"""Diagnose a recorded session.

    python tools/analyze_session.py                    # the newest session
    python tools/analyze_session.py --latest           # same thing, explicitly
    python tools/analyze_session.py sessions/foo.jsonl
    python tools/analyze_session.py --json --top 5

Two rotation reports, because there are two ways rotation goes wrong:

* **refused rotations** -- "it will not turn". Where they happened, with which
  piece, and why every SRS offset failed.
* **successful but suspicious rotations** -- "it turns but lands wrong", which a
  refusal count cannot see. A far kick (the first published offset did not fit), a
  resulting state that is not the one the turn asked for, or the same piece rotated
  over and over in a fraction of a second, all mean the piece is being fought into
  place rather than placed.

Around them sit the things a session can only tell you after the fact -- whether
DAS is doing what the config says, which keys resolved to nothing, and stretches
where the board moved without input.

To record one:

    python -m tetrisrl play --record        # or: battle --record
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import recorder as R                      # noqa: E402

MOVEMENT_ACTIONS = {'move_left': -1, 'move_right': 1}
#: A quiet stretch longer than this is reported: it usually means the player was
#: away, or that input stopped arriving while the board kept playing.
GAP_SECONDS = 3.0
#: A successful rotation that needed this kick index or later. Index 0 is "no
#: offset at all", 1 is the second published test; from 2 on the piece had to be
#: lifted or shifted to turn, which is where T-spins live and where a kick table
#: that does not match the shapes shows up.
FAR_KICK_INDEX = 2
#: Rotations of one piece closer together than this count as one burst. Rotating
#: twice is ordinary; four times in half a second is a fight.
BURST_GAP = 1.0

RECORD_HINT = 'python -m tetrisrl play --record   (or: battle --record)'


def summarize(events):
    """A one-screen summary of a session's events."""
    if not events:
        return {'events': 0}
    start = next((e for e in events if e.get('kind') == 'start'), {})
    stopped = next((e for e in reversed(events) if e.get('kind') == 'stopped'), {})
    duration = max((e.get('t') or 0.0) for e in events)
    locks = [e for e in events if e.get('kind') == 'lock']
    attacks = [e for e in events if e.get('kind') == 'attack']
    rotations = [e for e in events if e.get('kind') == 'rotate']
    keys = [e for e in events if e.get('kind') == 'key']
    moves = [e for e in events if e.get('kind') == 'move']
    clears = Counter(e.get('lines', 0) for e in locks)
    clearing = sum(n for lines, n in clears.items() if lines)
    seed = start.get('seed')
    for event in events:
        if event.get('seed') is not None:
            seed = event['seed']
            break
    return {
        'events': len(events),
        'duration': duration,
        'mode': start.get('mode'),
        'version': start.get('version'),
        'seed': seed,
        'settings': start.get('settings') or {},
        'bindings': start.get('bindings') or {},
        'input_problems': start.get('input_problems') or [],
        'stopped': stopped.get('reason'),
        'pieces': len(locks),
        'pps': (len(locks) / duration) if duration > 0 else 0.0,
        'lines': sum(e.get('lines', 0) for e in locks),
        'score': max((e.get('score', 0) for e in events
                      if e.get('kind') == 'state'), default=0),
        'tetrises': clears.get(4, 0),
        'tetris_share': (100.0 * clears.get(4, 0) / clearing) if clearing else 0.0,
        'clear_mix': {k: v for k, v in sorted(clears.items()) if k},
        'rotations': len(rotations),
        'refused_rotations': sum(1 for e in rotations if not e.get('ok')),
        'keys': len(keys),
        'key_misses': sum(1 for e in keys if e.get('action') is None),
        'moves': len(moves),
        'attacks': len(attacks),
        'garbage_tanked': sum(e.get('tanked', 0) for e in attacks),
        'garbage_sent': sum(e.get('sent', 0) for e in attacks),
        'deps': dependency_history(events),
    }


def refused_rotations(events, top=10):
    """Where rotations were refused, by piece, position, direction and reason."""
    refused = [e for e in events if e.get('kind') == 'rotate' and not e.get('ok')]
    attempts = [e for e in events if e.get('kind') == 'rotate']
    by_piece = Counter(e.get('piece') for e in refused)
    by_position = Counter((e.get('x'), e.get('y')) for e in refused)
    by_reason = Counter(e.get('reason') for e in refused)
    by_direction = Counter(e.get('direction') for e in refused)
    contexts = Counter(
        ('wall' if (e.get('left_blocked') and e.get('right_blocked'))
         else 'left wall' if e.get('left_blocked')
         else 'right wall' if e.get('right_blocked')
         else 'floor' if e.get('floor')
         else 'open board')
        for e in refused)
    examples = []
    for event in refused[:max(0, top)]:
        examples.append({
            't': event.get('t'),
            'piece': event.get('piece'),
            'direction': event.get('direction'),
            'from': event.get('from'),
            'to': event.get('to'),
            'x': event.get('x'),
            'y': event.get('y'),
            'reason': event.get('reason'),
            'left_blocked': event.get('left_blocked'),
            'right_blocked': event.get('right_blocked'),
            'floor': event.get('floor'),
            'candidates': event.get('candidates'),
        })
    # The worst cell: the position that refused most often.
    worst = by_position.most_common(top)
    return {
        'attempts': len(attempts),
        'refused': len(refused),
        'rate': (100.0 * len(refused) / len(attempts)) if attempts else 0.0,
        'by_piece': dict(by_piece.most_common()),
        'by_position': [{'x': x, 'y': y, 'count': n} for (x, y), n in worst],
        'by_direction': {('cw' if d == 1 else 'ccw' if d == -1 else f'180({d})'): n
                         for d, n in by_direction.most_common()},
        'by_reason': dict(by_reason.most_common()),
        'by_context': dict(contexts.most_common()),
        'examples': examples,
    }


def suspicious_rotations(events, top=10, burst_gap=BURST_GAP):
    """Rotations that *succeeded* but should not have needed to work that hard.

    The refused-rotation report answers "it will not turn". This one answers "it
    turns, but not the way I asked", which is invisible in a refusal count and was
    exactly the shape of the bug that made T-spin triples impossible: every
    rotation succeeded, they just landed somewhere else.

    Four signals, all from the ``rotate`` events the recorder already writes:

    * **far kicks** -- a successful turn whose first fitting offset was index
      ``FAR_KICK_INDEX`` or later, meaning the piece had to be shifted or lifted to
      turn at all. The distribution by piece is the interesting part: "the I always
      needs a far kick" is a table or shape problem, not a player problem.
    * **wrong resulting state** -- ``to`` is not ``(from + direction) % 4``, which
      means the rotation served a different turn than the one requested.
    * **bursts** -- the same piece rotated again within ``burst_gap`` seconds, with
      no lock in between. One extra turn is normal; a burst of four is someone
      fighting a piece into a slot.
    * **kick index by piece** -- the whole distribution, so a pattern is visible
      without reading examples.
    """
    rotations = [e for e in events if e.get('kind') == 'rotate']
    ok = [e for e in rotations if e.get('ok')]

    far = [e for e in ok if (e.get('kick_index') or 0) >= FAR_KICK_INDEX]
    far_by_piece = Counter(e.get('piece') for e in far)

    wrong = []
    for event in ok:
        old, new = event.get('from'), event.get('to')
        direction = event.get('direction')
        if old is None or new is None or not direction:
            continue
        if new != (old + direction) % 4:
            wrong.append(event)

    # Bursts: rotations of one piece with no lock between them and no long pause.
    bursts = []
    current = []
    for event in events:
        kind = event.get('kind')
        if kind == 'lock':
            if current:
                bursts.append(current)
                current = []
        elif kind == 'rotate':
            same_piece = current and current[-1].get('piece') == event.get('piece')
            close = current and (event.get('t') or 0) - (current[-1].get('t') or 0) \
                <= burst_gap
            if current and not (same_piece and close):
                bursts.append(current)
                current = []
            current.append(event)
    if current:
        bursts.append(current)
    long_bursts = sorted((b for b in bursts if len(b) >= 2),
                         key=lambda b: (-len(b), b[0].get('t') or 0))

    kick_by_piece = defaultdict(Counter)
    attempts_by_piece = Counter()
    for event in ok:
        attempts_by_piece[event.get('piece')] += 1
        kick_by_piece[event.get('piece')][event.get('kick_index') or 0] += 1

    def brief(event):
        return {
            't': event.get('t'),
            'piece': event.get('piece'),
            'from': event.get('from'),
            'to': event.get('to'),
            'direction': event.get('direction'),
            'x': event.get('x'),
            'y': event.get('y'),
            'kick_index': event.get('kick_index'),
            'kick': event.get('kick'),
            'left_blocked': event.get('left_blocked'),
            'right_blocked': event.get('right_blocked'),
            'floor': event.get('floor'),
        }

    return {
        'attempts': len(rotations),
        'succeeded': len(ok),
        'far_kicks': len(far),
        'far_kick_rate': (100.0 * len(far) / len(ok)) if ok else 0.0,
        'far_kick_by_piece': dict(far_by_piece.most_common()),
        'far_kick_examples': [brief(e) for e in far[:max(0, top)]],
        'wrong_state': len(wrong),
        'wrong_state_examples': [brief(e) for e in wrong[:max(0, top)]],
        'bursts': len(long_bursts),
        'worst_bursts': [
            {
                't': burst[0].get('t'),
                'piece': burst[0].get('piece'),
                'length': len(burst),
                'rotations': [f"{e.get('from')}->{e.get('to')}" for e in burst],
                'positions': [[e.get('x'), e.get('y')] for e in burst],
            }
            for burst in long_bursts[:max(0, top)]
        ],
        'burst_lengths': dict(sorted(Counter(
            len(b) for b in long_bursts).items())),
        'kick_index_by_piece': {piece: dict(sorted(counts.items()))
                                for piece, counts in sorted(kick_by_piece.items())},
        'attempts_by_piece': dict(attempts_by_piece.most_common()),
        'clean': not far and not wrong and not long_bursts,
    }


def key_latency(events, top=5):
    """Observed DAS and ARR, from the press-to-first-repeat gap.

    This is the check a config cannot do for itself: the settings file says what
    DAS *should* be, and only the recording says what the game did with it.
    """
    pending = {}                       # direction -> press time
    first_done = {}                    # direction -> already measured DAS
    last_auto = {}                     # direction -> time of the previous repeat
    das_samples = defaultdict(list)
    arr_samples = defaultdict(list)
    for event in events:
        kind = event.get('kind')
        if kind == 'key':
            action = event.get('action')
            if action in MOVEMENT_ACTIONS:
                if event.get('event') == 'down':
                    pending[action] = event.get('t')
                    first_done[action] = False
                    last_auto[action] = None
                else:
                    pending.pop(action, None)
        elif kind == 'move' and event.get('auto'):
            dx = event.get('dx')
            action = 'move_left' if dx == -1 else 'move_right' if dx == 1 else None
            if action is None or action not in pending:
                continue
            if not first_done.get(action):
                das_samples[action].append(event.get('t', 0) - pending[action])
                first_done[action] = True
                last_auto[action] = event.get('t', 0)
            elif last_auto.get(action) is not None:
                arr_samples[action].append(event.get('t', 0) - last_auto[action])
                last_auto[action] = event.get('t', 0)

    def stats(samples):
        if not samples:
            return None
        ordered = sorted(samples)
        return {
            'n': len(ordered),
            'min': ordered[0],
            'median': statistics.median(ordered),
            'max': ordered[-1],
            'p90': ordered[int(0.9 * (len(ordered) - 1))],
        }

    out = {'das': {}, 'arr': {}}
    for action in sorted(set(list(das_samples) + list(arr_samples))):
        out['das'][action] = stats(das_samples.get(action) or [])
        out['arr'][action] = stats(arr_samples.get(action) or [])
    out['worst'] = [
        {'action': action, 'median': s['median'], 'n': s['n']}
        for action, s in sorted(out['das'].items(), key=lambda kv: -(kv[1] or
                                                                   {'median': 0})['median'])
        if s
    ][:top]
    return out


def suspicious(events, top=10, gap_seconds=GAP_SECONDS):
    """Keys that did nothing, overlapping bindings, quiet stretches, silent locks."""
    keys = [e for e in events if e.get('kind') == 'key']
    misses = Counter(e.get('name') for e in keys if e.get('action') is None)
    start = next((e for e in events if e.get('kind') == 'start'), {})
    bindings = start.get('bindings') or {}
    owners = defaultdict(list)
    for action, names in bindings.items():
        for name in names or ():
            owners[name].append(action)
    overlaps = {name: actions for name, actions in owners.items()
                if len(actions) > 1}

    gaps = []
    previous = None
    for event in events:
        if event.get('kind') not in ('key', 'move', 'rotate', 'lock'):
            continue
        if previous is not None:
            delta = (event.get('t') or 0) - previous
            if delta >= gap_seconds:
                gaps.append({'t': event.get('t'), 'seconds': delta,
                             'after': event.get('kind')})
        previous = event.get('t') or 0

    silent = []
    touched = False
    for event in events:
        kind = event.get('kind')
        if kind in ('key', 'move', 'rotate'):
            touched = True
        elif kind == 'lock':
            if not touched:
                silent.append({'t': event.get('t'), 'piece': event.get('piece'),
                               'y': event.get('y')})
            touched = False
    return {
        'key_misses': dict(misses.most_common(top)),
        'overlapping_bindings': overlaps,
        'input_problems': start.get('input_problems') or [],
        'long_gaps': sorted(gaps, key=lambda g: -g['seconds'])[:top],
        'gap_threshold': gap_seconds,
        'locks_without_input': silent[:top],
        'locks_without_input_count': len(silent),
    }


def report(path, top=10):
    """The whole diagnosis for one session file."""
    events = R.load(path)
    return {
        'file': path,
        'summary': summarize(events),
        'refused_rotations': refused_rotations(events, top=top),
        'suspicious_rotations': suspicious_rotations(events, top=top),
        'key_latency': key_latency(events, top=top),
        'suspicious': suspicious(events, top=top),
    }


# --- printing ---------------------------------------------------------------

def _fmt(value, digits=1, dash='-' ):
    if value is None:
        return dash
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def dependency_history(events):
    """Peak I-dependencies, when they peaked, and whether the board ended owing one.

    Reads the ``deps`` field of the lock events. A session recorded before that
    field existed simply reports nothing rather than guessing.
    """
    locks = [e for e in events if e.get('kind') == 'lock']
    counts = [(e.get('t') or 0.0, e.get('deps')) for e in locks]
    counts = [(t, d) for t, d in counts if isinstance(d, int)]
    if not counts:
        return {'locks': 0}
    peak = max(d for _t, d in counts)
    peak_at = next(t for t, d in counts if d == peak)
    return {
        'locks': len(counts),
        'peak': peak,
        'peak_at': peak_at,
        'open_share': 100.0 * sum(1 for _t, d in counts if d >= 1) / len(counts),
        'two_share': 100.0 * sum(1 for _t, d in counts if d >= 2) / len(counts),
        'final': counts[-1][1],
    }


def print_report(data, stream=None):
    # ``stream=None`` rather than ``stream=sys.stdout``: a default argument is
    # bound once, at import, so the original form ignored stdout redirection --
    # which made the report impossible to capture in a test or a pipe.
    stream = stream or sys.stdout
    summary = data['summary']
    out = stream.write
    out(f"session {data['file']}\n")
    out(f"  mode {summary.get('mode')}  seed {summary.get('seed')}  "
        f"version {summary.get('version')}  stopped: {summary.get('stopped')}\n")
    out(f"  {summary['events']} events over {summary['duration']:.0f}s  "
        f"({summary['pieces']} pieces, {summary['pps']:.2f} PPS)\n")
    out(f"  lines {summary['lines']}  score {summary['score']}  "
        f"tetrises {summary['tetrises']} ({summary['tetris_share']:.1f}% of clears)  "
        f"mix {summary['clear_mix']}\n")
    settings = summary.get('settings') or {}
    if settings:
        out(f"  settings: DAS {_fmt(settings.get('das_ms'), 0)}ms  "
            f"ARR {_fmt(settings.get('arr_ms'), 0)}ms  "
            f"SDF {_fmt(settings.get('sdf'))}  "
            f"lock {_fmt(settings.get('lock_delay_ms'), 0)}ms\n")
    if summary.get('attacks'):
        out(f"  battle: sent {summary['garbage_sent']}  "
            f"tanked {summary['garbage_tanked']}\n")

    deps = summary.get('deps') or {}
    if deps.get('locks'):
        out('\ndependency history (one-wide gaps only an I can fill)\n')
        out(f"  peak {deps['peak']} at t={deps['peak_at']:.0f}s  "
            f"open at {deps['open_share']:.0f}% of locks  "
            f"two-or-more at {deps['two_share']:.0f}%  "
            f"ended with {deps['final']}\n")
        if deps['peak'] >= 2:
            out('  two at once is the doom loop: the board is waiting on two I '
                'pieces and the bag gives one per seven\n')

    refused = data['refused_rotations']
    out('\nrefused rotations\n')
    out(f"  {refused['refused']} of {refused['attempts']} attempts "
        f"({refused['rate']:.1f}%)\n")
    if refused['refused']:
        out(f"  by piece     {refused['by_piece']}\n")
        out(f"  by direction {refused['by_direction']}\n")
        out(f"  by reason    {refused['by_reason']}\n")
        out(f"  by context   {refused['by_context']}\n")
        out('  worst cells  ' + ', '.join(
            f"x{p['x']}y{p['y']} ({p['count']})"
            for p in refused['by_position']) + '\n')
        for example in refused['examples'][:5]:
            offsets = example.get('candidates') or []
            shown = offsets[:8]
            tried = ', '.join(f"{o['offset']}{'ok' if o['ok'] else 'x'}"
                              for o in shown)
            if len(offsets) > len(shown):
                tried += f", ... ({len(offsets)} tried)"
            out(f"    t={_fmt(example['t'])} {example['piece']} "
                f"{example['from']}->{example['to']} at "
                f"({example['x']},{example['y']}) {example['reason']}"
                f"  left={example['left_blocked']} right={example['right_blocked']}"
                f" floor={example['floor']}\n")
            if tried:
                out(f"        offsets tried: {tried}\n")

    spins = data['suspicious_rotations']
    out('\nsuccessful but suspicious rotations\n')
    if spins['clean']:
        out(f"  nothing suspicious: {spins['succeeded']} rotations turned, "
            f"none needed a far kick, none landed in an unexpected state, "
            f"none repeated on one piece\n")
    else:
        out(f"  {spins['far_kicks']} of {spins['succeeded']} successful rotations "
            f"needed a far kick, index >= {FAR_KICK_INDEX} "
            f"({spins['far_kick_rate']:.1f}%)\n")
        if spins['far_kick_by_piece']:
            out(f"  far kicks by piece  {spins['far_kick_by_piece']}\n")
        for example in spins['far_kick_examples'][:5]:
            out(f"    t={_fmt(example['t'])} {example['piece']} "
                f"{example['from']}->{example['to']} at "
                f"({example['x']},{example['y']}) kick index "
                f"{example['kick_index']} offset {example['kick']}"
                f"  left={example['left_blocked']} right={example['right_blocked']}"
                f" floor={example['floor']}\n")
        if spins['wrong_state']:
            out(f"  rotations that landed in an unexpected state: "
                f"{spins['wrong_state']}\n")
            for example in spins['wrong_state_examples'][:5]:
                expected = (example['from'] + example['direction']) % 4
                out(f"    t={_fmt(example['t'])} {example['piece']} "
                    f"{example['from']}->{example['to']} "
                    f"(direction {example['direction']} should give {expected}) "
                    f"at ({example['x']},{example['y']})\n")
        if spins['bursts']:
            worst = spins['worst_bursts'][0]
            out(f"  pieces rotated repeatedly without locking: {spins['bursts']} "
                f"(longest {worst['length']} in a row, {worst['piece']} at "
                f"t={_fmt(worst['t'])}, lengths {spins['burst_lengths']})\n")
            for burst in spins['worst_bursts'][:5]:
                out(f"    t={_fmt(burst['t'])} {burst['piece']} "
                    f"{' then '.join(burst['rotations'])}"
                    f"  at {burst['positions']}\n")
        out(f"  kick index by piece  {spins['kick_index_by_piece']}   "
            f"(0 = no offset needed)\n")

    out('\nkey latency (observed, from the recording)\n')
    latency = data['key_latency']
    any_latency = False
    for action, stats in latency['das'].items():
        if not stats:
            continue
        any_latency = True
        arr = latency['arr'].get(action)
        out(f"  {action:<11} DAS median {stats['median'] * 1000:.0f}ms "
            f"(min {stats['min'] * 1000:.0f}, max {stats['max'] * 1000:.0f}, "
            f"n={stats['n']})"
            + (f"   ARR median {arr['median'] * 1000:.0f}ms" if arr else '') + '\n')
    if not any_latency:
        out('  (no auto-repeat observed: nothing was held long enough)\n')

    bad = data['suspicious']
    out('\nsuspicious\n')
    out(f"  keys that resolved to nothing: {bad['key_misses'] or 'none'}\n")
    if bad['overlapping_bindings']:
        out(f"  keys bound to more than one action: {bad['overlapping_bindings']}\n")
    if bad['input_problems']:
        out(f"  input problems at startup: {bad['input_problems']}\n")
    if bad['long_gaps']:
        worst = bad['long_gaps'][0]
        out(f"  quiet stretches ≥{bad['gap_threshold']:.0f}s: "
            f"{len(bad['long_gaps'])} (longest {worst['seconds']:.1f}s at "
            f"t={_fmt(worst['t'])})\n")
    if bad['locks_without_input_count']:
        out(f"  pieces locked with no input at all: "
            f"{bad['locks_without_input_count']} "
            f"(e.g. {bad['locks_without_input'][:3]})\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'to record a new session:  {RECORD_HINT}')
    parser.add_argument('sessions', nargs='*',
                        help='session files (default: the newest in sessions/)')
    parser.add_argument('--latest', action='store_true',
                        help='analyse the newest session file, ignoring any paths')
    parser.add_argument('--json', action='store_true',
                        help='machine-readable output')
    parser.add_argument('--top', type=int, default=10,
                        help='how many examples to show (default 10)')
    args = parser.parse_args(argv)

    paths = [] if args.latest else list(args.sessions)
    if not paths:
        newest = R.newest_session()
        if newest is None:
            print(f'no sessions in {R.session_dir()}; record one with '
                  f'{RECORD_HINT}')
            return 1
        paths = [newest]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for path in missing:
            print(f'no such session file: {path}', file=sys.stderr)
        return 1

    reports = [report(path, top=args.top) for path in paths]
    if args.json:
        json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, indent=2)
        sys.stdout.write('\n')
        return 0
    for index, data in enumerate(reports):
        if index:
            print()
        print_report(data)
    print(f'\nrecord a new session:  {RECORD_HINT}\n'
          f'then re-run:           python tools/analyze_session.py --latest')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
