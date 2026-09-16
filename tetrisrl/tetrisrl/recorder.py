"""Session recording: newline-delimited JSON of what actually happened.

Why this exists
---------------
A bug report arrived as one sentence -- *"I can't rotate when there's a wall"* --
and diagnosing it took a synthetic test written by hand, because the only evidence
was the description. The engine's kick tables, the input layer and the game state
at the moment of the failure were all gone by the time anyone looked.

A recording answers those questions directly. In particular **a refused rotation is
a first-class event**: it carries the piece, the direction, the position, whether
the piece was against a wall or the floor, every kick offset that was tried, and
why none of them fitted. That single record is the difference between "rotating
feels broken" and "the I piece at x=0 could not turn counter-clockwise because the
(0,0) and (-2,0) offsets collide and the table has no entry for (1,0)".

Design rules
------------
* **One JSON object per line.** A session that ends in a crash is still readable up
  to the last complete line, and ``iter_events`` streams it without loading
  everything.
* **Bounded.** A session stops at ``MAX_EVENTS`` and says so in a final ``stopped``
  event. No grid dumps per frame -- only events, plus a periodic ``state``.
* **Never fatal.** Every write is guarded. A recording that cannot be written
  disables itself and warns once, because losing a recording must never lose the
  game.
* **Off unless asked for.** ``--record`` on ``play``/``battle``; nothing else pays
  for it. ``recorder=None`` is the default everywhere.

Event kinds
-----------
``start``   the effective settings, bindings, seed, mode and version -- everything
            needed to reproduce or explain the session
``key``     a dispatched key press or release, with the action it resolved to
            (``null`` when it resolved to nothing, which is itself a bug report)
``move``    a horizontal step, flagged ``auto`` when it came from DAS/ARR rather
            than from the press itself, which is what makes DAS measurable
``rotate``  every attempt, successful or refused
``lock``    every piece that came to rest, with the placement and the timings
``attack``  battle: what a lock sent, cancelled and tanked
``state``   a periodic snapshot (every ``STATE_INTERVAL`` seconds)
``stopped`` why the session ended: closed, or the event cap
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

from .engine import IKICKS, IKICKS_180, KICKS, KICKS_180

#: Hard cap on one session. A long game is ~10 events per piece, so this is hours
#: of play; it exists so a stuck loop cannot fill a disk.
MAX_EVENTS = 20000
#: Seconds between ``state`` snapshots.
STATE_INTERVAL = 5.0
#: Flush every event: a crash then costs at most the event in flight.
_FLUSH_EVERY = 1

DEFAULT_DIRNAME = 'sessions'


# --- paths ------------------------------------------------------------------

def session_dir(project_dir=None):
    """``tetrisrl/sessions``, created on demand."""
    if project_dir is None:
        from . import PROJECT_DIR
        project_dir = PROJECT_DIR
    path = os.path.join(project_dir, DEFAULT_DIRNAME)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def default_path(project_dir=None, now=None):
    """``sessions/<UTC timestamp>.jsonl``, unique to the second."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(session_dir(project_dir), f'{stamp}.jsonl')
    if os.path.exists(path):                       # two sessions in one second
        for n in range(2, 100):
            alt = os.path.join(session_dir(project_dir), f'{stamp}-{n}.jsonl')
            if not os.path.exists(alt):
                return alt
    return path


def session_files(directory=None):
    """Every session file, oldest first."""
    directory = directory or session_dir()
    try:
        names = [n for n in os.listdir(directory) if n.endswith('.jsonl')]
    except OSError:
        return []
    return [os.path.join(directory, n) for n in sorted(names)]


def newest_session(directory=None):
    files = session_files(directory)
    return files[-1] if files else None


# --- reading ----------------------------------------------------------------

def iter_events(path):
    """Stream the events of a session file, skipping lines that are not JSON.

    A truncated final line is expected when a session ends in a crash, so it is
    skipped rather than raised.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def load(path):
    """Every event in a session file, as a list."""
    return list(iter_events(path))


# --- describing a rotation --------------------------------------------------

def rotation_report(game, direction):
    """What a rotation attempt would do, and why.

    Mirrors ``Game.can_rotate`` exactly -- same tables, same order, same fallback
    -- and adds the detail the engine has no reason to expose: which offset of the
    SRS table was used, which ones were tried and refused, and whether the piece
    was boxed in against a wall or resting on the floor.

    The engine is not modified for this. The tables are read, and the collision
    test is the engine's own ``Board.collides``, so the report cannot disagree
    with what the game actually did; if it ever did, the ``rotate`` events and the
    game would diverge visibly.
    """
    piece = game.current
    if piece is None:
        return {'ok': False, 'reason': 'no active piece', 'candidates': []}
    old = piece.rotation
    new = (old + direction) % 4
    report = {
        'piece': piece.kind,
        'from': old,
        'to': new,
        'direction': direction,
        'x': piece.x,
        'y': piece.y,
        'ok': False,
        'kick': None,
        'kick_index': None,
        'left_blocked': not game.can_move(-1),
        'right_blocked': not game.can_move(1),
        'floor': not game.can_move(0, 1),
        'candidates': [],
        'reason': None,
    }
    if piece.kind == 'O':
        # The engine short-circuits O: its cells are identical in every state.
        report.update(ok=True, kick=[0, 0], kick_index=0)
        return report
    if direction == 2:
        table = IKICKS_180 if piece.kind == 'I' else KICKS_180
    else:
        table = IKICKS if piece.kind == 'I' else KICKS
    offsets = table.get((old, new))
    if offsets is None:
        # The engine falls back to a bare (0, 0) here, which is exactly why a
        # missing table entry shows up as "rotation refused while against a wall".
        report['reason'] = 'table miss'
        offsets = ((0, 0),)
    for index, (dx, dy) in enumerate(offsets):
        fits = not game.board.collides(
            piece.cells(rotation=new, x=piece.x + dx, y=piece.y + dy))
        report['candidates'].append({'offset': [dx, dy], 'ok': fits})
        if fits and not report['ok']:
            report['ok'] = True
            report['kick'] = [dx, dy]
            report['kick_index'] = index
    if not report['ok'] and report['reason'] is None:
        report['reason'] = 'all offsets collide'
    if report['ok'] and report['kick'] == [0, 0]:
        report['kick'] = None                      # no kick was needed
    return report


# --- the recorder -----------------------------------------------------------

class SessionRecorder:
    """Writes one session as JSONL. Safe to call from a 60 fps loop.

    ``stream`` exists for tests: pass an ``io.StringIO`` and nothing touches the
    filesystem. ``clock`` is injectable for the same reason.
    """

    def __init__(self, path=None, meta=None, max_events=MAX_EVENTS,
                 stream=None, clock=time.monotonic, state_interval=STATE_INTERVAL,
                 echo=True):
        self.max_events = int(max_events)
        self.state_interval = float(state_interval)
        self.warnings = []
        self.count = 0
        self.frame = 0
        self.stopped = False
        self.enabled = True
        self._clock = clock
        self._start = clock()
        # Negative so the *first* state_if_due call fires: a session should open
        # with a snapshot of the board it started from.
        self._last_state = -self.state_interval
        self._echo = echo
        self._owns_stream = stream is None
        self.path = path
        if stream is not None:
            self.stream = stream
        else:
            path = path or default_path()
            self.path = os.path.abspath(path)
            try:
                parent = os.path.dirname(self.path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.stream = open(self.path, 'w', encoding='utf-8')
            except OSError as exc:
                self._disable(exc)
                self.stream = None
        if meta is not None and self.enabled:
            self.record('start', **meta)

    # -- lifecycle --------------------------------------------------------

    def elapsed(self):
        return self._clock() - self._start

    def tick(self, dt=None):
        """Advance the frame counter (and optionally the clock, for tests)."""
        self.frame += 1
        if dt is not None:
            self._start -= dt                  # pretend dt seconds have passed
        return self.frame

    def record(self, kind, **fields):
        """Append one event. Returns False when recording is off or stopped.

        Never raises: a full disk or a closed stream disables recording rather
        than interrupting play.
        """
        if not self.enabled:
            return False
        if self.count >= self.max_events:
            self._stop('event cap reached')
            return False
        payload = {'t': round(self.elapsed(), 4), 'frame': self.frame, 'kind': kind}
        for key, value in fields.items():
            payload[key] = value
        try:
            line = json.dumps(payload, separators=(',', ':'), default=str)
        except (TypeError, ValueError) as exc:      # unserialisable field
            self._warn(f'could not serialise a {kind} event ({exc})')
            return False
        try:
            self.stream.write(line + '\n')
            self.count += 1
            if self.count % _FLUSH_EVERY == 0:
                self.stream.flush()
        except Exception as exc:                    # noqa: BLE001 - never fatal
            self._disable(exc)
            return False
        return True

    def close(self, reason='closed'):
        """Finish the session: write ``stopped``, flush, and close our own file."""
        if self.stopped:
            return self.count
        self._stop(reason)
        return self.count

    def _stop(self, reason):
        if not self.stopped and self.enabled:
            self.stopped = True
            payload = {'t': round(self.elapsed(), 4), 'frame': self.frame,
                       'kind': 'stopped', 'reason': reason,
                       'events': self.count}
            try:
                self.stream.write(json.dumps(payload, separators=(',', ':')) + '\n')
                self.count += 1
                self.stream.flush()
            except Exception as exc:                # noqa: BLE001
                self._disable(exc)
                return
        self.enabled = False
        if self._owns_stream and self.stream is not None:
            try:
                self.stream.close()
            except Exception:                       # noqa: BLE001
                pass

    def _disable(self, exc):
        self.enabled = False
        self._warn(f'{exc}')

    def _warn(self, message):
        note = f'recorder disabled: {message}'
        if note not in self.warnings:
            self.warnings.append(note)
            if self._echo:
                print(note, file=sys.stderr)

    # -- domain hooks -----------------------------------------------------

    def key(self, key, action, pressed, name=None):
        """A dispatched key event, including keys that resolve to no action."""
        from .controls import key_name
        return self.record('key', key=int(key), name=name or key_name(key),
                           action=action, event='down' if pressed else 'up')

    def move(self, dx, auto, ok=True):
        """A horizontal step; ``auto`` marks a DAS/ARR repeat."""
        return self.record('move', dx=int(dx), auto=bool(auto), ok=bool(ok))

    def rotate(self, game, direction, report=None):
        """Record a rotation attempt, refused or not."""
        report = report if report is not None else rotation_report(game, direction)
        return self.record('rotate', **report)

    def lock(self, piece, lines, combo, b2b, used_hold, settings=None,
             top=None, holes=None, deps=None):
        """Record a piece coming to rest. ``piece`` is the piece *before* locking."""
        fields = {
            'piece': getattr(piece, 'kind', None),
            'rotation': getattr(piece, 'rotation', None),
            'x': getattr(piece, 'x', None),
            'y': getattr(piece, 'y', None),
            'lines': int(lines),
            'combo': int(combo),
            'b2b': bool(b2b),
            'hold': bool(used_hold),
        }
        if top is not None:
            fields['top'] = int(top)
        if holes is not None:
            fields['holes'] = int(holes)
        if deps is not None:
            fields['deps'] = int(deps)
        if settings is not None:
            fields['das_ms'] = settings.das_ms
            fields['arr_ms'] = settings.arr_ms
            fields['sdf'] = settings.sdf
            fields['lock_delay_ms'] = settings.lock_delay_ms
        return self.record('lock', **fields)

    def attack(self, side, summary, battle=None):
        """Battle: what one lock sent, cancelled and tanked."""
        attack = summary.get('attack')
        fields = {
            'side': side,
            'sent': int(summary.get('sent', 0)),
            'cancelled': int(summary.get('cancelled', 0)),
            'tanked': len(summary.get('tanked') or ()),
            'lines': getattr(attack, 'lines', 0),
            'chunks': list(getattr(attack, 'chunks', ()) or ()),
            'b2b': getattr(attack, 'b2b', 0),
            'ko': bool(summary.get('ko')),
        }
        if battle is not None:
            fields['pending'] = [battle.sides[0].queue.pending,
                                 battle.sides[1].queue.pending]
            fields['ready'] = [battle.ready(0), battle.ready(1)]
            fields['wins'] = [battle.sides[0].wins, battle.sides[1].wins]
            fields['round'] = battle.round
        return self.record('attack', **fields)

    def state_if_due(self, game, battle=None, force=False, extra=None):
        """A periodic snapshot: score, lines, height, holes, queue."""
        if not self.enabled:
            return False
        now = self.elapsed()
        if not force and now - self._last_state < self.state_interval:
            return False
        self._last_state = now
        return self.state(game, battle=battle, extra=extra)

    def state(self, game, battle=None, extra=None):
        heights = game.board.column_heights()
        fields = {
            'score': game.score,
            'lines': game.lines,
            'level': game.level,
            'pieces': game.pieces_placed,
            'top': max(heights) if heights else 0,
            'holes': count_holes(game),
            'game_over': bool(game.game_over),
            'queue': list(game.queue_kinds),
            'hold': game.hold_kind,
        }
        if battle is not None:
            for index, side in enumerate(battle.sides):
                prefix = 'you' if index == 0 else 'foe'
                fields[f'{prefix}_top'] = max(side.game.board.column_heights() or [0])
                fields[f'{prefix}_lines'] = side.game.lines
                fields[f'{prefix}_sent'] = side.sent
                fields[f'{prefix}_tanked'] = side.tanked
                fields[f'{prefix}_pending'] = side.queue.pending
        if extra:
            fields.update(extra)
        return self.record('state', **fields)


def count_i_dependencies(game):
    """One-cell-wide gaps at least three rows deep -- holes only an I can fill.

    Same definition the search prices (Blockfish's ``i_dependencies``), computed
    here so a session records it even when the recorder runs without the search
    weights that act on it. This is the number that explains a board which keeps
    growing with two holes in it.
    """
    from .search import board_masks, heights_of, i_dependencies
    masks = board_masks(game)
    return i_dependencies(masks, heights_of(masks))[0]


def count_holes(game):
    """Holes on a board: empty cells with something above them in the column.

    Computed here rather than through ``features.board_metrics`` so the recorder
    stays free of numpy -- this runs inside the frame loop.
    """
    holes = 0
    grid = game.board.grid
    for x in range(game.cols):
        seen = False
        for y in range(game.rows):
            if grid[y][x] is not None:
                seen = True
            elif seen:
                holes += 1
    return holes


def start_meta(mode, seed=None, settings=None, extra=None):
    """The ``start`` event's payload: what makes a session explainable.

    Settings and bindings are the point. A recording of "the rotation did not
    work" is worthless without DAS, ARR, SDF and which keys were bound at the
    time, because those are exactly what a later reader would otherwise assume.
    """
    from . import __version__
    meta = {
        'version': __version__,
        'mode': mode,
        'seed': seed,
        'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    if settings is not None:
        meta['settings'] = {
            'das_ms': settings.das_ms,
            'arr_ms': settings.arr_ms,
            'sdf': settings.sdf,
            'lock_delay_ms': settings.lock_delay_ms,
            'lock_reset_limit': settings.lock_reset_limit,
        }
        meta['bindings'] = {name: settings.binding_names(name)
                            for name in settings.bindings}
        problems = list(settings.problems())
        if problems or settings.warnings:
            meta['input_problems'] = list(settings.warnings) + problems
    if extra:
        meta.update(extra)
    return meta
