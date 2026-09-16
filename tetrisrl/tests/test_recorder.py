"""Tests for the session recorder and the analyser. Run: python tests/test_recorder.py

The recorder exists because a bug report ("I can't rotate when there's a wall")
could not be diagnosed from the report alone. So the tests are mostly about the
guarantees that make a recording worth having: it is valid JSONL even when
truncated, it never grows without bound, it never takes the game down with it, and
**a refused rotation is recorded with the offsets that were tried**.
"""
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import analyze_session as A                          # noqa: E402
import pygame                                        # noqa: E402
from tetrisrl import controls as C                   # noqa: E402
from tetrisrl import recorder as R                   # noqa: E402
from tetrisrl.engine import Game, Piece              # noqa: E402
from tetrisrl.render import HumanInput, _BattleState, _battle_step  # noqa: E402
from tetrisrl.versus import Battle                   # noqa: E402

FAILS = []
CHECKS = 0


def check(name, cond, detail=''):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


def settings(**kw):
    kw.setdefault('das_ms', 100)
    return C.InputSettings(**kw)


def boxed_in_game():
    """A game whose active I can rotate in no direction: a one-wide shaft."""
    game = Game(seed=1)
    for y in range(game.rows):
        for x in range(game.cols):
            game.board.grid[y][x] = 'X'
    for y in range(6, game.rows):
        game.board.grid[y][4] = None
    game.current = Piece('I', 4, 6, rotation=1)
    return game


class Boom:
    """A stream that fails, to prove recording never takes the game down."""

    def write(self, _data):
        raise OSError('disk full')

    def flush(self):
        pass


# --- the file format --------------------------------------------------------

def test_writes_valid_jsonl():
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, meta=R.start_meta('test', seed=7))
    rec.key(100, 'move_left', True)
    rec.move(-1, auto=False)
    rec.close()
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]
    check('every line is a JSON object', all(isinstance(e, dict) for e in parsed))
    check('every event carries t, frame and kind',
          all({'t', 'frame', 'kind'} <= set(e) for e in parsed))
    check('the start event has the settings and the seed',
          parsed[0]['kind'] == 'start' and parsed[0]['seed'] == 7)
    check('a close is recorded as a stopped event', parsed[-1]['kind'] == 'stopped')


def test_round_trips_through_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'session.jsonl')
        rec = R.SessionRecorder(path=path, meta=R.start_meta('play', seed=3,
                                                            settings=settings()),
                                echo=False)
        rec.record('lock', piece='T', lines=2)
        rec.close()
        events = R.load(path)
        streamed = list(R.iter_events(path))
        check('load returns the events in order',
              [e['kind'] for e in events] == ['start', 'lock', 'stopped'],
              [e['kind'] for e in events])
        check('iter_events streams the same events', streamed == events)
        check('the settings survived the round trip',
              events[0]['settings']['das_ms'] == 100)


def test_truncated_line_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'cut.jsonl')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('{"kind": "start", "t": 0}\n')
            handle.write('{"kind": "lock", "t": 1}\n')
            handle.write('{"kind": "loc')          # a crash mid-write
        events = R.load(path)
        check('a half-written last line is ignored, not raised', len(events) == 2,
              len(events))


def test_event_cap_stops_cleanly():
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, max_events=4, echo=False)
    for i in range(10):
        rec.record('noise', i=i)
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    recorded = [e for e in events if e['kind'] != 'stopped']
    check('the cap stops the session at max_events', len(recorded) == 4,
          len(recorded))
    check('and says why', events[-1]['kind'] == 'stopped'
          and events[-1]['reason'] == 'event cap reached', events[-1])
    check('further records are refused', rec.record('noise', i=99) is False)
    check('recording is disabled afterwards', rec.enabled is False)


def test_failed_write_disables_recording():
    rec = R.SessionRecorder(stream=Boom(), echo=False)
    ok = rec.record('lock', piece='I')
    check('a failing write returns False instead of raising', ok is False)
    check('recording disables itself', rec.enabled is False)
    check('and warns once', len(rec.warnings) == 1, rec.warnings)
    check('further records are harmless', rec.record('lock') is False)
    rec.close()                                   # must not raise either


def test_default_path_is_in_sessions():
    path = R.default_path()
    check('sessions land in the sessions/ directory',
          os.path.dirname(path) == R.session_dir(), path)
    check('the filename is a UTC timestamp',
          os.path.basename(path).endswith('.jsonl') and 'T' in os.path.basename(path),
          os.path.basename(path))
    ignored = open(os.path.join(os.path.dirname(R.session_dir()), '.gitignore'),
                   encoding='utf-8').read()
    check('recordings are git-ignored', 'sessions/' in ignored)


# --- the rotation report ----------------------------------------------------

def test_rotation_report_mirrors_the_engine():
    """The report must never disagree with what the game actually does."""
    disagree = 0
    checked = 0
    for seed in range(8):
        game = Game(seed=seed)
        for step in range(60):
            if game.current is None:
                break
            for direction in (1, -1, 2):
                report = R.rotation_report(game, direction)
                actual = game.can_rotate(direction) is not None
                checked += 1
                if report['ok'] != actual:
                    disagree += 1
            from tetrisrl.engine import valid_actions
            from tetrisrl.heuristic import apply_move
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            apply_move(game, (False, actions[step % len(actions)][0],
                              actions[step % len(actions)][1]))
    check('rotation_report agrees with Game.can_rotate on every attempt',
          disagree == 0, f'{disagree}/{checked}')


def test_refused_rotation_is_recorded():
    """The headline case: a rotation that cannot happen must leave a record."""
    game = boxed_in_game()
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, echo=False)
    inp = HumanInput(settings(), rec)
    inp.key_down(pygame.key.key_code('x'), game)      # rotate clockwise
    inp.key_down(pygame.key.key_code('z'), game)      # and counter-clockwise
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    rotations = [e for e in events if e['kind'] == 'rotate']
    check('both attempts were recorded', len(rotations) == 2, len(rotations))
    refused = [e for e in rotations if not e['ok']]
    check('both were refused', len(refused) == 2, [e['ok'] for e in rotations])
    check('the record names the piece and the position',
          refused[0]['piece'] == 'I' and refused[0]['x'] == 4
          and refused[0]['y'] == 6, refused[0])
    check('it says why', refused[0]['reason'] == 'all offsets collide',
          refused[0]['reason'])
    check('it lists every offset that was tried, all failing',
          refused[0]['candidates'] and
          all(not c['ok'] for c in refused[0]['candidates']),
          refused[0]['candidates'])
    check('it notes the piece was walled in',
          refused[0]['left_blocked'] and refused[0]['right_blocked']
          and refused[0]['floor'], refused[0])
    check('and the rotation really did not happen', game.current.rotation == 1)


def test_successful_rotation_records_its_kick():
    game = Game(seed=2)
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, echo=False)
    inp = HumanInput(settings(), rec)
    inp.key_down(pygame.key.key_code('x'), game)
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    rotation = next(e for e in events if e['kind'] == 'rotate')
    check('a successful rotation is recorded as ok', rotation['ok'] is True)
    check('with the offsets it considered', 'candidates' in rotation)


# --- wiring -----------------------------------------------------------------

def test_off_by_default():
    check('HumanInput records nothing unless asked',
          HumanInput(settings()).recorder is None)
    import inspect
    from tetrisrl import render
    for fn in (render.run_human, render.run_battle, render._human_loop,
               render._BattleState.__init__):
        params = inspect.signature(fn).parameters
        check(f'{fn.__name__} defaults recorder to None',
              params.get('recorder') is not None
              and params['recorder'].default is None)


def test_every_event_kind_appears():
    """Driven through the real input handler and one frame of a battle."""
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, meta=R.start_meta('battle', seed=5,
                                                          settings=settings()),
                            state_interval=0.0, echo=False)
    game = Game(seed=5)
    inp = HumanInput(settings(), rec)
    inp.key_down(pygame.key.key_code('down'), game)   # soft drop
    inp.key_down(pygame.key.key_code('left'), game)   # and a horizontal move
    for _ in range(20):
        inp.update(1 / 60, game)
    inp.key_down(pygame.key.key_code('x'), game)      # rotate
    inp.key_down(pygame.key.key_code('space'), game)  # hard drop locks
    rec.state_if_due(game, force=True)
    rec.key(pygame.key.key_code('q'), None, True)      # a key bound to nothing
    rec.close()

    kinds = {json.loads(line)['kind'] for line in buf.getvalue().splitlines()}
    for kind in ('start', 'key', 'move', 'rotate', 'lock', 'state', 'stopped'):
        check(f'a {kind} event is written when it happens', kind in kinds,
              sorted(kinds))
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    lock = next(e for e in events if e['kind'] == 'lock')
    check('the lock names the piece that locked', lock['piece'] == game.current.kind
          or lock['piece'] is not None, lock)
    check('the lock carries the timings in force',
          lock['das_ms'] == 100 and 'lock_delay_ms' in lock, lock)
    state = next(e for e in events if e['kind'] == 'state')
    check('the state snapshot has score, height and holes',
          {'score', 'top', 'holes', 'pieces'} <= set(state), sorted(state))
    missed = [e for e in events if e['kind'] == 'key' and e['action'] is None]
    check('a key that resolves to nothing is still recorded', len(missed) == 1)


def test_battle_attack_events():
    """A battle frame writes attack events with the queue depth."""
    class Stub:
        pps = 20.0
        reaction = 0.0

        def __call__(self, game, actions):
            return actions[-1]

    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, echo=False)
    battle = Battle(seed=3)
    state = _BattleState(battle, Stub(), settings=settings(), recorder=rec)
    keys = [pygame.key.key_code('space')]
    for _ in range(90):
        _battle_step(state, 1 / 60, keys_down=keys, keys_up=[])
    rec.close()
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    kinds = {e['kind'] for e in events}
    check('driving a battle produces lock events', 'lock' in kinds, sorted(kinds))
    attacks = [e for e in events if e['kind'] == 'attack']
    check('and attack events', bool(attacks), sorted(kinds))
    check('an attack records what was sent, cancelled and tanked',
          {'sent', 'cancelled', 'tanked', 'pending'} <= set(attacks[0]),
          sorted(attacks[0]))
    check('both sides are recorded', {a['side'] for a in attacks} == {0, 1},
          {a['side'] for a in attacks})


def test_state_snapshots_are_periodic():
    clock = [0.0]
    buf = io.StringIO()
    rec = R.SessionRecorder(stream=buf, clock=lambda: clock[0],
                            state_interval=5.0, echo=False)
    game = Game(seed=1)
    check('the first snapshot is due immediately', rec.state_if_due(game))
    check('the next one is not', rec.state_if_due(game) is False)
    clock[0] += 4.0
    check('still not after four seconds', rec.state_if_due(game) is False)
    clock[0] += 1.5
    check('but due after five', rec.state_if_due(game))


# --- the analyser -----------------------------------------------------------

def synthetic_session(path):
    """A hand-built session with one refusal, one miss and two locks."""
    events = [
        {'t': 0.0, 'frame': 0, 'kind': 'start', 'version': '1.0.0', 'mode': 'play',
         'seed': 11,
         'settings': {'das_ms': 100, 'arr_ms': 0, 'sdf': 41.0, 'lock_delay_ms': 500},
         'bindings': {'move_left': ['left'], 'rotate_cw': ['x'], 'hold': ['c']},
         'input_problems': []},
        {'t': 0.5, 'frame': 30, 'kind': 'key', 'key': 120, 'name': 'left',
         'action': 'move_left', 'event': 'down'},
        {'t': 0.6, 'frame': 36, 'kind': 'move', 'dx': -1, 'auto': False, 'ok': True},
        {'t': 0.7, 'frame': 42, 'kind': 'move', 'dx': -1, 'auto': True, 'ok': True},
        {'t': 0.72, 'frame': 43, 'kind': 'move', 'dx': -1, 'auto': True, 'ok': True},
        {'t': 0.8, 'frame': 48, 'kind': 'rotate', 'piece': 'I', 'from': 1, 'to': 0,
         'direction': 1, 'x': 4, 'y': 6, 'ok': False, 'kick': None,
         'kick_index': None, 'left_blocked': True, 'right_blocked': True,
         'floor': True, 'reason': 'all offsets collide',
         'candidates': [{'offset': [0, 0], 'ok': False}]},
        {'t': 0.85, 'frame': 51, 'kind': 'key', 'key': 113, 'name': 'q',
         'action': None, 'event': 'down'},
        {'t': 0.9, 'frame': 54, 'kind': 'lock', 'piece': 'T', 'rotation': 0, 'x': 3,
         'y': 18, 'lines': 4, 'combo': 0, 'b2b': False, 'hold': False, 'top': 4,
         'holes': 0},
        {'t': 20.5, 'frame': 1230, 'kind': 'lock', 'piece': 'L', 'rotation': 0,
         'x': 7, 'y': 19, 'lines': 0, 'combo': -1, 'b2b': False, 'hold': False},
        {'t': 21.0, 'frame': 1260, 'kind': 'stopped', 'reason': 'closed',
         'events': 10},
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event) + '\n')
    return path


def test_analyzer_reports_from_a_synthetic_session():
    with tempfile.TemporaryDirectory() as tmp:
        path = synthetic_session(os.path.join(tmp, 'synthetic.jsonl'))
        data = A.report(path)
        summary = data['summary']
        check('the summary counts the pieces', summary['pieces'] == 2,
              summary['pieces'])
        check('and the lines', summary['lines'] == 4, summary['lines'])
        check('and the tetris share', summary['tetris_share'] == 100.0,
              summary['tetris_share'])
        check('the seed is recovered', summary['seed'] == 11, summary['seed'])
        refused = data['refused_rotations']
        check('the refusal is the headline', refused['refused'] == 1, refused)
        check('with its position', refused['by_position'][0]['x'] == 4,
              refused['by_position'])
        check('and its reason', 'all offsets collide' in refused['by_reason'],
              refused['by_reason'])
        check('the context is reported as walled in',
              refused['by_context'].get('wall') == 1, refused['by_context'])
        latency = data['key_latency']['das'].get('move_left')
        check('the observed DAS is measured from the recording',
              latency and abs(latency['median'] - 0.2) < 1e-6, latency)
        arr = data['key_latency']['arr'].get('move_left')
        check('and the observed ARR', arr and abs(arr['median'] - 0.02) < 1e-6, arr)
        bad = data['suspicious']
        check('a key that did nothing is reported',
              bad['key_misses'].get('q') == 1, bad['key_misses'])
        check('a long quiet stretch is reported', bool(bad['long_gaps']),
              bad['long_gaps'])
        check('a piece locked with no input is reported',
              bad['locks_without_input_count'] == 1,
              bad['locks_without_input_count'])


def test_analyzer_prints_and_emits_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = synthetic_session(os.path.join(tmp, 'synthetic.jsonl'))
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            code = A.main([path, '--json', '--top', '3'])
        finally:
            sys.stdout = old
        check('--json exits cleanly', code == 0, code)
        payload = json.loads(buf.getvalue())
        check('--json output has the three sections',
              {'summary', 'refused_rotations', 'key_latency', 'suspicious'}
              <= set(payload), sorted(payload))

        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            A.main([path])
        finally:
            sys.stdout = old
        text = buf.getvalue()
        check('the text report leads with the refusals',
              'refused rotations' in text and 'all offsets collide' in text)
        check('and mentions the settings it was recorded with',
              'DAS 100ms' in text, text[:200])


def suspicious_session(path, clean=False):
    """A session whose *successful* rotations are the problem.

    Three signals, one piece each and a lock between them so they stay separable:
    a T that only turned on the fourth published offset, a Z whose resulting state
    is not the one its direction asks for, and a J rotated twice with no lock in
    between. ``clean=True`` builds the control: one rotation per piece, no offset,
    every state as requested.
    """
    events = [
        {'t': 0.0, 'frame': 0, 'kind': 'start', 'version': '1.0.0', 'mode': 'play',
         'seed': 5, 'settings': {'das_ms': 100, 'arr_ms': 0, 'sdf': 41.0,
                                 'lock_delay_ms': 500},
         'bindings': {'rotate_cw': ['x'], 'rotate_ccw': ['z']},
         'input_problems': []},
    ]
    if clean:
        for index, piece in enumerate('TZJ'):
            events.append({'t': 1.0 + index, 'frame': 60 * index, 'kind': 'rotate',
                           'piece': piece, 'from': 0, 'to': 1, 'direction': 1,
                           'x': 4, 'y': 8, 'ok': True, 'kick': None, 'kick_index': 0,
                           'left_blocked': False, 'right_blocked': False,
                           'floor': False, 'candidates': [{'offset': [0, 0],
                                                           'ok': True}]})
            events.append({'t': 1.1 + index, 'frame': 60 * index + 6, 'kind': 'lock',
                           'piece': piece, 'rotation': 1, 'x': 4, 'y': 18,
                           'lines': 0, 'combo': -1, 'b2b': False, 'hold': False})
    else:
        events += [
            # a far kick: index 3 and an offset that had to shift the piece
            {'t': 1.0, 'frame': 60, 'kind': 'rotate', 'piece': 'T', 'from': 0,
             'to': 1, 'direction': 1, 'x': 3, 'y': 5, 'ok': True, 'kick': [1, -1],
             'kick_index': 3, 'left_blocked': True, 'right_blocked': False,
             'floor': False,
             'candidates': [{'offset': [0, 0], 'ok': False},
                            {'offset': [-1, 0], 'ok': False},
                            {'offset': [1, 0], 'ok': False},
                            {'offset': [1, -1], 'ok': True}]},
            {'t': 1.2, 'frame': 72, 'kind': 'lock', 'piece': 'T', 'rotation': 1,
             'x': 3, 'y': 18, 'lines': 0, 'combo': -1, 'b2b': False, 'hold': False},
            # a rotation that landed in a state its direction did not ask for
            {'t': 2.0, 'frame': 120, 'kind': 'rotate', 'piece': 'Z', 'from': 0,
             'to': 3, 'direction': 1, 'x': 6, 'y': 7, 'ok': True, 'kick': None,
             'kick_index': 0, 'left_blocked': False, 'right_blocked': False,
             'floor': False, 'candidates': [{'offset': [0, 0], 'ok': True}]},
            {'t': 2.2, 'frame': 132, 'kind': 'lock', 'piece': 'Z', 'rotation': 3,
             'x': 6, 'y': 18, 'lines': 0, 'combo': -1, 'b2b': False, 'hold': False},
            # the same piece turned twice with no lock between: a fight
            {'t': 3.0, 'frame': 180, 'kind': 'rotate', 'piece': 'J', 'from': 0,
             'to': 1, 'direction': 1, 'x': 8, 'y': 6, 'ok': True, 'kick': None,
             'kick_index': 0, 'left_blocked': False, 'right_blocked': False,
             'floor': False, 'candidates': [{'offset': [0, 0], 'ok': True}]},
            {'t': 3.2, 'frame': 192, 'kind': 'rotate', 'piece': 'J', 'from': 1,
             'to': 0, 'direction': -1, 'x': 8, 'y': 6, 'ok': True, 'kick': None,
             'kick_index': 0, 'left_blocked': False, 'right_blocked': False,
             'floor': False, 'candidates': [{'offset': [0, 0], 'ok': True}]},
            {'t': 3.4, 'frame': 204, 'kind': 'lock', 'piece': 'J', 'rotation': 0,
             'x': 8, 'y': 18, 'lines': 0, 'combo': -1, 'b2b': False, 'hold': False},
        ]
    events.append({'t': 4.0, 'frame': 240, 'kind': 'stopped', 'reason': 'closed',
                   'events': len(events)})
    with open(path, 'w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event) + '\n')
    return path


def test_analyzer_reports_suspicious_rotations():
    with tempfile.TemporaryDirectory() as tmp:
        data = A.report(suspicious_session(os.path.join(tmp, 'sus.jsonl')))
        spins = data['suspicious_rotations']
        check('the far kick is counted', spins['far_kicks'] == 1, spins)
        check('and attributed to its piece',
              spins['far_kick_by_piece'] == {'T': 1}, spins['far_kick_by_piece'])
        check('with the offset it needed',
              spins['far_kick_examples'][0]['kick_index'] == 3
              and spins['far_kick_examples'][0]['kick'] == [1, -1],
              spins['far_kick_examples'])
        check('the wrong resulting state is reported', spins['wrong_state'] == 1,
              spins['wrong_state'])
        check('and names the state it should have been',
              spins['wrong_state_examples'][0]['to'] == 3
              and (spins['wrong_state_examples'][0]['from']
                   + spins['wrong_state_examples'][0]['direction']) % 4 == 1,
              spins['wrong_state_examples'])
        check('the repeated rotation of one piece is reported',
              spins['bursts'] == 1, spins['bursts'])
        check('and the worst burst is two turns of one piece',
              spins['worst_bursts'][0]['length'] == 2
              and spins['worst_bursts'][0]['piece'] == 'J',
              spins['worst_bursts'])
        check('the kick index distribution is per piece',
              spins['kick_index_by_piece'].get('T') == {3: 1},
              spins['kick_index_by_piece'])
        check('and the session is not called clean', spins['clean'] is False)


def test_analyzer_reports_clean_when_nothing_is_wrong():
    with tempfile.TemporaryDirectory() as tmp:
        path = suspicious_session(os.path.join(tmp, 'clean.jsonl'), clean=True)
        data = A.report(path)
        spins = data['suspicious_rotations']
        check('a clean session reports no far kicks', spins['far_kicks'] == 0, spins)
        check('no wrong states', spins['wrong_state'] == 0, spins)
        check('no repeated rotations', spins['bursts'] == 0, spins)
        check('and is marked clean', spins['clean'] is True)
        check('every rotation is still counted',
              spins['succeeded'] == 3 and spins['attempts'] == 3, spins)
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            A.main([path])
        finally:
            sys.stdout = old
        text = buf.getvalue()
        check('the report says so rather than printing an empty table',
              'successful but suspicious rotations' in text
              and 'nothing suspicious' in text, text[-600:])
        check('and the footer says how to record another session',
              'play --record' in text, text[-200:])


def test_analyzer_latest_flag():
    with tempfile.TemporaryDirectory() as tmp:
        path = suspicious_session(os.path.join(tmp, 'newest.jsonl'))
        original = A.R.newest_session
        A.R.newest_session = lambda *a, **k: path
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            code = A.main(['--latest', '--json'])
        finally:
            sys.stdout = old
            A.R.newest_session = original
        check('--latest exits cleanly', code == 0, code)
        payload = json.loads(buf.getvalue())
        check('--latest analyses the newest session',
              payload['file'] == path, payload['file'])
        check('--json carries the new section',
              'suspicious_rotations' in payload, sorted(payload))


def test_analyzer_handles_an_empty_or_missing_session():
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, 'empty.jsonl')
        open(empty, 'w', encoding='utf-8').close()
        data = A.report(empty)
        check('an empty session summarises without exploding',
              data['summary']['events'] == 0, data['summary'])
        buf = io.StringIO()
        old, sys.stdout, sys.stderr = sys.stdout, buf, buf
        try:
            code = A.main([os.path.join(tmp, 'nope.jsonl')])
        finally:
            sys.stdout, sys.stderr = old, sys.stderr
        check('a missing file is an error, not a crash', code == 1, code)


def main():
    for fn in (test_writes_valid_jsonl,
               test_round_trips_through_disk,
               test_truncated_line_is_skipped,
               test_event_cap_stops_cleanly,
               test_failed_write_disables_recording,
               test_default_path_is_in_sessions,
               test_rotation_report_mirrors_the_engine,
               test_refused_rotation_is_recorded,
               test_successful_rotation_records_its_kick,
               test_off_by_default,
               test_every_event_kind_appears,
               test_battle_attack_events,
               test_state_snapshots_are_periodic,
               test_analyzer_reports_from_a_synthetic_session,
               test_analyzer_reports_suspicious_rotations,
               test_analyzer_reports_clean_when_nothing_is_wrong,
               test_analyzer_latest_flag,
               test_analyzer_prints_and_emits_json,
               test_analyzer_handles_an_empty_or_missing_session):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all recorder tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
