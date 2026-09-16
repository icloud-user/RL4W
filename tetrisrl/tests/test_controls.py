"""Tests for input settings and the settings screen.

Run: python tests/test_controls.py

Two halves: `controls.py` itself (parsing, bindings, serialisation, merging) and
the parts of the renderer that consume it (``HumanInput``'s DAS/ARR/SDF/lock
behaviour and the settings screen's key handling), driven headlessly -- no display
is ever opened, and the drawing check uses an off-screen Surface.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame                                        # noqa: E402

from tetrisrl import controls as C                   # noqa: E402
from tetrisrl import render as R                     # noqa: E402
from tetrisrl.engine import Game                     # noqa: E402

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


def key(name):
    return C.parse_key(name)


def grounded(game):
    """Put the current piece on the floor, so lock delay is what happens next."""
    game.current.y = game.ghost_y()
    return game


# --- parsing and bindings ---------------------------------------------------

def test_parse_key():
    check('a plain name parses', key('left') == pygame.K_LEFT)
    check('lctrl is an alias for left ctrl', key('lctrl') == key('left ctrl'))
    check('esc is an alias for escape', key('esc') == pygame.K_ESCAPE)
    check('space and a bare space agree', key(' ') == key('space') == pygame.K_SPACE)
    check('arrow aliases work', key('arrowup') == key('up') == pygame.K_UP)
    check('shift means the left one', key('shift') == key('left shift'))
    check('a code passes through', C.parse_key(pygame.K_LEFT) == pygame.K_LEFT)
    try:
        key('not a key at all')
        check('a bad name raises ControlsError', False)
    except C.ControlsError:
        check('a bad name raises ControlsError', True)
    try:
        key('')
        check('an empty name raises', False)
    except C.ControlsError:
        check('an empty name raises', True)


def test_defaults_and_doubles():
    s = C.InputSettings()
    check('movement is single-bound by default',
          s.binding_names('move_left') == ['left'])
    check('rotation ships double-bound', len(s.binding_names('rotate_cw')) == 2)
    check('both rotate_cw keys map to the action',
          all(s.action_for(code) == 'rotate_cw' for code in s.codes('rotate_cw')))
    check('action_for ignores unbound keys', s.action_for(pygame.K_F13) is None)
    check('the defaults have no problems', s.problems() == [], s.problems())


def test_rebinding_steals_the_key():
    s = C.InputSettings()
    check('hard drop starts on space', s.binding_names('hard_drop') == ['space'])
    s.set_binding('hold', 0, 'space')
    check('the new binding is in place', 'space' in s.binding_names('hold'))
    check('and it was removed from hard drop', s.binding_names('hard_drop') == [],
          s.binding_names('hard_drop'))
    check('space now performs hold', s.action_for(pygame.K_SPACE) == 'hold')
    s.clear_binding('hold', 0)
    check('clearing removes the slot it was given',
          'space' not in s.binding_names('hold'), s.binding_names('hold'))
    check('but leaves the other slot alone',
          s.binding_names('hold') == ['left shift'], s.binding_names('hold'))
    check('and the key is free again', s.action_for(pygame.K_SPACE) is None)


def test_to_dict_round_trip():
    s = C.InputSettings(das_ms=90, arr_ms=25, sdf=12, lock_delay_ms=250,
                        lock_reset_limit=8)
    s.set_binding('rotate_180', 0, 'q')
    again = C.InputSettings.from_dict(s.to_dict())
    check('timings survive a round trip',
          (again.das_ms, again.arr_ms, again.sdf, again.lock_delay_ms,
           again.lock_reset_limit) == (90, 25, 12, 250, 8),
          (again.das_ms, again.arr_ms, again.sdf))
    check('bindings survive a round trip',
          again.binding_names('rotate_180') == ['q'],
          again.binding_names('rotate_180'))
    check('and the two agree on every action',
          all(s.binding_names(a) == again.binding_names(a) for a in C.ACTION_NAMES))


def test_bad_input_is_clamped_not_fatal():
    s = C.InputSettings(das_ms=-10, arr_ms=-1, lock_delay_ms=-5, sdf=-2)
    check('negative DAS clamps to zero', s.das_ms == 0)
    check('negative ARR clamps to zero', s.arr_ms == 0)
    check('negative lock delay clamps to zero', s.lock_delay_ms == 0)
    check('negative SDF means instant', s.sdf == 0 and s.soft_instant)
    check('and it said so', len(s.warnings) >= 4, s.warnings)
    junk = C.InputSettings.from_dict('not a mapping')
    check('a non-mapping section warns and uses defaults',
          junk.das_ms == C.DEFAULT_TIMING['das_ms'] and junk.warnings, junk.warnings)
    mixed = C.InputSettings.from_dict({
        'bindings': {'move_left': 'left', 'nonsense_action': ['a'],
                     'move_right': 'notakey', 'hold': ['c', 'v', 'b']},
        'das_ms': 'soon'})
    check('a single key name is accepted as a one-item list',
          mixed.binding_names('move_left') == ['left'])
    check('an unknown action is reported, not applied',
          any('nonsense_action' in w for w in mixed.warnings), mixed.warnings)
    check('an unknown key is reported, not applied',
          any('notakey' in w for w in mixed.warnings), mixed.warnings)
    check('extra bindings beyond two are dropped',
          len(mixed.binding_names('hold')) == 2, mixed.binding_names('hold'))


# --- timing -----------------------------------------------------------------

def test_soft_drop_timing():
    instant = C.InputSettings(sdf=0)
    check('SDF 0 is instant', instant.soft_instant)
    check('and has no interval', instant.soft_interval(1) is None)
    cleared = C.InputSettings()
    cleared.sdf = None                  # what a hand-edited settings file can set
    check('SDF None is instant too', cleared.soft_instant)
    fast = C.InputSettings(sdf=41)
    check('a positive SDF is not instant', not fast.soft_instant)
    slow = C.InputSettings(sdf=4)
    check('a bigger factor is a faster drop',
          fast.soft_interval(1) < slow.soft_interval(1),
          (fast.soft_interval(1), slow.soft_interval(1)))
    check('the interval follows gravity down as the level rises',
          fast.soft_interval(10) <= fast.soft_interval(1),
          (fast.soft_interval(1), fast.soft_interval(10)))
    # This used to assert a 1/60 floor "so a frame cannot be outrun", which was
    # the bug rather than the spec: it made every SDF above ~51 identical, so 200x
    # was silently 60x. The frame is no longer the speed limit -- the renderer
    # takes as many rows as the interval earns, up to MAX_SOFT_ROWS_PER_FRAME --
    # so the floor only has to keep the repeat maths finite.
    check('the interval is still floored, but far below a frame',
          0 < fast.soft_interval(20) < 1 / 60.0, fast.soft_interval(20))
    check('a very high SDF stays finite rather than zero',
          C.InputSettings(sdf=1e6).soft_interval(1) > 0,
          C.InputSettings(sdf=1e6).soft_interval(1))
    check('and the renderer caps how many rows one frame may take',
          R.MAX_SOFT_ROWS_PER_FRAME >= 20,
          R.MAX_SOFT_ROWS_PER_FRAME)


def test_seconds_conversions():
    s = C.InputSettings(das_ms=133, arr_ms=0, lock_delay_ms=500)
    check('DAS converts to seconds', abs(s.das - 0.133) < 1e-9, s.das)
    check('ARR 0 stays 0', s.arr == 0.0)
    check('lock delay converts to seconds', abs(s.lock_delay - 0.5) < 1e-9, s.lock_delay)


# --- files ------------------------------------------------------------------

def test_load_merges_settings_over_config():
    with tempfile.TemporaryDirectory() as tmp:
        config = {'input': {'das_ms': 50, 'sdf': 10,
                            'bindings': {'hold': ['v']}}}
        merged = C.load(config, directory=tmp)
        check('config supplies the value when nothing is saved',
              merged.das_ms == 50 and merged.sdf == 10)
        check('and its bindings win too',
              merged.binding_names('hold') == ['v'], merged.binding_names('hold'))
        saved = C.InputSettings(das_ms=200, sdf=3)
        saved.set_binding('hold', 0, 'left shift')
        C.save(saved, directory=tmp)
        loaded = C.load(config, directory=tmp)
        check('settings.yaml overrides the config', loaded.das_ms == 200, loaded.das_ms)
        check('for bindings as well',
              loaded.binding_names('hold') == ['left shift'],
              loaded.binding_names('hold'))
        check('untouched values still come from the config', loaded.sdf == 3)
        check('config.yaml is not what got written',
              not os.path.exists(os.path.join(tmp, 'config.yaml')))


def test_load_never_raises_on_junk():
    with tempfile.TemporaryDirectory() as tmp:
        path = C.settings_path(tmp)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('this: [is not: valid yaml')
        loaded = C.load({'input': {'das_ms': 40}}, directory=tmp)
        check('a broken settings.yaml warns instead of raising', loaded.warnings)
        check('and the config value is still in force', loaded.das_ms == 40)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('input:\n  das_ms: 80\n  bindings:\n    hold: [v]\n')
        loaded = C.load({}, directory=tmp)
        check('a well-formed file is read', loaded.das_ms == 80, loaded.das_ms)
        check('including a flat binding list',
              loaded.binding_names('hold') == ['v'], loaded.binding_names('hold'))


def test_save_then_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        s = C.InputSettings(das_ms=77, arr_ms=11, sdf=0, lock_delay_ms=333)
        s.set_binding('quit', 0, 'escape')
        s.set_binding('pause', 0, 'f1')
        path = C.save(s, directory=tmp)
        check('save writes settings.yaml', os.path.basename(path) == C.SETTINGS_NAME)
        check('and the file explains itself',
              open(path, encoding='utf-8').readline().startswith('#'))
        loaded = C.load({}, directory=tmp)
        check('a saved file round-trips exactly',
              loaded.to_dict() == s.to_dict(),
              (loaded.to_dict(), s.to_dict()))


# --- driven input -----------------------------------------------------------

def test_human_input_moves_and_das():
    s = C.InputSettings(das_ms=100, arr_ms=0, sdf=41)
    game = Game(seed=0)
    inp = R.HumanInput(s)
    start = game.current.x
    check('a bound key is consumed', inp.key_down(key('left'), game))
    check('and moves the piece immediately', game.current.x == start - 1,
          game.current.x)
    check('an unbound key is not consumed', not inp.key_down(pygame.K_F13, game))
    after_press = game.current.x
    inp.update(0.05, game)
    check('DAS holds the piece still before it expires',
          game.current.x == after_press, game.current.x)
    inp.update(0.06, game)
    check('ARR 0 then slides it to the wall', game.current.x == 0, game.current.x)
    check('and the key release is consumed', inp.key_up(key('left'), game))
    check('after which it stops sliding',
          (inp.update(0.1, game), game.current.x == 0)[1])


def test_arr_zero_is_faster_than_a_real_arr():
    def run(arr_ms):
        game = Game(seed=0)
        inp = R.HumanInput(C.InputSettings(das_ms=100, arr_ms=arr_ms))
        inp.key_down(key('left'), game)
        inp.update(0.12, game)          # just past DAS, well inside one ARR tick
        return game.current.x

    instant, stepped = run(0), run(50)
    check('ARR 0 reaches the wall', instant == 0, instant)
    check('a non-zero ARR does not', stepped > instant, (stepped, instant))


def test_doubled_bindings_behave_identically():
    def press(name):
        game = Game(seed=0)
        inp = R.HumanInput(C.InputSettings())
        inp.key_down(key(name), game)
        return game

    by_c, by_shift = press('c'), press('left shift')
    check('both hold keys hold the piece',
          by_c.hold_piece is not None and by_shift.hold_piece is not None)
    check('and they hold the same piece',
          by_c.hold_piece.kind == by_shift.hold_piece.kind)
    cw_up, cw_x = press('up'), press('x')
    check('both clockwise keys rotate the same way',
          cw_up.current.rotation == cw_x.current.rotation == 1,
          (cw_up.current.rotation, cw_x.current.rotation))


def test_rotate_180():
    game = Game(seed=0)
    inp = R.HumanInput(C.InputSettings())
    check('the 180 key is consumed', inp.key_down(key('a'), game))
    check('and spins the piece twice round', game.current.rotation == 2,
          game.current.rotation)


def test_soft_drop_behaviour():
    instant = C.InputSettings(sdf=0)
    game = Game(seed=0)
    inp = R.HumanInput(instant)
    landing = game.ghost_y()
    inp.key_down(key('down'), game)
    check('SDF 0 snaps to the landing row', game.current.y == landing,
          (game.current.y, landing))
    check('without locking the piece', game.pieces_placed == 0)
    check('and it can still be moved afterwards',
          game.move(-1) and game.current.y == landing)

    stepped = C.InputSettings(sdf=41)
    game2 = Game(seed=0)
    inp2 = R.HumanInput(stepped)
    start_y = game2.current.y
    inp2.key_down(key('down'), game2)
    inp2.update(0.05, game2)
    check('a positive SDF only takes a step or two',
          start_y < game2.current.y < game2.ghost_y(),
          (start_y, game2.current.y, game2.ghost_y()))


def test_lock_delay_comes_from_settings():
    s = C.InputSettings(lock_delay_ms=200)
    game = grounded(Game(seed=0))
    inp = R.HumanInput(s)
    inp.update(0.1, game)
    check('a resting piece does not lock early', game.pieces_placed == 0)
    inp.update(0.15, game)
    check('and locks once the configured delay is up', game.pieces_placed == 1)

    slow = C.InputSettings(lock_delay_ms=800)
    game2 = grounded(Game(seed=0))
    inp2 = R.HumanInput(slow)
    inp2.update(0.6, game2)
    check('a longer delay really is longer', game2.pieces_placed == 0)


def test_lock_reset_limit_comes_from_settings():
    s = C.InputSettings(lock_delay_ms=1000, lock_reset_limit=0)
    game = grounded(Game(seed=0))
    inp = R.HumanInput(s)
    inp.lock_timer = 5.0                       # pretend it has been resting a while
    inp._touched()
    check('a zero reset limit refuses to refresh the timer', inp.lock_timer == 5.0)
    s.lock_reset_limit = 15
    inp._touched()
    check('and a real limit refreshes it', inp.lock_timer == 0.0)


# --- the settings screen ----------------------------------------------------

def test_settings_screen_navigation():
    state = R._SettingsState(C.InputSettings())
    check('it starts on the first action',
          state.current['action'] == C.ACTION_NAMES[0], state.current)
    R._settings_step(state, pygame.K_DOWN)
    check('down moves a row', state.current['action'] == C.ACTION_NAMES[1])
    R._settings_step(state, pygame.K_UP)
    check('up moves back', state.current['action'] == C.ACTION_NAMES[0])
    R._settings_step(state, pygame.K_RIGHT)
    check('right selects the second slot', state.slot == 1)
    R._settings_step(state, pygame.K_LEFT)
    check('left selects the first', state.slot == 0)
    for _ in range(len(state.rows)):
        R._settings_step(state, pygame.K_UP)
    check('the selection wraps', 0 <= state.row < len(state.rows), state.row)
    check('every action has a row',
          all(any(r.get('action') == a for r in state.rows)
              for a in C.ACTION_NAMES))
    check('and every timing has one',
          all(any(r.get('key') == t[0] for r in state.rows)
              for t in R.SETTINGS_TIMING))


def test_settings_screen_binding_and_capture():
    state = R._SettingsState(C.InputSettings())
    state.row = [i for i, r in enumerate(state.rows)
                 if r.get('action') == 'hard_drop'][0]
    check('ENTER starts a capture', R._settings_step(state, pygame.K_RETURN) is None
          and state.capture == ('hard_drop', 0), state.capture)
    R._settings_step(state, pygame.K_f)
    check('the next key is bound', state.capture is None)
    check('and it took effect',
          state.settings.binding_names('hard_drop') == ['f'],
          state.settings.binding_names('hard_drop'))
    check('the screen says what happened', 'hard drop' in state.message.lower(),
          state.message)

    R._settings_step(state, pygame.K_RETURN)
    R._settings_step(state, pygame.K_ESCAPE)
    check('ESC cancels a capture rather than leaving', state.capture is None)
    check('and the binding is untouched',
          state.settings.binding_names('hard_drop') == ['f'])

    R._settings_step(state, pygame.K_DELETE)
    check('DELETE clears the slot',
          state.settings.binding_names('hard_drop') == [],
          state.settings.binding_names('hard_drop'))


def test_settings_screen_steals_and_edits():
    state = R._SettingsState(C.InputSettings())
    state.row = [i for i, r in enumerate(state.rows)
                 if r.get('action') == 'hold'][0]
    R._settings_step(state, pygame.K_RETURN)
    R._settings_step(state, pygame.K_SPACE)          # space was hard drop's
    check('binding onto a taken key removes it from its old action',
          state.settings.binding_names('hard_drop') == [],
          state.settings.binding_names('hard_drop'))
    check('and the screen mentions it', 'removed from' in state.message,
          state.message)

    state.row = [i for i, r in enumerate(state.rows) if r.get('key') == 'das_ms'][0]
    before = state.settings.das_ms
    R._settings_step(state, pygame.K_RIGHT)
    check('right nudges a timing value up by its step',
          state.settings.das_ms == before + 5, state.settings.das_ms)
    R._settings_step(state, pygame.K_LEFT)
    check('left nudges it back', state.settings.das_ms == before)
    R._settings_step(state, pygame.K_RIGHT, fine=True)
    check('shift makes the step fine', state.settings.das_ms == before + 1,
          state.settings.das_ms)
    for _ in range(200):
        R._settings_step(state, pygame.K_LEFT)
    check('values clamp at their floor rather than going negative',
          state.settings.das_ms == 0, state.settings.das_ms)

    state.row = [i for i, r in enumerate(state.rows) if r.get('key') == 'sdf'][0]
    for _ in range(250):
        R._settings_step(state, pygame.K_RIGHT)
    cap = [t for t in R.SETTINGS_TIMING if t[0] == 'sdf'][0][5]
    check('and at their ceiling', state.settings.sdf == cap, state.settings.sdf)


def test_settings_screen_saves_and_exits():
    with tempfile.TemporaryDirectory() as tmp:
        old_path = C.settings_path
        C.settings_path = lambda directory=None: os.path.join(tmp, C.SETTINGS_NAME)
        try:
            state = R._SettingsState(C.InputSettings())
            state.row = [i for i, r in enumerate(state.rows)
                         if r.get('key') == 'das_ms'][0]
            R._settings_step(state, pygame.K_RIGHT)
            check('editing marks the state dirty', state.dirty)
            check('S saves', R._settings_step(state, pygame.K_s) == 'save')
            check('and clears the dirty flag', not state.dirty)
            check('the file exists', os.path.exists(C.settings_path()))
            again = C.load({}, directory=tmp)
            check('with the edit in it', again.das_ms == state.settings.das_ms,
                  again.das_ms)
            check('ESC leaves the screen',
                  R._settings_step(state, pygame.K_ESCAPE) == 'exit')
        finally:
            C.settings_path = old_path


def test_settings_screen_draws_headlessly():
    """Draw the settings screen onto an off-screen Surface -- no display."""
    surface = pygame.Surface((900, 700))
    renderer = R.Renderer(surface)
    state = R._SettingsState(C.InputSettings())
    state.message = 'hello'
    state.settings.warnings.append('a deliberate warning')
    state.row = 1
    state.capture = ('hold', 0)              # exercises the "press a key" branch
    try:
        R._settings_draw(surface, renderer, state)
        R._settings_draw(surface, renderer, R._SettingsState(C.InputSettings()))
        check('the settings screen draws without a display', True)
    except Exception as exc:                  # noqa: BLE001 - report anything
        check('the settings screen draws without a display', False, repr(exc))


def test_menu_offers_the_settings_screen():
    """The menu hands back a settings choice, driven without any display.

    ``pygame.event.get`` and ``display.flip`` are stubbed, so the real menu loop
    runs against a bare Surface: this is what catches the wiring (a menu that
    offers Settings but calls the mode with an argument it does not accept).
    """
    from types import SimpleNamespace

    real_get, real_flip = pygame.event.get, pygame.display.flip
    scripted = [[SimpleNamespace(type=pygame.KEYDOWN, key=key)]
                for key in (pygame.K_DOWN, pygame.K_DOWN, pygame.K_DOWN,
                            pygame.K_RETURN)]
    pygame.event.get = lambda *a, **k: scripted.pop(0) if scripted else []
    pygame.display.flip = lambda *a, **k: None
    try:
        screen = pygame.Surface((900, 700))
        choice = R._menu_loop(screen, R.Layout(), C.InputSettings())
    finally:
        pygame.event.get, pygame.display.flip = real_get, real_flip
    check('the menu offers Settings as its fourth entry',
          choice == ('settings', None), choice)

    real_get, real_flip = pygame.event.get, pygame.display.flip
    scripted = [[SimpleNamespace(type=pygame.KEYDOWN, key=key)]
                for key in (pygame.K_DOWN, pygame.K_DOWN, pygame.K_DOWN,
                            pygame.K_DOWN, pygame.K_RETURN)]
    pygame.event.get = lambda *a, **k: scripted.pop(0) if scripted else []
    pygame.display.flip = lambda *a, **k: None
    try:
        choice = R._menu_loop(pygame.Surface((900, 700)), R.Layout(),
                              C.InputSettings())
    finally:
        pygame.event.get, pygame.display.flip = real_get, real_flip
    check('and Quit is still last', choice is None, choice)


def test_soft_drop_instant():
    """Infinitely fast soft drop: the flag, the legacy spellings, and the speed."""
    soft = C.parse_key('down')

    # Three ways to ask for the same thing. ``sdf: null`` used to do nothing at
    # all: the constructor reads None as "unspecified", so only 0 ever worked.
    by_zero = C.InputSettings.from_dict({'sdf': 0})
    by_null = C.InputSettings.from_dict({'sdf': None})
    by_flag = C.InputSettings.from_dict({'sdf': 41, 'soft_drop_instant': True})
    absent = C.InputSettings.from_dict({})
    check('sdf 0 means instant', by_zero.soft_instant)
    check('sdf null means instant', by_null.soft_instant)
    check('the soft_drop_instant flag means instant', by_flag.soft_instant)
    check('an absent sdf is still an ordinary factor', not absent.soft_instant)
    check('instant has no interval to interpolate',
          by_flag.soft_interval(1) is None)

    def rows(sdf, frames=1):
        """Rows the piece falls in ``frames`` frames of held soft drop."""
        game = Game(seed=5)
        player = R.HumanInput(C.InputSettings(sdf=sdf))
        start = game.current.y
        player.key_down(soft, game)
        for _ in range(frames):
            player.update(1 / 60, game)
        return game.current.y - start

    # The old code took at most one row per frame, so every SDF above ~51 was
    # identical and 200x felt exactly like 60x.
    check('a large SDF moves several rows in one frame', rows(200) >= 3, rows(200))
    check('and is strictly faster than a mid SDF', rows(200) > rows(60), 
          (rows(200), rows(60)))
    check('a low SDF still moves at most one row per frame', rows(1, 10) <= 1,
          rows(1, 10))
    check('SDF above the old ceiling is no longer all the same speed',
          C.InputSettings(sdf=200).soft_interval(1)
          < C.InputSettings(sdf=60).soft_interval(1))

    # Instant teleports to the landing row without locking, which is the semantic
    # that makes it usable for spinning a piece into a slot.
    game = Game(seed=5)
    for x in range(game.cols):                     # a stack to land on
        for y in range(18, game.rows):
            game.board.grid[y][x] = 'X'
    player = R.HumanInput(C.InputSettings(sdf=0))
    before = game.pieces_placed
    ghost = game.ghost_y()
    player.key_down(soft, game)
    check('instant lands exactly on the ghost row',
          game.current is not None and game.current.y == ghost,
          (None if game.current is None else game.current.y, ghost))
    check('instant does not place the piece', game.pieces_placed == before,
          game.pieces_placed)
    check('instant does not end the game on a resting piece', not game.game_over)
    player.update(1 / 60, game)
    check('and the piece stays put on the following frame',
          game.current is not None and game.current.y == ghost
          and not game.game_over)

    check('the flag round-trips through a config dict',
          C.InputSettings.from_dict(
              C.InputSettings(sdf=41, soft_drop_instant=True).to_dict()
          ).soft_drop_instant)
    check('and the word "off" in a config file is not truthy',
          not C.InputSettings.from_dict(
              {'soft_drop_instant': 'off'}).soft_drop_instant)

    state = R._SettingsState(C.InputSettings(sdf=41))
    while state.current.get('key') != 'sdf':
        state.move(1)
    R._settings_step(state, pygame.K_i)
    check('the settings screen reaches instant in one press',
          state.settings.soft_instant)
    check('and the row says "instant" rather than a number',
          R._timing_text(state.settings, state.current)[0] == 'instant',
          R._timing_text(state.settings, state.current))
    R._settings_step(state, pygame.K_i)
    check('and pressing it again gives the factor back',
          not state.settings.soft_instant and state.settings.sdf == 41,
          state.settings.sdf)


# --- DCD (DAS Cut Delay) ----------------------------------------------------

DCD_DT = 1 / 60.0
#: Frame the rotation lands on in the scripted runs below.
DCD_ROTATE_AT = 10


def dcd_run(dcd_ms, rotate_at=DCD_ROTATE_AT, frames=40, das_ms=0, arr_ms=50):
    """Hold left from frame 2, optionally rotate, and record x per frame.

    DAS 0 and ARR 50 ms on purpose: the repeat fires every ~3 frames, so a freeze
    of a few hundred milliseconds is unmistakable against it.
    """
    settings = C.InputSettings(das_ms=das_ms, arr_ms=arr_ms, dcd_ms=dcd_ms)
    game = Game(seed=3)
    inp = R.HumanInput(settings)
    xs, timers = [], []
    for frame in range(frames):
        if frame == 2:
            inp.key_down(key('left'), game)
        if rotate_at is not None and frame == rotate_at:
            inp.key_down(key('x'), game)
        inp.update(DCD_DT, game)
        xs.append(game.current.x)
        timers.append(inp.dcd_timer)
    return xs, timers, inp, game


def first_change(xs, after):
    """Index of the first frame after ``after`` where x differs from its value."""
    for i in range(after + 1, len(xs)):
        if xs[i] != xs[after]:
            return i
    return None


def test_dcd_defaults_and_conversion():
    check('DCD is off by default',
          C.InputSettings().dcd_ms == 0 and C.InputSettings().dcd == 0.0)
    check('DCD converts ms to seconds', C.InputSettings(dcd_ms=200).dcd == 0.2)
    check('DCD has no upper cap', C.InputSettings(dcd_ms=5000).dcd_ms == 5000)
    data = C.InputSettings(dcd_ms=250).to_dict()
    check('DCD is serialised', data['dcd_ms'] == 250, data.get('dcd_ms'))
    check('and read back', C.InputSettings.from_dict(data).dcd_ms == 250)
    negative = C.InputSettings(dcd_ms=-40)
    check('a negative DCD clamps to 0',
          negative.dcd_ms == 0, negative.dcd_ms)
    check('with a warning naming it',
          any('dcd_ms' in w for w in negative.warnings), negative.warnings)
    junk = C.InputSettings.from_dict({'dcd_ms': 'soon'})
    check('a junk DCD falls back to the default',
          junk.dcd_ms == C.DEFAULT_TIMING['dcd_ms'] and bool(junk.warnings),
          (junk.dcd_ms, junk.warnings))
    check('the startup line mentions DCD',
          'DCD' in R.settings_summary(C.InputSettings(dcd_ms=200)),
          R.settings_summary(C.InputSettings(dcd_ms=200)))


def test_dcd_off_changes_nothing():
    """dcd_ms=0 must repeat exactly as it always did: no freeze anywhere."""
    xs, timers, _inp, _game = dcd_run(0)
    check('the freeze timer never arms at DCD 0', set(timers) == {0.0}, set(timers))
    check('the piece keeps sliding after the rotation frame',
          first_change(xs, DCD_ROTATE_AT) is not None
          and first_change(xs, DCD_ROTATE_AT) <= DCD_ROTATE_AT + 3,
          (DCD_ROTATE_AT, first_change(xs, DCD_ROTATE_AT)))
    gaps = [b - a for a, b in zip(_changes(xs), _changes(xs)[1:])
            if b < DCD_ROTATE_AT]           # before the rotation, which can kick x
    check('at the ARR cadence, every ~3 frames', set(gaps) <= {3}, gaps)


def _changes(xs):
    return [i for i in range(1, len(xs)) if xs[i] != xs[i - 1]]


def test_dcd_freezes_repeat_after_a_rotation():
    frozen, timers, inp, _game = dcd_run(200)
    free, _t, _i, _g = dcd_run(0)
    at = DCD_ROTATE_AT
    check('the rotation arms the freeze', timers[at] > 0.0, timers[at])
    check('and it is armed for the configured time',
          abs(timers[at] - 200 / 1000.0) <= DCD_DT, timers[at])
    window = range(at, at + 11)                  # ~180 ms of the 200 ms window
    check('the piece does not slide during the freeze',
          len({frozen[i] for i in window}) == 1,
          [frozen[i] for i in window])
    check('while without DCD it does slide in the same window',
          len({free[i] for i in window}) > 1, [free[i] for i in window])
    resumed = first_change(frozen, at)
    check('the repeat resumes after the window, and promptly',
          resumed is not None and at + 12 <= resumed <= at + 16,
          (at, resumed))
    check('the freeze decays to zero rather than sticking',
          all(t >= 0.0 for t in timers) and timers[-1] == 0.0, timers[-1])


def test_dcd_freezes_repeat_after_a_spawn():
    settings = C.InputSettings(das_ms=0, arr_ms=50, dcd_ms=200)
    game = Game(seed=3)
    inp = R.HumanInput(settings)
    inp.key_down(key('left'), game)
    inp.update(DCD_DT, game)                     # DAS 0: one repeat fires at once
    x_at_spawn = game.current.x
    check('the piece still has room to move', x_at_spawn > 0, x_at_spawn)
    inp.reset()                                  # what a spawn/hard drop/hold calls
    check('a new piece arms the freeze',
          abs(inp.dcd_timer - 0.2) <= DCD_DT, inp.dcd_timer)
    xs = []
    for _ in range(24):
        inp.update(DCD_DT, game)
        xs.append(game.current.x)
    check('a held direction does not slide the new piece',
          set(xs[:11]) == {x_at_spawn}, xs[:12])
    check('and it starts moving once the window closes', xs[-1] < x_at_spawn,
          (x_at_spawn, xs[-1]))


def test_dcd_does_not_gate_a_fresh_press():
    """DCD cuts the repeat, not the first step: a tap must still move one cell."""
    settings = C.InputSettings(das_ms=0, arr_ms=50, dcd_ms=1000)
    game = Game(seed=3)
    inp = R.HumanInput(settings)
    inp.key_down(key('left'), game)
    inp.key_down(key('x'), game)                 # rotate: a long freeze
    frozen_x = game.current.x
    for _ in range(3):
        inp.update(DCD_DT, game)
    check('the freeze is still running', inp.dcd_timer > 0.5, inp.dcd_timer)
    check('a fresh press is consumed', inp.key_down(key('right'), game))
    check('and moves the piece one cell during the freeze',
          game.current.x == frozen_x + 1, (frozen_x, game.current.x))


def test_dcd_leaves_soft_drop_gravity_and_lock_alone():
    settings = C.InputSettings(das_ms=0, arr_ms=50, dcd_ms=1000, sdf=41)
    game = Game(seed=3)
    inp = R.HumanInput(settings)
    inp.key_down(key('x'), game)                 # freeze
    start_y = game.current.y
    for _ in range(10):
        inp.update(DCD_DT, game)
    check('gravity still pulls the piece down during the freeze',
          game.current.y > start_y, (start_y, game.current.y))
    check('and the freeze is still counting', inp.dcd_timer > 0.5, inp.dcd_timer)
    inp.key_down(key('down'), game)              # soft drop, still frozen
    y_before = game.current.y
    for _ in range(6):
        inp.update(DCD_DT, game)
    check('soft drop still runs during the freeze',
          game.current.y > y_before, (y_before, game.current.y))

    # Lock delay is a separate clock and must not be paused by DCD either.
    late = C.InputSettings(das_ms=0, arr_ms=50, dcd_ms=5000,
                           lock_delay_ms=100)
    board = grounded(Game(seed=3))
    player = R.HumanInput(late)
    player.key_down(key('x'), board)             # arm a 5 s freeze
    locked = False
    for _ in range(10):
        locked = player.update(DCD_DT, board) or locked
    check('the lock timer still runs during a long freeze', locked)


def test_settings_screen_dcd_row():
    state = R._SettingsState(C.InputSettings())
    rows = [i for i, r in enumerate(state.rows) if r.get('key') == 'dcd_ms']
    check('the settings screen has a DCD row', len(rows) == 1, rows)
    state.row = rows[0]
    check('labelled DCD, in ms',
          state.current['label'] == 'DCD' and state.current['unit'] == 'ms',
          state.current)
    start = state.settings.dcd_ms
    R._settings_step(state, pygame.K_RIGHT)
    check('right nudges it by 5 ms',
          state.settings.dcd_ms == start + 5, state.settings.dcd_ms)
    R._settings_step(state, pygame.K_RIGHT, fine=True)
    check('shift-right nudges by 1 ms',
          state.settings.dcd_ms == start + 6, state.settings.dcd_ms)
    for _ in range(40):
        R._settings_step(state, pygame.K_LEFT)
    check('and it cannot be pushed below zero',
          state.settings.dcd_ms == 0, state.settings.dcd_ms)
    for _ in range(500):
        R._settings_step(state, pygame.K_RIGHT)
    check('there is no ceiling on it',
          state.settings.dcd_ms == 2500, state.settings.dcd_ms)
    check('the row renders its value',
          R._timing_text(state.settings, state.current)[0] == '2500 ms',
          R._timing_text(state.settings, state.current))


def test_settings_rows_fit_with_dcd():
    """The list grew to 17 rows: it must still fit a small window.

    ``_settings_row_height`` shares the room above the footer between the rows,
    with a 9 px floor -- and a floor can overshoot when the window is short, which
    is exactly the failure the fitter was added to prevent.
    """
    for cell in (10, 30):
        config = {'game': {'cell_size': cell, 'side_panel_width': 200}}
        layout = R.Layout.from_config(config)
        surface = pygame.Surface((layout.width, layout.height))
        renderer = R.Renderer(surface, layout, config)
        state = R._SettingsState(C.InputSettings())
        top = layout.margin + 4 + renderer._height(renderer.big, 22) + 6
        row_h = R._settings_row_height(renderer, len(state.rows), top)
        used = row_h * len(state.rows)
        room = layout.height - top - (2 * max(14, renderer._height(renderer.small, 12) + 4) + 10)
        check(f'the {len(state.rows)} settings rows fit at cell {cell}',
              used <= room, (used, room, row_h))
        try:
            R._settings_draw(surface, renderer, state)
            check(f'and the screen draws at cell {cell}', True)
        except Exception as exc:              # noqa: BLE001 - report anything
            check(f'and the screen draws at cell {cell}', False, repr(exc))


def main():
    for fn in (test_parse_key,
               test_defaults_and_doubles,
               test_rebinding_steals_the_key,
               test_to_dict_round_trip,
               test_bad_input_is_clamped_not_fatal,
               test_soft_drop_timing,
               test_soft_drop_instant,
               test_seconds_conversions,
               test_load_merges_settings_over_config,
               test_load_never_raises_on_junk,
               test_save_then_load_round_trip,
               test_human_input_moves_and_das,
               test_arr_zero_is_faster_than_a_real_arr,
               test_doubled_bindings_behave_identically,
               test_rotate_180,
               test_soft_drop_behaviour,
               test_lock_delay_comes_from_settings,
               test_lock_reset_limit_comes_from_settings,
               test_settings_screen_navigation,
               test_settings_screen_binding_and_capture,
               test_settings_screen_steals_and_edits,
               test_settings_screen_saves_and_exits,
               test_settings_screen_draws_headlessly,
               test_menu_offers_the_settings_screen,
               test_dcd_defaults_and_conversion,
               test_dcd_off_changes_nothing,
               test_dcd_freezes_repeat_after_a_rotation,
               test_dcd_freezes_repeat_after_a_spawn,
               test_dcd_does_not_gate_a_fresh_press,
               test_dcd_leaves_soft_drop_gravity_and_lock_alone,
               test_settings_screen_dcd_row,
               test_settings_rows_fit_with_dcd):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all controls tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
