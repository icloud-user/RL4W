"""Tests for the beam search. Run: python tests/test_search.py

The differential tests are the important ones: the search runs on its own column
bitmask model rather than the engine's grid, and a fast model that disagrees with
``Game.simulate_placement`` would make every measurement in the README a
measurement of the wrong thing. They are the same style of guard the engine's own
enumeration is held to.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import search as S                    # noqa: E402
from tetrisrl.engine import (Game, Piece, spawn_anchor,  # noqa: E402
                             valid_actions)
from tetrisrl.heuristic import apply_move           # noqa: E402

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


def grid_masks(grid, rows, cols):
    """The engine's grid in the bitmask form the search uses."""
    return tuple(sum(1 << (rows - 1 - y) for y in range(rows)
                     if grid[y][x] is not None) for x in range(cols))


def random_states(trials=40, steps=25, seed=0):
    """Yield reachable (game) states by playing random legal moves."""
    rng = random.Random(seed)
    for _ in range(trials):
        game = Game(seed=rng.randrange(10 ** 6))
        for _ in range(steps):
            if game.current is None or game.game_over:
                break
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            yield game, actions
            rot, x = rng.choice(actions)
            apply_move(game, (False, rot, x))


def make_board(heights, kind='I', seed=0):
    """A Game whose columns are filled solidly to ``heights``, with ``kind`` in hand."""
    game = Game(seed=seed)
    for x, h in enumerate(heights):
        for y in range(game.rows - h, game.rows):
            game.board.grid[y][x] = 'X'
    game.current = Piece(kind, *spawn_anchor(kind))
    game.next_queue.clear()
    for k in ('J', 'L', 'S', 'Z', 'T'):
        game.next_queue.append(Piece(k))
    return game


# --- the bitboard model -----------------------------------------------------

def test_board_masks_match_grid():
    bad = 0
    n = 0
    for game, _actions in random_states():
        n += 1
        if S.board_masks(game) != grid_masks(game.board.grid, game.rows, game.cols):
            bad += 1
    check('board_masks reproduces the engine grid', bad == 0, f'{bad}/{n}')


def test_placements_match_valid_actions():
    """The fast enumerator must offer exactly what the engine offers."""
    bad = 0
    n = 0
    for game, actions in random_states():
        masks = S.board_masks(game)
        heights = S.heights_of(masks)
        mine = {(r, x) for r, x, _y in
                S.placements(masks, heights, game.current.kind,
                             game.rows, game.cols)}
        n += 1
        if mine != set(actions):
            bad += 1
    check('placements() agrees with engine.valid_actions', bad == 0,
          f'{bad}/{n} states disagree')


def test_apply_placement_matches_simulate_placement():
    """Every legal placement must produce the board the engine would produce."""
    bad = 0
    n = 0
    for game, actions in random_states():
        masks = S.board_masks(game)
        heights = S.heights_of(masks)
        for rot, x in actions:
            y = S.landing_y(heights, game.current.kind, rot, x, game.rows)
            got, cleared = S.apply_placement(masks, game.current.kind, rot, x, y,
                                             game.rows)
            grid, _cells, full = game.simulate_placement(rot, x)
            n += 1
            if got != grid_masks(grid, game.rows, game.cols) or cleared != len(full):
                bad += 1
    check('apply_placement() agrees with simulate_placement()', bad == 0,
          f'{bad}/{n} placements disagree')


def test_landing_row_matches_ghost():
    bad = 0
    n = 0
    for game, actions in random_states(trials=20):
        heights = S.heights_of(S.board_masks(game))
        for rot, x in actions:
            probe = Piece(game.current.kind, *spawn_anchor(game.current.kind))
            probe.rotation, probe.x = rot, x
            n += 1
            if S.landing_y(heights, game.current.kind, rot, x, game.rows) != \
                    game.ghost_y(probe):
                bad += 1
    check('landing_y() agrees with the engine ghost row', bad == 0, f'{bad}/{n}')


def test_spawn_blocked_matches_engine():
    """``spawn_blocked`` must reproduce the engine's own spawn test exactly.

    Note what the engine actually does: ``Board.collides`` treats cells above the
    playfield as free, and every piece spawns at a negative row, so a board filled
    to the ceiling still reports a playable spawn. That is a known property of this
    engine (games end by lock-out, not block-out), and the search has to model the
    engine it plays in rather than the Guideline rule it would prefer. The test is
    therefore a differential, not an assertion about Tetris.
    """
    bad = 0
    n = 0
    boards = [make_board([0] * 10), make_board([5, 3, 7, 2, 6, 4, 8, 1, 5, 0]),
              make_board([22] * 10), make_board([21] * 9 + [0])]
    for game in boards:
        masks = S.board_masks(game)
        for kind in ('I', 'J', 'L', 'O', 'S', 'T', 'Z'):
            probe = Piece(kind, *spawn_anchor(kind))
            n += 1
            if S.spawn_blocked(masks, kind, game.rows, game.cols) != \
                    game.board.collides(probe.cells()):
                bad += 1
    check('spawn_blocked() agrees with the engine spawn test', bad == 0,
          f'{bad}/{n}')


def test_well_rules():
    """Both well definitions, on boards built by hand."""
    heights = [4, 4, 4, 4, 4, 4, 4, 4, 4, 0]
    game = make_board(heights)
    masks = S.board_masks(game)
    live = S.heights_of(masks)
    depth, well = S.ready_depth(masks, live, game.cols, game.rows, 4)
    check('ready_depth finds a four-deep well on a flat stack',
          depth == 4 and well == 9, f'{depth} at column {well}')
    depth, well = S.well_depth(masks, live, game.cols, game.rows, 4)
    check('well_depth finds the same well', depth == 4 and well == 9,
          f'{depth} at column {well}')

    game = make_board([4, 4, 4, 4, 4, 4, 4, 4, 4, 1])
    masks = S.board_masks(game)
    depth, _well = S.ready_depth(masks, S.heights_of(masks), game.cols,
                                 game.rows, 4)
    check('a three-row well scores three, not four', depth == 3, depth)

    game = make_board([6, 6, 6, 6, 6, 6, 6, 6, 6, 1])
    masks = S.board_masks(game)
    depth, _well = S.ready_depth(masks, S.heights_of(masks), game.cols,
                                 game.rows, 4)
    check('well depth is capped at the four rows an I can clear', depth == 4, depth)


# --- the search -------------------------------------------------------------

def test_search_returns_a_legal_move():
    """Whatever the search picks must be executable, hold or not."""
    bad = 0
    n = 0
    for game, actions in random_states(trials=20, steps=8):
        move = S.search_move(game, actions, depth=2, beam=4)
        n += 1
        if move is None:
            bad += 1
            continue
        plain = tuple(move[-2:])
        is_hold = len(move) == 3 and bool(move[0])
        if not is_hold and plain not in {tuple(a) for a in actions}:
            bad += 1
            continue
        placed_before = game.pieces_placed
        apply_move(game, move)
        if game.pieces_placed != placed_before + 1:
            bad += 1
    check('search_move returns an executable move', bad == 0, f'{bad}/{n}')


def test_search_takes_an_available_tetris():
    """With a four-deep well and an I in hand, the tetris must be taken."""
    game = make_board([4, 4, 4, 4, 4, 4, 4, 4, 4, 0], kind='I')
    actions = valid_actions(game.board, game.current)
    move = S.search_move(game, actions, depth=2, beam=6)
    cleared = apply_move(game, move)
    check('search takes a tetris that is available now', cleared == 4,
          f'cleared {cleared} with {move}')


def test_policy_uses_the_hold_slot():
    """Hold is the lever that makes the tetris rate jump, so it must be used.

    Banking an I for a tetris that is still three pieces away is a prediction no
    one-ply score can express -- the README measured hold as worthless for the
    greedy evaluator for exactly that reason. Over a game the search must spend the
    slot, and this pins that it does rather than trusting the claim.
    """
    from tetrisrl.policies import make_search_policy

    policy = make_search_policy(weights='tetris', depth=3, beam=6)
    game = Game(seed=7)
    holds = 0
    tetrises = 0
    pieces = 0
    while not game.game_over and pieces < 200:
        actions = valid_actions(game.board, game.current)
        if not actions:
            break
        move = policy(game, actions)
        if len(move) == 3 and move[0]:
            holds += 1
        if apply_move(game, move) == 4:
            tetrises += 1
        pieces += 1
    check('the search spends the hold slot', holds > 0,
          f'{holds} holds in {pieces} pieces')
    check('and the hold slot buys tetrises', tetrises > 0,
          f'{tetrises} tetrises in {pieces} pieces')


def test_search_is_deterministic():
    game = make_board([5, 4, 6, 3, 5, 7, 4, 6, 5, 0], kind='T')
    actions = valid_actions(game.board, game.current)
    first = S.search_move(game, actions, depth=3, beam=6)
    second = S.search_move(game, actions, depth=3, beam=6)
    check('the same board yields the same move', first == second,
          f'{first} vs {second}')


def test_search_avoids_topping_out():
    """When one legal move ends the game and another does not, take the other.

    Column 0 is left empty while the rest of the field is stacked to row 1, so a
    vertical I drops safely into column 0 while a horizontal one comes to rest in
    the buffer and locks the game out.
    """
    game = make_board([0] + [21] * 9, kind='I')
    actions = valid_actions(game.board, game.current)
    heights = S.heights_of(S.board_masks(game))

    def fatal(move):
        if len(move) == 3 and move[0]:
            return False              # a hold move plays another piece: not judged
        rot, x = move[-2], move[-1]
        y = S.landing_y(heights, 'I', rot, x, game.rows)
        return y is None or S.locks_out('I', rot, y, game.buffer_rows)

    survivors = [a for a in actions if not fatal((False,) + tuple(a))]
    if not survivors:
        check('search prefers a surviving placement when one exists', True,
              '(no surviving placement on this board)')
        return
    move = S.search_move(game, actions, depth=2, beam=6)
    check('search prefers a surviving placement when one exists',
          not fatal(move), f'chose {move}, survivors={survivors}')


def test_search_alone_is_not_enough():
    """Search without the tetris terms is not the thing doing the work.

    A control for the README's claim: the same beam search over plain board
    quality produces a far worse clear mix than the tetris weights. Measured on a
    fixed seed over a short game, so the bar is deliberately loose.
    """
    from tetrisrl.policies import make_search_policy

    def share(policy, pieces):
        game = Game(seed=11)
        clears = {1: 0, 2: 0, 3: 0, 4: 0}
        n = 0
        while not game.game_over and n < pieces:
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            cleared = apply_move(game, policy(game, actions))
            if cleared:
                clears[cleared] += 1
            n += 1
        events = sum(clears.values())
        return 100.0 * clears[4] / events if events else 0.0

    tetris = make_search_policy(weights='tetris', depth=2, beam=6)
    plain = make_search_policy(weights='default', depth=2, beam=6)
    a = share(tetris, 400)
    b = share(plain, 400)
    check('the tetris weights beat the plain-search control on tetris share',
          a > b, f'{a:.1f}% vs {b:.1f}%')
    check('the tetris weights clear a real share of tetrises', a >= 10.0,
          f'{a:.1f}% of clears')


def test_policy_plays_a_game():
    """End to end through the registry: load, play, clear lines, do not crash."""
    from tetrisrl.policies import load_policy

    policy, desc = load_policy('search_tetris')
    game = Game(seed=5)
    cleared_total = 0
    pieces = 0
    while not game.game_over and pieces < 250:
        actions = valid_actions(game.board, game.current)
        if not actions:
            break
        cleared_total += apply_move(game, policy(game, actions))
        pieces += 1
    check('search_tetris loads from the registry', 'search' in desc, desc)
    check('search_tetris plays and clears lines',
          pieces > 50 and cleared_total > 0,
          f'{pieces} pieces, {cleared_total} lines')


def test_weight_presets():
    tetris = S.resolve_search_weights('tetris')
    plain = S.resolve_search_weights('default')
    cc = S.resolve_search_weights('cold_clear')
    check('the tetris preset prices a single below a tetris',
          tetris['clear1'] < 0 < tetris['clear4'])
    check('the plain preset has no clear shaping',
          plain['clear1'] == 0 and plain['clear4'] == 0)
    check('the Cold Clear preset keeps their published clear values',
          cc['clear1'] == -143 and cc['clear4'] == 390)
    try:
        S.resolve_search_weights('nope')
        check('an unknown preset raises', False)
    except ValueError:
        check('an unknown preset raises', True)


def main():
    for fn in (test_board_masks_match_grid,
               test_placements_match_valid_actions,
               test_apply_placement_matches_simulate_placement,
               test_landing_row_matches_ghost,
               test_spawn_blocked_matches_engine,
               test_well_rules,
               test_search_returns_a_legal_move,
               test_search_takes_an_available_tetris,
               test_policy_uses_the_hold_slot,
               test_search_is_deterministic,
               test_search_avoids_topping_out,
               test_search_alone_is_not_enough,
               test_policy_plays_a_game,
               test_weight_presets):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all search tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
