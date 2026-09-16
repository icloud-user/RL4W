"""Correctness tests for the engine. Run: python tests/test_engine.py"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import engine as E

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


def cells_of(rot_cells):
    """Normalise a cell set to a comparable form (sorted tuples)."""
    return sorted(rot_cells)


def cw_about_origin(point):
    """Rotate a single point a quarter turn clockwise in the board frame.

    The board frame has y growing *down*, and in it this map is a clockwise
    quarter turn: the "up" direction (0, -1) becomes (1, 0), i.e. right. (The same
    formula is counter-clockwise in a y-up frame, which is a good way to get the
    shape table subtly mirrored -- see ``test_box_convention``.)

    The table stores each state in its own box, so a rotated result has to be
    translated back before it can be compared.
    """
    x, y = point
    return (-y, x)


def centre_norm(cells):
    """Cell set normalised about its centroid.

    NOTE: this is *not* rotation invariant, and deliberately not used for shape
    comparison -- a 90-degree turn changes the centroid-relative offsets of the
    cells for I/J/L/T. Use ``distance_signature`` for rotation-invariant
    comparisons.
    """
    cx = sum(x for x, _ in cells) / len(cells)
    cy = sum(y for _, y in cells) / len(cells)
    return sorted((x - cx, y - cy) for x, y in cells)


def distance_signature(cells):
    """Rotation- and translation-invariant fingerprint of a set of 4 cells.

    The multiset of the 6 pairwise squared distances determines a 4-point set up
    to rigid motion, so every rotation of a piece shares a signature. Note this
    does *not* separate mirror-image pairs (L/J, S/Z) -- those are distinguished
    by their cell layouts instead.
    """
    import itertools
    return sorted(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        for a, b in itertools.combinations(cells, 2)
    )


def norm_shift(cells):
    """Translate a cell set so its bounding box starts at the origin."""
    mx = min(x for x, _ in cells)
    my = min(y for _, y in cells)
    return sorted((x - mx, y - my) for x, y in cells)


def test_shapes_are_tetrominoes():
    print('shapes:')
    for kind in E.PIECES:
        for rot in range(4):
            cs = E.SHAPES[kind][rot]
            ok = len(cs) == 4 and len(set(cs)) == 4
            check(f'{kind} rot{rot} has 4 distinct cells', ok, cs)
        # All four rotations must be the same rigid shape. The distance
        # signature is the correct invariant; centroid or bounding-box
        # normalisation is not (a 90-degree turn shifts them for I/J/L/T).
        base = distance_signature(E.SHAPES[kind][0])
        for rot in (1, 2, 3):
            check(f'{kind} rot{rot} is the same tetromino as rot0',
                  distance_signature(E.SHAPES[kind][rot]) == base,
                  f'\n     rot0 {base}\n     rot{rot} '
                  f'{distance_signature(E.SHAPES[kind][rot])}')
    # L and J are mirror images: identical pairwise distances, so they cannot be
    # separated by a distance signature, but their cell layouts must differ.
    check('L and J have the same distance signature (they are mirrors)',
          distance_signature(E.SHAPES['L'][0]) == distance_signature(E.SHAPES['J'][0]))
    check('L and J have different cell layouts',
          E.SHAPES['L'][0] != E.SHAPES['J'][0])
    check('S and Z have different cell layouts',
          E.SHAPES['S'][0] != E.SHAPES['Z'][0])
    # Chiral pairs must not be superpositionable by any rotation.
    for a, b in (('L', 'J'), ('S', 'Z')):
        b_states = {tuple(sorted(norm_shift(s))) for s in E.SHAPES[b]}
        a_states = {tuple(sorted(norm_shift(s))) for s in E.SHAPES[a]}
        check(f'{a} cannot be rotated into {b}', not (a_states & b_states))


def test_rotation_is_consistent():
    """Rotating rot N one step clockwise must equal the published rot N+1."""
    print('rotation consistency (cw 90 == next table entry):')
    for kind in E.PIECES:
        for rot in range(4):
            cur = [list(c) for c in E.SHAPES[kind][rot]]
            # One rotation step, renormalised: the tables store each state with
            # its own origin, so the rotated result must be translated back.
            nxt = norm_shift([cw_about_origin(p) for p in cur])
            published = norm_shift(E.SHAPES[kind][(rot + 1) % 4])
            check(f'{kind} rot{rot} -> rot{(rot + 1) % 4} is one rotation step',
                  sorted(nxt) == sorted(published),
                  f'\n     computed {sorted(nxt)}\n     table    {sorted(published)}')

    # Four cw steps must return to the spawn state. Rotate about the shape's
    # true centre (its centroid), not its bounding-box corner: corner rotation
    # translates the shape and would never close the cycle.
    print('shape invariants:')
    for kind in E.PIECES:
        cells = list(E.SHAPES[kind][0])
        for _ in range(4):
            cells = [cw_about_origin(p) for p in cells]
        check(f'{kind}: four cw steps return to spawn',
              distance_signature(cells) == distance_signature(E.SHAPES[kind][0]),
              f'\n     got  {sorted(cells)}\n     want {sorted(E.SHAPES[kind][0])}')

        # Every rotation must keep the piece on the board's 10 columns when the
        # anchor is inside them: width is at most 4, height at most 4.
        for rot in range(4):
            xs = {x for x, _ in E.SHAPES[kind][rot]}
            ys = {y for _, y in E.SHAPES[kind][rot]}
            check(f'{kind} rot{rot} fits in a 4x4 box', len(xs) <= 4 and len(ys) <= 4,
                  f'{len(xs)}x{len(ys)}')


def test_spawn_positions():
    print('spawn:')
    # SRS spawn columns: I on 3-6, O on 4-5, the rest on 3-5.
    expected = {'I': (3, 6), 'O': (4, 5), 'J': (3, 5), 'L': (3, 5),
                'S': (3, 5), 'T': (3, 5), 'Z': (3, 5)}
    for kind in E.PIECES:
        p = E.Piece(kind)
        p.x, p.y = E.spawn_anchor(kind)
        cells = p.cells()
        ys = [c[1] for c in cells]
        span = E.spawn_span(kind)
        check(f'{kind} spawns on SRS columns {expected[kind]}', span == expected[kind], span)
        check(f'{kind} spawns in the hidden buffer, not the playfield',
              max(ys) < E.BUFFER_ROWS, f'max y {max(ys)}')
        check(f'{kind} spawn does not collide on empty board',
              not E.Board().collides(cells))
        # The spawn state sits fully above the playfield (negative rows) and
        # falls into view, which is what the Guideline does.
        check(f'{kind} spawn is above the playfield', max(ys) < 0, max(ys))


def test_bag():
    print('7-bag:')
    rng = __import__('random').Random(1)
    bag = E.Bag(rng)
    drawn = [bag.next() for _ in range(700)]
    for i in range(0, 700, 7):
        group = Counter(drawn[i:i + 7])
        if sorted(group.values()) != [1] * 7 or set(group) != set(E.PIECES):
            check(f'bag group {i // 7} contains each piece once', False, group)
            break
    else:
        check('every consecutive group of 7 is a permutation of all 7 pieces', True)


def test_line_clear_and_scoring():
    print('line clearing:')
    # Fill columns 0-5 of the bottom row, then lay a horizontal I across
    # columns 6-9. It rests exactly in row 21 and completes the line.
    g = E.Game(seed=0)
    for x in range(6):
        g.board.grid[21][x] = 'L'
    g.current = E.Piece('I', 0, 0, 0)
    g.current.x = 6
    g.current.y = E.spawn_anchor('I')[1]
    # ghost_y() is the anchor row; the occupied row is derived from the cells.
    landed = g.current.cells(y=g.ghost_y())
    check('horizontal I occupies columns 6-9',
          sorted(x for x, _ in g.current.cells()) == [6, 7, 8, 9],
          sorted(g.current.cells()))
    check('horizontal I comes to rest in the bottom row 21',
          {y for _, y in landed} == {21}, sorted(landed))
    cleared, rows = g.hard_drop()
    check('single line cleared', cleared == 1, f'cleared={cleared} rows={rows}')
    check('lines counter incremented', g.lines == 1)
    check('score awarded', g.score == 100, f'score={g.score}')
    check('level still 1', g.level == 1)
    check('all 4 piece cells were eroded', g.last_eroded == 4, g.last_eroded)

    # Tetris = 800: rows 18-21 filled across columns 0-8, column 9 empty, so a
    # vertical I dropped into column 9 completes all four rows at once.
    g2 = E.Game(seed=0)
    for y in range(18, 22):
        for x in range(9):
            g2.board.grid[y][x] = 'L'
    # Rotation 1 puts the four cells in a single column at board offset x+2 --
    # the SRS box convention keeps the vertical I in the middle columns of its
    # 4x4 box -- and the cells span board rows anchor_y .. anchor_y+3.
    g2.current = E.Piece('I', 7, 18, rotation=1)
    check('vertical I occupies column 9',
          {x for x, _ in g2.current.cells()} == {9}, sorted(g2.current.cells()))
    landed2 = g2.current.cells(y=g2.ghost_y())
    check('vertical I rests across rows 18-21',
          {y for _, y in landed2} == {18, 19, 20, 21}, sorted(landed2))
    cleared, _ = g2.hard_drop()
    check('tetris clears 4 lines', cleared == 4, cleared)
    check('tetris scores 800', g2.score == 800, g2.score)

    # Level scaling: level 2 makes a single worth 200 and a tetris 1600.
    g3 = E.Game(seed=0)
    g3.lines = 10
    g3.level = 2
    for x in range(6):
        g3.board.grid[21][x] = 'L'
    g3.current = E.Piece('I', 0, 0, 0)
    g3.current.x = 6
    g3.current.y = E.spawn_anchor('I')[1]
    g3.hard_drop()
    check('level 2 single scores 200', g3.score == 200, g3.score)

    # Back-to-back tetrises get a 1.5x bonus on the second one.
    g4 = E.Game(seed=0)
    g4.back_to_back = True
    for y in range(18, 22):
        for x in range(9):
            g4.board.grid[y][x] = 'L'
    g4.current = E.Piece('I', 7, 18, rotation=1)   # vertical I, cells at x+2
    g4.hard_drop()
    check('back-to-back tetris scores 1200', g4.score == 1200, g4.score)
    check('back_to_back stays set after another tetris', g4.back_to_back is True)


class BlockedAbove(E.Board):
    """A board whose non-existent negative rows are occupied.

    Used to test block-out, since ``Board.grid`` has no rows above the playfield
    and ``grid[-1]`` would index the bottom row instead.
    """

    def __init__(self, blocked_cols, rows=E.ROWS, cols=E.COLS):
        super().__init__(rows, cols)
        self.above = {(x, y) for x in blocked_cols for y in range(-4, 0)}

    def collides(self, cells):
        for c in cells:
            if c in self.above:
                return True
        return super().collides(cells)


def test_game_over_on_blockout():
    print('game over:')
    # Block out: fill the cells where the J/L spawn states sit, so every spawn
    # collides immediately and ``spawn_blocked`` is true.
    b = BlockedAbove(range(3, 6))
    g = E.Game(seed=0)
    g.board = b
    check('spawn_blocked detects a blocked spawn area', g.spawn_blocked())

    empty = BlockedAbove(range(0))
    g_norm = E.Game(seed=0)
    g_norm.board = empty
    check('spawn_blocked is false on a normal board', not g_norm.spawn_blocked())

    # Lock out: a piece that comes to rest entirely above the visible field ends
    # the game even though nothing collided. Block row 2 in the O's columns so it
    # settles at rows 0-1, both of which are invisible buffer rows.
    g2 = E.Game(seed=0)
    for x in (3, 4):
        g2.board.grid[2][x] = 'L'
    g2.current = E.Piece('O', 0, 0, 0)
    g2.current.x, g2.current.y = 3, 0
    landed = g2.ghost_y()
    check('O settles inside the buffer', landed < E.BUFFER_ROWS, landed)
    g2.hard_drop()
    check('game over on lock out above the visible field', g2.game_over)

    # Random legal play tops out quickly (it never clears lines), which is the
    # correct outcome and exercises the block-out path. What must hold is that
    # the game ends cleanly and the board is left consistent.
    import random as _random
    rng = _random.Random(7)
    g3 = E.Game(seed=7)
    placed = 0
    for _ in range(300):
        if g3.game_over:
            break
        acts = E.valid_actions(g3.board, g3.current)
        if not acts:
            check('valid_actions never empty while the game is running', False, g3.current.kind)
            break
        rot, x = rng.choice(acts)
        g3.current.rotation = rot
        g3.current.x = x
        g3.current.y = E.spawn_anchor(g3.current.kind)[1]
        g3.hard_drop()
        placed += 1
    check('random play eventually tops out and sets game_over', g3.game_over, placed)
    check('random play lasted a plausible number of pieces', 10 <= placed <= 60, placed)
    check('counters stay consistent after many placements',
          g3.lines >= 0 and g3.score >= 0 and g3.pieces_placed == placed,
          f'lines={g3.lines} score={g3.score} placed={g3.pieces_placed}/{placed}')
    # Every locked piece must be supported: each must touch the floor or another
    # block directly beneath one of its cells at the moment it locks. (Checking
    # the finished board instead would wrongly flag pieces like T, whose arms
    # necessarily have empty cells beneath them.)
    print('support invariant:')
    original_lock = E.Game.lock
    unsupported = []

    def checking_lock(self):
        snapshot = {id(r): list(r) for r in self.board.grid}
        cells = self.current.cells() if self.current else []
        kind = self.current.kind if self.current else None
        cleared, rows = original_lock(self)
        if self.game_over or not cells or self.current is None:
            return cleared, rows
        # Reconstruct: a piece is supported if any cell had the floor or a
        # filled cell directly below it in the pre-lock snapshot.
        full = set(rows)
        supported = False
        for x, y in cells:
            if y == self.rows - 1:
                supported = True
                break
            below = snapshot[id(self.board.grid[y + 1])][x] if y + 1 < self.rows else None
            # The cell below counts if it was already filled before the lock.
            if below is not None:
                supported = True
                break
        if not supported:
            unsupported.append((kind, sorted(cells)))
        return cleared, rows

    E.Game.lock = checking_lock
    try:
        rng = _random.Random(58)
        g4 = E.Game(seed=58)
        played = 0
        for _ in range(300):
            if g4.game_over:
                break
            acts = E.valid_actions(g4.board, g4.current)
            if not acts:
                break
            rot, x = rng.choice(acts)
            g4.current.rotation = rot
            g4.current.x = x
            g4.current.y = E.spawn_anchor(g4.current.kind)[1]
            g4.hard_drop()
            played += 1
    finally:
        E.Game.lock = original_lock
    check('every locked piece rested on the floor or on another block',
          not unsupported, unsupported[:3])
    check('support check actually exercised the game', played >= 15, played)


def test_hold():
    print('hold:')
    g = E.Game(seed=0)
    first = g.current.kind
    check('hold empty initially', g.hold_kind is None)
    check('hold succeeds', g.hold() is True)
    check('hold slot now has the first piece', g.hold_kind == first)
    check('hold_used flag set', g.hold_used is True)
    check('second hold in same turn rejected', g.hold() is False)
    # after locking, hold is available again
    g.hard_drop()
    check('hold available again after lock', g.hold_used is False)


def test_valid_actions():
    print('valid actions:')
    b = E.Board()
    p = E.Piece('T')
    p.x, p.y = E.spawn_anchor('T')
    acts = E.valid_actions(b, p)
    check('non-empty on empty board', len(acts) > 0, len(acts))
    check('all rotations present on empty board',
          {r for r, _ in acts} == {0, 1, 2, 3}, {r for r, _ in acts})
    check('no duplicate actions', len(acts) == len(set(acts)))
    # Every action must actually be playable.
    for rot, x in acts:
        g = E.Game(seed=0)
        g.current = E.Piece('T')
        g.current.x, g.current.y = E.spawn_anchor('T')
        g.current.rotation = 0
        g.current.x, g.current.y, g.current.rotation = x, g.current.y, rot
        ok = not g.board.collides(g.current.cells())
        if not ok:
            check(f'action {rot},{x} is colliding', False)
            break
    else:
        check('every returned action is collision-free at spawn row', True)

    # The O spans columns x and x+1, so x can be 0..8: 9 columns, and all four
    # rotations are geometrically identical (36 equivalent actions).
    p = E.Piece('O')
    p.x, p.y = E.spawn_anchor('O')
    acts_o = E.valid_actions(b, p)
    check('O has 9 usable columns x 4 identical rotations = 36', len(acts_o) == 36, len(acts_o))
    check('O never sticks out past the right wall',
          all(0 <= x and x + 1 < E.COLS for _, x in acts_o))

    # Valid actions are checked at the spawn position, which is above the
    # playfield. Blocking every spawn cell leaves no legal placement.
    b2 = BlockedAbove(range(E.COLS))
    acts2 = E.valid_actions(b2, p)
    check('a blocked spawn area leaves no valid actions', len(acts2) == 0, len(acts2))
    check('an empty board gives the expected count',
          len(E.valid_actions(E.Board(), p)) == len(acts_o))


def test_ghost_and_drop():
    print('ghost / drop:')
    g = E.Game(seed=0)
    g.current = E.Piece('O', 3, 0, 0)
    g.current.x, g.current.y = 0, 0
    gy = g.ghost_y()
    dropped = g.current.cells(y=gy)
    check('ghost lands on the floor', max(c[1] for c in dropped) == 21,
          max(c[1] for c in dropped))
    check('ghost does not move the real piece', g.current.y == 0)


def reference_valid_actions(board, piece):
    """Slow, obviously-correct placement enumeration by simulating the drop."""
    actions = []
    for rot in range(4):
        base = E._CELLS[piece.kind][rot]
        xs = [ox for ox, _ in base]
        # Anchors that keep the piece's *cells* on the board. They are not
        # necessarily 0..cols-1: under the SRS box convention a rotation's cells
        # can start at a non-zero offset, so the vertical I needs an anchor of -2
        # to reach column 0. Enumerating 0..cols-1 here would have made this
        # reference agree with the same mistake valid_actions was making.
        for x in range(-min(xs), board.cols - max(xs)):
            if board.collides([(x + ox, piece.y + oy) for ox, oy in base]):
                continue
            y = piece.y
            while not board.collides([(x + ox, y + 1 + oy) for ox, oy in base]):
                y += 1
            if not board.collides([(x + ox, y + oy) for ox, oy in base]):
                actions.append((rot, x))
    return actions


def test_every_rotation_reaches_every_column():
    """Every rotation of every piece must reach every column on an empty board.

    This is the invariant that was missing. `valid_actions` used to enumerate
    anchors as if each rotation's cells started at offset 0, so once the shape
    tables moved to the SRS box convention the vertical I could not reach columns
    0 or 1 -- the exact well a tetris is built in. Every policy in the project
    collapsed (the heuristic stalled at ~100 pieces, the search scored zero
    tetrises) and no differential test noticed, because both sides of the
    comparison shared the assumption. Reachability has to be asserted directly.
    """
    print('rotation reachability:')
    board = E.Board()
    for kind in E.PIECES:
        bad = []
        for rot in range(4):
            piece = E.Piece(kind, *E.spawn_anchor(kind))
            piece.rotation = rot
            cells = E._CELLS[kind][rot]
            cols = set()
            for r, x in E.valid_actions(board, piece):
                if r == rot:
                    cols.update(x + ox for ox, _ in cells)
            if cols != set(range(board.cols)):
                bad.append((rot, sorted(cols)))
        check(f'{kind}: every rotation can reach every column', not bad, bad)


def test_valid_actions_matches_simulation():
    """The fast height-based enumeration must equal the simulate-the-drop one.

    This is the single most important test in the file: valid_actions is the
    agent's entire legal-move set, and it is implemented in column space for
    speed rather than by simulating drops.
    """
    print('valid_actions vs simulated drop (differential):')
    import random as _random
    rng = _random.Random(1234)
    mismatches = []
    checked = 0
    for trial in range(400):
        g = E.Game(seed=trial)
        # Build a random stack so boards vary in shape, holes and overhangs.
        for _ in range(rng.randrange(0, 90)):
            if g.game_over:
                break
            acts = E.valid_actions(g.board, g.current)
            if not acts:
                break
            rot, x = rng.choice(acts)
            g.current.rotation = rot
            g.current.x = x
            g.current.y = E.spawn_anchor(g.current.kind)[1]
            g.hard_drop()
        if g.game_over or g.current is None:
            continue
        fast = sorted(E.valid_actions(g.board, g.current))
        slow = sorted(reference_valid_actions(g.board, g.current))
        checked += 1
        if fast != slow:
            mismatches.append((g.current.kind, fast, slow))
    check(f'fast and simulated enumeration agree on {checked} boards',
          not mismatches,
          f'{len(mismatches)} mismatches, first: {mismatches[0] if mismatches else None}')

    # And on a hand-built overhang: a lone block creates a slot the piece can
    # drop into but cannot be slid sideways into.
    b = E.Board()
    for y in range(18, 22):
        b.grid[y][0] = 'L'
    b.grid[18][1] = 'L'      # overhang over column 1
    p = E.Piece('L')
    p.x, p.y = E.spawn_anchor('L')
    check('overhang case agrees with simulation',
          sorted(E.valid_actions(b, p)) == sorted(reference_valid_actions(b, p)),
          f'{sorted(E.valid_actions(b, p))} vs {sorted(reference_valid_actions(b, p))}')
def test_simulate_matches_execution():
    """``simulate_placement`` must produce exactly the board real play produces.

    This underpins the entire agent: every placement is chosen by scoring the
    board that ``simulate_placement`` predicts, so any disagreement between the
    prediction and reality silently teaches the network about a game that is not
    being played.
    """
    print('simulate_placement vs real play:')
    import random as _random
    rng = _random.Random(3)
    mismatches = []
    checked = 0
    g = E.Game(seed=3)
    while not g.game_over and checked < 250:
        acts = E.valid_actions(g.board, g.current)
        if not acts:
            break
        rot, x = rng.choice(acts)
        kind = g.current.kind
        spawn_y = E.spawn_anchor(kind)[1]

        # Both the simulation and the real play must start from the spawn
        # anchor: that is how an action is realised.
        g.current.y = spawn_y
        sim_grid, sim_cells, sim_cleared_rows = g.simulate_placement(rot, x)
        sim_cleared = len(sim_cleared_rows)

        # Replay the same placement on a copy and compare.
        clone = E.Game(seed=0)
        clone.board.grid = [row[:] for row in g.board.grid]
        clone.current = E.Piece(kind)
        clone.lines = g.lines
        clone.current.rotation = rot
        clone.current.x = x
        clone.current.y = spawn_y
        before = clone.lines
        clone.hard_drop()
        real_cleared = clone.lines - before

        if sim_grid != clone.board.grid or sim_cleared != real_cleared:
            mismatches.append((kind, rot, x, sorted(sim_cells), sim_cleared, real_cleared))
        checked += 1

        g.current.rotation = rot
        g.current.x = x
        g.current.y = E.spawn_anchor(kind)[1]
        g.hard_drop()

    check(f'simulation matched real play on {checked} placements',
          checked > 0 and not mismatches,
          f'{len(mismatches)} mismatches, first: {mismatches[0] if mismatches else None}')


def test_srs_wall_kick():
    print('SRS wall kick:')
    # A vertical I against the left wall should kick when rotated to horizontal.
    g = E.Game(seed=0)
    g.current = E.Piece('I', 0, 0, 1)
    g.current.x, g.current.y = 0, 0
    check('vertical I at x=0 is placeable', not g.board.collides(g.current.cells()),
          g.current.cells())
    ok = g.rotate(1)
    check('rotation into the wall succeeds via kick', ok)
    if ok:
        check('resulting cells are in bounds',
              all(0 <= x < g.cols for x, _ in g.current.cells()),
              g.current.cells())

    # A T against the left wall rotated should also resolve.
    g2 = E.Game(seed=0)
    g2.current = E.Piece('T', 0, 0, 0)
    g2.current.x, g2.current.y = 0, 0
    check('T rotation at the left wall resolves', g2.rotate(3) or g2.rotate(1))


def test_tspin_kick_available():
    print('T-spin kick (TST setup):')
    # Classic T-spin triple slot: a 3-deep well with an overhang.
    g = E.Game(seed=0)
    for y in range(18, 22):
        for x in range(10):
            g.board.grid[y][x] = 'L'
    for y in range(18, 22):
        g.board.grid[y][3] = None
        g.board.grid[y][4] = None
        g.board.grid[y][5] = None
    g.board.grid[18][5] = 'L'  # overhang
    g.board.grid[19][5] = 'L'
    g.board.grid[20][5] = 'L'
    g.board.grid[21][5] = 'L'
    # A vertical T dropped into the well then rotated should be able to kick in
    g.current = E.Piece('T', 0, 0, 1)
    g.current.x, g.current.y = 4, 0
    landed = g.ghost_y()
    g.current.y = landed
    ok = g.rotate(1) or g.rotate(3) or g.rotate(2)
    check('a legal rotation exists in the T-slot', ok)


def test_metrics_helpers():
    print('board metrics:')
    b = E.Board()
    # column heights via helper used by features
    for y in range(18, 22):
        b.grid[y][0] = 'L'
    b.grid[19][1] = 'L'  # a floating block -> a hole at row 20-21 under it
    hs = [0] * b.cols
    for x in range(b.cols):
        for y in range(b.rows):
            if b.grid[y][x] is not None:
                hs[x] = b.rows - y
                break
    check('column 0 height is 4', hs[0] == 4, hs[0])
    check('column 1 height is 3', hs[1] == 3, hs[1])


def test_box_convention():
    """Every rotation state must live in the piece's own SRS box.

    This is the invariant the SRS/SRS+ kick tables are written against: the
    offsets describe where the *box* moves. An earlier version of the shape table
    rotated each state about the origin instead, so states sat at different offsets
    inside their box, kicks pushed the piece to the wrong place, and the T spawned
    nub-up (mirrored against the real game) -- which made T-spins impossible.
    """
    print('SRS box convention:')
    boxes = {'I': 4, 'O': 2}
    for kind in E.PIECES:
        size = boxes.get(kind, 3)
        inside = True
        for rot in range(4):
            xs = {x for x, _ in E.SHAPES[kind][rot]}
            ys = {y for _, y in E.SHAPES[kind][rot]}
            if min(xs) < 0 or max(xs) >= size or min(ys) < 0 or max(ys) >= size:
                inside = False
        check(f'{kind}: all four states fit the {size}x{size} box', inside)
        # And the box is actually used, rather than the states having been
        # normalised into a corner: at least one state must touch each edge.
        span_x = set()
        span_y = set()
        for rot in range(4):
            for x, y in E.SHAPES[kind][rot]:
                span_x.add(x)
                span_y.add(y)
        check(f'{kind}: the box spans its full width', span_x == set(range(size)),
              sorted(span_x))
        check(f'{kind}: the box spans its full height', span_y == set(range(size)),
              sorted(span_y))


def test_srs_plus_wall_rotations():
    """SRS+ never refuses a rotation the board has room for.

    SRS guarantees this: there is always somewhere for the piece to go, and the
    kick table is supposed to find it. It is the check that would have caught both
    rotation bugs at once -- the counter-clockwise turns that were being sent to
    the 180-degree tables, and the mirrored shape convention those offsets assumed.

    "Room" is the operative word, and it is why the piece rests two rows above the
    floor here rather than on it. A piece sits in its box, and SRS lets that box
    hang below the surface -- a flat I on the *bottom* row has its box two rows
    below the floor, and standing it up needs a two-row lift that the published I
    table does not have (its largest upward kick is one row). That refusal is the
    real table's behaviour, not something to paper over, and it is pinned
    separately below.
    """
    print('SRS+ rotations against both walls:')
    refused = []
    attempts = 0
    for side in ('left', 'right'):
        for clearance in (None, 2):        # None = in the air, 2 = two rows of room
            for kind in E.PIECES:
                for rot in range(4):
                    for direction in (1, -1, 2):
                        attempts += 1
                        game = E.Game(seed=0)
                        piece = E.Piece(kind, *E.spawn_anchor(kind))
                        piece.rotation = rot
                        game.current = piece
                        if clearance is not None:
                            piece.y = game.ghost_y(piece) - clearance
                        step = -1 if side == 'left' else 1
                        for _ in range(12):
                            if not game.move(step):
                                break
                        if not game.rotate(direction):
                            refused.append((kind, side, clearance, rot, direction))
    check(f'all {attempts} rotations succeed (every piece, state, direction, '
          f'both walls, in the air and with room below)', not refused,
          f'{len(refused)} refused: {refused[:6]}')

    # The floor. SRS+ lifts a piece to complete a rotation, so even a flat I on the
    # bottom row stands up -- through the fifth published test, (1, -2): right one,
    # up two. This case used to be pinned here as *impossible*, and that was the
    # tell: the published (1, -2) had been mirrored into (1, +2) by negating dy on
    # a table that was already y-down, so the only offset that fits pushed the bar
    # further below the floor.
    bottom = E.Game(seed=0)
    flat = E.Piece('I', *E.spawn_anchor('I'))
    bottom.current = flat
    flat.y = bottom.ghost_y(flat)
    check('a flat I on the bottom row stands up via the published (1, -2)',
          bottom.rotate(1)
          and bottom.last_kick_index == 4
          and bottom.last_kick_offset == E.IKICKS[(0, 1)][4] == (1, -2),
          f'index {bottom.last_kick_index} offset {bottom.last_kick_offset}')
    raised = E.Game(seed=0)
    lifted = E.Piece('I', *E.spawn_anchor('I'))
    raised.current = lifted
    lifted.y = raised.ghost_y(lifted) - 1
    check('one row up, the same published kick is the one that fits',
          raised.rotate(1)
          and raised.last_kick_index == 4
          and raised.last_kick_offset == E.IKICKS[(0, 1)][4] == (1, -2),
          f'index {raised.last_kick_index} offset {raised.last_kick_offset}')
    roomy = E.Game(seed=0)
    standing = E.Piece('I', *E.spawn_anchor('I'))
    roomy.current = standing
    standing.y = roomy.ghost_y(standing) - 2
    check('two rows of clearance lets it stand up with no kick at all',
          roomy.rotate(1) and roomy.last_kick_index == 0,
          f'index {roomy.last_kick_index}')


def test_z_and_s_spin_into_a_notch():
    """A Z and an S tucked into a well -- the case that was reported broken.

    A Z-spin is a rotation that needs a *vertical* kick: the piece has to be lifted
    as it turns, or it rotates into the terrain beside the well. With dy negated on
    a table that was already y-down, the lift became a push downwards and the
    rotation was refused -- "I can't spin the Z in". Every offset asserted here is
    the first fitting entry of the published SRS+ row, which is what SRS+ does.
    """
    print('Z/S spin into a notch:')

    def well(kind):
        """A 2-wide well with an overhang at row 16: the shape a Z or S spins into."""
        game = E.Game(seed=0)
        for y in range(16, 22):
            for x in range(10):
                game.board.grid[y][x] = 'X'
        for x in (4, 5):
            for y in range(17, 22):
                game.board.grid[y][x] = None
        game.board.grid[16][4] = None
        game.board.grid[16][5] = None
        game.current = E.Piece(kind, *E.spawn_anchor(kind))
        return game

    def spin(kind, direction, column):
        game = well(kind)
        piece = game.current
        piece.x = column
        piece.y = game.ghost_y(piece)          # seated on the terrain above the well
        before = (piece.rotation, piece.x, piece.y)
        return game, piece, before, game.rotate(direction)

    game, piece, before, ok = spin('Z', 1, 2)
    check('a Z spins into the well', ok, before)
    check('and it uses the published (-1, -1) kick',
          game.last_kick_index == 2
          and game.last_kick_offset == E.KICKS[(0, 1)][2] == (-1, -1),
          f'index {game.last_kick_index} offset {game.last_kick_offset}')
    check('landing where SRS+ puts it',
          (piece.rotation, piece.x, piece.y) == (1, 1, 13),
          (piece.rotation, piece.x, piece.y))

    game, piece, before, ok = spin('Z', -1, 2)
    check('a Z also spins in the other way', ok, before)
    check('using the published (1, -1) kick',
          game.last_kick_index == 2
          and game.last_kick_offset == E.KICKS[(0, 3)][2] == (1, -1),
          f'index {game.last_kick_index} offset {game.last_kick_offset}')
    check('landing where SRS+ puts it, the other way round',
          (piece.rotation, piece.x, piece.y) == (3, 3, 13),
          (piece.rotation, piece.x, piece.y))

    game, piece, before, ok = spin('S', 1, 4)
    check('an S spins into the well', ok, before)
    check('using the published (-1, 0) kick',
          game.last_kick_index == 1
          and game.last_kick_offset == E.KICKS[(0, 1)][1] == (-1, 0),
          f'index {game.last_kick_index} offset {game.last_kick_offset}')

    # The sign convention itself, pinned. A vertically mirrored table is still
    # internally consistent -- it passes the wall sweep, the spawn tests and every
    # rotation-count check -- so this is the assertion that would catch it coming
    # back: on this y-down board a positive dy moves the piece *down* the screen.
    check('the kick rows are y-down: (0, +2) is a two-row drop',
          E.KICKS[(0, 1)][3] == (0, 2) and E.KICKS[(1, 0)][3] == (0, -2),
          (E.KICKS[(0, 1)][3], E.KICKS[(1, 0)][3]))
    check('and the S/Z rows match the published SRS+ table entry for entry',
          E.KICKS[(0, 1)] == ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2))
          and E.KICKS[(2, 3)] == ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
          (E.KICKS[(0, 1)], E.KICKS[(2, 3)]))


def test_tspin_detection():
    """T-spin classification, and a real T-spin double built with a kick."""
    print('T-spin detection:')

    def classify(kind='T', rot=0, x=3, y=18, filled=(), moved=False, kick=0):
        game = E.Game(seed=0)
        game.current = E.Piece(kind, x, y, rot)
        for fx, fy in filled:
            game.board.grid[fy][fx] = 'X'
        if moved:
            game.move(1)
            game.move(-1)
        else:
            game.rotation_was_last_move = True
        game.last_kick_index = kick
        game.lock()
        return game.spin_kind

    # A T in state 0 at (3, 18): its cells are the nub at (4, 18) and the row
    # (3, 19), (4, 19), (5, 19), so its 3x3 box -- anchor at the top-left corner --
    # has corners (3, 18), (5, 18), (3, 20) and (5, 20). The nub points up, so the
    # two front corners are the top pair.
    both_front = [(3, 18), (5, 18), (3, 20)]
    one_front = [(3, 18), (3, 20), (5, 20)]
    check('three corners including both front corners is a full T-spin',
          classify(filled=both_front) == 'full')
    check('three corners with only one front corner is a mini T-spin',
          classify(filled=one_front) == 'mini')
    check('a far kick promotes a mini to a full T-spin',
          classify(filled=one_front, kick=4) == 'full')
    check('two corners is not a spin at all',
          classify(filled=[(3, 18), (5, 20)]) is None)
    check('a move before the lock clears the rotation flag',
          classify(filled=both_front, moved=True) is None)
    check('only the T is classified', classify(kind='L', filled=both_front) is None)

    # The published table, test for test: a T resting on the floor turns clockwise
    # through its *third* test, (-1, -1) -- left one, up one. The first two (turn
    # in place, then left one) both leave the bar a row below the floor. An earlier
    # version of this check expected index 3 and the offset (0, -2), which is the
    # mirrored table talking: the published index 3 is (0, +2), a two-row *drop*
    # used to spin a T into a slot, not to lift it off the floor.
    flooring = E.Game(seed=0)
    standing = E.Piece('T', *E.spawn_anchor('T'))
    flooring.current = standing
    standing.y = flooring.ghost_y(standing)
    check('a T on the floor turns clockwise', flooring.rotate(1))
    check('and the offset used is the published third test, (-1, -1)',
          flooring.last_kick_index == 2
          and flooring.last_kick_offset == E.KICKS[(0, 1)][2]
          and flooring.last_kick_offset == (-1, -1),
          f'index {flooring.last_kick_index} offset {flooring.last_kick_offset}')

    # End to end, on a hand-built seat. Rows 18 and up are solid except for a
    # 2-wide mouth at row 18, a 3-wide notch at row 19, a 1-wide gap at row 20 and
    # the cell under it, so the T has four distinct landing places and the clear
    # count depends on how it is turned:
    #
    #   row 18: ###..#####     (3,18) (4,18)
    #   row 19: ###...####     (3,19) (4,19) (5,19)
    #   row 20: ####.#####     (4,20)
    #   row 21: ##########
    #
    # Dropped standing in state 3 (nub left) the T falls into the notch and rests
    # with its bar down the (4, 18..20) column. It is already a spin position --
    # three of its box corners are filled -- and which way it turns decides the
    # clear: a 180 puts the nub in the right-hand mouth and clears two rows, a
    # counter-clockwise turn lays it flat across row 19 and clears three.
    def seat():
        board = E.Game(seed=0)
        for y in range(18, 22):
            for x in range(board.cols):
                board.board.grid[y][x] = 'L'
        for cell in ((3, 18), (3, 19), (4, 18), (4, 19), (4, 20), (5, 19)):
            board.board.grid[cell[1]][cell[0]] = None
        return board

    def drop_into_seat(direction):
        board = seat()
        board.current = E.Piece('T', 3, E.spawn_anchor('T')[1], 3)
        while board.soft_drop():
            pass
        check(f'state 3 T rests in the notch (dir {direction})',
              sorted(board.current.cells())
              == [(3, 19), (4, 18), (4, 19), (4, 20)],
              sorted(board.current.cells()))
        spun = board.rotate(direction)
        check(f'and turns with no kick needed (dir {direction})',
              spun and board.last_kick_offset == (0, 0), board.last_kick_offset)
        while board.soft_drop():
            pass
        cleared, _rows = board.lock()
        return board, cleared

    board, cleared = drop_into_seat(2)             # 3 -> 1, the 180 turn
    check('a 180 turn into the seat is a full T-spin',
          board.spin_kind == 'full', board.spin_kind)
    check('and clears two rows', cleared == 2, cleared)

    board, cleared = drop_into_seat(-1)            # 3 -> 2, counter-clockwise
    check('a counter-clockwise turn into the seat is a full T-spin',
          board.spin_kind == 'full', board.spin_kind)
    check('and clears three rows -- a T-spin triple', cleared == 3, cleared)
    check('the triple left the mouth of the seat standing',
          [board.board.grid[21][x] for x in range(6)]
          == ['L', 'L', 'L', None, None, 'L'],
          [board.board.grid[21][x] for x in range(6)])

    # The same board filled through instead of spun: a plain drop is not a spin.
    plain = E.Game(seed=0)
    for x in range(plain.cols):
        plain.board.grid[19][x] = 'L'
        plain.board.grid[20][x] = 'L'
    for x in (3, 4, 5):
        plain.board.grid[19][x] = None
    plain.board.grid[20][4] = None
    plain.current = E.Piece('T', 3, 0, 0)
    plain.hard_drop()
    check('a piece that never rotated is not a spin', plain.spin_kind is None,
          plain.spin_kind)


def main():
    for fn in (test_shapes_are_tetrominoes,
               test_rotation_is_consistent,
               test_box_convention,
               test_spawn_positions,
               test_bag,
               test_line_clear_and_scoring,
               test_game_over_on_blockout,
               test_hold,
               test_valid_actions,
               test_every_rotation_reaches_every_column,
               test_valid_actions_matches_simulation,
               test_simulate_matches_execution,
               test_ghost_and_drop,
               test_srs_wall_kick,
               test_srs_plus_wall_rotations,
               test_tspin_kick_available,
               test_z_and_s_spin_into_a_notch,
               test_tspin_detection,
               test_metrics_helpers):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S): {FAILS}')
        return 1
    print('all engine tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
