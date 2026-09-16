"""Beam-search placement selection.

Why this file exists
--------------------
The one-ply evaluator in ``heuristic.py`` cannot build tetrises, and the reason is
structural rather than a matter of weights. Clearing one line and keeping a well
open score *identically right now*, so the greedy choice is a coin flip, and every
measured attempt to force it through the static score either did nothing or made
the evaluator hoard a well it could never use and top out (README, "the tetris
problem").

What real bots do instead is *search*: they look several pieces ahead over the
actual next queue and the hold slot and pick the placement whose continuation is
best. This module is a small CPU-sized version of that idea, built to answer one
question: **how do I get more tetrises on average?**

How it works
------------
* The board is a tuple of column bitmasks (bit ``i`` of column ``x`` is the cell
  at row ``rows - 1 - i``), so placing a piece, clearing lines and every board
  statistic are integer operations. That is what makes a search affordable in pure
  Python.
* Moves are ``(use_hold, rotation, x)`` with a hard drop -- the engine's action
  abstraction. Legality and landing row come from column heights, the same
  reduction ``engine.valid_actions`` uses.
* A beam search runs ``depth`` pieces deep and ``beam`` boards wide, expanding
  every legal placement (including the hold swap).
* Ranking a node means path rewards (line clears, exactly as Cold Clear does it)
  plus a static evaluation of the board it reached.

The evaluation is a deliberate port of Cold Clear's ``standard.rs``
(MinusKelvin/cold-clear, MIT), rescaled only where this project has no equivalent:
cavity cells and their squares, bumpiness with the well column excluded, maximum
height, the ``top_half``/``top_quarter`` height walls, a well-depth bonus with
their per-column bias array, and their exact clear values (-143 / -100 / -58 /
+390). Their ratios matter more than their scale: a hole is worth about four rows
of height in their weighting, and the version of this file that used a hole at
-170 against a height of -2 was effectively telling the search that a hole costs
thirty-four rows. It played accordingly -- parking at height 13 and farming
singles.

Nothing here is copied verbatim; the port is of the *structure* of their
evaluator. Time- and attack-dependent terms (``jeopardy``, ``move_time``, combo
garbage, T-spin slots, perfect clear) are omitted because this engine has no
garbage, no opponent and no T-spin model.

Everything is deterministic given the board: no RNG, no wall-clock cutoffs.
``tests/test_search.py`` pins the bitboard against ``Game.simulate_placement`` so
the fast model and the engine cannot drift apart.
"""
from __future__ import annotations

from .engine import PIECES, SHAPES, spawn_anchor, valid_actions


# --- piece geometry, board convention (y grows downward) --------------------

def _rotations(kind):
    """Per rotation: cells, covered columns, bottom profile, top profile, width."""
    out = []
    for rot in range(4):
        # ``SHAPES`` is already in board convention (y grows downward, offsets
        # from the box's top-left corner). This used to negate y here as well,
        # which quietly mirrored every piece relative to the engine: the search
        # then scored placements the engine could never reach.
        cells = tuple(SHAPES[kind][rot])
        dxs = tuple(sorted({ox for ox, _ in cells}))
        bottoms = tuple(max(oy for ox, oy in cells if ox == dx) for dx in dxs)
        tops = tuple(min(oy for ox, oy in cells if ox == dx) for dx in dxs)
        out.append((cells, dxs, bottoms, tops, dxs[-1] + 1))
    return tuple(out)


_ROT = {kind: _rotations(kind) for kind in PIECES}
_SPAWN = {kind: spawn_anchor(kind) for kind in PIECES}


# --- compact board ----------------------------------------------------------

def board_masks(game):
    """``game``'s board as one bitmask per column. Bit i is row ``rows - 1 - i``."""
    rows = game.rows
    grid = game.board.grid
    return tuple(
        sum(1 << (rows - 1 - y) for y in range(rows) if grid[y][x] is not None)
        for x in range(game.cols)
    )


def heights_of(masks):
    """Column heights: rows up to and including the top filled cell."""
    return [m.bit_length() for m in masks]


def broken_cells(masks, heights):
    """Empty cells with a filled cell above them (holes), summed over columns."""
    return sum(h - m.bit_count() for m, h in zip(masks, heights))


def _remove_rows(mask, full):
    """Delete the rows whose bits are set in ``full``, dropping everything above."""
    removed = 0
    f = full
    while f:
        low = f & -f
        i = low.bit_length() - 1 - removed
        mask = (mask & ((1 << i) - 1)) | ((mask >> (i + 1)) << i)
        removed += 1
        f ^= low
    return mask


# --- placement --------------------------------------------------------------

def landing_y(heights, kind, rot, x, rows):
    """Row the piece comes to rest on, or None when the placement is illegal.

    A placement is legal when the piece fits at its spawn row -- which is what
    forbids sliding under an overhang -- and drops from there. Working in heights
    rather than cells is exact, because a piece only ever descends onto the top of
    each column it covers.
    """
    _cells, dxs, bottoms, _tops, _width = _ROT[kind][rot]
    if x + dxs[0] < 0 or x + dxs[-1] >= len(heights):
        return None
    y = rows - 1 - heights[x + dxs[0]] - bottoms[0]
    for i in range(1, len(dxs)):
        other = rows - 1 - heights[x + dxs[i]] - bottoms[i]
        if other < y:
            y = other
    if y < _SPAWN[kind][1]:
        return None
    return y


def placements(masks, heights, kind, rows, cols):
    """Every legal ``(rotation, x, landing_y)`` for ``kind`` on this board.

    ``x`` is an anchor column and may be negative: a rotation's cells do not have
    to start at offset 0 in the SRS box convention, so the vertical I (offset 2 of
    its 4x4 box) must be allowed an anchor of -2 to sit in column 0. Enumerating
    box-aligned anchors instead silently forbade the well a tetris is built in.
    """
    out = []
    for rot in range(4):
        dxs = _ROT[kind][rot][1]
        for x in range(-dxs[0], cols - dxs[-1]):
            y = landing_y(heights, kind, rot, x, rows)
            if y is not None:
                out.append((rot, x, y))
    return out


def apply_placement(masks, kind, rot, x, y, rows):
    """Board after dropping ``kind`` at ``(rot, x)`` from row ``y``.

    Returns ``(masks, cleared)``. Cells above the board are discarded, matching
    ``Board.lock``.
    """
    new = list(masks)
    for ox, oy in _ROT[kind][rot][0]:
        cy = y + oy
        if cy >= 0:
            new[x + ox] |= 1 << (rows - 1 - cy)
    full = new[0]
    for m in new[1:]:
        full &= m
    cleared = full.bit_count()
    if cleared:
        new = [_remove_rows(m, full) for m in new]
    return tuple(new), cleared


def spawn_blocked(masks, kind, rows, cols):
    """True when ``kind`` cannot spawn -- the engine's definition of a terminal board."""
    sx, sy = _SPAWN[kind]
    for ox, oy in _ROT[kind][0][0]:
        cx, cy = sx + ox, sy + oy
        if cx < 0 or cx >= cols or cy >= rows:
            return True
        if cy >= 0 and (masks[cx] >> (rows - 1 - cy)) & 1:
            return True
    return False


def topped_out(masks, rows, cols):
    """True when no piece can spawn, so the game is over on this board."""
    return all(spawn_blocked(masks, kind, rows, cols) for kind in PIECES)


def locks_out(kind, rot, y, buffer_rows):
    """True when the piece would come to rest entirely inside the hidden buffer.

    This is the engine's other death: ``Game.lock`` ends the game when every row a
    piece touches is above the visible playfield. It is reachable well before the
    board is full, so a search that only knows about block-out will walk into it.
    """
    return max(y + oy for _ox, oy in _ROT[kind][rot][0]) < buffer_rows


# --- board statistics -------------------------------------------------------

def well_depth(masks, heights, cols, rows, cap):
    """``(depth, column)`` of the well, by Cold Clear's rule.

    The well is the **lowest column, rightmost on a tie**, and its depth is the
    number of rows above it that are occupied in all nine other columns -- counted
    contiguously upward from the well's top and capped at ``cap``.

    Kept alongside :func:`ready_depth` because the two disagree in an instructive
    way. This one is robust and cheap and never looks at holes below the well; the
    other is stricter and measured better in this engine. Ported weights that use
    this rule are available (see ``WEIGHT_PRESETS`` in the module docstring) but
    are off by default.
    """
    well = 0
    for x in range(1, cols):
        if heights[x] <= heights[well]:
            well = x
    depth = 0
    for bit in range(heights[well], rows):
        for x in range(cols):
            if x != well and not (masks[x] >> bit) & 1:
                return depth, well
        depth += 1
        if depth >= cap:
            break
    return depth, well


def ready_depth(masks, heights, cols, rows, cap, pref=None):
    """``(depth, column)`` of the best tetris well, gapless, capped at ``cap``.

    A column's depth is how far it sits below the *gapless* height of the lowest
    other column: an I dropped there clears exactly that many rows. ``pref`` breaks
    ties towards the columns a player would pick, so the well does not wander.

    The strictness is load-bearing and was measured. A more literal version --
    count the rows above the well that are filled in every other column -- looks
    more honest and plays much worse: 379 pieces and 8 deaths in 8 games against
    930 pieces and 5 deaths on identical seeds. "Gapless" quietly refuses to see a
    tetris on a board with holes in it, which is exactly the board where chasing a
    tetris is the wrong plan; the literal version commits to a well on a broken
    board and dies there.
    """
    solid = solid_heights(masks)
    lo, hi, lo_i = _two_smallest(solid)
    if lo is None:
        return 0, -1
    best_val = None
    best = (0, -1)
    for w in range(cols):
        other = hi if w == lo_i else lo
        if other is None:
            continue
        depth = other - heights[w]
        if depth > cap:
            depth = cap
        elif depth < 0:
            depth = 0
        val = depth * 4 + (pref[w] if (pref and depth) else 0)
        if best_val is None or val > best_val:
            best_val = val
            best = (depth, w)
    return best


def solid_heights(masks):
    """Height of the gapless run from the floor -- where the column first has a hole."""
    return [((~m) & (m + 1)).bit_length() - 1 for m in masks]


def _two_smallest(values):
    """(smallest, second smallest, index of the smallest) in one pass."""
    lo = hi = None
    lo_i = -1
    for i, v in enumerate(values):
        if lo is None or v < lo:
            hi = lo
            lo, lo_i = v, i
        elif hi is None or v < hi:
            hi = v
    return lo, hi, lo_i


def dip_cost(heights, cols, cap=4):
    """Summed ``depth^2`` of single-column dips, capped per column.

    Off by default -- Cold Clear has no such term and measurement agreed that
    adding one made the policy worse -- but kept because it is the direct
    expression of "one well is an investment, two are a coffin" and a cheaper
    guard than the well rule above if the well rule is ever changed.
    """
    total = 0
    for x in range(cols):
        left = heights[x - 1] if x else heights[x]
        right = heights[x + 1] if x < cols - 1 else heights[x]
        gap = (left if left < right else right) - heights[x]
        if gap > 0:
            if gap > cap:
                gap = cap
            total += gap * gap
    return total


# --- evaluation and clear values --------------------------------------------

# The weights that ship. Two families live here.
#
# The tetris terms (`ready`, the negative small-clear values, the height pressure)
# were tuned in this engine and measured: 21% of clear events are tetrises against
# 3.9% for the one-ply heuristic, with roughly four times as many tetrises per
# piece.
#
# The Cold Clear terms (`holes_sq`, `bumpiness_sq`, `top_half`, `top_quarter`,
# `well_depth`, `well_column`, `skip_well_bumpiness`) are ports of their published
# defaults in their own units, and are OFF by default. Porting their whole table --
# cavity -173 against height -39, a hole worth four rows of height instead of
# thirty-four -- was tried and measured *worse* here: 6% tetrises and 7 deaths in 8
# games. That is worth recording rather than hiding. Their weights are tuned for a
# deep MCTS with garbage, combos and T-spins; a shallow beam search that keeps the
# stack pinned low never builds the four-row structure a tetris needs. The terms
# stay available so each can be tested on its own instead of as a block.
TETRIS_SEARCH_WEIGHTS = {
    # Board quality.
    'holes': -170.0,
    'bumpiness': -2.2,
    'aggregate_height': -0.50,
    'max_height': -2.0,

    # Tetris structure.
    'ready': 11.0,            # per row of well depth, 0..4
    'ready_full': 12.0,       # extra once four rows are genuinely available
    'max_well_depth': 4,
    'well_pref': (2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0),
    # Charge for well depth past the four rows a tetris can use. This is the single
    # most valuable term in the set after `ready` itself, and the reason is
    # survival rather than scoring: rows under a deep well can never be cleared
    # until the well is filled, so a well that outruns the stack is a trap that
    # only closes. Measured over 8 seeds: -40 gave 26.0 tetrises per game and 965
    # pieces against 19.8 and 810 with the charge off, and heavier (-70) was worse
    # again.
    'well_excess': -40.0,
    'dip': 0.0,               # charge for extra pits beyond the well

    # Pieces. Cold Clear has no hold term -- their search is deep enough that
    # banking an I pays off on its own. Ours is 2-3 plies, so the value of an I in
    # hand has to be visible statically or the search simply spends it.
    'i_hold': 20.0,
    'i_hold_base': 6.0,
    'wasted_i': -20.0,

    # Clear values: a single is a cost, the tetris is the payday.
    'clear1': -80.0,
    'clear2': -100.0,
    'clear3': -80.0,
    'clear4': 480.0,

    # Quadratic height pressure above `height_soft`. This replaced a "danger mode"
    # that flipped small-clear values positive above a threshold; that made digging
    # *profitable*, so the policy grew to height 13, farmed singles at +140 each and
    # never kept a well at all (readiness 0 on 84% of moves, 3.9% tetrises -- the
    # exact behaviour it was meant to cure).
    'height_soft': 8,
    'height_k': 12.0,

    # Cold Clear ports, off by default. See the note above.
    'holes_sq': 0.0,
    'bumpiness_sq': 0.0,
    'top_half': 0.0,
    'top_half_at': 10,
    'top_quarter': 0.0,
    'top_quarter_at': 15,
    'well_depth': 0.0,
    'well_column': None,
    'skip_well_bumpiness': False,

    # Versus terms, off by default so the solo measurements above keep their
    # meaning; VERSUS_SEARCH_WEIGHTS turns them on.
    #
    # `covered` is Cold Clear's dig-priority term: per column, how many blocks sit
    # on top of each hole, capped per hole and squared. A plain hole count cannot
    # tell a reachable hole from one buried under six blocks, and on a garbage
    # board that difference is the whole game.
    'covered': 0.0,
    'covered_sq': 0.0,
    'covered_cap': 6,
    # Incoming garbage makes height lethal rather than merely bad. This is scored
    # per board, and `search_move` additionally refuses moves that cannot survive
    # the queue at all.
    'incoming_danger': 0.0,
    # Gaps only a vertical I can fill. Off here so the solo measurements keep
    # their meaning; VERSUS_SEARCH_WEIGHTS turns it on.
    'i_dependency': 0.0,
    'i_dependency_sq': 0.0,
    'i_dependency_rows': 0.0,
    'dep_suppress': 0.0,
}

# Cold Clear's published defaults as a preset, for anyone re-testing the claim
# above: this is the configuration that measured worse.
COLD_CLEAR_WEIGHTS = dict(
    TETRIS_SEARCH_WEIGHTS,
    holes=-173.0, bumpiness=-24.0, max_height=-39.0,
    holes_sq=-3.0, bumpiness_sq=-7.0,
    top_half=-150.0, top_quarter=-511.0,
    skip_well_bumpiness=True,
    well_depth=57.0,
    well_column=(20.0, 23.0, 20.0, 50.0, 59.0, 21.0, 59.0, 10.0, -10.0, 24.0),
    ready=0.0, ready_full=0.0, well_pref=None, well_excess=0.0,
    i_hold=0.0, i_hold_base=0.0, wasted_i=-152.0,
    clear1=-143.0, clear2=-100.0, clear3=-58.0, clear4=390.0,
    height_soft=None, height_k=0.0,
)

# The same search with no tetris shaping at all: board quality and no clear
# rewards. Exists so "the search helped" and "the tetris terms helped" can be told
# apart rather than assumed.
DEFAULT_SEARCH_WEIGHTS = dict(
    TETRIS_SEARCH_WEIGHTS,
    ready=0.0,
    ready_full=0.0,
    well_pref=None,
    well_depth=0.0,
    well_column=None,
    i_hold=0.0,
    i_hold_base=0.0,
    wasted_i=0.0,
    clear1=0.0,
    clear2=0.0,
    clear3=0.0,
    clear4=0.0,
)

SEARCH_WEIGHT_PRESETS = {
    'tetris': TETRIS_SEARCH_WEIGHTS,
    'default': DEFAULT_SEARCH_WEIGHTS,
    'cold_clear': COLD_CLEAR_WEIGHTS,
    'versus': None,          # filled in below, once VERSUS_SEARCH_WEIGHTS exists
}

# Versus play. Same tetris-first core, plus the two things a garbage board needs:
# digging is priced by how buried each hole is, and height is priced by what is
# about to land on it. The well terms are left alone because the gapless readiness
# measure already declines to chase tetrises on a holed board -- under garbage the
# policy downstacks, which is the correct behaviour rather than a compromise.
VERSUS_SEARCH_WEIGHTS = dict(
    TETRIS_SEARCH_WEIGHTS,
    covered=-17.0,           # Cold Clear's covered_cells weight
    covered_sq=-1.0,
    incoming_danger=-70.0,
    # The doom loop: two unreachable-at-once holes and no I pieces left to pay
    # them. See the measurement table in the README's doom-loop section.
    i_dependency=-120.0,
    i_dependency_sq=-90.0,
    i_dependency_rows=-6.0,
    # Above this stack height a board that owes an I may only consider moves that
    # pay the debt down (see the gate in search_move). Below it, building still
    # wins, which is what keeps this from becoming a well-hoarding term.
    dep_gate_height=8,
    dep_suppress=1.0,
)
SEARCH_WEIGHT_PRESETS['versus'] = VERSUS_SEARCH_WEIGHTS


def resolve_search_weights(weights):
    """Preset name -> dict; dicts and None pass through (None means tetris-first)."""
    if isinstance(weights, str):
        try:
            return SEARCH_WEIGHT_PRESETS[weights]
        except KeyError:
            raise ValueError(
                f'unknown search weight preset {weights!r}; '
                f'known: {sorted(SEARCH_WEIGHT_PRESETS)}') from None
    return TETRIS_SEARCH_WEIGHTS if weights is None else weights


def clear_value(cleared, w):
    """Reward for clearing ``cleared`` rows. Never positive for a single."""
    if not cleared:
        return 0.0
    return w.get(f'clear{min(cleared, 4)}', 0.0)


def well_of(masks, heights, cols, rows, w):
    """``(depth, column)`` of the well this weight set pays for, or ``(0, -1)``.

    Two definitions live here and the weights choose between them: ``ready`` is the
    gapless measure that ships, ``well_depth`` is Cold Clear's lowest-column rule.
    """
    if w['ready'] or w['ready_full'] or w['i_hold'] or w['well_excess'] or w['dip']:
        return ready_depth(masks, heights, cols, rows, w['max_well_depth'],
                           w['well_pref'])
    if w['well_depth'] or w['well_column']:
        return well_depth(masks, heights, cols, rows, w['max_well_depth'])
    return 0, -1


def covered_cells(masks, heights, cap=6):
    """``(total, total_sq)`` blocks sitting on top of each hole -- Cold Clear's term.

    A hole under six blocks can only be reached by clearing the six rows above it;
    a hole under one is fixed by the next piece. A plain hole count cannot tell
    those apart, and on a garbage board that difference is the whole game: it is
    what makes digging beat stacking on top of the mess.
    """
    total = 0
    total_sq = 0
    for m, h in zip(masks, heights):
        holes = (~m) & ((1 << h) - 1)
        while holes:
            low = holes & -holes
            cover = (m >> low.bit_length()).bit_count()
            if cover > cap:
                cover = cap
            total += cover
            total_sq += cover * cover
            holes ^= low
    return total, total_sq


def i_dependencies(masks, heights):
    """``(columns, rows)`` of gaps only a vertical I can fill -- Blockfish's term.

    From the dedicated downstacking engine Blockfish
    (``blockfish-engine/src/ai/eval.rs``, whose entire board cost is
    ``5 * rows + 10 * piece_estimate + 10 * i_dependencies``): a column counts when
    it holds **at least three consecutive empty rows that are one cell wide** --
    both neighbours filled on each of those rows. Nothing but a vertical I is one
    cell wide, and pieces only descend from above, so such a gap can be neither
    filled nor reached sideways. That is a hole the board is *waiting on a
    specific piece* to fix, which is a different problem from a hole.

    ``columns`` is how many columns carry one, which is Blockfish's count.
    ``rows`` is a cheap depth proxy: the number of rows that begin a run of three
    or more, so a four-deep gap counts twice a three-deep one. Both come out of
    the bitmasks -- about five integer operations per column, no pixel walking,
    because this runs at every search node.

    Blockfish prices one dependency at twice a row of height. The weights here go
    higher, and the second one higher again, because the failure being guarded
    against is not one unreachable hole but two at once: the board then needs two
    I pieces while the bag supplies one every seven, and the bot stacks in the
    middle waiting for pieces that are not coming.
    """
    cols = len(masks)
    if cols < 2:
        return 0, 0
    columns = 0
    rows = 0
    for x in range(cols):
        height = heights[x]
        if height <= 0:
            continue
        gap = (~masks[x]) & ((1 << height) - 1)     # empty cells below the top
        # The board edge is as solid as a filled column, so both edge columns are
        # scanned too. An earlier version looped over range(1, cols - 1) and so
        # missed the single most common place for one of these to appear: a
        # covered one-wide shaft dug at the wall, which nothing but an I can fill.
        left = masks[x - 1] if x > 0 else ~0
        right = masks[x + 1] if x < cols - 1 else ~0
        one_wide = gap & left & right
        if not one_wide:
            continue
        # A row begins a run of >= 3 when it and the two rows above it are all
        # one-wide gaps; the higher rows of the board are the higher bits.
        deep = one_wide & (one_wide >> 1) & (one_wide >> 2)
        if deep:
            columns += 1
            rows += deep.bit_count()
    return columns, rows


def evaluate(masks, heights, kind, hold, w, rows, incoming=0):
    """Static value of a board, plus what the piece situation is worth.

    Every term is skipped when its weight is zero, which keeps the common case
    (the shipping weight set) to a handful of integer operations per node.

    ``incoming`` is the garbage queued against this board. It is zero in solo play
    and in every measurement in the README; versus turns on ``incoming_danger`` so
    that height is priced by what is about to land on it rather than on its own.
    """
    cols = len(masks)
    top = max(heights)
    score = w['max_height'] * top

    if w['holes'] or w['holes_sq']:
        holes = 0
        holes_sq = 0
        for m, h in zip(masks, heights):
            c = h - m.bit_count()
            holes += c
            holes_sq += c * c
        score += w['holes'] * holes + w['holes_sq'] * holes_sq

    if w['covered'] or w['covered_sq']:
        cover, cover_sq = covered_cells(masks, heights, w['covered_cap'])
        score += w['covered'] * cover + w['covered_sq'] * cover_sq

    deps = 0
    dep_rows = 0
    if (w['i_dependency'] or w['i_dependency_sq'] or w['i_dependency_rows']
            or w['dep_suppress']):
        deps, dep_rows = i_dependencies(masks, heights)
        if deps:
            # Linear plus squared on purpose: the first dependency is expensive,
            # the second is what actually loses the game, and a linear term barely
            # separates "one hole I owe an I" from "two, and the bag gives one I
            # per seven pieces".
            score += (w['i_dependency'] * deps
                      + w['i_dependency_sq'] * deps * deps
                      + w['i_dependency_rows'] * dep_rows)

    # A board that owes an I must not also be building a well and hoarding I
    # pieces to fill it. That is the doom loop in one sentence: the evaluator pays
    # for a four-deep well and for an I in hand, so the bot keeps stacking in the
    # middle waiting for I pieces that are already spoken for, while the debt it
    # cannot pay grows underneath. While a dependency is open, the well and
    # I-hoarding terms stand down.
    keep = 1.0
    if deps and w['dep_suppress']:
        keep = max(0.0, 1.0 - w['dep_suppress'])

    if w['aggregate_height']:
        score += w['aggregate_height'] * sum(heights)

    depth, well = well_of(masks, heights, cols, rows, w)

    if w['bumpiness'] or w['bumpiness_sq']:
        bumpiness = 0
        bumpiness_sq = 0
        skip = well if w['skip_well_bumpiness'] else -1
        for x in range(cols - 1):
            if x == skip or x + 1 == skip:
                continue              # Cold Clear excludes the well from bumpiness
            d = heights[x] - heights[x + 1]
            if d < 0:
                d = -d
            bumpiness += d
            bumpiness_sq += d * d
        score += w['bumpiness'] * bumpiness + w['bumpiness_sq'] * bumpiness_sq

    if w['top_half'] and top > w['top_half_at']:
        score += w['top_half'] * (top - w['top_half_at'])
    if w['top_quarter'] and top > w['top_quarter_at']:
        score += w['top_quarter'] * (top - w['top_quarter_at'])

    if depth:
        score += w['ready'] * depth * keep
        if depth >= w['max_well_depth']:
            score += w['ready_full'] * keep
        score += w['well_depth'] * depth
        bias = w['well_column']
        if bias:
            score += bias[well]
        if w['well_excess'] and well >= 0:
            solid = solid_heights(masks)
            lo, hi, lo_i = _two_smallest(solid)
            other = hi if well == lo_i else lo
            over = (other - heights[well]) - w['max_well_depth'] if other else 0
            if over > 0:
                score += w['well_excess'] * over

    if w['i_hold'] and (kind == 'I' or hold == 'I'):
        # Outside the `depth` branch on purpose: the flat part is what makes the
        # *first* I worth saving, which matters most on a board whose well is not
        # deep yet.
        score += (w['i_hold_base'] + w['i_hold'] * (depth / 4.0)) * keep

    if w['dip']:
        # Charged on every dip *except* the well in use. Without the exemption the
        # term taxes the very pit it is meant to protect: a four-deep well costs
        # 16 * dip, which at any useful weight cancels the whole well bonus, and
        # the measured result was a policy that stopped building wells at all.
        spare = dip_cost(heights, cols) - depth * depth
        if spare > 0:
            score += w['dip'] * spare
    soft = w['height_soft']
    if soft is not None and top > soft:
        over = top - soft
        score -= w['height_k'] * over * over
    if incoming and w['incoming_danger']:
        over = top + incoming - (rows - 1)
        if over > 0:
            score += w['incoming_danger'] * over * over
    return score


# --- the search -------------------------------------------------------------

class _Node:
    """One board in the search, plus the piece situation around it.

    ``path`` is what the line of play has collected so far (clear values and
    action costs) and ``leaf`` is the static score of the board it reached. They
    are kept apart on purpose: ranking on ``path + leaf`` compares boards at the
    same ply fairly, whereas folding ``leaf`` into ``path`` would count the static
    evaluation once per ply and make a depth-3 search quietly weight board quality
    three times as heavily as a depth-1 one. Cold Clear splits the same two
    quantities (``Reward`` accumulates along the path, ``Value`` describes the
    board).
    """

    __slots__ = ('masks', 'heights', 'kind', 'hold', 'qpos', 'path', 'leaf',
                 'value', 'first', 'cleared')

    def __init__(self, masks, heights, kind, hold, qpos, path, leaf, first,
                 cleared=0):
        self.masks = masks
        self.heights = heights
        self.kind = kind
        self.hold = hold
        self.qpos = qpos
        self.path = path
        self.leaf = leaf
        self.value = path + leaf
        self.first = first
        self.cleared = cleared


def _swapped(node, use_hold, queue):
    """(piece actually placed, hold afterwards, queue position afterwards)."""
    if not use_hold:
        return node.kind, node.hold, node.qpos + 1
    if node.hold is not None:
        return node.hold, node.kind, node.qpos
    if node.qpos < len(queue):
        return queue[node.qpos], node.kind, node.qpos + 1
    return None, node.hold, node.qpos


def _step(node, use_hold, rot, x, y, queue, rows, cols, w, buffer_rows,
          incoming=0):
    """Advance a node by one move. Returns the child, or None past the queue."""
    placed, new_hold, qpos = _swapped(node, use_hold, queue)
    if placed is None:
        return None
    masks, cleared = apply_placement(node.masks, placed, rot, x, y, rows)
    heights = heights_of(masks)
    nxt = queue[qpos] if qpos < len(queue) else None
    path = node.path
    if locks_out(placed, rot, y, buffer_rows):
        path -= 1e6                        # lock out: the game ends here
    elif nxt is not None and spawn_blocked(masks, nxt, rows, cols):
        path -= 1e6                        # block out on the next piece
    elif nxt is None and topped_out(masks, rows, cols):
        path -= 1e6
    path += clear_value(cleared, w)
    # Throwing the tetris piece away: an I that clears nothing on a board where a
    # tetris was already set up. Cold Clear charges its equivalent (``wasted_t``)
    # and this is the one place our shallower search cannot see the loss itself.
    if w['wasted_i'] and placed == 'I' and not cleared:
        if well_of(node.masks, node.heights, cols, rows, w)[0] >= 4:
            path += w['wasted_i']
    leaf = evaluate(masks, heights, nxt, new_hold, w, rows, incoming)
    return _Node(masks, heights, nxt, new_hold, qpos, path, leaf, node.first,
                 cleared)


def _children(node, queue, rows, cols, w, buffer_rows, incoming=0):
    """Every legal child of ``node`` as ``(use_hold, rot, x, child)``."""
    out = []
    for rot, x, y in placements(node.masks, node.heights, node.kind, rows, cols):
        child = _step(node, False, rot, x, y, queue, rows, cols, w, buffer_rows,
                      incoming)
        if child is not None:
            out.append((False, rot, x, child))
    if node.hold is not None or node.qpos < len(queue):
        alt = node.hold if node.hold is not None else queue[node.qpos]
        for rot, x, y in placements(node.masks, node.heights, alt, rows, cols):
            child = _step(node, True, rot, x, y, queue, rows, cols, w,
                          buffer_rows, incoming)
            if child is not None:
                out.append((True, rot, x, child))
    return out


def search_move(game, actions=None, depth=3, beam=8, weights=None,
                allow_hold=True, incoming=0):
    """Choose a move by beam search. Returns ``(use_hold, rotation, x)`` or None.

    ``actions`` are the caller's legal moves (three-tuples or bare ``(rotation,
    x)``); the returned move is always one of them, so the search cannot hand back
    a move the engine would reject. Deeper plies enumerate their own placements on
    the fast bitboard.

    ``depth`` is how many pieces to look ahead -- 1 is a plain greedy choice over
    the same evaluation -- and ``beam`` how many boards survive each ply.

    ``incoming`` is the garbage queued against this board. When it is non-zero the
    search also applies Cold Clear's survival gate: among the legal moves it takes
    the best one that can absorb what is about to land, and only when *nothing*
    survives does it fall back to the best move outright, which is the "take the
    max-damage move" half of their ``pick_move``.
    """
    from .heuristic import hold_options

    if game.current is None or game.game_over:
        return None
    if actions is None:
        actions = (hold_options(game) if allow_hold
                   else [(False, r, x)
                         for r, x in valid_actions(game.board, game.current)])
    if not actions:
        return None

    w = resolve_search_weights(weights)
    rows, cols = game.rows, game.cols
    masks = board_masks(game)
    heights = heights_of(masks)
    queue = tuple(p.kind for p in game.next_queue)
    hold = game.hold_piece.kind if (game.hold_piece is not None
                                    and not game.hold_used) else None
    root = _Node(masks, heights, game.current.kind, hold, 0, 0.0, 0.0, None)

    # Ply 1 walks the caller's actions, so the answer is legal by construction.
    scored = []
    for action in actions:
        if len(action) == 2:
            use_hold, rot, x = False, action[0], action[1]
        else:
            use_hold, rot, x = action
        if use_hold and (game.hold_used or not game.allow_hold):
            continue
        placed, _new_hold, _qpos = _swapped(root, use_hold, queue)
        if placed is None:
            continue
        y = landing_y(heights, placed, rot, x, rows)
        if y is None:
            continue                      # the fast model disagrees: skip safely
        child = _step(root, use_hold, rot, x, y, queue, rows, cols, w,
                      game.buffer_rows, incoming)
        if child is None:
            continue
        child.first = (use_hold, rot, x)
        scored.append((child.value, (use_hold, rot, x), child))

    if not scored:
        move = actions[0]
        return (False, move[0], move[1]) if len(move) == 2 else tuple(move)

    if incoming > 0 and w['incoming_danger']:
        def survives(child):
            # The move's own clear cancels part of the queue, so what has to fit
            # is the board after it, plus whatever is still coming.
            net = incoming - child.cleared
            if net < 0:
                net = 0
            return (max(child.heights) if child.heights else 0) + net <= rows - 1

        safe = [item for item in scored if survives(item[2])]
        if safe:
            scored = safe

    # The dependency gate. A soft price for an I-dependency is not enough, and the
    # measurement says so plainly: with `i_dependency` at -120 and its square at
    # -90, ten games of garbage pressure came out *byte-identical* to the same ten
    # with the term switched off. The reason is that a dependency is a symptom of a
    # tall board, and at height 14 the quadratic height pressure is already ~430
    # and at 18 it is ~1200, so a few hundred points of dependency penalty is
    # inside the noise of the choice it was supposed to change.
    #
    # So the dependency is treated like incoming garbage: as a constraint on the
    # root move rather than as a line in the score. If the board owes an I and the
    # stack is high, the search may only consider moves that pay it down (or at
    # least do not make it worse); if no such move exists it falls back to its
    # normal ranking, which is the "take the best move anyway" half of Cold Clear's
    # pick_move.
    gate_height = w.get('dep_gate_height')
    if gate_height and scored:
        deps_now, _rows = i_dependencies(masks, heights)
        if deps_now and max(heights) >= gate_height:
            graded = []
            for item in scored:
                child = item[2]
                deps_after = i_dependencies(child.masks, child.heights)[0]
                graded.append((deps_after, item))
            better = [item for deps_after, item in graded if deps_after < deps_now]
            if better:
                scored = better
            else:
                no_worse = [item for deps_after, item in graded
                            if deps_after <= deps_now]
                if no_worse:
                    scored = no_worse

    scored.sort(key=lambda item: (-item[0], item[1]))
    beam_nodes = [child for _v, _m, child in scored[:beam]]

    for _ply in range(max(1, depth) - 1):
        if not beam_nodes:
            break
        seen = {}
        children = []
        for node in beam_nodes:
            for _uh, _rot, _x, child in _children(node, queue, rows, cols, w,
                                                  game.buffer_rows):
                key = (child.masks, child.kind, child.hold)
                prev = seen.get(key)
                if prev is None or child.value > prev:
                    seen[key] = child.value
                    children.append(child)
        if not children:
            break
        children.sort(key=lambda c: -c.value)
        beam_nodes = []
        kept = set()
        for child in children:
            key = (child.masks, child.kind, child.hold)
            if key in kept:
                continue
            kept.add(key)
            beam_nodes.append(child)
            if len(beam_nodes) >= beam:
                break

    if beam_nodes and beam_nodes[0].first is not None:
        return beam_nodes[0].first
    return scored[0][1]
