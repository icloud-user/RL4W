"""Feature extraction.

The agent never sees the raw board. It sees a compact, interpretable vector --
the difference between an agent that learns Tetris on a CPU in an hour and one
that never learns at all.

The vector describes a **(board, candidate placement)** pair, not just a board.
That distinction matters: an earlier version scored each candidate purely by the
board it produced, so all ~40 candidates for a piece looked almost identical to
the network (they differ by four cells out of 220) and the network could not
rank them.

Feature versions
----------------
``version=1`` (37 features) is the original set, kept because a trained
checkpoint depends on its exact layout.

``version=2`` (44 features, the default) adds the classic Tetris board-quality
metrics: row and column transitions, the sum of well depths, and the maximum
column height. These are what let a network see *stacking quality* rather than
just "how tall" and "how many holes": transitions count the ragged edges that
make future placements awkward, and well depth measures the deep single-column
dips that only an I piece can fill.

New versions must never reorder or resize an existing one -- a checkpoint is
bound to the layout it was trained on.

Layout (version 2, N_FEATURES = 44):

    [ 0:10]  column heights, normalised by rows
    [10]     holes            (empty cells with a filled cell above them)
    [11]     bumpiness        (sum of |h[i] - h[i+1]|)
    [12]     aggregate height (sum of column heights)
    [13]     row transitions  (filled/empty boundaries along each row)
    [14]     column transitions (filled/empty boundaries down each column)
    [15]     well sum         (depth of wells: dips with walls on both sides)
    [16]     maximum column height
    [17:24]  one-hot of the piece being placed (7)
    [24:31]  one-hot of the next piece (7)
    [31]     landing depth    (how far down the board the piece comes to rest)
    [32]     lines this placement clears (0, 1, 2, 3 or 4)
    [33]     landing height   (lowest occupied row of the placed piece)
    [34]     cells of the placed piece that survive (4 - eroded)
    [35]     holes created by this placement
    [36]     resulting aggregate height
    [37]     resulting column transitions
    [38]     grid holes after this placement
    [39]     resulting row transitions
    [40]     resulting well sum
    [41]     resulting holes minus the holes before the placement
    [42]     filled cells removed by the line clear
    [43]     resulting maximum column height

Counts are divided by constants so every input is roughly 0..1, which keeps the
network well conditioned without normalisation layers.
"""
from __future__ import annotations

import numpy as np

from .engine import KINDS, PIECES

N_BOARD_FEATURES = 16          # shared board statistics
N_EXTRA_BOARD = 1              # max height (versions 2+)
N_PLACEMENT_COMMON = 7         # placement statistics in both versions
N_PLACEMENT_EXTRA = 6          # extra placement statistics (versions 2+)

N_FEATURES_V1 = N_BOARD_FEATURES + 2 * len(PIECES) + N_PLACEMENT_COMMON   # 37
N_FEATURES_V2 = (N_BOARD_FEATURES + N_EXTRA_BOARD + 2 * len(PIECES)
                 + N_PLACEMENT_COMMON + N_PLACEMENT_EXTRA)                 # 44
# Version 3 = version 2 plus ``rows_with_holes`` on both sides of the pair: as a
# board statistic and as a property of the placement's resulting board.
N_FEATURES_V3 = N_FEATURES_V2 + 2                                          # 46
# Version 4 = version 3 plus a *tetris-readiness* summary. This is the one thing
# missing from every earlier version: not how the board looks, but whether an I
# dropped right now would complete four lines, and how deep the best well is.
# A one-ply evaluator (and a value network scoring single placements) cannot see
# that "this placement looks slightly worse but leaves the tetris one piece away".
N_TETRIS_READY = 3
N_FEATURES_V4 = N_FEATURES_V3 + N_TETRIS_READY                             # 49
# Board-side width per version (the part shared by every candidate placement).
_BOARD_WIDTH = {
    1: N_BOARD_FEATURES,
    2: N_BOARD_FEATURES + N_EXTRA_BOARD,
    3: N_BOARD_FEATURES + 2 * N_EXTRA_BOARD,
    4: N_BOARD_FEATURES + 2 * N_EXTRA_BOARD + N_TETRIS_READY,
}

VERSIONS = (1, 2, 3, 4)
DEFAULT_VERSION = 1


def n_features(version: int = DEFAULT_VERSION) -> int:
    """Feature-vector length for a version."""
    if version == 1:
        return N_FEATURES_V1
    if version == 2:
        return N_FEATURES_V2
    if version == 3:
        return N_FEATURES_V3
    if version == 4:
        return N_FEATURES_V4
    raise ValueError(f'unknown feature version {version!r}; '
                     f'expected one of {VERSIONS}')


def board_width(version):
    """Width of the board-side (per-decision) part of the vector."""
    try:
        return _BOARD_WIDTH[version]
    except KeyError:
        raise ValueError(f'unknown feature version {version!r}') from None


# The default length, for callers that do not care about versions.
N_FEATURES = N_FEATURES_V2

_HEIGHT_SCALE = 20.0
_COUNT_SCALE = 30.0
_TRANSITION_SCALE = 40.0

# One-hot rows indexed by piece letter.
_PIECE_ONEHOT = {
    kind: np.eye(len(KINDS), dtype=np.float32)[i]
    for i, kind in enumerate(KINDS)
}
_MISSING_PIECE = np.zeros(len(KINDS), dtype=np.float32)

# Board metrics are memoised on the board: the same board recurs constantly
# (every candidate for one piece shares a board; a placement's successor is the
# next piece's board).
_CACHE: dict[tuple, tuple] = {}
_CACHE_MAX = 200_000


def _key(grid):
    return tuple(tuple(row) for row in grid)


def board_metrics(grid, rows, cols):
    """Board statistics.

    Returns ``(heights, holes, bumpiness, aggregate, row_trans, col_trans,
    wells, max_height, rows_with_holes)``.

    ``wells`` is the summed depth of wells: empty cells whose left and right
    neighbours are both filled (a wall counts as filled). A deep well is the
    shape an I piece is worth saving for, so its depth is the natural summary.

    ``rows_with_holes`` counts rows containing at least one hole. In the
    Dellacherie-20 set this carries the *largest* weight of any term, larger than
    the hole count itself: twenty holes spread over one row are far less damaging
    than twenty holes spread over twenty rows, because the latter cannot be
    cleared by any single line clear.
    """
    heights = [0] * cols
    for x in range(cols):
        for y in range(rows):
            if grid[y][x] is not None:
                heights[x] = rows - y
                break

    holes = 0
    rows_with_holes = 0
    for x in range(cols):
        seen = False
        for y in range(rows):
            if grid[y][x] is not None:
                seen = True
            elif seen:
                holes += 1

    for y in range(rows):
        row = grid[y]
        for x in range(cols):
            if row[x] is not None:
                continue
            # A hole in this row: something filled above it in the same column.
            above = False
            for yy in range(y):
                if grid[yy][x] is not None:
                    above = True
                    break
            if above:
                rows_with_holes += 1
                break

    bumpiness = 0
    for x in range(cols - 1):
        bumpiness += abs(heights[x] - heights[x + 1])

    aggregate = sum(heights)
    max_height = max(heights) if heights else 0

    # Row transitions: filled<->empty boundaries, treating outside as empty.
    row_trans = 0
    for y in range(rows):
        prev = True          # border counts as filled so the edge is counted
        row = grid[y]
        for x in range(cols):
            filled = row[x] is not None
            if filled != prev:
                row_trans += 1
            prev = filled
        if not prev:
            row_trans += 1

    # Column transitions: the same down each column.
    col_trans = 0
    for x in range(cols):
        prev = True
        for y in range(rows):
            filled = grid[y][x] is not None
            if filled != prev:
                col_trans += 1
            prev = filled
        if not prev:
            col_trans += 1

    # Wells: empty cells with walls on both sides.
    wells = 0
    for x in range(cols):
        for y in range(rows):
            if grid[y][x] is not None:
                continue
            left_wall = (x == 0) or (grid[y][x - 1] is not None)
            right_wall = (x == cols - 1) or (grid[y][x + 1] is not None)
            if left_wall and right_wall:
                wells += 1

    return heights, holes, bumpiness, aggregate, row_trans, col_trans, wells, \
        max_height, rows_with_holes


def metrics_for(grid, rows, cols):
    """Memoised ``board_metrics``."""
    key = _key(grid)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.clear()
    value = board_metrics(grid, rows, cols)
    _CACHE[key] = value
    return value


def holes_of(grid, rows, cols):
    """Count holes: empty cells with a filled cell somewhere above them.

    Cannot be derived from column heights alone, because a gap *inside* a column
    is invisible to heights. Exposed here so there is one implementation.
    """
    holes = 0
    for x in range(cols):
        seen = False
        for y in range(rows):
            if grid[y][x] is not None:
                seen = True
            elif seen:
                holes += 1
    return holes


def tetris_readiness(heights, cols):
    """(best_well_depth, is_tetris_ready, columns_at_min_height) for a board.

    ``is_tetris_ready`` is 1.0 when some column would take an I piece and clear
    exactly four rows -- i.e. it is 4 deeper than the wall beside it. That single
    bit is the difference between "this board looks fine" and "this board is one
    I away from a tetris", which a per-placement value cannot otherwise express.
    """
    min_h = min(heights) if heights else 0
    at_min = 0
    depth = 0
    ready = 0.0
    for x in range(cols):
        if heights[x] == min_h:
            at_min += 1
        left = heights[x - 1] if x > 0 else heights[x]
        right = heights[x + 1] if x < cols - 1 else heights[x]
        gap = min(4, min(left, right) - heights[x])
        if gap > depth:
            depth = gap
        if gap >= 4:
            ready = 1.0
    return float(depth), ready, float(at_min)


def board_vector_from_metrics(metrics, current_kind, next_kind,
                              version=DEFAULT_VERSION):
    """Features that depend only on the board and the two piece identities.

    Computed once per decision; the per-candidate features are appended by
    ``placement_vector``.
    """
    (heights, holes, bumpiness, aggregate, row_trans, col_trans, wells,
     max_height, rows_with_holes) = metrics[:9]
    size = board_width(version) + 2 * len(PIECES)
    vec = np.empty(size, dtype=np.float32)
    inv_h = 1.0 / _HEIGHT_SCALE
    inv_c = 1.0 / _COUNT_SCALE
    for i, h in enumerate(heights):
        vec[i] = h * inv_h
    vec[10] = holes * inv_c
    vec[11] = bumpiness * inv_c
    vec[12] = aggregate * inv_c
    vec[13] = row_trans / _TRANSITION_SCALE
    vec[14] = col_trans / _TRANSITION_SCALE
    vec[15] = wells * inv_c

    piece_at = N_BOARD_FEATURES
    if version >= 2:
        vec[16] = max_height * inv_h
        piece_at = N_BOARD_FEATURES + N_EXTRA_BOARD
    if version >= 3:
        vec[17] = rows_with_holes * inv_c
        piece_at = N_BOARD_FEATURES + 2 * N_EXTRA_BOARD
    if version >= 4:
        depth, ready, at_min = tetris_readiness(heights, len(heights))
        vec[18] = depth / 4.0
        vec[19] = ready
        vec[20] = at_min / max(1, len(heights))
        piece_at = N_BOARD_FEATURES + 2 * N_EXTRA_BOARD + N_TETRIS_READY

    vec[piece_at:piece_at + 7] = (_MISSING_PIECE if current_kind is None
                                  else _PIECE_ONEHOT[current_kind])
    vec[piece_at + 7:piece_at + 14] = (_MISSING_PIECE if next_kind is None
                                       else _PIECE_ONEHOT[next_kind])
    return vec


def board_vector(grid, rows, cols, current_kind, next_kind,
                 version=DEFAULT_VERSION):
    return board_vector_from_metrics(
        metrics_for(grid, rows, cols), current_kind, next_kind, version)


def board_vector_game(game, version=DEFAULT_VERSION):
    """Features for the game's current decision point."""
    current = game.current.kind if game.current is not None else None
    nxt = game.next_queue[0].kind if game.next_queue else None
    return board_vector(game.board.grid, game.rows, game.cols, current, nxt,
                        version)


def placement_vector(base, placement, version=DEFAULT_VERSION):
    """Append a candidate placement's statistics to ``base``.

    ``placement`` is the dict returned by ``agent.placement_stats``.

    Version 2 appends 13 statistics; version 1 appends the original 7, ending
    with the resulting aggregate height. Getting that last slot wrong is not a
    cosmetic error: an unwritten slot in an ``np.empty`` array holds whatever the
    allocator left behind, and a network will happily key on it.
    """
    size = n_features(version)
    # Zeros, not empty: a slot that a version does not write must be 0 rather
    # than uninitialised memory.
    vec = np.zeros(size, dtype=np.float32)
    n_base = len(base)
    vec[:n_base] = base

    rows = placement.get('rows', 22.0)
    landing_top = placement.get('landing_top', rows)
    landing_bottom = placement.get('landing_bottom', rows)
    eroded = placement.get('eroded', 0)

    inv_h = 1.0 / rows
    inv_c = 1.0 / _COUNT_SCALE
    o = n_base
    # Landing depth: how far down the board the piece comes to rest. Larger
    # means lower, which is usually safer.
    vec[o + 0] = (rows - landing_top) * inv_h
    vec[o + 1] = placement.get('cleared', 0) / 4.0
    vec[o + 2] = landing_bottom * inv_h
    vec[o + 3] = (4 - eroded) / 4.0
    vec[o + 4] = placement.get('holes_created', 0) / 4.0

    if version == 1:
        # Original layout: four placement statistics, then the resulting maximum
        # height at slot 5... except that the original code wrote the *aggregate*
        # height into slot 5 as well, so slots 5 and 6 both carry it and there is
        # no max-height input at all. That is reproduced faithfully here, because
        # a checkpoint trained on it depends on the exact values.
        agg = placement.get('aggregate', 0) / (rows * 10.0)
        vec[o + 5] = agg
        vec[o + 6] = agg
        return vec

    vec[o + 5] = placement.get('aggregate', 0) / (rows * 10.0)
    vec[o + 6] = placement.get('result_col_trans', 0) / _TRANSITION_SCALE
    vec[o + 7] = placement.get('result_holes', 0) * inv_c
    vec[o + 8] = placement.get('result_row_trans', 0) / _TRANSITION_SCALE
    vec[o + 9] = placement.get('result_wells', 0) * inv_c
    vec[o + 10] = placement.get('holes_delta', 0) * inv_c
    vec[o + 11] = placement.get('filled_removed', 0) / (rows * 10.0)
    vec[o + 12] = placement.get('result_max_height', 0) * inv_h
    if version >= 3:
        vec[o + 13] = placement.get('result_rows_with_holes', 0) * inv_c
    return vec


def fingerprint(version=DEFAULT_VERSION):
    """A stable identifier for a feature layout.

    Recorded in checkpoints so that loading weights trained on a different
    layout is detected instead of silently producing worse play. This matters
    because a wrong-but-same-width layout is *shape valid*: ``load_state_dict``
    succeeds, every forward pass runs, and the agent just plays badly. That
    failure mode cost real debugging time, so it is now checked explicitly.

    Bump this whenever the meaning or order of any feature changes.
    """
    import hashlib
    spec = (f'v{version}|board={N_BOARD_FEATURES}|extra={N_EXTRA_BOARD}'
            f'|place={N_PLACEMENT_COMMON}|place_extra={N_PLACEMENT_EXTRA}'
            f'|pieces={len(PIECES)}')
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def check_fingerprint(saved, version=DEFAULT_VERSION):
    """Compare a checkpoint's recorded fingerprint with this build's."""
    current = fingerprint(version)
    if saved and saved != current:
        return (f'feature layout changed: checkpoint was trained on {saved}, '
                f'this build computes {current}. Weights are still loadable but '
                f'the agent will play worse than it did in training; retrain '
                f'from scratch or restore the matching layout.')
    return None


def clear_cache():
    _CACHE.clear()


def cache_size():
    return len(_CACHE)
