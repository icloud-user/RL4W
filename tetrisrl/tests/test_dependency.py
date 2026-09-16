"""Tests for the I-dependency term. Run: python tests/test_dependency.py

An "I dependency" is a gap that is one cell wide and at least three rows deep,
with filled cells on both sides at those rows: nothing but a vertical I can fill
it, because every other piece is two or more cells wide and pieces only descend
from above. Blockfish calls it `i_dependencies` and prices one at twice a row of
height; `search.i_dependencies` is that term, and these checks pin its definition,
its edges, and that the versus weights actually pay for it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import search as S                    # noqa: E402

FAILS = []
CHECKS = 0

ROWS = 22
COLS = 10


def check(name, cond, detail=''):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


def col(rows):
    """A column bitmask from the rows that are filled. Bit 0 is the bottom row."""
    mask = 0
    for row in rows:
        mask |= 1 << (ROWS - 1 - row)
    return mask


def solid(top, bottom=0):
    """Rows ``bottom`` .. ``top`` inclusive (top is the smaller row index)."""
    return col(range(bottom, top + 1) if top >= bottom else [])


def shaft(shaft_col, depth, cover_from=3, cols=COLS):
    """A covered one-cell-wide shaft of ``depth`` rows at the board's floor.

    Rows ``ROWS - depth`` .. ``ROWS - 1`` in ``shaft_col`` are empty, the column
    above them is filled, and every neighbour (the wall counts as one) is filled
    at exactly the gap rows. That is the shape Blockfish's term exists for.
    """
    gap_top = ROWS - depth
    masks = [0] * cols
    masks[shaft_col] = solid(gap_top - 1, cover_from) if gap_top - 1 >= cover_from \
        else 0
    for neighbour in (shaft_col - 1, shaft_col + 1):
        if 0 <= neighbour < cols:
            masks[neighbour] = solid(ROWS - 1, gap_top)
    return tuple(masks)


def deps(masks):
    return S.i_dependencies(masks, S.heights_of(masks))


def cost(masks, weights=None):
    w = weights or S.resolve_search_weights('versus')
    return S.evaluate(masks, S.heights_of(masks), 'I', None, w, ROWS)


# --- the definition ---------------------------------------------------------

def test_empty_board_is_clean():
    masks = (0,) * COLS
    check('an empty board has no dependencies', deps(masks) == (0, 0), deps(masks))


def test_four_deep_one_wide_gap_counts():
    masks = shaft(4, 4)
    got = deps(masks)
    check('a four-deep one-wide gap is detected', got[0] == 1, (got, masks))
    check('and its depth is counted in rows', got[1] == 2, got)


def test_three_deep_counts():
    got = deps(shaft(4, 3))
    check('three rows deep is the threshold and counts', got == (1, 1), got)


def test_two_deep_does_not_count():
    got = deps(shaft(4, 2))
    check('two rows deep is below the threshold', got == (0, 0), got)


def test_six_deep_counts_more_rows():
    got = deps(shaft(4, 6))
    check('a deeper shaft is still one column', got[0] == 1, got)
    check('but carries more rows', got[1] > 2, got)


def test_flat_hole_does_not_count():
    masks = list(shaft(4, 4))
    masks[4] = solid(17, 3) | col([19, 20, 21])   # the shaft filled but one row
    got = deps(tuple(masks))
    check('one flat hole is not a dependency', got == (0, 0), got)


def test_open_well_does_not_count():
    """A one-wide column with nothing above the gap is a well, not a dependency.

    An I dropped from above fills it, which is what the well is for, so pricing it
    as a dependency would recreate the well-hoarding failure the README documents.
    """
    masks = [0] * COLS
    masks[3] = solid(21, 12)
    masks[5] = solid(21, 12)
    check('an open one-wide well is not counted', deps(tuple(masks)) == (0, 0),
          deps(tuple(masks)))


def test_gap_wider_than_one_cell_does_not_count():
    """Two cells wide is fillable by more than an I, so it is just a hole."""
    masks = [0] * COLS
    masks[4] = solid(17, 3)
    masks[5] = solid(17, 3)
    masks[3] = solid(21, 18)
    masks[6] = solid(21, 18)
    check('a two-wide gap is not a dependency', deps(tuple(masks)) == (0, 0),
          deps(tuple(masks)))


def test_gap_with_no_wall_at_its_own_rows_does_not_count():
    """One cell wide at the floor but the neighbours are lower: fillable sideways.

    Only the bottom two rows are one-wide here, which is below the three-row
    threshold, so this is a hole rather than something the board is waiting on an
    I for.
    """
    masks = [0] * COLS
    masks[4] = solid(17, 3)
    masks[3] = solid(21, 20)
    masks[5] = solid(21, 20)
    check('a gap whose neighbours are lower is not a dependency',
          deps(tuple(masks)) == (0, 0), deps(tuple(masks)))


def test_two_dependencies():
    masks = list(shaft(4, 4))
    other = shaft(7, 4)
    for i in range(COLS):
        masks[i] |= other[i]
    got = deps(tuple(masks))
    check('two shafts count as two columns', got[0] == 2, got)
    check('and their rows add up', got[1] == 4, got)


def test_wall_columns_count():
    """A covered one-wide gap in the wall column: the wall is the second wall.

    Nothing but an I fits a one-wide shaft, and the board edge is as solid as a
    filled column, so this is a dependency like any other -- and the most common
    place for one to appear, since a well is usually dug at the edge.
    """
    got = deps(shaft(0, 4))
    check('a covered one-wide gap at the left wall is detected', got[0] == 1, got)
    got = deps(shaft(COLS - 1, 4))
    check('and at the right wall', got[0] == 1, got)


# --- the weights ------------------------------------------------------------

def test_term_is_off_in_solo_and_on_in_versus():
    solo = S.resolve_search_weights('tetris')
    versus = S.resolve_search_weights('versus')
    check('the solo tetris preset does not price dependencies',
          not solo['i_dependency'] and not solo['i_dependency_sq'])
    check('the versus preset does', versus['i_dependency'] < 0,
          versus['i_dependency'])


def test_a_dependency_costs_something():
    clean = tuple(solid(21, 3) for _ in range(COLS))
    check('the same board without the shaft is worth more',
          cost(shaft(4, 4)) < cost(clean),
          (cost(shaft(4, 4)), cost(clean)))


def test_second_dependency_costs_more_than_twice_the_first():
    w = S.resolve_search_weights('versus')
    clean = tuple(solid(21, 3) for _ in range(COLS))
    one = shaft(4, 4)
    two = list(shaft(4, 4))
    other = shaft(7, 4)
    for i in range(COLS):
        two[i] |= other[i]
    c0, c1, c2 = cost(clean, w), cost(one, w), cost(tuple(two), w)
    check('one dependency costs something', c1 < c0, (c0, c1))
    check('the second costs more than twice the first',
          (c0 - c2) > 2 * (c0 - c1), (c0, c1, c2))


def test_filling_a_dependency_beats_stacking_elsewhere():
    """The decision the term exists to change: pay the debt, do not build on it."""
    w = S.resolve_search_weights('versus')
    owed = shaft(4, 4)
    paid = list(owed)
    paid[4] = solid(19, 3)                   # filled to within two rows of cover
    stacked = list(owed)
    stacked[8] = solid(21, 16)               # a tower built far away instead
    if deps(tuple(paid))[0]:
        check('the partly-filled board is no longer a dependency', False,
              deps(tuple(paid)))
        return
    check('paying the debt scores better than building elsewhere',
          cost(tuple(paid), w) > cost(tuple(stacked), w),
          (cost(tuple(paid), w), cost(tuple(stacked), w)))


def main():
    for fn in (test_empty_board_is_clean,
               test_four_deep_one_wide_gap_counts,
               test_three_deep_counts,
               test_two_deep_does_not_count,
               test_six_deep_counts_more_rows,
               test_flat_hole_does_not_count,
               test_open_well_does_not_count,
               test_gap_wider_than_one_cell_does_not_count,
               test_gap_with_no_wall_at_its_own_rows_does_not_count,
               test_two_dependencies,
               test_wall_columns_count,
               test_term_is_off_in_solo_and_on_in_versus,
               test_a_dependency_costs_something,
               test_second_dependency_costs_more_than_twice_the_first,
               test_filling_a_dependency_beats_stacking_elsewhere):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all dependency tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
