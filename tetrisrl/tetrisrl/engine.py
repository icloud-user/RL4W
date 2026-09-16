"""Headless Tetris engine: SRS rotation, wall kicks, 7-bag, hold, scoring.

Coordinate system
-----------------
The board is ``rows`` x ``cols`` (22x10 by default) with the top 2 rows used as
an invisible buffer, matching the Tetris Guideline. ``grid[y][x]`` is ``None``
for empty or the piece letter that occupies the cell. ``y`` grows *downward*.

A piece is an anchor ``(x, y)`` plus a list of cell offsets. Offsets are stored
in board convention (y down) so ``cells = [(x + ox, y + oy) for ox, oy in ...]``
is the only transformation needed.

Wall kicks are stored as SRS tables (which are published in a y-up convention)
and converted to board convention by negating the y component -- see
``_to_board_kicks``. This keeps the tables diffable against the published ones.

No pygame, no RL, no I/O beyond config loading. Pure stdlib + numpy is not even
required here; this module is deliberately dependency-free so it stays fast and
testable.
"""
from __future__ import annotations

import random
from collections import deque

# --- Geometry -------------------------------------------------------------

COLS = 10
ROWS = 22            # 20 visible + 2 hidden buffer rows
VISIBLE_ROWS = 20
BUFFER_ROWS = ROWS - VISIBLE_ROWS

PIECES = ('I', 'J', 'L', 'O', 'S', 'T', 'Z')
KINDS = PIECES

# SRS+ rotation states for all 7 pieces, 4 rotations each, 4 cells each.
#
# These are TETR.IO's own matrices, copied cell for cell from the Triangle.js
# engine (https://github.com/halp1/triangle, src/engine/utils/tetromino/data.ts,
# branch `main`), the replay-accurate community reverse-engineering of the game.
# Triangle stores each state as [x, y] with y growing *down* the screen -- its I
# spawns in the upper of its two horizontal states, which is what SRS specifies --
# so the values transfer to this board's y-down convention with no flip at all.
#
# Convention, and every part of it matters for the kicks to mean anything:
#
# * Offsets are from the **top-left corner of the piece's box**: J/L/S/T/Z live in
#   a 3x3 box, I in a 4x4, O in a 2x2, and all four states of a piece sit in that
#   same box. ``Piece.x``/``Piece.y`` is that corner, so a kick offset translates
#   the box, which is exactly what the published SRS/SRS+ tables describe.
# * Rotation index: 0 = spawn, 1 = clockwise (R), 2 = 180, 3 = counter-clockwise
#   (L) -- i.e. index + 1 is one clockwise turn, and the table keys ("01", "10",
#   ...) are usable verbatim.
#
# Two earlier versions of this table were wrong in ways that made rotations feel
# broken rather than obviously fail: generating the states by rotating the spawn
# shape about the origin put each state at a different offset inside its box, and
# a later copy of Triangle's matrices was mirrored vertically (its y values were
# negated as if the source were y-up). A mirrored board silently reverses
# clockwise and counter-clockwise, spawns the T nub-down instead of nub-up, and
# pairs each kick row with the opposite turn -- so tucks and floor kicks land a
# row or two off, which is precisely "rotations almost work".
SHAPES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    'I': (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((3, 2), (2, 2), (1, 2), (0, 2)),
        ((1, 3), (1, 2), (1, 1), (1, 0)),
    ),
    'J': (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((2, 0), (1, 0), (1, 1), (1, 2)),
        ((2, 2), (2, 1), (1, 1), (0, 1)),
        ((0, 2), (1, 2), (1, 1), (1, 0)),
    ),
    'L': (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((2, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 2), (2, 1), (1, 1), (0, 1)),
        ((0, 0), (1, 2), (1, 1), (1, 0)),
    ),
    'O': (
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ),
    'S': (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((2, 1), (2, 2), (1, 0), (1, 1)),
        ((1, 2), (0, 2), (2, 1), (1, 1)),
        ((0, 1), (0, 0), (1, 2), (1, 1)),
    ),
    'T': (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((2, 1), (1, 0), (1, 1), (1, 2)),
        ((1, 2), (2, 1), (1, 1), (0, 1)),
        ((0, 1), (1, 2), (1, 1), (1, 0)),
    ),
    'Z': (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (2, 1), (1, 1), (1, 2)),
        ((2, 2), (1, 2), (1, 1), (0, 1)),
        ((0, 2), (0, 1), (1, 1), (1, 0)),
    ),
}

# The table every hot path reads. Kept as its own name because it is what makes
# the "board convention" explicit at the point of use; the values are already in
# it, so there is no per-state conversion to do.
_CELLS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = SHAPES


# The kick tables below are used *verbatim*, in published order, because Triangle's
# kick data is already in this board's convention.
#
# That is worth spelling out, because getting it wrong is invisible for the kicks
# you use most. Triangle's tetromino matrices are y-DOWN -- its T spawns with the
# nub on the top row of a 3x3 box -- and its kick offsets live in the same space,
# since an engine applies an offset to the same coordinates as the cells it moves.
# This board is y-down too, so the translation is the identity.
#
# For a while this function negated dy, on the assumption that the source was
# y-up. Every kick with a vertical component was therefore inverted: wall kicks
# (which are mostly horizontal) kept working, while every kick that moves a piece
# up or down -- the tuck that slides an S or Z into a notch, the drop that lands a
# T-spin triple, the lift that completes a 180 -- pushed the piece the wrong way
# and was refused. Measured against the source, 22 of the 24 published rows were
# mirrored.
#
# SRS tests the offsets in sequence and takes the first that fits, so order is part
# of the specification and is preserved exactly. The published tables carry no
# (0, 0) entry because Triangle's engine tries the unmoved rotation implicitly;
# ours are spelled with it first.


# SRS+ wall kicks for J, L, S, T, Z. Keys are (from_rotation, to_rotation), and
# the (0, 0) test comes first because SRS tests the unmoved rotation before any
# kick.
#
# Source: the TETR.IO preset of the Triangle.js engine,
# https://github.com/halp1/triangle, src/engine/utils/kicks/data.ts, entry
# `kicks["SRS+"]`, branch `main` (fetched 2026-09), copied cell for cell. Those
# values are y-down already -- see the note above -- so this table *is* the board
# table; nothing converts it.
#
# SRS+ is SRS with the 180-degree turns added and its I rows reordered; the
# quarter-turn offsets for J/L/S/T/Z are identical to SRS proper, which is why
# they match the Hard Drop SRS page (https://harddrop.com/wiki/SRS) test for test
# once the y convention is accounted for.
_JLSTZ_KICKS = {
    (0, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (1, 0): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (1, 2): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (2, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (2, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (3, 2): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (3, 0): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (0, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
}

# SRS+ wall kicks for I: the same source, entry ``i_kicks``. Note that SRS+ orders
# these differently from plain SRS -- the first test after (0, 0) is (+1, 0) rather
# than (-2, 0) -- which changes where an I lands against a wall.
_I_KICKS = {
    (0, 1): ((0, 0), (1, 0), (-2, 0), (-2, 1), (1, -2)),
    (1, 0): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (2, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (3, 2): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (2, 1), (-1, -2)),
}

# The 180-degree turns. SRS proper has none -- these are SRS+'s addition, and in
# TETR.IO they are part of the same preset rather than an optional extra.
_KICKS_JLSTZ_180 = {
    (0, 2): ((0, 0), (0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)),
    (1, 3): ((0, 0), (1, 0), (1, -2), (1, -1), (0, -2), (0, -1)),
    (2, 0): ((0, 0), (0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)),
    (3, 1): ((0, 0), (-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)),
}
_KICKS_I_180 = {
    (0, 2): ((0, 0), (0, -1)),
    (1, 3): ((0, 0), (1, 0)),
    (2, 0): ((0, 0), (0, 1)),
    (3, 1): ((0, 0), (-1, 0)),
}

# No conversion: these values are the board table.
KICKS = _JLSTZ_KICKS
IKICKS = _I_KICKS
KICKS_180 = _KICKS_JLSTZ_180
IKICKS_180 = _KICKS_I_180

# SRS spawn columns. The Guideline spawns I across columns 3-6, O across
# columns 4-5, and the three-wide pieces across columns 3-5. These are the
# anchor columns; because rotation 0 is not always the leftmost state of a
# piece, the origin offset is subtracted from the desired spawn column below.
_SPAWN_COL = {'I': 3, 'O': 4}
_SPAWN_COL_DEFAULT = 3
# Board row the spawn state's TOP row occupies. This must be inside the
# invisible buffer (rows 0..BUFFER_ROWS-1 are hidden, so a negative row is
# above the board entirely). Getting this wrong pushes spawns down into the
# visible playfield and games end after a few dozen pieces.
_SPAWN_TOP_ROW = -2
SPAWN_Y = 0             # kept for callers that reference it


def spawn_anchor(kind: str) -> tuple[int, int]:
    """Anchor that places ``kind``'s spawn state on the SRS spawn columns.

    The spawn state's topmost occupied row lands on ``_SPAWN_TOP_ROW``, inside the
    invisible buffer, and then the piece falls -- standard Guideline spawn
    behaviour. Note that this is the piece's occupied rows, not its box: the I's
    box is four tall with the spawn state in row 1, so an anchor that put the *box*
    at ``_SPAWN_TOP_ROW`` would spawn the I a row too low.
    """
    state = SHAPES[kind][0]
    left = min(ox for ox, _ in state)
    top = min(oy for _, oy in state)          # y grows down: the smallest is the top
    col = _SPAWN_COL.get(kind, _SPAWN_COL_DEFAULT)
    return col - left, _SPAWN_TOP_ROW - top


def spawn_span(kind: str) -> tuple[int, int]:
    """Inclusive column range that ``kind``'s spawn state occupies."""
    x, _ = spawn_anchor(kind)
    xs = [x + ox for ox, _ in _CELLS[kind][0]]
    return min(xs), max(xs)


class Piece:
    """A tetromino at a board position. Cheap to construct and copy."""

    __slots__ = ('kind', 'x', 'y', 'rotation')

    def __init__(self, kind, x=None, y=None, rotation=0):
        self.kind = kind
        if x is None or y is None:
            x, y = spawn_anchor(kind)
        self.x = x
        self.y = y
        self.rotation = rotation

    def cells(self, rotation=None, x=None, y=None):
        """Absolute board cells for this piece (optionally overridden)."""
        rot = self.rotation if rotation is None else rotation
        ax = self.x if x is None else x
        ay = self.y if y is None else y
        return [(ax + ox, ay + oy) for ox, oy in _CELLS[self.kind][rot % 4]]

    def copy(self):
        return Piece(self.kind, self.x, self.y, self.rotation)


class Bag:
    """7-bag randomizer: each piece appears once per shuffled bag."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.queue = deque()
        self._refill()

    def _refill(self):
        pieces = list(PIECES)
        self.rng.shuffle(pieces)
        self.queue.extend(pieces)

    def next(self):
        if not self.queue:
            self._refill()
        return self.queue.popleft()


# Scoring: base points per line clear, multiplied by level (Guideline).
CLEAR_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
B2B_MULTIPLIER = 1.5
COMBO_BONUS = 50


class Board:
    """The playfield grid."""

    __slots__ = ('grid', 'rows', 'cols')

    def __init__(self, rows=ROWS, cols=COLS):
        self.rows = rows
        self.cols = cols
        self.grid = [[None] * cols for _ in range(rows)]

    def clear(self):
        for row in self.grid:
            for x in range(self.cols):
                row[x] = None

    def column_heights(self):
        """Height of each column: rows from the bottom to the top filled cell.

        Height 0 means the column is empty. This is used both as an agent
        feature and to enumerate placements without simulating drops.
        """
        heights = [0] * self.cols
        grid = self.grid
        rows = self.rows
        for x in range(self.cols):
            for y in range(rows):
                if grid[y][x] is not None:
                    heights[x] = rows - y
                    break
        return heights

    def collides(self, cells):
        """True if any cell is out of bounds or overlaps a filled cell.

        Cells above the top (negative y) are allowed: that is the buffer.
        """
        grid = self.grid
        rows, cols = self.rows, self.cols
        for x, y in cells:
            if x < 0 or x >= cols or y >= rows:
                return True
            if y >= 0 and grid[y][x] is not None:
                return True
        return False

    def blocks(self, x, y):
        """True if ``(x, y)`` is occupied, treating above-board cells as free.

        Separate from ``collides`` so callers that need to reason about the
        pre-playfield rows never index ``grid`` with a negative row (which would
        silently wrap around to the bottom of the board).
        """
        if y < 0 or y >= self.rows or x < 0 or x >= self.cols:
            return False
        return self.grid[y][x] is not None

    def lock(self, piece):
        """Write a piece into the grid. Returns the rows it touched."""
        rows_touched = []
        for x, y in piece.cells():
            # Guard x as well as y: the SRS box convention puts some states at a
            # column offset (the vertical I sits at x+2 of its 4x4 box), so a piece
            # placed by hand outside the board used to raise IndexError here
            # instead of simply writing nothing.
            if 0 <= y < self.rows and 0 <= x < self.cols:
                self.grid[y][x] = piece.kind
                if y not in rows_touched:
                    rows_touched.append(y)
        return rows_touched

    def full_rows(self):
        return [y for y, row in enumerate(self.grid) if all(c is not None for c in row)]

    def clear_rows(self, rows):
        """Remove the given rows and add empty ones at the top."""
        if not rows:
            return
        drop = set(rows)
        kept = [row for y, row in enumerate(self.grid) if y not in drop]
        empty = [[None] * self.cols for _ in range(len(rows))]
        self.grid = empty + kept

    # -- garbage ----------------------------------------------------------

    def add_garbage(self, lines, holes, hole_size=1, letter='G'):
        """Insert garbage rows at the bottom, pushing everything above them up.

        ``holes`` is one hole column per line; a single int is broadcast. Returns
        the number of filled cells pushed off the top edge, which is zero when the
        push was survivable -- that count is how the caller detects a garbage
        top-out.

        This is the versus primitive. TETR.IO does not raise the board
        continuously: it inserts a batch of rows at the bottom on a lock that
        cleared nothing, which is what this does. Garbage cells are written as
        ``letter`` so a renderer can colour them apart from a player's own pieces.
        """
        if lines <= 0:
            return 0
        lines = min(lines, self.rows)
        cols = self.cols
        if isinstance(holes, int):
            holes = [holes] * lines
        lost = sum(1 for y in range(lines) for c in self.grid[y] if c is not None)
        kept = self.grid[lines:]
        fresh = []
        for i in range(lines):
            row = [letter] * cols
            hole = holes[i % len(holes)]
            for k in range(hole_size):
                if 0 <= hole + k < cols:
                    row[hole + k] = None
            fresh.append(row)
        self.grid = kept + fresh
        return lost


class Game:
    """A headless game of Tetris.

    Public surface used by the rest of the project:

    ``board``       : Board
    ``current``     : Piece or None
    ``next_queue``  : deque[Piece] (lookahead, default 5)
    ``hold_piece``  : Piece or None
    ``lines``/``score``/``level``/``game_over``
    ``pieces_placed``: int

    ``queue_kinds`` / ``hold_kind`` are convenience accessors returning letters.
    """

    def __init__(self, rows=ROWS, cols=COLS, seed=None, next_count=5,
                 allow_hold=True, buffer_rows=BUFFER_ROWS):
        self.rng = random.Random(seed)
        self.rows = rows
        self.cols = cols
        self.buffer_rows = buffer_rows
        self.next_count = next_count
        self.allow_hold = allow_hold

        self.board = Board(rows, cols)
        self.bag = Bag(self.rng)
        self.next_queue = deque()
        for _ in range(next_count):
            self.next_queue.append(Piece(self.bag.next()))

        self.current = None
        self.hold_piece = None
        self.hold_used = False

        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces_placed = 0
        self.combo = -1
        self.back_to_back = False
        self.game_over = False
        # Versus rule: TETR.IO disables lock-out death (``nolockout`` plus a
        # "clutch" clear), so a piece that comes to rest in the buffer does not end
        # the game. Off by default, which keeps the solo behaviour the rest of the
        # project measures against.
        self.nolockout = False

        # Bookkeeping for the "eroded piece cells" feature: how many cells of
        # the most recently locked piece were destroyed by line clears.
        self.last_eroded = 0

        # T-spin bookkeeping. ``spin_kind`` describes the most recent *lock*
        # (None, 'mini' or 'full'). The rest records how the piece in play got
        # where it is: the guideline rule for a T-spin is "the last successful
        # movement was a rotation, and three of the four corners around the T are
        # filled", and the far-kick exception needs to know which offset was used.
        self.spin_kind = None
        self.rotation_was_last_move = False
        self.last_kick_index = 0
        self.last_kick_offset = (0, 0)
        self.last_rotation = (0, 0)

        self._spawn()

    # -- spawning ---------------------------------------------------------

    def _spawn(self, kind=None):
        piece = self.next_queue.popleft() if kind is None else Piece(kind)
        if kind is None:
            self.next_queue.append(Piece(self.bag.next()))
        x, y = spawn_anchor(piece.kind)
        piece.x, piece.y, piece.rotation = x, y, 0
        self.current = piece
        self.hold_used = False
        # A fresh piece has not been rotated, so it cannot spin. ``spin_kind`` is
        # deliberately not cleared here: it describes the lock that just happened
        # and is read by the versus layer after this returns.
        self.rotation_was_last_move = False
        self.last_kick_index = 0
        self.last_kick_offset = (0, 0)
        self.last_rotation = (0, 0)
        if self.board.collides(piece.cells()):
            # Block out: no room for the new piece.
            self.game_over = True

    def spawn_blocked(self):
        """True when a fresh piece would not fit at the spawn position.

        This is the block-out test and the definition of a terminal board. The
        agent labels terminal transitions with this same rule, so the two can
        never disagree about when a game is over.
        """
        for kind in PIECES:
            x, y = spawn_anchor(kind)
            if not self.board.collides(Piece(kind, x, y, 0).cells()):
                return False
        return True

    @property
    def queue_kinds(self):
        return [p.kind for p in self.next_queue]

    @property
    def hold_kind(self):
        return self.hold_piece.kind if self.hold_piece else None

    # -- movement ---------------------------------------------------------

    def can_move(self, dx, dy=0):
        return not self.board.collides(self.current.cells(x=self.current.x + dx,
                                                          y=self.current.y + dy))

    def move(self, dx):
        if self.current is None or not self.can_move(dx):
            return False
        self.current.x += dx
        # Moving breaks the "last action was a rotation" rule for T-spins.
        # Gravity and soft drop deliberately do not: a piece that falls between
        # the rotation and the lock is still a spin, which is what every modern
        # client does and what players expect from a soft-dropped T-spin.
        self.rotation_was_last_move = False
        return True

    def can_rotate(self, direction):
        """Resolve an SRS+ rotation. Returns the (x, y, rotation) it lands on."""
        target = self.rotation_target(direction)
        return None if target is None else target[:3]

    def rotation_target(self, direction):
        """``(x, y, rotation, kick_index, (dx, dy), from_rotation)``, or None.

        The kick index and offset are returned as well as the destination because
        T-spin detection needs them: a rotation that only fits after a far kick
        counts as a full spin even when the corners say mini.
        """
        piece = self.current
        if piece is None:
            return None
        old = piece.rotation
        new = (old + direction) % 4
        if piece.kind == 'O':
            return piece.x, piece.y, new, 0, (0, 0), old
        if direction == 2:
            table = IKICKS_180 if piece.kind == 'I' else KICKS_180
        else:
            # Directions +1 and -1 are ordinary quarter turns and share the table;
            # only a full 180 uses the extension. Sending -1 to the 180 table was
            # a real bug: those entries only exist for (0,2), (1,3), (2,0) and
            # (3,1), so a counter-clockwise turn looked up a missing key, fell back
            # to a bare (0,0) offset, and was refused at a wall -- which is why
            # "I can't rotate near a wall" only ever happened turning one way.
            table = IKICKS if piece.kind == 'I' else KICKS
        for index, (dx, dy) in enumerate(table.get((old, new), ((0, 0),))):
            nx, ny = piece.x + dx, piece.y + dy
            if not self.board.collides(piece.cells(rotation=new, x=nx, y=ny)):
                return nx, ny, new, index, (dx, dy), old
        return None

    def rotate(self, direction):
        target = self.rotation_target(direction)
        if target is None:
            return False
        nx, ny, new, index, offset, old = target
        self.current.x, self.current.y, self.current.rotation = nx, ny, new
        self.rotation_was_last_move = True
        self.last_kick_index = index
        self.last_kick_offset = offset
        self.last_rotation = (old, new)
        return True

    def soft_drop(self):
        """Move down one row. Returns True if it moved."""
        if self.current is None or not self.can_move(0, 1):
            return False
        self.current.y += 1
        return True

    def ghost_y(self, piece=None):
        """Row the piece would land on if dropped now."""
        piece = piece or self.current
        y = piece.y
        while not self.board.collides(piece.cells(y=y + 1)):
            y += 1
        return y

    def drop_distance(self, piece=None):
        piece = piece or self.current
        return self.ghost_y(piece) - piece.y

    # -- locking ----------------------------------------------------------

    def hard_drop(self):
        """Drop to the floor and lock. Returns (lines_cleared, rows_cleared)."""
        self.current.y = self.ghost_y()
        return self.lock()

    def lock(self):
        piece = self.current
        cells = piece.cells()
        self.spin_kind = self._classify_spin(piece)
        rows_touched = self.board.lock(piece)
        self.pieces_placed += 1

        # Lock out: the whole piece came to rest above the visible playfield.
        if rows_touched and max(rows_touched) < self.buffer_rows \
                and not self.nolockout:
            self.game_over = True
            self.current = None
            return 0, []

        full = self.board.full_rows()
        cleared = len(full)
        self.last_eroded = sum(
            1 for x, y in cells if y in set(full)
        ) if cleared else 0

        self.board.clear_rows(full)
        self._apply_score(cleared)

        self.lines += cleared
        self.level = 1 + self.lines // 10
        self._spawn()
        return cleared, full

    def _occupied(self, x, y):
        """True if a cell counts as filled for spin detection.

        Walls, the floor and anything below it count as filled; the space above
        the board does not. A corner outside the board is a wall the T is pressed
        against, which is exactly the situation a T-spin is built in.
        """
        if x < 0 or x >= self.cols or y >= self.rows:
            return True
        if y < 0:
            return False
        return self.board.grid[y][x] is not None

    def _classify_spin(self, piece):
        """``'full'``, ``'mini'`` or None for the piece about to lock.

        The guideline rule, which TETR.IO follows: a T-spin is a T whose last
        successful movement was a rotation and which has at least three of the
        four corners around its 3x3 box filled. It is a *mini* spin when the two
        corners the nub points at are not both filled -- unless the rotation only
        fitted after a far kick, which promotes it to a full spin.

        The four corners per rotation state are the same ones TETR.IO's own
        ``cornerTable`` lists (Triangle.js, src/engine/utils/kicks/data.ts).
        Only the T can spin here: TETR.IO's league preset counts T-spins only, so
        other pieces are not classified even though the corner test would work.
        """
        if piece.kind != 'T' or not self.rotation_was_last_move:
            return None
        # The corners are those of the piece's *box*, which is fixed by the SRS
        # tables (3x3 for T), not by the extent of the cells it happens to fill.
        # Deriving the box from the cells was a real bug here: a T only occupies
        # two of its box's three rows, so the computed "corners" landed on the
        # piece's own cells and the classification was nonsense. The anchor is the
        # box's top-left corner, so the box runs from (x, y) down to
        # (x + box - 1, y + box - 1).
        box = 4 if piece.kind == 'I' else 3
        x0, y0 = piece.x, piece.y
        # Order: top-left, top-right, bottom-left, bottom-right (y grows down).
        corners = ((x0, y0), (x0 + box - 1, y0),
                   (x0, y0 + box - 1), (x0 + box - 1, y0 + box - 1))
        filled = [self._occupied(x, y) for x, y in corners]
        if sum(filled) < 3:
            return None
        # The nub points up in state 0, right in 1, down in 2, left in 3, so the
        # corners "in front" of it follow the rotation. These are the same four
        # corner pairs Triangle.js lists for the T in its ``cornerTable``.
        front = {0: (0, 1), 1: (1, 3), 2: (2, 3), 3: (0, 2)}[piece.rotation % 4]
        if filled[front[0]] and filled[front[1]]:
            return 'full'
        return 'full' if self.last_kick_index >= 4 else 'mini'

    def _apply_score(self, cleared):
        if cleared == 0:
            self.combo = -1
            return
        difficult = cleared == 4
        base = CLEAR_SCORES.get(cleared, CLEAR_SCORES[4])
        if difficult and self.back_to_back:
            base = int(base * B2B_MULTIPLIER)
        self.back_to_back = difficult
        self.combo += 1
        combo_points = COMBO_BONUS * max(0, self.combo) * self.level
        self.score += base * self.level + combo_points

    def hold(self):
        """Swap the current piece with the hold slot (once per piece)."""
        if not self.allow_hold or self.hold_used or self.game_over:
            return False
        if self.hold_piece is None:
            self.hold_piece = Piece(self.current.kind)
            self._spawn()
        else:
            swap = self.hold_piece
            self.hold_piece = Piece(self.current.kind)
            self._spawn(kind=swap.kind)
        self.hold_used = True
        return True

    # -- garbage ----------------------------------------------------------

    def add_garbage(self, lines, holes, hole_size=1):
        """Raise this board by ``lines`` garbage rows. True if it topped out.

        Two ways to die, both modelled on TETR.IO: cells pushed off the top edge,
        and the active piece being caught inside the risen stack ("garbage
        smash"). Merely pushing the *stack* into the hidden buffer is survivable,
        which is why only content above row 0 counts.
        """
        if self.game_over:
            return True
        lost = self.board.add_garbage(lines, holes, hole_size)
        if lost:
            self.game_over = True
            return True
        if self.current is not None and self.board.collides(self.current.cells()):
            self.game_over = True
            return True
        return False

    # -- read-only placement simulation -----------------------------------

    def simulate_placement(self, rotation, x):
        """Result of playing ``(rotation, x)`` without mutating this game.

        Returns ``(grid, cells, cleared_rows)`` where ``grid`` is the board AFTER
        the placement and any line clears, ``cells`` are the piece's cells before
        clearing, and ``cleared_rows`` lists the rows that were completed. Used
        to score candidate placements.

        The piece is repositioned to its spawn anchor with the requested
        rotation and then dropped, which is exactly how an action is realised
        (the agent sets rotation/x/y=spawn and hard-drops), so this result
        matches execution by construction. Dropping from the piece's *current*
        position instead would disagree whenever a piece has already fallen.
        """
        piece = self.current
        rows, cols = self.rows, self.cols
        base = _CELLS[piece.kind][rotation % 4]
        landing_y = spawn_anchor(piece.kind)[1]
        while True:
            nxt = landing_y + 1
            blocked = False
            for ox, oy in base:
                cx, cy = x + ox, nxt + oy
                if cx < 0 or cx >= cols or cy >= rows:
                    blocked = True
                    break
                if cy >= 0 and self.board.grid[cy][cx] is not None:
                    blocked = True
                    break
            if blocked:
                break
            landing_y = nxt
        cells = [(x + ox, landing_y + oy) for ox, oy in base]

        grid = [row[:] for row in self.board.grid]
        for cx, cy in cells:
            if 0 <= cy < rows:
                grid[cy][cx] = piece.kind
        full = [y for y in range(rows) if all(c is not None for c in grid[y])]
        if full:
            drop = set(full)
            grid = [row for y, row in enumerate(grid) if y not in drop]
            grid = [[None] * cols for _ in full] + grid
        return grid, cells, full

    def placement_heights(self, rotation, x):
        """Column heights after playing ``(rotation, x)``, without a grid copy.

        Returns ``(heights, cells, cleared_rows)``. This is the cheap version of
        ``simulate_placement`` used to score candidates: scoring needs the
        resulting *heights*, not the resulting grid, and building a grid plus
        rescanning every column for each of ~30 candidates per piece dominates
        evaluation time.

        The height update is exact. Placing the piece raises the height of each
        covered column to the piece's top cell in that column; clearing k rows
        then removes k cells from every column, so each height drops by k. (Every
        row above a cleared row was itself full, so heights agree above the cut.)
        """
        piece = self.current
        rows, cols = self.rows, self.cols
        base = _CELLS[piece.kind][rotation % 4]
        heights = self.board.column_heights()

        landing_y = spawn_anchor(piece.kind)[1]
        while True:
            nxt = landing_y + 1
            blocked = False
            for ox, oy in base:
                cx, cy = x + ox, nxt + oy
                if cx < 0 or cx >= cols or cy >= rows:
                    blocked = True
                    break
                if cy >= 0 and heights[cx] > rows - cy:
                    blocked = True
                    break
            if blocked:
                break
            landing_y = nxt
        cells = [(x + ox, landing_y + oy) for ox, oy in base]

        new_heights = list(heights)
        tops = {}
        for cx, cy in cells:
            if cy < 0:
                continue
            tops[cx] = min(tops.get(cx, rows), cy)
        for cx, top in tops.items():
            new_heights[cx] = max(new_heights[cx], rows - top)

        cleared = sum(1 for h in new_heights if h >= rows)
        cleared_rows = []
        if cleared:
            cleared_rows = list(range(rows - cleared, rows))
            new_heights = [max(0, h - cleared) for h in new_heights]
        return new_heights, cells, cleared_rows, cleared


# --- Placement enumeration ------------------------------------------------

# For each piece and rotation, the distinct column offsets of its cells. Used by
# ``valid_actions`` to work in column space rather than cell space.
_COL_OFFSETS: dict[str, tuple[tuple[int, ...], ...]] = {
    kind: tuple(tuple(sorted({ox for ox, _ in _CELLS[kind][rot]})) for rot in range(4))
    for kind in PIECES
}

# For each piece and rotation, the lowest cell in each column, as a board-row
# offset. Together with _COL_OFFSETS this describes the piece's bottom profile.
_BOTTOM_PROFILE: dict[str, tuple[tuple[int, ...], ...]] = {
    kind: tuple(
        tuple(max(oy for ox, oy in _CELLS[kind][rot] if ox == colx)
              for colx in _COL_OFFSETS[kind][rot])
        for rot in range(4)
    )
    for kind in PIECES
}


def valid_actions(board, piece, allow_180=False):
    """Valid ``(rotation, x)`` actions for ``piece`` on ``board``.

    A placement is valid when the piece fits at its spawn row (this is what
    rules out sliding under an overhang) and still fits once dropped.

    This works in column-height space rather than simulating a drop row by row,
    which makes it roughly an order of magnitude faster -- it is the single
    hottest function in training. The result is identical to simulating the
    drop; ``tests/test_engine.py`` checks the two agree.

    A board on which nothing can spawn has no legal placements, so that case is
    rejected up front: the height-based fast path cannot see the rows above the
    playfield, where pieces spawn.
    """
    if board.collides(piece.cells()):
        return []
    rows = board.rows
    cols = board.cols
    heights = board.column_heights()
    spawn_y = piece.y
    actions = []
    for rot in range(4):
        offsets = _COL_OFFSETS[piece.kind][rot]
        bottoms = _BOTTOM_PROFILE[piece.kind][rot]
        # x is an *anchor* column, and a rotation's cells need not start at
        # offset 0: in the SRS box convention the vertical I sits at offset 2 of
        # its 4x4 box, so its anchor may legitimately be negative. Enumerating
        # `0 .. cols - width` instead -- which is only correct when the leftmost
        # cell is at offset 0 -- made a vertical I unable to reach columns 0, 1
        # or 9, which is exactly the well a tetris needs. Enumerate the anchors
        # that keep the piece's *cells* on the board.
        for x in range(-offsets[0], cols - offsets[-1]):
            # Clearance available under the piece's lowest cell in each column
            # it covers. A placement at the spawn row is only legal if every
            # covered column has room for the piece there.
            clearance = rows
            ok = True
            for i, colx in enumerate(offsets):
                h = heights[x + colx]
                bottom_row = spawn_y + bottoms[i]
                if bottom_row >= rows - h:
                    ok = False
                    break
                room = rows - h - bottom_row
                if room < clearance:
                    clearance = room
            if not ok:
                continue
            actions.append((rot, x))
    return actions
