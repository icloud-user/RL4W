"""Pygame front-end for the headless engine: human play, agent playback, menu.

Public API
----------
``run_human(config=None, seed=None)``    play yourself, blocks until quit
``run_agent(policy, config=None, seed=None, fps=30.0)``    watch an AI play
``run_menu(config=None)``    title screen / launcher

``Layout``, ``Renderer``, ``HumanInput`` and ``AgentPlayer`` are internal but
importable on purpose, so the drawing and the per-frame logic can be driven in a
test without a visible window.

``tetrisrl.engine`` (pure stdlib) is imported at module level; ``tetrisrl.policies``
is imported *lazily inside the menu*, because it may pull in torch and a missing
model must not stop the renderer from starting -- that is exactly when you want
to see the screen.  Every timer here is in seconds, advanced by ``dt`` from
``clock.tick()``: DAS/ARR, gravity, the lock delay and the agent animation all
keep their feel if the frame rate changes.
"""
from __future__ import annotations

import random

import pygame

try:                                    # the normal case: imported as a package
    from tetrisrl import controls
    from tetrisrl import recorder as recording
    from tetrisrl.engine import (BUFFER_ROWS, COLS, PIECES, ROWS, VISIBLE_ROWS,
                                 Game, Piece, spawn_anchor, valid_actions)
    from tetrisrl.versus import Battle
except ImportError:                     # `python render.py` from inside the package
    import controls
    import recorder as recording
    from engine import (BUFFER_ROWS, COLS, PIECES, ROWS, VISIBLE_ROWS, Game,
                        Piece, spawn_anchor, valid_actions)
    from versus import Battle

# Palette, identical to vibecode.py so both front-ends look like one project.
# 'G' is the versus garbage row written by Board.add_garbage: a neutral grey, so a
# risen row never reads as one of your own pieces.
COLORS = {'I': (80, 220, 255), 'J': (40, 40, 255), 'L': (255, 160, 40),
          'O': (240, 240, 60), 'S': (80, 255, 80), 'T': (200, 80, 255),
          'Z': (255, 80, 80), 'G': (124, 126, 140)}
BG, GRID_LINE, PANEL_BG = (18, 18, 24), (40, 40, 52), (28, 28, 36)
TEXT, DIM, FRAME = (225, 225, 235), (150, 150, 165), (170, 170, 180)
BLOCK_EDGE = (10, 10, 14)            # 1px dark outline around every block
GAME_OVER_COLOR = (255, 120, 120)

DEFAULT_CELL, PANEL_W, MARGIN = 30, 200, 10
#: The HOLD box gets a panel of its own on the *left* of the board (NEXT and the
#: stats keep the right panel). It only ever shows one tetromino, so it needs one
#: preview wide plus padding -- about a third of a full side panel, which is why
#: this is its own constant rather than a share of ``PANEL_W``. Configurable as
#: ``game.hold_panel_width``; the default fits a 4-wide I preview drawn with the
#: renderer's ``preview_cell``.
HOLD_PANEL_W = 96
NEXT_PREVIEW = 5                     # pieces shown in the NEXT box

# Timings in seconds, grouped so the feel of the game is one block of tunables.
#
# These are only fallbacks now: every one of them is owned by ``controls.py``
# (which speaks the milliseconds and SDF TETR.IO players expect, and which the
# in-game settings screen edits). They are kept because callers that predate that
# module still import them, and because they document what the defaults were.
DAS_DELAY, ARR = 0.133, 0.0           # hold, then horizontal auto-repeat rate
SOFT_DROP_INTERVAL = 0.05             # held down-arrow repeat when SDF is a number
#: Most rows soft drop may take in one frame. A 20-row board can never need more
#: than 22, so this is a spin guard rather than a speed limit -- the speed comes
#: from SDF. It is what lets SDF above ~51 mean something: the old code took one
#: row per frame, so 200x was silently identical to 60x.
MAX_SOFT_ROWS_PER_FRAME = 24
LOCK_DELAY, LOCK_RESET_LIMIT = 0.5, 15
AGENT_MOVE_TIME, AGENT_DROP_TIME = 0.08, 0.07
AGENT_SETTLE_TIME = 0.05
SPEED_MIN, SPEED_MAX, SPEED_STEP = 0.25, 8.0, 1.25
MENU_FPS = 30.0

#: The actions the on-screen hint lists, in the order a player reads them.
HINT_ACTIONS = ('move_left', 'move_right', 'soft_drop', 'hard_drop', 'rotate_cw',
                'rotate_ccw', 'rotate_180', 'hold')
HUMAN_KEYS = ('left/right move   down soft drop   SPACE hard drop\n'
              'Z/X rotate   C hold   P pause   R restart   N new seed   ESC quit')

# --- battle mode ----------------------------------------------------------
#: AgentPlayer's three phases take this long per piece at speed 1.0, so the
#: animation speed that produces a target pieces-per-second rate is just this
#: multiplied by the rate -- that is how a difficulty preset's ``pps`` is enforced.
AGENT_PIECE_TIME = AGENT_MOVE_TIME + AGENT_DROP_TIME + AGENT_SETTLE_TIME
#: The largest cell size that still fits two boards, two hold boxes and two side
#: panels side by side.
BATTLE_CELL_MAX = 26
#: Width budget for the two-board screen. The battle cell is additionally clamped
#: so one half cannot exceed half of this once the panels and margins are paid for;
#: at the shipped defaults the cell cap above is what actually applies, and the
#: window comes out 1,192 px wide.
BATTLE_MAX_WIDTH = 1280
#: Battle-only clamp on the side panel. ``config.yaml``'s ``side_panel_width`` is
#: honoured up to here; past it two boards plus panels stop fitting across a
#: screen, and a battle that opens off-screen is worse than a narrower panel.
BATTLE_PANEL_MAX = 260
#: Beat between rounds, so a K.O. is legible before the boards reset.
ROUND_PAUSE = 1.8
#: Garbage meter: ready lines are bright, still-travelling ones dim (TETR.IO's
#: yellow-to-red meter, split at the travel delay).
GARBAGE_READY, GARBAGE_PENDING = (235, 90, 90), (150, 140, 70)


def _binding_hint(settings, extra=''):
    """One or two lines of hint text built from the *live* bindings.

    The hardcoded ``HUMAN_KEYS`` string is kept above for callers that have no
    settings, but a player who rebinds hard drop to ``F`` should not be told to
    press SPACE, so every on-screen hint goes through here instead.
    """
    if settings is None:
        return HUMAN_KEYS
    parts = []
    for action in HINT_ACTIONS:
        names = settings.binding_names(action)
        if names:
            parts.append(f'{" ".join(names)} {controls.ACTION_LABELS[action].lower()}')
    line_one = '   '.join(parts[0:4])
    line_two = '   '.join(parts[4:])
    loop = []
    for action in controls.LOOP_ACTIONS:
        names = settings.binding_names(action)
        if names:
            loop.append(f'{names[0]} {controls.ACTION_LABELS[action].lower()}')
    text = '\n'.join(part for part in (line_one, line_two) if part)
    if loop:
        text = f'{text}\n' + '   '.join(loop)
    if extra:
        text = f'{text}\n{extra}'
    return text


def settings_summary(settings):
    """One line describing the settings in force, for the startup printout."""
    if settings is None:
        return 'input: defaults'
    soft = 'instant' if settings.soft_instant else f'{settings.sdf:g}x'
    dcd = f'{settings.dcd_ms:.0f}ms' if settings.dcd_ms > 0 else 'off'
    return (f'input: DAS {settings.das_ms:.0f}ms  ARR {settings.arr_ms:.0f}ms'
            f'{" (instant)" if settings.arr <= 0 else ""}  DCD {dcd}'
            f'  soft drop {soft}'
            f'  lock {settings.lock_delay_ms:.0f}ms'
            f'  ({controls.settings_path()})')


def _agent_hint(settings):
    """The watcher's hint line, with the rebindable parts read from settings."""
    def keys(action, fallback):
        if settings is None:
            return fallback
        return '/'.join(settings.binding_names(action)) or fallback

    return (f'{keys("pause", "SPACE")} pause   up/down speed   S step (paused)\n'
            f'{keys("restart", "R")} restart   {keys("new_seed", "N")} new seed   '
            f'{keys("quit", "ESC")} quit')


def _cfg(config, path, default, cast=None):
    """Read a nested config value, falling back to ``default`` for anything odd.

    ``config`` is caller-supplied and may be ``None``, not a dict, or hold a
    value of the wrong type, and a bad config must never be the reason the window
    fails to open.
    """
    node = config
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    if cast is None:
        return node
    try:
        return cast(node)
    except (TypeError, ValueError):
        return default


def _shade(color, factor):
    """Scale a colour's channels, clamped: the 3D bevel's light/dark shades."""
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def _make_font(size, bold=False):
    """A SysFont preferring consolas; ``None`` if no font could be made at all.

    ``None`` turns every text draw into a no-op, which keeps the board watchable
    instead of crashing.
    """
    try:
        pygame.font.init()
        return pygame.font.SysFont('consolas', int(size), bold=bold)
    except Exception:
        pass
    try:
        return pygame.font.Font(None, int(size * 1.2))
    except Exception:
        return None


def gravity_interval(level):
    """Seconds per gravity step, floored so high levels stay controllable.

    The engine exposes no gravity (only single-row ``soft_drop``), so human play
    drives it from this curve.
    """
    return max(0.06, 0.85 * (0.82 ** (max(1, int(level)) - 1)))


class HoldRepeat:
    """Timed key auto-repeat: fire on press, then after ``delay`` seconds and
    every ``rate`` seconds after that.

    DAS/ARR, held soft drop and the agent's speed keys are the same shape of
    problem, so they share this.
    """

    __slots__ = ('delay', 'rate', 'held', 'timer', 'cap')

    def __init__(self, delay, rate, cap=8):
        self.delay = max(0.0, delay)
        self.rate = max(0.001, rate)     # never 0: that would loop forever
        self.held, self.timer = False, 0.0
        #: Most repeats one frame may owe. The default suits DAS/ARR and the
        #: agent's speed keys; soft drop raises it, because a high SDF is *meant*
        #: to move several rows per frame and 8 would be a second speed ceiling.
        self.cap = max(1, int(cap))

    def press(self):
        """Mark the key down; True if this is a fresh press, not a repeat."""
        if self.held:
            return False
        self.held, self.timer = True, self.delay
        return True

    def release(self):
        self.held, self.timer = False, 0.0

    def due(self, dt):
        """Consume ``dt``; return how many repeats are owed this frame."""
        if not self.held:
            return 0
        self.timer -= dt
        count = 0
        while self.timer <= 0.0 and count < self.cap:   # capped: a long dt must not spin
            count += 1
            self.timer += self.rate
        return count


class Layout:
    """Pixel geometry for the playfield, side panel and status strip.

    Everything derives from ``cell`` and ``panel_w``, and ``cell_rect`` is the
    single place board coordinates become screen pixels -- including the y offset
    that hides the spawn buffer -- so all the drawing code shares one convention:
    board row ``r`` is drawn at visible row ``r - BUFFER_ROWS``, and rows inside
    the hidden buffer return ``None`` so a spawning piece can never paint over the
    panel or off-screen.
    """

    def __init__(self, cell=DEFAULT_CELL, panel_w=PANEL_W, margin=MARGIN,
                 hold_w=HOLD_PANEL_W):
        self.cell, self.panel_w, self.margin = int(cell), int(panel_w), int(margin)
        self.hold_w = int(hold_w)
        self.board_w, self.board_h = COLS * self.cell, VISIBLE_ROWS * self.cell
        # Left to right: margin, the HOLD box, one gap (the margin's width -- the
        # only spacing constant this layout has), the board, a gap, the side panel,
        # and a margin. The board no longer starts at the margin, so board_x is
        # derived rather than assigned.
        self.hold_x = self.margin
        self.board_x = self.hold_x + self.hold_w + self.margin
        self.board_y = self.margin
        self.panel_x = self.board_x + self.board_w + self.margin
        # The hold box is a box, not a column: a label plus one preview. Its height
        # is derived from what goes in it, mirroring Renderer's font and preview
        # floors -- pads, a label, a two-cell-tall preview, and a little slack.
        # Deriving it keeps the box tight at the default cell size and still big
        # enough at the small end, where those two sizes stop shrinking.
        preview = max(8, self.cell // 2)
        label = max(10, self.cell // 2 - 3)
        self.hold_h = 2 * max(4, self.cell // 6) + label + 2 * preview + 4
        self.status_h = int(round(1.4 * self.cell))         # room for two hints
        self.width = self.panel_x + self.panel_w + self.margin
        self.height = self.board_h + 2 * self.margin + self.status_h
        self.status_y = self.board_y + self.board_h + 4

    @classmethod
    def from_config(cls, config=None):
        """Layout from an optional config dict, clamped to sane values.

        ``game.cell_size``, ``game.side_panel_width`` and ``game.hold_panel_width``
        are read; the clamps matter because a zero cell size would otherwise make
        an unopenable window. Board dimensions are not configurable -- the renderer
        is written against the engine's fixed 10x20 visible field.
        """
        cell = _cfg(config, ('game', 'cell_size'), DEFAULT_CELL, int)
        panel = _cfg(config, ('game', 'side_panel_width'), PANEL_W, int)
        hold = _cfg(config, ('game', 'hold_panel_width'), HOLD_PANEL_W, int)
        return cls(cell=max(10, min(80, cell)), panel_w=max(120, min(600, panel)),
                   hold_w=max(56, min(200, hold)))

    @property
    def board_rect(self):
        return pygame.Rect(self.board_x, self.board_y, self.board_w, self.board_h)

    @property
    def panel_rect(self):
        return pygame.Rect(self.panel_x, 0, self.panel_w, self.height)

    @property
    def hold_rect(self):
        """The HOLD box, top-aligned with the board on its left."""
        return pygame.Rect(self.hold_x, self.board_y, self.hold_w, self.hold_h)

    def cell_rect(self, col, row):
        """Rect for board cell ``(col, row)``, or ``None`` if it is clipped."""
        col, visible = int(col), int(row) - BUFFER_ROWS
        if not (0 <= visible < VISIBLE_ROWS and 0 <= col < COLS):
            return None
        return pygame.Rect(self.board_x + col * self.cell,
                           self.board_y + visible * self.cell,
                           self.cell, self.cell)


def draw_block(surface, rect, color):
    """One cell with a subtle 3D bevel.

    A lighter band on the top/left and a darker one on the bottom/right, both
    derived from the base colour so every piece keeps its hue, plus a 1px dark
    border. The band scales with the cell so it still reads when drawn small.
    """
    band = max(2, rect.width // 6)
    light, dark = _shade(color, 1.35), _shade(color, 0.55)
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, light, (rect.x, rect.y, rect.w, band))
    pygame.draw.rect(surface, light, (rect.x, rect.y, band, rect.h))
    pygame.draw.rect(surface, dark, (rect.x, rect.bottom - band, rect.w, band))
    pygame.draw.rect(surface, dark, (rect.right - band, rect.y, band, rect.h))
    pygame.draw.rect(surface, BLOCK_EDGE, rect, 1)


class Renderer:
    """Draws a ``Game`` into a surface: playfield, side panel, overlays.

    Holds the surface, layout and fonts, so the run loops only pass the game plus
    whatever is special about this frame: a paused banner, a frozen board during
    the line-clear flash, or the agent's in-flight piece.
    """

    def __init__(self, surface, layout=None, config=None):
        self.surface = surface
        self.layout = layout or Layout.from_config(config)
        cell = self.layout.cell
        self.font = _make_font(max(12, cell // 2))          # body text
        self.small = _make_font(max(10, cell // 2 - 3))     # hints and labels
        self.big = _make_font(max(18, cell), bold=True)     # banner / menu title
        self.preview_cell = max(8, cell // 2)               # NEXT/HOLD block size
        # Translucent overlays, sized once and reused every frame.
        self._dim = pygame.Surface((self.layout.board_w, self.layout.board_h),
                                   pygame.SRCALPHA)
        self._dim.fill((0, 0, 0, 165))

    def label(self, text, pos, color=TEXT, font=None, center=False):
        """Blit one line of text; returns the width drawn (0 if nothing drawn)."""
        active = font or self.font
        if active is None or text is None:
            return 0
        rendered = active.render(str(text), True, color)
        rect = (rendered.get_rect(center=pos) if center
                else rendered.get_rect(topleft=pos))
        self.surface.blit(rendered, rect)
        return rect.width

    @staticmethod
    def _height(font, fallback=14):
        return fallback if font is None else font.get_height()

    def draw(self, game, current=None, ghost_cells=None, show_current=True,
             grid=None, highlight=(), hud=(), status=None, paused=False,
             message=None, fill_bg=True):
        """Draw one complete frame.

        ``current``/``ghost_cells`` are the piece and ghost to draw, defaulting to
        ``game.current`` and the engine's own ghost; the agent passes its in-flight
        animation here, which is how a piece can appear at a fractional position
        the engine never holds. ``show_current=False`` draws the field only, and
        ``grid``/``highlight`` substitute a board snapshot plus rows to flash white
        (the rows that just cleared). ``hud`` appends extra ``(label, value)``
        stats, ``status`` is the hint text under the board (``\\n`` splits lines),
        and ``message`` overrides the banner, which otherwise reads PAUSED or
        GAME OVER.

        ``fill_bg=False`` leaves the background alone, which is what lets two
        renderers share one surface in battle mode: everything else this method
        draws is confined to its own layout's board, panel and status strip.
        """
        if fill_bg:
            self.surface.fill(BG)
        self._draw_board(game, current, ghost_cells, show_current, grid, highlight)
        self._draw_hold(game)
        self._draw_panel(game, hud)
        for offset, line in enumerate(str(status or '').split('\n')):
            self.label(line, (self.layout.board_x, self.layout.status_y + offset * 16),
                       DIM, self.small)
        pygame.draw.rect(self.surface, FRAME, self.layout.board_rect, 2)
        banner = message
        if banner is None:
            banner = 'GAME OVER' if game.game_over else ('PAUSED' if paused else None)
        if banner:
            board = self.layout.board_rect
            self.surface.blit(self._dim, board.topleft)
            color = GAME_OVER_COLOR if banner.upper().startswith('GAME') else TEXT
            self.label(banner, board.center, color, self.big, center=True)

    def _draw_board(self, game, current, ghost_cells, show_current, grid, highlight):
        board, cell = self.layout.board_rect, self.layout.cell
        for col in range(COLS + 1):                     # faint grid, drawn first
            x = board.x + col * cell
            pygame.draw.line(self.surface, GRID_LINE, (x, board.y), (x, board.bottom - 1))
        for row in range(VISIBLE_ROWS + 1):
            y = board.y + row * cell
            pygame.draw.line(self.surface, GRID_LINE, (board.x, y), (board.right - 1, y))

        rows = grid if grid is not None else game.board.grid
        for board_row in range(BUFFER_ROWS, ROWS):
            line = rows[board_row]
            for col in range(min(COLS, len(line))):
                rect = self.layout.cell_rect(col, board_row) if line[col] else None
                if rect is not None:
                    draw_block(self.surface, rect, COLORS.get(line[col], FRAME))

        if not show_current or game.game_over:
            return
        piece = game.current if current is None else current
        if piece is None:
            return
        color = COLORS.get(piece.kind, FRAME)
        ghost = self._ghost_for(game, piece) if ghost_cells is None else ghost_cells
        for col, row in ghost:                          # outline only, never filled
            rect = self.layout.cell_rect(col, row)
            if rect is not None:
                pygame.draw.rect(self.surface, color, rect, 2)
        for col, row in piece.cells():
            rect = self.layout.cell_rect(round(col), round(row))
            if rect is not None:
                draw_block(self.surface, rect, color)

    @staticmethod
    def _ghost_for(game, piece):
        """Landing cells for ``piece`` as the engine would drop it.

        The piece may be an animation stand-in with fractional coordinates while
        the engine indexes its grid with integers, so it is rounded into a
        throwaway ``Piece`` first. If that rounded stand-in overlaps the stack
        (possible mid-slide, where the interpolated state is not yet a legal
        placement) there is no meaningful ghost, so none is drawn.
        """
        probe = Piece(piece.kind, int(round(piece.x)), int(round(piece.y)),
                      int(piece.rotation) % 4)
        if game.board.collides(probe.cells()):
            return ()
        return probe.cells(y=game.ghost_y(probe))

    def _draw_hold(self, game):
        """The HOLD box on the left of the board: a label and one preview.

        Clipped to its own rect like the side panel, so a preview can never spill
        onto the playfield even at the smallest cell size.
        """
        box = self.layout.hold_rect
        pygame.draw.rect(self.surface, PANEL_BG, box)
        pygame.draw.rect(self.surface, FRAME, box, 1)
        previous = self.surface.get_clip()
        self.surface.set_clip(box)
        try:
            pad = max(4, self.layout.cell // 6)
            self.label('HOLD', (box.x + pad, box.y + pad), DIM, self.small)
            top = box.y + pad + self._height(self.small, 12) + 2
            preview = pygame.Rect(box.x + pad, top, box.w - 2 * pad,
                                  max(self.preview_cell, box.bottom - pad - top))
            hold = game.hold_piece
            self._preview(preview, None if hold is None else hold.kind,
                          dim=game.hold_used)
        finally:
            self.surface.set_clip(previous)

    def _draw_panel(self, game, hud=()):
        """NEXT and the stats, laid out top to bottom on a running cursor.

        HOLD used to be the second section here; it now has its own box on the
        board's left (``_draw_hold``), so this panel is NEXT followed by the
        numbers, unchanged otherwise.
        """
        panel = self.layout.panel_rect
        pygame.draw.rect(self.surface, PANEL_BG, panel)
        pygame.draw.rect(self.surface, FRAME, panel, 1)
        previous = self.surface.get_clip()
        self.surface.set_clip(panel)                    # never spill into the board
        try:
            pad = max(6, self.layout.cell // 5)
            box_w, box_h = panel.w - 2 * pad, 3 * self.preview_cell
            gap = max(4, self.layout.cell // 6)
            x, y = panel.x + pad, panel.y + pad
            next_kinds = [p.kind for p in list(game.next_queue)[:NEXT_PREVIEW]]
            for title, kinds in (('NEXT', next_kinds),):
                self.label(title, (x, y), DIM, self.small)
                y += self._height(self.small, 12) + 4
                for kind in kinds:
                    self._preview(pygame.Rect(x, y, box_w, box_h), kind)
                    y += box_h + gap
                y += gap                                # gap between sections

            stats = [('SCORE', game.score), ('LINES', game.lines),
                     ('LEVEL', game.level), ('PIECES', game.pieces_placed)] + list(hud)
            for name, value in stats:
                y += gap // 2
                self.label(name, (x, y), DIM, self.small)
                text = str(value)                       # right-aligned value
                width = self.font.size(text)[0] if self.font else 0
                self.label(text, (x + box_w - width, y), TEXT)
                y += self._height(self.font) - gap // 2
        finally:
            self.surface.set_clip(previous)

    def _preview(self, box, kind, dim=False):
        """A tetromino drawn small and centred inside ``box``.

        Uses the engine's own rotation-0 cell layout (a throwaway ``Piece`` at its
        spawn anchor) normalised to its bounding box, so each piece is centred
        rather than drawn at its spawn offset.
        """
        pygame.draw.rect(self.surface, BG, box)
        pygame.draw.rect(self.surface, GRID_LINE, box, 1)
        if kind is None:
            return
        cells = Piece(kind).cells()
        cols, rows = [c for c, _ in cells], [r for _, r in cells]
        min_col, min_row = min(cols), min(rows)
        size = self.preview_cell
        x = box.x + (box.w - (max(cols) - min_col + 1) * size) // 2 - min_col * size
        y = box.y + (box.h - (max(rows) - min_row + 1) * size) // 2 - min_row * size
        color = _shade(COLORS.get(kind, FRAME), 0.45) if dim else COLORS.get(kind, FRAME)
        for col, row in cells:
            draw_block(self.surface, pygame.Rect(x + col * size, y + row * size,
                                                 size, size), color)

    def draw_menu(self, title, subtitle, items, selected, footer, message=None):
        """Title screen / submenu: a heading, a list of choices and a footer."""
        self.surface.fill(BG)
        board, panel = self.layout.board_rect, self.layout.panel_rect
        pygame.draw.rect(self.surface, PANEL_BG, board)
        pygame.draw.rect(self.surface, PANEL_BG, self.layout.hold_rect)
        pygame.draw.rect(self.surface, PANEL_BG, panel)
        pygame.draw.rect(self.surface, FRAME, panel, 1)
        self.label(title, (board.x + 16, board.y + 16), TEXT, self.big)

        # Decorative strip of all seven tetrominoes, drawn with the same preview
        # routine the side panel uses.
        size = self.preview_cell
        top = board.y + 16 + self._height(self.big, 28) + 10
        for index, kind in enumerate(PIECES):
            self._preview(pygame.Rect(board.x + 16 + index * (4 * size + 6), top,
                                      4 * size, 2 * size), kind)

        y = top + 2 * size + 20
        self.label(subtitle, (board.x + 16, y), DIM)
        y += self._height(self.font) + 12
        for index, item in enumerate(items):
            self.label(f'{" >" if index == selected else "  "} {index + 1}. {item}',
                       (board.x + 20, y), TEXT if index == selected else DIM)
            y += self._height(self.font) + 8
        if message:
            self.label(message, (board.x + 16, board.bottom - 76), GAME_OVER_COLOR,
                       self.small)
        for offset, line in enumerate(str(footer).split('\n')):
            self.label(line, (board.x + 16, board.bottom - 44 + offset * 16), DIM,
                       self.small)


class HumanInput:
    """DAS/ARR, soft drop, gravity and lock delay for human play.

    Key *state* lives here rather than in the run loop's event pump, which makes
    the timing testable: drive ``key_down``/``key_up`` then ``update(dt, game)``
    without a window.

    Every binding and every timing comes from an :class:`controls.InputSettings`,
    so a rebind in the settings screen takes effect on the next piece without the
    renderer knowing which key does what.
    """

    def __init__(self, settings=None, recorder=None):
        self.settings = settings if settings is not None else controls.load()
        # Optional session recorder. ``None`` (the default) costs one attribute
        # lookup per event and nothing else -- recording is off unless asked for.
        self.recorder = recorder
        self.reset_repeaters()
        self.gravity_timer = self.lock_timer = 0.0
        self.lock_resets = 0

    def reset_repeaters(self):
        """(Re)build the auto-repeat timers from the current settings."""
        self.left = HoldRepeat(self.settings.das, self.settings.arr)
        self.right = HoldRepeat(self.settings.das, self.settings.arr)
        # Fires immediately (delay 0) and may take a whole board's worth of rows
        # in one frame, so a high SDF is not capped at the frame rate.
        self.soft = HoldRepeat(0.0, SOFT_DROP_INTERVAL,
                               cap=MAX_SOFT_ROWS_PER_FRAME)
        self.direction = 0                # -1/0/+1: the most recent press wins
        #: DAS Cut Delay countdown, in seconds. Armed by a rotation and by a new
        #: piece; while it is positive ``update`` leaves the horizontal repeat
        #: alone. Zero unless the player sets ``dcd_ms``.
        self.dcd_timer = 0.0

    # -- key events --------------------------------------------------------

    def key_down(self, key, game):
        """Handle one gameplay key press; True if this class consumed the key."""
        action = self.settings.action_for(key)
        if action is None:
            return False
        if action == 'move_left' or action == 'move_right':
            step = -1 if action == 'move_left' else 1
            active, other = ((self.left, self.right) if step < 0
                             else (self.right, self.left))
            other.release()                         # the newest direction wins
            if active.press():
                moved = game.move(step)
                if moved:
                    self._touched()
                if self.recorder is not None:
                    self.recorder.move(step, auto=False, ok=moved)
            self.direction = step
        elif action == 'soft_drop':
            self.soft.press()
            if self.settings.soft_instant:
                self._soft_to_floor(game)
        elif action == 'rotate_cw':
            self._rotate(game, 1)
        elif action == 'rotate_ccw':
            self._rotate(game, -1)
        elif action == 'rotate_180':
            self._rotate(game, 2)                   # engine has the 180 kicks
        elif action == 'hard_drop':
            piece, lines_before = game.current, game.lines
            game.hard_drop()                        # locks the piece
            self._record_lock(game, piece, game.lines - lines_before)
            self.reset()
        elif action == 'hold':
            if game.hold():
                self.reset()
        else:
            return False                            # a loop action: not ours
        return True

    def key_up(self, key, game=None):
        """Handle one gameplay key release; True if this class consumed it."""
        action = self.settings.action_for(key)
        if action == 'move_left':
            self.left.release()
            self._hand_over(-1, self.right)
        elif action == 'move_right':
            self.right.release()
            self._hand_over(1, self.left)
        elif action == 'soft_drop':
            self.soft.release()
        else:
            return False
        return True

    # -- actions -----------------------------------------------------------

    def _rotate(self, game, direction):
        """Rotate through the engine (it resolves wall kicks) and refresh the lock."""
        # Work out what the attempt will do *before* it happens, so a refusal can
        # be recorded with the offsets that were tried and why none of them fitted.
        # This is the record that turns "I can't rotate near a wall" into a
        # diagnosis, so it is taken on every attempt, successful or not.
        report = None
        if self.recorder is not None:
            report = recording.rotation_report(game, direction)
        if game.rotate(direction):
            self._touched()
            # Remember that this frame already produced a rotation. A wall kick
            # can lift the piece (that is exactly how a T-spin is built), and
            # soft drop running later in the *same* frame would push it straight
            # back down, undoing the kick before the player can see it or hard
            # drop into it. Real clients apply gravity before input for the same
            # reason; here the frame is simply skipped.
            self.rotated_this_frame = True
            # DCD starts at the same instant: horizontal auto-repeat pauses for
            # ``dcd_ms`` so a spin does not also fling the piece sideways. The
            # frame-skip above is a different, shorter thing (one frame of no soft
            # drop, to keep a wall kick visible); this one outlives it.
            self.dcd_timer = self.settings.dcd
        if report is not None:
            self.recorder.rotate(game, direction, report)

    def _record_lock(self, game, piece, lines):
        """Record a piece coming to rest. ``piece`` is the piece as it locked.

        Called after the lock, because the cleared count is only known then, but
        handed the piece from before it, because by now the next one has spawned.
        """
        if self.recorder is None or piece is None:
            return
        heights = game.board.column_heights()
        self.recorder.lock(piece, lines, game.combo, game.back_to_back,
                           bool(getattr(game, 'hold_used', False)), self.settings,
                           top=max(heights) if heights else 0,
                           holes=recording.count_holes(game),
                           deps=recording.count_i_dependencies(game))

    def _soft_to_floor(self, game):
        """SDF 0: snap to the landing row *without* locking.

        TETR.IO's unlimited soft drop drops the piece to where it would land and
        leaves it there, so the player can still slide or spin it during the lock
        delay. Locking here instead would take that away and make SDF 0 a hard
        drop.
        """
        if game.current is None:
            return False
        target = game.ghost_y()
        moved = False
        while game.current.y < target and game.soft_drop():
            moved = True
        if moved:
            self._touched()
        return moved

    def _hand_over(self, direction, other):
        """Give control to the opposite key if still held, with DAS recharged."""
        if other.held:
            self.direction = direction
            other.held, other.timer = True, other.delay
        else:
            self.direction = 0

    def reset(self):
        """Forget per-piece timing (a new piece just spawned).

        Also re-arms DCD. Every caller here is a new piece arriving -- a spawn, a
        hard drop (which locks and spawns) and a hold (which swaps one in) -- and a
        fresh piece is exactly when a held direction would otherwise carry the
        piece off its spawn column before the player can aim it.
        """
        self.gravity_timer = self.lock_timer = 0.0
        self.lock_resets = 0
        self.dcd_timer = self.settings.dcd

    def release_all(self):
        """Drop every held key: a key released while input was frozen (pause,
        restart) would otherwise still look held and the piece would keep moving."""
        self.left.release()
        self.right.release()
        self.soft.release()
        self.direction = 0

    def _touched(self):
        """A successful move/rotate restarts the lock timer, up to the cap.

        The cap is the guideline's move-reset rule: without it a piece resting on
        the stack could be spun forever and never lock.
        """
        if self.lock_timer > 0.0 and self.lock_resets < self.settings.lock_reset_limit:
            self.lock_timer = 0.0
            self.lock_resets += 1

    # -- per-frame ---------------------------------------------------------

    def _slide_repeat(self, dt, game):
        """Horizontal auto-repeat, including ARR 0.

        ARR 0 means "arrive instantly", which ``HoldRepeat`` cannot express -- it
        floors ``rate`` at a millisecond so that its own loop terminates. So the
        instant case is handled here: wait out DAS, then slide until the wall.

        Every step is offered to the recorder when one is attached: the gap between
        a press and the first ``auto`` step is the only way to observe DAS from a
        recording, and the spacing after that is ARR.
        """
        repeater = self.left if self.direction < 0 else self.right
        if self.settings.arr > 0:
            for _ in range(repeater.due(dt)):
                moved = game.move(self.direction)
                if moved:
                    self._touched()
                if self.recorder is not None:
                    self.recorder.move(self.direction, auto=True, ok=moved)
            return
        if not repeater.held:
            return
        repeater.timer -= dt
        if repeater.timer > 0.0:
            return
        repeater.timer = 0.0                      # stay expired while held
        for _ in range(10):                       # a 10-wide board: enough to reach
            moved = game.move(self.direction)
            if self.recorder is not None:
                self.recorder.move(self.direction, auto=True, ok=moved)
            if not moved:
                break
            self._touched()

    def update(self, dt, game):
        """Advance auto-repeat, soft drop, gravity and lock delay by ``dt``."""
        if game.game_over or game.current is None:
            return False
        # A rotation happened during this frame's events: hold the piece still so
        # the kick survives, then clear the flag for the next frame.
        rotated = getattr(self, 'rotated_this_frame', False)
        self.rotated_this_frame = False
        # DCD: freeze horizontal auto-repeat for its window after a rotation or a
        # new piece. Only the *repeat* waits -- and because the repeaters are only
        # advanced inside ``_slide_repeat``, skipping it freezes the DAS charge
        # rather than resetting it, so a key that was most of the way to firing
        # keeps that progress and fires as soon as the window closes. Soft drop,
        # gravity and the lock timer below all keep running, and a fresh press
        # still moves a cell in ``key_down``.
        if self.dcd_timer > 0.0:
            self.dcd_timer = max(0.0, self.dcd_timer - dt)
        if self.direction and self.dcd_timer <= 0.0:
            self._slide_repeat(dt, game)

        # Soft drop replaces gravity rather than adding to it, so holding down
        # never double-steps. Its rate follows the level, because SDF is a
        # multiple of gravity and gravity changes as the level rises.
        moved = False
        if self.soft.held and not rotated:
            if self.settings.soft_instant:
                moved = self._soft_to_floor(game)
            else:
                interval = self.settings.soft_interval(game.level)
                # The real interval, not a frame-floored one: ``due`` accumulates
                # the fraction and hands back every row this frame has earned, so
                # SDF 200 moves several rows at 60 fps instead of one.
                self.soft.rate = interval if interval else SOFT_DROP_INTERVAL
                for _ in range(self.soft.due(dt)):
                    if not game.soft_drop():
                        break                       # resting: stop asking
                    moved = True
        else:
            self.gravity_timer -= dt
            if self.gravity_timer <= 0.0:
                self.gravity_timer = gravity_interval(game.level)
                if game.soft_drop():
                    moved = True
        if moved:
            self.lock_timer, self.lock_resets = 0.0, 0
            return False
        if game.can_move(0, 1):                         # still falling
            self.lock_timer = 0.0
            return False
        self.lock_timer += dt                           # resting: count to the lock
        if self.lock_timer >= self.settings.lock_delay:
            piece, lines_before = game.current, game.lines
            game.lock()
            self._record_lock(game, piece, game.lines - lines_before)
            self.reset()
            return True
        return False


class AgentPlayer:
    """Plays one piece at a time so the placement is visible.

    Per piece: call the policy exactly once, slide/rotate along the spawn row,
    fall to the landing row, hard drop, then hold for a beat. Every duration is
    divided by ``speed``, so the speed keys scale the animation uniformly. The
    engine piece is only mutated when a drop *finishes*, so an interrupted
    animation can never leave the board half-applied; while a piece is in flight
    ``display_piece()`` returns a stand-in with fractional coordinates which the
    renderer draws in place of ``game.current``.
    """

    def __init__(self, game, policy, speed=1.0):
        self.game = game
        self.policy = policy
        self.speed = speed
        self.phase, self.timer = 'plan', 0.0
        self.plan = None            # (rotation, x) chosen for the piece in flight
        self.anim = None            # animation parameters for that piece
        self.locked = 0

    @staticmethod
    def _coerce(choice, actions):
        """The policy's pick as plain ints, else the first legal action.

        Accepts a bare ``(rotation, x)`` or a ``(use_hold, rotation, x)`` move --
        the hold flag must be ignored here, not read as the rotation. An earlier
        version unpacked ``choice[0], choice[1]`` unconditionally, so a 3-tuple
        yielded ``rot=False``, never matched a legal action, and fell back to
        ``actions[0]`` on every piece: it played the first legal move every time
        and died almost immediately. The contract guarantees a legal pick; this
        only stops a misbehaving policy from taking the window down.
        """
        try:
            rot, x = int(choice[-2]), int(choice[-1])
        except (TypeError, ValueError, IndexError, KeyError):
            return actions[0]
        return (rot, x) if (rot, x) in actions else actions[0]

    def _begin_piece(self):
        """Ask the policy once and set up the animation. False if it cannot."""
        game, piece = self.game, self.game.current
        if piece is None or game.game_over:
            return False
        actions = valid_actions(game.board, piece)
        if not actions:
            return False

        # A policy may ask to hold first: (use_hold, rotation, x). Hold swaps in
        # a different piece, so it must be applied before the target placement is
        # validated -- the requested rotation/column belongs to the *new* piece.
        move = self.policy(game, actions)
        if len(move) == 3 and move[0]:
            if game.hold():
                piece = game.current
                actions = valid_actions(game.board, piece)
                if not actions:
                    return False
            rot, x = self._coerce((move[1], move[2]), actions)
        else:
            rot, x = self._coerce(move, actions)

        # Mirror the engine's placement contract: rotation and column apply at the
        # spawn row and the piece hard drops from there, so what is drawn is
        # exactly what is executed.
        spawn_y = spawn_anchor(piece.kind)[1]
        target = Piece(piece.kind, x, spawn_y, rot)
        landing = game.ghost_y(target)
        steps = (rot - piece.rotation) % 4
        if steps == 3:
            steps = -1                     # three cw == one ccw: spin the short way

        # Pieces spawn above the board (the engine's spawn row is negative), so a
        # slide along the spawn row would happen entirely off-screen. The slide is
        # drawn at the first row where the piece is fully inside the field instead.
        # That row is on the same vertical path the hard drop takes, which the
        # engine has already verified is clear, and it never goes below the
        # landing row -- so this only changes what is drawn, never where the piece
        # ends up.
        top_offset = min(row - spawn_y for _, row in target.cells())
        from_y = min(max(piece.y, BUFFER_ROWS - top_offset), landing)

        self.plan = (rot, x)
        self.anim = {'kind': piece.kind, 'from_x': piece.x, 'to_x': x,
                     'from_y': from_y, 'landing': landing,
                     'from_rot': piece.rotation, 'rot_steps': steps,
                     'ghost': target.cells(y=landing)}
        self.phase, self.timer = 'move', 0.0
        return True

    def _duration(self):
        """Seconds the current phase lasts, scaled by the speed multiplier."""
        return {'move': AGENT_MOVE_TIME, 'drop': AGENT_DROP_TIME,
                'settle': AGENT_SETTLE_TIME}.get(self.phase, 0.0) / max(self.speed, 0.01)

    def _progress(self):
        duration = self._duration()
        return 0.0 if duration <= 0 else min(1.0, max(0.0, self.timer / duration))

    def display_piece(self):
        """The piece as it should be drawn this frame.

        Rotation snaps to whole steps along the shortest direction (the engine has
        only four states) while x and y interpolate, which reads as the piece
        spinning as it slides across.
        """
        if self.anim is None:
            return self.game.current
        anim, t = self.anim, self._progress()
        if self.phase == 'drop':
            y = anim['from_y'] + (anim['landing'] - anim['from_y']) * t
            x, rot = anim['to_x'], (anim['from_rot'] + anim['rot_steps']) % 4
        else:
            x = anim['from_x'] + (anim['to_x'] - anim['from_x']) * t
            y, steps = anim['from_y'], anim['rot_steps']
            done = min(int(t * abs(steps)), abs(steps)) if steps else 0
            rot = (anim['from_rot'] + (1 if steps > 0 else -1) * done) % 4
        return Piece(anim['kind'], x, y, rot)

    def ghost_cells(self):
        """Landing cells of the piece being animated, or ``None``.

        Taken from the *plan*, not the in-flight position, so the ghost stays put
        while the piece slides instead of jittering along with it.
        """
        if self.anim is None or self.game.game_over:
            return None
        return self.anim['ghost']

    def _finish_drop(self):
        """Apply the plan to the engine, hard drop, and start the pause."""
        piece = self.game.current
        piece.rotation, piece.x = self.plan
        piece.y = spawn_anchor(piece.kind)[1]
        self.game.hard_drop()

        # No line-clear pause or flash: play continues straight into the next
        # piece so the board is never frozen.
        self.anim, self.plan, self.timer = None, None, 0.0
        self.locked += 1
        self.phase = 'settle'
        return 1

    def _advance(self):
        """Move to the next phase; returns pieces locked by the transition."""
        if self.phase == 'move':
            self.phase = 'drop'
            return 0
        if self.phase == 'drop':
            return self._finish_drop()
        self.phase = 'plan'                             # settle finished
        return 0

    def update(self, dt):
        """Advance by ``dt`` seconds; returns how many pieces were locked.

        This is the whole per-frame behaviour, so a test can drive the playback
        without a window and without a real clock.
        """
        game = self.game
        if game.game_over or game.current is None:
            self.anim = self.plan = None
            self.phase = 'done'
            return 0
        if self.phase in ('plan', 'done'):
            if not self._begin_piece():
                self.phase = 'done'
                return 0
        else:
            self.timer += dt

        locked = 0
        # One frame can span a whole phase at high speed, so keep consuming
        # finished phases until the timer no longer overflows.
        for _ in range(8):
            duration = self._duration()
            if self.timer < duration:
                break
            self.timer -= duration
            locked += self._advance()
            if self.phase == 'plan':
                if not self._begin_piece():
                    self.phase = 'done'
                    break
                self.timer = 0.0
        return locked

    def step(self):
        """Place the current piece immediately (the S key while paused)."""
        if self.game.game_over or self.game.current is None:
            return False
        if self.anim is None and not self._begin_piece():
            return False
        self._finish_drop()
        return True

    def apply_speed(self, factor):
        """Scale playback speed by ``factor``, clamped to a sane range."""
        self.speed = max(SPEED_MIN, min(SPEED_MAX, self.speed * factor))
        return self.speed


# --- Run loops ------------------------------------------------------------

def _new_game(config=None, seed=None):
    """Create a ``Game``, honouring ``config['game']['seed']`` when unset.

    Returns ``(game, seed)`` so callers can show (and reuse) the seed in play;
    ``None`` means the engine picked one at random.
    """
    if seed is None:
        seed = _cfg(config, ('game', 'seed'), None, int)
    return Game(seed=seed), seed


def _run_window(layout, title, body, size=None):
    """Open a window, run ``body(screen)``, and always shut pygame down.

    ``set_mode`` raises ``pygame.error`` when no display is available (headless
    without the dummy driver), reported as a hint rather than a traceback.
    ``size`` overrides the window dimensions for a layout that describes one half
    of a wider screen (battle mode); it defaults to the layout's own size.
    """
    pygame.init()
    try:
        pygame.key.set_repeat()      # OS key repeat would double-fire our DAS
        screen = pygame.display.set_mode(size or (layout.width, layout.height))
        pygame.display.set_caption(title)
    except pygame.error as exc:
        # pygame.init() itself only reports failures, so the display may be
        # missing here; treat that as "no window" rather than a traceback.
        print(f'render: cannot open a window ({exc}); try SDL_VIDEODRIVER=dummy')
        pygame.quit()
        return None
    try:
        return body(screen)
    finally:
        pygame.quit()


def run_human(config=None, seed=None, settings=None, recorder=None):
    """Play Tetris yourself. Blocks until the window is closed.

    Every key comes from the input settings (``config.yaml``'s ``input:`` section,
    overridden by ``settings.yaml``, editable from the menu's Settings screen), so
    the defaults are TETR.IO's: arrows move, UP or X rotate clockwise, Z or LCTRL
    counter-clockwise, A spins 180, SPACE hard drops, C or LSHIFT holds, P pauses,
    R restarts, N reseeds, ESC quits.

    ``settings`` is passed in when the caller already has a live object (the menu
    does), so a rebind made there is in force immediately.
    """
    layout = Layout.from_config(config)
    settings = settings if settings is not None else controls.load(config)
    print(settings_summary(settings))
    for warning in settings.warnings + settings.problems():
        print(f'input: {warning}')
    _run_window(layout, 'Tetris RL - play',
                lambda screen: _human_loop(screen, layout, config, seed, settings,
                                           recorder))


def _new_human_input(settings, recorder=None):
    """A fresh input handler, sharing one live settings object with the loop."""
    return HumanInput(settings, recorder)


def _human_loop(screen, layout, config, seed, settings=None, recorder=None):
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    fps = max(20.0, min(240.0, _cfg(config, ('game', 'fps'), 60.0, float)))
    settings = settings if settings is not None else controls.load(config)
    game, seed = _new_game(config, seed)
    player_input = _new_human_input(settings, recorder)
    if recorder is not None:
        # The seed the game actually used, which is not the one the CLI was given
        # when it was None: a recording you cannot replay is half a recording.
        recorder.state_if_due(game, force=True,
                              extra={'seed': seed, 'note': 'session start'})
    paused, running = False, True

    while running:
        # Clamp dt: dragging the window or a slow frame must not teleport the
        # piece downwards or fire a burst of auto-repeats.
        dt = min(clock.tick(fps) / 1000.0, 0.05)
        if recorder is not None:
            recorder.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = settings.action_for(event.key)
                if recorder is not None:
                    # Every dispatched key is recorded, including the ones bound to
                    # a game action and the ones bound to nothing at all: "my key
                    # does nothing" is a bug report, and only the record can tell
                    # it apart from a key that never arrived.
                    recorder.key(event.key, action, True)
                if action == 'quit' or event.key == pygame.K_q:
                    running = False
                elif action == 'pause':
                    paused = not paused
                    player_input.release_all()
                elif action in ('restart', 'new_seed'):
                    if action == 'new_seed':
                        seed = random.randrange(1, 1 << 30)
                    game, seed = _new_game(config, seed)
                    player_input, paused = _new_human_input(settings, recorder), False
                    if recorder is not None:
                        recorder.state_if_due(game, force=True,
                                             extra={'note': 'restart'})
                elif not paused and not game.game_over:
                    player_input.key_down(event.key, game)
            elif event.type == pygame.KEYUP and not paused:
                if recorder is not None:
                    recorder.key(event.key, settings.action_for(event.key), False)
                player_input.key_up(event.key, game)

        if not paused and not game.game_over:
            player_input.update(dt, game)
            if recorder is not None:
                recorder.state_if_due(game)
        renderer.draw(game, paused=paused, hud=_seed_hud(seed),
                      status=_binding_hint(settings))
        pygame.display.flip()


def run_agent(policy, config=None, seed=None, fps=30.0, settings=None):
    """Watch ``policy`` play. Blocks until the window is closed.

    ``policy(game, valid_actions) -> (rotation, x)``, called exactly once per
    piece, *before* that piece's animation starts, so a slow (neural net) policy
    stalls one frame per placement instead of stalling mid-animation.

    Controls: SPACE pause/resume, UP/DOWN faster/slower, S steps one piece while
    paused, R restart, N new game with a new seed, ESC or Q quit -- the four loop
    keys come from the input settings and are shown with their live bindings. The
    HUD shows lines, score, level, pieces placed, speed and the paused/game-over
    state.
    """
    layout = Layout.from_config(config)
    settings = settings if settings is not None else controls.load(config)
    _run_window(layout, 'Tetris RL - agent',
                lambda screen: _agent_loop(screen, layout, config, seed, policy,
                                           fps, settings))


def _agent_loop(screen, layout, config, seed, policy, fps, settings=None):
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    fps = max(5.0, min(240.0, float(fps)))
    settings = settings if settings is not None else controls.load(config)
    game, seed = _new_game(config, seed)
    player = AgentPlayer(game, policy)
    speed_up, speed_down = HoldRepeat(0.25, 0.09), HoldRepeat(0.25, 0.09)
    paused, running = False, True

    while running:
        dt = min(clock.tick(fps) / 1000.0, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # The watcher keeps its own speed/step keys on top of the
                # rebindable loop actions, since none of them are piece controls.
                action = settings.action_for(event.key)
                if action == 'quit' or event.key == pygame.K_q:
                    running = False
                elif action == 'pause':
                    paused = not paused
                elif event.key == pygame.K_UP and speed_up.press():
                    player.apply_speed(SPEED_STEP)
                elif event.key == pygame.K_DOWN and speed_down.press():
                    player.apply_speed(1.0 / SPEED_STEP)
                elif event.key == pygame.K_s and paused:
                    player.step()
                elif action in ('restart', 'new_seed'):
                    if action == 'new_seed':
                        seed = random.randrange(1, 1 << 30)
                    game, seed = _new_game(config, seed)
                    player, paused = AgentPlayer(game, policy, player.speed), False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    speed_up.release()
                elif event.key == pygame.K_DOWN:
                    speed_down.release()

        # Speed keys repeat while held; the animation itself only runs unpaused.
        for _ in range(speed_up.due(dt)):
            player.apply_speed(SPEED_STEP)
        for _ in range(speed_down.due(dt)):
            player.apply_speed(1.0 / SPEED_STEP)
        if not paused:
            player.update(dt)

        renderer.draw(game, current=player.display_piece(),
                      ghost_cells=player.ghost_cells(), paused=paused,
                      hud=_seed_hud(seed) + [('SPEED', f'{player.speed:.2f}x')],
                      status=_agent_hint(settings))
        pygame.display.flip()


# --- Battle mode ----------------------------------------------------------

def _battle_layouts(config=None):
    """Two single-board layouts side by side, plus the window size they need.

    Each half keeps the ordinary geometry -- hold box, board, side panel -- and the
    right half is simply shifted by one half's width, so ``Renderer``, ``cell_rect``
    and every drawing routine work unchanged: there is no notion of a "second
    board" anywhere in them.

    Four panels now share the window (two hold boxes and two side panels), so the
    cell is capped twice: by ``BATTLE_CELL_MAX`` and by what is left of
    ``BATTLE_MAX_WIDTH`` once the panels and margins are paid for. At the shipped
    defaults (cell 30, panel 200, hold 96) the first cap wins and the window is
    1,192 x 576 -- which fits a 1,280-wide screen, hence the budget below. A
    ``side_panel_width`` large enough to break that is clamped for battle only, so
    the two boards stay on screen.
    """
    cell = _cfg(config, ('game', 'cell_size'), DEFAULT_CELL, int)
    panel = _cfg(config, ('game', 'side_panel_width'), PANEL_W, int)
    hold = _cfg(config, ('game', 'hold_panel_width'), HOLD_PANEL_W, int)
    panel = max(120, min(BATTLE_PANEL_MAX, panel))
    hold = max(56, min(200, hold))
    # One half is three margins + the hold box + the panel + the board.
    fixed = 3 * MARGIN + hold + panel
    room = BATTLE_MAX_WIDTH // 2 - fixed
    cell = max(10, min(BATTLE_CELL_MAX, cell, room // COLS))
    left = Layout(cell=cell, panel_w=panel, margin=MARGIN, hold_w=hold)
    right = Layout(cell=cell, panel_w=panel, margin=MARGIN, hold_w=hold)
    right.hold_x += left.width
    right.board_x += left.width
    right.panel_x += left.width
    return left, right, (left.width * 2, left.height)


def _bot_speed(policy):
    """Animation speed that makes ``AgentPlayer`` place ``policy.pps`` pieces/second.

    The pacing handicap is enforced here rather than by sleeping: a piece takes
    ``AGENT_PIECE_TIME / speed`` seconds, so inverting that gives the speed for a
    target rate. ``policy.pps`` comes from the difficulty preset, which is
    calibrated against published TETRA LEAGUE attack rates.
    """
    pps = float(getattr(policy, 'pps', 1.0) or 1.0)
    return max(0.05, min(SPEED_MAX, AGENT_PIECE_TIME * max(0.05, pps)))


class _BattleState:
    """Everything a battle owns, so one frame of it can run without a window.

    Held in one object because the per-frame update has to be drivable headlessly
    (see ``_battle_step``): nothing here touches pygame's display.
    """

    def __init__(self, battle, policy, difficulty=None, messiness='default',
                 settings=None, recorder=None):
        self.battle = battle
        self.policy = policy
        self.difficulty = difficulty
        self.messiness = messiness
        self.settings = settings if settings is not None else controls.load()
        self.recorder = recorder
        self.paused = False
        self.over = False
        self.result = None
        self.round_message = None
        self.round_hold = 0.0
        self.reaction = 0.0                 # bot pause after garbage lands on it
        self.agent = AgentPlayer(battle.sides[1].game, policy,
                                 speed=_bot_speed(policy))
        self.player_input = HumanInput(self.settings, recorder)
        self.locks = [0, 0]

    def reset(self, seed=None):
        """Restart the match on the *same* ``Battle`` object, score back to zero.

        Keeping the object matters: the policy's ``incoming_fn`` closes over it, so
        a fresh ``Battle`` would leave the bot reading a dead queue.
        """
        battle = self.battle
        if seed is not None:
            battle.seed = seed
        battle.rng = random.Random(battle.seed)
        battle.log = []
        battle.new_round()                  # fresh boards, same seed unless set
        battle.round = 1
        for side in battle.sides:
            side.wins = 0
        self.agent = AgentPlayer(battle.sides[1].game, self.policy,
                                 speed=_bot_speed(self.policy))
        self.player_input = HumanInput(self.settings, self.recorder)
        self.paused = self.over = False
        self.result = self.round_message = None
        self.round_hold = self.reaction = 0.0
        self.locks = [0, 0]
        return battle


def _battle_step(state, dt, keys_down=(), keys_up=()):
    """Advance the match by ``dt``. Returns a list of ``(kind, ...)`` events.

    Everything the match does per frame lives here -- the human's input and locks,
    the bot's pacing, the rules, the K.O. and the round reset -- and none of it
    needs a window, so the whole battle can be driven by a test.

    Locks are detected by comparing ``pieces_placed``/``lines`` across the frame
    rather than from a return value, because a hard drop locks inside
    ``key_down`` while a lock-delay lock happens inside ``update()``; the counter
    catches both and also gives the cleared count the rules need.
    """
    events = []
    battle = state.battle
    battle.update(dt)
    if state.over:
        return events
    if state.round_hold > 0.0:              # K.O. beat, then the next round
        state.round_hold -= dt
        if state.round_hold <= 0.0:
            battle.new_round()
            state.agent = AgentPlayer(battle.sides[1].game, state.policy,
                                      speed=_bot_speed(state.policy))
            state.player_input = HumanInput(state.settings, state.recorder)
            state.reaction = 0.0
            state.round_message = None
            events.append(('round', battle.round))
        return events
    if state.paused:
        return events

    game = battle.sides[0].game               # the human
    if not game.game_over:
        before = (game.pieces_placed, game.lines)
        for key in keys_down:
            if state.recorder is not None:
                state.recorder.key(key, state.settings.action_for(key), True)
            state.player_input.key_down(key, game)
        for key in keys_up:
            if state.recorder is not None:
                state.recorder.key(key, state.settings.action_for(key), False)
            state.player_input.key_up(key, game)
        state.player_input.update(dt, game)
        if game.pieces_placed != before[0]:
            summary = battle.after_lock(0, game.lines - before[1])
            state.locks[0] += 1
            events.append(('lock', 0, summary))
            if state.recorder is not None:
                state.recorder.attack(0, summary, battle)

    game = battle.sides[1].game               # the bot
    if not game.game_over:
        if state.reaction > 0.0:
            state.reaction -= dt
        else:
            before = (game.pieces_placed, game.lines)
            state.agent.update(dt)
            if game.pieces_placed != before[0]:
                summary = battle.after_lock(1, game.lines - before[1])
                state.locks[1] += 1
                events.append(('lock', 1, summary))
                if state.recorder is not None:
                    state.recorder.attack(1, summary, battle)
                if summary['tanked']:
                    state.reaction = max(
                        0.0, float(getattr(state.policy, 'reaction', 0.0) or 0.0))

    if state.recorder is not None:
        state.recorder.tick()
        state.recorder.state_if_due(battle.sides[0].game, battle=battle)

    for index in (0, 1):                      # who topped out, and what it costs
        if not battle.sides[index].game.game_over:
            continue
        foe = battle.other(index)
        foe.wins += 1
        won = battle.winner()
        if won is not None:
            state.over = True
            state.result = ('You win the match' if won == 0
                            else 'The bot wins the match')
            events.append(('match_over', won))
        else:
            state.round_hold = ROUND_PAUSE
            state.round_message = f'{foe.name} takes the round'
            events.append(('round_over', index))
        break
    return events


def _draw_garbage_meter(surface, queue, now, x, top, height, width=6):
    """A vertical bar of what is about to land on a board.

    Bright means ready: those lines enter on the next lock that clears nothing.
    Dim means still travelling -- TETR.IO's travel delay, which is the window in
    which a clear can still cancel them.
    """
    pending = queue.pending
    if pending <= 0:
        return
    ready = min(queue.ready(now), 20)
    travelling = min(pending - ready, 20)
    step = max(3, height // 20)
    y = top + height
    for _ in range(travelling):
        y -= step
        pygame.draw.rect(surface, GARBAGE_PENDING, (x, y, width, step - 1))
    for _ in range(ready):
        y -= step
        pygame.draw.rect(surface, GARBAGE_READY, (x, y, width, step - 1))


def _battle_draw(screen, renderers, state, layouts, label=None):
    """One frame of a battle: both boards, both panels, both meters, overlays."""
    battle = state.battle
    screen.fill(BG)                           # once: each renderer skips the fill
    for index, (renderer, layout) in enumerate(zip(renderers, layouts)):
        side = battle.sides[index]
        current = ghost = None
        if index == 1 and not state.over:
            current, ghost = state.agent.display_piece(), state.agent.ghost_cells()
        hud = [('SENT', side.sent), ('QUEUED', side.queue.pending),
               ('WINS', side.wins)]
        if index == 0:
            status = (f'you {battle.sides[0].wins} - {battle.sides[1].wins} bot\n'
                      + _binding_hint(state.settings))
        else:
            status = (f'bot: {label or state.difficulty or "custom"}   '
                      f'reaction {getattr(state.policy, "reaction", 0.0):.2f}s\n'
                      f'garbage: {state.messiness}')
        renderer.draw(side.game, current=current, ghost_cells=ghost, hud=hud,
                      status=status, fill_bg=False)
        meter_x = (layout.board_x + layout.board_w + 3 if index == 0
                   else layout.board_x - 9)
        _draw_garbage_meter(screen, side.queue, battle.time, meter_x,
                            layout.board_y, layout.board_h)

    center = (layouts[0].width, layouts[0].height // 2)
    if state.over:
        renderers[0].label(state.result or 'match over', center,
                           GAME_OVER_COLOR if (state.result or '').startswith('The bot')
                           else TEXT, renderers[0].big, center=True)
        renderers[0].label('R restart    N new seed    ESC quit',
                           (center[0], center[1] + 34), DIM, renderers[0].font,
                           center=True)
    elif state.round_hold > 0.0:
        renderers[0].label(state.round_message or 'K.O.', center, TEXT,
                           renderers[0].big, center=True)
    elif state.paused:
        renderers[0].label('PAUSED', center, TEXT, renderers[0].big, center=True)


def _battle_loop(screen, layouts, state, config=None, label=None):
    """The pygame frame loop: collect input, step the match, draw it."""
    renderers = tuple(Renderer(screen, layout) for layout in layouts)
    clock = pygame.time.Clock()
    fps = max(20.0, min(240.0, _cfg(config, ('game', 'fps'), 60.0, float)))
    running = True
    while running:
        dt = min(clock.tick(fps) / 1000.0, 0.05)
        keys_down, keys_up = [], []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = state.settings.action_for(event.key)
                if action == 'quit' or event.key == pygame.K_q:
                    running = False
                elif action == 'pause':
                    state.paused = not state.paused
                    state.player_input.release_all()
                elif action in ('restart', 'new_seed'):
                    seed = (random.randrange(1, 1 << 30)
                            if action == 'new_seed' else state.battle.seed)
                    state.reset(seed=seed)
                else:
                    keys_down.append(event.key)
            elif event.type == pygame.KEYUP:
                keys_up.append(event.key)
        if not running:
            break
        _battle_step(state, dt, keys_down, keys_up)
        _battle_draw(screen, renderers, state, layouts, label)
        pygame.display.flip()
    return None


def _difficulty_label(module, difficulty):
    """A preset's human-readable label, or the raw name when it is unavailable."""
    try:
        return module.HANDICAP_PRESETS[difficulty]['label']
    except Exception:                                  # noqa: BLE001
        return difficulty or 'custom bot'


def run_battle(policy=None, config=None, seed=None, difficulty=None,
               messiness='default', rounds=3, settings=None, recorder=None):
    """Play a first-to-``rounds`` versus match against the bot. Blocks until quit.

    You are the left board and the bot is the right one. The bot's strength comes
    from ``policies.make_versus_policy``'s handicap preset (search depth and beam,
    a pieces-per-second cap and a mistake rate) and its garbage awareness from the
    battle's own incoming queue, which the preset policy reads through
    ``incoming_fn`` -- so it can price a tall stack against what is about to land
    on it instead of playing as if the board were clean.

    Garbage is sent by clearing lines (TETR.IO's attack table, combo multiplier,
    B2B charging and Surge) and lands only on a lock that clears nothing, at most
    eight lines at a time. Controls are ``play``'s, plus P pause, R restart, N new
    seed and ESC to leave.
    """
    if seed is None:
        seed = _cfg(config, ('game', 'seed'), None, int)
    if seed is None:
        seed = random.randrange(1, 1 << 30)

    battle = Battle(seed=seed, rounds_to_win=max(1, int(rounds)),
                    messiness=messiness, names=('you', 'bot'))
    if recorder is not None:
        recorder.record('note', mode='battle', seed=seed, rounds=rounds,
                        messiness=messiness, note='session start')
    module, error = _load_policies()
    if policy is None:
        if error:
            print(f'render: {error}')
            return None
        difficulty = difficulty or module.DEFAULT_DIFFICULTY
        policy = module.make_versus_policy(
            difficulty=difficulty, incoming_fn=lambda: battle.incoming(1),
            seed=seed)
    label = _difficulty_label(module, difficulty) if module else None
    settings = settings if settings is not None else controls.load(config)
    print(settings_summary(settings))
    for warning in settings.warnings + settings.problems():
        print(f'input: {warning}')
    state = _BattleState(battle, policy, difficulty=difficulty,
                         messiness=messiness, settings=settings,
                         recorder=recorder)
    left, right, size = _battle_layouts(config)
    print(f'render: battle against {getattr(policy, "policy_name", "policy")} '
          f'(seed {seed}, first to {state.battle.rounds_to_win}, '
          f'{messiness} garbage)')
    return _run_window(left, 'Tetris RL - battle',
                       lambda screen: _battle_loop(screen, (left, right), state,
                                                   config, label),
                       size=size)


# --- Menu -----------------------------------------------------------------

def _seed_hud(seed):
    """The stats row naming the seed in play (or that it was random)."""
    return [('SEED', 'random' if seed is None else seed)]


def _load_policies():
    """Import ``tetrisrl.policies`` lazily; returns ``(module, error)``.

    Deliberately not a module-level import: that module pulls in torch and the
    model files, and a broken checkpoint must not stop the renderer from starting.
    """
    try:
        from tetrisrl import policies
    except Exception as exc:                          # noqa: BLE001 - report anything
        return None, f'tetrisrl.policies unavailable: {exc}'
    return policies, None


def _policy_names(policies_module):
    """Names to offer, with a default entry when the catalogue is empty."""
    try:
        names = [str(name) for name in policies_module.list_policies()]
    except Exception as exc:                          # noqa: BLE001
        return ['default'], f'list_policies() failed: {exc}'
    # load_policy defaults name=None, so an empty catalogue is still usable.
    return (names or ['default']), None


def _difficulty_names(policies_module):
    """Difficulty presets to offer the battle menu, in the module's own order."""
    try:
        return [str(name) for name in policies_module.HANDICAP_PRESETS]
    except Exception as exc:                          # noqa: BLE001
        return []


# --- Settings screen -------------------------------------------------------
#
# The whole interaction lives in ``_settings_step`` (no pygame calls) and the whole
# presentation in ``_settings_draw``, so the screen can be driven and drawn
# headlessly. That split is the same one battle mode uses, for the same reason: a
# UI that can only be exercised by a human at a keyboard is a UI nobody tests.

#: The timing rows: ``(attribute, label, arrow step, shift step, min, max, unit)``.
#: The bounds are wide but real -- a 5-second DAS is a mistake rather than a
#: preference, and SDF above 200x is indistinguishable from instant.
#: ``(key, label, step, fine, lo, hi, unit)``. ``hi`` may be ``None`` for a value
#: with no ceiling -- DCD is a delay and the player is allowed a silly one.
SETTINGS_TIMING = (
    ('das_ms', 'DAS', 5, 1, 0, 500, 'ms'),
    ('arr_ms', 'ARR', 5, 1, 0, 100, 'ms'),
    ('dcd_ms', 'DCD', 5, 1, 0, None, 'ms'),
    ('sdf', 'SDF', 1, 1, 0, 200, 'x'),
    ('lock_delay_ms', 'Lock delay', 5, 1, 0, 2000, 'ms'),
)
#: Keys per action. Two, as ``controls`` documents.
SETTINGS_SLOTS = 2


def settings_rows(settings=None):
    """The rows the settings screen lists: one per action, then the timings."""
    rows = [{'kind': 'binding', 'action': action, 'label': label}
            for action, label, _group in controls.ACTIONS]
    rows += [{'kind': 'timing', 'key': key, 'label': label, 'step': step,
              'fine': fine, 'lo': lo, 'hi': hi, 'unit': unit}
             for key, label, step, fine, lo, hi, unit in SETTINGS_TIMING]
    return rows


def _timing_text(settings, row):
    """A timing row's value as the screen shows it."""
    value = getattr(settings, row['key'])
    if row['key'] == 'sdf':
        if settings.soft_instant:
            return 'instant', ('SDF instant: soft drop teleports to the landing '
                               'row without locking   (I toggles)')
        return f'{float(value):g}x', 'a multiple of gravity   (I = instant)'
    if row['key'] == 'arr_ms' and float(value) <= 0:
        return 'instant', 'ARR 0 slides to the wall once DAS expires'
    return f'{float(value):.0f} {row["unit"]}', ''


class _SettingsState:
    """Selection, capture and messages for the settings screen.

    Holds no pygame state, so ``_settings_step`` can be run by a test.
    """

    def __init__(self, settings=None, rows=None):
        self.settings = settings if settings is not None else controls.load()
        self.rows = rows if rows is not None else settings_rows(self.settings)
        self.row = 0
        self.slot = 0
        self.capture = None                # (action, slot) while waiting for a key
        self.message = ''
        self.dirty = False
        #: The last numeric SDF, so toggling instant off gives the player their
        #: factor back instead of dumping them at the default.
        self.sdf_memory = None

    @property
    def current(self):
        return self.rows[self.row]

    def move(self, step):
        self.row = (self.row + step) % len(self.rows)
        self.slot = 0

    def nudge(self, direction, fine=False):
        """LEFT/RIGHT: pick a slot on a binding row, change a value on a timing row."""
        row = self.current
        if row['kind'] != 'timing':
            self.slot = (self.slot + (1 if direction > 0 else -1)) % SETTINGS_SLOTS
            return
        amount = (row['fine'] if fine else row['step']) * (1 if direction > 0 else -1)
        value = getattr(self.settings, row['key'])
        value = 0.0 if value is None else float(value)
        value = max(row['lo'], value + amount)
        if row['hi'] is not None:
            value = min(row['hi'], value)
        if row['key'] == 'sdf':
            # A number takes soft drop out of instant; the bottom of the range is
            # instant by definition, so remember the factor to come back to.
            if value > 0:
                self.settings.soft_drop_instant = False
            if value > 0 and value != getattr(self.settings, 'sdf', None):
                self.sdf_memory = value
        setattr(self.settings, row['key'], value)
        self.dirty = True
        self.message = ''

    def toggle_soft_instant(self):
        """``I`` on the SDF row: instant in one press, and back again.

        From the default SDF of 41 this used to be 41 nudges, and nothing on the
        screen said instant existed at all.
        """
        if self.settings.soft_instant:
            # Also the way back from ``sdf: 0`` in a config file, not just the flag.
            self.settings.soft_drop_instant = False
            self.settings.sdf = (self.sdf_memory
                                 or float(controls.DEFAULT_TIMING['sdf']))
            self.message = f'SDF {self.settings.sdf:g}x'
        else:
            self.sdf_memory = float(self.settings.sdf
                                    or controls.DEFAULT_TIMING['sdf'])
            self.settings.soft_drop_instant = True
            self.message = ('SDF instant: soft drop teleports down without locking')
        self.dirty = True

    def bind(self, action, slot, key):
        """Assign a captured key; returns the message to show."""
        name = controls.key_name(key)
        # Read the clash *before* binding it: set_binding removes the key from
        # wherever it was, so afterwards there is nothing left to report.
        moved = [other for other in controls.ACTION_NAMES
                 if other != action and name in self.settings.binding_names(other)]
        try:
            self.settings.set_binding(action, slot, name)
        except controls.ControlsError as exc:
            return str(exc)
        self.dirty = True
        note = f' (removed from {", ".join(moved)})' if moved else ''
        return f'{controls.ACTION_LABELS[action]} slot {slot + 1} = {name}{note}'

    def clear(self):
        row = self.current
        if row['kind'] != 'binding':
            return 'nothing to clear on this row'
        self.settings.clear_binding(row['action'], self.slot)
        self.dirty = True
        return f'{row["label"]} slot {self.slot + 1} cleared'


def _settings_step(state, key, fine=False):
    """Handle one key press on the settings screen.

    Returns ``'exit'`` to leave, ``'save'`` after writing the settings file, or
    None. A pending capture swallows whatever key arrives next, which is how "press
    a key..." binds anything at all -- including ESC, which has to cancel instead.
    """
    if state.capture is not None:
        action, slot = state.capture
        state.capture = None
        if key == pygame.K_ESCAPE:
            state.message = 'binding cancelled'
            return None
        state.message = state.bind(action, slot, key)
        return None
    if key == pygame.K_ESCAPE:
        return 'exit'
    if key in (pygame.K_UP, pygame.K_DOWN):
        state.move(-1 if key == pygame.K_UP else 1)
    elif key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_MINUS,
                 pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS,
                 pygame.K_KP_MINUS):
        left = key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS)
        state.nudge(-1 if left else 1, fine)
    elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        if state.current['kind'] == 'binding':
            state.capture = (state.current['action'], state.slot)
            state.message = 'press a key...  (ESC cancels)'
        else:
            state.message = 'left/right changes this value (shift = 1ms steps)'
    elif key == pygame.K_i and state.current.get('key') == 'sdf':
        # One press, either direction: SDF instant is a mode, not a value you
        # should have to nudge down to from 41.
        state.toggle_soft_instant()
    elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        state.message = state.clear()
    elif key == pygame.K_s:
        try:
            path = controls.save(state.settings)
        except OSError as exc:
            state.message = f'could not save: {exc}'
            return None
        state.dirty = False
        state.message = f'saved to {path}'
        return 'save'
    return None


def _settings_row_height(renderer, rows, y):
    """Row height that fits ``rows`` settings lines between ``y`` and the footer.

    A small ``cell_size`` makes a short window, and a fixed row height then runs
    the last rows through the footer. Fitting them instead means the text clips
    gracefully where an overflowing list would just overlap. Measured at
    ``cell_size`` 10 the default list needed 272 px in a 234 px window.
    """
    base = max(14, renderer._height(renderer.small, 12) + 4)
    room = renderer.layout.height - y - (2 * base + 10)   # footer plus some air
    return max(9, min(base, room // max(1, rows)))


def _settings_draw(screen, renderer, state):
    """Draw the settings list, the values, any warnings and the footer."""
    layout = renderer.layout
    small, body = renderer.small, renderer.font
    screen.fill(BG)
    row_h = max(14, renderer._height(small, 12) + 4)
    top = layout.margin + 4
    x = layout.margin + 8
    renderer.label('SETTINGS', (x, top), TEXT, renderer.big)
    renderer.label('input bindings and timing', (x + 130, top + 6), DIM, small)
    y = top + renderer._height(renderer.big, 22) + 6
    row_h = _settings_row_height(renderer, len(state.rows), y)

    slot_x = x + 240
    for index, row in enumerate(state.rows):
        selected = index == state.row
        if selected:
            pygame.draw.rect(screen, PANEL_BG,
                             pygame.Rect(x - 4, y - 2, layout.width - 2 * x + 8,
                                         row_h))
        if row['kind'] == 'binding':
            renderer.label(row['label'], (x, y), TEXT if selected else DIM, small)
            names = state.settings.binding_names(row['action'])
            for slot in range(SETTINGS_SLOTS):
                box = pygame.Rect(slot_x + slot * 130, y - 2, 120, row_h - 2)
                focused = selected and slot == state.slot
                if state.capture is not None and focused:
                    text = 'press a key...'
                elif slot < len(names):
                    text = names[slot]
                else:
                    text = '(none)'
                pygame.draw.rect(screen, (60, 60, 78) if focused else PANEL_BG, box)
                pygame.draw.rect(screen, FRAME if focused else GRID_LINE, box, 1)
                renderer.label(text, (box.x + 6, y),
                               TEXT if slot < len(names) else DIM, small)
        else:
            renderer.label(row['label'], (x, y), TEXT if selected else DIM, small)
            text, note = _timing_text(state.settings, row)
            renderer.label(text, (slot_x, y),
                           TEXT if selected else DIM, small)
            if note and selected:
                renderer.label(note, (slot_x + 150, y), DIM, small)
        y += row_h

    warnings = list(state.settings.warnings) + state.settings.problems()
    for text in warnings[:2]:
        renderer.label(f'! {text}', (x, y), GAME_OVER_COLOR, small)
        y += row_h
    if state.message:
        renderer.label(state.message, (x, y), TEXT, small)
        y += row_h
    footer = ('up/down row   left/right slot or value (shift = fine)   ENTER bind'
              '\nI = SDF instant   DEL clear   S save   ESC back')
    renderer.label(footer, (x, layout.height - 2 * row_h - 6), DIM, small)


def _settings_screen(screen, layout, settings=None, config=None):
    """Run the settings screen until ESC. Returns the settings in force."""
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    settings = settings if settings is not None else controls.load(config)
    state = _SettingsState(settings)
    while True:
        clock.tick(MENU_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return state.settings
            if event.type != pygame.KEYDOWN:
                continue
            fine = bool(event.mod & pygame.KMOD_SHIFT)
            if _settings_step(state, event.key, fine) == 'exit':
                return state.settings
        _settings_draw(screen, renderer, state)
        pygame.display.flip()


def run_menu(config=None):
    """Title screen: watch, play, battle, settings, quit.

    Selectable by number key or arrow + Enter. "Watch AI" first picks a policy and
    a seed, "Battle the bot" picks a difficulty and a seed, then each hands off to
    ``run_agent``/``run_battle``; a missing or broken policies module shows an
    in-window message instead of a traceback. "Settings" opens the input screen and
    comes back here, because rebinding is something you do between games.
    """
    layout = Layout.from_config(config)
    settings = controls.load(config)
    print(settings_summary(settings))
    for warning in settings.warnings + settings.problems():
        print(f'input: {warning}')
    while True:
        choice = _run_window(layout, 'Tetris RL',
                             lambda screen: _menu_loop(screen, layout, settings))
        if choice is None:
            return
        action, payload = choice
        if action == 'settings':
            _run_window(layout, 'Tetris RL - settings',
                        lambda screen: _settings_screen(screen, layout, settings))
            continue
        if action == 'agent':
            run_agent(payload[0], config=config, seed=payload[1],
                      settings=settings)
        elif action == 'battle':
            run_battle(difficulty=payload[0], seed=payload[1], config=config,
                       settings=settings)
        else:
            run_human(config=config, settings=settings)
        return


def _menu_loop(screen, layout, settings=None):
    """Run the menu until it produces a choice.

    Returns ``('agent', (policy, seed))``, ``('human', None)`` or ``None`` to quit.
    The policy is loaded here (possibly slowly) so a failure can still be shown
    inside the window rather than killing the process.
    """
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    items = ['Watch AI', 'Play yourself', 'Battle the bot', 'Settings', 'Quit']
    state, selected, message, seed_text = 'main', 0, None, ''
    names, difficulties = [], []

    while True:
        clock.tick(MENU_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                if state == 'main':
                    return None
                state, message = 'main', None      # ESC backs out of a submenu
                continue
            activate = event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
            step = -1 if event.key == pygame.K_UP else 1
            if state == 'main':
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected = (selected + step) % len(items)
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    selected, activate = event.key - pygame.K_1, True
                if not activate:
                    continue
                if selected == 4:
                    return None
                if selected == 3:
                    return ('settings', None)
                if selected == 1:
                    return ('human', None)
                module, error = _load_policies()
                if selected == 2:                   # battle: pick a difficulty
                    difficulties, error = ([], error) if error else (
                        _difficulty_names(module), None)
                    state, selected, message = 'difficulty', 0, error
                    continue
                names, error = ([], error) if error else _policy_names(module)
                state, selected, message = 'policies', 0, error
            elif state == 'difficulty':
                if event.key in (pygame.K_UP, pygame.K_DOWN) and difficulties:
                    selected = (selected + step) % len(difficulties)
                elif (pygame.K_1 <= event.key <= pygame.K_9
                      and event.key - pygame.K_1 < len(difficulties)):
                    selected, activate = event.key - pygame.K_1, True
                if activate and difficulties:
                    seed_text, state = '', 'battle_seed'
            elif state == 'battle_seed':
                if event.key == pygame.K_BACKSPACE:
                    seed_text = seed_text[:-1]
                elif activate:
                    return ('battle', (difficulties[selected],
                                       int(seed_text) if seed_text else None))
                elif getattr(event, 'unicode', '').isdigit() and len(seed_text) < 9:
                    seed_text += event.unicode
            elif state == 'policies':
                if event.key in (pygame.K_UP, pygame.K_DOWN) and names:
                    selected = (selected + step) % len(names)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    if event.key - pygame.K_1 < len(names):
                        selected, activate = event.key - pygame.K_1, True
                if activate and names:
                    seed_text, state = '', 'seed'
            elif state == 'seed':
                if event.key == pygame.K_BACKSPACE:
                    seed_text = seed_text[:-1]
                elif activate:
                    # Paint "loading" first: load_policy may read a checkpoint.
                    renderer.draw_menu('TETRIS RL', f'Loading {names[selected]} ...',
                                       [], 0, '')
                    pygame.display.flip()
                    module, error = _load_policies()
                    policy = None
                    if not error:
                        try:
                            policy, description = module.load_policy(
                                None if names[selected] == 'default' else names[selected])
                        except Exception as exc:          # noqa: BLE001
                            error = f'could not load "{names[selected]}": {exc}'
                    if error:
                        state, message = 'policies', error
                        continue
                    print(f'render: policy loaded ({description})')
                    return ('agent', (policy, int(seed_text) if seed_text else None))
                elif getattr(event, 'unicode', '').isdigit() and len(seed_text) < 9:
                    seed_text += event.unicode

        if state == 'main':
            renderer.draw_menu('TETRIS RL', 'Choose a mode', items, selected,
                               'up/down + ENTER, or press 1-5\nESC quits')
        elif state == 'difficulty' and difficulties:
            renderer.draw_menu('BATTLE THE BOT', 'Choose a difficulty',
                               [f'{name} - {_difficulty_label(module, name)}'
                                for name in difficulties],
                               selected,
                               'up/down or 1-9, ENTER to start\nESC back', message)
        elif state == 'difficulty':
            renderer.draw_menu('BATTLE THE BOT', 'No difficulties available', [], 0,
                               'ESC back', message)
        elif state == 'battle_seed':
            shown = seed_text if seed_text else '(random)'
            renderer.draw_menu('BATTLE THE BOT',
                               f'Difficulty: {difficulties[selected]}',
                               [f'Seed: {shown}'], 0,
                               'type digits, BACKSPACE to edit\nENTER starts, ESC back')
        elif state == 'policies' and names:
            renderer.draw_menu('WATCH AI', 'Choose a policy', names, selected,
                               'up/down or 1-9, ENTER to start\nESC back', message)
        elif state == 'policies':
            renderer.draw_menu('WATCH AI', 'No policies available', [], 0,
                               'ESC back', message)
        else:
            shown = seed_text if seed_text else '(random)'
            renderer.draw_menu('WATCH AI', f'Policy: {names[selected]}',
                               [f'Seed: {shown}'], 0,
                               'type digits, BACKSPACE to edit\nENTER starts, ESC back')
        pygame.display.flip()


if __name__ == '__main__':
    run_menu()
