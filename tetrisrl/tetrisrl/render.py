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
    from tetrisrl.engine import (BUFFER_ROWS, COLS, PIECES, ROWS, VISIBLE_ROWS,
                                 Game, Piece, spawn_anchor, valid_actions)
except ImportError:                     # `python render.py` from inside the package
    from engine import (BUFFER_ROWS, COLS, PIECES, ROWS, VISIBLE_ROWS, Game,
                        Piece, spawn_anchor, valid_actions)

# Palette, identical to vibecode.py so both front-ends look like one project.
COLORS = {'I': (80, 220, 255), 'J': (40, 40, 255), 'L': (255, 160, 40),
          'O': (240, 240, 60), 'S': (80, 255, 80), 'T': (200, 80, 255),
          'Z': (255, 80, 80)}
BG, GRID_LINE, PANEL_BG = (18, 18, 24), (40, 40, 52), (28, 28, 36)
TEXT, DIM, FRAME = (225, 225, 235), (150, 150, 165), (170, 170, 180)
BLOCK_EDGE = (10, 10, 14)            # 1px dark outline around every block
GAME_OVER_COLOR = (255, 120, 120)

DEFAULT_CELL, PANEL_W, MARGIN = 30, 200, 10
NEXT_PREVIEW = 5                     # pieces shown in the NEXT box

# Timings in seconds, grouped so the feel of the game is one block of tunables.
DAS_DELAY, ARR = 0.17, 0.03          # hold, then horizontal auto-repeat rate
SOFT_DROP_INTERVAL = 0.05            # held down-arrow repeat
LOCK_DELAY, LOCK_RESET_LIMIT = 0.45, 15
AGENT_MOVE_TIME, AGENT_DROP_TIME = 0.08, 0.07
AGENT_SETTLE_TIME = 0.05
SPEED_MIN, SPEED_MAX, SPEED_STEP = 0.25, 8.0, 1.25
MENU_FPS = 30.0

HUMAN_KEYS = ('left/right move   down soft drop   SPACE hard drop\n'
              'Z/X rotate   C hold   P pause   R restart   N new seed   ESC quit')
AGENT_KEYS = ('SPACE pause   up/down speed   S step (paused)\n'
              'R restart   N new seed   ESC quit')


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

    __slots__ = ('delay', 'rate', 'held', 'timer')

    def __init__(self, delay, rate):
        self.delay = max(0.0, delay)
        self.rate = max(0.001, rate)     # never 0: that would loop forever
        self.held, self.timer = False, 0.0

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
        while self.timer <= 0.0 and count < 8:   # capped: a long dt must not spin
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

    def __init__(self, cell=DEFAULT_CELL, panel_w=PANEL_W, margin=MARGIN):
        self.cell, self.panel_w, self.margin = int(cell), int(panel_w), int(margin)
        self.board_w, self.board_h = COLS * self.cell, VISIBLE_ROWS * self.cell
        self.board_x = self.board_y = self.margin
        self.panel_x = self.board_x + self.board_w + self.margin
        self.status_h = int(round(1.4 * self.cell))         # room for two hints
        self.width = self.panel_x + self.panel_w + self.margin
        self.height = self.board_h + 2 * self.margin + self.status_h
        self.status_y = self.board_y + self.board_h + 4

    @classmethod
    def from_config(cls, config=None):
        """Layout from an optional config dict, clamped to sane values.

        Only ``game.cell_size`` and ``game.side_panel_width`` are read; the clamps
        matter because a zero cell size would otherwise make an unopenable window.
        Board dimensions are not configurable -- the renderer is written against
        the engine's fixed 10x20 visible field.
        """
        cell = _cfg(config, ('game', 'cell_size'), DEFAULT_CELL, int)
        panel = _cfg(config, ('game', 'side_panel_width'), PANEL_W, int)
        return cls(cell=max(10, min(80, cell)), panel_w=max(120, min(600, panel)))

    @property
    def board_rect(self):
        return pygame.Rect(self.board_x, self.board_y, self.board_w, self.board_h)

    @property
    def panel_rect(self):
        return pygame.Rect(self.panel_x, 0, self.panel_w, self.height)

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
             message=None):
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
        """
        self.surface.fill(BG)
        self._draw_board(game, current, ghost_cells, show_current, grid, highlight)
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

    def _draw_panel(self, game, hud=()):
        """NEXT, HOLD and the stats, laid out top to bottom on a running cursor."""
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
            hold = game.hold_piece
            next_kinds = [p.kind for p in list(game.next_queue)[:NEXT_PREVIEW]]
            sections = (('NEXT', next_kinds),
                        ('HOLD', [None if hold is None else hold.kind]))
            for title, kinds in sections:
                self.label(title, (x, y), DIM, self.small)
                y += self._height(self.small, 12) + 4
                for kind in kinds:
                    self._preview(pygame.Rect(x, y, box_w, box_h), kind,
                                  dim=game.hold_used and title == 'HOLD')
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
    """

    def __init__(self):
        self.left = HoldRepeat(DAS_DELAY, ARR)
        self.right = HoldRepeat(DAS_DELAY, ARR)
        self.soft = HoldRepeat(0.0, SOFT_DROP_INTERVAL)   # fires immediately
        self.direction = 0                # -1/0/+1: the most recent press wins
        self.gravity_timer = self.lock_timer = 0.0
        self.lock_resets = 0

    def key_down(self, key, game):
        """Handle one gameplay key press; True if this class consumed the key."""
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            step = -1 if key == pygame.K_LEFT else 1
            active, other = (self.left, self.right) if step < 0 else (self.right, self.left)
            other.release()                         # the newest direction wins
            if active.press():
                game.move(step)
            self.direction = step
        elif key == pygame.K_DOWN:
            self.soft.press()
        elif key in (pygame.K_UP, pygame.K_x):      # clockwise, with SRS kicks
            self._rotate(game, 1)
        elif key in (pygame.K_z, pygame.K_LCTRL):
            self._rotate(game, -1)
        elif key == pygame.K_SPACE:
            game.hard_drop()                        # locks the piece
            self.reset()
        elif key in (pygame.K_c, pygame.K_LSHIFT):
            if game.hold():
                self.reset()
        else:
            return False
        return True

    def key_up(self, key):
        """Handle one gameplay key release; True if this class consumed it."""
        if key == pygame.K_LEFT:
            self.left.release()
            self._hand_over(-1, self.right)
        elif key == pygame.K_RIGHT:
            self.right.release()
            self._hand_over(1, self.left)
        elif key == pygame.K_DOWN:
            self.soft.release()
        else:
            return False
        return True

    def _rotate(self, game, direction):
        """Rotate through the engine (it resolves wall kicks) and refresh the lock."""
        if game.rotate(direction):
            self._touched()

    def _hand_over(self, direction, other):
        """Give control to the opposite key if still held, with DAS recharged."""
        if other.held:
            self.direction = direction
            other.held, other.timer = True, other.delay
        else:
            self.direction = 0

    def reset(self):
        """Forget per-piece timing (a new piece just spawned)."""
        self.gravity_timer = self.lock_timer = 0.0
        self.lock_resets = 0

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
        if self.lock_timer > 0.0 and self.lock_resets < LOCK_RESET_LIMIT:
            self.lock_timer = 0.0
            self.lock_resets += 1

    def update(self, dt, game):
        """Advance auto-repeat, soft drop, gravity and lock delay by ``dt``."""
        if game.game_over or game.current is None:
            return False
        if self.direction:                              # horizontal auto-repeat
            repeater = self.left if self.direction < 0 else self.right
            for _ in range(repeater.due(dt)):
                if game.move(self.direction):
                    self._touched()

        # Soft drop replaces gravity rather than adding to it, so holding down
        # never double-steps.
        moved = False
        if self.soft.held:
            for _ in range(self.soft.due(dt)):
                if game.soft_drop():
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
        if self.lock_timer >= LOCK_DELAY:
            game.lock()
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


def _run_window(layout, title, body):
    """Open a window, run ``body(screen)``, and always shut pygame down.

    ``set_mode`` raises ``pygame.error`` when no display is available (headless
    without the dummy driver), reported as a hint rather than a traceback.
    """
    pygame.init()
    try:
        pygame.key.set_repeat()      # OS key repeat would double-fire our DAS
        screen = pygame.display.set_mode((layout.width, layout.height))
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


def run_human(config=None, seed=None):
    """Play Tetris yourself. Blocks until the window is closed.

    Controls: arrows move / soft drop, UP or X rotate clockwise, Z or LCTRL
    rotate counter-clockwise, SPACE hard drop, C or LSHIFT hold, P pause, R
    restart, N new game with a new seed, ESC or Q quit.
    """
    layout = Layout.from_config(config)
    _run_window(layout, 'Tetris RL - play',
                lambda screen: _human_loop(screen, layout, config, seed))


def _human_loop(screen, layout, config, seed):
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    fps = max(20.0, min(240.0, _cfg(config, ('game', 'fps'), 60.0, float)))
    game, seed = _new_game(config, seed)
    controls = HumanInput()
    paused, running = False, True

    while running:
        # Clamp dt: dragging the window or a slow frame must not teleport the
        # piece downwards or fire a burst of auto-repeats.
        dt = min(clock.tick(fps) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    controls.release_all()
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        seed = random.randrange(1, 1 << 30)
                    game, seed = _new_game(config, seed)
                    controls, paused = HumanInput(), False
                elif not paused and not game.game_over:
                    controls.key_down(event.key, game)
            elif event.type == pygame.KEYUP and not paused:
                controls.key_up(event.key)

        if not paused and not game.game_over:
            controls.update(dt, game)
        renderer.draw(game, paused=paused, hud=_seed_hud(seed), status=HUMAN_KEYS)
        pygame.display.flip()


def run_agent(policy, config=None, seed=None, fps=30.0):
    """Watch ``policy`` play. Blocks until the window is closed.

    ``policy(game, valid_actions) -> (rotation, x)``, called exactly once per
    piece, *before* that piece's animation starts, so a slow (neural net) policy
    stalls one frame per placement instead of stalling mid-animation.

    Controls: SPACE pause/resume, UP/DOWN faster/slower, S steps one piece while
    paused, R restart, N new game with a new seed, ESC or Q quit. The HUD shows
    lines, score, level, pieces placed, speed and the paused/game-over state.
    """
    layout = Layout.from_config(config)
    _run_window(layout, 'Tetris RL - agent',
                lambda screen: _agent_loop(screen, layout, config, seed, policy, fps))


def _agent_loop(screen, layout, config, seed, policy, fps):
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    fps = max(5.0, min(240.0, float(fps)))
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
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP and speed_up.press():
                    player.apply_speed(SPEED_STEP)
                elif event.key == pygame.K_DOWN and speed_down.press():
                    player.apply_speed(1.0 / SPEED_STEP)
                elif event.key == pygame.K_s and paused:
                    player.step()
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
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
                      status=AGENT_KEYS)
        pygame.display.flip()


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


def run_menu(config=None):
    """Title screen: (1) Watch AI, (2) Play yourself, (3) Quit.

    Selectable by number key or arrow + Enter. "Watch AI" first picks a policy and
    a seed, then hands off to ``run_agent``; a missing or broken policies module
    shows an in-window message instead of a traceback.
    """
    layout = Layout.from_config(config)
    choice = _run_window(layout, 'Tetris RL', lambda screen: _menu_loop(screen, layout))
    if choice is None:
        return
    action, payload = choice
    if action == 'agent':
        run_agent(payload[0], config=config, seed=payload[1])
    else:
        run_human(config=config)


def _menu_loop(screen, layout):
    """Run the menu until it produces a choice.

    Returns ``('agent', (policy, seed))``, ``('human', None)`` or ``None`` to quit.
    The policy is loaded here (possibly slowly) so a failure can still be shown
    inside the window rather than killing the process.
    """
    renderer = Renderer(screen, layout)
    clock = pygame.time.Clock()
    items = ['Watch AI', 'Play yourself', 'Quit']
    state, selected, message, seed_text, names = 'main', 0, None, '', []

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
                    selected = (selected + step) % 3
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    selected, activate = event.key - pygame.K_1, True
                if not activate:
                    continue
                if selected == 2:
                    return None
                if selected == 1:
                    return ('human', None)
                module, error = _load_policies()
                names, error = ([], error) if error else _policy_names(module)
                state, selected, message = 'policies', 0, error
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
                               'up/down + ENTER, or press 1-3\nESC quits')
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
