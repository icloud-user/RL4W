"""
Modern Tetris (single-file) in Python using pygame
Features included:
- Seven-bag randomizer
- SRS rotation system with wall kicks
- Soft drop and hard drop
- Hold piece
- Ghost piece
- Lock delay and line clear
- Scoring, levels, back-to-back, combos
- Basic T-spin detection (standard rules)

Controls (default):
- Left / Right arrows: Move
- Up arrow / X / W / Z: Rotate clockwise
- Shift / C / Up: Rotate counter-clockwise (also Z)
- Down arrow: Soft drop
- Space: Hard drop
- Left Shift: Hold
- P: Pause
- Esc / Q: Quit

Requirements: pip install pygame
Run: python modern_tetris.py
"""

import pygame
import random
import sys
from collections import deque

# -------------------------------
# Config / Constants
# -------------------------------
CELL = 30
COLS = 10
ROWS = 22  # include invisible rows at top (standard is 20 visible + 2 buffer)
VISIBLE_ROWS = 20
WIDTH = CELL * COLS
HEIGHT = CELL * VISIBLE_ROWS
SIDE_PANEL = 200
FPS = 60

# timings (in frames at 60fps)
ARE_FRAMES = 10  # spawn delay (ARE)
# Lock delay in frames: how many frames a piece may rest before locking.
# Modern guideline: ~30-60 frames (0.5-1s at 60fps). Tune as desired.
LOCK_DELAY = 2  # lock delay in frames (tunable)
# If True, successful move/rotation/soft-drop while piece is resting resets lock timer.
LOCK_DELAY_RESET_ON_MOVE = False
DAS_INITIAL = 5.9  # initial DAS (frames). fractional values allowed.
DAS_REPEAT = 0     # ARR (auto-repeat rate) in frames; 0 = instant repeat (every frame)
SOFT_DROP_SPEED = 1  # legacy: not used directly here
# When holding soft-drop, drop once every N frames. Lower = faster.
# Set to 0 for infinite-speed soft drop (instant drop to contact when pressed).
SOFT_DROP_INTERVAL = 0
GRAVITY_LEVELS = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]  # frames per cell (approx)

# Colors
COLORS = {
    'I': (80, 220, 255),
    'J': (40, 40, 255),
    'L': (255, 160, 40),
    'O': (240, 240, 60),
    'S': (80, 255, 80),
    'T': (200, 80, 255),
    'Z': (255, 80, 80),
    'X': (120, 120, 120),  # for garbage/outline
}

# SRS Tetromino shapes (matrix of coordinates for rotation state 0)
# We'll store tetromino blocks for each rotation using standard reference
TETROMINO_BLOCKS = {
    'I': [(-1,0),(0,0),(1,0),(2,0)],
    'J': [(-1,-1),(-1,0),(0,0),(1,0)],
    'L': [(-1,0),(0,0),(1,0),(1,-1)],
    'O': [(0,0),(1,0),(0,-1),(1,-1)],
    'S': [(-1,0),(0,0),(0,-1),(1,-1)],
    'T': [(-1,0),(0,0),(1,0),(0,-1)],
    'Z': [(-1,-1),(0,-1),(0,0),(1,0)],
}

# SRS+ kick tables for non-I pieces (expanded for advanced setups like L-spin triples, J-spin triples, etc)
# key: (from, to) rotation indexes among 0,1,2,3
# These are the full SRS+ tables with comprehensive kicks
KICKS = {
    # 0 -> 1 (CW)
    (0,1): [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
    # 1 -> 0 (CCW from 1)
    (1,0): [(0,0),(1,0),(1,-1),(0,2),(1,2)],
    # 1 -> 2 (CW)
    (1,2): [(0,0),(1,0),(1,-1),(0,2),(1,2)],
    # 2 -> 1 (CCW from 2)
    (2,1): [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
    # 2 -> 3 (CW)
    (2,3): [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
    # 3 -> 2 (CCW from 3)
    (3,2): [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
    # 3 -> 0 (CW)
    (3,0): [(0,0),(-1,0),(-1,-1),(0,+2),(-1,2)],
    # 0 -> 3 (CCW from 0)
    (0,3): [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
}

# I-piece kicks - SRS+ expanded (I has different kick table due to its shape)
IKICKS = {
    (0,1): [(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
    (1,0): [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
    (1,2): [(0,0),(-1,0),(2,0),(-1,2),(2,-1)],
    (2,1): [(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
    (2,3): [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
    (3,2): [(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
    (3,0): [(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
    (0,3): [(0,0),(-1,0),(2,0),(-1,2),(2,-1)],
}

# 180-degree kick candidates (full SRS+ style with extensive offsets for advanced setups)
# Keys are (from,to) where to = (from+2)%4.
KICKS_180 = {}
IKICKS_180 = {}
_common_180 = [(0,0),(-1,0),(1,0),(0,1),(0,-1),(-1,1),(1,1),(-1,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,1),(2,-1),(-2,-1),(3,0),(-3,0),(0,3),(0,-3),(1,2),(-1,2),(1,-2),(-1,-2),(2,2),(-2,2),(2,-2),(-2,-2)]
for a in range(4):
    b = (a+2) % 4
    KICKS_180[(a,b)] = list(_common_180)
    IKICKS_180[(a,b)] = list(_common_180)

# rotation states are 0,1,2,3 clockwise

# -------------------------------
# Utilities
# -------------------------------

def rotate_point(x, y, r):
    # rotate (x,y) around origin r times 90deg clockwise
    for _ in range(r % 4):
        x, y = y, -x
    return x, y

# -------------------------------
# Game Classes
# -------------------------------

class Tetromino:
    def __init__(self, kind):
        self.kind = kind
        self.rotation = 0
        # spawn position (x,y) - use standard spawn near top-center
        self.x = 4
        self.y = 1  # y grows downward; some pieces need negative start - we included buffer rows
        self.blocks = TETROMINO_BLOCKS[kind]

    def get_cells(self, rot=None, xoff=None, yoff=None):
        rot = self.rotation if rot is None else rot
        xoff = self.x if xoff is None else xoff
        yoff = self.y if yoff is None else yoff
        cells = []
        for bx, by in self.blocks:
            rx, ry = rotate_point(bx, by, rot)
            cells.append((xoff + rx, yoff + ry))
        return cells

    def rotate(self, dir):
        # dir = +1 clockwise, -1 counter
        old = self.rotation
        self.rotation = (self.rotation + dir) % 4
        return old, self.rotation


class Bag:
    def __init__(self):
        self.q = deque()
        self._refill()

    def _refill(self):
        pieces = list(TETROMINO_BLOCKS.keys())
        random.shuffle(pieces)
        self.q.extend(pieces)

    def next(self):
        # Only refill when the queue is empty so each 7-piece bag remains intact.
        if len(self.q) == 0:
            self._refill()
        return Tetromino(self.q.popleft())


class Board:
    def __init__(self):
        self.grid = [[None for _ in range(COLS)] for _ in range(ROWS)]

    def inside(self, x, y):
        return 0 <= x < COLS and y < ROWS

    def cell(self, x, y):
        if not self.inside(x, y):
            return None
        return self.grid[y][x]

    def set_cell(self, x, y, val):
        if 0 <= y < ROWS and 0 <= x < COLS:
            self.grid[y][x] = val

    def collide(self, cells):
        for x, y in cells:
            if x < 0 or x >= COLS or y >= ROWS:
                return True
            if y >= 0 and self.grid[y][x] is not None:
                return True
        return False

    def lock(self, tetromino):
        for x, y in tetromino.get_cells():
            if 0 <= y < ROWS:
                self.grid[y][x] = tetromino.kind

    def clear_lines(self):
        cleared = 0
        new = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = ROWS - len(new)
        while len(new) < ROWS:
            new.insert(0, [None for _ in range(COLS)])
        self.grid = new
        return cleared

    def is_empty_at_spawn(self, tetromino):
        return not self.collide(tetromino.get_cells())

    def get_top_heights(self):
        heights = [0]*COLS
        for c in range(COLS):
            h = 0
            for r in range(ROWS):
                if self.grid[r][c] is not None:
                    h = ROWS - r
                    break
            heights[c] = h
        return heights

# -------------------------------
# Game State
# -------------------------------

class Game:
    def __init__(self):
        self.board = Board()
        self.bag = Bag()
        self.next_queue = deque()
        for _ in range(5):
            self.next_queue.append(self.bag.next())
        # Spawn the initial current piece from the pre-filled next_queue.
        # `spawn_next()` will pop one entry from `next_queue` and append a new
        # piece from the bag, so we should not pop here (that caused the bag
        # order to shift and overlap). Just call `spawn_next()` to set `current`.
        self.spawn_next()
        self.hold_piece = None
        self.hold_used = False
        self.level = 0
        self.score = 0
        self.lines = 0
        self.combo = -1
        self.back_to_back = False

        self.gravity_timer = 0
        self.gravity_frames = GRAVITY_LEVELS[min(self.level, len(GRAVITY_LEVELS)-1)]
        self.lock_delay = 0
        self.locked = False
        self.are = 0
        self.game_over = False

        # input handling
        self.left_held = False
        self.right_held = False
        self.das_dir = 0
        self.das_timer = 0
        # soft-drop hold handling
        self.down_held = False
        self.soft_drop_timer = 0

    def spawn_next(self):
        # place current as next_queue[0] and spawn
        self.current = self.next_queue.popleft()
        self.next_queue.append(self.bag.next())
        self.current.x = 4
        self.current.y = 0
        self.current.rotation = 0
        if self.board.collide(self.current.get_cells()):
            # game over
            self.game_over = True

    def hold(self):
        if self.hold_used:
            return
        if self.hold_piece is None:
            # store current piece in hold
            self.hold_piece = Tetromino(self.current.kind)
            self.spawn_next()
        else:
            # swap: save current kind, restore held kind
            current_kind = self.current.kind
            self.current = Tetromino(self.hold_piece.kind)
            self.hold_piece = Tetromino(current_kind)
            self.current.x = 4
            self.current.y = 0
            self.current.rotation = 0
            if self.board.collide(self.current.get_cells()):
                self.game_over = True
        self.hold_used = True
        self.lock_delay = 0
        self.gravity_timer = 0
        # Reset input states to prevent input ghosting
        self.left_held = False
        self.right_held = False
        self.down_held = False
        self.das_dir = 0
        self.das_timer = 0
        self.soft_drop_timer = 0

    def hard_drop(self):
        # drop to lowest possible
        while True:
            cells = self.current.get_cells(yoff=self.current.y+1)
            if self.board.collide(cells):
                break
            self.current.y += 1
        self.board.lock(self.current)
        cleared = self.board.clear_lines()
        self.after_lock(cleared, hard=True)

    def soft_drop(self):
        # move down one if possible
        cells = self.current.get_cells(yoff=self.current.y+1)
        if not self.board.collide(cells):
            self.current.y += 1
            self.score += 1  # standard soft drop scoring 1pt per cell
            # Reset lock delay when soft-dropping into a new position so that
            # soft-drop + move doesn't cause an immediate lock (like modern Tetris).
            if LOCK_DELAY_RESET_ON_MOVE:
                self.lock_delay = 0
            return True
        return False

    def try_move(self, dx):
        if self.board.collide(self.current.get_cells(xoff=self.current.x+dx)):
            return False
        self.current.x += dx
        # Successful horizontal movement should reset lock delay so the player
        # has time to react after moving while the piece is resting.
        if LOCK_DELAY_RESET_ON_MOVE:
            self.lock_delay = 0
        return True

    def try_rotate(self, dir):
        # support dir = +1 (90 cw), -1 (90 ccw), +2/-2 (180)
        old = self.current.rotation
        new = (old + (dir % 4)) % 4
        # apply tentative rotation on the tetromino object (we'll revert if it fails)
        self.current.rotation = new

        # choose appropriate kick table
        # if abs(dir) == 2:
        #     kicks = IKICKS_180 if self.current.kind == 'I' else KICKS_180
        # else:
        kicks = IKICKS if self.current.kind == 'I' else KICKS

        for ox, oy in kicks.get((old, new), [(0,0)]):
            nx = self.current.x + ox
            ny = self.current.y + oy
            if not self.board.collide(self.current.get_cells(rot=new, xoff=nx, yoff=ny)):
                # apply
                self.current.x = nx
                self.current.y = ny
                self.current.rotation = new
                # rotation counts as move for lock delay
                if LOCK_DELAY_RESET_ON_MOVE:
                    self.lock_delay = 0
                return True

        # no valid kick found; revert rotation
        self.current.rotation = old
        return False

    def gravity_step(self):
        cells = self.current.get_cells(yoff=self.current.y+1)
        if not self.board.collide(cells):
            self.current.y += 1
            self.lock_delay = 0
            return False
        else:
            # piece rests on ground or blocks
            # increment lock timer while piece is resting
            self.lock_delay += 1
            if self.lock_delay >= LOCK_DELAY:
                # lock piece
                self.board.lock(self.current)
                cleared = self.board.clear_lines()
                self.after_lock(cleared)
                return True
        return False

    def after_lock(self, cleared, hard=False):
        # scoring & state updates
        is_tspin = self.detect_tspin(self.current) if self.current.kind == 'T' else False
        if is_tspin:
            # basic tspin scoring (simple)
            if cleared == 1:
                base = 800
            elif cleared == 2:
                base = 1200
            elif cleared == 3:
                base = 1600
            else:
                base = 400
        else:
            base = {0:0,1:100,2:300,3:500,4:800}.get(cleared, 0)
        # apply back-to-back
        b2b_bonus = 1.5 if (self.back_to_back and (cleared==4 or is_tspin and cleared>0)) else 1
        self.score += int(base * b2b_bonus)
        if hard:
            self.score += 2 * (ROWS - self.current.y)
        # soft drop scoring already added incrementally
        self.lines += cleared
        if cleared > 0:
            self.combo = self.combo + 1 if self.combo >=0 else 0
            if cleared == 4 or (is_tspin and cleared>0):
                if self.back_to_back:
                    self.score += 50  # small b2b bonus
                self.back_to_back = True
            else:
                self.back_to_back = False
        else:
            self.combo = -1
        # level up every 10 lines
        self.level = self.lines // 10
        self.gravity_frames = GRAVITY_LEVELS[min(self.level, len(GRAVITY_LEVELS)-1)]
        # spawn next
        self.hold_used = False
        self.lock_delay = 0
        self.spawn_next()

    def detect_tspin(self, tetromino):
        # naive T-spin detection: last action was rotation and three of four corners around T are blocked
        if tetromino.kind != 'T':
            return False
        cx = tetromino.x
        cy = tetromino.y
        # corners relative to center
        corners = [(-1,-1),(1,-1),(-1,1),(1,1)]
        blocked = 0
        for dx, dy in corners:
            x = cx + dx
            y = cy + dy
            if x < 0 or x >= COLS or y < 0 or y >= ROWS:
                blocked += 1
            elif self.board.cell(x,y) is not None:
                blocked += 1
        return blocked >= 3

# -------------------------------
# Pygame Rendering and Main Loop
# -------------------------------

def draw_cell(surface, x, y, color, alpha=255, outline=True):
    rect = pygame.Rect(x*CELL, y*CELL - (ROWS - VISIBLE_ROWS)*CELL, CELL, CELL)
    s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
    s.fill((*color, alpha))
    surface.blit(s, rect.topleft)
    if outline:
        pygame.draw.rect(surface, (10,10,10), rect, 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH + SIDE_PANEL, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 18)
    bigfont = pygame.font.SysFont('Consolas', 28)

    game = Game()
    gravity_counter = 0

    running = True
    paused = False

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_p:
                    paused = not paused
                if paused:
                    continue
                if event.key == pygame.K_LEFT:
                    game.try_move(-1)
                    game.left_held = True
                    game.das_dir = -1
                    game.das_timer = DAS_INITIAL
                if event.key == pygame.K_RIGHT:
                    game.try_move(1)
                    game.right_held = True
                    game.das_dir = 1
                    game.das_timer = DAS_INITIAL
                if event.key == pygame.K_DOWN:
                    # start holding soft-drop
                    game.down_held = True
                    game.soft_drop_timer = 0
                    # If interval==0 treat soft-drop as infinite speed: drop to contact immediately
                    if SOFT_DROP_INTERVAL == 0:
                        # repeatedly soft-drop until we can't (this won't lock the piece)
                        while game.soft_drop():
                            pass
                    else:
                        game.soft_drop()
                if event.key == pygame.K_SPACE:
                    game.hard_drop()
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_c:
                    game.hold()
                if event.key in (pygame.K_z, pygame.K_LCTRL):
                    game.try_rotate(1)
                if event.key in (pygame.K_x, pygame.K_UP):
                    game.try_rotate(-1)
                if event.key == pygame.K_a:
                    # rotate 180 degrees (SRS+ style)
                    game.try_rotate(2)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    game.left_held = False
                    if game.right_held:
                        game.das_dir = 1
                        game.das_timer = DAS_INITIAL
                    else:
                        game.das_dir = 0
                if event.key == pygame.K_RIGHT:
                    game.right_held = False
                    if game.left_held:
                        game.das_dir = -1
                        game.das_timer = DAS_INITIAL
                    else:
                        game.das_dir = 0
                if event.key == pygame.K_DOWN:
                    game.down_held = False
                    game.soft_drop_timer = 0

        if paused or game.game_over:
            screen.fill((8,8,8))
            txt = 'PAUSED' if paused else 'GAME OVER'
            t = bigfont.render(txt, True, (240,240,240))
            screen.blit(t, (20, 20))
            pygame.display.flip()
            continue

        # apply DAS
        if game.das_dir != 0:
            game.das_timer -= 1
            if game.das_timer <= 0:
                game.try_move(game.das_dir)
                game.das_timer = DAS_REPEAT

        # gravity
        # handle soft-drop hold: call soft_drop at an interval while key is held
        if game.down_held and SOFT_DROP_INTERVAL > 0:
            game.soft_drop_timer += 1
            if game.soft_drop_timer >= SOFT_DROP_INTERVAL:
                game.soft_drop_timer = 0
                game.soft_drop()

        gravity_counter += 1
        if gravity_counter >= game.gravity_frames:
            gravity_counter = 0
            game.gravity_step()

        # render
        screen.fill((8,8,8))

        # draw board background
        board_surf = pygame.Surface((WIDTH, HEIGHT))
        board_surf.fill((18,18,18))

        # draw locked cells
        for y in range(ROWS - VISIBLE_ROWS, ROWS):
            for x in range(COLS):
                cell = game.board.grid[y][x]
                if cell is not None:
                    draw_cell(board_surf, x, y, COLORS.get(cell, (200,200,200)))

        # ghost piece
        ghost = Tetromino(game.current.kind)
        ghost.x = game.current.x
        ghost.y = game.current.y
        ghost.rotation = game.current.rotation
        while not game.board.collide(ghost.get_cells(yoff=ghost.y+1)):
            ghost.y += 1
        for x, y in ghost.get_cells():
            if y >= ROWS - VISIBLE_ROWS:
                draw_cell(board_surf, x, y, COLORS.get(game.current.kind, (200,200,200)), alpha=150, outline=False)

        # current piece
        for x, y in game.current.get_cells():
            if y >= ROWS - VISIBLE_ROWS:
                draw_cell(board_surf, x, y, COLORS.get(game.current.kind, (200,200,200)))

        screen.blit(board_surf, (0,0))

        # side panel
        panel_x = WIDTH + 10
        # next queue
        screen.blit(font.render('Next:', True, (220,220,220)), (panel_x, 10))
        for i, nxt in enumerate(list(game.next_queue)[:5]):
            px = panel_x
            py = 40 + i*60
            # draw a small representation
            for bx, by in nxt.blocks:
                rx, ry = rotate_point(bx, by, 0)
                rect = pygame.Rect(px + (rx+1)*10, py + (ry+1)*10, 10, 10)
                pygame.draw.rect(screen, COLORS[nxt.kind], rect)
                pygame.draw.rect(screen, (10,10,10), rect,1)

        # hold
        screen.blit(font.render('Hold:', True, (220,220,220)), (panel_x, 340))
        if game.hold_piece:
            hp = game.hold_piece
            px = panel_x
            py = 360
            for bx, by in hp.blocks:
                rx, ry = rotate_point(bx, by, 0)
                rect = pygame.Rect(px + (rx+1)*10, py + (ry+1)*10, 10, 10)
                pygame.draw.rect(screen, COLORS[hp.kind], rect)
                pygame.draw.rect(screen, (10,10,10), rect,1)

        # stats
        screen.blit(font.render(f'Score: {game.score}', True, (220,220,220)), (panel_x, 460))
        screen.blit(font.render(f'Lines: {game.lines}', True, (220,220,220)), (panel_x, 490))
        screen.blit(font.render(f'Level: {game.level}', True, (220,220,220)), (panel_x, 520))
        screen.blit(font.render('Controls: ← → move  ↓ soft  space hard  z/x rotate  shift hold', True, (180,180,180)), (10, HEIGHT-30))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()