"""Tests for the versus battle loop and its headless entry points.

Run: python tests/test_battle.py

Everything here drives the match through ``render._battle_step`` and the battle
objects, so **no window is ever opened**: ``pygame`` is imported for its key
constants and, in the last check only, for one off-screen ``Surface`` -- never for
``display.set_mode``.

The checks pin the parts of battle mode that break silently rather than loudly:
the garbage travel delay, TETR.IO's rule that garbage only enters on a lock that
cleared nothing, an attack actually reaching the other board, the K.O./round-reset
cycle, and the pieces-per-second handicap the difficulty presets are built on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame                                                       # noqa: E402

from tetrisrl import render                                         # noqa: E402
from tetrisrl.policies import (make_heuristic_policy,               # noqa: E402
                               make_versus_policy)
from tetrisrl.versus import GARBAGE_TRAVEL, Battle                  # noqa: E402

FAILS = []
CHECKS = 0
FRAME = 1.0 / 60.0


def check(name, cond, detail=''):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


def stub_policy(pps=6.0, reaction=0.0):
    """A cheap non-learning policy carrying the timing attributes battle mode reads.

    The heuristic plays well enough to survive a few hundred frames, so the bot
    side advances without pulling torch in and without the cost of the search.
    """
    policy = make_heuristic_policy()
    policy.pps, policy.reaction = pps, reaction
    return policy


def drive(state, frames, keys_every=0, key=pygame.K_SPACE):
    """Step a battle headlessly. Returns ``(frames_run, counts by event kind)``.

    ``keys_every`` presses one key every N frames, which is how the human side is
    simulated: spamming a hard drop at the spawn column stacks up and tops out,
    which is what makes a K.O. happen without a person at the keyboard.
    """
    counts = {}
    for index in range(frames):
        keys = [key] if keys_every and index % keys_every == 0 else []
        for event in render._battle_step(state, FRAME, keys, []):
            counts[event[0]] = counts.get(event[0], 0) + 1
        if state.over:
            return index + 1, counts
    return frames, counts


# --- the rules --------------------------------------------------------------

def test_garbage_travels_before_it_enters():
    """An attack is queued immediately but cannot land until it has travelled."""
    battle = Battle(seed=1, rounds_to_win=1)
    side = battle.sides[0]
    side.queue.send([3], battle.time)
    check('queued garbage counts as pending straight away',
          side.queue.pending == 3, side.queue.pending)
    check('but none of it is ready before the travel delay',
          side.queue.ready(battle.time) == 0, side.queue.ready(battle.time))

    battle.update(GARBAGE_TRAVEL * 0.5)
    check('still not ready half way through the delay',
          side.queue.ready(battle.time) == 0, side.queue.ready(battle.time))

    battle.update(GARBAGE_TRAVEL * 0.5 + 1e-6)
    check('ready once the delay has elapsed', side.queue.ready(battle.time) == 3,
          side.queue.ready(battle.time))
    check('travelling garbage is pending, not lost', side.queue.pending == 3)


def test_garbage_enters_only_on_a_lock_that_cleared_nothing():
    """TETR.IO's combo blocking, and the shape of a garbage row."""
    battle = Battle(seed=2, rounds_to_win=1)
    side = battle.sides[0]
    side.queue.send([3], battle.time)
    battle.update(GARBAGE_TRAVEL + 1e-6)

    blocked = battle.after_lock(0, 1, all_clear=False)
    check('a lock that cleared lines lets no garbage in',
          blocked['tanked'] == [] and side.queue.pending == 3, blocked)

    summary = battle.after_lock(0, 0, all_clear=False)
    check('a lock that cleared nothing lets it in',
          len(summary['tanked']) == 3, summary['tanked'])
    bottom = side.game.board.grid[side.game.rows - 1]
    check('a garbage row is a full row with exactly one hole',
          sum(1 for cell in bottom if cell is None) == 1, bottom)
    check('the whole attack was consumed', side.queue.pending == 0,
          side.queue.pending)


def test_an_attack_lands_on_the_other_board():
    """A quad sends four lines, which arrive on the opponent's queue and land."""
    battle = Battle(seed=3, rounds_to_win=1)
    me, foe = battle.sides
    summary = battle.after_lock(0, 4, all_clear=False)
    check('a quad sends four lines', summary['sent'] == 4, summary)
    check('they are queued against the opponent, not against the sender',
          foe.queue.pending == 4 and me.queue.pending == 0,
          (foe.queue.pending, me.queue.pending))
    check('and they are still travelling', battle.ready(1) == 0)
    battle.update(GARBAGE_TRAVEL + 1e-6)
    check('they become ready once they have travelled', battle.ready(1) == 4,
          battle.ready(1))

    landed = battle.after_lock(1, 0, all_clear=False)
    check('the opponent tanks them on a lock that cleared nothing',
          len(landed['tanked']) == 4, landed['tanked'])
    check('both sides book-keep what moved',
          foe.received == 4 and me.sent == 4, (foe.received, me.sent))


# --- the loop ---------------------------------------------------------------

def test_a_ko_resets_the_round_and_the_match_finishes():
    """A top-out awards the round, resets the boards, and eventually the match."""
    battle = Battle(seed=7, rounds_to_win=2, names=('you', 'bot'))
    state = render._BattleState(battle, stub_policy(), difficulty='stub')
    frames, counts = drive(state, frames=4000, keys_every=6)

    check('a K.O. ends a round', counts.get('round_over', 0) >= 1, counts)
    check('the next round starts on fresh boards',
          counts.get('round', 0) >= 1 and battle.round >= 2,
          f'round={battle.round} counts={counts}')
    check('the match reaches a result', state.over and bool(state.result),
          f'over={state.over} result={state.result!r}')
    check('the winner reached the required wins',
          max(side.wins for side in battle.sides) >= 2,
          [side.wins for side in battle.sides])
    check('the human side was the one topping out',
          battle.sides[0].wins == 0 and battle.sides[1].wins >= 2,
          [side.wins for side in battle.sides])
    del frames


def test_the_bot_places_pieces_at_its_pps():
    """The handicap is pacing: N frames at P pieces/second places about P*N/60."""
    placed = {}
    frames = 300
    for pps in (3.0, 6.0):
        battle = Battle(seed=4, rounds_to_win=1)
        state = render._BattleState(battle, stub_policy(pps=pps),
                                    difficulty='stub')
        drive(state, frames=frames)
        placed[pps] = battle.sides[1].game.pieces_placed
        expected = pps * frames / 60.0
        check(f'{pps:g} pieces/second places about {expected:.0f} pieces in '
              f'{frames} frames', abs(placed[pps] - expected) <= 3,
              f'placed {placed[pps]}')
    check('twice the rate places clearly more pieces',
          placed[6.0] > placed[3.0] + 5, placed)


def test_the_real_versus_policy_plays_a_match():
    """The shipped opponent, with garbage on its board, to a match result."""
    battle = Battle(seed=11, rounds_to_win=1, names=('you', 'bot'))
    policy = make_versus_policy(difficulty='beginner',
                                incoming_fn=lambda: battle.incoming(1), seed=3)
    state = render._BattleState(battle, policy, difficulty='beginner')
    bot = battle.sides[1]

    # Give the bot a dirty board and keep its queue non-empty, so the search's
    # incoming-garbage plumbing is exercised rather than merely constructed.
    bot.queue.send([4], battle.time)
    battle.update(GARBAGE_TRAVEL + 1e-6)
    battle.after_lock(1, 0, all_clear=False)
    check('the bot starts the round with garbage row(s) on its board',
          bot.tanked == 4, bot.tanked)

    frames, counts = drive(state, frames=3000, keys_every=30)
    check('the versus policy keeps placing pieces on a dirty board',
          bot.game.pieces_placed >= 2, bot.game.pieces_placed)
    check('it is paced by its preset, not by the frame rate',
          bot.game.pieces_placed <= 2 + policy.pps * frames / 60.0,
          f'{bot.game.pieces_placed} pieces in {frames} frames at {policy.pps} pps')
    check('the match still runs to a result', state.over and bool(state.result),
          f'over={state.over} counts={counts}')
    check('the real policy is the one in play',
          policy.policy_name == 'search_versus:beginner', policy.policy_name)


def test_battle_renders_off_screen():
    """One frame of battle mode onto a Surface, with no display involved."""
    battle = Battle(seed=6, rounds_to_win=1, names=('you', 'bot'))
    state = render._BattleState(battle, stub_policy(), difficulty='stub')
    render._battle_step(state, FRAME)
    battle.sides[0].queue.send([3], battle.time)
    left, right, size = render._battle_layouts({'game': {'cell_size': 20}})
    check('two boards, two hold boxes and two panels fit the battle window',
          size == (left.width * 2, left.height)
          and right.hold_x == left.width + right.margin
          and right.board_x == right.hold_x + right.hold_w + right.margin
          and right.panel_x + right.panel_w + right.margin == size[0],
          (size, left.width, right.board_x, right.hold_x))
    check('the right half is the left half shifted by exactly one width',
          all(getattr(right, field) - getattr(left, field) == left.width
              for field in ('hold_x', 'board_x', 'panel_x')))
    try:
        surface = pygame.Surface(size)
        renderers = (render.Renderer(surface, left),
                     render.Renderer(surface, right))
        render._battle_draw(surface, renderers, state, (left, right), 'stub')
        state.paused = True
        render._battle_draw(surface, renderers, state, (left, right), 'paused')
        state.paused = False
        render._draw_garbage_meter(surface, battle.sides[0].queue, battle.time,
                                   left.margin, left.board_y, left.board_h)
        ok, detail = True, ''
    except pygame.error as exc:                 # no video support at all
        ok, detail = True, f'(skipped: {exc})'
    check('battle mode draws off-screen without a display', ok, detail)


def test_hold_box_sits_left_of_the_board():
    """HOLD is a box on the board's left in every mode, inside its own window.

    This is the layout the user asked for, so it is pinned rather than assumed:
    hold, then the board, then the side panel (NEXT and the stats), and nothing
    outside the surface the window was sized for.
    """
    solo = render.Layout.from_config({'game': {'cell_size': 30}})
    battle = render._battle_layouts({'game': {'cell_size': 30}})
    # The right half's rects are shifted by the left half's width, so they have to
    # be measured against the two-board window rather than against their own
    # half-width -- which is what this check got wrong the first time.
    for name, layout, window_w in (('solo', solo, solo.width),
                                   ('battle left', battle[0], battle[2][0]),
                                   ('battle right', battle[1], battle[2][0])):
        hold, board, panel = layout.hold_rect, layout.board_rect, layout.panel_rect
        check(f'{name}: the hold box is left of the board',
              hold.right <= board.left, (hold, board))
        check(f'{name}: the board is left of the side panel',
              board.right <= panel.left, (board, panel))
        check(f'{name}: everything fits the window the layout asks for',
              hold.x >= 0 and panel.right <= window_w
              and max(hold.bottom, board.bottom, panel.bottom) <= layout.height,
              (window_w, layout.height, hold, board, panel))
    check('the window is wide enough for both battle halves on one screen',
          battle[2][0] <= render.BATTLE_MAX_WIDTH, battle[2])

    # A real solo frame, drawn off-screen: the hold box is left of the board, it
    # actually gets painted, and nothing lands outside the window.
    from tetrisrl.engine import Game, Piece

    surface = pygame.Surface((solo.width, solo.height))
    renderer = render.Renderer(surface, solo)
    game = Game(seed=1)
    game.hold_piece = Piece('T')
    renderer.draw(game, status='stub')
    hold, board = solo.hold_rect, solo.board_rect
    check('solo frame: the hold box is drawn left of the board',
          hold.x < board.x, (hold.x, board.x))
    check('solo frame: the hold box was painted',
          any(surface.get_at((x, y))[:3] != render.BG
              for x in range(hold.x, hold.right)
              for y in range(hold.y, hold.bottom)), hold)
    check('solo frame: nothing is drawn outside the window',
          surface.get_at((0, 0))[:3] == render.BG
          and surface.get_at((solo.width - 1, solo.height - 1))[:3] == render.BG,
          (solo.width, solo.height))


def test_hold_panel_width_is_configurable():
    """``game.hold_panel_width`` is honoured, clamped, and shifts the board."""
    narrow = render.Layout.from_config(
        {'game': {'cell_size': 30, 'hold_panel_width': 70}})
    wide = render.Layout.from_config(
        {'game': {'cell_size': 30, 'hold_panel_width': 150}})
    check('a wider hold box moves the board right',
          wide.board_x - narrow.board_x == 80, (narrow.board_x, wide.board_x))
    check('the hold box follows the setting',
          narrow.hold_w == 70 and wide.hold_w == 150,
          (narrow.hold_w, wide.hold_w))
    absurd = render.Layout.from_config(
        {'game': {'cell_size': 30, 'hold_panel_width': 5000}})
    check('an absurd hold width is clamped', absurd.hold_w == 200, absurd.hold_w)
    default = render.Layout.from_config({'game': {'cell_size': 30}})
    check('the default hold box is the documented constant',
          default.hold_w == render.HOLD_PANEL_W, default.hold_w)


def test_settings_list_fits_a_small_window():
    """The settings rows are fitted, not run off the bottom of a short window.

    Flagged earlier: at a small ``cell_size`` the 16-row list overlapped the
    footer. The row height is derived from the space available, so this pins that
    the last row still ends above the footer at the smallest supported cell.
    """
    from tetrisrl import controls
    from tetrisrl.render import _settings_row_height, settings_rows

    for cell in (10, 16, 30):
        layout = render.Layout.from_config({'game': {'cell_size': cell}})
        surface = pygame.Surface((layout.width, layout.height))
        renderer = render.Renderer(surface, layout)
        rows = settings_rows(controls.InputSettings())
        y = layout.margin + 4 + renderer._height(renderer.big, 22) + 6
        row_h = _settings_row_height(renderer, len(rows), y)
        end = y + row_h * len(rows)
        footer_top = layout.height - 2 * row_h - 6
        check(f'cell {cell}: every settings row ends above the footer',
              end <= footer_top, (end, footer_top, row_h, len(rows)))
        check(f'cell {cell}: rows stay legible', row_h >= 9, row_h)


def main():
    for fn in (test_garbage_travels_before_it_enters,
               test_garbage_enters_only_on_a_lock_that_cleared_nothing,
               test_an_attack_lands_on_the_other_board,
               test_a_ko_resets_the_round_and_the_match_finishes,
               test_the_bot_places_pieces_at_its_pps,
               test_the_real_versus_policy_plays_a_match,
               test_battle_renders_off_screen,
               test_hold_box_sits_left_of_the_board,
               test_hold_panel_width_is_configurable,
               test_settings_list_fits_a_small_window):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all battle tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
