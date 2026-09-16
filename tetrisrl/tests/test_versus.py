"""Tests for the versus rules. Run: python tests/test_versus.py

The attack numbers are TETR.IO's, so these checks are the record of what the mode
actually implements: if a number here is wrong the mode is wrong, and the doc
comments in `versus.py` plus `docs/tetrio-versus-ruleset.md` say where each one
came from.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import versus as V                     # noqa: E402
from tetrisrl.engine import Game                     # noqa: E402
from tetrisrl.heuristic import apply_move            # noqa: E402

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


def fill(game, rows, cols=None):
    """Fill the given board rows completely."""
    for y in rows:
        for x in range(game.cols):
            game.board.grid[y][x] = 'X'


# --- attack -----------------------------------------------------------------

def test_attack_table():
    check('a single sends nothing', V.attack_for(1, combo=0).lines == 0)
    check('a double sends one', V.attack_for(2, combo=0).lines == 1)
    check('a triple sends two', V.attack_for(3, combo=0).lines == 2)
    check('a tetris sends four', V.attack_for(4, combo=0).lines == 4)
    check('five lines send five', V.attack_for(5, combo=0).lines == 5)
    check('six lines send six', V.attack_for(6, combo=0).lines == 6)
    check('an all clear sends five', V.attack_for(4, combo=0, all_clear=True).lines == 5)
    check('no clear sends nothing', V.attack_for(0, combo=5).lines == 0)


def test_combo():
    # base * (1 + 0.25 * combo), rounded down.
    check('a double at combo 1 sends 1', V.attack_for(2, combo=1).lines == 1)
    check('a double at combo 3 sends 1 (floor of 1.75)',
          V.attack_for(2, combo=3).lines == 1)
    check('a tetris at combo 4 sends 8', V.attack_for(4, combo=4).lines == 8)
    # A single has a zero base, so the log curve is what keeps a chain alive.
    check('a single in a combo sends nothing at combo 0',
          V.attack_for(1, combo=0).lines == 0)
    check('a single at combo 2 sends one', V.attack_for(1, combo=2).lines == 1)
    check('a single at combo 6 sends two', V.attack_for(1, combo=6).lines == 2)


def test_b2b_and_surge():
    a = V.attack_for(4, combo=0, b2b=1)
    check('B2B charging adds one to a tetris', a.lines == 5, a)
    check('and the streak grows', a.b2b == 2, a)
    a = V.attack_for(2, combo=0, b2b=3)
    check('a double breaks the B2B streak', a.b2b == 0, a)
    a = V.attack_for(4, combo=0, b2b=3)
    check('the fourth tetris in a row surges', a.b2b == 4, a)
    check('the tetris and the surge arrive as separate chunks',
          len(a.chunks) == 4, a)
    check('surge chunks sum to the attack', sum(a.chunks) == a.lines, a)
    # Surge pays the streak itself: 4 lines from a streak of 4, +1 per level. So a
    # quad is 4, charging adds 1, and the surge adds 4 more.
    check('surge pays the streak: 4 lines at a streak of 4', a.lines == 4 + 1 + 4, a)
    a = V.attack_for(4, combo=0, b2b=7)
    check('and 8 lines at a streak of 8 (the published example)',
          a.lines == 4 + 1 + 8, a)


# --- queue ------------------------------------------------------------------

def test_queue_travel_and_cancel():
    q = V.GarbageQueue()
    q.send([4], 0.0)
    check('garbage is not ready before it has travelled', q.ready(0.1) == 0)
    check('garbage is ready after 20 frames', q.ready(V.GARBAGE_TRAVEL + 0.001) == 4)
    check('the queue reports what is pending', q.pending == 4)
    check('cancelling spends the oldest first', q.cancel(3) == 0 and q.pending == 1)
    check('cancelling more than is queued returns the remainder',
          q.cancel(5) == 4 and q.pending == 0)


def test_queue_offsets_are_oldest_first():
    q = V.GarbageQueue()
    q.send([2], 0.0)
    q.send([3], 0.0)
    q.cancel(3)
    check('the first item is fully consumed, then the second',
          q.pending == 2, q.pending)


def test_cap_limits_entry():
    q = V.GarbageQueue(cap=8)
    q.send([20], 0.0)
    holes = q.take(1.0, random.Random(0), 'clean', 10)
    check('at most the cap enters on one lock', len(holes) == 8, len(holes))
    check('the rest stays queued', q.pending == 12, q.pending)


def test_messiness():
    rng = random.Random(0)
    q = V.GarbageQueue()
    q.send([6], 0.0)
    clean = q.take(1.0, rng, 'clean', 10)
    check('clean garbage keeps one hole column', len(set(clean)) == 1, clean)
    q2 = V.GarbageQueue()
    q2.send([6], 0.0)
    chaotic = q2.take(1.0, random.Random(0), 'chaotic', 10)
    check('chaotic garbage re-rolls every line', len(set(chaotic)) > 1, chaotic)
    check('holes stay inside the board', all(0 <= h < 10 for h in chaotic))
    q3 = V.GarbageQueue()
    q3.send([6], 0.0)
    messy = q3.take(1.0, random.Random(3), 'messy', 10)
    check('messy garbage is between the two', 1 <= len(set(messy)) <= 6, messy)


# --- garbage on the board ---------------------------------------------------

def test_garbage_raises_the_stack():
    game = Game(seed=0)
    game.board.grid[21][0] = 'X'
    game.add_garbage(2, 5)
    heights = game.board.column_heights()
    check('garbage adds rows at the bottom', heights[0] == 3, heights)
    check('the hole column stays open', heights[5] == 0, heights)
    check('an empty row remains where the stack was', heights[9] == 2, heights)


def test_garbage_top_out():
    game = Game(seed=0)
    fill(game, range(0, 22))
    check('a full board tops out when garbage arrives', game.add_garbage(1, 3))
    game2 = Game(seed=0)
    fill(game2, range(4, 22))
    check('a stack with room above it survives', not game2.add_garbage(1, 3))
    game3 = Game(seed=0)
    fill(game3, range(2, 22))
    check('cells pushed off the top edge end the game', game3.add_garbage(4, 3))


def test_garbage_smash():
    """A piece caught inside the risen stack ends the game, as in TETR.IO."""
    game = Game(seed=0)
    fill(game, range(18, 22))
    game.current.y = 19                      # pretend the piece is down in the stack
    check('a piece buried by the rise tops out', game.add_garbage(2, 3))
    game2 = Game(seed=0)
    check('a piece well above the stack is fine', not game2.add_garbage(1, 3))


def test_no_lockout_in_versus():
    game = Game(seed=0)
    game.nolockout = True
    fill(game, range(0, 2))                  # a piece locking here would lock out
    game.current.y = 0
    game.lock()
    check('nolockout keeps the game alive', not game.game_over)
    solo = Game(seed=0)
    fill(solo, range(0, 2))
    solo.current.y = 0
    solo.lock()
    check('the solo default still ends on a lock out', solo.game_over)


# --- the match --------------------------------------------------------------

def test_battle_tanks_only_on_a_clean_lock():
    battle = V.Battle(seed=0)
    side = battle.sides[0]
    side.queue.send([4], battle.time)
    battle.update(1.0)
    # all_clear is stated explicitly because these boards are empty by
    # construction, not because a real all clear happened.
    battle.after_lock(0, 2, all_clear=False)   # a double: 1 line of attack
    check('a clearing lock blocks garbage', battle.ready(0) == 3,
          battle.ready(0))
    check('and cancels what it can', side.queue.pending == 3, side.queue.pending)
    check('nothing is sent on when the attack is spent cancelling',
          battle.sides[1].queue.pending == 0)


def test_battle_cancel_and_send():
    battle = V.Battle(seed=0)
    side = battle.sides[0]
    side.queue.send([2], battle.time)
    battle.update(1.0)
    out = battle.after_lock(0, 4, all_clear=False)   # a tetris: 4 lines
    check('attack cancels incoming first', out['cancelled'] == 2, out)
    check('only the remainder is sent', out['sent'] == 2, out)
    check('and it lands on the opponent',
          battle.sides[1].queue.pending == 2, battle.sides[1].queue.pending)


def test_battle_takes_garbage_on_a_dry_lock():
    battle = V.Battle(seed=0, messiness='clean')
    side = battle.sides[0]
    side.queue.send([3], battle.time)
    battle.update(1.0)
    before = side.game.board.column_heights()
    out = battle.after_lock(0, 0)
    after = side.game.board.column_heights()
    check('a dry lock takes the garbage', out['tanked'] and len(out['tanked']) == 3,
          out['tanked'])
    check('the board rises accordingly', sum(after) - sum(before) == 9 * 3,
          (before, after))
    check('and the board stays clean apart from the holes',
          len(set(out['tanked'])) == 1, out['tanked'])


def test_battle_resets_between_rounds():
    battle = V.Battle(seed=4)
    battle.sides[0].wins = 1
    battle.new_round()
    check('wins survive a round reset', battle.sides[0].wins == 1)
    check('boards are emptied for the new round',
          all(h == 0 for h in battle.sides[0].game.board.column_heights()))
    check('rounds are counted', battle.round == 2)


def test_match_plays_out():
    """An attacker against a passive board, so the garbage actually lands.

    Two identical policies on the same piece sequence cancel every attack
    perfectly -- measured, and correct rather than a bug: the log is a mirror. So
    the receiving side here is deliberately one that never clears a line, which is
    the only way to see garbage enter.
    """
    from tetrisrl.heuristic import heuristic_choice

    def attacker(game, actions):
        return heuristic_choice(game, actions) or actions[0]

    def passive(game, actions):
        # Always the leftmost placement: it never completes a row, so everything
        # it receives can only pile up.
        return min(actions, key=lambda a: (a[-1], a[-2]))

    battle = V.Battle(seed=7, rounds_to_win=1, messiness='default')
    sent = []
    winner = V.play_out(battle, [attacker, passive], max_pieces=1200,
                        on_lock=lambda i, s: sent.append(s['sent']) if s['sent'] else None)
    check('the attacker beats a board that never clears', winner == 0, winner)
    check('attacks were exchanged', sum(sent) > 0, sum(sent))
    check('garbage reached the receiving board',
          battle.sides[1].tanked > 0, battle.sides[1].tanked)
    check('and the loser is the one that took it',
          battle.sides[1].tanked > battle.sides[0].tanked,
          (battle.sides[0].tanked, battle.sides[1].tanked))


def test_same_sequence_is_shared():
    a = V.Battle(seed=11)
    b = V.Battle(seed=11)
    check('the same seed gives the same bags',
          a.sides[0].game.queue_kinds == b.sides[0].game.queue_kinds)
    check('and both sides of a battle share one sequence',
          a.sides[0].game.queue_kinds == a.sides[1].game.queue_kinds)


def test_spin_attacks():
    print('T-spin attacks:')
    check('a T-spin single sends two',
          V.attack_for(1, combo=0, spin='full').lines == 2)
    check('a T-spin double sends four',
          V.attack_for(2, combo=0, spin='full').lines == 4)
    check('a T-spin triple sends six',
          V.attack_for(3, combo=0, spin='full').lines == 6)
    check('a T-spin quad sends ten',
          V.attack_for(4, combo=0, spin='full').lines == 10)
    check('a spin with no clear sends nothing',
          V.attack_for(0, combo=0, spin='full').lines == 0)
    check('a mini single sends nothing',
          V.attack_for(1, combo=0, spin='mini').lines == 0)
    check('a mini double sends one',
          V.attack_for(2, combo=0, spin='mini').lines == 1)
    check('a mini triple sends two',
          V.attack_for(3, combo=0, spin='mini').lines == 2)
    # The whole point: without detection the same lock is an ordinary double.
    check('an ordinary double still sends one',
          V.attack_for(2, combo=0).lines == 1)
    check('a spin continues the B2B chain',
          V.attack_for(2, combo=0, b2b=1, spin='full').b2b == 2)
    check('and is paid the B2B charging bonus',
          V.attack_for(2, combo=0, b2b=1, spin='full').lines == 5)
    check('a T-spin triple on a B2B streak sends seven',
          V.attack_for(3, combo=0, b2b=1, spin='full').lines == 7)
    check('a T-spin triple on a combo sends seven',
          V.attack_for(3, combo=1, spin='full').lines == 7)
    check('an ordinary double still breaks the chain',
          V.attack_for(2, combo=0, b2b=1).b2b == 0)


def test_battle_pays_a_tspin():
    """Real T-spins, through the battle layer, are worth four and six lines.

    The board is the seat from the engine tests: rows 18 and up solid except for a
    2-wide mouth at row 18, a 3-wide notch at row 19, a 1-wide gap at row 20 and
    the cell under it. The same dropped T spins into it either way -- a 180 turn
    clears two rows, a counter-clockwise turn clears three -- so this covers
    detection, the attack table and the cancel/send path at once.
    """
    from tetrisrl.engine import Piece, spawn_anchor

    def spun_seat(direction):
        battle = V.Battle(seed=0)
        game = battle.sides[0].game
        for y in range(18, 22):
            for x in range(game.cols):
                game.board.grid[y][x] = 'L'
        for x, y in ((3, 18), (3, 19), (4, 18), (4, 19), (4, 20), (5, 19)):
            game.board.grid[y][x] = None
        game.current = Piece('T', 3, spawn_anchor('T')[1], 3)
        while game.soft_drop():
            pass
        check(f'the T turns in the seat (dir {direction})', game.rotate(direction))
        cleared, _rows = game.lock()
        return battle, game, cleared

    battle, game, cleared = spun_seat(2)
    check('the 180 turn clears two lines', cleared == 2, cleared)
    check('and the engine calls it a full spin', game.spin_kind == 'full',
          game.spin_kind)
    summary = battle.after_lock(0, cleared)
    check('the battle pays four lines for a T-spin double',
          summary['sent'] == 4, summary)
    check('and reports the spin', summary['spin'] == 'full', summary)

    battle, game, cleared = spun_seat(-1)
    check('the counter-clockwise turn clears three lines', cleared == 3, cleared)
    check('and is a full spin too', game.spin_kind == 'full', game.spin_kind)
    summary = battle.after_lock(0, cleared)
    check('the battle pays six lines for a T-spin triple',
          summary['sent'] == 6, summary)
    check('and reports the spin', summary['spin'] == 'full', summary)
    check('and starts a B2B chain', summary['b2b'] == 1, summary)


def main():
    for fn in (test_attack_table,
               test_combo,
               test_b2b_and_surge,
               test_spin_attacks,
               test_battle_pays_a_tspin,
               test_queue_travel_and_cancel,
               test_queue_offsets_are_oldest_first,
               test_cap_limits_entry,
               test_messiness,
               test_garbage_raises_the_stack,
               test_garbage_top_out,
               test_garbage_smash,
               test_no_lockout_in_versus,
               test_battle_tanks_only_on_a_clean_lock,
               test_battle_cancel_and_send,
               test_battle_takes_garbage_on_a_dry_lock,
               test_battle_resets_between_rounds,
               test_match_plays_out,
               test_same_sequence_is_shared):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S) out of {CHECKS} checks: {FAILS}')
        return 1
    print(f'all versus tests passed ({CHECKS} checks)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
