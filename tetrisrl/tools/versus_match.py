"""Play the versus bot against an opponent, headlessly.

    python tools/versus_match.py --difficulty advanced --games 5
    python tools/versus_match.py --difficulty max --opponent heuristic --verbose

The opponent is `heuristic` by default -- the strong one-ply evaluator, which
never dies on a clean board and so makes a real attacker. `passive` never clears a
line, which is the fastest way to watch garbage land, and `mirror` plays the same
policy on both sides, where every attack cancels and the match is a stalemate by
construction.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import versus as V                        # noqa: E402
from tetrisrl.engine import valid_actions               # noqa: E402
from tetrisrl.heuristic import apply_move, heuristic_choice   # noqa: E402
from tetrisrl.policies import (DEFAULT_DIFFICULTY, HANDICAP_PRESETS,  # noqa: E402
                               make_versus_policy)


def passive_policy(game, actions):
    """Leftmost placement: never completes a row, so garbage can only pile up."""
    return min(actions, key=lambda a: (a[-1], a[-2]))


def heuristic_policy(game, actions):
    return heuristic_choice(game, actions) or actions[0]


def play_match(difficulty, opponent, seed, rounds, messiness, cap=4000,
               verbose=False, foe_pps=2.0, frame=1 / 60.0):
    """One match, driven by a clock so the speed handicap is real.

    Each side may place a piece only every ``1/pps`` seconds of match time, which
    is what a difficulty preset means in the UI. An earlier version of this tool
    let both sides move every iteration and the ``pps`` knob did nothing, so every
    preset measured the same speed and the ladder was meaningless.
    """
    battle = V.Battle(seed=seed, rounds_to_win=rounds, messiness=messiness)
    bot = make_versus_policy(difficulty=difficulty,
                             incoming_fn=lambda: battle.incoming(0))
    foe = bot if opponent == 'mirror' else (
        passive_policy if opponent == 'passive' else heuristic_policy)
    rate = [bot.pps, bot.pps if opponent == 'mirror' else foe_pps]
    ready_at = [0.0, 0.0]
    pieces = 0
    while battle.winner() is None and pieces < cap:
        moved = False
        for index, policy in enumerate((bot, foe)):
            if battle.time + 1e-9 < ready_at[index]:
                continue
            side = battle.sides[index]
            game = side.game
            if game.game_over:
                continue
            actions = valid_actions(game.board, game.current)
            if not actions:
                game.game_over = True
                continue
            cleared = apply_move(game, policy(game, actions))
            summary = battle.after_lock(index, cleared)
            ready_at[index] = battle.time + 1.0 / rate[index]
            if getattr(policy, 'reaction', 0) and summary['tanked']:
                ready_at[index] += policy.reaction
            moved = True
            if verbose and (summary['sent'] or summary['tanked']):
                print(f'  t={battle.time:6.1f} {side.name:<4} cleared={cleared} '
                      f'sent={summary["sent"]} tanked={len(summary["tanked"])} '
                      f'incoming={battle.incoming(index)}')
            if game.game_over:
                battle.other(index).wins += 1
                if battle.winner() is None:
                    battle.new_round()
                    ready_at = [battle.time, battle.time]
                break
        battle.update(frame)
        if moved:
            pieces += 1
    return battle, pieces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--difficulty', default=DEFAULT_DIFFICULTY,
                        choices=sorted(HANDICAP_PRESETS))
    parser.add_argument('--opponent', default='heuristic',
                        choices=('heuristic', 'passive', 'mirror'))
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=1,
                        help='rounds needed to win a match')
    parser.add_argument('--messiness', default='default',
                        choices=sorted(V.MESSINESS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--foe-pps', type=float, default=2.0,
                        help='pieces per second for the non-mirror opponent')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    wins = 0
    lengths = []
    rows = []
    for i in range(args.games):
        seed = args.seed + i
        battle, pieces = play_match(args.difficulty, args.opponent, seed,
                                    args.rounds, args.messiness,
                                    verbose=args.verbose, foe_pps=args.foe_pps)
        winner = battle.winner()
        bot_won = winner == 0
        wins += 1 if bot_won else 0
        lengths.append(pieces)
        rows.append((seed, winner, pieces, battle.sides[0].sent,
                     battle.sides[0].tanked, battle.sides[1].tanked))
        print(f'  game {i + 1}: seed={seed} winner='
              f'{"bot" if bot_won else ("opponent" if winner == 1 else "draw/cap")} '
              f'pieces={pieces} bot sent={battle.sides[0].sent} '
              f'tanked={battle.sides[0].tanked} foe tanked={battle.sides[1].tanked}')

    preset = HANDICAP_PRESETS[args.difficulty]
    print()
    print(f'difficulty {args.difficulty} ({preset["label"]}) vs {args.opponent}: '
          f'{wins}/{args.games} wins, mean {statistics.mean(lengths):.0f} pieces')
    print(f'  pps={preset["pps"]} reaction={preset["reaction"]}s '
          f'mistake={preset["mistake"]} depth={preset["depth"]} beam={preset["beam"]}')
    print(f'  bot tanked {sum(r[4] for r in rows)} lines, '
          f'sent {sum(r[3] for r in rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
