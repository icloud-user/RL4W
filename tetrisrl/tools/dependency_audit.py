"""Measure the I-dependency doom loop, before and after the weights that fix it.

    python tools/dependency_audit.py --games 10 --arm both
    python tools/dependency_audit.py --games 10 --arm new --difficulty max

Plays the versus bot against the heuristic opponent on a clock (so the
pieces-per-second handicap is real), and records, per game, what the *bot's* board
looked like after every one of its locks:

* ``max`` / ``mean`` dependencies -- Blockfish's count of one-cell-wide gaps at
  least three rows deep, which only a vertical I can fill;
* how long they stayed open (share of locks with one, and with two or more);
* mean and peak stack height, lines, pieces, tetris share and whether it died.

Two arms, same seeds, same opponent:

* ``old`` -- the versus weights as they shipped before the dependency term;
* ``new`` -- the shipped versus weights, which price a dependency and price the
  second one much harder.

Run both arms on the same seeds or the comparison means nothing.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import search as S                         # noqa: E402
from tetrisrl import versus as V                         # noqa: E402
from tetrisrl.engine import valid_actions                # noqa: E402
from tetrisrl.heuristic import apply_move, heuristic_choice   # noqa: E402
from tetrisrl.policies import make_versus_policy         # noqa: E402

#: The versus weights as they were before the dependency work: no price on a
#: dependency and no gate. Both have to be zeroed -- leaving the gate inherited
#: from the shipped preset made the two arms identical and hid the effect.
OLD_WEIGHTS = dict(S.VERSUS_SEARCH_WEIGHTS, i_dependency=0.0, i_dependency_sq=0.0,
                   i_dependency_rows=0.0, dep_gate_height=0)
NEW_WEIGHTS = dict(S.VERSUS_SEARCH_WEIGHTS)
ARMS = {'old': OLD_WEIGHTS, 'new': NEW_WEIGHTS}


def play(difficulty, weights, seed, messiness='default', rounds=1, cap=4000,
         foe_pps=2.0, frame=1 / 60.0, garbage_every=0, garbage_lines=4):
    """One game. ``garbage_every`` > 0 plays the bot solo against injected
    garbage instead of against the heuristic, which is the only way to reproduce
    the doom loop: garbage is what manufactures one-wide gaps.

    Both arms receive byte-identical garbage (same seed, same schedule), so the
    comparison is of the weights, not of the dice.
    """
    import random as _random

    battle = V.Battle(seed=seed, rounds_to_win=rounds, messiness=messiness)
    bot = make_versus_policy(difficulty=difficulty, weights=weights,
                             incoming_fn=lambda: battle.incoming(0))
    solo = garbage_every > 0
    if not solo:
        def foe(game, actions):
            return heuristic_choice(game, actions) or actions[0]
    rng = _random.Random(seed)

    rate = [bot.pps, foe_pps]
    ready_at = [0.0, 0.0]
    deps, tops, clears = [], [], {1: 0, 2: 0, 3: 0, 4: 0}
    pieces = 0
    while battle.winner() is None and pieces < cap:
        for index, policy in enumerate((bot, foe) if not solo else (bot,)):
            if battle.time + 1e-9 < ready_at[index]:
                continue
            game = battle.sides[index].game
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
            if index == 0:
                pieces += 1
                if solo and pieces % garbage_every == 0:
                    # The opponent "attacks": queued, then tanked on the next lock
                    # that clears nothing, exactly as versus does it.
                    battle.sides[0].queue.send([garbage_lines], battle.time)
                masks = S.board_masks(game)
                heights = S.heights_of(masks)
                deps.append(S.i_dependencies(masks, heights)[0])
                tops.append(max(heights))
                if cleared:
                    clears[min(cleared, 4)] = clears.get(min(cleared, 4), 0) + 1
            if game.game_over:
                if not solo:
                    battle.other(index).wins += 1
                    if battle.winner() is None:
                        battle.new_round()
                        ready_at = [battle.time, battle.time]
                break
        battle.update(frame)
        if solo and battle.sides[0].game.game_over:
            break

    events = sum(clears.values())
    return {
        'pieces': pieces,
        'lines': battle.sides[0].game.lines,
        'max_deps': max(deps) if deps else 0,
        'mean_deps': statistics.mean(deps) if deps else 0.0,
        'locks_with_dep': 100.0 * sum(1 for d in deps if d >= 1) / max(1, len(deps)),
        'locks_with_two': 100.0 * sum(1 for d in deps if d >= 2) / max(1, len(deps)),
        'final_deps': deps[-1] if deps else 0,
        'mean_top': statistics.mean(tops) if tops else 0.0,
        'peak_top': max(tops) if tops else 0,
        'tetris_pct': 100.0 * clears[4] / max(1, events),
        'dead': battle.sides[0].game.game_over,
    }


def run_arm(name, difficulty, games, seed0, messiness, rounds, verbose,
            garbage_every=0, garbage_lines=4):
    rows = [play(difficulty, ARMS[name], seed0 + i, messiness, rounds,
                 garbage_every=garbage_every, garbage_lines=garbage_lines)
            for i in range(games)]
    mean = statistics.mean
    print(f'--- arm {name} ({games} games, difficulty {difficulty}, seeds '
          f'{seed0}..{seed0 + games - 1}'
          + (f', garbage {garbage_lines} lines every {garbage_every} pieces'
             if garbage_every else ', vs heuristic') + ') ---')
    if verbose:
        print('  seed  deps(max/mean)  locks>=1  locks>=2  top(mean/peak)  '
              'lines  pieces  tetris%  dead')
        for i, r in enumerate(rows):
            print(f'  {seed0 + i:<5} {r["max_deps"]:>4}/{r["mean_deps"]:<9.2f} '
                  f'{r["locks_with_dep"]:>7.1f}%  {r["locks_with_two"]:>7.1f}%  '
                  f'{r["mean_top"]:>6.1f}/{r["peak_top"]:<7} {r["lines"]:>5}  '
                  f'{r["pieces"]:>6}  {r["tetris_pct"]:>6.1f}%  '
                  f'{"yes" if r["dead"] else "no"}')
    summary = {
        'max_deps': mean(r['max_deps'] for r in rows),
        'locks_with_dep': mean(r['locks_with_dep'] for r in rows),
        'locks_with_two': mean(r['locks_with_two'] for r in rows),
        'mean_top': mean(r['mean_top'] for r in rows),
        'peak_top': mean(r['peak_top'] for r in rows),
        'lines': mean(r['lines'] for r in rows),
        'pieces': mean(r['pieces'] for r in rows),
        'tetris_pct': mean(r['tetris_pct'] for r in rows),
        'deaths': sum(1 for r in rows if r['dead']),
        'final_deps': mean(r['final_deps'] for r in rows),
    }
    print(f'  mean  deps(max {summary["max_deps"]:.2f})  '
          f'locks>=1 {summary["locks_with_dep"]:.1f}%  '
          f'locks>=2 {summary["locks_with_two"]:.1f}%  '
          f'top {summary["mean_top"]:.1f}/{summary["peak_top"]:.1f}  '
          f'lines {summary["lines"]:.0f}  pieces {summary["pieces"]:.0f}  '
          f'tetris {summary["tetris_pct"]:.1f}%  '
          f'deaths {summary["deaths"]}/{len(rows)}  '
          f'final deps {summary["final_deps"]:.2f}')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--arm', default='both', choices=('old', 'new', 'both'))
    parser.add_argument('--difficulty', default='max')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--messiness', default='default')
    parser.add_argument('--garbage-every', type=int, default=0,
                        help='play solo and inject garbage every N pieces '
                             '(0 = play against the heuristic instead)')
    parser.add_argument('--garbage-lines', type=int, default=4)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    for name, weights in ARMS.items():
        print(f'arm {name}: i_dependency={weights["i_dependency"]} '
              f'i_dependency_sq={weights["i_dependency_sq"]} '
              f'i_dependency_rows={weights["i_dependency_rows"]}')

    arms = ('old', 'new') if args.arm == 'both' else (args.arm,)
    out = {}
    for name in arms:
        out[name] = run_arm(name, args.difficulty, args.games, args.seed,
                            args.messiness, args.rounds, not args.quiet,
                            garbage_every=args.garbage_every,
                            garbage_lines=args.garbage_lines)
    if len(out) == 2:
        old, new = out['old'], out['new']
        print()
        print('delta (new - old): '
              f'deps(max) {new["max_deps"] - old["max_deps"]:+.2f}  '
              f'locks>=2 {new["locks_with_two"] - old["locks_with_two"]:+.1f}pp  '
              f'top(mean) {new["mean_top"] - old["mean_top"]:+.2f}  '
              f'lines {new["lines"] - old["lines"]:+.0f}  '
              f'pieces {new["pieces"] - old["pieces"]:+.0f}  '
              f'tetris {new["tetris_pct"] - old["tetris_pct"]:+.1f}pp  '
              f'deaths {new["deaths"] - old["deaths"]:+d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
