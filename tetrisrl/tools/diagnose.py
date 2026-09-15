"""Diagnose how the trained agent fails.

Reports the distribution of game lengths, the line-clear mix, and the board
state at the moment of death. The point is to separate two very different
problems:

* the agent dies with a *low, clean* board -> it is making bad placement
  decisions (a value/policy problem), or
* the agent dies with a *tall, holed* board -> it is stacking badly over a long
  horizon (a credit-assignment / feature problem).

Usage:  python tools/diagnose.py [--policy best] [--games 200]
"""
import argparse
import multiprocessing
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default='best')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    from tetrisrl.engine import Game, spawn_anchor, valid_actions
    from tetrisrl.features import holes_of, metrics_for
    from tetrisrl.heuristic import apply_move
    from tetrisrl.policies import load_policy

    policy, desc = load_policy(args.policy)
    print(f'policy: {desc}')
    print(f'games : {args.games} (seeds {args.seed}..{args.seed + args.games - 1})')

    rows = []
    for i in range(args.games):
        game = Game(seed=args.seed + i)
        clears = Counter()
        pieces_at = []
        while not game.game_over:
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            n = apply_move(game, policy(game, actions))
            if n:
                clears[n] += 1
            pieces_at.append(game.pieces_placed)

        m = metrics_for(game.board.grid, game.rows, game.cols)
        heights, holes, bumpiness, wells = m[0], m[1], m[2], m[6]
        rows.append({
            'lines': game.lines,
            'score': game.score,
            'pieces': game.pieces_placed,
            'max_height': max(heights),
            'holes': holes,
            'bumpiness': bumpiness,
            'wells': wells,
            'clears': clears,
            'top_out': game.game_over,
        })

    n = len(rows)
    lines = sorted(r['lines'] for r in rows)
    pieces = sorted(r['pieces'] for r in rows)

    def pct(vals, p):
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    print()
    print('game length (pieces):')
    print(f'  min {pieces[0]}  p10 {pct(pieces, .10)}  p25 {pct(pieces, .25)}  '
          f'median {pct(pieces, .50)}  p75 {pct(pieces, .75)}  '
          f'p90 {pct(pieces, .90)}  max {pieces[-1]}')

    print('lines cleared:')
    print(f'  min {lines[0]}  p10 {pct(lines, .10)}  median {pct(lines, .50)}  '
          f'mean {sum(lines) / n:.1f}  p90 {pct(lines, .90)}  max {lines[-1]}')

    print('board at death:')
    for key in ('max_height', 'holes', 'bumpiness', 'wells'):
        vals = sorted(r[key] for r in rows)
        print(f'  {key:11s} median {pct(vals, .50):6.1f}   '
              f'mean {sum(vals) / n:6.1f}   max {vals[-1]}')

    # The key comparison: how often did it die with a board that was NOT full?
    tall = sum(1 for r in rows if r['max_height'] >= 20)
    print()
    print(f'died with the stack at height >= 20 (genuinely full): {tall}/{n}')
    print(f'died with the stack below height 20:                 {n - tall}/{n}'
          '  <- these are premature if the board was otherwise clean')

    total_clears = sum(sum(r['clears'].values()) for r in rows)
    mix = Counter()
    for r in rows:
        mix.update(r['clears'])
    print()
    print(f'line-clear events: {total_clears}')
    for k in sorted(mix):
        label = {1: 'single', 2: 'double', 3: 'triple', 4: 'tetris'}.get(k, str(k))
        print(f'  {label:7s} {mix[k]:6d}  ({100 * mix[k] / max(1, total_clears):5.1f}%)')

    survivors = sum(1 for r in rows if not r['top_out'])
    print()
    print(f'games that ran to the piece cap instead of dying: {survivors}/{n}')

    short = sum(1 for r in rows if r['pieces'] < 100)
    print(f'games dead in under 100 pieces: {short}/{n} '
          f'({100 * short / n:.0f}%)  <- the "it just dies sometimes" cases')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
