"""Report the clear mix of a policy: singles / doubles / triples / tetrises.

Total lines alone hides what a policy is actually doing. A policy clearing 900
lines as 81% singles is playing a very different game from one clearing 780 lines
with 4% tetrises -- and for Guideline scoring the second can be worth more,
because a tetris pays 800 against 100 for a single.

Usage:
    python tools/clear_mix.py --policy heuristic
    python tools/clear_mix.py --policy heuristic_tetris --games 6 --cap 2500
    python tools/clear_mix.py --policy best_rewardfix --games 20 --cap 3000
"""
import argparse
import multiprocessing
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default='heuristic')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--cap', type=int, default=2500,
                        help='max pieces per game')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    from tetrisrl.engine import Game, spawn_anchor, valid_actions
    from tetrisrl.features import metrics_for
    from tetrisrl.heuristic import apply_move
    from tetrisrl.policies import load_policy

    policy, desc = load_policy(args.policy, args.checkpoint)
    print(f'policy: {desc}')
    print(f'{args.games} games x {args.cap} pieces, seeds '
          f'{args.seed}..{args.seed + args.games - 1}')

    mix = Counter()
    total_lines = total_score = total_pieces = total_holes = 0
    holds = 0
    lengths = []
    died = 0
    for i in range(args.games):
        game = Game(seed=args.seed + i)
        while not game.game_over and game.pieces_placed < args.cap:
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            move = policy(game, actions)
            if len(move) == 3 and move[0]:
                holds += 1
            n = apply_move(game, move)
            if n:
                mix[n] += 1
        total_lines += game.lines
        total_score += game.score
        total_pieces += game.pieces_placed
        total_holes += metrics_for(game.board.grid, game.rows, game.cols)[1]
        lengths.append(game.pieces_placed)
        if game.game_over:
            died += 1

    n = args.games
    events = sum(mix.values()) or 1
    print()
    print(f'lines/game  {total_lines / n:9.1f}      score/game {total_score / n:12,.0f}')
    print(f'pieces/game {total_pieces / n:9.1f}      holes at end {total_holes / n:6.1f}')
    print(f'games ended by topping out: {died}/{n}')
    print(f'holds used: {holds} ({100 * holds / max(1, total_pieces):.1f}% of pieces)')
    print()
    print(f'line-clear events: {events}')
    for k in (1, 2, 3, 4):
        label = {1: 'single', 2: 'double', 3: 'triple', 4: 'tetris'}[k]
        print(f'  {label:7s} {mix[k]:6d}   {100 * mix[k] / events:5.1f}%')
    if events:
        per_tetris = events / max(1, mix[4])
        print(f'  -> one tetris per {per_tetris:.1f} clear events')
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
