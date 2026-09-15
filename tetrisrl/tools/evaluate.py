"""Benchmark policies over full games.

Reports real game outcomes (lines cleared, Guideline score, pieces survived),
never the shaped training reward.

Usage:
    python tools/evaluate.py                     # heuristic + best + latest
    python tools/evaluate.py --games 30
    python tools/evaluate.py --policy latest --games 50
"""
import argparse
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default=None,
                        help='heuristic | best | latest | path. '
                             'Default: evaluate all available.')
    parser.add_argument('--checkpoint', default=None,
                        help='explicit path to a checkpoint file to evaluate')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-pieces', type=int, default=20000)
    args = parser.parse_args(argv)

    from tetrisrl.evaluate import evaluate_policy, format_summary
    from tetrisrl.policies import list_policies, load_policy

    if args.checkpoint:
        policy, desc = load_policy(args.policy or 'best',
                                   checkpoint=args.checkpoint)
        t0 = time.time()
        summary = evaluate_policy(policy, episodes=args.games, seed0=args.seed,
                                  max_pieces=args.max_pieces)
        summary['label'] = desc
        print(format_summary(summary))
        print(f'         ({time.time() - t0:.0f}s)')
        return 0

    names = [args.policy] if args.policy else list_policies()
    for name in names:
        policy, desc = load_policy(name)
        t0 = time.time()
        summary = evaluate_policy(policy, episodes=args.games, seed0=args.seed,
                                  max_pieces=args.max_pieces)
        summary['label'] = desc
        print(format_summary(summary))
        print(f'         ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
