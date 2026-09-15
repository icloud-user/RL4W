"""Fine-tune an existing checkpoint.

The first long run peaked around round 189 and then slowly degraded while its
training loss kept climbing -- the usual Q-value overestimation. This continues
from the best weights with a lower learning rate and a re-annealed epsilon to
see whether it recovers, keeping whichever weights actually measure best.

Usage:  python tools/finetune.py [rounds] [workers] [episodes_per_worker]
                                [--from PATH] [--lr LR]
"""
import argparse
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rounds', nargs='?', type=int, default=150)
    parser.add_argument('workers', nargs='?', type=int, default=12)
    parser.add_argument('episodes_per_worker', nargs='?', type=int, default=6)
    parser.add_argument('--from', dest='source', default=None,
                        help='checkpoint to start from (default: best.pt)')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epsilon', type=float, default=0.10,
                        help='restart epsilon here and re-anneal down')
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args(argv)

    from tetrisrl import CHECKPOINT_DIR, load_config
    from tetrisrl.train import BEST_PATH, train

    source = args.source or BEST_PATH
    if not os.path.exists(source):
        print(f'no checkpoint at {source}')
        return 1

    # Keep the fine-tuned results in their own files so a bad run cannot
    # overwrite the checkpoint that is known to be good.
    import tetrisrl.train as T
    T.BEST_PATH = os.path.join(CHECKPOINT_DIR, 'best_finetuned.pt')
    T.LATEST_PATH = os.path.join(CHECKPOINT_DIR, 'latest_finetuned.pt')

    cfg = load_config()
    cfg['agent']['lr'] = args.lr
    cfg['agent']['epsilon_start'] = args.epsilon
    cfg['agent']['epsilon_decay_steps'] = 60000
    cfg['training'].update({
        'imitation_steps': 0,
        'grad_steps_per_round': 300,
        'checkpoint_every': 25,
        'eval_every': 10,
        'eval_games': 12,
    })

    print(f'fine-tuning from {source} at lr={args.lr}, '
          f'epsilon restart {args.epsilon} -> {cfg["agent"]["epsilon_end"]}')
    t0 = time.time()
    train(cfg, rounds=args.rounds, workers=args.workers,
          episodes_per_worker=args.episodes_per_worker, resume=source)
    print(f'wall clock: {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
