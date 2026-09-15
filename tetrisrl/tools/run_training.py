"""Run the full training job.

Guarded by ``__main__`` because Windows child processes re-import this module.

Usage:  python tools/run_training.py [rounds] [workers] [episodes_per_worker]
"""
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv):
    from tetrisrl import load_config
    from tetrisrl.train import train

    cfg = load_config()
    rounds = int(argv[1]) if len(argv) > 1 else cfg['training']['rounds']
    workers = int(argv[2]) if len(argv) > 2 else cfg['training']['workers']
    epw = int(argv[3]) if len(argv) > 3 else cfg['training']['episodes_per_worker']

    t0 = time.time()
    train(cfg, rounds=rounds, workers=workers, episodes_per_worker=epw)
    print(f'wall clock: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(sys.argv)
