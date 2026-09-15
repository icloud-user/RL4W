"""Compare feature versions in a controlled setting.

Counterfactual test for "did the new features make learning worse?". Instead of
spending an hour per training run, this trains a small model to predict the
heuristic teacher's choice from each feature version on identical data, and
measures held-out accuracy. A better-conditioned feature set should support a
more accurate policy at the same model size.

This cannot prove a feature set is better for *long-horizon RL*, but it cleanly
separates "these inputs do not carry the information" from "the RL loop is at
fault", which is the question that matters when a change regresses.

Usage:  python tools/compare_features.py [--train-games 12] [--test-games 6]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect(games, seed0, version, step_cap=400):
    """(features, action_id, legal_mask) examples from teacher play."""
    from tetrisrl.agent import action_id, placement_stats, valid_mask
    from tetrisrl.engine import Game, spawn_anchor, valid_actions
    from tetrisrl.features import board_vector_from_metrics, metrics_for, \
        placement_vector
    from tetrisrl.heuristic import heuristic_choice

    rows = []
    for i in range(games):
        g = Game(seed=seed0 + i)
        while not g.game_over and g.pieces_placed < step_cap:
            if g.current is None:
                break
            acts = valid_actions(g.board, g.current)
            if not acts:
                break
            current = g.current.kind if g.current else None
            nxt = g.next_queue[0].kind if g.next_queue else None
            base = board_vector_from_metrics(
                metrics_for(g.board.grid, g.rows, g.cols), current, nxt, version)
            choice = heuristic_choice(g, acts)
            stats, _r, _d = placement_stats(g, choice[0], choice[1])
            vec = placement_vector(base, stats, version)
            rows.append((vec, action_id(*choice), valid_mask(acts)))
            g.current.rotation, g.current.x = choice
            g.current.y = spawn_anchor(g.current.kind)[1]
            g.hard_drop()
    return rows


def train_eval(train_rows, test_rows, steps=4000, lr=1e-3, seed=0, hidden=256):
    from tetrisrl.model import QNet

    torch.manual_seed(seed)
    random.seed(seed)
    n_in = len(train_rows[0][0])
    V = torch.from_numpy(np.stack([r[0] for r in train_rows]))
    A = torch.tensor([r[1] for r in train_rows])
    M = torch.from_numpy(np.stack([r[2] for r in train_rows]))
    Vs = torch.from_numpy(np.stack([r[0] for r in test_rows]))
    As = torch.tensor([r[1] for r in test_rows])
    Ms = torch.from_numpy(np.stack([r[2] for r in test_rows]))

    net = QNet(n_features=n_in, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = 256
    for _ in range(steps):
        idx = torch.tensor(random.sample(range(len(train_rows)), bs))
        logits = net(V[idx]).masked_fill(~M[idx], -1e9)
        loss = nn.functional.cross_entropy(logits, A[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    def acc(X, Y, Mask):
        with torch.no_grad():
            pred = net(X).masked_fill(~Mask, -1e9).argmax(1)
            return float((pred == Y).float().mean().item()) * 100

    return acc(V, A, M), acc(Vs, As, Ms), loss.item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-games', type=int, default=12)
    parser.add_argument('--test-games', type=int, default=6)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args(argv)

    from tetrisrl.features import VERSIONS, n_features

    print(f'train games {args.train_games} (seeds 0..), '
          f'test games {args.test_games} (held-out seeds)')
    results = {}
    for version in VERSIONS:
        t0 = time.time()
        train_rows = collect(args.train_games, 0, version)
        test_rows = collect(args.test_games, 900000, version)
        accs = []
        for seed in range(args.seeds):
            _tr, te, _loss = train_eval(train_rows, test_rows, steps=args.steps,
                                        seed=seed)
            accs.append(te)
        results[version] = (float(np.mean(accs)), float(np.std(accs)))
        print(f'  v{version} ({n_features(version):3d} features): '
              f'held-out imitation accuracy {np.mean(accs):5.1f}% '
              f'(+-{np.std(accs):.1f}, {args.seeds} seeds)  '
              f'[{len(train_rows)} train / {len(test_rows)} test examples, '
              f'{time.time() - t0:.0f}s]')

    print()
    if len(results) == 2:
        a, b = results[1][0], results[2][0]
        spread = max(results[1][1], results[2][1])
        verdict = ('v2 is better' if b > a + spread else
                   'v1 is better' if a > b + spread else 'no clear difference')
        print(f'v2 - v1 = {b - a:+.1f} points (seed spread ~{spread:.1f}) -> {verdict}')
        print('If the feature sets are equivalent on this test, a training '
              'regression is coming from the RL loop, not from the inputs.')
    return 0


if __name__ == '__main__':
    main()
