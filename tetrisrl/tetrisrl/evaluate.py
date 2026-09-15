"""Evaluation: measure a policy over many games and report honest numbers.

Reported metrics are REAL game outcomes -- lines cleared and the standard
Guideline score -- never the shaped training reward.
"""
from __future__ import annotations

import random

import numpy as np

from .engine import Game, valid_actions
from .heuristic import apply_move


def run_episode(policy, seed=None, max_pieces=20000, greedy=True):
    """Play one full game with ``policy``. Returns a stats dict."""
    game = Game(seed=seed)
    while not game.game_over:
        actions = valid_actions(game.board, game.current)
        if not actions:
            break
        # apply_move accepts (use_hold, rotation, x) or a bare (rotation, x).
        apply_move(game, policy(game, actions))
        if game.pieces_placed >= max_pieces:
            break
    return {
        'lines': game.lines,
        'score': game.score,
        'pieces': game.pieces_placed,
        'level': game.level,
    }


def evaluate_policy(policy, episodes=50, seed0=0, verbose=False, label='policy',
                    max_pieces=20000):
    """Run ``episodes`` games and summarise. No shaping reward anywhere."""
    results = []
    for i in range(episodes):
        stats = run_episode(policy, seed=seed0 + i, max_pieces=max_pieces)
        results.append(stats)
        if verbose:
            print(f'  game {i + 1:3d}/{episodes}: lines {stats["lines"]:4d} '
                  f'score {stats["score"]:7d} pieces {stats["pieces"]:5d}')
    lines = np.array([r['lines'] for r in results])
    scores = np.array([r['score'] for r in results])
    pieces = np.array([r['pieces'] for r in results])
    summary = {
        'label': label,
        'games': episodes,
        'lines_mean': float(lines.mean()),
        'lines_median': float(np.median(lines)),
        'lines_max': int(lines.max()),
        'lines_min': int(lines.min()),
        'score_mean': float(scores.mean()),
        'score_max': int(scores.max()),
        'pieces_mean': float(pieces.mean()),
        'cleared_any': int((lines > 0).sum()),
        'cleared_rate': float((lines > 0).mean()),
    }
    return summary


def format_summary(s):
    return (f'{s["label"]}: {s["games"]} games | '
            f'lines/game mean {s["lines_mean"]:.2f} median {s["lines_median"]:.1f} '
            f'max {s["lines_max"]} | score/game mean {s["score_mean"]:.0f} '
            f'max {s["score_max"]} | pieces/game {s["pieces_mean"]:.0f} | '
            f'cleared >=1 line in {s["cleared_any"]}/{s["games"]} '
            f'({100 * s["cleared_rate"]:.0f}%)')
