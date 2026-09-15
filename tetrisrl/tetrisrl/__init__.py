"""Tetris reinforcement learning.

A small, self-contained project: a correct SRS Tetris engine, a feature-based
Double DQN agent, parallel CPU training, and a pygame front end for watching or
playing.

Typical use::

    python -m tetrisrl train
    python -m tetrisrl watch --policy best
    python -m tetrisrl play
"""
from __future__ import annotations

import os

import yaml

__version__ = '1.0.0'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')


def load_config(path=None):
    """Load config.yaml, merged over built-in defaults.

    Missing keys never crash: the defaults below are complete, so a partial or
    hand-edited config still runs.
    """
    cfg = {
        'game': {
            'cols': 10, 'rows': 22, 'visible_rows': 20,
            'cell_size': 30, 'seed': None,
        },
        'agent': {
            'lr': 0.0007, 'gamma': 0.99, 'n_step': 3,
            'batch_size': 256, 'buffer_size': 400000, 'target_sync': 1500,
            'epsilon_start': 1.0, 'epsilon_end': 0.02,
            'epsilon_decay_steps': 250000,
            # Supervised warm start uses a larger step than TD learning.
            'warmstart_lr': 0.001,
            # Learning-rate annealing. DEFAULT OFF: measured in a matched
            # ~420-round A/B, annealing produced a worse agent (21.2 vs 30.2
            # lines/game) even though it lowered the training loss. See config.yaml.
            'lr_decay_fraction': 0.0,
            'lr_warmup_fraction': 0.3,
            'lr_min': 0.00007,
            # Feature set. 1 is what the shipping checkpoints use; 2-4 add
            # stacking-quality, rows-with-holes and tetris-readiness terms. A
            # checkpoint records its own version and a layout fingerprint.
            'feature_version': 1,
        },
        'reward': {
            'survival_bonus': -1.0,
            'single': 10.0, 'double': 30.0, 'triple': 60.0, 'tetris': 100.0,
            'death_penalty': 10.0,
            'height_penalty': 0.0, 'hole_penalty': 0.0, 'bumpiness_penalty': 0.0,
        },
        'training': {
            'rounds': 300, 'workers': 12, 'episodes_per_worker': 6,
            'grad_steps_auto': True, 'grad_steps_per_sample': 0.25,
            'min_grad_steps_per_round': 200,
            'grad_steps_per_round': 300, 'checkpoint_every': 25,
            'eval_every': 10, 'eval_games': 12,
            'imitation_episodes': 15, 'imitation_steps': 0,
            'imitation_batch_size': 256,
            'teacher_fraction': 0.0, 'teacher_rounds': 0,
        },
    }
    path = path or DEFAULT_CONFIG
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return cfg
    except Exception as exc:                      # malformed yaml: keep going
        print(f'warning: could not read {path} ({exc}); using defaults')
        return cfg
    for section, values in loaded.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


__all__ = ['load_config', 'CHECKPOINT_DIR', 'PROJECT_DIR', '__version__']
