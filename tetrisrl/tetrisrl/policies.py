"""Policy factories used by the renderer and the CLI.

Kept separate from the renderer so that a missing or broken checkpoint can never
stop the game from starting: ``load_policy`` falls back to the built-in
heuristic and reports what it did.
"""
from __future__ import annotations

import os

from .engine import valid_actions
from .heuristic import heuristic_choice

BUILTIN = 'heuristic'

# The lookahead variant of the same evaluator. Same lines/game as `heuristic` but
# a better clear mix, at roughly 30x the cost per move. Opt-in.
LOOKAHEAD = 'heuristic2'

# Tetris-seeking evaluator: Cold Clear's negative-small-clear scheme plus a
# survival override. Roughly 83% of the default's lines and about 6x the tetris
# rate. Opt-in because it trades total lines for a better clear mix.
TETRIS_AWARE = 'heuristic_tetris'

# Beam search over the next few pieces, with the hold slot in the action space.
# This is what actually builds tetrises: the greedy evaluator cannot see that
# holding an I for three pieces is worth anything, and the search can. Measured at
# 53% of clear events being tetrises against 3.9% for the plain tetris-aware
# heuristic. Opt-in because it costs roughly 15x the plain heuristic per move.
SEARCH = 'search'
SEARCH_TETRIS = 'search_tetris'
#: The versus opponent: same search, garbage-aware weights, plus a handicap.
SEARCH_VERSUS = 'search_versus'

#: Difficulty presets for the versus opponent.
#:
#: The knobs are the ones published Tetris bots actually expose -- Zetris ships
#: Speed, Previews and Intelligence, MisaMino ships a "smartness" level -- plus a
#: mistake rate modelled on Stockfish's Skill Level, which plays a weaker move
#: with a probability instead of weakening every move.
#:
#: ``pps`` is calibrated against published TETRA LEAGUE attack rates
#: (neozt/tetra-league-stats, 39,477 players): D+ ~6 APM at 0.61 PPS, C ~9/0.74,
#: B ~15/0.93, A ~21/1.13, S ~33/1.40, X ~91/2.49. A tetris every ~7 pieces is
#: about 4 lines per 7 pieces, so a preset's ceiling is roughly
#: ``pps * 60 * 4 / 7`` APM; the bot has no T-spins and no opener, so treat these
#: as upper bounds rather than promises.
#:
#: ``mistake`` is the chance of playing the one-ply heuristic's move instead of the
#: search's. That is a real downgrade rather than noise: the heuristic is the
#: policy this project measured as unable to build tetrises at all.
HANDICAP_PRESETS = {
    'beginner': {'depth': 1, 'beam': 1, 'mistake': 0.40, 'pps': 0.7,
                 'reaction': 0.40, 'label': 'beginner (~C rank pace)'},
    'intermediate': {'depth': 1, 'beam': 6, 'mistake': 0.25, 'pps': 0.9,
                     'reaction': 0.30, 'label': 'intermediate (~B rank pace)'},
    'advanced': {'depth': 2, 'beam': 6, 'mistake': 0.12, 'pps': 1.1,
                 'reaction': 0.20, 'label': 'advanced (~A rank pace)'},
    'expert': {'depth': 3, 'beam': 6, 'mistake': 0.04, 'pps': 1.4,
               'reaction': 0.12, 'label': 'expert (~S rank pace)'},
    'max': {'depth': 3, 'beam': 6, 'mistake': 0.0, 'pps': 6.0,
            'reaction': 0.0, 'label': 'max (no handicap)'},
}
DEFAULT_DIFFICULTY = 'advanced'

# Named checkpoints. ``best`` is the best-measuring checkpoint from the main
# training run; the ``*_finetuned`` entries come from tools/finetune.py, which
# continues from it at a lower learning rate and an re-annealed epsilon.
CHECKPOINT_NAMES = {
    'best': 'best.pt',
    'latest': 'latest.pt',
    'best_finetuned': 'best_finetuned.pt',
    'latest_finetuned': 'latest_finetuned.pt',
    'best_rewardfix': 'best_rewardfix.pt',
    'rewardfix': 'rewardfix.pt',
}

# Menu order: strongest measured first, so the best policy is the obvious pick.
PREFERRED_ORDER = ('best_rewardfix', 'best_finetuned', 'best',
                   'rewardfix', 'latest_finetuned', 'latest')


def _checkpoint_path(name):
    from . import CHECKPOINT_DIR
    if name in CHECKPOINT_NAMES:
        return os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAMES[name])
    return name


def make_versus_policy(difficulty=DEFAULT_DIFFICULTY, weights='versus',
                       incoming_fn=None, seed=0, name=None):
    """The bot you play against: garbage-aware search, handicapped.

    ``incoming_fn`` is called for the garbage queued against this board, which the
    search needs to price height by what is about to land on it and to apply its
    survival gate. Pass the battle's ``incoming`` bound to this side.

    The returned policy carries ``pps`` and ``reaction`` so a caller driving a real
    match can pace it: pieces per second, and a pause after garbage lands before
    the next decision. Nothing in the policy itself sleeps -- timing belongs to the
    loop that owns the clock, exactly as in the rest of this project.
    """
    import random as _random

    from . import search as search_mod
    from .heuristic import heuristic_choice

    preset = HANDICAP_PRESETS[difficulty]
    rng = _random.Random(seed)

    def policy(game, actions):
        incoming = incoming_fn() if incoming_fn else 0
        if preset['mistake'] and rng.random() < preset['mistake']:
            return heuristic_choice(game, actions) or actions[0]
        move = search_mod.search_move(
            game, None, depth=preset['depth'], beam=preset['beam'],
            weights=weights, incoming=incoming)
        if move is None:
            raise RuntimeError('no legal placement')
        return move

    policy.policy_name = name or f'{SEARCH_VERSUS}:{difficulty}'
    policy.difficulty = difficulty
    policy.pps = preset['pps']
    policy.reaction = preset['reaction']
    policy.mistake = preset['mistake']
    policy.depth = preset['depth']
    policy.beam = preset['beam']
    return policy


def list_policies(checkpoint_dir=None):
    """Available policy names: the heuristics plus whichever checkpoints exist."""
    from . import CHECKPOINT_DIR as default_dir
    names = [BUILTIN, LOOKAHEAD, TETRIS_AWARE, SEARCH_TETRIS, SEARCH]
    directory = checkpoint_dir or default_dir
    for label in PREFERRED_ORDER:
        if os.path.exists(os.path.join(directory, CHECKPOINT_NAMES[label])):
            names.append(label)
    return names


def make_search_policy(weights='tetris', depth=3, beam=6, allow_hold=True,
                       name=None):
    """Beam-search policy over ``search.py``'s evaluator.

    ``actions`` is ignored when ``allow_hold`` is set, because the search needs the
    hold swap in its candidate list and a caller that passes only
    ``valid_actions`` cannot offer it. Every move the search returns is checked
    against the engine's own placement rules first, so a hold move is as legal as
    any other.

    ``depth`` is pieces of lookahead and ``beam`` the number of boards kept per
    ply; the shipped default (3, 6) costs roughly 15x the plain heuristic per move.
    """
    from . import search as search_mod

    def policy(game, actions):
        move = search_mod.search_move(
            game, None if allow_hold else actions, depth=depth, beam=beam,
            weights=weights, allow_hold=allow_hold)
        if move is None:
            raise RuntimeError('no legal placement')
        return move

    policy.policy_name = name or (SEARCH_TETRIS if weights == 'tetris' else SEARCH)
    policy.depth = depth
    policy.beam = beam
    policy.allow_hold = allow_hold
    return policy


def make_heuristic_policy(lookahead=0, weights=None, name=None):
    def policy(game, actions):
        from .heuristic import heuristic_choice
        choice = heuristic_choice(game, actions, weights=weights,
                                  sample_next=lookahead)
        if choice is None:
            raise RuntimeError('no legal placement')
        return choice

    policy.policy_name = name or (LOOKAHEAD if lookahead else BUILTIN)
    policy.sample_next = lookahead
    return policy


def make_dqn_policy(checkpoint):
    """Greedy DQN policy from a checkpoint. Imports torch lazily.

    The checkpoint records the feature version and input width it was trained
    with, so a policy built from an older checkpoint still produces the right
    inputs for it rather than crashing on a shape mismatch.
    """
    import numpy as np
    import torch

    from .agent import placement_stats
    from .features import (DEFAULT_VERSION, board_vector_from_metrics,
                           metrics_for, n_features, placement_vector)
    from .model import QNet

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    from .features import VERSIONS
    saved_n = int(ckpt.get('n_features', n_features(1)))
    version = int(ckpt.get('feature_version') or 0)
    if version not in VERSIONS:
        # Older checkpoints did not record the version; the input width pins it
        # down, and using the wrong layout silently produces garbage Q-values
        # (a shape-valid but semantically wrong input).
        version = next((v for v in VERSIONS if n_features(v) == saved_n), None)
        if version is None:
            raise ValueError(
                f'checkpoint {checkpoint} has {saved_n} inputs, which matches no '
                f'feature version (known widths: '
                f'{[n_features(v) for v in VERSIONS]})')

    net = QNet(n_features=saved_n)
    net.load_state_dict(ckpt['policy'])
    net.eval()

    saved_cfg = ckpt.get('cfg') or {}
    reward_cfg = saved_cfg.get('reward', {}) if isinstance(saved_cfg, dict) else {}

    def policy(game, actions):
        current = game.current.kind if game.current is not None else None
        nxt = game.next_queue[0].kind if game.next_queue else None
        base = board_vector_from_metrics(
            metrics_for(game.board.grid, game.rows, game.cols), current, nxt,
            version)
        vectors = [placement_vector(
            base, placement_stats(game, rot, x, reward_cfg)[0], version)
            for rot, x in actions]
        with torch.no_grad():
            q = net(torch.from_numpy(np.stack(vectors))).max(dim=1).values.numpy()
        # Only legal placements are ever evaluated, so this argmax is already
        # restricted to actions the piece can actually perform.
        return actions[int(q.argmax())]

    policy.policy_name = os.path.basename(checkpoint)
    policy.feature_version = version
    return policy


def load_policy(name=None, checkpoint=None):
    """Return ``(policy, description)``.

    Falls back to the heuristic if a checkpoint is missing or fails to load, so
    the caller can always render *something*.
    """
    name = name or BUILTIN
    if checkpoint:
        path = checkpoint
    elif name == BUILTIN:
        return (make_heuristic_policy(),
                'built-in heuristic, 1-ply (no learning)')
    elif name == LOOKAHEAD:
        return (make_heuristic_policy(lookahead=2),
                'built-in heuristic, 2-ply lookahead (no learning)')
    elif name == TETRIS_AWARE:
        return (make_heuristic_policy(weights='tetris_aware'),
                'built-in heuristic, tetris-seeking (no learning)')
    elif name == SEARCH_TETRIS:
        return (make_search_policy(weights='tetris'),
                'beam search, tetris-first (no learning)')
    elif name == SEARCH:
        return (make_search_policy(weights='default'),
                'beam search, board quality only (no learning)')
    else:
        path = _checkpoint_path(name)

    if not os.path.exists(path):
        policy = make_heuristic_policy()
        return policy, f'{name} not found at {path}; using heuristic instead'

    try:
        policy = make_dqn_policy(path)
        return policy, f'DQN from {path}'
    except Exception as exc:
        policy = make_heuristic_policy()
        return policy, f'could not load {path} ({exc}); using heuristic instead'


def dqn_policy_or_none(name=None, checkpoint=None):
    """Like ``load_policy`` but returns None instead of falling back."""
    try:
        policy, desc = load_policy(name, checkpoint)
        if getattr(policy, 'policy_name', None) == BUILTIN:
            return None, desc
        return policy, desc
    except Exception as exc:
        return None, str(exc)
