"""The learning agent: Double DQN over placement actions, with n-step returns.

Actions are "final placements": one of 4 rotations x 10 columns, realised by
rotating, moving and hard-dropping. Illegal placements are masked so they can
never be chosen. One forward pass scores every legal placement for the current
piece, which is what makes this cheap enough to train on a CPU.
"""
from __future__ import annotations

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .engine import COLS, valid_actions
from .features import (DEFAULT_VERSION, N_FEATURES_V1, VERSIONS,
                       board_vector_from_metrics, check_fingerprint,
                       fingerprint, holes_of, metrics_for, n_features,
                       placement_vector)
from .model import N_ACTIONS, QNet

NEG_INF = -1e9


def action_id(rotation, x):
    return rotation * COLS + x


def action_from_id(idx):
    return divmod(idx, COLS)


def valid_mask(actions):
    """Bool mask of length N_ACTIONS marking legal actions."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for rot, x in actions:
        mask[action_id(rot, x)] = True
    return mask


# Reward presets, so a scheme can be A/B'd with one config line.
#
# The numbers are not arbitrary. The key quantity is the *tetris-to-step ratio*:
# building a tetris costs roughly 12 placements (about 9 assorted pieces plus the
# I), so a tetris should be worth on the order of 12 steps of survival or it
# cannot pay for its own setup. Measured on an early version: tetris 100 against
# a step cost of -1 gave 1.6% tetrises and 72.7% singles.
#
#   v1_original   tetris/step = 100   <- not worth building a well for
#   michielcox    tetris/step = 1200  <- published agent clearing 1000+ lines
#   strong_death  as michielcox with a much larger terminal penalty
#   tetris_heavy  michielcox with an even stronger multi-line skew
REWARD_PRESETS = {
    'v1_original': {
        'survival_bonus': -1.0, 'single': 10.0, 'double': 30.0,
        'triple': 60.0, 'tetris': 100.0, 'death_penalty': 10.0,
    },
    'michielcox': {
        'survival_bonus': 1.0, 'single': 40.0, 'double': 100.0,
        'triple': 300.0, 'tetris': 1200.0, 'death_penalty': 5.0,
    },
    'strong_death': {
        'survival_bonus': 1.0, 'single': 40.0, 'double': 100.0,
        'triple': 300.0, 'tetris': 1200.0, 'death_penalty': 99.0,
    },
    'tetris_heavy': {
        'survival_bonus': 1.0, 'single': 20.0, 'double': 100.0,
        'triple': 500.0, 'tetris': 2400.0, 'death_penalty': 50.0,
    },
    # Same 1200:1 ratio as michielcox but with a *small* absolute scale, and the
    # step term kept negative. Two things vary between the presets above and the
    # original: the ratio, and the magnitude. A large magnitude makes the value
    # function's range huge (a 3000-piece game accumulates ~3000 from step
    # bonuses alone), which is harder to fit. This preset isolates scale.
    'michielcox_small': {
        'survival_bonus': -1.0, 'single': 40.0, 'double': 100.0,
        'triple': 300.0, 'tetris': 1200.0, 'death_penalty': 10.0,
    },
}


def reward_config(cfg):
    """Resolve ``cfg['reward']``, expanding a ``preset`` if one is named."""
    reward = dict(cfg.get('reward') or {})
    name = reward.pop('preset', None)
    if name:
        if name not in REWARD_PRESETS:
            raise ValueError(f'unknown reward preset {name!r}; '
                             f'known: {sorted(REWARD_PRESETS)}')
        merged = dict(REWARD_PRESETS[name])
        merged.update(reward)          # explicit keys still win
        return merged
    return reward


# --- terminal detection ---------------------------------------------------

def _spawn_blocked(grid, rows, cols, buffer_rows):
    """True if no piece could appear, i.e. the board is terminal.

    Mirrors ``Game.spawn_blocked``: the board is finished only when *every*
    piece would collide at its spawn position. The spawn cells are precomputed
    because this runs once per candidate placement.
    """
    for cells in _SPAWN_CELLS:
        free = True
        for x, y in cells:
            if y >= rows or (y >= 0 and grid[y][x] is not None):
                free = False
                break
        if free:
            return False
    return True


def _build_spawn_cells():
    from .engine import PIECES, Piece, spawn_anchor
    out = []
    for kind in PIECES:
        x, y = spawn_anchor(kind)
        out.append(tuple(Piece(kind, x, y, 0).cells()))
    return tuple(out)


_SPAWN_CELLS = _build_spawn_cells()


# --- placement scoring ----------------------------------------------------

def placement_stats(game, rotation, x, reward_cfg=None, before_holes=None,
                    heights=None):
    """Statistics and reward for playing ``(rotation, x)`` now.

    Returns ``(stats, reward, done)``. ``stats`` feeds
    ``features.placement_vector``; ``done`` marks a placement that ends the game.

    ``before_holes`` and ``heights`` are the *current* board's hole count and
    column heights. They describe the board the move is made on, so they are
    identical across every candidate for one piece and are passed in to avoid a
    board scan per candidate.

    The placement's own statistics (``max_height``, ``aggregate``) describe the
    board the placement *produces*, including any line clears, because that is
    what the value function was trained to read. Computing them from the
    pre-clear heights would silently change the agent's inputs.
    """
    grid, cells, cleared_rows = game.simulate_placement(rotation, x)
    rows, cols = game.rows, game.cols
    cleared = len(cleared_rows)

    # Everything about the board the placement *produces* comes from this one
    # memoised metrics call, so the extra v2/v3 features are nearly free.
    (result_heights, result_holes, _result_bump, result_aggregate,
     result_row_trans, result_col_trans, result_wells, result_max,
     result_rows_with_holes) = metrics_for(grid, rows, cols)

    if before_holes is None:
        before_holes = holes_of(game.board.grid, rows, cols)

    # Shaping reads the board the piece was placed on, not the post-clear board.
    if heights is None:
        heights = game.board.column_heights()
    bumpiness = 0
    for i in range(cols - 1):
        bumpiness += abs(heights[i] - heights[i + 1])

    done = _spawn_blocked(grid, rows, cols, game.buffer_rows)
    reward = placement_reward(cleared, heights, result_holes, bumpiness, done,
                              reward_cfg,
                              holes_delta=result_holes - before_holes)

    cleared_set = set(cleared_rows)
    landing_top = min((y for _, y in cells), default=rows)
    landing_bottom = max((y for _, y in cells), default=rows)
    stats = {
        'rows': rows,
        'landing_top': landing_top,
        'landing_bottom': landing_bottom,
        'cleared': cleared,
        # Cells of the placed piece destroyed by the lines it completed.
        'eroded': sum(1 for _c, y in cells if y in cleared_set),
        'holes_created': max(0, result_holes - before_holes),
        'max_height': result_max,
        'aggregate': result_aggregate,
        # Version-2 statistics: board quality after the placement.
        'result_holes': result_holes,
        'result_row_trans': result_row_trans,
        'result_col_trans': result_col_trans,
        'result_wells': result_wells,
        'result_max_height': result_max,
        'holes_delta': result_holes - before_holes,
        'filled_removed': len(cleared_rows) * cols,
        # Version-3 statistics.
        'result_rows_with_holes': result_rows_with_holes,
    }
    return stats, reward, done


def _height_summary(grid, rows, cols):
    """(max height, total filled cells) of a board, in one pass."""
    max_h = 0
    filled = 0
    for x in range(cols):
        for y in range(rows):
            if grid[y][x] is not None:
                filled += 1
                h = rows - y
                if h > max_h:
                    max_h = h
                break
    return max_h, filled


def _holes_of(grid, rows, cols):
    """Empty cells with a filled cell above them, counted per column."""
    return holes_of(grid, rows, cols)

def _spawn_blocked(grid, rows, cols, buffer_rows):
    """True if no piece could appear, i.e. the board is terminal.

    Mirrors ``Game.spawn_blocked``: the board is finished only when *every*
    piece would collide at its spawn position. The spawn cells are precomputed
    because this runs once per candidate placement.
    """
    for cells in _SPAWN_CELLS:
        free = True
        for x, y in cells:
            if y >= rows or (y >= 0 and grid[y][x] is not None):
                free = False
                break
        if free:
            return False
    return True


def _build_spawn_cells():
    from .engine import PIECES, Piece, spawn_anchor
    out = []
    for kind in PIECES:
        x, y = spawn_anchor(kind)
        out.append(tuple(Piece(kind, x, y, 0).cells()))
    return tuple(out)


_SPAWN_CELLS = _build_spawn_cells()


def placement_reward(cleared, heights, holes, bumpiness, done, cfg=None,
                     holes_delta=0):
    """Shaped reward for one placement.

    Two groups of terms:

    * **Progress** -- a small per-step survival bonus, line clears scaled so
      multi-line clears are worth disproportionately more, and a death penalty.
      The per-step bonus also applies gentle pressure to keep the stack low,
      because a lower stack means more steps before dying.
    * **Board quality** -- all zero by default. ``hole_penalty`` charges for the
      *stock* of holes (a state cost, which is what actually predicts topping
      out), while ``new_hole_penalty`` charges only for holes this placement
      created. The distinction matters: a stock penalty teaches "boards with
      holes are bad", a creation penalty teaches "don't make holes", and only
      the former gives any credit for digging out of trouble.

    ``holes_delta`` is the change in hole count caused by this placement.
    """
    c = cfg or {}
    reward = c.get('survival_bonus', 0.0)
    if cleared == 1:
        reward += c.get('single', 10.0)
    elif cleared == 2:
        reward += c.get('double', 30.0)
    elif cleared == 3:
        reward += c.get('triple', 60.0)
    elif cleared >= 4:
        reward += c.get('tetris', 100.0)

    height_penalty = c.get('height_penalty', 0.0)
    if height_penalty:
        reward -= height_penalty * max(heights)
    hole_penalty = c.get('hole_penalty', 0.0)
    if hole_penalty:
        reward -= hole_penalty * holes
    bump_penalty = c.get('bumpiness_penalty', 0.0)
    if bump_penalty:
        reward -= bump_penalty * bumpiness
    new_hole_penalty = c.get('new_hole_penalty', 0.0)
    if new_hole_penalty and holes_delta > 0:
        reward -= new_hole_penalty * holes_delta
    if done:
        reward -= c.get('death_penalty', 10.0)
    return float(reward)


# --- agent ----------------------------------------------------------------

class Agent:
    """Double DQN agent with a target network and masked placement actions."""

    def __init__(self, cfg, device='cpu', feature_version=None):
        self.cfg = cfg
        self.gamma = cfg['gamma']
        self.batch_size = cfg['batch_size']
        self.target_sync = cfg['target_sync']
        self.n_step = cfg.get('n_step', 3)
        self.epsilon = cfg['epsilon_start']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay_steps = cfg['epsilon_decay_steps']
        self.reward_cfg = reward_config(cfg)
        self.device = torch.device(device)
        self.feature_version = int(feature_version if feature_version is not None
                                   else cfg.get('feature_version',
                                                DEFAULT_VERSION))
        self.n_inputs = n_features(self.feature_version)

        self.policy = QNet(n_features=self.n_inputs).to(self.device)
        self.target = QNet(n_features=self.n_inputs).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.env_steps = 0
        self.grad_steps = 0

    # -- inference -------------------------------------------------------

    def score_placements(self, game, actions):
        """Q-values for every candidate placement, from one batched forward pass.

        Each candidate is scored as a (board, placement) pair: the board part is
        computed once and the placement's own statistics are appended per
        candidate. Returns ``(q_values, rewards, dones)`` where ``q_values`` is
        aligned with ``actions`` (i.e. only the legal placements).
        """
        base = self.board_vector(game)
        before_holes = holes_of(game.board.grid, game.rows, game.cols)
        heights = game.board.column_heights()
        vecs = []
        rewards = []
        dones = []
        for rot, x in actions:
            stats, reward, done = placement_stats(
                game, rot, x, self.reward_cfg, before_holes, heights)
            vecs.append(placement_vector(base, stats, self.feature_version))
            rewards.append(reward)
            dones.append(done)
        batch = torch.from_numpy(np.stack(vecs)).to(self.device)
        with torch.no_grad():
            q = self.policy(batch).max(dim=1).values.numpy()
        return q, rewards, dones

    def board_vector(self, game):
        """The board/piece part of the input, shared by every candidate."""
        current = game.current.kind if game.current is not None else None
        nxt = game.next_queue[0].kind if game.next_queue else None
        return board_vector_from_metrics(
            metrics_for(game.board.grid, game.rows, game.cols), current, nxt,
            self.feature_version)

    def choose(self, game, actions, epsilon=0.0, rng=None):
        """Pick an action index (into ``actions``) by epsilon-greedy."""
        rng = rng or random
        if epsilon > 0.0 and rng.random() < epsilon:
            return rng.randrange(len(actions))
        q, _, _ = self.score_placements(game, actions)
        return int(np.argmax(q))

    def best_placement(self, game):
        """Greedy placement choice, or None when there is nothing legal."""
        actions = valid_actions(game.board, game.current)
        if not actions:
            return None
        return actions[self.choose(game, actions, epsilon=0.0)]

    # -- learning --------------------------------------------------------

    def n_step_targets(self, episode):
        """Turn a finished episode into (feature, action, target) examples.

        ``episode`` is a list of
        ``(vec, action_id, reward, next_vec, next_mask, done)`` in time order.
        Each target is the n-step bootstrapped return, with the bootstrap value
        taken from the target network.

        The bootstrap values are computed in ONE batched forward pass rather
        than one per transition -- with n-step returns that is the difference
        between a few hundred forward passes per episode and one.
        """
        n = self.n_step
        length = len(episode)
        # Work out, for each t, how many steps we can accumulate and the index
        # of the bootstrap state (if any).
        boot_vecs = []
        boot_masks = []
        plan = []                      # (n_usable, boot_slot or None)
        for t in range(length):
            total_reward = 0.0
            usable = 0
            boot_slot = None
            for k in range(n):
                if t + k >= length:
                    break
                total_reward += (self.gamma ** k) * episode[t + k][2]
                usable += 1
                if episode[t + k][5]:          # a terminal transition
                    boot_slot = None
                    break
            else:
                if usable == n:
                    last = episode[t + n - 1]
                    if not last[5] and last[4].any():
                        boot_slot = len(boot_vecs)
                        boot_vecs.append(last[3])
                        boot_masks.append(last[4])
            plan.append((total_reward, boot_slot))

        boot_values = []
        if boot_vecs:
            batch = torch.from_numpy(np.stack(boot_vecs)).to(self.device)
            masks = torch.from_numpy(np.stack(boot_masks)).to(self.device)
            with torch.no_grad():
                q = self.target(batch).masked_fill(~masks, NEG_INF).max(dim=1).values
            boot_values = q.cpu().numpy()

        examples = []
        for t, (total_reward, boot_slot) in enumerate(plan):
            target = total_reward
            if boot_slot is not None:
                target += (self.gamma ** n) * float(boot_values[boot_slot])
            examples.append((episode[t][0], episode[t][1], target))
        return examples

    def learn(self, examples, steps):
        """Run ``steps`` gradient updates sampled uniformly from ``examples``."""
        if len(examples) < self.batch_size:
            return None
        losses = []
        device = self.device
        for _ in range(steps):
            batch = random.sample(examples, self.batch_size)
            vecs = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            acts = torch.tensor([b[1] for b in batch], dtype=torch.long,
                                device=device)
            targets = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                                   device=device)
            q = self.policy(vecs).gather(1, acts.unsqueeze(1)).squeeze(1)
            per_sample = self.loss_fn(q, targets)
            loss = per_sample.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.optimizer.step()
            self.grad_steps += 1
            losses.append(loss.item())
            if self.grad_steps % self.target_sync == 0:
                self.target.load_state_dict(self.policy.state_dict())
        return float(np.mean(losses))

    def learn_batch(self, batch):
        """One weighted gradient step on a sampled batch.

        ``batch`` is a list of ``(example, weight, index)``; the weight is the
        importance-sampling correction from prioritized replay (1.0 for a
        uniform buffer). Returns ``(loss, td_errors, indices)`` so the caller can
        write the fresh TD errors back as priorities.
        """
        vecs = torch.from_numpy(np.stack([b[0][0] for b in batch])).to(self.device)
        acts = torch.tensor([b[0][1] for b in batch], dtype=torch.long,
                            device=self.device)
        targets = torch.tensor([b[0][2] for b in batch], dtype=torch.float32,
                               device=self.device)
        weights = torch.tensor([b[1] for b in batch], dtype=torch.float32,
                               device=self.device)

        q = self.policy(vecs).gather(1, acts.unsqueeze(1)).squeeze(1)
        td_error = (q - targets).detach()
        # Weight the per-sample Huber loss, then average. Normalising by the
        # sum of weights keeps the effective learning rate independent of how
        # the weights happen to be scaled.
        per_sample = self.loss_fn(q, targets)
        loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return (loss.item(), td_error.cpu().numpy(),
                [b[2] for b in batch])

    def decay_epsilon(self, steps):
        """Linear decay from epsilon_start to epsilon_end over decay_steps."""
        self.env_steps += steps
        frac = min(1.0, self.env_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def anneal_lr(self):
        """Decay the learning rate over training, then hold it constant.

        The measured failure this addresses: a 300-round run peaked near round
        189 and then declined while the training loss climbed from 1.35 to 17.4.
        Its best checkpoint beat the most recent one by 60% (42.6 against 26.7
        lines/game) even though the later one had 20% more gradient steps. A
        constant learning rate is the usual cause of that pattern in off-policy
        RL -- the critic keeps taking full-size steps while its target
        distribution shifts underneath it.

        Shape follows common practice: hold the initial rate for a warmup
        fraction, then decay linearly to a floor and hold. Set
        ``lr_decay_fraction: 0`` to disable and reproduce the old constant-rate
        behaviour.
        """
        frac = self.cfg.get('lr_decay_fraction', 0.0)
        if not frac:
            return
        total = self.epsilon_decay_steps
        if total <= 0:
            return
        progress = min(1.0, self.env_steps / total)
        warm = self.cfg.get('lr_warmup_fraction', 0.3)
        floor = self.cfg.get('lr_min', self.cfg['lr'] * 0.1)
        if progress <= warm:
            target = self.cfg['lr']
        else:
            t = (progress - warm) / max(1e-9, 1.0 - warm)
            target = self.cfg['lr'] + t * (floor - self.cfg['lr'])
        for group in self.optimizer.param_groups:
            group['lr'] = target
        self.current_lr = target

    # -- imitation -------------------------------------------------------

    def imitate(self, examples, steps, batch_size=256, lr=None):
        """Cross-entropy warm start on (features, chosen action) pairs.

        Uses a masked softmax so the network learns to rank the expert's choice
        above the other *legal* placements only.

        ``lr`` defaults to ``cfg['warmstart_lr']`` and is deliberately higher
        than the RL learning rate, with its own optimizer: fitting a supervised
        target needs a larger step than TD learning, which is noisy and
        destabilised by a large rate. Measured on identical data, the RL rate
        (7e-4) reached 78% imitation accuracy while 1e-3 reached 91%, and only
        the latter learned to clear lines.
        """
        if len(examples) < batch_size:
            return None
        if lr is None:
            lr = self.cfg.get('warmstart_lr', 1e-3)
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            batch = random.sample(examples, batch_size)
            vecs = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
            acts = torch.tensor([b[1] for b in batch], dtype=torch.long,
                                device=self.device)
            masks = torch.from_numpy(np.stack([b[2] for b in batch])).to(self.device)
            logits = self.policy(vecs).masked_fill(~masks, NEG_INF)
            loss = nn.functional.cross_entropy(logits, acts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
        # The RL optimizer must start fresh from the warm-started weights.
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.cfg['lr'])
        self.target.load_state_dict(self.policy.state_dict())
        return float(np.mean(losses))

    def imitation_accuracy(self, examples, batch_size=1024):
        """Fraction of examples where the greedy action matches the expert's."""
        if not examples:
            return 0.0
        sample = examples if len(examples) <= batch_size else \
            random.sample(examples, batch_size)
        vecs = torch.from_numpy(np.stack([b[0] for b in sample])).to(self.device)
        acts = torch.tensor([b[1] for b in sample], dtype=torch.long,
                            device=self.device)
        masks = torch.from_numpy(np.stack([b[2] for b in sample])).to(self.device)
        with torch.no_grad():
            pred = self.policy(vecs).masked_fill(~masks, NEG_INF).argmax(dim=1)
            return float((pred == acts).float().mean().item())

    # -- persistence -----------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'env_steps': self.env_steps,
            'grad_steps': self.grad_steps,
            'n_features': self.n_inputs,
            'feature_version': self.feature_version,
            'feature_fingerprint': fingerprint(self.feature_version),
            'n_actions': N_ACTIONS,
            'cfg': self.cfg,
        }, path)

    def load(self, path, load_optimizer=True):
        """Load weights, rebuilding the network if the feature width differs.

        A checkpoint records the input width and feature version it was trained
        with. Loading a wider or narrower one would otherwise fail deep inside
        ``load_state_dict`` with a shape error, so the mismatch is detected here
        and reported in terms of feature versions.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_n = int(ckpt.get('n_features', N_FEATURES_V1))
        if saved_n != self.n_inputs:
            saved_version = ckpt.get('feature_version')
            self.feature_version = (int(saved_version) if saved_version
                                    else _guess_version(saved_n))
            self.n_inputs = n_features(self.feature_version)
            self.policy = QNet(n_features=self.n_inputs).to(self.device)
            self.target = QNet(n_features=self.n_inputs).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=self.cfg['lr'])
            load_optimizer = False        # shapes changed: old state is invalid
        self.policy.load_state_dict(ckpt['policy'])
        self.target.load_state_dict(ckpt.get('target', ckpt['policy']))
        # A same-width but different-layout checkpoint loads cleanly and then
        # plays badly. Say so rather than letting it look like a training result.
        warning = check_fingerprint(ckpt.get('feature_fingerprint'),
                                    self.feature_version)
        if warning:
            self.feature_warning = warning
            print(f'warning: {path}: {warning}')
        if load_optimizer and 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError):
                pass
        self.epsilon = ckpt.get('epsilon', self.epsilon_end)
        self.env_steps = ckpt.get('env_steps', 0)
        self.grad_steps = ckpt.get('grad_steps', 0)
        return ckpt


def _guess_version(n_inputs):
    """Infer the feature version from an input width, for old checkpoints."""
    for version in VERSIONS:
        if n_features(version) == n_inputs:
            return version
    raise ValueError(f'no feature version has {n_inputs} inputs; '
                     f'known widths are '
                     f'{[n_features(v) for v in VERSIONS]}')
