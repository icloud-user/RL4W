"""Replay buffers.

Two buffers with the same interface:

``Replay``
    Uniform sampling with a separate index of "interesting" transitions (line
    clears and terminal states). Most transitions in Tetris are identical and
    boring -- "placed a piece, cleared nothing, still alive" -- and they carry
    almost no signal, so guaranteeing a fraction of every batch comes from the
    informative minority speeds up the discovery of line clearing.

``PrioritizedReplay``
    Proportional prioritisation by TD error (Schaul et al., 2016). Transitions
    the network currently predicts badly are replayed more often, which is the
    standard fix for a uniform buffer spending nearly all its capacity on states
    it has already learned. Importance-sampling weights correct the bias.

``make_replay`` picks one from config so callers do not care which.
"""
from __future__ import annotations

import random

import numpy as np


class Replay:
    """Uniform replay with rare-event oversampling."""

    def __init__(self, capacity, interesting_frac=0.25):
        self.capacity = capacity
        self.interesting_frac = interesting_frac
        self.data = []                 # ring buffer
        self.next_idx = 0
        self.interesting = []          # indices into self.data
        self._interesting = set()

    def __len__(self):
        return len(self.data)

    def _mark(self, idx):
        self.interesting.append(idx)
        self._interesting.add(idx)

    def _unmark(self, idx):
        if idx in self._interesting:
            self._interesting.discard(idx)
            try:
                self.interesting.remove(idx)
            except ValueError:
                pass

    def push(self, example, interesting=False):
        """Add an example; flag rare/high-value ones for oversampling."""
        if len(self.data) < self.capacity:
            idx = len(self.data)
            self.data.append(example)
        else:
            idx = self.next_idx
            self._unmark(idx)
            self.data[idx] = example
        self.next_idx = (idx + 1) % self.capacity
        if interesting:
            self._mark(idx)

    def extend(self, examples, interesting_flags=None):
        if interesting_flags is None:
            interesting_flags = [False] * len(examples)
        for ex, flag in zip(examples, interesting_flags):
            self.push(ex, flag)

    def sample(self, n):
        """Return a batch as ``(example, weight, index)`` triples.

        The triples match ``PrioritizedReplay`` so the learner can be written
        once for either buffer.
        """
        size = len(self.data)
        if size <= n:
            return [(ex, 1.0, i) for i, ex in enumerate(self.data)]
        want_interesting = int(n * self.interesting_frac)
        # Indices in `interesting` may be stale after ring-buffer wraparound.
        pool = [i for i in self.interesting if i < size]
        picked = []
        if pool and want_interesting:
            if len(pool) >= want_interesting:
                picked = random.sample(pool, want_interesting)
            else:
                picked = list(pool)
        chosen = set(picked)
        while len(chosen) < n:
            chosen.add(random.randrange(size))
        return [(self.data[i], 1.0, i) for i in chosen]

    def update_priorities(self, indices, errors):
        """No-op: uniform replay ignores TD errors."""

    def rebuild_index(self):
        """Drop stale interesting indices (called periodically)."""
        size = len(self.data)
        if len(self.interesting) > 4 * max(1, size):
            self.interesting = [i for i in self.interesting if i < size]
            self._interesting = set(self.interesting)


class PrioritizedReplay:
    """Proportional prioritized replay with importance sampling.

    A transition's sampling probability is ``p_i^alpha / sum(p^alpha)`` where
    ``p_i`` is its last TD error plus a small floor, so a transition that was
    once predicted perfectly can still be revisited. Each batch carries weights
    ``(1 / (N * P(i)))^beta`` normalised by the batch maximum, which keeps the
    gradient an unbiased estimate of the uniform-batch gradient as ``beta -> 1``.

    Priorities are held in a flat numpy array and sampled by inverse-CDF. A
    Fenwick tree is the textbook structure and gives O(log N) updates, but a
    cumulative sum over <=1M floats is fast enough here and far easier to check.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=200000,
                 epsilon=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = max(1, beta_steps)
        self.epsilon = epsilon
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.next_idx = 0
        self.max_priority = 1.0
        self._samples = 0

    def __len__(self):
        return len(self.data)

    @property
    def beta(self):
        frac = min(1.0, self._samples / self.beta_steps)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def push(self, example, interesting=False):
        if len(self.data) < self.capacity:
            idx = len(self.data)
            self.data.append(example)
        else:
            idx = self.next_idx
            self.data[idx] = example
        self.next_idx = (idx + 1) % self.capacity
        # New transitions enter at the current maximum priority so that every
        # transition is replayed at least once.
        self.priorities[idx] = self.max_priority

    def extend(self, examples, interesting_flags=None):
        for ex in examples:
            self.push(ex)

    def sample(self, n):
        size = len(self.data)
        if size <= n:
            self._samples += 1
            return [(ex, 1.0, i) for i, ex in enumerate(self.data)]

        scaled = np.power(self.priorities[:size], self.alpha)
        total = scaled.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(size, 1.0 / size)
        else:
            probs = scaled / total

        idxs = np.random.choice(size, size=n, p=probs, replace=False)
        self._samples += 1

        beta = self.beta
        weights = np.power(size * probs[idxs], -beta)
        weights = weights / weights.max()
        return [(self.data[int(i)], float(weights[k]), int(i))
                for k, i in enumerate(idxs)]

    def update_priorities(self, indices, errors):
        """Set new priorities from observed TD errors."""
        for i, err in zip(indices, errors):
            if i is None or i < 0 or i >= self.capacity:
                continue
            p = abs(float(err)) + self.epsilon
            self.priorities[i] = p
            if p > self.max_priority:
                self.max_priority = p

    def rebuild_index(self):
        """Nothing to rebuild: priorities are indexed by slot."""


def make_replay(cfg):
    """Build the replay buffer described by ``cfg``.

    ``cfg['replay']`` selects ``'prioritized'`` or ``'uniform'``.
    """
    kind = str(cfg.get('replay', 'uniform')).lower()
    capacity = cfg['buffer_size']
    if kind in ('prioritized', 'per', 'prioritised'):
        return PrioritizedReplay(
            capacity,
            alpha=cfg.get('per_alpha', 0.6),
            beta_start=cfg.get('per_beta_start', 0.4),
            beta_steps=cfg.get('per_beta_steps', 200000),
        )
    return Replay(capacity, interesting_frac=cfg.get('interesting_frac', 0.25))
