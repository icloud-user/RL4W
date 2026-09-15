"""Tests for the learning machinery: n-step targets, replay sampling, learning.

Run: python tests/test_train.py
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import load_config
from tetrisrl.agent import Agent
from tetrisrl.features import N_FEATURES
from tetrisrl.replay import PrioritizedReplay, Replay, make_replay
from tetrisrl.train import _to_targets

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


def _agent():
    cfg = load_config()
    acfg = dict(cfg['agent'])
    acfg['reward'] = cfg['reward']
    return Agent(acfg)


def _vec(seed, width=None):
    """A dummy feature vector.

    ``width`` defaults to the configured agent's input width, so these tests
    follow the config rather than a hardcoded feature version (which is exactly
    the mismatch that once crashed every training run).
    """
    if width is None:
        width = _agent().n_inputs
    rng = np.random.RandomState(seed)
    return rng.randn(width).astype(np.float32)


class _ZeroNet(torch.nn.Module):
    """A network that always outputs zero, so targets are pure reward sums."""

    def forward(self, x):
        return torch.zeros(x.shape[0], 40)


def test_n_step_target_math():
    """Targets must equal the discounted reward sum plus the n-step bootstrap.

    With a zero network the bootstrap term vanishes, leaving only the reward
    sums, which can be checked by hand.
    """
    print('n-step targets:')
    gamma, n = 1.0, 2
    mask = np.ones(40, dtype=bool)
    # rewards 1, 2, 3, 4 over four non-terminal steps
    episode = [(_vec(i), i, float(i + 1), 0, False, mask) for i in range(4)]
    out = _to_targets(episode, _ZeroNet(), gamma, n)
    targets = [t for _v, _a, t in out]
    # t=0 -> r0 + r1 = 3 ; t=1 -> r1 + r2 = 5 ; t=2 -> r2 + r3 = 7 ; t=3 -> r3 = 4
    check('targets are the 2-step reward sums', targets == [3.0, 5.0, 7.0, 4.0],
          targets)
    check('every transition produced a target', len(out) == 4, len(out))

    # A terminal inside the window stops accumulation (the terminal
    # transition's own reward still counts) and drops the bootstrap.
    episode2 = [(_vec(i), i, float(i + 1), 0, i == 1, mask) for i in range(4)]
    targets2 = [t for _v, _a, t in _to_targets(episode2, _ZeroNet(), gamma, n)]
    # t=0 -> 1 + 2 = 3 (terminal at t=1, so no bootstrap)
    # t=1 -> 2 (terminal itself)
    # t=2 -> 3 + 4 = 7 ; t=3 -> 4
    check('a terminal inside the window stops accumulation',
          targets2 == [3.0, 2.0, 7.0, 4.0], targets2)

    # gamma < 1 discounts correctly.
    episode3 = [(_vec(i), i, 1.0, 0, False, mask) for i in range(3)]
    targets3 = [t for _v, _a, t in _to_targets(episode3, _ZeroNet(), 0.5, 3)]
    # t=0 -> 1 + 0.5 + 0.25 = 1.75 ; t=1 -> 1 + 0.5 = 1.5 ; t=2 -> 1.0
    check('gamma discounts the reward window',
          [round(t, 6) for t in targets3] == [1.75, 1.5, 1.0], targets3)


def test_bootstrap_uses_target_net():
    """A non-zero value network must contribute exactly gamma^n * V(successor)."""
    print('bootstrap term:')
    gamma, n = 0.9, 2

    class Const(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x):
            return torch.full((x.shape[0], 40), self.value)

    mask = np.ones(40, dtype=bool)
    episode = [(_vec(i), i, 0.0, 0, False, mask) for i in range(3)]
    targets = [t for _v, _a, t in _to_targets(episode, Const(10.0), gamma, n)]
    # All rewards are zero, so a target is gamma^steps * 10 for however many
    # steps the window actually consumed.
    #   t=0: consumes 2 transitions, lands in state 2 -> 0.81 * 10 = 8.1
    #   t=1: consumes transitions 1 and 2, lands in state 3, which does not
    #        exist, so there is nothing to bootstrap from -> 0
    #   t=2: consumes 1 transition, lands in state 3, again absent -> 0
    check('bootstrap is gamma^steps * V(successor state)',
          [round(t, 6) for t in targets] == [8.1, 0.0, 0.0], targets)

    # With a longer episode the middle steps do bootstrap.
    episode_long = [(_vec(i), i, 0.0, 0, False, mask) for i in range(6)]
    targets_long = [t for _v, _a, t in
                    _to_targets(episode_long, Const(10.0), gamma, n)]
    check('interior steps bootstrap from their successor',
          [round(t, 6) for t in targets_long] == [8.1, 8.1, 8.1, 8.1, 0.0, 0.0],
          targets_long)


def test_replay_sampling():
    print('uniform replay:')
    buf = Replay(capacity=100, interesting_frac=0.25)
    for i in range(100):
        buf.push((_vec(i), i % 40, float(i)), interesting=(i % 10 == 0))
    check('buffer holds its capacity', len(buf) == 100, len(buf))
    check('10% of entries are flagged interesting', len(buf.interesting) == 10,
          len(buf.interesting))

    random.seed(0)
    batch = buf.sample(40)
    check('sample returns the requested size', len(batch) == 40, len(batch))
    check('sample yields (example, weight, index) triples',
          all(len(b) == 3 and isinstance(b[1], float) and isinstance(b[2], int)
              for b in batch))
    flagged = {(v.tobytes(), a, t) for v, a, t in
               [e for i, e in enumerate(buf.data) if i % 10 == 0]}
    hits = sum(1 for ex, _w, _i in batch
               if (ex[0].tobytes(), ex[1], ex[2]) in flagged)
    check('rare events are oversampled into the batch', hits >= 5, hits)
    check('uniform replay reports unit weights',
          all(w == 1.0 for _e, w, _i in batch))

    # Ring-buffer wraparound must not leave dangling indices.
    for i in range(200):
        buf.push((_vec(1000 + i), i % 40, 0.0), interesting=True)
    buf.rebuild_index()
    check('no stale interesting indices remain',
          all(i < len(buf.data) for i in buf.interesting),
          max(buf.interesting, default=0))
    check('sampling still works after wraparound', len(buf.sample(32)) == 32)


def test_prioritized_replay():
    """Prioritized replay must favour high-error transitions and weight them."""
    print('prioritized replay:')
    buf = PrioritizedReplay(capacity=500, alpha=0.6, beta_start=0.4,
                            beta_steps=1000)
    for i in range(500):
        buf.push((_vec(i), i % 40, 0.0), interesting=False)
    check('buffer holds its capacity', len(buf) == 500, len(buf))

    # Give 50 transitions a much larger error than the rest.
    hot = list(range(0, 500, 10))
    buf.update_priorities(hot, [1e3] * len(hot))

    random.seed(1)
    hits = 0
    trials = 40
    for _ in range(trials):
        batch = buf.sample(64)
        hits += sum(1 for _ex, _w, i in batch if i in set(hot))
    # The hot set is 10% of the buffer, so uniform sampling would give ~6.4 per
    # batch; prioritisation should raise that substantially.
    expected_uniform = trials * 64 * 0.1
    check('high-error transitions are sampled more often than uniform',
          hits > expected_uniform * 1.5,
          f'{hits} vs uniform expectation {expected_uniform:.0f}')

    check('importance weights are in (0, 1]',
          all(0.0 < w <= 1.0 for _e, w, _i in buf.sample(32)))
    check('at least one weight is 1.0 (the batch maximum)',
          any(abs(w - 1.0) < 1e-6 for _e, w, _i in buf.sample(32)))

    # beta must anneal towards 1 as sampling proceeds.
    beta_before = buf.beta
    for _ in range(1200):
        buf.sample(8)
    check('beta anneals towards 1.0', buf.beta > beta_before and buf.beta <= 1.0,
          f'{beta_before:.3f} -> {buf.beta:.3f}')

    check('make_replay builds a prioritized buffer from config',
          isinstance(make_replay({'replay': 'prioritized', 'buffer_size': 10}),
                     PrioritizedReplay))
    check('make_replay builds a uniform buffer from config',
          isinstance(make_replay({'replay': 'uniform', 'buffer_size': 10}),
                     Replay))


def test_prioritized_replay_covers_new_transitions():
    """A new transition must be replayable even if never sampled before."""
    print('prioritized replay coverage:')
    buf = PrioritizedReplay(capacity=1000, alpha=0.6)
    for i in range(100):
        buf.push((_vec(i), i % 40, 0.0))
        buf.update_priorities([i], [0.0])       # all errors zero
    # With every priority floored at epsilon, sampling must still work and must
    # reach every stored index eventually.
    seen = set()
    for _ in range(400):
        for _ex, _w, i in buf.sample(32):
            seen.add(i)
    check('every transition can be sampled', len(seen) == 100, len(seen))


def test_learning_updates_weights():
    """A few gradient steps must actually change the network."""
    print('learning:')
    agent = _agent()
    before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    # Targets far from the initial (near-zero) outputs give a clear gradient.
    examples = [(_vec(i), i % 40, 5.0) for i in range(600)]
    losses = [agent.learn(examples, steps=1) for _ in range(5)]
    after = agent.policy.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    check('gradient steps modify the weights', changed)
    check('loss is finite', all(l is not None and np.isfinite(l) for l in losses),
          losses)
    check('gradient step count advances', agent.grad_steps >= 5, agent.grad_steps)


def test_learning_reduces_loss():
    """Fitting a constant target should reduce the loss substantially."""
    print('loss decreases:')
    agent = _agent()
    examples = [(_vec(i), i % 40, 4.0) for i in range(2000)]
    first = agent.learn(examples, steps=1)
    mid = agent.learn(examples, steps=60)
    last = agent.learn(examples, steps=60)
    check('loss falls over training', last < mid < first,
          f'{first:.4f} -> {mid:.4f} -> {last:.4f}')
    check('loss gets close to the target', abs(last) < 1.0, last)


def test_epsilon_schedule():
    print('epsilon schedule:')
    agent = _agent()
    start, end = agent.epsilon_start, agent.epsilon_end
    check('starts at epsilon_start', abs(agent.epsilon - start) < 1e-9, agent.epsilon)
    agent.decay_epsilon(agent.epsilon_decay_steps // 2)
    check('decays towards the midpoint',
          end < agent.epsilon < start, agent.epsilon)
    agent.decay_epsilon(agent.epsilon_decay_steps)
    check('never falls below epsilon_end', agent.epsilon >= end - 1e-9, agent.epsilon)


def test_agent_config_version_matches_feature_builds():
    """The agent's feature version must drive every feature it consumes.

    Regression test for a real crash: the training workers built features with
    the feature module's *default* version while the agent's config selected
    version 1, so workers returned 44-wide vectors to a 37-input network and
    every run died with "mat1 and mat2 shapes cannot be multiplied" a few rounds
    in. Any code path that constructs features for an agent must use its
    version.
    """
    print('feature version consistency:')
    from tetrisrl.engine import Game, spawn_anchor, valid_actions
    from tetrisrl.features import VERSIONS, n_features
    from tetrisrl.train import _score_all

    for version in VERSIONS:
        cfg = load_config()
        acfg = dict(cfg['agent'])
        acfg['reward'] = cfg['reward']
        acfg['feature_version'] = version
        agent = Agent(acfg)
        expected = n_features(version)
        check(f'v{version}: agent input width is {expected}',
              agent.n_inputs == expected, agent.n_inputs)

        g = Game(seed=2)
        actions = valid_actions(g.board, g.current)
        vectors, _rewards, _dones, best = _score_all(
            agent.policy, g, actions, agent.reward_cfg,
            version=agent.feature_version)
        check(f'v{version}: scored vectors have the network\'s width',
              len(vectors[0]) == agent.n_inputs,
              f'{len(vectors[0])} vs {agent.n_inputs}')
        # The forward pass is the real test: a width mismatch raises here.
        try:
            agent.policy(torch.from_numpy(np.stack(vectors)))
            ok = True
            err = ''
        except RuntimeError as exc:
            ok = False
            err = str(exc)
        check(f'v{version}: network accepts the built features', ok, err)
        check(f'v{version}: best index is in range', 0 <= best < len(actions), best)


def main():
    for fn in (test_n_step_target_math,
               test_bootstrap_uses_target_net,
               test_replay_sampling,
               test_prioritized_replay,
               test_prioritized_replay_covers_new_transitions,
               test_learning_updates_weights,
               test_learning_reduces_loss,
               test_epsilon_schedule,
               test_agent_config_version_matches_feature_builds):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S): {FAILS}')
        return 1
    print('all training tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
