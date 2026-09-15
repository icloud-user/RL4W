"""Tests for the agent's feature/scoring plumbing and the heuristic.

Run: python tests/test_agent.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrisrl import load_config
from tetrisrl.agent import (Agent, action_id, placement_reward,
                            placement_stats, valid_mask)
from tetrisrl.engine import Game, Piece as E_Piece, spawn_anchor, valid_actions
from tetrisrl.features import (DEFAULT_VERSION, board_vector_game, n_features,
                               placement_vector)
from tetrisrl.heuristic import heuristic_choice
from tetrisrl.model import N_ACTIONS, QNet

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


def _configured_version():
    """The feature version the shipped config selects."""
    return int(load_config()['agent'].get('feature_version', DEFAULT_VERSION))


def test_feature_shape():
    print('features:')
    version = _configured_version()
    want = n_features(version)
    g = Game(seed=0)
    base = board_vector_game(g, version)
    actions = valid_actions(g.board, g.current)
    vecs = []
    for rot, x in actions:
        stats, reward, done = placement_stats(g, rot, x, None)
        vec = placement_vector(base, stats, version)
        vecs.append(vec)
    V = np.stack(vecs)
    check('one vector per legal action', len(vecs) == len(actions), len(vecs))
    check(f'v{version} vector length is {want}', V.shape[1] == want, V.shape)
    check('no NaN or inf in features', np.isfinite(V).all())
    check('candidates are distinguishable (not all identical)',
          len({tuple(np.round(v, 5)) for v in vecs}) > 1)
    check('feature values are roughly normalised (<= 3)', np.abs(V).max() <= 3.0,
          float(np.abs(V).max()))


def test_model_output_shape():
    print('model:')
    version = _configured_version()
    net = QNet(n_features=n_features(version))
    out = net(torch.zeros(5, n_features(version)))
    check(f'network emits {N_ACTIONS} Q-values', out.shape == (5, N_ACTIONS), out.shape)
    check('network output is finite', torch.isfinite(out).all())


def test_placement_stats_semantics():
    print('placement stats:')
    g = Game(seed=0)
    actions = valid_actions(g.board, g.current)
    rot, x = actions[0]
    stats, reward, done = placement_stats(g, rot, x, None)
    for key in ('landing_top', 'landing_bottom', 'cleared', 'eroded',
                'holes_created', 'max_height', 'aggregate'):
        check(f'stats contain {key}', key in stats)
    check('landing_top <= landing_bottom',
          stats['landing_top'] <= stats['landing_bottom'],
          (stats['landing_top'], stats['landing_bottom']))
    check('landing is on the board', 0 <= stats['landing_bottom'] < g.rows,
          stats['landing_bottom'])
    check('done is a bool', isinstance(done, bool))

    # A placement that completes a line must report cleared == 1 and eroded == 4
    # (all four cells of a horizontal I are destroyed). The I must actually be
    # the current piece, and sits at its spawn anchor exactly as the agent would
    # place it.
    cfg = load_config()['reward']
    g2 = Game(seed=0)
    for x2 in range(6):
        g2.board.grid[21][x2] = 'L'
    g2.current = E_Piece('I')
    g2.current.x = 6
    g2.current.y = spawn_anchor('I')[1]
    g2.current.rotation = 0
    check('the placement under test clears the bottom row',
          len(g2.simulate_placement(0, 6)[2]) == 1,
          g2.simulate_placement(0, 6)[2])
    stats2, reward2, done2 = placement_stats(g2, 0, 6, cfg)
    check('clearing a line reports cleared == 1', stats2['cleared'] == 1,
          stats2['cleared'])
    check('all 4 cells reported as eroded', stats2['eroded'] == 4, stats2['eroded'])
    check('a line clear earns positive reward', reward2 > 0, reward2)


def test_reward_ordering():
    print('reward:')
    cfg = load_config()['reward']          # the real shaping, not empty defaults
    heights = [0] * 10
    bad_heights = [10] * 10
    good = placement_reward(4, heights, 0, 0, False, cfg)
    plain = placement_reward(0, heights, 0, 0, False, cfg)
    death = placement_reward(0, heights, 0, 0, True, cfg)
    check('tetris rewards more than a normal placement', good > plain, (good, plain))
    check('dying is worse than surviving', death < plain, (death, plain))
    check('per-step penalty applies', plain < 0, plain)
    check('height penalty lowers reward when configured',
          placement_reward(0, bad_heights, 0, 0, False,
                           dict(cfg, height_penalty=1.0)) < plain)


def test_masking_prevents_illegal_actions():
    """The agent must never select a placement outside ``valid_actions``."""
    print('action masking:')
    agent = _agent()
    g = Game(seed=3)
    for _ in range(120):
        if g.game_over:
            break
        actions = valid_actions(g.board, g.current)
        if not actions:
            break
        q, rewards, dones = agent.score_placements(g, actions)
        check_len = len(q) == len(actions)
        if not check_len:
            check('q-values align with legal actions', False, (len(q), len(actions)))
            break
        idx = agent.choose(g, actions, epsilon=0.0)
        if not (0 <= idx < len(actions)):
            check('chosen index is within the legal action list', False, idx)
            break
        rot, x = actions[idx]
        g.current.rotation = rot
        g.current.x = x
        g.current.y = spawn_anchor(g.current.kind)[1]
        g.hard_drop()
    else:
        check('q-values align with legal actions', True)
        check('chosen index is always within the legal action list', True)


def test_masked_bootstrap_ignores_illegal_outputs():
    """n-step bootstrapping must mask illegal placements.

    The network emits 40 Q-values. Outputs for placements that are illegal in a
    given state are never trained, so an unmasked max could pick an arbitrarily
    large value and poison every target.
    """
    print('masked bootstrap:')

    class HugeOnIllegal(torch.nn.Module):
        """Emits an enormous Q-value for one action id.

        Real networks do this too: outputs for placements that are illegal in a
        given state are never trained, so they are free to drift anywhere. The
        poisoned id is chosen so that it is illegal in *every* state of the test
        episode, otherwise a large value would legitimately propagate.
        """

        def __init__(self, poisoned):
            super().__init__()
            self.poisoned = poisoned

        def forward(self, x):
            out = torch.zeros(x.shape[0], N_ACTIONS)
            out[:, self.poisoned] = 1e6
            return out

    # Build the episode first so we can find an always-illegal action id.
    from tetrisrl.train import _to_targets

    g = Game(seed=1)
    episode = []
    while not g.game_over and len(episode) < 12:
        actions = valid_actions(g.board, g.current)
        if not actions:
            break
        mask = valid_mask(actions)
        base = board_vector_game(g)
        stats, reward, done = placement_stats(g, *actions[0], None)
        vec = placement_vector(base, stats)
        episode.append((vec, action_id(*actions[0]), reward, 0, done, mask))
        g.current.rotation, g.current.x = actions[0]
        g.current.y = spawn_anchor(g.current.kind)[1]
        g.hard_drop()

    ever_legal = set()
    for e in episode:
        ever_legal.update(i for i, ok in enumerate(e[5]) if ok)
    never_legal = [i for i in range(N_ACTIONS) if i not in ever_legal]
    check('the episode has an action that is never legal', bool(never_legal),
          f'{len(never_legal)} candidates')
    if not never_legal:
        return

    poisoned = never_legal[0]
    check('the poisoned action was never played',
          all(e[1] != poisoned for e in episode))
    targets = _to_targets(episode, HugeOnIllegal(poisoned), 0.99, 3)
    finite = all(abs(t) < 1e5 for _v, _a, t in targets)
    check('targets ignore Q-values of illegal actions', finite,
          [round(t, 1) for _v, _a, t in targets][:5])


def test_heuristic_clears_lines():
    """The heuristic baseline must actually clear lines -- it is a teacher."""
    print('heuristic baseline:')
    from tetrisrl.heuristic import apply_move

    total_lines = 0
    pieces = 0
    for seed in range(3):
        g = Game(seed=seed)
        while not g.game_over and g.pieces_placed < 300:
            actions = valid_actions(g.board, g.current)
            if not actions:
                break
            # heuristic_choice returns (use_hold, rotation, x); apply_move takes
            # either that or a bare (rotation, x).
            apply_move(g, heuristic_choice(g, actions))
        total_lines += g.lines
        pieces += g.pieces_placed
    check('heuristic clears a substantial number of lines', total_lines > 100,
          f'{total_lines} lines over 3 games')
    check('heuristic survives a long time', pieces > 500, pieces)


def test_hold_is_available_but_off_by_default():
    """Hold must work when asked for, and stay off unless asked.

    Measured: holding never helps this evaluator (the board after "hold then
    place" equals placing directly), and it made the clear mix worse in every
    configuration tried. So it is implemented and available, but opt-in.
    """
    print('hold:')
    from tetrisrl.heuristic import apply_move, hold_options

    g = Game(seed=0)
    opts = hold_options(g)
    check('hold_options offers hold moves', any(o[0] for o in opts), len(opts))
    check('hold_options offers non-hold moves', any(not o[0] for o in opts))

    # Default: no hold.
    g2 = Game(seed=0)
    before = g2.pieces_placed
    for _ in range(40):
        if g2.game_over or not valid_actions(g2.board, g2.current):
            break
        move = heuristic_choice(g2, valid_actions(g2.board, g2.current))
        check_len = len(move) == 3
        if not check_len:
            check('heuristic returns a 3-tuple', False, move)
            break
        apply_move(g2, move)
    else:
        check('heuristic returns a 3-tuple', True)

    g3 = Game(seed=0)
    holds = 0
    for _ in range(120):
        if g3.game_over or not valid_actions(g3.board, g3.current):
            break
        move = heuristic_choice(g3, valid_actions(g3.board, g3.current))
        holds += 1 if move[0] else 0
        apply_move(g3, move)
    check('heuristic does not hold by default', holds == 0, holds)

    # apply_move must cope with a stale hold request (hold already used).
    g4 = Game(seed=0)
    g4.hold()
    apply_move(g4, (True, 0, 0))
    check('apply_move survives a hold that cannot happen', not g4.game_over)


def test_feature_layouts_are_complete_and_finite():
    """Every feature slot must be written, for every version.

    Regression test for a real bug: ``placement_vector`` allocated with
    ``np.empty`` and, for one feature version, left the highest-weight slot
    unwritten. The network read whatever the allocator had left behind -- values
    around 8.8e12 -- and an otherwise-trained checkpoint dropped from ~130 lines
    per game to ~60 with no error anywhere. Any unwritten slot must be zero and
    every value must be finite.
    """
    print('feature layout completeness:')
    from tetrisrl.features import VERSIONS, n_features

    g = Game(seed=4)
    for _ in range(25):
        if g.game_over or g.current is None:
            break
        acts = valid_actions(g.board, g.current)
        if not acts:
            break
        _v, _r, _d, idx = None, None, None, 0
        rot, x = acts[0]
        g.current.rotation, g.current.x = rot, x
        g.current.y = spawn_anchor(g.current.kind)[1]
        g.hard_drop()

    if g.current is None or not valid_actions(g.board, g.current):
        return
    actions = valid_actions(g.board, g.current)

    for version in VERSIONS:
        base = board_vector_game(g, version)
        before_holes = None
        heights = None
        vecs = []
        for rot, x in actions:
            stats, _reward, _done = placement_stats(g, rot, x, None,
                                                    before_holes, heights)
            vecs.append(placement_vector(base, stats, version))
        V = np.stack(vecs)
        check(f'v{version}: vector width is {n_features(version)}',
              V.shape[1] == n_features(version), V.shape)
        check(f'v{version}: all values finite', np.isfinite(V).all())
        check(f'v{version}: no absurd magnitudes (|x| < 100)',
              np.abs(V).max() < 100, float(np.abs(V).max()))
        # A slot that is never written is the bug this test exists for.
        constant = [i for i in range(V.shape[1])
                    if np.allclose(V[:, i], V[0, i]) and abs(V[0, i]) > 50]
        check(f'v{version}: no slot carries a garbage constant', not constant,
              constant)


def test_fingerprint_detects_layout_change():
    print('feature fingerprint:')
    from tetrisrl.features import check_fingerprint, fingerprint
    fp = fingerprint(2)
    check('fingerprint is stable', fingerprint(2) == fp)
    check('versions have different fingerprints',
          fingerprint(1) != fingerprint(2))
    check('matching fingerprint reports no problem',
          check_fingerprint(fp, 2) is None)
    warning = check_fingerprint('deadbeef0000', 2)
    check('mismatched fingerprint reports a problem', warning is not None)
    check('the warning explains the consequence',
          warning is not None and 'play worse' in warning)
    check('a missing fingerprint is tolerated', check_fingerprint(None, 2) is None)


def test_renderer_coercion_accepts_hold_moves():
    """The renderer must read a move's rotation/column, not its hold flag.

    Regression test for a real bug: `_coerce` unpacked `choice[0], choice[1]`
    unconditionally. Once the heuristic started returning `(use_hold, rot, x)`,
    that gave `rot=False`, which never matched a legal action, so the renderer
    fell back to `actions[0]` on every piece -- it watched the agent play the
    first legal move forever and die almost immediately.
    """
    print('renderer move coercion:')
    try:
        from tetrisrl import render  # needs pygame
    except ImportError as exc:
        print(f'  skip (pygame unavailable: {exc})')
        return

    actions = [(0, 3), (1, 5), (2, 7)]
    check('bare (rotation, x) is honoured',
          render.AgentPlayer._coerce((2, 7), actions) == (2, 7))
    check('(use_hold, rotation, x) is honoured',
          render.AgentPlayer._coerce((False, 1, 5), actions) == (1, 5))
    check('a hold move still yields its placement',
          render.AgentPlayer._coerce((True, 0, 3), actions) == (0, 3))
    check('an illegal pick falls back to the first action',
          render.AgentPlayer._coerce((9, 9), actions) == actions[0])
    check('a malformed pick falls back to the first action',
          render.AgentPlayer._coerce(None, actions) == actions[0])

    # And the heuristic's real output must survive coercion.
    g = Game(seed=0)
    acts = valid_actions(g.board, g.current)
    move = heuristic_choice(g, acts)
    got = render.AgentPlayer._coerce(move, acts)
    check('real heuristic output coerces to a legal placement',
          got in acts, (move, got))


def main():
    for fn in (test_feature_shape,
               test_model_output_shape,
               test_placement_stats_semantics,
               test_reward_ordering,
               test_masking_prevents_illegal_actions,
               test_masked_bootstrap_ignores_illegal_outputs,
               test_feature_layouts_are_complete_and_finite,
               test_fingerprint_detects_layout_change,
               test_heuristic_clears_lines,
               test_hold_is_available_but_off_by_default,
               test_renderer_coercion_accepts_hold_moves):
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S): {FAILS}')
        return 1
    print('all agent tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
