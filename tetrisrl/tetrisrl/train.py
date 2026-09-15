"""Training: parallel self-play workers plus a central learner.

Each round the learner broadcasts the current weights to worker processes; the
workers play games with an epsilon-greedy policy, convert each finished episode
into n-step targets and return them. The learner pushes everything into a replay
buffer and takes gradient steps.

Doing the n-step conversion inside the workers is deliberate: it overlaps that
work with simulation across processes instead of serialising it in the learner.
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .agent import (Agent, NEG_INF, action_id, placement_reward,
                    placement_stats, reward_config, valid_mask)
from .engine import Game, spawn_anchor, valid_actions
from .features import DEFAULT_VERSION, n_features
from .heuristic import heuristic_choice
from .model import QNet
from .replay import make_replay

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints')
BEST_PATH = os.path.join(CHECKPOINT_DIR, 'best.pt')
LATEST_PATH = os.path.join(CHECKPOINT_DIR, 'latest.pt')

# Cap on pieces per training episode. Once the agent plays well, games can run
# for tens of thousands of pieces; uncapped, a single episode would dominate a
# round and training would appear to hang. Truncation is NOT treated as death:
# the episode simply stops and the final transition bootstraps normally.
MAX_PIECES_PER_EPISODE = 3000


def _local_net(state_dict, version=1):
    net = QNet(n_features=n_features(version))
    net.load_state_dict(state_dict)
    net.eval()
    return net


def play_episode(net, epsilon, cfg, rng, seed=None, teacher=None,
                 version=1):
    """Play one game and return (examples, stats).

    ``examples`` are ``(features, action_id, n_step_target)`` triples ready for
    gradient updates; ``stats`` reports the real game outcome.
    """
    gamma = cfg['gamma']
    n_step = cfg.get('n_step', 3)
    reward_cfg = cfg.get('reward', {})

    game = Game(seed=seed)
    episode = []
    while not game.game_over:
        actions = valid_actions(game.board, game.current)
        if not actions:
            break
        if game.pieces_placed >= MAX_PIECES_PER_EPISODE:
            break

        # Score every legal placement. One forward pass covers all candidates.
        if teacher is not None and rng.random() < teacher:
            vectors, rewards, dones, _ = _score_all(net, game, actions,
                                                    reward_cfg, version=version)
            choice = heuristic_choice(game, actions)
            idx = actions.index(choice)
        elif epsilon > 0 and rng.random() < epsilon:
            vectors, rewards, dones, _ = _score_all(net, game, actions,
                                                    reward_cfg, version=version)
            idx = rng.randrange(len(actions))
        else:
            vectors, rewards, dones, idx = _score_all(
                net, game, actions, reward_cfg, version=version)

        rot, x = actions[idx]
        vec = vectors[idx]

        game.current.rotation = rot
        game.current.x = x
        game.current.y = spawn_anchor(game.current.kind)[1]
        before = game.lines
        game.hard_drop()
        # The engine is ground truth for the outcome, not the prediction: a
        # predicted line clear or predicted terminal is only an estimate of the
        # state the placement produces.
        cleared = game.lines - before
        done = game.game_over

        reward = placement_reward(cleared, *_board_stats(game), done, reward_cfg)
        episode.append((vec, action_id(rot, x), reward, cleared, done,
                        valid_mask(actions)))

    stats = {
        'lines': game.lines,
        'score': game.score,
        'pieces': game.pieces_placed,
        'steps': len(episode),
    }
    return _to_targets(episode, net, gamma, n_step), stats


def _board_stats(game):
    """(heights, holes, bumpiness) for the current board, for reward shaping."""
    from .features import metrics_for
    metrics = metrics_for(game.board.grid, game.rows, game.cols)
    heights, holes, bumpiness = metrics[0], metrics[1], metrics[2]
    return heights, holes, bumpiness


def _to_targets(episode, net, gamma, n_step):
    """Convert a finished episode into (features, action_id, target) examples.

    For each step t the target is the discounted sum of the next ``n_step``
    rewards plus the bootstrapped value of the state that follows them.

    Two subtleties this handles:

    * The successor state of step t is step t+1's own feature vector, so no extra
      simulation is needed.
    * The bootstrap is a max over the successor's **legal** actions only. The
      network emits Q-values for 40 placements regardless of the piece, and
      outputs for placements that are illegal in that state are never trained,
      so an unrestricted max could pick an arbitrary large value and poison the
      target. Masks are stored alongside each transition for this reason.
    """
    length = len(episode)
    if length == 0:
        return []

    boot_values = [0.0] * length
    # The value of every state in the episode, each masked to the actions that
    # were legal in *that* state. Note it is the state's own mask that is used,
    # not its successor's: the value of a state is the max over its own moves.
    with torch.no_grad():
        batch = torch.from_numpy(np.stack([episode[i][0] for i in range(length)]))
        masks = torch.from_numpy(np.stack([episode[i][5] for i in range(length)]))
        vals = net(batch).masked_fill(~masks, NEG_INF).max(dim=1).values
    for i in range(length):
        boot_values[i] = float(vals[i].item())

    # A transition at index t holds the state *before* the move at index t, so
    # looking k steps ahead consumes transitions t .. t+k-1 and lands in the
    # state at index t+k. That is the state the bootstrap is taken from.
    out = []
    for t in range(length):
        total = 0.0
        steps = 0                        # transitions consumed by the window
        for k in range(n_step):
            if t + k >= length:
                break                    # ran out of episode
            total += (gamma ** k) * episode[t + k][2]
            steps = k + 1
            if episode[t + k][4]:
                # A real game over inside the window: no future value. The
                # terminal transition's own reward still counts.
                steps = -1
                break
        bootstrap = 0.0
        if steps > 0:
            successor = t + steps
            if successor < length:
                # Truncation is not a terminal state, so this value is real.
                # Dropping it would bias every target in the window.
                bootstrap = (gamma ** steps) * boot_values[successor]
        out.append((episode[t][0], episode[t][1], total + bootstrap))
    return out


def _worker(payload):
    """Entry point for a worker process. Must stay module-level (Windows)."""
    state_dict, epsilon, cfg, episodes, seeds, teacher, version = payload
    torch.set_num_threads(1)          # many workers: per-process threading hurts
    rng = random.Random(seeds[0])
    np.random.seed(seeds[0] % (2 ** 31))
    net = _local_net(state_dict, version)

    examples = []
    stats = []
    for i in range(episodes):
        ex, st = play_episode(net, epsilon, cfg, rng, seed=seeds[i],
                              teacher=teacher, version=version)
        examples.extend(ex)
        stats.append(st)
    return examples, stats


def _score_all(net, game, actions, reward_cfg, base=None,
               version=None):
    """Score every legal placement. Returns ``(vectors, rewards, dones, best)``.

    The board part of the input is computed once and shared across candidates;
    only the placement statistics differ. One forward pass scores them all.

    ``best`` is the index (into ``actions``) of the highest-valued *legal*
    placement. Only legal placements are ever evaluated, so the argmax is
    inherently restricted -- the network's outputs for placements that are
    illegal for this piece are never consulted.

    ``version`` must match the network's input width. It defaults to 1 rather
    than to the feature module's default so that a caller who forgets it fails
    loudly on a shape mismatch instead of silently feeding the wrong layout.
    """
    from .features import (board_vector_from_metrics, holes_of, metrics_for,
                           placement_vector)
    if version is None:
        version = 1
    if base is None:
        current = game.current.kind if game.current is not None else None
        nxt = game.next_queue[0].kind if game.next_queue else None
        base = board_vector_from_metrics(
            metrics_for(game.board.grid, game.rows, game.cols), current, nxt,
            version)
    # These are identical for every candidate, so compute them once per
    # decision rather than once per placement.
    before_holes = holes_of(game.board.grid, game.rows, game.cols)
    heights = game.board.column_heights()
    stats = [placement_stats(game, rot, x, reward_cfg, before_holes, heights)
             for rot, x in actions]
    vectors = [placement_vector(base, s[0], version) for s in stats]
    with torch.no_grad():
        q = net(torch.from_numpy(np.stack(vectors))).max(dim=1).values.numpy()
    return vectors, [s[1] for s in stats], [s[2] for s in stats], int(q.argmax())


def _candidate_vector(game, rot, x, version=1):
    """Full feature vector for one placement (used by the imitation teacher)."""
    from .features import board_vector_game, placement_vector
    stats, _, _ = placement_stats(game, rot, x, None)
    return placement_vector(board_vector_game(game, version), stats, version)


def evaluate(agent, episodes=30, seed0=100000, temperature=0.0):
    """Greedy evaluation. Returns mean/max lines and scores over full games."""
    net = agent.policy
    net.eval()
    reward_cfg = agent.reward_cfg
    version = agent.feature_version
    lines, scores, pieces = [], [], []
    rng = random.Random(seed0)
    for i in range(episodes):
        game = Game(seed=seed0 + i)
        while not game.game_over:
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            if temperature > 0 and rng.random() < temperature:
                idx = rng.randrange(len(actions))
            else:
                _vecs, _r, _d, idx = _score_all(net, game, actions, reward_cfg,
                                                version=version)
            rot, x = actions[idx]
            game.current.rotation = rot
            game.current.x = x
            game.current.y = spawn_anchor(game.current.kind)[1]
            game.hard_drop()
            if game.pieces_placed >= MAX_PIECES_PER_EPISODE:
                break
        lines.append(game.lines)
        scores.append(game.score)
        pieces.append(game.pieces_placed)
    net.train()
    return {
        'lines_mean': float(np.mean(lines)),
        'lines_max': int(np.max(lines)),
        'score_mean': float(np.mean(scores)),
        'score_max': int(np.max(scores)),
        'pieces_mean': float(np.mean(pieces)),
        'games': episodes,
    }


def pretrain_imitation(agent, episodes=40, steps=1200, cfg=None, teacher=1.0,
                       batch_size=256, verbose=False):
    """Warm start the network on the heuristic teacher's choices.

    Each game is capped at ``MAX_PIECES_PER_EPISODE``: the heuristic can play
    thousands of pieces, which would make this phase take longer than the
    training it is meant to accelerate.
    """
    cfg = cfg or agent.cfg
    rng = random.Random(0)
    examples = []
    for i in range(episodes):
        game = Game(seed=500000 + i)
        while not game.game_over:
            actions = valid_actions(game.board, game.current)
            if not actions:
                break
            if game.pieces_placed >= MAX_PIECES_PER_EPISODE:
                break
            if teacher < 1.0 and rng.random() > teacher:
                break
            choice = heuristic_choice(game, actions)
            vec = _candidate_vector(game, choice[0], choice[1],
                                    version=agent.feature_version)
            examples.append((vec, action_id(*choice), valid_mask(actions)))
            game.current.rotation, game.current.x = choice
            game.current.y = spawn_anchor(game.current.kind)[1]
            game.hard_drop()
        if verbose and (i + 1) % 10 == 0:
            print(f'  imitation: {i + 1}/{episodes} games, '
                  f'{len(examples)} examples')
    if not examples:
        return None
    if verbose:
        print(f'  imitation: {len(examples)} examples, {steps} steps '
              f'({steps * batch_size / max(1, len(examples)):.1f} passes)')
    return agent.imitate(examples, steps, batch_size=batch_size)


def train(config, rounds=None, workers=None, episodes_per_worker=None,
          resume=None, quiet=False, tag=None):
    """Train an agent.

    ``tag`` names the output checkpoints (``<tag>.pt`` and ``best_<tag>.pt``).
    Without a tag this uses the canonical ``best.pt`` / ``latest.pt`` names and
    will therefore OVERWRITE existing checkpoints -- so any experiment or short
    test run should pass a tag, and the CLI does that automatically.
    """
    tcfg = config['training']
    acfg = dict(config['agent'])
    acfg['reward'] = reward_config(config)
    rounds = rounds or tcfg['rounds']
    workers = workers or tcfg['workers']
    episodes_per_worker = episodes_per_worker or tcfg['episodes_per_worker']
    eval_every = tcfg.get('eval_every', 25)
    eval_games = tcfg.get('eval_games', 30)
    teacher = tcfg.get('teacher_fraction', 0.0)
    teacher_rounds = tcfg.get('teacher_rounds', 0)

    if tag:
        best_path = os.path.join(CHECKPOINT_DIR, f'best_{tag}.pt')
        latest_path = os.path.join(CHECKPOINT_DIR, f'{tag}.pt')
    else:
        best_path, latest_path = BEST_PATH, LATEST_PATH

    agent = Agent(acfg)
    start_round = 0
    best_lines = -1.0

    if resume and os.path.exists(resume):
        ckpt = agent.load(resume)
        best_lines = ckpt.get('best_lines', -1.0)
        start_round = ckpt.get('round', 0)
        if not quiet:
            print(f'resumed {resume}: round {start_round}, '
                  f'epsilon {agent.epsilon:.3f}, grad steps {agent.grad_steps}')
    elif tcfg.get('imitation_steps', 0) > 0:
        t_imp = time.time()
        loss = pretrain_imitation(
            agent,
            episodes=tcfg.get('imitation_episodes', 40),
            steps=tcfg['imitation_steps'],
            cfg=acfg,
            batch_size=tcfg.get('imitation_batch_size', 256),
            verbose=not quiet,
        )
        if not quiet:
            print(f'imitation warm start: loss {loss:.4f} '
                  f'({time.time() - t_imp:.0f}s)' if loss is not None
                  else 'imitation warm start skipped (no examples)')
        baseline = evaluate(agent, episodes=eval_games)
        best_lines = baseline['lines_mean']
        if not quiet:
            print(f'  after warm start: {baseline["lines_mean"]:.2f} lines/game '
                  f'(max {baseline["lines_max"]}), score {baseline["score_mean"]:.0f}')

    replay = make_replay(acfg)
    rng = random.Random(12345)

    if not quiet:
        print(f'training: rounds={rounds} workers={workers} '
              f'episodes/worker={episodes_per_worker} '
              f'grad_steps/round={tcfg["grad_steps_per_round"]} device=cpu')
        print(f'  lr={acfg["lr"]} gamma={acfg["gamma"]} n_step={acfg["n_step"]} '
              f'eps {acfg["epsilon_start"]}->{acfg["epsilon_end"]} '
              f'over {acfg["epsilon_decay_steps"]} steps')

    t_start = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rnd in range(start_round + 1, rounds + 1):
                t0 = time.time()
                state_dict = {k: v.detach().clone()
                              for k, v in agent.policy.state_dict().items()}
                payloads = []
                for w in range(workers):
                    seeds = [rng.randrange(2 ** 31) for _ in range(episodes_per_worker)]
                    payloads.append((state_dict, agent.epsilon, acfg,
                                     episodes_per_worker, seeds,
                                     teacher if rnd <= teacher_rounds else None,
                                     agent.feature_version))

                all_examples = []
                stats = []
                for examples, st in pool.map(_worker, payloads):
                    all_examples.extend(examples)
                    stats.extend(st)

                # Flag rare, informative transitions so a uniform buffer can
                # oversample them: line clears and terminal states.
                targets = np.array([e[2] for e in all_examples], dtype=np.float32)
                interesting = (targets > 0.5).tolist()
                replay.extend(all_examples, interesting)

                loss = None
                n_steps = tcfg['grad_steps_per_round']
                if tcfg.get('grad_steps_auto', True):
                    # Scale the update budget with how much experience arrived.
                    # With a fixed step count, an improving agent (whose episodes
                    # get longer and longer) collects more and more transitions
                    # per round for the same number of updates -- so the updates
                    # per sample collapse exactly when learning matters most.
                    ratio = tcfg.get('grad_steps_per_sample', 1 / 4)
                    n_steps = max(tcfg.get('min_grad_steps_per_round', 200),
                                  int(len(all_examples) * ratio))
                if len(replay) >= acfg['batch_size'] and n_steps:
                    losses = []
                    for _ in range(n_steps):
                        batch = replay.sample(acfg['batch_size'])
                        step_loss, td_errors, indices = agent.learn_batch(batch)
                        # Prioritized replay needs the fresh TD errors back.
                        replay.update_priorities(indices, td_errors)
                        losses.append(step_loss)
                    loss = float(np.mean(losses))
                agent.decay_epsilon(len(all_examples))
                agent.anneal_lr()

                if rnd % tcfg.get('checkpoint_every', 25) == 0 or rnd == rounds:
                    agent.save(latest_path)

                if rnd % eval_every == 0 or rnd == rounds:
                    ev = evaluate(agent, episodes=eval_games)
                    marker = ''
                    if ev['lines_mean'] > best_lines:
                        best_lines = ev['lines_mean']
                        agent.save(best_path)
                        # Record the score alongside the weights so a resumed
                        # run keeps comparing against the real best instead of
                        # treating its first evaluation as a new record.
                        _stamp_best(best_path, rnd, best_lines)
                        marker = ' *best*'
                    if not quiet:
                        print(f'  eval: {ev["lines_mean"]:.2f} lines/game '
                              f'(max {ev["lines_max"]}), '
                              f'score {ev["score_mean"]:.0f}, '
                              f'pieces {ev["pieces_mean"]:.0f}{marker}')
                else:
                    ev = None

                if not quiet:
                    mean_lines = float(np.mean([s['lines'] for s in stats]))
                    mean_score = float(np.mean([s['score'] for s in stats]))
                    loss_s = f'{loss:.4f}' if loss is not None else '  n/a '
                    print(f'round {rnd:4d}/{rounds} | '
                          f'lines {mean_lines:7.2f} | score {mean_score:8.0f} | '
                          f'eps {agent.epsilon:.3f} | buf {len(replay):7d} | '
                          f'loss {loss_s} | +{len(all_examples):6d} | '
                          f'{time.time() - t0:5.1f}s')

                if rnd % 50 == 0:
                    replay.rebuild_index()
    except KeyboardInterrupt:
        if not quiet:
            print('\ninterrupted - saving and exiting')

    agent.save(latest_path)
    if not quiet:
        print(f'done in {time.time() - t_start:.0f}s. '
              f'checkpoints: {latest_path}, {best_path}')
    return agent


def _stamp_best(path, rnd, lines_mean):
    """Record which round produced the best checkpoint.

    ``Agent.save`` writes the weights; this adds the provenance so ``--resume``
    can restore the running best instead of resetting it.
    """
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        ckpt['round'] = rnd
        ckpt['best_lines'] = lines_mean
        torch.save(ckpt, path)
    except Exception as exc:                      # provenance is not critical
        print(f'warning: could not stamp {path}: {exc}')
