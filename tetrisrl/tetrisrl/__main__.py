"""Command line interface.

    python -m tetrisrl train [--rounds N] [--workers N] [--resume PATH]
    python -m tetrisrl eval  [--policy best|latest|heuristic|PATH] [--games N]
    python -m tetrisrl watch [--policy best] [--fps N]
    python -m tetrisrl play
    python -m tetrisrl battle [--difficulty advanced] [--rounds 3]
    python -m tetrisrl menu
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time

from . import CHECKPOINT_DIR, __version__, load_config
from .evaluate import evaluate_policy, format_summary
from .policies import HANDICAP_PRESETS, list_policies, load_policy
from .versus import MESSINESS


def _default_workers():
    count = multiprocessing.cpu_count() or 4
    return max(1, min(12, count - 1))


def cmd_train(args):
    from .train import train
    config = load_config(args.config)
    if args.workers:
        config['training']['workers'] = args.workers
    else:
        config['training']['workers'] = min(config['training']['workers'],
                                            _default_workers())
    if args.rounds:
        config['training']['rounds'] = args.rounds
    if args.episodes_per_worker:
        config['training']['episodes_per_worker'] = args.episodes_per_worker
    if args.reward_preset:
        config.setdefault('reward', {})['preset'] = args.reward_preset
        print(f'reward preset: {args.reward_preset}')
    if args.epsilon_decay_steps:
        config['agent']['epsilon_decay_steps'] = args.epsilon_decay_steps
        print(f'epsilon decay over {args.epsilon_decay_steps} transitions')
    if args.no_lr_decay:
        config['agent']['lr_decay_fraction'] = 0.0
        print('learning-rate annealing DISABLED (constant lr)')
    if args.lr_decay_fraction is not None:
        config['agent']['lr_decay_fraction'] = args.lr_decay_fraction
        print(f"lr_decay_fraction = {args.lr_decay_fraction}")
    resume = args.resume
    if resume is True:
        import os
        resume = os.path.join(CHECKPOINT_DIR, 'latest.pt')
        if not os.path.exists(resume):
            print(f'--resume: no checkpoint at {resume}; starting fresh')
            resume = None
    # A run writing to the canonical best.pt/latest.pt destroys whatever is
    # already there, so anything other than an explicit --publish run gets its
    # own files. This makes a short test run safe by default.
    tag = args.tag
    if tag is None and not args.publish:
        tag = time.strftime('run%m%d-%H%M%S')
        print(f'checkpoints -> best_{tag}.pt and {tag}.pt')
        print('  (pass --publish to write the canonical best.pt/latest.pt)')
    train(config, rounds=args.rounds, workers=config['training']['workers'],
          episodes_per_worker=args.episodes_per_worker, resume=resume,
          quiet=args.quiet, tag=tag)


def cmd_eval(args):
    policy, desc = load_policy(args.policy, args.checkpoint)
    print(f'policy: {desc}')
    summary = evaluate_policy(policy, episodes=args.games, seed0=args.seed,
                              verbose=args.verbose,
                              max_pieces=args.max_pieces)
    print(format_summary(summary))
    return 0


def cmd_watch(args):
    try:
        from . import render
    except ImportError as exc:
        print(f'rendering needs pygame: {exc}')
        print('  install with: pip install pygame')
        return 1
    config = load_config(args.config)
    if args.fps:
        config.setdefault('game', {})['fps'] = args.fps
    policy, desc = load_policy(args.policy, args.checkpoint)
    print(f'watching: {desc}')
    render.run_agent(policy, config=config, seed=args.seed,
                     fps=args.fps or 30.0)
    return 0


def _open_recorder(args, mode, seed, config):
    """A session recorder for play/battle, or None.

    Recording is opt-in: without ``--record`` nothing is opened, nothing is
    imported beyond what already was, and not a byte is written. ``--no-record``
    wins over everything, so a wrapper script can force it off.
    """
    if getattr(args, 'no_record', False):
        return None
    where = getattr(args, 'record', None)
    if where is None:
        return None
    from . import controls, recorder

    path = None if where in ('', 'default') else where
    settings = controls.load(config)
    session = recorder.SessionRecorder(
        path, meta=recorder.start_meta(mode, seed=seed, settings=settings,
                                       extra={'recorded_from': 'cli'}))
    if session.enabled:
        print(f'recording session to {session.path}')
        print('  (sessions/ is git-ignored; analyze with '
              'python tools/analyze_session.py)')
    else:
        print('recording unavailable: ' + '; '.join(session.warnings))
    return session


def _close_recorder(session):
    """Finish a session and say where it went."""
    if session is None:
        return
    session.close()
    if session.path:
        print(f'session written to {session.path} '
              f'({session.count} events, {session.elapsed():.0f}s)')


def cmd_play(args):
    try:
        from . import render
    except ImportError as exc:
        print(f'rendering needs pygame: {exc}')
        print('  install with: pip install pygame')
        return 1
    config = load_config(args.config)
    session = _open_recorder(args, 'play', args.seed, config)
    try:
        render.run_human(config=config, seed=args.seed, recorder=session)
    finally:
        _close_recorder(session)
    return 0


def cmd_menu(args):
    try:
        from . import render
    except ImportError as exc:
        print(f'rendering needs pygame: {exc}')
        print('  install with: pip install pygame')
        return 1
    render.run_menu(config=load_config(args.config))
    return 0


def cmd_battle(args):
    try:
        from . import render
    except ImportError as exc:
        print(f'rendering needs pygame: {exc}')
        print('  install with: pip install pygame')
        return 1
    config = load_config(args.config)
    session = _open_recorder(args, 'battle', args.seed, config)
    try:
        render.run_battle(config=config, seed=args.seed,
                          difficulty=args.difficulty, messiness=args.messiness,
                          rounds=args.rounds, recorder=session)
    finally:
        _close_recorder(session)
    return 0


def cmd_policies(args):
    from .policies import BUILTIN, LOOKAHEAD, SEARCH, SEARCH_TETRIS, TETRIS_AWARE
    print('available policies:')
    for name in list_policies():
        if name == BUILTIN:
            print('  heuristic        built-in evaluator, 1-ply (no learning)')
        elif name == LOOKAHEAD:
            print('  heuristic2       built-in evaluator, 2-ply lookahead: same '
                  'lines/game, more multi-line clears (~30x slower)')
        elif name == TETRIS_AWARE:
            print('  heuristic_tetris tetris-seeking: ~83% of the lines, ~6x the '
                  'tetris rate (no learning)')
        elif name == SEARCH_TETRIS:
            print('  search_tetris    beam search, 3 pieces deep with hold: 53% of '
                  'clears are tetrises (~15x slower)')
        elif name == SEARCH:
            print('  search           same search, board quality only, no tetris '
                  'shaping (A/B control)')
        else:
            print(f'  {name:16s} checkpoints/{name}.pt')
    return 0


def _add_record_flags(parser):
    """``--record [PATH]`` / ``--no-record``, shared by play and battle."""
    parser.add_argument('--record', nargs='?', const='default', default=None,
                        metavar='PATH',
                        help='record this session as JSONL for later diagnosis '
                             '(default: sessions/<UTC timestamp>.jsonl)')
    parser.add_argument('--no-record', action='store_true',
                        help='never record, whatever else is set')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m tetrisrl',
        description='Tetris reinforcement learning (SRS engine + Double DQN)')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', help='train the agent')
    p.add_argument('--rounds', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--episodes-per-worker', type=int)
    p.add_argument('--resume', nargs='?', const=True, default=None,
                   help='resume from checkpoints/latest.pt (or a given path)')
    p.add_argument('--tag', default=None,
                   help='name for the output checkpoints '
                        '(best_<tag>.pt and <tag>.pt)')
    p.add_argument('--publish', action='store_true',
                   help='write the canonical best.pt/latest.pt '
                        '(overwrites them); default is a tagged run')
    p.add_argument('--reward-preset', default=None,
                   help='reward scheme: v1_original | michielcox | '
                        'strong_death | tetris_heavy | michielcox_small')
    p.add_argument('--no-lr-decay', action='store_true',
                   help='disable learning-rate annealing (constant lr)')
    p.add_argument('--lr-decay-fraction', type=float, default=None,
                   help='0 disables annealing, 1 anneals fully (default)')
    p.add_argument('--epsilon-decay-steps', type=int, default=None,
                   help='transitions over which epsilon anneals to its minimum; '
                        'size this to the run so exploration actually decays')
    p.add_argument('--config', default=None)
    p.add_argument('--quiet', action='store_true')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('eval', help='measure a policy over full games')
    p.add_argument('--policy', default='best')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--max-pieces', type=int, default=20000,
                   help='stop a game after this many pieces (the heuristic can '
                        'play indefinitely, so cap it to keep evaluation short)')
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser('watch', help='watch a policy play')
    p.add_argument('--policy', default='best')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--config', default=None)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser('play', help='play it yourself')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--config', default=None)
    _add_record_flags(p)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser('battle', help='play a versus match against the bot')
    p.add_argument('--difficulty', default=None,
                   choices=sorted(HANDICAP_PRESETS),
                   help='bot strength: PPS cap, search depth/beam and mistake rate '
                        '(default: the module default)')
    p.add_argument('--messiness', default='default', choices=sorted(MESSINESS),
                   help='how the garbage hole column is chosen')
    p.add_argument('--rounds', type=int, default=3,
                   help='rounds needed to win the match (default 3)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--config', default=None)
    _add_record_flags(p)
    p.set_defaults(func=cmd_battle)

    p = sub.add_parser('menu', help='interactive launcher')
    p.add_argument('--config', default=None)
    p.set_defaults(func=cmd_menu)

    p = sub.add_parser('policies', help='list available policies')
    p.set_defaults(func=cmd_policies)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
