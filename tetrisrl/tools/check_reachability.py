"""Test whether every action the agent can choose is reachable by normal play.

The agent abstracts a move as a final placement ``(rotation, x)`` and realises it
by setting the piece's rotation and column directly, then hard-dropping. That
bypasses the rotate/move/kick machinery entirely, which raises a fair question:
does the abstraction let the agent choose placements a real player could not
reach? And conversely, does requiring the piece to fit at the spawn row exclude
placements a real player *could* reach by tucking under an overhang?

This script answers both by breadth-first search over the real move primitives
(rotate with SRS kicks, move, soft drop) for many random boards.

Usage:  python tools/check_reachability.py [--boards 300]
"""
import argparse
import os
import random
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def placements_by_play(game, max_states=20000):
    """All (rotation, x, landed_cells) reachable by real moves from spawn.

    State is (rotation, x, y). Successors come from the engine's own primitives,
    so SRS wall kicks are included exactly as the engine implements them.

    A state is a *placement* when the piece cannot fall any further; such a
    state is still expanded, because a resting piece may slide sideways and
    settle lower elsewhere.
    """
    start = (game.current.rotation, game.current.x, game.current.y)
    seen = {start}
    queue = deque([start])
    found = {}

    def falls(rot, x, y):
        return not game.board.collides(
            [(x + ox, y + 1 + oy) for ox, oy in _base(game.current.kind, rot)])

    while queue and len(seen) < max_states:
        rot, x, y = queue.popleft()
        if not falls(rot, x, y):
            found.setdefault((rot, x), [(x + ox, y + oy)
                                        for ox, oy in _base(game.current.kind, rot)])
        # Fall
        if falls(rot, x, y):
            nxt = (rot, x, y + 1)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
        # Slide
        for dx in (-1, 1):
            cand = (rot, x + dx, y)
            if cand in seen:
                continue
            if not game.board.collides([(x + dx + ox, y + oy)
                                        for ox, oy in _base(game.current.kind, rot)]):
                seen.add(cand)
                queue.append(cand)
        # Rotate, using the engine's resolved kick position
        for direction in (1, -1):
            piece = game.current
            saved = (piece.rotation, piece.x, piece.y)
            piece.rotation, piece.x, piece.y = rot, x, y
            target = game.can_rotate(direction)
            piece.rotation, piece.x, piece.y = saved
            if target is None:
                continue
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return found


def _base(kind, rot):
    from tetrisrl.engine import _CELLS
    return _CELLS[kind][rot % 4]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boards', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    from tetrisrl.engine import Game, spawn_anchor, valid_actions

    rng = random.Random(args.seed)
    tried = 0
    unreachable = 0
    extra_reachable = 0
    examples = []

    for trial in range(args.boards):
        # Build a random board by playing random legal placements.
        game = Game(seed=trial)
        for _ in range(rng.randrange(0, 60)):
            if game.game_over:
                break
            acts = valid_actions(game.board, game.current)
            if not acts:
                break
            rot, x = rng.choice(acts)
            game.current.rotation = rot
            game.current.x = x
            game.current.y = spawn_anchor(game.current.kind)[1]
            game.hard_drop()
        if game.game_over or game.current is None:
            continue

        abstract = set(valid_actions(game.board, game.current))
        if not abstract:
            continue
        playable = set(placements_by_play(game))
        tried += 1

        missing = abstract - playable
        if missing:
            unreachable += 1
            if len(examples) < 5:
                examples.append((game.current.kind, sorted(missing)[:4],
                                 len(abstract), len(playable)))

    print(f'boards examined                      : {tried}')
    print(f'boards where an abstract action was')
    print(f'NOT reachable by real SRS play       : {unreachable}'
          f'  ({100 * unreachable / max(1, tried):.1f}%)')
    if examples:
        print('  examples (piece, unreachable actions, #abstract, #playable):')
        for e in examples:
            print(f'    {e}')

    # The other direction: placements real play can reach that the abstraction
    # forbids. These are the "tuck under an overhang" moves.
    tucked_total = 0
    boards_with_tucks = 0
    for trial in range(args.boards):
        game = Game(seed=10000 + trial)
        for _ in range(rng.randrange(0, 60)):
            if game.game_over:
                break
            acts = valid_actions(game.board, game.current)
            if not acts:
                break
            rot, x = rng.choice(acts)
            game.current.rotation = rot
            game.current.x = x
            game.current.y = spawn_anchor(game.current.kind)[1]
            game.hard_drop()
        if game.game_over or game.current is None:
            continue
        abstract = set(valid_actions(game.board, game.current))
        playable = set(placements_by_play(game))
        extra = playable - abstract
        if extra:
            boards_with_tucks += 1
            tucked_total += len(extra)

    print()
    print(f'boards where real play can reach a placement the abstraction FORBIDS: '
          f'{boards_with_tucks}')
    print(f'total such placements: {tucked_total}')
    print('  (these are overhang "tucks": the engine requires a placement to fit')
    print('   at the spawn row, so a piece cannot be slid under an overhang)')
    return 0


if __name__ == '__main__':
    main()
