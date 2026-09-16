"""TETR.IO-style versus rules: attack, cancelling, garbage and match state.

The engine is a solo engine: it has no notion of an opponent, garbage, combos
worth sending or a match. This module adds exactly that layer and nothing else, so
the solo measurements in the README keep meaning what they meant.

Numbers are TETR.IO's, taken from the published ruleset (see
``docs/tetrio-versus-ruleset.md`` for the full sourcing):

* **Attack table** — single 0, double 1, triple 2, quad 4; anything above four
  lines is ``5 + (n - 5)``; an all clear sends 5.
* **Combo** — ``base * (1 + 0.25 * combo)``, rounded down, in the default
  "Multiplier" combo table. A clear with a base of zero still sends
  ``ln(1 + 1.25 * combo)`` from a 2-combo up, and the two are compared with the
  larger winning.
* **B2B** — a continuing difficult clear adds 1 ("charging"), and from a streak of
  4 a **Surge** is charged: it pays ``4 + (streak - 4)`` lines, i.e. the streak
  itself, split into three chunks. (QUICK PLAY starts its Surge at 1 instead of 4;
  the rooms and leagues this mirrors use 4.)
* **Cancelling** — outgoing lines strike the oldest incoming garbage first, one
  line at a time, with no offset limit; only the remainder is sent on.
* **Entry** — garbage lands **only on a lock that cleared nothing**. Any clear
  blocks it, which is TETR.IO's "combo blocking".
* **Queue** — 20 frames of travel (0.333 s) before an item can enter, and at most
  8 lines enter per lock.
* **Messiness** — how the hole column is chosen: never re-rolled ("clean"), or
  re-rolled per attack and per line ("messy").

Spins are detected by the engine (last move a rotation, and three of the four
corners of the piece's box filled -> see ``engine.Game._classify_spin``), so the
spin rows of the attack table are live: a T-spin double sends four lines, not one.
Only the T is classified, which is what the TETR.IO preset does. Sources for every
number, and the attack table in full, are in ``docs/tetrio-versus-ruleset.md``.
"""
from __future__ import annotations

import math
import random
from collections import namedtuple

from .engine import Game

# --- attack -----------------------------------------------------------------

#: Garbage sent per clear, before combo and B2B.
ATTACK_TABLE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
#: An all clear is worth its own value rather than the clear's.
ALL_CLEAR_GARBAGE = 5
#: Spins are paid on their own scale, and a spin with no clear sends nothing.
#: TETR.IO's league preset (docs/tetrio-versus-ruleset.md, from Triangle.js
#: ``src/engine/utils/damageCalc``): spin single 2, double 4, triple 6, quad 10,
#: penta 12; mini single 0, mini double 1, mini triple 2. This is the row that a
#: T-spin double being worth four lines comes from -- without spin detection the
#: same lock is an ordinary double and pays one.
SPIN_ATTACK = {0: 0, 1: 2, 2: 4, 3: 6, 4: 10}
SPIN_ATTACK_OVER = 12
MINI_ATTACK = {0: 0, 1: 0, 2: 1, 3: 2}
COMBO_MULTIPLIER = 0.25
COMBO_LOG = 1.25
B2B_SURGE_AT = 4
B2B_SURGE_BASE = 4

#: Seconds an incoming attack spends in the queue before it can enter.
GARBAGE_TRAVEL = 20 / 60.0
#: Most lines that can enter a board on one lock.
GARBAGE_CAP = 8
FPS = 60.0

Attack = namedtuple('Attack', 'lines chunks b2b')


def base_attack(cleared, all_clear=False, spin=None):
    """Garbage for a clear, before combo and B2B.

    ``spin`` is ``'full'``, ``'mini'`` or None, straight from the engine's
    detection. A spin replaces the ordinary clear value rather than adding to it:
    a T-spin double is worth four lines *instead of* one.
    """
    if spin == 'full':
        if cleared <= 0:
            return 0
        return SPIN_ATTACK.get(cleared, SPIN_ATTACK_OVER)
    if spin == 'mini':
        if cleared <= 0:
            return 0
        # A mini quad does not really exist; if one is ever reported, pay it as a
        # full spin rather than as nothing.
        return MINI_ATTACK.get(cleared, SPIN_ATTACK.get(cleared, 0))
    if cleared <= 0:
        return 0
    if all_clear:
        return ALL_CLEAR_GARBAGE
    if cleared > 4:
        return 5 + (cleared - 5)
    return ATTACK_TABLE.get(cleared, 0)


def attack_for(cleared, combo=-1, b2b=0, all_clear=False, spin=None):
    """``(lines, chunks, b2b)`` for one lock, by TETR.IO's rules.

    ``combo`` is the chain length *after* this clear (0 for the first clear of a
    chain), which is what ``Game.combo`` holds once the lock has been applied.
    ``b2b`` is the streak before it. ``chunks`` is how the lines are delivered:
    normally one chunk, but a Surge arrives as three.

    A spin counts as a *difficult* clear for the B2B chain, exactly like a quad:
    it continues the chain and is paid the charging bonus.
    """
    if cleared <= 0:
        return Attack(0, (), b2b + 1 if spin else 0)
    base = base_attack(cleared, all_clear, spin)
    if base:
        lines = int(base * (1 + COMBO_MULTIPLIER * combo))
    else:
        # A single has a base of zero, so the multiplier can never pay. TETR.IO
        # keeps long single chains alive with a log curve instead, and takes
        # whichever of the two is larger.
        lines = int(math.log(1 + COMBO_LOG * combo)) if combo >= 2 else 0

    difficult = cleared == 4 or all_clear or spin is not None
    b2b_after = b2b + 1 if difficult else 0
    if difficult and b2b:
        lines += 1                      # B2B "charging": +1 while the chain lives

    chunks = [lines] if lines > 0 else []
    if b2b_after >= B2B_SURGE_AT:
        # Surge: the streak itself, from 4 lines at a streak of 4 upwards.
        surge = B2B_SURGE_BASE + (b2b_after - B2B_SURGE_AT)
        third = round(surge / 3)
        chunks.extend([third, third, surge - 2 * third])
    return Attack(sum(chunks), tuple(chunks), b2b_after)


# --- garbage queue ----------------------------------------------------------

#: Hole-column behaviour. ``change`` re-rolls between attacks, ``within`` between
#: the lines of one attack, ``nosame`` forbids repeating a column, ``center``
#: excludes the middle fifth of the board.
MESSINESS = {
    'clean': {'change': 0.0, 'within': 0.0, 'nosame': True, 'center': False},
    'default': {'change': 0.35, 'within': 0.15, 'nosame': True, 'center': False},
    'messy': {'change': 0.8, 'within': 0.55, 'nosame': True, 'center': False},
    'chaotic': {'change': 1.0, 'within': 1.0, 'nosame': False, 'center': False},
}


def new_hole(cols, rng, avoid=None, center=False):
    """A hole column, optionally excluding ``avoid`` and the centre fifth."""
    low, high = 0, cols - 1
    if center:
        margin = max(1, round(cols / 5))
        low, high = margin, cols - 1 - margin
    choices = [c for c in range(low, high + 1) if c != avoid]
    if not choices:
        choices = list(range(cols))
    return rng.choice(choices)


class GarbageQueue:
    """Incoming garbage for one board: what was sent, and when it can land."""

    def __init__(self, cap=GARBAGE_CAP, travel=GARBAGE_TRAVEL):
        self.items = []                 # [lines, ready_time], oldest first
        self.cap = cap
        self.travel = travel

    @property
    def pending(self):
        """Total lines queued, ready or not -- what an opponent has sent."""
        return sum(n for n, _t in self.items)

    def ready(self, now):
        """Lines that have finished travelling and may enter on the next lock."""
        return sum(n for n, t in self.items if t <= now)

    def send(self, chunks, now):
        for lines in chunks:
            if lines:
                self.items.append([lines, now + self.travel])

    def cancel(self, lines):
        """Spend ``lines`` against the oldest garbage first.

        Returns what is left to send on. TETR.IO has no offset limit: one line of
        attack removes one line of incoming garbage, wherever it is in the queue.
        """
        while lines > 0 and self.items:
            item = self.items[0]
            take = item[0] if item[0] < lines else lines
            item[0] -= take
            lines -= take
            if item[0] <= 0:
                self.items.pop(0)
        return lines

    def take(self, now, rng, messiness, cols, previous=None):
        """Remove up to ``cap`` ready lines and return their hole columns.

        Hole columns are chosen here rather than when the attack was sent, because
        that is when TETR.IO decides them: messiness is a property of the garbage
        as it lands, not of the attack that sent it.
        """
        mess = MESSINESS[messiness] if isinstance(messiness, str) else messiness
        budget = self.ready(now)
        if budget > self.cap:
            budget = self.cap
        holes = []
        col = previous
        while budget > 0 and self.items and self.items[0][1] <= now:
            item = self.items[0]
            take = item[0] if item[0] < budget else budget
            for i in range(take):
                chance = mess['within'] if i else mess['change']
                if col is None or rng.random() < chance:
                    col = new_hole(cols, rng,
                                   avoid=col if mess['nosame'] else None,
                                   center=mess['center'])
                holes.append(col)
            item[0] -= take
            budget -= take
            if item[0] <= 0:
                self.items.pop(0)
        return holes


# --- match ------------------------------------------------------------------

class Side:
    """One player's board, its incoming queue, and its attack bookkeeping."""

    __slots__ = ('game', 'queue', 'b2b', 'wins', 'name', 'last_hole',
                 'sent', 'received', 'tanked')

    def __init__(self, game, name='player', cap=GARBAGE_CAP,
                 travel=GARBAGE_TRAVEL):
        self.game = game
        self.queue = GarbageQueue(cap, travel)
        self.b2b = 0
        self.wins = 0
        self.name = name
        self.last_hole = None
        self.sent = 0
        self.received = 0
        self.tanked = 0


class Battle:
    """A first-to-N versus match between two boards.

    Both sides get the *same* piece sequence by default, which is how MisaMino
    plays and the only way a match against a bot is a fair comparison of
    decisions rather than of bags.

    The caller owns the clocks: feed it ``update(dt)`` and tell it about every
    lock with ``after_lock(side, cleared)``. It never reaches into a policy.
    """

    def __init__(self, seed=None, rounds_to_win=3, messiness='default',
                 cap=GARBAGE_CAP, travel=GARBAGE_TRAVEL, same_pieces=True,
                 names=('you', 'bot')):
        self.seed = seed
        self.rounds_to_win = rounds_to_win
        self.messiness = messiness
        self.cap = cap
        self.travel = travel
        self.same_pieces = same_pieces
        self.names = names
        self.rng = random.Random(seed)
        self.time = 0.0
        self.round = 1
        self.log = []
        self._new_boards()

    # -- setup ------------------------------------------------------------

    def _side_seed(self, index):
        if self.same_pieces:
            return self.seed if self.seed is not None else 0
        return self.rng.randrange(2 ** 31)

    def _new_boards(self):
        self.sides = []
        for i, name in enumerate(self.names[:2]):
            game = Game(seed=self._side_seed(i))
            game.nolockout = True          # TETR.IO's nolockout + clutch
            self.sides.append(Side(game, name=name, cap=self.cap,
                                   travel=self.travel))

    def new_round(self):
        """Fresh boards, keeping the score."""
        wins = [s.wins for s in self.sides]
        names = [s.name for s in self.sides]
        self.round += 1
        self.time = 0.0
        self._new_boards()
        for side, w, name in zip(self.sides, wins, names):
            side.wins = w
            side.name = name
        return self.sides

    # -- queries ----------------------------------------------------------

    def other(self, index):
        return self.sides[1 - index]

    def incoming(self, index):
        """Lines queued against a side -- what a survival check needs to know."""
        return self.sides[index].queue.pending

    def ready(self, index):
        return self.sides[index].queue.ready(self.time)

    def winner(self):
        """Index of the side that has taken the match, or None."""
        for i, side in enumerate(self.sides):
            if side.wins >= self.rounds_to_win:
                return i
        return None

    def update(self, dt):
        self.time += dt

    # -- the rules --------------------------------------------------------

    def after_lock(self, index, cleared, all_clear=None, spin=None):
        """Apply one lock's consequences. Returns a summary dict.

        Order matters and follows TETR.IO: the attack is computed, then spent
        cancelling this board's own incoming garbage oldest-first, and only the
        remainder is sent on. Garbage enters this board only if the lock cleared
        nothing.

        ``spin`` defaults to the engine's own classification of that lock, so a
        T-spin is paid as a T-spin without the caller having to notice it.
        """
        side = self.sides[index]
        foe = self.other(index)
        game = side.game
        if all_clear is None:
            all_clear = bool(cleared) and all(
                c is None for row in game.board.grid for c in row)
        if spin is None:
            # The engine classified the lock the side just made. Taking it from
            # there rather than from the caller means a T-spin is paid correctly
            # however the lock was driven -- human, bot, or a policy in a test.
            spin = getattr(game, 'spin_kind', None)

        attack = attack_for(cleared, max(0, game.combo), side.b2b, all_clear, spin)
        side.b2b = attack.b2b

        cancelled = 0
        sent = 0
        if attack.lines:
            before = side.queue.pending
            remainder = side.queue.cancel(attack.lines)
            cancelled = before - side.queue.pending
            if remainder:
                sent = remainder
                foe.queue.send([remainder], self.time)
                foe.received += remainder
                side.sent += remainder
            self.log.append((self.time, side.name, cleared, attack.lines,
                             cancelled, sent, spin))

        tanked = []
        if cleared == 0:
            holes = side.queue.take(self.time, self.rng, self.messiness,
                                    game.cols, side.last_hole)
            if holes:
                side.last_hole = holes[-1]
                side.tanked += len(holes)
                # One row per line, all inserted together at the bottom.
                game.add_garbage(len(holes), holes)
                tanked = holes

        return {'attack': attack, 'cancelled': cancelled, 'sent': sent,
                'tanked': tanked, 'ko': game.game_over, 'spin': spin,
                'b2b': attack.b2b}


def play_out(battle, policies, max_pieces=6000, on_lock=None, piece_seconds=0.25):
    """Drive a whole match headlessly -- the harness the tests and tools use.

    ``policies`` is one callable per side, each ``policy(game, actions)`` exactly
    like the rest of the project. Returns the winning index.

    ``piece_seconds`` advances the match clock once per exchange. It matters:
    garbage only becomes enterable after its travel delay, so a harness that never
    advances the clock would queue attacks that can never land -- which is exactly
    what an earlier version of this function did.
    """
    from .heuristic import apply_move
    from .engine import valid_actions

    pieces = 0
    while battle.winner() is None and pieces < max_pieces:
        for index, policy in enumerate(policies):
            side = battle.sides[index]
            game = side.game
            if game.game_over:
                continue
            actions = valid_actions(game.board, game.current)
            if not actions:
                game.game_over = True
                continue
            move = policy(game, actions)
            cleared = apply_move(game, move)
            summary = battle.after_lock(index, cleared)
            if on_lock is not None:
                on_lock(index, summary)
            if game.game_over:
                foe = battle.other(index)
                foe.wins += 1
                if battle.winner() is None:
                    battle.new_round()
                break
        battle.update(piece_seconds)
        pieces += 1
    return battle.winner()
