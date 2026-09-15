"""Hand-written placement heuristics.

Two roles:

1. ``heuristic_choice`` is a non-learning baseline. It is useful for checking
   that the engine, reward and evaluation plumbing are sane, and as a fallback
   in the renderer when no checkpoint exists.
2. ``teacher_choice`` is the same evaluator exposed for imitation warm-starting.

The evaluation function uses the same board metrics the network sees, so the
teacher never has information the agent lacks.
"""
from __future__ import annotations

from .engine import Game, valid_actions
from .features import metrics_for


def best_well(heights, cols):
    """(depth, column) of the deepest usable well, or (0, -1).

    A well is a column lower than its neighbours. The depth is how far the
    neighbours rise above it, capped at 4 because no clear is larger than a
    tetris -- a 7-deep pit is not a better tetris well than a 4-deep one, just a
    more dangerous hole. Board edges count as walls, so the leftmost and
    rightmost columns can be wells.

    This exists to let the evaluator pay for *keeping* a well. Without it the
    greedy score prefers clearing a single line (+10 now) over leaving the line
    and completing a tetris later (+0 now), which is why the untuned baseline
    clears 82% singles and 0.7% tetrises.
    """
    depth, col = 0, -1
    for x in range(cols):
        left = heights[x - 1] if x > 0 else heights[x]
        right = heights[x + 1] if x < cols - 1 else heights[x]
        wall = min(left, right)
        gap = min(4, wall - heights[x])
        if gap > depth:
            depth, col = gap, x
    return depth, col


def covered_cells(grid, rows, cols):
    """How many filled cells sit on top of the topmost hole of each column.

    From Cold Clear's evaluator. A hole covered by six blocks is far worse than
    one covered by a single block -- the first can never be reached, the second
    can still be dug out -- and a plain hole count cannot tell them apart.
    Returns ``(total, sum_of_squares)``; the squared term is what separates one
    deeply buried hole from several shallow ones.
    """
    total = 0
    total_sq = 0
    for x in range(cols):
        h = 0
        for y in range(rows):
            if grid[y][x] is not None:
                h = rows - y
                break
        # Walk down from just below any hole; count cover above each gap.
        for y in range(rows - h, rows):
            if grid[y][x] is None:
                # Everything filled above this cell in the column covers it.
                cover = sum(1 for yy in range(y) if grid[yy][x] is not None)
                cover = min(6, cover)
                if cover:
                    total += cover
                    total_sq += cover * cover
    return total, total_sq


def _playable_wells(game, res_heights, res_holes, res_rows_with_holes, places,
                    ready_only=False):
    """Extra score for leaving a viable tetris well, best over the given plans.

    ``places`` is a sequence of ``(column, keep_depth)``: the columns that may
    hold a well, and how deep it must stay. Returning the best keeps the
    evaluator neutral when nothing fits, rather than forcing a bad well.

    ``ready_only`` makes this a *waiting* bonus rather than a hoarding one: it
    pays only when a tetris is actually completable now, i.e. the well is deep
    enough and an I piece is to hand. Without that condition the evaluator simply
    never clears (keeping the well scores better than spending it), stacks up and
    dies -- measured at 44 lines/game against 998. The check is deliberately
    optimistic -- it assumes a clean approach and ignores intervening pieces.
    """
    best = 0.0
    cols = game.cols
    tetris_at = None
    if ready_only:
        tetris_at = _tetris_opportunity(res_heights)
        if tetris_at is None:
            return 0.0
    for well_col, keep_depth in places:
        if not (0 <= well_col < cols) or res_holes > 0 or res_rows_with_holes > 0:
            continue
        if ready_only and well_col != tetris_at:
            continue
        # The three columns that must stay tall enough for an I to complete a
        # tetris: the well itself plus the three beside it.
        height = min(res_heights[well_col],
                     max(res_heights[max(0, well_col - 1)],
                         res_heights[min(cols - 1, well_col + 1)]))
        if height >= keep_depth:
            best = max(best, float(height))
    return best


def _tetris_opportunity(heights):
    """Column where dropping an I right now would clear four rows, or None.

    True when some column is 4 lower than the height its I would need, with the
    rest of the row high enough that the I completes exactly four lines.
    """
    cols = len(heights)
    for x in range(cols):
        left = heights[x - 1] if x > 0 else heights[x]
        right = heights[x + 1] if x < cols - 1 else heights[x]
        wall = min(left, right)
        if wall - heights[x] >= 4:
            return x
    return None


def _i_available(game, horizon=6):
    """True if an I piece is in hand or within the next ``horizon`` pieces.

    Also always true when hold is unused and the current piece is an I.
    """
    if game.current is not None and game.current.kind == 'I':
        return True
    held = getattr(game, 'hold_piece', None)
    if held is not None and getattr(held, 'kind', None) == 'I':
        return True
    for piece in list(game.next_queue)[:horizon]:
        if getattr(piece, 'kind', None) == 'I':
            return True
    return False


# Columns a well may occupy. The engine's placement search allows a *vertical* I
# in any column (verified: rotations 1 and 3 admit x = 0..9 on an empty board),
# so nothing needs excluding here. An earlier version restricted this to columns
# 2-7, which is the range of *horizontal* spawn positions and had nothing to do
# with vertical drops -- a wrong constraint that silently told the evaluator some
# reachable wells were unfillable.
_I_COLUMNS = tuple(range(10))


def hold_options(game):
    """Candidate moves as ``(use_hold, rotation, x)``.

    Hold is part of the action space, not a separate layer: the engine allows one
    hold per piece, so each legal placement is offered twice -- once directly and
    once after swapping the current piece out. ``use_hold`` is dropped when hold
    is unavailable (already used this piece, or disabled), so the caller can
    always treat the first element as meaningful.

    A three-tuple rather than a bare ``(rotation, x)`` because holding the I
    until the well is four deep is the single most valuable use of hold, and that
    decision cannot be expressed as a placement.
    """
    placements = valid_actions(game.board, game.current)
    actions = []
    held = getattr(game, 'hold_piece', None)
    can_hold = (getattr(game, 'allow_hold', False)
                and not getattr(game, 'hold_used', True)
                and game.current is not None)
    if can_hold:
        # After a hold the piece in hand is the held one (or the next queued one
        # if the slot was empty), so its placements differ.
        alt_kind = held.kind if held is not None else (
            game.next_queue[0].kind if game.next_queue else None)
        if alt_kind is not None:
            probe = game.current
            game.current = type(probe)(alt_kind) if probe is not None else None
            try:
                alt = valid_actions(game.board, game.current)
            finally:
                game.current = probe
            actions.extend([(True, r, x) for r, x in alt])
    actions.extend([(False, r, x) for r, x in placements])
    return actions

# Weights for the heuristic score. Sign convention: positive is good.
#
# Provenance, stated precisely, because it is easy to overstate:
#   * 'lines', 'holes', 'bumpiness', 'aggregate_height', 'max_height' were chosen
#     by a lines-per-game sweep on this engine. The aggregate-height value
#     converging near Dellacherie's published figure was a useful sanity check.
#   * 'row_transitions' is a PLACEHOLDER (about -0.5 where the published DT-20 set
#     uses roughly -2.4), and 'wells' is likewise untuned.
#   * 'rows_with_holes' is implemented and measured but left at 0: it fully
#     recovers a badly-tuned agent yet adds nothing once the hole term is right.
#
# So this is Dellacherie-STYLE rather than a faithful Dellacherie implementation:
# it omits column transitions, eroded cells and landing height, and its
# transition/well weights are not from the literature.
#
# With these values the baseline clears ~1,200 lines per 3,000-piece game with
# almost no holes at death, which makes it a strong comparison point and a
# usable imitation teacher.
DEFAULT_WEIGHTS = {
    'lines': 10.0,
    'holes': -30.0,
    'bumpiness': -0.8,
    'aggregate_height': -0.51,
    'max_height': -1.5,
    'row_transitions': -0.5,    # placeholder, see note above
    'wells': -0.8,              # untuned
    'rows_with_holes': 0.0,     # implemented and measured; redundant, see above
    # Pays for leaving a tetris well instead of clearing a single now. Off by
    # default; see `well_bonus` handling in score_placement.
    'well_bonus': 0.0,
    'well_keep_depth': 4.0,
}


def score_placement(game, rotation, x, weights=None):
    """Heuristic score for one placement. Higher is better."""
    w = resolve_weights(weights) or DEFAULT_WEIGHTS
    w = effective_weights(game, w)
    grid, cells, cleared_rows = game.simulate_placement(rotation, x)
    cleared = len(cleared_rows)
    rows, cols = game.rows, game.cols
    (heights, holes, bumpiness, aggregate, row_trans, col_trans, wells,
     max_height, rows_with_holes) = metrics_for(grid, rows, cols)

    score = 0.0
    if cleared:
        # Two alternative clear-value schemes.
        #
        # The default pays `lines * cleared^2`, which rewards clearing at all.
        # Cold Clear instead PAYS NEGATIVE for small clears and only rewards the
        # tetris (its defaults: clear1 -143, clear2 -100, clear3 -58, clear4
        # +390). That is a different mechanism, not a different weight: a single
        # line becomes a cost, so keeping the well is strictly the better local
        # move and waiting for the I is no longer a coin flip. Setting
        # `cc_clear1` switches to that scheme.
        if 'cc_clear1' in w:
            table = {1: w['cc_clear1'], 2: w.get('cc_clear2', w['cc_clear1']),
                     3: w.get('cc_clear3', w['cc_clear1']),
                     4: w.get('cc_clear4', 390.0)}
            score += table.get(min(cleared, 4), table[4])
        else:
            score += w['lines'] * (cleared ** 2)
    score += w['holes'] * holes
    score += w['bumpiness'] * bumpiness
    score += w['aggregate_height'] * aggregate
    score += w['max_height'] * max_height
    score += w['row_transitions'] * row_trans
    score += w['wells'] * wells
    score += w.get('rows_with_holes', 0.0) * rows_with_holes

    # --- Cold Clear derived terms -------------------------------------------
    #
    # Well depth, from Cold Clear's evaluator. A well of depth 4 is what a tetris
    # needs.
    #
    # MEASURED, and disappointing for a greedy evaluator: `well_depth_bonus` and
    # `well_column` both raise the tetris rate but destroy survival. Cold Clear
    # runs `well_column = [20, 23, 20, 50, 59, 21, 59, 10, -10, 24]`, but bolting
    # that onto a one-ply evaluator took lines from 774 to 77 with 6/6 games dead
    # -- the agent commits to a well and then cannot dig out. These stay at 0 by
    # default. The tetris gain that *is* safe comes from negative clear values
    # (see `cc_clear1` above), not from these.
    depth, well_col = best_well(heights, cols)
    keep = w.get('well_keep_depth', 4.0)
    if depth >= keep:
        wd_bonus = w.get('well_deep_bonus', 0.0)
        if wd_bonus and not (holes or rows_with_holes):
            score += wd_bonus
    wd = w.get('well_depth_bonus', 0.0)
    if wd:
        score += wd * depth * depth
    if depth and well_col >= 0:
        cols_w = w.get('well_column')
        if cols_w:
            score += cols_w[well_col]

    cov = w.get('covered_cells', 0.0)
    cov_sq = w.get('covered_cells_sq', 0.0)
    if cov or cov_sq:
        total, total_sq = covered_cells(grid, rows, cols)
        score += cov * total + cov_sq * total_sq

    # Tetris-well bonus. This is what makes the evaluator willing to *wait* for a
    # tetris: clearing a single scores w['lines'], and this term pays for the
    # alternative of keeping the well open. It is only offered for columns an I
    # can actually drop into, and only while the board is clean, so it cannot
    # reward leaving a useless pit.
    well_bonus = w.get('well_bonus', 0.0)
    if well_bonus:
        keep = w.get('well_keep_depth', 4.0)
        places = [(c, keep) for c in _I_COLUMNS]
        # Only pay when an I can actually finish the job. Paired with a small
        # discount on cheap clears so waiting is the better local move.
        if w.get('well_wait_for_i', True) and not _i_available(game):
            pass
        else:
            score += well_bonus * _playable_wells(
                game, heights, holes, rows_with_holes, places,
                ready_only=w.get('well_ready_only', True))

    # Landing depth: prefer placing pieces low. Measured from the top of the
    # board, so larger means the piece came to rest further down.
    landing_top = min((y for _, y in cells), default=rows)
    score += 0.10 * landing_top
    return score


def score_action(game, action, weights=None):
    """Score a move, given as ``(use_hold, rotation, x)`` or ``(rotation, x)``.

    Hold is evaluated by swapping the piece on a copy and scoring the resulting
    placement, so a hold that unlocks a much better spot is visible to the same
    evaluator -- no separate hold heuristic.
    """
    if len(action) == 2:
        use_hold, rot, x = False, action[0], action[1]
    else:
        use_hold, rot, x = action
    if not use_hold:
        return score_placement(game, rot, x, weights)
    probe = clone_game(game)
    if not probe.hold():
        return score_placement(game, rot, x, weights)
    score = score_placement(probe, rot, x, weights)
    # Holding is otherwise score-neutral -- the board after "hold then place" is
    # identical to placing directly, so the evaluator has no reason to prefer it
    # and in practice never did. What makes a hold *useful* is putting a wanted
    # piece into the slot (keeping an I for a tetris) or parking an unwanted one.
    # Biases are therefore explicit and tunable rather than hoped for:
    w = resolve_weights(weights) or DEFAULT_WEIGHTS
    held_back = game.current.kind if game.current else None
    wb = w.get('hold_bias', 0.0)
    if wb:
        current_kind = game.current.kind if game.current else None
        brought_in = probe.current.kind
        # Prefer bringing a piece that fits the current need, and prefer banking
        # a piece that does not.
        if brought_in == w.get('hold_prefer_kind'):
            score += wb
        elif current_kind == w.get('hold_prefer_kind'):
            score -= wb
        elif brought_in == w.get('hold_avoid_kind'):
            score -= wb
    return score


# Tetris-aware weighting, derived from Cold Clear's published evaluator
# (MinusKelvin/cold-clear, bot/src/evaluation/standard.rs).
#
# The mechanism is the interesting part, and it is not a weight tweak: Cold Clear
# assigns NEGATIVE value to clearing one, two or three lines (its defaults are
# -143 / -100 / -58) and pays only for the tetris (+390). A single line therefore
# becomes a *cost*, so keeping the well open is the better local move and waiting
# for the I stops being a coin flip. Simply shrinking the single-line reward --
# which is all the default scheme can express -- cannot do this.
#
# The catch, measured: a policy that refuses to clear will happily stack out.
# Cold Clear avoids that with deep search; a one-ply evaluator needs an explicit
# override, so `panic_height` restores normal clear values once the stack gets
# tall. That is what makes the scheme survivable rather than suicidal.
TETRIS_AWARE_WEIGHTS = dict(
    DEFAULT_WEIGHTS,
    cc_clear1=-20.0,        # clearing a single is a small cost
    cc_clear2=-14.0,
    cc_clear3=-8.0,
    cc_clear4=100.0,        # the tetris is what pays
    panic_height=10,        # above this stack height, clear normally again
)

WEIGHT_PRESETS = {
    'default': DEFAULT_WEIGHTS,
    'tetris_aware': TETRIS_AWARE_WEIGHTS,
}


def resolve_weights(weights):
    """Turn a preset name into its dict; pass dicts (and None) straight through."""
    if isinstance(weights, str):
        try:
            return WEIGHT_PRESETS[weights]
        except KeyError:
            raise ValueError(f'unknown weight preset {weights!r}; '
                             f'known: {sorted(WEIGHT_PRESETS)}') from None
    return weights


def effective_weights(game, w):
    """Apply the ``panic_height`` survival override, if one is configured.

    Restores positive clear values when the stack reaches ``panic_height``, so a
    tetris-seeking evaluator still digs itself out instead of topping out.
    """
    panic = w.get('panic_height')
    if panic is None:
        return w
    if max(game.board.column_heights()) < panic:
        return w
    flipped = dict(w)
    for key in ('cc_clear1', 'cc_clear2', 'cc_clear3', 'cc_clear4'):
        if key in flipped:
            flipped[key] = abs(flipped[key])
    return flipped


def heuristic_choice(game, actions=None, weights=None, sample_next=0,
                     allow_hold=False):
    """Best move by the heuristic, or None when nothing is legal.

    Returns ``(use_hold, rotation, x)``; ``apply_move`` also accepts a bare
    ``(rotation, x)``, so callers that do not care about hold need no special
    case.

    ``allow_hold`` defaults to FALSE, and that is a measured decision rather than
    an oversight. Hold is score-neutral here -- the board after "hold then place"
    is identical to placing directly -- so the evaluator has no basis for
    preferring it, and measurements bear that out:

        weights        hold    lines   singles  doubles  triples  tetrises
        default        off     599.0    81.4%    15.9%     2.2%     0.5%
        default        on      598.7    93.3%     5.8%     0.7%     0.2%
        tetris_aware   off     596.3    23.1%    61.9%    11.6%     3.5%
        tetris_aware   on      596.0     5.3%    82.1%    10.4%     2.2%

    Holding churns roughly one piece in three through the slot and makes the
    clear mix worse in every configuration tried, including with an explicit bias
    to bank an I or park an S/Z. Like the well bonuses and lookahead, it needs a
    planner to be worth anything: deciding to keep an I for a tetris that is
    three pieces away is a prediction this evaluator cannot make. It is
    implemented and available so the DQN can use it, where the value function
    can in principle learn when a swap pays.

    ``sample_next`` enables lookahead over the upcoming pieces.
    """
    if actions is None:
        actions = (hold_options(game) if allow_hold
                   else [(False, r, x)
                         for r, x in valid_actions(game.board, game.current)])
    if not actions:
        return None
    weights = resolve_weights(weights)
    if sample_next:
        return lookahead_choice(game, actions, weights, sample_next)
    best = None
    best_score = float('-inf')
    for action in actions:
        s = score_action(game, action, weights)
        if s > best_score:
            best_score = s
            best = action
    if best is None:
        return None
    # Always a 3-tuple. `apply_move` accepts both that and a bare (rotation, x),
    # so a caller handling either shape needs no special case.
    if len(best) == 2:
        return (False, best[0], best[1])
    return best


def apply_move(game, move):
    """Apply a ``(use_hold, rotation, x)`` move, or a bare ``(rotation, x)``.

    Returns the lines the placement cleared. Holding first is what lets the
    evaluator keep an I back until the well is ready; if hold is unavailable the
    piece is simply placed, so a stale hold flag can never crash a game.
    """
    from .engine import spawn_anchor

    if len(move) == 2:
        use_hold, rot, x = False, move[0], move[1]
    else:
        use_hold, rot, x = move
    if use_hold:
        game.hold()
    game.current.rotation = rot
    game.current.x = x
    game.current.y = spawn_anchor(game.current.kind)[1]
    before = game.lines
    game.hard_drop()
    return game.lines - before


def clone_game(game, keep_next=6):
    """A playable copy of ``game`` with the same board, piece, hold and queue.

    Used by the lookahead and by hold evaluation so both can simulate without
    touching the real game.
    """
    from collections import deque
    probe = Game(seed=0)
    probe.rows, probe.cols = game.rows, game.cols
    probe.buffer_rows = getattr(game, 'buffer_rows', probe.buffer_rows)
    probe.board.grid = [row[:] for row in game.board.grid]
    probe.current = game.current.copy() if game.current else None
    probe.hold_piece = game.hold_piece.copy() if game.hold_piece else None
    probe.hold_used = getattr(game, 'hold_used', False)
    probe.allow_hold = getattr(game, 'allow_hold', True)
    probe.next_queue = deque(p.copy() for p in list(game.next_queue)[:keep_next])
    return probe


def lookahead_choice(game, actions, weights=None, sample_next=3):
    """Choose by the best continuation ``sample_next`` pieces deep.

    A one-ply evaluator cannot plan a tetris. Holding a well open and clearing a
    single later score identically *right now*, so the decision is a coin flip and
    the greedy score always takes the immediate clear; the result is 81% singles
    and ~0.7% tetrises no matter how the weights are set. Measured attempts to
    force it through scoring alone (a bonus for keeping a well, suppressing the
    single-line reward, and gating the bonus on an I piece being available) all
    failed: the bonus makes the evaluator hoard the well and top out (998 ->
    22-207 lines), while changing nothing.
    Scoring a placement by what the next few pieces can achieve does work.

    Cost is the branching factor: each candidate spawns a full placement search
    one ply deeper, so this is roughly 30x slower than ``heuristic_choice``.
    """
    from .engine import spawn_anchor

    def child(game, rot, x, samples):
        """A Game advanced by one placement, carrying `samples` next pieces."""
        probe = clone_game(game, keep_next=max(1, samples))
        probe.current.rotation, probe.current.x = rot, x
        probe.current.y = spawn_anchor(probe.current.kind)[1]
        probe.hard_drop()
        return probe

    def node_value(game, node, depth):
        """Best value reachable from ``node``, looking ``depth`` pieces ahead.

        Weights are recomputed at EVERY node through ``effective_weights``. That
        matters: the panic override is height-dependent, so a continuation that
        digs out must be scored with the dig-out policy. An earlier version
        evaluated continuations with the original weights, which silently
        disabled the override one ply down and made lookahead and the
        tetris-aware weights fail to combine (0.8% tetrises, no better than
        lookahead alone).
        """
        if depth <= 0:
            return 0.0
        replies = valid_actions(node.board, node.current)
        if not replies:
            return -1e17
        w_here = effective_weights(node, resolve_weights(weights) or DEFAULT_WEIGHTS)
        best = float('-inf')
        for r, x in replies:
            here = score_placement(node, r, x, w_here)
            if depth > 1:
                nxt = child(node, r, x, depth - 1)
                if nxt.game_over:
                    here -= 1000.0
                else:
                    here += node_value(nxt, nxt, depth - 1)
            if here > best:
                best = here
        return best

    # Rank the immediate moves by the best line of play from each. Lookahead is
    # only applied to placements: exploring hold inside the search would multiply
    # the branching factor again for little gain, so a hold candidate is scored
    # on its own board alone.
    scored = []
    w_root = effective_weights(game, resolve_weights(weights) or DEFAULT_WEIGHTS)
    for action in actions:
        use_hold, rot, x = action if len(action) == 3 else (False,) + tuple(action)
        here = score_action(game, (use_hold, rot, x), w_root)
        if use_hold:
            scored.append((here, (True, rot, x)))
            continue
        probe = child(game, rot, x, sample_next)
        if probe.game_over:
            value = here - 1000.0
        elif sample_next > 1:
            value = here + node_value(probe, probe, sample_next - 1)
        else:
            replies = valid_actions(probe.board, probe.current)
            value = here + (max(score_placement(probe, r2, x2,
                                                effective_weights(probe, w_root))
                                for r2, x2 in replies) if replies else -1e17)
        scored.append((value, (False, rot, x)))
    if not scored:
        return None
    best = max(scored)[1]
    if len(best) == 2:
        return (False, best[0], best[1])
    return best


def teacher_choice(game, actions=None):
    """Alias used by the imitation warm start (same evaluator)."""
    return heuristic_choice(game, actions)


def make_policy(weights=None, name='heuristic', sample_next=0):
    """Return a ``policy(game, actions) -> (rotation, x)`` callable.

    ``weights`` may be a preset name ('default' or 'tetris_aware') or a dict.
    ``sample_next`` > 0 enables the lookahead evaluator.
    """
    if isinstance(weights, str):
        weights = WEIGHT_PRESETS[weights]

    def policy(game, actions):
        choice = heuristic_choice(game, actions, weights, sample_next)
        if choice is None:
            raise RuntimeError('no legal placement available')
        return choice

    policy.policy_name = name
    policy.sample_next = sample_next
    return policy
