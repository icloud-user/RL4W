[README.md](https://github.com/user-attachments/files/32247533/README.md)
# tetrisrl — learning to play Tetris

A from-scratch Guideline Tetris engine and a reinforcement-learning agent that
learns to clear lines on CPU, plus a hand-tuned evaluator strong enough to be a
useful yardstick and an optional teacher.

The majority of the contents of this project was generated with ARTIFICIAL INTELLIGENCE. 

This project was written after reviewing the older `vibecode.py` / `dqn.py` code
in the parent folder. Their Tetris *engine* was usable and their pygame front end
was worth imitating; their *learning* was not, and the reasons are instructive
enough that this README documents them rather than hiding them.

```
pip install torch numpy pyyaml          # pygame too, for the renderer
cd tetrisrl

python -m tetrisrl train                # learn from scratch (~40 min on 12 cores)
python -m tetrisrl eval --games 60      # measure it over full games
python -m tetrisrl watch --policy best_rewardfix   # watch it play
python -m tetrisrl watch --policy search_tetris    # watch it build tetrises
python -m tetrisrl play                 # play it yourself
python -m tetrisrl menu                 # interactive launcher
python -m tetrisrl policies             # list every policy that can be loaded
```

**Run every command from `tetrisrl/`** (the directory containing this README and
`config.yaml`), not from the parent. `tetrisrl/` is the project root and also
contains the inner `tetrisrl/` package, so this layout means `python -m tetrisrl`
resolves to the package. From the parent directory it resolves to a namespace
package and `python -m tetrisrl` fails with *"No module named tetrisrl.\_\_main\_\_;
'tetrisrl' is a package and cannot be directly executed."*

Pre-trained weights ship in `checkpoints/`:

```
python -m tetrisrl eval  --policy best_rewardfix --games 60
python -m tetrisrl watch --policy best_rewardfix
```

## Results

Everything below is a **real game outcome** — lines cleared and the standard
Guideline score — measured on full games with a greedy policy, never the shaped
training reward. Rows are not all measured over the same sample size; the
`n` column says how many games, and the small-`n` rows should be read as rough.

| Policy | n | Lines/game (mean) | Median | Best | Score/game | Pieces/game | Cleared ≥1 line |
|---|---|---|---|---|---|---|---|
| **Learned DQN** (`best_rewardfix.pt`) | 40 | **66.9** | 63.0 | 158 | 47,360 | 212 | **40/40 (100%)** |
| Learned DQN (`best_finetuned.pt`) | 40 | 58.9 | 45.0 | 270 | 44,225 | 192 | 40/40 (100%) |
| Learned DQN (`best.pt`) | 60 | 42.6 | 30.0 | 168 | 26,536 | 140 | 56/60 (93%) |
| Learned DQN (`latest.pt`) | 60 | 26.7 | 25.0 | 79 | 8,142 | 113 | 60/60 (100%) |
| Heuristic `heuristic` (1-ply) | 5 | 3,293.6 | 3,996.0 | 5,255 | 89,836,400 | 8,266 | 5/5 (100%) |
| Heuristic `heuristic_tetris` | 6 | 2,420.0 | — | — | — | — | 6/6 (100%) |
| **Search `search_tetris`** (3-ply, beam 6) | 6 | 551.2 | 498.0 | 1,304 | 4,892,100 | 1,420 | 6/6 (100%) |
| Random legal placement | — | 0.0 | 0.0 | 0 | 0 | ~24 | 0% |

`search_tetris` is the newest policy and the one built specifically for the tetris
goal: **52.9% of its clear events are tetrises**, against 3.9% for
`heuristic_tetris` and 0.6% for the default heuristic. It clears far fewer lines
than the plain heuristic — that is the trade it was asked to make — and about
eight times as many tetrises per piece. See "the search that finally built
tetrises" below.

Two honesty notes on this table.

**The learned agent does not beat the heuristic, and it is not close.** The
heuristic is a hand-tuned Dellacherie-style evaluator, and hand-tuned evaluators
are extraordinarily strong at Tetris — it clears thousands of lines and never
dies within the episode cap at all. Its numbers are therefore *lower bounds*
governed entirely by `--max-pieces`, not by ability. What the DQN demonstrates is
that the learning pipeline works end to end, from random play to routine
line-clearing, with no hand-written placement rules anywhere in the agent.

**The 131.3 figure that appeared in an earlier revision of this file was a
small-sample artifact** and has been removed. `best_finetuned.pt` was measured at
131.3 lines/game on one 60-game sample and at 58.9 on a seeded 40-game sample
(seed 7). Game-to-game variance is enormous — the same checkpoint has produced a
single 533-line game — so no result below ~40 games means much, and the
comparison that matters (`best_rewardfix` vs `best_finetuned` on an identical
seed) now favours `best_rewardfix`.

`best_rewardfix.pt` is the current champion at **66.9 lines/game, 100% of games
clearing at least one line, 212 pieces survived on average.**

### Learning curve

```
round    1/300 | lines    0.03 | score        3 | eps 0.99  | loss 1.35
round   40/300 | lines    0.11 | score       11 | eps 0.75  | loss 1.38
round  100/300 | lines    0.42 | score       42 | eps 0.38  | loss 1.81
round  140/300 | lines    0.99 | score      103 | eps 0.13  | loss 2.08
round  189/300 | eval 50.42 lines/game (max 193)              *best*
round  216/300 | lines   17.68 | score     4031 | eps 0.02  | loss 13.19
```

Performance peaked around round 189 and then declined while the training loss
kept climbing — the classic signature of Q-value overestimation once exploration
finishes, and why `best.pt` (42.6) beats `latest.pt` (26.7) despite `latest`
having 20% more gradient steps.

`tools/finetune.py` continues from the best weights at a lower learning rate and a
re-annealed epsilon, writing to separate files so a bad run cannot overwrite the
checkpoint that is known to be good.

## How it works

**Engine** (`engine.py`) — a correct Guideline implementation: SRS rotation with
the published wall-kick tables, 7-bag randomiser, hold, lock delay, block-out and
lock-out, and Guideline scoring (100/300/500/800 × level, back-to-back and combo
bonuses). No dependencies, no RL, fully testable in isolation.

**Features** (`features.py`) — the agent never sees the raw board. It sees 37
numbers describing a **(board, candidate placement)** pair: column heights, holes,
bumpiness, aggregate height, row and column transitions, well depths, one-hot
encodings of the current and next piece, and the placement's own statistics
(landing depth, lines cleared, holes created, resulting height).

**Agent** (`agent.py`, `model.py`) — Double DQN with a target network, n-step
returns (n=3), masked actions, and a replay buffer that oversamples rare
informative transitions. A plain MLP, not a duelling network (see problem 6).

**Reward** — deliberately mild shaping: a small per-step cost, line clears
dominating, and a death penalty. The per-step cost is what makes survival worth
something, and it also pressures the stack to stay low, since a low stack means
more steps before dying.

### The action abstraction

A move is `(use_hold, rotation, x)` — one of 4 rotations × 10 columns, optionally
after a hold swap. Roughly 34–36 of the 40 candidates are legal for any given
piece, and all of them are scored in a single batched forward pass, which is what
makes CPU training viable.

There is no hardcoded placement table and no per-piece case analysis. Legality
comes from `valid_actions()`, which works in **column-height space** rather than
simulating drops: a placement is legal if the piece fits at its spawn row, at
which point the clearance is the minimum height over the columns it covers. That
is about 8× faster than simulating every drop, and `tests/test_engine.py` asserts
it agrees exactly with actually playing the moves.

The abstraction is a **superset of physically reachable placements**: a move is
realised by *teleporting* to `(rotation, x)` at the spawn anchor and calling
`hard_drop()`. The agent never rotates, moves, or wall-kicks, so SRS kicks are
never exercised by the agent — they matter only for human play and for the
renderer's animation, though they are fully implemented and tested.
`tools/check_reachability.py` measures the gap: **4.5% of enumerated placements
are unreachable** by real play, while roughly 110 reachable placements per board
are forbidden by the abstraction.

**T-spins are not represented at all.** There is no T-spin scoring, no T-spin
feature, and no heuristic term for them, and kick-only T-spin slots may be
excluded outright. This is a genuine capability gap, not a deliberate scoping
decision, and it is the largest single thing missing relative to real Guideline
play.

## Every file, and what it is for

`tetrisrl/tetrisrl/` is the package (inner), `tetrisrl/` is the project root
(outer, containing this README).

### Package — `tetrisrl/`

| File | Lines | Purpose |
|---|---|---|
| `engine.py` | 678 | The Guideline Tetris engine and nothing else. `Piece`, `Bag` (7-bag randomiser), `Board`, `Game`. Owns the SRS cell tables and all four kick tables (`KICKS`, `IKICKS`, `KICKS_180`, `IKICKS_180` — the 180° ones are the widely used extension, not SRS proper), `spawn_anchor`, `Game.spawn_blocked`, the fast `valid_actions` enumerator, `simulate_placement`, and kick-aware `try_rotate`/`rotate` for real play. Pure Python, no torch. |
| `features.py` | 438 | Everything that turns a board into numbers. `board_metrics` returns the 9-tuple `(heights, holes, bumpiness, aggregate, row_trans, col_trans, wells, max_height, rows_with_holes)`, with `metrics_for` as its memoised entry point (`clear_cache`, `cache_size`). Also `placement_vector` (the scored input), `board_vector_game`, `n_features`/`board_width`/`fingerprint`/`check_fingerprint` for the four feature layouts, `holes_of`, and `tetris_readiness`. Knows nothing about learning. |
| `model.py` | 68 | `QNet`, the plain MLP, and `N_ACTIONS = 40`. Deliberately tiny: small (0.1) initialisation so the untrained network does not emit large spreads. |
| `agent.py` | 656 | The learner. Double DQN with target network, masked action selection, the reward calculation (`placement_reward`, `placement_stats`, `reward_config`, `REWARD_PRESETS`), the `Agent` class (`learn`/`learn_batch`/`imitate`/`anneal_lr`/`save`/`load`), the `action_id`/`action_from_id`/`valid_mask` action encoding, and `_guess_version` for width-based feature-version inference. |
| `replay.py` | 207 | Two buffers behind one interface: `Replay` (uniform, with rare-event oversampling) and `PrioritizedReplay`. Both yield `(example, weight, index)` triples so the update code is shared. `make_replay` picks by config. |
| `train.py` | 493 | Parallel self-play and the training loop. `play_episode` (one worker's episode, with epsilon/teacher/exploration), `_to_targets` (the n-step return with a masked bootstrap — the bug-prone part, and validated), `_worker`, `_score_all`, `_candidate_vector`, `evaluate`, `pretrain_imitation`, and `train`. Owns `MAX_PIECES_PER_EPISODE = 3000` and `_stamp_best`. |
| `evaluate.py` | 70 | Measurement, deliberately separate from training. `run_episode` plays a full game through `apply_move` using real outcomes only; `evaluate_policy(..., max_pieces=20000)` aggregates; `format_summary` prints the line you see in every result above. |
| `heuristic.py` | 615 | The hand-written evaluator, and the optional teacher. `DEFAULT_WEIGHTS`, `TETRIS_AWARE_WEIGHTS`, `WEIGHT_PRESETS`, `score_placement`/`score_action`, `clone_game`, `covered_cells`, `best_well`, `hold_options`, `apply_move`, `heuristic_choice(..., allow_hold=False)`, `lookahead_choice`, `teacher_choice`, `make_policy`, and `effective_weights`/`resolve_weights` (which is where the `panic_height` override lives). |
| `policies.py` | 216 | Policy factories so callers never touch torch directly. `load_policy` by name with lazy imports and safe fallbacks, `make_dqn_policy` (feature-version aware), `make_heuristic_policy`, `make_search_policy`, the `CHECKPOINT_NAMES`/`PREFERRED_ORDER` table, and `list_policies` for the CLI. |
| `search.py` | 697 | The beam search, and the reason `search_tetris` exists. A compact column-bitmask board model (`board_masks`, `placements`, `apply_placement`, `locks_out`, `spawn_blocked`), two well definitions (`ready_depth`, `well_depth`), the evaluation (`evaluate`, `TETRIS_SEARCH_WEIGHTS`, `DEFAULT_SEARCH_WEIGHTS`, `COLD_CLEAR_WEIGHTS`, `clear_value`), and `search_move` — a depth-3, beam-6 search over placements *and the hold swap*, ranked by path rewards plus a static board score. Pure Python, no torch, no engine mutation. |
| `render.py` | 1046 | The whole pygame front end: `Layout`, `Renderer`, `HumanInput` (DAS/ARR, soft drop, lock delay), `AgentPlayer` (animates a policy's chosen move before dropping), and the `run_human`/`run_agent`/`run_menu` entry points. Also the only place that draws a ghost piece, NEXT ×5, HOLD, score/level, and the stack histogram. |
| `__main__.py` | 217 | The CLI: `train`, `eval`, `watch`, `play`, `menu`, `policies`. Training flags include `--rounds --workers --episodes-per-worker --resume --tag --publish --reward-preset --epsilon-decay-steps --no-lr-decay --lr-decay-fraction --quiet`. |
| `__init__.py` | 90 | `load_config`, with a complete set of defaults so every key is defined even if `config.yaml` omits it, and the package version. |

### Project root

| File | Lines | Purpose |
|---|---|---|
| `README.md` | this file | Results, architecture, the full file inventory, the chronology of what was tried, and the problems that were hit. |
| `config.yaml` | 151 | Every hyperparameter and reward shaping knob in one place: net width, buffer, n-step, epsilon schedule, `feature_version`, `replay` type, target sync, gradient budget, reward preset, panic height. Current defaults: `feature_version: 1`, `replay: uniform`, `epsilon_decay_steps: 250000`, `lr_decay_fraction: 0.0`, `grad_steps_auto: true`, `grad_steps_per_sample: 0.25`, `imitation_steps: 0`. |
| `checkpoints/` | — | See the checkpoint table below. |
| `*.log`, `*.err` | — | stdout/stderr captures from the long A/B runs (`ab_uniform`, `fin_on`, `fin_off`, `fix`, `train_v2`, `uniA`). Kept because the schedules and per-round numbers in them are the evidence for several claims in this README. Safe to delete. |

### Checkpoints — `checkpoints/`

| File | What it is |
|---|---|
| `best_rewardfix.pt` | **Champion.** 66.9 lines/game (40 games, seed 7). Best of the reward-fix lineage. |
| `rewardfix.pt` | Latest weights of that run, not necessarily best. |
| `best_finetuned.pt` | Best of the `finetune.py` continuation. 58.9 on the same seed. |
| `latest_finetuned.pt` | Latest of that continuation. |
| `best.pt` | Best of the original 300-round run. 42.6 lines/game. |
| `latest.pt` | Latest of the original run. 26.7 lines/game. |
| `fin_on.pt`, `best_fin_on.pt` | A/B leftovers: learning-rate annealing **on**. 21.2 lines/game. |
| `fin_off.pt`, `best_fin_off.pt` | A/B leftovers: annealing **off** (the winner). 30.2 lines/game. |

`fin_*` are dead experiment artifacts not registered in `policies.py`; they are
kept only as the record of the annealing A/B and can be deleted. Prefer
`best_rewardfix.pt`, then `best_finetuned.pt`.

### Tools — `tools/`

Every script is runnable standalone and inserts the project root on `sys.path`,
and each is guarded by `if __name__ == '__main__'` because training uses
`ProcessPoolExecutor`.

| File | Lines | Purpose |
|---|---|---|
| `evaluate.py` | 60 | The measurement command used for every result in this README. Evaluates one policy or all of them, over full games, with `--games --seed --max-pieces --policy --checkpoint`. |
| `run_training.py` | 31 | Minimal launcher for a fresh training run with the config defaults — the pattern to copy when starting a long run. |
| `finetune.py` | 70 | Continues from the best checkpoint at a lower learning rate and re-annealed epsilon, writing to separate filenames so a bad run cannot clobber a good checkpoint. |
| `clear_mix.py` | 87 | The clear-type histogram (singles/doubles/triples/tetrises as percentages of clear events). This is the instrument that made the tetris problem measurable. |
| `diagnose.py` | 119 | Why did it die? Reports the distribution of game lengths, the clear mix, and the board state at death, to separate "dies on a low clean board" (a policy problem) from "dies on a tall holed board" (a credit-assignment problem). |
| `compare_runs.py` | 126 | Side-by-side comparison of two training logs, so an A/B is read off the per-round numbers rather than a single final eval. |
| `compare_features.py` | 134 | Held-out comparison of feature versions on a fixed dataset — this is what showed v2/v3 gained nothing once the sample was large enough. |
| `check_reachability.py` | 172 | Measures how much of the enumerated action space is actually reachable by rotate/move/soft-drop play, and how many reachable placements the abstraction forbids. |

The two most useful of these:

```
python tools/clear_mix.py --policy search_tetris --games 6 --cap 2500
python tools/clear_mix.py --policy heuristic_tetris --games 6 --cap 2500
python tools/diagnose.py  --policy best_rewardfix --games 30
```

### Tests — `tests/`

Run all four directly; each is self-contained and prints one line per check —
**311 checks total, all passing.**

| File | Lines | Checks | Covers |
|---|---|---|---|
| `test_engine.py` | 639 | 195 | Rotation tables self-consistent under 90° rotation (the one invariant that catches a table built by reflection), 7-bag properties, scoring and level scaling, lock-out and block-out, hold, wall kicks, and — most importantly — that the fast placement enumeration and `simulate_placement` both agree exactly with actually playing the moves. |
| `test_agent.py` | 420 | 47 | Feature layout (every slot written and finite, per version), reward shaping, action masking, masked bootstrap targets, checkpoint round-trip, and the renderer's choice-coercion path. |
| `test_train.py` | 329 | 46 | n-step return maths, replay sampling and weights, that learning actually reduces error on a toy problem, and that each feature version's vectors are accepted by that version's network (the feature-drift guard). |
| `test_search.py` | 382 | 23 | The search's bitmask model against the engine — `placements` versus `valid_actions`, `apply_placement` versus `simulate_placement`, the landing row versus the ghost row, spawn and lock-out tests — plus the well rules, move legality, determinism, that an available tetris is taken, that the hold slot is actually spent, and that the tetris weights beat the plain-search control on clear mix. |

```
python tests/test_engine.py     # 195 checks
python tests/test_agent.py      #  47 checks
python tests/test_train.py      #  46 checks
python tests/test_search.py     #  23 checks
```

## The process — what was tried, in order

### 1. Build the engine, and get the tables right

The engine came first, because every wrong number downstream would be
uninterpretable otherwise. The subtlest failure here was the rotation tables: the
hand-written SRS tables were **reflections**, not rotations, so rotation state 2
was wrong for the I, S and Z pieces. The fix was to stop writing tables by hand
and *generate* them — rotate the spawn state 90° three times, mapping `(x, y) →
(-y, x)` in y-up coordinates and negating y to reach the board's y-down
convention. `tests/test_engine.py` now pins the invariant that caught it: each
state must be the previous state rotated, and rotating four times must return to
the start.

### 2. Pick the abstraction, then fix it twice

Decision: learn over **final placements** `(rotation, x)` with hard drop, not over
keystrokes. That collapses a variable-length action sequence into one choice, makes
the action space a fixed 40, and lets every candidate be scored in one batched
forward pass. The cost is that wall kicks and T-spins leave the agent's world —
see the abstraction section above.

Two bugs followed directly from this choice. `simulate_placement` dropped the
piece from its *current* position instead of the spawn anchor, so the board the
evaluator scored was not the board that got played; a differential test against
real play now guards it. And `valid_actions` originally worked by simulating every
drop, until it was rewritten in column-height space (~8× faster), with the same
differential test proving the two agree.

### 3. First training runs: everything worked, badly

Three silent bugs made the first runs meaningless, and all three produced
*plausible* numbers rather than crashes.

**Pieces spawned inside the visible playfield.** A wrong constant put each spawn
state's top row on board row 0 instead of in the hidden buffer
(`_SPAWN_TOP_ROW`); it is −2. Games ended after ~50 pieces and the heuristic
scored 3.7 lines/game. One constant took it to 1,197. *Everything measured before
that fix was invalid.*

**The action argmax was unmasked.** The network emits 40 Q-values regardless of
the piece; only ~34 are legal. Illegal outputs are never trained and sit near
zero, while legal ones are trained toward negative rewards — so an unrestricted
`argmax` *always* chose an impossible placement. Symptom: the agent survived
hundreds of pieces and cleared exactly zero lines. Fixed by only ever scoring
legal candidates, with a fallback for an illegal pick.

**The n-step bootstrap was unmasked.** Same root cause, worse consequence: the
target is a max over next-state values, so one untrained illegal output could
poison every target in the batch. Fixed by storing action masks with each
transition and masking the bootstrap.

### 4. The reward, in three attempts

The reward went through more revisions than anything else, and the honest summary
is that **most of them did not help.**

| Attempt | Result | Verdict |
|---|---|---|
| Per-step cost to make survival valuable | made the first runs survivable | kept |
| **Rescale the reward / add a positive per-step bonus** | **66.4 → 10.6 lines, a 6× regression** | rejected |
| Alternative presets (`michielcox`, `michielcox_small`, `strong_death`, `tetris_heavy`) | no reliable improvement over `v1_original` | default stays `v1_original` |

The rescaling regression has a clear mechanism worth remembering: a positive
per-step bonus makes the *return* dominated by survival, so the agent optimises
for not dying rather than for clearing lines, and it stops clearing. That
experiment is why `v1_original` is still the default despite being the first thing
written.

### 5. Richer features — and finding out the first "gain" was noise

Feature versions 2, 3 and 4 (44, 46 and 49 numbers) added hold state, next-piece
detail, and placement-conditioned terms. On small samples v2 looked like an
improvement. On a properly held-out comparison with adequate data it was **not**:

| Features | Held-out performance |
|---|---|
| v1 (37) | 68.9% |
| v2 (44) | 68.6% |
| v3 (46) | 68.9% |

The "gain" was small-sample noise. `feature_version` stays at 1, and
`tools/compare_features.py` exists specifically so that claim can be re-checked
rather than remembered.

The same fate met `rows_with_holes`, the highest-weighted term in DT-20: it is
real, but it is **redundant with a correctly weighted hole term**. It fully
recovers a badly-tuned agent (123 → 598 lines) and adds nothing at all once the
hole weight is right.

### 6. Architecture: a duelling head that diverged, and a starved network

**The duelling head diverged.** After imitation its Q-values reached ~−32,000
with spreads in the 100,000s. It was replaced with a plain MLP, which sits at
Q ≈ ±30, and `model.py` has been a plain MLP since. A smaller initialisation
(0.1) was added for the same reason.

**`placement_features` starved the network.** An early design fed the network
board features plus a description of the placement, and every candidate placement
looked identical to the network, so there was nothing to discriminate on. The fix
was to score **(board, placement) pairs** — the design that survives today.

### 7. Prioritized replay — implemented, measured, rejected

`PrioritizedReplay` was implemented properly and it **regressed**: loss climbed
1.27 → 1.64 and measured performance fell. The default is `uniform`, and the
prioritized path is kept in `replay.py` so the negative result is reproducible
rather than folklore.

### 8. Finding the real lever: the exploration schedule

After all the reward and architecture experiments moved things by a few percent,
the thing that actually mattered was **epsilon**. An early config comment of mine
advised "sizing epsilon decay to the run", which led to a 40,000-step decay; that
floored epsilon at round 76 with a still-untrained network, and both A/B arms
collapsed to 0.04 and 0.29 lines/round. The run that worked kept epsilon at 0.61
at round 189 and reached 66 lines/game in 150 rounds.

This is the clearest lesson in the project: **the arm that explored longest and
learned at a constant rate won.** Everything about reward shaping, feature
engineering and buffer strategy was second-order by comparison.

### 9. Adaptive update budget — a real bug, fixed

Training took a fixed 300 gradient steps per round while the experience per round
grew as the agent improved (longer episodes), so updates-per-transition collapsed
from ~1:5 early to worse than 1:2 later — precisely when learning matters most.
The budget now scales with the data:

```yaml
grad_steps_auto: true
grad_steps_per_sample: 0.25     # updates per collected transition
min_grad_steps_per_round: 200
```

### 10. The tetris problem — the user-facing goal that took the longest

The agent cleared lines but almost never cleared four at once. Measured with
`tools/clear_mix.py`:

| Policy | Singles | Doubles | Triples | **Tetrises** |
|---|---|---|---|---|
| `heuristic` | 81.1% | 15.9% | 2.2% | 0.6% |
| `heuristic2` (2-ply) | 74.6% | — | — | 0.7% |
| `best_rewardfix` (DQN) | 51.4% | — | — | 1.7% |
| `heuristic_tetris` | 26.9% | 61.9% | 11.6% | 3.9% |
| **`search_tetris`** (3-ply beam + hold) | 18.5% | 14.3% | 14.3% | **52.9%** |

The last row is one tetris per 1.9 clear events, measured over the same protocol
(six games, 2,500-piece cap) — see section 16 for how it got there and what it
cost.

The working idea came from [Cold Clear](https://github.com/MinusKelvin/cold-clear)
(`bot/src/evaluation/standard.rs`), whose key mechanism is **negative clear
values**: `clear1: -143, clear2: -100, clear3: -58, clear4: +390`. Clearing one
line is *worse than nothing*; only a tetris pays. Paired with a `panic_height`
override that restores positive clear values when the stack reaches a threshold
(default 10 in `tetris_aware`), this took the tetris rate from 0.6% to **3.9% —
6.5× better** — at the cost of ~17% of the lines.

Everything else from Cold Clear, bolted on without its search, **failed**:

| Change | Lines | Tetrises | Outcome |
|---|---|---|---|
| `well_depth_bonus = 3` | 37 | **15.4%** | hoards wells, dies in 37 lines |
| `well_deep_bonus = 40` | 233 | 8.9% | same failure, slower |
| `well_column` (Cold Clear's own array) | 77 | 6.6% | 6/6 games dead |
| `covered_cells` penalty | 317 | 3.0% | no net gain |

Three further attempts to raise it past 3.9% all failed, and the reason is
geometric rather than a tuning failure. Instrumenting the board-height histogram
during play shows the stack sitting at **height 3–9 for most of the game**, with a
4-deep well occupying 40 of the 200 cells. There is no room to build a tetris from
height ~4 without first clearing lines, and clearing is what destroys the well.
The panic override that keeps the agent alive fires often enough (7.6% of moves)
that its threshold and reward values barely matter — which is why "strong dig" and
"cheap dig" panic settings produced **byte-identical** results.

Combining lookahead with the tetris-aware weights does not compound either: 2-ply
on top of tetris-aware gives 0.8% tetrises, the same as 2-ply on the default
weights, and *worse* than 1-ply tetris-aware. The cause is a bug I introduced and
then fixed — the lookahead's inner evaluation bypassed the `panic_height`
override — but fixing it made results worse, because **lookahead and tetris-aware
are substitutes, not complements.**

Cold Clear runs all of these inside a deep MCTS search. Bolted onto a greedy
one-ply evaluator, every one of them converts directly into well-hoarding and
death. **The safe part of its scheme is the negative clear values; the rest needs
the search Cold Clear pairs them with.**

### 11. Hold — implemented, usable, and measurably not worth it

Hold was a genuine gap: the engine supported it (`Game.hold()`, `hold_used`) and
the feature vector even carried a hold one-hot, but **nothing ever called it**. It
is now part of the move representation — a move is `(use_hold, rotation, x)`,
`hold_options()` offers every placement both directly and after a swap, and
`apply_move()` executes either shape.

The heuristic then measurably did not want it:

| Weights | Hold | Lines | Singles | Doubles | Triples | Tetrises |
|---|---|---|---|---|---|---|
| default | off | 599.0 | 81.4% | 15.9% | 2.2% | 0.5% |
| default | **on** | 598.7 | **93.3%** | 5.8% | 0.7% | **0.2%** |
| tetris_aware | off | 596.3 | 23.1% | 61.9% | 11.6% | **3.5%** |
| tetris_aware | **on** | 596.0 | 5.3% | 82.1% | 10.4% | 2.2% |

The cause is structural and is the same one behind the well bonuses and
lookahead: **holding is score-neutral.** The board after "hold, then place" is
identical to placing directly, so a one-ply evaluator has no basis for preferring
it, and it churns roughly one piece in three through the slot for nothing. An
explicit bias — bank an I, or park an S/Z — did not rescue it (the avoid-S/Z
variant cut lines 596 → 477 while tetrises stayed at 3.8%).

Deciding to keep an I for a tetris three pieces away is a *prediction*, and a
one-ply evaluator cannot make it. So `allow_hold` defaults to **False** for the
heuristic — the capability is implemented and tested so the DQN, whose value
function can in principle learn when a swap pays, can use it.

### 12. Learning-rate annealing — tested, and it HURT

The hypothesis: a 300-round run peaked near round 189 and then declined while the
loss climbed 1.35 → 17.4, and `best.pt` beat `latest.pt` by 60% despite fewer
gradient steps. That points at value-function divergence, whose standard cause is
a constant learning rate. Annealing was implemented — hold for a warmup fraction,
decay linearly to a floor, hold — on the same transition clock as epsilon.

A matched A/B, ~420 rounds, 7 workers per arm, identical in every other respect:

| `lr_decay_fraction` | Lines/game (60 games) | Pieces/game | Final loss |
|---|---|---|---|
| 1.0 (annealed) | 21.2 | 85 | 5.29 |
| **0.0 (constant)** | **30.2** | 122 | 9.89 |

**Annealing lowered the training loss and produced a worse agent.** The decay is
tied to `epsilon_decay_steps`, so it completed around round 385 and then froze the
rate while the agent was still improving, while the constant-rate arm kept
improving and survived 44% longer per game.

The default is now `lr_decay_fraction: 0.0`, and the divergence remains
**unexplained**. It is worth recording that a lower loss did not mean a better
policy; loss and measured strength disagreed in *both* directions across these
experiments.

### 13. The renderer bug that looked like suicide

`python -m tetrisrl watch --policy heuristic` appeared to kill itself
immediately — the heuristic plays thousands of lines headless. The cause was in
`render.AgentPlayer._coerce`, which read `choice[0], choice[1]` from the choice
tuple. For a heuristic choice of `(use_hold, rotation, x)` that reads
`rotation=False` — which never matches a legal action, so it silently fell back to
`actions[0]` for every single piece. Distinct placements used went from **1 to
29–33** after the fix, with a regression test added.

A related non-bug worth recording: the heuristic **never sets `game_over`** — it
just stops at `--max-pieces`. Seeing "800 pieces, 319 lines, game_over=False" is
the cap, not a death.

### 14. Bugs worth remembering, and the guards added

Both of these were silent — no exception, just worse play.

**An uninitialised feature slot.** `placement_vector` allocated with `np.empty`
and left one slot per version unwritten, so it held whatever the allocator left
behind (observed: ~8.8e12, in the *highest-weight* input). A checkpoint cannot be
reproduced when part of its input is undefined memory; a trained checkpoint
dropped from 137.7 to 62 lines with no error. Now `np.zeros`, every slot written
explicitly, plus a **feature-layout fingerprint** recorded in every checkpoint
that warns on mismatch, and tests pinning every slot as written and finite.

**Feature-version drift between the agent and its workers.** The training workers
built features with the feature module's default version while the agent's config
selected version 1, so workers returned 44-wide vectors to a 37-input network.
Every run died after ~9 rounds with `mat1 and mat2 shapes cannot be multiplied
(34x44 and 37x256)`. The agent's version is now threaded through every feature
construction, and `tests/test_train.py` asserts each version's features are
accepted by that version's network.

Two smaller ones: the original v1 layout could not be reconstructed faithfully
(the `max_agg` variant is correct — 110 lines vs 62 for `agg_agg`), so the
original 137.7 is not reproducible; and `_I_COLUMNS` was set to `(2,3,4,5,6,7)`,
which is the *horizontal* spawn range, when a vertical I is placeable in any
column 0–9 in this abstraction. That one silently removed legal I placements.

**Known dead code, found while writing this README.** A reference-count pass over
the package turned up four definitions that nothing calls:

| Symbol | File | Note |
|---|---|---|
| `_spawn_blocked`, `_build_spawn_cells`, `_SPAWN_CELLS` | `agent.py` | Defined **twice, verbatim** (lines 101–128 and again 221–248). The second shadows the first, so behaviour is unaffected. |
| `_holes_of` | `agent.py` | A one-line wrapper around `features.holes_of`. Both real call sites (lines 161, 340) import and call `holes_of` directly, so the wrapper is never used. |
| `_height_summary` | `agent.py` | Never referenced. |
| `make_policy` | `heuristic.py` | Never referenced; `policies.make_heuristic_policy` is the live path. |

All are harmless — none changes behaviour — but they are leftovers from
refactors, and this is the kind of thing a verification pass over documentation
tends to surface. Worth deleting; `simulate_placement`, `teacher_choice`,
`metrics_for`, `board_vector_game` and `action_id` were checked the same way and
are all genuinely live.

### 15. Operational lessons that cost real time

**Long runs must be launcher-managed background jobs.** Launching training with
`Start-Process` from a transient shell got the process killed silently, with an
empty stderr file that looked like a crash. Every long run in this project was
subsequently launched as a managed background job with its stdout and stderr
captured to files.

**`ProcessPoolExecutor` needs a real console.** Under a restricted sandbox the
worker pool fails with `PermissionError: [WinError 5]` because it cannot open
named pipes; training needs unrestricted file access. On Windows the `__main__`
guard is also mandatory, which is why every tool and the CLI have one.

**Training volume is the binding constraint.** Roughly **460k transitions** were
needed to reach 64 lines/game; ~100k is simply below the learning threshold, and
this explains most of the "regressions" seen in short runs.

### 16. The search that finally built tetrises

Ten rounds of weight tuning could not push the one-ply evaluator past 3.9%
(section 10), and the reason is structural: a score that looks at one placement
cannot value an I it has not spent yet. The fix was not another term but another
*look* — `search.py`, a beam search over the next few pieces with the hold slot in
the action space.

The board is ten column bitmasks, so a placement and a line clear are integer
operations; that is what makes a search affordable in pure Python. A move is still
`(use_hold, rotation, x)` with a hard drop, so nothing about the action
abstraction changed. Each node is ranked by the rewards collected along the way
(Cold Clear's clear values) plus a static score of the board it reached. The
weights, the failed experiments and the two well definitions all live in that file
with their measurements attached.

**What moved the number, in order of measured effect** — six seeds, 1,500-piece
cap, `search_tetris`:

| Change | Tetrises/game | Tetris share | Pieces/game |
|---|---|---|---|
| Beam search, 3 ply, no hold in the candidate list | 29.3 | 19.2% | 988 |
| **+ the hold swap at the root** | **72.3** | **56.9%** | 1,048 |
| − hold again (control) | 29.3 | 19.2% | 988 |

Hold is the whole story, and it is the thing section 11 measured as *worthless*
for the greedy evaluator. That is not a contradiction: holding is score-neutral
for a one-ply score — the board after "hold, then place" is identical to placing
directly — so only a search can see that keeping an I is worth something. With the
swap in its candidate list the policy banks an I, builds the well over the next
few pieces, and spends it; holds are used on 44% of pieces.

**What failed here as well**, all measured and all reverted:

- **Porting Cold Clear's whole weight table.** Their ratios are the interesting
  part — a cavity cell at −173 against height at −39, so a hole costs about four
  rows — and adopting them wholesale measured *worse* in this engine: 6.2%
  tetrises and 7 deaths in 8 games. Their weights are tuned for a deep MCTS with
  garbage, combos and T-spins; a shallow beam that keeps the stack pinned low
  never builds the four-row structure a tetris needs. The preset survives as
  `cold_clear` in `search.py` so the claim can be re-checked rather than
  remembered.
- **A "danger mode"** that flipped the small-clear values positive above a height
  threshold. It made digging *profitable*: the policy grew to height 13, farmed
  singles at +140 each and never kept a well at all (readiness 0 on 84% of moves,
  3.9% tetrises). Replaced by a quadratic height pressure, which is never positive
  for a single.
- **A more literal readiness test** — count the rows above the well that are
  filled in every other column, rather than requiring the other columns to be
  gapless. It reads better and plays much worse: 379 pieces and 8 deaths in 8
  games against 930 and 5. The strict version quietly refuses to chase a tetris on
  a board with holes in it, which is exactly the board where chasing one is the
  wrong plan.
- **Extra-pit charges, wider beams (12), 4-ply.** No better; 4-ply is far too slow
  to measure with.
- **Well-depth discipline** is the one survival term that earned its place
  (`well_excess: -40` in `search.py`): rows under a deep well can never be cleared
  until it is filled, so a well that outruns the stack is a trap that only closes.
  It took tetrises/game from 19.8 to 26.0 and pieces/game from 810 to 965.

**What it costs.** About 15× the plain heuristic per move (roughly 14 ms/piece
here, so a six-game 2,500-piece measurement runs in about two minutes); fewer
lines (551/game against 785 for `heuristic_tetris` and thousands for the plain
heuristic); and more deaths — 5 of 6 games topped out inside the 2,500-piece cap,
against 2 of 6 for `heuristic_tetris`. It is a tetris machine, not a survival
machine, which is the trade it was asked to make.

## What this says about the approach

The agent's architecture has been validated from several directions: grouped
final-placement actions with hard drop, Dellacherie-style board statistics, and a
Double DQN with masked n-step targets — the same shape used by published agents
that clear 1000+ lines. The experiments above consistently show the bottleneck is
not representation and not reward design. It is **training volume and exploration
scheduling.**

For the ceiling: the strongest modern (Guideline/SRS) bots, such as
[Cold Clear](https://github.com/MinusKelvin/cold-clear), get their strength from
**search** over the 7-bag with hand-picked features whose weights are tuned by
genetic algorithm — not from a learned value function. A DQN over board statistics
will not match that, and this project does not claim to.

## Notes and limitations

- **The learned agent is far weaker than the heuristic.** See the results table.
- **`search_tetris` is the tetris specialist, not the best policy overall.** It
  clears 53% tetrises and far fewer total lines, and it tops out more often than
  any other policy here. Use it when tetrises are the goal.
- **Hold is the strongest single lever in the search** — without the swap in its
  candidate list the same search drops from 57% to 19% tetrises.
- **Small samples lie.** Game-to-game variance is huge; use ≥40 games and a fixed
  `--seed` before believing any comparison, including the ones in this file.
- **Loss is not strength.** They disagreed in both directions here.
- **T-spins are not modelled** — no scoring, no feature, no heuristic term.
- The action abstraction is a superset of reachable placements (4.5% unreachable).
- **Training is CPU-only** and uses `ProcessPoolExecutor`; on Windows the entry
  point must be guarded by `if __name__ == '__main__'`.
- **The replay buffer is not checkpointed**, so a resumed run refills it over the
  first few rounds.
- `latest.pt` is the most recent weights, which are not always the best. Prefer
  `best_rewardfix.pt`, then `best_finetuned.pt`.
- Episodes are capped at 3,000 pieces during training. A well-trained agent would
  otherwise play indefinitely and one episode would dominate a round. Truncation
  is not treated as death.
- The heuristic never sets `game_over`, so it is always stopped by a piece cap.
