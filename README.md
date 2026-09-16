# tetrisrl — learning to play Tetris

> ## The majority of the contents of this project was generated with ARTIFICIAL INTELLIGENCE.
>
> Nearly every line of code here the engine, the features, the agent, the search,
> the versus rules, the renderer, the tools, the tests and most of this README
> was written by AI. The human author set the goals, chose the
> direction, ran the games, reported what felt wrong, and verified behaviour in
> play; the code and prose were machine-generated and then measured.
>
> Practically, that means: **treat the claims in this file as measured, not
> authoritative.** Every number is reproducible from the commands given, the test
> suite pins the behaviour that matters, and the sections below deliberately keep
> the failures and dead ends alongside the wins — because a language model's
> mistakes are invisible unless somebody writes them down. Corrections that came
> from actually playing the thing (wall kicks, T-spins, soft drop speed, the
> i-dependency doom loop) are the clearest evidence of where the generated code
> was wrong and how it was found.

A from-scratch Guideline Tetris engine and a reinforcement-learning agent that
learns to clear lines on CPU, plus a hand-tuned evaluator strong enough to be a
useful yardstick and an optional teacher, a beam-search bot that plays TETR.IO
rules, and a playable local versus mode.

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
python -m tetrisrl battle               # play against it, TETR.IO versus rules
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
| Learned DQN**†** (`best_rewardfix.pt`) | 40 | 66.9 | 63.0 | 158 | 47,360 | 212 | 40/40 (100%) |
| Learned DQN**†** (`best_finetuned.pt`) | 40 | 58.9 | 45.0 | 270 | 44,225 | 192 | 40/40 (100%) |
| Learned DQN**†** (`best.pt`) | 60 | 42.6 | 30.0 | 168 | 26,536 | 140 | 56/60 (93%) |
| Learned DQN**†** (`latest.pt`) | 60 | 26.7 | 25.0 | 79 | 8,142 | 113 | 60/60 (100%) |
| Heuristic `heuristic` (1-ply) | 5 | 1,898.8 | 957.0 | 3,847 | 36,679,800 | 4,777 | 5/5 (100%) |
| Heuristic `heuristic_tetris` | 6 | 3,487.8 | 2,704.0 | 6,072 | 116,760,833 | 8,748 | 6/6 (100%) |
| **Search `search_tetris`** (3-ply, beam 6) | 6 | 505.2 | 389.0 | 996 | 4,628,683 | 1,300 | 6/6 (100%) |
| Random legal placement | — | 0.0 | 0.0 | 0 | 0 | ~24 | 0% |

**†** The learned rows were measured before the SRS+ rotation fix (section 17). That
fix reordered the rotation states, which *could* perturb a DQN, since its action
index maps to a placement — but it has not been measured properly, and the kick
tables are not on the DQN's path at all: the agent teleports to `(rotation, x)` and
hard-drops, never calling `rotate`. Two small samples say it still plays normally
(63.5 lines/game over 4 games and 70.5 over 2, both at seed 7, clearing lines in
every game), which is why an earlier footnote claiming these checkpoints are stale
and "die in all six games" was withdrawn. A 40-game fixed-seed comparison is what
would settle it; until then, treat these rows as pre-fix. The heuristic and search
rows *are* re-measured, on the same commands, and they are small-`n` samples whose
means are dominated by one or two long games (the plain heuristic's median is half
its mean).

`search_tetris` is the newest policy and the one built specifically for the tetris
goal: **57.0% of its clear events are tetrises**, against 3.4% for
`heuristic_tetris` and 0.7% for the default heuristic (all three re-measured after
the SRS+ rotation fix, section 17). It clears far fewer lines than the plain
heuristic — that is the trade it was asked to make — and about eight times as many
tetrises per piece. See "the search that finally built tetrises" below.

**The learned rows above predate the SRS+ rotation fix, so treat them as unverified.**
That fix changed which cells rotation states 0 and 2 occupy (before it, state 0 was
the vertical mirror of the real one — the T spawned nub-down and the S and Z pieces
were drawn the wrong way round). A heuristic or a search re-scores every board from
scratch, so it is unaffected; a *learned* Q-function is keyed to the
action→placement mapping it was trained on, and that mapping changed under it. The
checkpoints still load and play. An earlier revision of this note claimed they now
average 64 lines and die in every game; two independent measurements contradict that
(63.5 lines/game over four games at seed 7 and 70.5 over two, clearing lines in
every game), and the kick tables cannot affect this agent anyway, because it
teleports to `(rotation, x)` and hard-drops without ever rotating. The honest
statement is therefore narrower: the reorder *should* perturb a DQN, it has not been
measured properly, and a 40-game fixed-seed comparison is what would settle it.

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

`best_rewardfix.pt` was the champion of that engine at **66.9 lines/game, 100% of
games clearing at least one line, 212 pieces survived on average** — a title it
holds only in the record now, for the reason in the † note above.

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

**Engine** (`engine.py`) — a correct Guideline implementation: **SRS+** rotation —
TETR.IO's own preset, i.e. SRS with the 180° turns folded into the same table — with
the cell matrices and all four kick tables copied from the game's reverse-engineered
engine rather than written by hand (section 17), 7-bag randomiser,
hold, lock delay, block-out and lock-out, Guideline scoring (100/300/500/800 ×
level, back-to-back and combo bonuses), and **T-spin detection** by the guideline
rule: last move a rotation, three of the four corners of the piece's box filled,
mini unless the two corners the nub points at are filled or the rotation only fitted
after a far kick. No dependencies, no RL, fully testable in isolation.

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
`hard_drop()`. The agent never rotates, moves, or wall-kicks, so SRS+ kicks are
never exercised by the agent — they matter for human play, for the versus mode's
T-spin detection, and for the renderer's animation, though they are fully
implemented and tested.
`tools/check_reachability.py` measures the gap: **4.5% of enumerated placements
are unreachable** by real play, while roughly 110 reachable placements per board
are forbidden by the abstraction.

**The policies do not use T-spins, but the engine does.** The engine detects them
and the versus layer pays them (a T-spin double is worth four lines, a triple six —
see the versus section). What is still missing is *on the agent's side*: there is no
T-spin feature, no heuristic term, and no T-spin bonus in Guideline scoring. And
because a move is a spawn-anchor placement plus `hard_drop()`, a T-spin slot that can
only be reached with a wall kick is outside the action space altogether. That is a
genuine capability gap in the learned agent, and the largest single thing missing
relative to real Guideline play.

## Every file, and what it is for

`tetrisrl/tetrisrl/` is the package (inner), `tetrisrl/` is the project root
(outer, containing this README).

### Package — `tetrisrl/`

| File | Lines | Purpose |
|---|---|---|
| `engine.py` | 895 | The Guideline Tetris engine and nothing else. `Piece`, `Bag` (7-bag randomiser), `Board`, `Game`. Owns the **SRS+** cell tables and all four kick tables (`KICKS`, `IKICKS`, `KICKS_180`, `IKICKS_180` — the 180° rows are part of TETR.IO's preset, not a separate extension), copied verbatim from Triangle.js **in that file's own y-down convention, with no conversion**, plus T-spin detection (`Game.spin_kind`, `last_kick_index`, `last_kick_offset`), `spawn_anchor`, `Game.spawn_blocked`, the fast `valid_actions` enumerator, `simulate_placement`, and kick-aware `try_rotate`/`rotate` for real play. Versus additions: `Board.add_garbage` / `Game.add_garbage` (raise the stack by a batch of rows, with the two top-out rules) and `Game.nolockout` for TETR.IO's clutch rule. Pure Python, no torch. |
| `features.py` | 438 | Everything that turns a board into numbers. `board_metrics` returns the 9-tuple `(heights, holes, bumpiness, aggregate, row_trans, col_trans, wells, max_height, rows_with_holes)`, with `metrics_for` as its memoised entry point (`clear_cache`, `cache_size`). Also `placement_vector` (the scored input), `board_vector_game`, `n_features`/`board_width`/`fingerprint`/`check_fingerprint` for the four feature layouts, `holes_of`, and `tetris_readiness`. Knows nothing about learning. |
| `model.py` | 68 | `QNet`, the plain MLP, and `N_ACTIONS = 40`. Deliberately tiny: small (0.1) initialisation so the untrained network does not emit large spreads. |
| `agent.py` | 656 | The learner. Double DQN with target network, masked action selection, the reward calculation (`placement_reward`, `placement_stats`, `reward_config`, `REWARD_PRESETS`), the `Agent` class (`learn`/`learn_batch`/`imitate`/`anneal_lr`/`save`/`load`), the `action_id`/`action_from_id`/`valid_mask` action encoding, and `_guess_version` for width-based feature-version inference. |
| `replay.py` | 207 | Two buffers behind one interface: `Replay` (uniform, with rare-event oversampling) and `PrioritizedReplay`. Both yield `(example, weight, index)` triples so the update code is shared. `make_replay` picks by config. |
| `train.py` | 493 | Parallel self-play and the training loop. `play_episode` (one worker's episode, with epsilon/teacher/exploration), `_to_targets` (the n-step return with a masked bootstrap — the bug-prone part, and validated), `_worker`, `_score_all`, `_candidate_vector`, `evaluate`, `pretrain_imitation`, and `train`. Owns `MAX_PIECES_PER_EPISODE = 3000` and `_stamp_best`. |
| `evaluate.py` | 70 | Measurement, deliberately separate from training. `run_episode` plays a full game through `apply_move` using real outcomes only; `evaluate_policy(..., max_pieces=20000)` aggregates; `format_summary` prints the line you see in every result above. |
| `heuristic.py` | 615 | The hand-written evaluator, and the optional teacher. `DEFAULT_WEIGHTS`, `TETRIS_AWARE_WEIGHTS`, `WEIGHT_PRESETS`, `score_placement`/`score_action`, `clone_game`, `covered_cells`, `best_well`, `hold_options`, `apply_move`, `heuristic_choice(..., allow_hold=False)`, `lookahead_choice`, `teacher_choice`, `make_policy`, and `effective_weights`/`resolve_weights` (which is where the `panic_height` override lives). |
| `policies.py` | 291 | Policy factories so callers never touch torch directly. `load_policy` by name with lazy imports and safe fallbacks, `make_dqn_policy` (feature-version aware), `make_heuristic_policy`, `make_search_policy`, `make_versus_policy` (the handicapped opponent you play), `HANDICAP_PRESETS`, the `CHECKPOINT_NAMES`/`PREFERRED_ORDER` table, and `list_policies` for the CLI. |
| `search.py` | 920 | The beam search, and the reason `search_tetris` exists. A compact column-bitmask board model (`board_masks`, `placements`, `apply_placement`, `locks_out`, `spawn_blocked`), two well definitions (`ready_depth`, `well_depth`), the evaluation (`evaluate`, `TETRIS_SEARCH_WEIGHTS`, `DEFAULT_SEARCH_WEIGHTS`, `COLD_CLEAR_WEIGHTS`, `VERSUS_SEARCH_WEIGHTS`, `covered_cells`, `clear_value`), and `search_move` — a depth-3, beam-6 search over placements *and the hold swap*, ranked by path rewards plus a static board score, with Cold Clear's survival gate when garbage is incoming. Pure Python, no torch, no engine mutation — its geometry is derived from `engine.SHAPES` and its differential tests against the engine are the guard that the two never disagree. |
| `versus.py` | 419 | The TETR.IO versus rules and nothing else: the attack table **including the spin and mini-spin rows**, the combo multiplier with its log fallback, B2B charging and Surge (`streak`, i.e. 4+1+4 for the fourth tetris in a row), `GarbageQueue` (oldest-first cancelling, a 20-frame travel delay, a cap of 8, and the tank-only-on-a-dry-lock rule), `MESSINESS` presets and hole generation, `Side`/`Battle` for the match itself (`after_lock` reports the spin it paid), and `play_out` for headless matches. Numbers and their sources are in `docs/tetrio-versus-ruleset.md`. |
| `render.py` | 2098 | The whole pygame front end: `Layout`, `Renderer`, `HumanInput` (DAS/ARR/SDF, soft drop, lock delay, all driven by `controls.py`), `AgentPlayer` (animates a policy's chosen move before dropping), the `run_human`/`run_agent`/`run_menu`/`run_battle` entry points, and the settings screen. Also the only place that draws a ghost piece, NEXT ×5, HOLD, score/level, and the stack histogram. `run_battle` is the two-board versus screen: human left, bot right, with a garbage meter per side (bright = ready to enter, dim = still travelling), a score line, K.O./round overlays, and `P` pause / `R` restart / `N` new seed / `ESC` quit. Every input handler takes an optional `recorder=` (default `None`), which is where the session hooks live. |
| `controls.py` | 522 | Everything about player input: `ACTIONS` (12 bindable actions, **two keys each**), `DEFAULT_BINDINGS`, `DEFAULT_TIMING`, `parse_key`/`key_name` (with aliases like `lctrl`, `esc`, ` `), and `InputSettings` — which converts TETR.IO's units (ms, and SDF as a gravity multiplier) into the seconds the renderer clocks use, resolves a key code to an action, rebinds without leaving a key doing two jobs, and reports problems. Soft drop has three spellings of instant (`soft_drop_instant`, `sdf: 0`, `sdf:` empty) and a large SDF is deliberately *not* floored at a frame, so 200x means 200x. `load` merges defaults → `config.yaml`'s `input:` → `settings.yaml` and never raises; `save` writes only `settings.yaml`, so the commented config is never rewritten. |
| `recorder.py` | 479 | Session recording: a `SessionRecorder` writing newline-delimited JSON, plus `load`/`iter_events`, `rotation_report` (which mirror's the engine's `can_rotate` to explain a refusal), `state_if_due` for periodic snapshots, and `start_meta`. Bounded by `MAX_EVENTS`, one event per line so a crashed session is still readable, and every write guarded so a full disk disables recording instead of ending the game. Off unless asked for. |
| `__main__.py` | 305 | The CLI: `train`, `eval`, `watch`, `play`, `battle`, `menu`, `policies`. `battle` takes `--difficulty --messiness --rounds --seed`. `play` and `battle` also take `--record [PATH]` and `--no-record`. Training flags include `--rounds --workers --episodes-per-worker --resume --tag --publish --reward-preset --epsilon-decay-steps --no-lr-decay --lr-decay-fraction --quiet`. |
| `__init__.py` | 90 | `load_config`, with a complete set of defaults so every key is defined even if `config.yaml` omits it, and the package version. |

### Project root

| File | Lines | Purpose |
|---|---|---|
| `README.md` | this file | Results, architecture, the full file inventory, the chronology of what was tried, and the problems that were hit. |
| `config.yaml` | 221 | Every hyperparameter and reward shaping knob in one place: net width, buffer, n-step, epsilon schedule, `feature_version`, `replay` type, target sync, gradient budget, reward preset, panic height — plus an `input:` section documenting the two key slots per action, DAS/ARR/SDF (including `soft_drop_instant`) and lock delay. Current defaults: `feature_version: 1`, `replay: uniform`, `epsilon_decay_steps: 250000`, `lr_decay_fraction: 0.0`, `grad_steps_auto: true`, `grad_steps_per_sample: 0.25`, `imitation_steps: 0`. |
| `settings.yaml` | — | Written by the in-game settings screen; overrides `config.yaml`'s `input:` section. Not committed, so your bindings survive a config edit or a `git pull`. |
| `checkpoints/` | — | See the checkpoint table below. |
| `docs/` | — | `tetrio-versus-ruleset.md` — every versus number this project implements, with its source (wiki, FAQ, and the Triangle.js engine reverse-engineering), the **SRS+ rotation tables and the T-spin rule**, and a priority-ordered implementation checklist. `tetrio-bot-account.md` — what TETR.IO's rules actually permit, with the application steps for a bot account. |
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
| `analyze_session.py` | 593 | Reads one or more recorded sessions and diagnoses them. The headline is **refused rotations** — count, piece, position, direction, reason, and the offsets that were tried — followed by observed DAS/ARR (measured from press to first repeat, which is the only way to check a setting the config cannot check for itself), and suspicious patterns: keys that resolved to nothing, keys bound twice, quiet stretches, and pieces that locked with no input at all. `--json` for machine-readable output, `--top N` for how many examples to show. |
| `dependency_audit.py` | 208 | Audits a board (or a recorded session) for I-dependencies: columns with three or more consecutive one-wide gap rows flanked by filled cells, which only an I can fix. This is the instrument behind the doom-loop work -- it reports where they open, how long they stay open and how many the board is carrying at once. |
| `versus_match.py` | 135 | Headless versus matches: plays the handicapped bot against the heuristic, a passive board or itself, on a clock so the pieces-per-second handicap is real. `--difficulty --opponent --games --rounds --messiness --foe-pps --verbose`. This is where the difficulty table in the versus section comes from. |

The two most useful of these:

```
python tools/clear_mix.py --policy search_tetris --games 6 --cap 2500
python tools/versus_match.py --difficulty expert --games 6
```

### Tests — `tests/`

Run all nine directly; each is self-contained and prints one line per check —
**812 checks total, all passing.**

| File | Lines | Checks | Covers |
|---|---|---|---|
| `test_engine.py` | 986 | 255 | Rotation tables self-consistent under 90° clockwise rotation, the **SRS+ box convention** (every state in its own box), a 336-attempt sweep that every rotation of every piece succeeds at both walls (in the air *and* flush on the floor), the exact published kick row and index used where a kick is forced (a T's floor kick is the third test of `0 → R`, `(-1, -1)`; a Z tucked into a well is the third test, `(-1, -1)`; a flat I stands up on the fifth, `(1, -2)`), the y-down sign convention of the kick rows pinned against the fetched source, spin classification (full/mini/far-kick promotion), real T-spin double and **T-spin triple** seats played end to end, 7-bag properties, scoring and level scaling, lock-out and block-out, hold, and — most importantly — that the fast placement enumeration and `simulate_placement` both agree exactly with actually playing the moves. |
| `test_agent.py` | 420 | 47 | Feature layout (every slot written and finite, per version), reward shaping, action masking, masked bootstrap targets, checkpoint round-trip, and the renderer's choice-coercion path. |
| `test_train.py` | 329 | 46 | n-step return maths, replay sampling and weights, that learning actually reduces error on a toy problem, and that each feature version's vectors are accepted by that version's network (the feature-drift guard). |
| `test_search.py` | 497 | 40 | The search's bitmask model against the engine — `placements` versus `valid_actions`, `apply_placement` versus `simulate_placement`, the landing row versus the ghost row, spawn and lock-out tests — plus the well rules, move legality, determinism, that an available tetris is taken, that the hold slot is actually spent, and that the tetris weights beat the plain-search control on clear mix. |
| `test_versus.py` | 382 | 87 | The TETR.IO numbers themselves — attack table including the **T-spin and mini-spin rows**, the combo multiplier and its log fallback, B2B charging and Surge chunks — plus a real T-spin double and triple played through the battle layer and paid 4 and 6 lines, queue travel and oldest-first cancelling, the entry cap, clean/messy/chaotic hole generation, garbage raise and both top-out rules, clutch/no-lockout, tank-only-on-a-dry-lock, round resets, and a full match where garbage crosses the board. |
| `test_battle.py` | 348 | 54 | The battle screen's logic, headlessly (no display, ever): garbage queued → travelling → tanked with the timing band, a quad landing on the opponent's queue, K.O. and round reset, the pieces-per-second handicap measured at two rates, the real handicapped policy playing to a match result from a board that has already tanked garbage, and the two-board layout geometry. |
| `test_controls.py` | 845 | 171 | Input: key-name parsing including aliases and bad names, both binding slots, rebinding a key away from what held it, config round-trips, negative and non-numeric timings degrading to a warning instead of a crash, SDF as a multiplier versus instant (all three spellings), a high SDF moving several rows in one frame where a low one still moves at most one, instant teleporting to the ghost row without locking or placing, DCD freezing the repeat after a rotation and after a spawn while leaving a fresh press, soft drop, gravity and the lock timer alone, the settings screen reaching instant in one press, `settings.yaml` overriding the config, and `HumanInput` driven headlessly with a real game — a bound key moves the piece, DAS is respected, ARR 0 slides to the wall, 180 rotates, and a doubled binding behaves like its partner. |
| `test_recorder.py` | 600 | 91 | Recording: valid JSONL that round-trips through `load`/`iter_events`, a truncated last line skipped rather than raised, the event cap stopping cleanly, a failing write disabling recording instead of killing the game, `rotation_report` agreeing with `Game.can_rotate` on every attempt, **a refused rotation recorded with its piece, position, reason and the offsets that were tried**, every event kind appearing when driven through `HumanInput` and one battle frame, periodic snapshots, and the analyser producing its summary, refusals, latency and suspicious-pattern sections from a session file. |
| `test_dependency.py` | 247 | 21 | The I-dependency term's definition and edges: a four-deep one-cell shaft is counted and a deeper one counts more rows; two rows deep, one flat hole, a two-wide gap, an open well and a gap whose neighbours are lower are all *not* counted; **a covered one-wide shaft in a wall column is counted**, where the board edge acts as the second wall; the solo preset leaves the term at zero while `versus` pays it; a second dependency costs more than twice the first; and paying a debt scores better than stacking elsewhere. |

```
python tests/test_engine.py     # 255 checks
python tests/test_agent.py      #  47 checks
python tests/test_train.py      #  46 checks
python tests/test_search.py     #  40 checks
python tests/test_versus.py     #  87 checks
python tests/test_battle.py     #  54 checks
python tests/test_controls.py   # 171 checks
python tests/test_recorder.py   #  91 checks
python tests/test_dependency.py #  21 checks
```

## The process — what was tried, in order

### 1. Build the engine, and get the tables right

The engine came first, because every wrong number downstream would be
uninterpretable otherwise. The subtlest failure here was the rotation tables, and it
took **three** attempts to get them right — each one producing rotations that
"mostly worked", which is exactly why it survived so long:

1. The hand-written SRS tables were **reflections**, not rotations, so rotation
   state 2 was wrong for the I, S and Z pieces.
2. Replacing them with *generated* tables (rotate the spawn state 90° three times
   about the origin) fixed the chirality but broke the **box convention**: the
   states no longer sat at the same offset inside the piece's bounding box, and the
   published kick offsets describe where that box moves, so kicks pushed pieces to
   the wrong place.
3. Copying TETR.IO's own matrices from [Triangle.js][triangle] fixed the boxes, but
   the cells were then run through a y-negation meant for a y-up source. Triangle's
   matrices are already y-**down**, so every piece came out **mirrored vertically**:
   the T spawned nub-down instead of nub-up, clockwise and counter-clockwise were
   swapped, the I's kick rows were paired with the opposite turn, and a floor kick
   lifted the piece the wrong way. The S and Z pieces were drawn as each other's
   shape in state 0. Nothing crashed, and the wall test passed, because a mirrored
   table is still internally consistent.

   | Piece | state 0 before | state 0 now |
   |---|---|---|
   | T | `XXX` / `.X.` | `.X.` / `XXX` |
   | J | `XXX` / `X..` | `X..` / `XXX` |
   | L | `XXX` / `..X` | `..X` / `XXX` |
   | S | `XX.` / `.XX` (a Z) | `.XX` / `XX.` |
   | Z | `.XX` / `XX.` (an S) | `XX.` / `.XX` |

The fix is to take the source at face value: matrices copied cell for cell, stored
y-down with offsets measured from the **top-left corner of the piece's box**, and
state 0 = spawn, 1 = clockwise, 2 = 180, 3 = counter-clockwise so that the published
table keys (`"01"`, `"10"`, …) apply verbatim. Three tests now pin it: each state
must be the previous state turned a quarter clockwise, all 336 rotations (every
piece, every state, three directions, both walls) must succeed on an empty board,
and the offsets actually used must equal the published row — for example the T's
floor kick must be the fourth test of `0 → R`, `(0, +2)`.

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
`tools/clear_mix.py`, and re-measured after the SRS+ rotation fix (section 17), which
moved `search_tetris` up and left the heuristics within a point:

| Policy | Singles | Doubles | Triples | **Tetrises** |
|---|---|---|---|---|
| `heuristic` | 80.9% | 15.7% | 2.7% | 0.7% |
| `heuristic2` (2-ply) | 80.9% | 16.9% | 1.9% | 0.3% |
| `best_rewardfix` (DQN) | 40.8% | 42.3% | 11.7% | 5.2% |
| `heuristic_tetris` | 25.4% | 61.1% | 10.1% | 3.4% |
| **`search_tetris`** (3-ply beam + hold) | 16.0% | 14.8% | 12.2% | **57.0%** |

The DQN row is one 205-piece game, from a checkpoint trained before the
rotation-state reorder, so its action→placement mapping is the old one and its clear
mix is not comparable with the rest of the column. It is not that it cannot play —
two small samples had it clearing lines in every game — but it has not been
re-measured properly since (see the † note above). The two heuristic rows and
`search_tetris` score each position fresh and are unaffected.

The last row is one tetris per 1.8 clear events, measured over the same protocol
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
lines (505/game against 996 for `heuristic_tetris` and ~880 for the plain
heuristic, re-measured after the SRS+ fix); and more deaths — 4 of 6 games topped
out inside the 2,500-piece cap, against 0 of 6 for `heuristic_tetris`. It is a
tetris machine, not a survival machine, which is the trade it was asked to make.
(The A/B deltas quoted above are from the earlier engine; the clear mix, lines and
death counts in this paragraph are the current ones.)

### 17. SRS+ rotation, wall kicks and T-spins

The versus mode made rotation correctness matter for the first time: a human plays
against the bot, and a human notices when a T-spin triple will not go in. Three
rounds of "rotations kind of work" later, the engine now implements **SRS+ exactly
as TETR.IO does**, and the versus layer pays spins.

**What SRS+ is.** SRS is the Guideline rotation system: four states per piece, with
a five-position wall-kick test for each of the eight possible turns. SRS+ is
TETR.IO's default preset, and it is SRS *plus a 180° turn*, folded into the same
tables rather than offered as a separate extension. So the quarter-turn offsets are
ordinary SRS (they match the [Hard Drop SRS tables][harddrop] test for test), and
the 180° rows are TETR.IO's addition. The I piece is the interesting one: where SRS
tries `(-2, 0)` first after the in-place test, SRS+ tries `(+1, 0)`, which changes
which way an I lands against a wall.

**Sources, not memory.** The cell matrices and all four kick tables (`KICKS`,
`IKICKS`, `KICKS_180`, `IKICKS_180`) are copied from [Triangle.js][triangle]
(`src/engine/utils/tetromino/data.ts` and `src/engine/utils/kicks/data.ts`, entry
`kicks["SRS+"]`), the replay-accurate community reverse-engineering of TETR.IO, and
cross-checked against the [SRS page][harddrop] for the quarter turns. Triangle's
matrices are already stored with y growing **down**, which is this board's
convention, so they transfer with no flip at all.

**The convention that makes kicks mean anything.** Offsets are measured from the
**top-left corner of the piece's box** (3×3 for J/L/S/T/Z, 4×4 for I, 2×2 for O),
and every state of a piece sits in that same box. Rotation index 0 is the spawn
state, 1 is clockwise, 2 is the 180 and 3 is counter-clockwise, so the published
table keys `"01"`, `"10"`, — apply verbatim. Get any one of those three things
wrong — box, chirality or state order — and the table is still self-consistent, so
the failure mode is not a crash but rotations that land a row or two off.

**T-spin detection** follows the guideline rule, which is what TETR.IO's league
preset uses: the last successful movement must have been a rotation, and at least
three of the four corners of the piece's box must be filled (walls and floor count
as filled, the space above the board does not). It is a **mini** when the two corners
the nub points at are not both filled, unless the rotation only fitted after the
fifth kick test, which promotes it to a full spin. Only the T is classified, again
matching the preset (`spinbonusRules["T-spins"]` in the same file). The engine
reports it as `Game.spin_kind` (`'full'`/`'mini'`/`None`) after every lock, and
`Game.last_kick_index` / `last_kick_offset` record which table row was used.

**What that pays**, through `versus.py` (`SPIN_ATTACK`/`MINI_ATTACK`, from
Triangle's damage calculation):

| Clear | Ordinary | Full spin | Mini |
|---|---|---|---|
| single | 0 | 2 | 0 |
| double | 1 | **4** | 1 |
| triple | 2 | **6** | 2 |
| quad | 4 | **10** | — |
| no clear | 0 | 0 | 0 |

A spin is also a *difficult* clear for the B2B chain, so it continues the streak and
is paid the +1 charging bonus: a T-spin double on a live streak sends 5.

**The demonstration.** `tests/test_engine.py` builds this seat by hand — rows 18 and
up solid except for a 2-wide mouth, a 3-wide notch, a 1-wide gap and the cell under
it — drops a T in standing (state 3, nub left) and turns it:

```
row 18: ###..#####      the T falls into the notch and rests with its
row 19: ###...####      bar down the (4, 18..20) column: a spin position,
row 20: ####.#####      three of its four box corners already filled
row 21: ##########
```

* a **180°** turn puts the nub in the right-hand mouth → full T-spin, **2 rows**
  (4 lines through the versus layer);
* a **counter-clockwise** turn lays it flat across row 19 → full T-spin,
  **3 rows** — a **T-spin triple**, worth **6 lines** (7 on a B2B streak, 7 at a
  1-combo).

Both are asserted end to end, engine and attack table, and the board is re-checked
after the clear. The kick table itself is pinned by asserting the *row* used: a T
resting on the floor turns clockwise on the third test of `0 → R`, which the
published table gives as `(-1, -1)` — left one, up one. The two earlier tests (turn
in place, then left one) both leave the bar a row below the floor.

**The floor, and the sign bug that hid behind it.** SRS+ lifts a piece when that is
what a rotation needs, so a flat I on the very bottom row *can* stand up — via the
fifth published test, `(1, -2)`: right one, up two. An earlier revision of this file
claimed the opposite and called it "the real table's behaviour", and that claim is
worth leaving a mark on: it was the mirrored kick table talking. The tables had been
copied from a y-down source and then negated in y as if the source were y-up, so
every offset with a vertical component pushed the piece the wrong way. A mirrored
table is internally consistent, which is why it survived a 336-attempt wall sweep,
the spawn tests and every rotation-count check — while breaking exactly the kicks
players notice: the tuck that slides an S or Z into a slot, the drop that lands a
T-spin triple, and the lift that finishes a rotation on the floor.

Measured after the fix: 168 rotations in the air and 168 flush on the floor, all
336 accepted; and the *first* offset that fits in the published row is the one used
in every case, which is what SRS+ specifies. `tests/test_engine.py` asserts both,
including a Z and an S spun into a well through their published kicks.

[triangle]: https://github.com/halp1/triangle
[harddrop]: https://harddrop.com/wiki/SRS

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
  clears 57% tetrises and far fewer total lines, and it tops out more often than
  any other policy here. Use it when tetrises are the goal.
- **Hold is the strongest single lever in the search** — without the swap in its
  candidate list the same search drops from 57% to 19% tetrises.
- **Small samples lie.** Game-to-game variance is huge; use ≥40 games and a fixed
  `--seed` before believing any comparison, including the ones in this file.
- **Loss is not strength.** They disagreed in both directions here.
- **The learned agent does not use T-spins.** The engine detects them and versus pays
  them (double 4, triple 6), but there is no T-spin feature, heuristic term or
  Guideline spin bonus in the agent's world, and a slot that needs a wall kick is
  outside its spawn-anchor action space.
- **Versus is local only.** TETR.IO has no public bot API; a bot account is
  application-only and custom-room-only (`docs/tetrio-bot-account.md`).
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

## Controls and settings

Every input is configurable, with **two keys per action**: move left/right, soft
drop, hard drop, rotate right/left/180, hold, pause, restart, new seed and quit.

* Edit the commented `input:` section of `config.yaml`, or
* run `python -m tetrisrl menu` → **Settings** to rebind in-game (ENTER captures a
  key, `DEL` clears a slot, `S` saves) and nudge the timings with the arrow keys.

```
input:
  das_ms: 133          # delay before auto-repeat
  arr_ms: 0            # auto-repeat rate; 0 = slide to the wall instantly
  dcd_ms: 0            # DAS Cut Delay: freeze auto-repeat after a spin/spawn
  sdf: 41              # soft drop factor: a multiple of gravity
  soft_drop_instant: false   # true = teleport down instead of dropping fast
  lock_delay_ms: 500
  bindings:
    move_left: [left]
    rotate_cw: [up, x]      # two keys, either one works
```

### DCD — DAS Cut Delay

`dcd_ms` freezes **horizontal auto-repeat** for that many milliseconds after a
piece is **rotated** and after a **new piece appears**. It is what stops a spin
from also flinging the piece sideways, and what stops a held direction from
carrying a fresh piece off its spawn column before you can aim it.

Three details are the whole feature, and each has a test:

* it cuts the **repeat**, not the input — a *fresh press* during the window still
  moves one cell, so a tap never feels eaten;
* DAS is **paused, not reset** — the charge keeps its progress through the window,
  so a key that was most of the way to firing fires as soon as it closes, rather
  than starting its delay over;
* it touches nothing else — soft drop, gravity and the lock delay all keep
  running while the repeat is frozen.

`0` (the default) disables it and behaves exactly as before. The minimum is 0 ms
and there is **no maximum**: it is a delay, and any value you type is accepted.
The settings screen has a DCD row beside DAS/ARR/SDF/lock delay, stepping 5 ms at
a time (1 ms with shift).

### Soft drop: fast, faster, instant

`soft_drop_instant: true` makes soft drop **teleport the piece to its landing row
without locking it**, so it can still be slid or spun during the lock delay. That
is a different thing from a big SDF, and both are available:

* a large SDF is genuinely fast now — 200x moves about four rows per frame at
  60 fps, where it used to move one. The old code floored the interval at 1/60 s,
  which made every SDF above ~51 identical: 200x *was* 60x, and no setting could
  beat a row per frame;
* `soft_drop_instant` (or `sdf: 0`, or an empty `sdf:`) is the true teleport.

In the settings screen, **`I` on the SDF row toggles instant in one press** — the
row then reads `instant` instead of a number, and pressing `I` again restores the
factor you had. Reaching instant by nudging used to take 41 presses from the
default and nothing on the screen said it existed.

The settings screen writes `settings.yaml` (**not** `config.yaml`, which is heavily
commented and would be stripped by a YAML dump), and that file overrides the config,
so your bindings survive both a config edit and a `git pull`. It is git-ignored, so
your keybinds stay yours. Both DAS/ARR and SDF use TETR.IO's meanings: milliseconds
for the timings, and SDF as a multiplier of the gravity interval where 0 means
unlimited.

A bad key name or a negative timing never stops the game: `controls.load` collects
the problem, prints it at startup, shows it on the settings screen, and falls back
to the documented default for just that value. `tests/test_controls.py` covers that
path, and it is where four real bugs in the first version of `controls.py` were
caught — a crash on negative timings, a `ValueError` escaping `load()`, an
unreachable alias and a duplicate-slot rebind.

*Not verified:* the settings screen's appearance and real key capture — no window was
opened during development, so the code paths are tested but nothing has been seen
rendering on a real display. On a small `cell_size` the 16-row list may overflow.

## Recording a session

A bug report arrived as one sentence — *"I can't rotate when there's a wall"* — and
diagnosing it meant inventing a synthetic test, because by the time anyone looked,
the keys that arrived, the piece's position and the kick offsets that were tried
were all gone. So sessions can be recorded:

```powershell
python -m tetrisrl play   --record          # sessions/<UTC timestamp>.jsonl
python -m tetrisrl battle --record my.jsonl
python -m tetrisrl play   --no-record       # force it off
```

Recording is **off unless asked for**, `sessions/` is git-ignored, and nothing about
it can take the game down: every write is guarded, and a failure disables recording
and warns once. Each line is one JSON object with `t` (seconds), `frame` and `kind`,
so a session that ends in a crash is still readable up to the last complete line.
A session stops at 20,000 events and says so in a final `stopped` event.

The events are `start` (the effective settings, bindings, seed, version — what
makes a report reproducible), `key` (every dispatched key **including ones bound to
nothing**, because "my key does nothing" is a bug report too), `move` (flagged
`auto` when it came from DAS/ARR), `rotate` (**every attempt**, refused or not),
`lock`, `attack` in battle, `state` every five seconds, and `stopped`.

Then:

```powershell
python tools/analyze_session.py                 # the newest session
python tools/analyze_session.py --json --top 5
```

On a short recorded session it prints:

```
session sessions/20260916T021840Z.jsonl
  mode play  seed 42  version 1.0.0  stopped: closed
  1121 events over 13s  (86 pieces, 6.41 PPS)
  lines 14  score 0  tetrises 0 (0.0% of clears)  mix {1: 3, 2: 4, 3: 1}
  settings: DAS 133ms  ARR 30ms  SDF 41.0  lock 500ms

refused rotations
  2 of 92 attempts (2.2%)
  by piece     {'I': 2}
  by direction {'cw': 1, 'ccw': 1}
  by reason    {'all offsets collide': 2}
  by context   {'wall': 2}
  worst cells  x4y6 (2)
    t=9.4 I 1->2 at (4,6) all offsets collide  left=True right=True floor=True
        offsets tried: [0, 0]x, [-1, 0]x, [1, 0]x, [2, 0]x, [-2, 0]x, ... (49 tried)

key latency (observed, from the recording)
  move_left   DAS median 133ms (min 133, max 134, n=6)   ARR median 33ms

suspicious
  keys that resolved to nothing: none
```

The refusals section is the point: it names the piece, the cell, the direction, the
context (walled in versus resting on the floor) and every offset that failed. The
latency section is the second: the config says what DAS *should* be, and only a
recording says whether the game did it — above, 133 ms measured against 133 ms
configured, and ARR 33 ms against 30 ms configured, which is one 60 fps frame of
quantisation.

## Versus mode — playing against it

`python -m tetrisrl battle` puts you on the left board and the bot on the right,
with TETR.IO's multiplayer rules underneath: the attack table **including its spin
rows**, the combo multiplier, B2B charging and Surge, oldest-first cancelling, a
20-frame garbage queue, a cap of 8 lines per entry, and garbage that only lands on a
lock which cleared nothing. `docs/tetrio-versus-ruleset.md` records every number and
where it came from; `tests/test_versus.py` pins them.

T-spins are live: the engine detects them by the guideline corner rule (section 17),
so a T-spin double sends **4** and a T-spin triple sends **6** rather than the
ordinary double's 1 and the ordinary triple's 2, minis pay 0/1/2, and any spin counts
as a difficult clear for the B2B chain. The default "Multiplier" combo table is
TETR.IO's — `base * (1 + 0.25 * combo)`, with `ln(1 + 1.25 * combo)` for a zero base
from a 2-combo up — and its wiki page is the source for both that and the Surge
arithmetic.

**What the mode does not have.** No All-Spin/All-Mini+ (only the T is classified,
which is what the preset this mirrors does), no back-to-back display, no combo
counter in the HUD (yet), and no targeting beyond 1v1.

### Difficulty

The opponent is handicapped on the three knobs published bots actually expose —
search depth/beam (Zetris's "Intelligence"), a pieces-per-second cap (its "Speed")
and a reaction delay — plus a mistake rate modelled on Stockfish's Skill Level: with
probability *p* the bot plays the one-ply heuristic's move instead of the search's,
which is a real downgrade rather than noise, because the heuristic is the policy
this project measured as unable to build tetrises at all.

Measured with `python tools/versus_match.py --games 6 --opponent heuristic`, where
the opponent is the strong one-ply evaluator playing at 2 pieces/second:

| Preset | Depth/beam | PPS | Mistake | Wins vs heuristic | Bot tanked |
|---|---|---|---|---|---|
| `beginner` | 1/1 | 0.7 | 40% | 0/6 | 161 lines |
| `intermediate` | 1/6 | 0.9 | 25% | 2/6 | 143 |
| `advanced` (default) | 2/6 | 1.1 | 12% | 0/6 | 204 |
| `expert` | 3/6 | 1.4 | 4% | 4/6 | 155 |
| `max` | 3/6 | 6.0 | 0% | **6/6** | 16 |

Six games per row is a small sample and the middle of that table is noisy —
`advanced` losing more than `intermediate` is not a real ordering, it is variance
with a strong opponent. The ends are solid: `beginner` loses every game and `max`
wins every game, and `max` wins them in about 100 pieces because it plays three
times faster than the opponent rather than because it plays better per piece.

### The I-dependency doom loop

A one-cell-wide gap at least three rows deep, with filled columns either side, is a
hole **only a vertical I can fill** — nothing else is one cell wide, and pieces
descend from above so it cannot be reached sideways. Blockfish, the dedicated
downstacking engine, counts exactly that as `i_dependencies` and prices one at
*twice a row of height*. Two at once is the state that loses games: the board is
waiting on two I pieces while the bag supplies one every seven, so the bot keeps
stacking in the middle with holes it cannot pay for underneath.

`search.i_dependencies` implements Blockfish's definition — including the wall
columns, where the board edge is as solid as a filled column and a covered
one-wide shaft is the most common place for one of these to appear — and
`tools/dependency_audit.py` plays the arena and reports per game the peak count,
how long they stayed open, stack height, lines, pieces, tetris share, and whether
the game ended still owing one.

**The measured result** — ten games a side, same seeds, difficulty `max`, three
garbage lines every twenty pieces:

| | peak deps | locks ≥2 deps | lines | pieces | tetris share | ended owing |
|---|---|---|---|---|---|---|
| before | **2.80** (max 4) | 7.4% | 210 | 455 | 31.0% | 2.10 |
| after | **1.80** (max 3) | 2.3% | 209 | 459 | **36.0%** | 1.60 |

Raw per-game peak counts, same seeds `300..309`:
`before [4,2,2,3,2,3,3,3,3,3]` → `after [2,1,2,1,3,2,2,2,1,2]`.

It holds under a firehose too (four lines every ten pieces, ten games, seeds
`200..209`): peak `2.00 → 1.70`, locks holding two `6.6% → 4.8%`, lines `89 → 93`,
tetris share `31.8% → 33.1%`.

So it is a real improvement, it does **not** cost scoring — tetris share went up —
and it is **not a cure**. Every game in every arm still tops out, ten games out of
ten still end with a dependency open, and the mean peak only falls from 2.8 to 1.8:
two at once are roughly a third as often, not gone. An earlier revision of this
section claimed "two dependencies at once are gone" from a six-game sample that
read `1.33 → 1.00` and `2.1% → 0.0%`; on ten games at the same schedule the numbers
are the ones above, so that was a small-sample artifact and the claim is withdrawn.

**Which part does the work** was worth measuring rather than assuming. Four arms,
same seeds `200..205`, four garbage lines every twenty pieces:

| arm | gate | suppress | price | peak deps | locks ≥2 | lines | pieces | tetris |
|---|---|---|---|---|---|---|---|---|
| **shipped** | ✓ | ✓ | ✓ | 1.50 | 0.3% | **195** | **403** | 41.2% |
| gate only | ✓ | ✗ | ✓ | **1.33** | **0.1%** | 121 | 260 | 46.3% |
| suppress only | ✗ | ✓ | ✓ | 2.50 | 5.5% | 183 | 373 | 36.3% |
| price only | ✗ | ✗ | ✓ | 2.67 | 7.2% | 176 | 366 | 36.9% |

* **The gate is the mechanism.** Every arm without it sits at 2.5–2.7 peak
  dependencies; every arm with it sits at 1.3–1.5.
* **The price is decorative.** With the gate on, `i_dependency` at −60, −120 and
  −200 give *byte-identical* ten-game runs, and price-only (2.67) is no better than
  no term at all (2.50). It is kept because it is what makes an open dependency
  dominate at a *low* stack, where dependencies are created — but nothing measured
  here separates it from zero.
* **Suppression is a survival trade, not the engine**, contrary to what an earlier
  revision of this section said. Dropping it gives the fewest dependencies (1.33)
  and the highest tetris share (46.3%) but roughly *half* the lines and pieces
  (121/260 against 195/403), because standing the well terms down is what lets the
  bot keep building. The shipped default keeps it, for the survival.
* **A genuine cure probably needs a different *kind* of term.** Everything above is
  local: it reacts once the board already owes an I. Blockfish — where this
  definition comes from — pairs `i_dependencies` with a **piece estimate**: the
  minimum number of pieces needed to clear the board, built by decomposing the
  residue above each hole, which prices the *whole board's* debt instead of one
  column's shape. That is the term that would stop the debt being created, and it
  has not been implemented here. Until it is, this section describes mitigation:
  the mean peak fell from 2.8 to 1.8, and ten games out of ten still ended owing a
  dependency.

All of it lives in `VERSUS_SEARCH_WEIGHTS`; the solo `tetris` preset leaves the
three at zero, so every solo number in this README still means what it meant.
`tests/test_dependency.py` (21 checks) pins the definition, its edges — flat holes,
two-wide gaps, open wells, wall columns — and that the versus weights actually pay
for a dependency and pay more for a second one.

Sessions record it: every `lock` event carries `deps`, and
`tools/analyze_session.py` prints a **dependency history** — peak, when it peaked,
the share of locks holding one or two, and whether the board ended owing one — so
this can be confirmed from a real game rather than from a synthetic bench.

### Why the bot needed changes for this

On a clean board the search builds tetrises; on a garbage board it has to dig, and
two things were added for that, both off by default so every solo number above
still means what it meant:

* **`covered_cells`** (Cold Clear's dig-priority term) — how many blocks sit on top
  of each hole. A plain hole count cannot tell a reachable hole from one under six
  blocks, and that distinction is the whole game on a garbage board.
* **A survival gate** — when garbage is queued, `search_move` takes the best move
  that can absorb what is about to land, and only when nothing survives does it
  fall back to the best move outright, which is the "take the max-damage move" half
  of Cold Clear's `pick_move`. Height is also priced by the queue still coming.

The gapless well measure does the rest by itself: on a holed board it stops seeing
a well, the well bonus disappears, and the policy downstacks instead. That is the
correct behaviour under garbage rather than a compromise.

### Playing on TETR.IO itself

There is **no public bot API**, and pointing this bot at a normal account is
explicitly bannable. The only sanctioned route is a staff-granted **bot account**
for custom rooms, which you have to apply for by hand. The full picture, with the
rules quoted and the application steps, is in
[`docs/tetrio-bot-account.md`](docs/tetrio-bot-account.md).

## Session log — how all of this actually got built

Everything above this section was written as it was measured. This section is the
history of the work that produced it, including the parts that went wrong, because
the wrong parts are the useful ones. It was built in one long session with an AI
agent doing most of the typing and a human playing the game, reading the results,
and saying "this is still wrong".

The honest summary: **the code that exists was largely correct on arrival, and the
things that were wrong were wrong in ways that passed every test that existed at
the time.** Every significant bug below was found by a human playing, by a
deliberately adversarial test, or by a subagent contradicting another subagent —
never by the suite that was supposed to cover it.

### What was added, in order

1. **A beam search over placements** (`search.py`), because ten rounds of weight
   tuning could not push the one-ply evaluator past 3.9% tetrises and section 10
   explains why: a score that looks at one placement cannot value an I it has not
   spent. Depth 3, beam 6, the hold swap in the candidate list, a column-bitmask
   board model. Measured at **57.0% of clear events being tetrises** against 3.9%
   for `heuristic_tetris` — one tetris per 1.8 clears.
2. **The single biggest lever was hold.** Adding the hold swap to the root
   candidates took the search from 19.2% to 56.9% tetrises on identical seeds.
   Holding is score-neutral one ply deep, which is why section 11 measured it as
   worthless for the greedy evaluator and why only a search can use it.
3. **TETR.IO's versus rules** (`versus.py`): the attack table, the combo
   multiplier with its log fallback, B2B charging, Surge, oldest-first cancelling,
   a 20-frame queue, a cap of 8, tank-only-on-a-dry-lock, and configurable
   clean/messy hole generation. Every number is sourced in
   `docs/tetrio-versus-ruleset.md`.
4. **Garbage-aware terms**: Cold Clear's `covered_cells` (how buried a hole is) and
   a survival gate that refuses moves which cannot absorb the queued garbage.
5. **Difficulty presets** calibrated to published rank APM/PPS, with a mistake rate
   modelled on Stockfish's Skill Level.
6. **A two-board battle screen**, garbage meters, and a settings screen; then
   **configurable controls** with two keys per action and DAS/ARR/SDF in TETR.IO's
   units.
7. **A session recorder and analyser** (`recorder.py`, `tools/analyze_session.py`),
   which is what turned "rotations feel wrong" into "this rotation, at this cell,
   with these offsets tried".
8. **SRS+ and T-spin detection**, then HOLD moved left of the board, then true
   instant soft drop, then Blockfish's **i-dependency** term for the doom loop the
   human kept seeing: the bot would open two holes only an I can fix, stack in the
   middle, and spiral. That last one is described in its own section above; its
   before/after numbers are the one set in this file that this author did not
   re-run personally.

### The bugs, which are the point

**The y-up / y-down mirror.** The shape tables came from Triangle.js, whose
matrices are already y-down, and were then negated as if they were y-up. Every
piece was vertically mirrored: the T spawned nub-down, J/L were flipped, **S and Z
were literally each other's shape in state 0**, clockwise and counter-clockwise
were swapped, and the I-piece kick rows were paired with the opposite turn. **22 of
24 kick rows were wrong.** It passed the wall sweep, the spawn tests, the rotation
state checks and the T-spin suite, because a mirrored table is still internally
consistent. What it broke was every kick with a vertical component — the S/Z tuck,
the T-spin triple drop, the floor lift — which is exactly the set of moves a human
notices and a unit test does not.

**The same mistake one layer up, in the kicks.** `_to_board_kicks` negated `dy` to
"convert the published y-up table to our y-down board". Triangle's kick data is
already y-down. All four kick tables were mirrored vertically. A human reported "I
can't rotate when there's a wall"; the wall sweep then showed **38 of 336 rotations
refused** on an empty board, where SRS guarantees every rotation succeeds.

**`can_rotate` sent counter-clockwise turns to the 180° tables.** Directions were
dispatched as `if direction in (1, 3) ... else <180 table>`, so a plain `-1` looked
up a key that only exists for `(0,2)`, `(1,3)`, `(2,0)`, `(3,1)`, found nothing, fell
back to a bare `(0,0)` offset and gave up. That is the half of "can't rotate at a
wall" that only happened turning one way.

**A patch that treated the symptom.** Faced with wall refusals and not yet knowing
the cause, this author appended mirrored offsets and a distance-ordered fallback
sweep to every kick row. Rotations then succeeded — by moving the piece somewhere
SRS would never put it, which is precisely what makes T-spin geometry
unreproducible. The human's next report ("rotations kind of work but are still bad,
T-spin triples don't work") was the consequence. The patch was reverted once the
real causes were found. *Widening a table until the symptom stops is not a fix.*

**T-spins paid almost nothing, because nothing detected them.** The engine had no
T-spin classification at all, so a T-spin double was scored as an ordinary double:
1 line instead of 4, and a triple as 2 instead of 6. The attack table could not
carry spin rows because there was nothing to key them on.

**Four real defects in the first `controls.py`**, found by the subagent that
inherited it after this author had tested it and called it working: an
`AttributeError` on any negative timing (the warnings list was assigned after the
loop that appended to it, so the documented "clamp and warn" path could never run);
a `ValueError` escaping `load()`, which is documented never to raise, so a typo in
a config value stopped the game before a window opened; an unreachable `' '` alias;
and a rebind that could put one key in both slots of the same action.

**Two soft-drop ceilings.** `soft_interval` floored the interval at `1/60`, so
every SDF above ~51 was identical to every other — 200× and 1000× both meant 60
rows/second, which is why "200× is still too slow" was the only reasonable thing a
player could conclude. Behind it sat a second, hidden cap: `HoldRepeat.due` had a
hardcoded `count < 8`, i.e. 8 rows per frame, which would have bitten at SDF ≈ 480
even after the first was fixed.

**`sdf: null` never meant instant** despite the documentation saying so: the
constructor read `None` as "unspecified" and fell back to 41. Only `sdf: 0` had
ever worked.

**Surge was off by one** (`b2b - at + base + 1`); the wiki's own worked example
(streak 8 → 8 lines) rules the `+1` out.

**Smaller ones.** HOLD was drawn under the NEXT queue rather than beside the board.
`settings.yaml` was not git-ignored while this README claimed it was. And **three
tests had pinned wrong behaviour** — the `1/60` soft-drop floor, and the mirrored
kick rows — and were corrected rather than deleted, with the reasoning left in the
comments. A test that encodes a bug is worse than no test, because it defends it.

### What the workflow got right

The measurement culture is the reason any of this was findable, and it was already
here before this session started:

* **Differential tests against the engine.** The search's bitmask model is checked
  against `valid_actions` and `simulate_placement` on random boards (2,729
  placement-set and 93,564 board-simulation comparisons, zero mismatches). When the
  shapes were mirrored, those kept passing — which is itself the lesson: a
  differential test proves two things agree, not that either is right.
* **The 336-attempt wall sweep.** Every piece, every state, every direction, both
  walls, on an empty board, where SRS guarantees success. It reported 38 refusals
  when a human said "I can't rotate at a wall", and 0 once the causes were fixed.
  It is now a permanent test.
* **`tools/clear_mix.py`, and before/after arms on identical seeds.** The claim
  that hold was the lever (19.2% → 56.9%) is an A/B on the same seeds, not a
  before-and-after anecdote.
* **The recorder.** `--record` plus `tools/analyze_session.py` reports refused
  rotations with the piece, cell, direction, every offset tried and the reason, and
  now surfaces rotations that *succeeded but are suspicious* — far kicks, states
  that do not match `(from + direction) % 4`. On the first session it analysed it
  immediately showed an index-11 far kick: the old inflated table, visible in
  recorded data.
* **Negative results kept.** The Cold Clear port measured 6.2% tetrises against
  57% (it survives as the `cold_clear` preset so the claim can be re-checked); the
  danger mode farmed singles at height 13; a well-depth bonus died in 37 lines.
  These are in the file on purpose.

### What the workflow got wrong

* **Twice, a subagent reported "all suites green" and my own run showed
  failures** — once with three checks failing in `test_search` and `test_battle`
  while another agent was mid-rewrite of `engine.py`, and once when the suite stood
  at what were then 734 checks, where
  the claim was true but a concurrent writer had not been accounted for. The rule
  this produced: **verify the verification**, from a clean run, after every writer
  has stopped. Every count in this file was re-measured that way.
* **A README warning that should not have shipped.** One agent added a † note
  saying the DQN checkpoints are stale and "die in all six games". Two independent
  measurements contradict it (`best_rewardfix` at 63.5 lines/game over 4 games and
  70.5 over 2, clearing lines in every game), and the kick tables cannot affect that
  agent at all: it teleports to `(rotation, x)` and hard-drops, never rotating. The
  note is unverified at best; a proper 40-game fixed-seed run would settle it.
* **A "table-correct exception" that was the bug.** Sixteen floor-rotation refusals
  were documented as legitimate SRS+ behaviour ("a flat I cannot stand up"). They
  were the sign bug. The tell was that the claim implied a rule SRS+ does not have:
  the I stands up fine via the fifth published offset.
* **The human was a required part of the test suite.** "It feels wrong", "T-spin
  triples don't work", "200× is still too slow", "it creates two holes and doom
  loops" — none of those were reachable from the tests as written. The recorder
  exists because the alternative was guessing.
* **Nothing visual was ever verified.** No window was opened on the human's machine
  in the entire session. Layout is asserted by geometry (HOLD left of the board,
  both halves inside the window, non-background pixels present on an off-screen
  surface) and that is all: **the renderer's appearance, the settings screen's
  legibility and the garbage meters have never been looked at.**
* **The i-dependency term first shipped without tests or measurement** — a
  follow-up pass wrote `tests/test_dependency.py` (21 checks), re-ran the arena at
  ten games a side instead of six, corrected the "two at once are gone" claim that
  the small sample had produced, identified the gate as the mechanism and the price
  as decorative, and found that the function skipped the **wall columns** entirely.
  That last one was a real miss: a covered one-wide shaft dug at the edge is the
  most common place for a dependency to appear, and it was not being counted at
  all.

### Where the limits now stand

* **T-spins are detected and paid**, but the preset that ships is `t-spins`: only
  a T is classified. Z and S spins rotate correctly and correctly send nothing,
  because that is what the preset says — TETR.IO's current default is `all-mini+`,
  recorded as a stated deviation rather than silently implemented.
* **The DQN checkpoints' status after the rotation-state reorder is unverified.**
  See the — note above; the measurements disagree with it.
* **The renderer has never been seen.** Not once, in the whole session.
* **The search is CPU-only and costs roughly 15× the plain heuristic per move**
  (~14 ms/piece at depth 3, beam 6), which is why `search_tetris` is opt-in and why
  a six-game measurement takes about two minutes.
* **`search_tetris` is a tetris machine, not a survival machine**: 5 of 6 games
  topped out inside the 2,500-piece cap against 2 of 6 for `heuristic_tetris`.
  That trade was chosen deliberately and it is still a trade.
