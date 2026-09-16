# TETR.IO multiplayer (versus) garbage ruleset — sourced reference

Source tags: **[W]** [tetris.wiki/TETR.IO](https://tetris.wiki/TETR.IO) (community wiki, cites osk patch notes) · **[FAQ]** [tetrio.github.io/faq](https://tetrio.github.io/faq/mechanics.html) (TETR.IO-hosted FAQ, community authors ZaptorZap/deep4amthoughts) · **[T]** [Triangle.js `halp1/triangle`](https://github.com/halp1/triangle) (community reverse-engineering of the TETR.IO engine, replay-accurate) · **[HD]** [harddrop.com/wiki/TETR.IO](https://harddrop.com/wiki/TETR.IO) (community) · **[IS]** [github.com/tetrio/issues](https://github.com/tetrio/issues/issues/1305) (official tracker, reporter-derived).

## 1. Base attack table (lines of garbage per clear)

| Clear | Garbage | Source |
|---|---|---|
| Single | 0 | [T] `src/engine/utils/damageCalc/index.ts`; QUICK PLAY forces 0-combo non-spin Singles to 1 [W] |
| Double / Triple / Quad (Tetris) | 1 / 2 / 4 | [T] (same as guideline) |
| Spin, no lines | 0 | [T] |
| Mini Spin Single / Double / Triple | 0 / 1 / 2 | [T] |
| Spin Single (TSS) / Double (TSD) / Triple (TST) | 2 / 4 / 6 | [T] |
| Spin Quad / Penta | 10 / 12 | [T] |
| >5 lines | spin: `12+2(n−5)`, else `5+(n−5)` | [T] |
| All Clear (PC) | configurable `allclear_garbage`: **5** current TETRA LEAGUE, **10** league S1, **3** QUICK PLAY; `allclear_b2b`: 1 (league), 2 (QP) | [T] `src/classes/room/presets.ts`; [W] QP |
| B2B (charging = default) | +1 per attack while streak alive; Surge at B2B≥4 | [W] |
| B2B chaining (legacy, toggle) | +1/+2/+3/+4/+5/+6/+7/+8 at streaks 2-3 / 4-8 / 9-24 / 25-67 / 68-185 / 186-504 / 505-1370 / 1371-3725 | [W], [HD] |

Surge: stored lines = `b2b − at + base` where `at`=4 (start threshold), `base`=4 (non-QP) / 1 (QP); sent as 3 chunks `round(s/3), round(s/3), s−2·round(s/3)` [W], [T] `src/engine/index.ts`. That is, the Surge pays the streak itself: 4 lines at a streak of 4, 5 at 5, and 8 at 8 (5 in QUICK PLAY) — the wiki's worked example [W]. An earlier version of this file (and of `versus.py`) had a `+1` on the end, which would make a streak of 8 pay 9 and a QP streak of 8 pay 6; the worked example rules it out, so the `+1` is gone. A 1v1 warning appears when one player has ≥8 B2B less than the opponent [W]. Surge is unaffected by combo.

## 2. Combo ("Multiplier" table, default)

`garbage = base × (1 + 0.25 × combo)`; if base = 0 use `ln(1 + 1.25 × combo)` (2-combo and above); final value = max of the two [W], [T] (`comboBonus 0.25`, `comboMinifier 1`, `comboMinifierLog 1.25`). Rounding: **DOWN** (floor) is used by TETRA LEAGUE/custom defaults; **RNG** (weighted) by QUICK PLAY [W]; `roundmode` per room [T]. Alternative combo tables exist (`none`, `classic guideline` = +0,1,1,2,2,3,3,4,4,4,5; `modern guideline` = +0,1,1,2,2,2,3,3,3,3,3,3,4) [T].

Room-level **garbage multiplier**: `garbagemultiplier` = 1 by default; grows by `garbageincrease` **per second** once `garbagemargin` frames have elapsed. Default/LEAGUE: +0.008/s after 10800 frames (180 s); gravity analog +0.0035/s after 7200 frames [T] presets + `utils/increase`; QP identical [T]. Attack is `round(attack × multiplier + specialBonus)`, `garbagespecialbonus` adds +1 when a spin/quad clears garbage (true in league) [T].

## 3. Cancelling, queue, meter, delay, caps

**Order per lock** [T]: compute lines → clear → update combo/B2B → compute attack (Surge chunks first, then clear attack, then PC) → if lines > 0: cancel against queue front, send remainder; if lines = 0: tank garbage (up to cap).

- **Cancel**: incoming queue is consumed oldest-first, 1 line per decrement, no offset limit; leftover attack is sent. No "partial block" other than that [T].
- **Tank trigger**: garbage enters **only on a placement that clears 0 lines** (any clear blocks entry for that placement) — "combo blocking"; the only blocking mode Triangle implements [T] `Game.GarbageBlocking`; [HD] labels TETR.IO "change on attack"/full blocking.
- **Travel/delay**: incoming garbage is tankable when `item.frame + garbagespeed ≤ currentFrame`; **garbagespeed = 20 frames = 0.333 s** by default (all presets) [T], [W] (20 frames, patch 4.2.0).
- **Queue visuals**: translucent yellow → translucent red → opaque red before becoming active; placing a piece before it is active does not let it enter [W] (QUICK PLAY). `garbagequeue` + `garbageare` (royale: 7 frames) control this [T].
- **Caps**: `garbagecap` = lines that may enter per placement: **8** default/LEAGUE (classic, arcade, bombs 8; QP 4 rising +0.033/s to `garbagecapmax` 10; 4-wide 4→8; royale 100) [T]. `garbagecapmax` clamps the dynamic cap (40 default). `garbageabsolutecap` = max lines held pending (royale 12; unlimited otherwise).
- **Opener Phase**: for the first 14 pieces (LEAGUE), if lines sent this round < pending incoming, you cancel twice as much; QP/other modes can set length [W], [T].
- **Windup** (QUICK PLAY only): a single received attack ≥8 lines is split into ≤4-line chunks (max 4 splits, e.g. 21 → 4+4+4+9), 1 s before the first chunk and 0.5 s between chunks; one Windup at a time, others queue [W]. Outgoing attacks >4 lines are split into ≤4-line chunks with a distance-based "height multiplier" [W].

## 4. Messiness / hole generation

`messiness.change` = chance the hole column is **re-rolled after an attack is fully consumed** (between attacks — this is the "change on attack" identity); `messiness.within` = chance **each line inside the same attack** re-rolls; `messiness.nosame` = new column ≠ previous; `messiness.timeout` = re-roll if no garbage tanked for N frames; `messiness.center` = exclude `round(width/5)` centre columns [T] `src/engine/garbage/index.ts`. Columns are uniform over `width − (holeSize−1)` positions, `garbageholesize` = 1 by default [T].

Documented preset values [T]: **classic 0/0** (hole column never changes = "clean"), **enforced delays 0.25/0.25**, **100 battle royale 0.25/0.25**, **bombs within 0.3**. QUICK PLAY: per-line chance, multiplied **2.5×** between separate attacks (garbage not tied to the queue) [W]. The **default custom-room / TETRA LEAGUE messiness values are not stated in any source I found — treat as UNKNOWN** (the presets omit them, so they are server defaults); the `messiness` "Messier Garbage"/Expert QP mods raise them [W].

## 5. Rise, board behaviour, top-out

No continuous rise in versus: garbage is inserted in a discrete batch at the bottom on a non-clearing lock, pushing the stack up; entry can be "instant" or "delayed"/rolling line-by-line (`garbageentry`, `garbageare`) — QP's Expert Mode uses all-at-once entry with a lowered delay [T], [W]. Buffer is 20 rows above the visible 20 [T] `classes/game`.

Top-out types [T] `GameOverReason`: `topout` (block out: new piece illegal), `garbagesmash` (garbage pushes you out), `topout_clear`, `winner`, `forfeit`, `disconnect`. TETR.IO-specific [IS] #1305 (closed as intended): you top out when the **active piece is dragged by garbage into the vanish zone** (spike KO), but a stack pushed into the buffer by garbage alone does **not** top you out. `nolockout: true` and `clutch: true` in every versus preset: lock out is disabled and a Clutch Clear pushes the next piece above the stack instead of dying [T], [W].

## 6. Match end / 1v1

TETRA LEAGUE is a first-to-N-wins match: **FT3** for A+ and below, **FT5** for SS and below, **FT7** for U and above (higher-ranked player's rank decides) [W]; preset shows `match.ft 7, wb 1, stock 0` [T]. `stock` = lives (0 = single life) [T]. ROYALE = last player standing [W]. FFA target bonus (unused in 1v1): +1/+3/+5/+7/+9 for 2/3/4/5/6+ enemies, "defensive" = tracked, not added [T]. Spin systems: `spinbonuses` = `T-spins` (league S1), `all-mini+` (current league default, Beta 1.5.0), plus `all`, `all+`, `all-mini`, `mini-only`, `handheld` (halves non-T spin garbage), `stupid`, `none` [T], [W].

## 7. Rotation system: SRS+

TETR.IO's default rotation system is **SRS+**, not plain SRS [W] (the game's settings
list SRS+ (default), SRS, SRS-X, ARS, NRS, ASC, Tetra-X, None). SRS+ is SRS with the
**180° turns folded into the same table**: the quarter-turn offsets are ordinary SRS,
and the 180 rows are TETR.IO's own addition.

- **Cell matrices and all four kick tables** (`kicks`, `i_kicks`, plus the 180 rows):
  <https://github.com/halp1/triangle> — [T] `src/engine/utils/tetromino/data.ts` and
  `src/engine/utils/kicks/data.ts`, entry `kicks["SRS+"]`. Triangle's matrices are
  stored with y growing **down**, matching this project's board convention, and its
  state order is 0 = spawn, 1 = clockwise (R), 2 = 180, 3 = counter-clockwise (L) —
  the same order this project uses, so the table keys (`"01"`, `"10"`, …) apply
  verbatim.
- **The kick offsets are y-down too, and that is easy to get wrong.** An engine
  applies a kick offset in the same coordinate space as the cells it moves, so
  Triangle's offsets are screen coordinates: its `kicks["SRS+"]."01"` is
  `[-1,0], [-1,-1], [0,2], [-1,2]`, which is the published y-up SRS row
  `(0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)` *mirrored*, not copied. Copying those
  values into a y-down engine is correct and needs no conversion; negating `dy`
  "because the source is y-up" mirrors all four tables. Measured against the fetched
  file, that mistake left 22 of the 24 published rows inverted — and the two rows it
  spared are the ones whose offsets happen to have `dy = 0`.
- **What a mirrored table costs.** It is internally consistent, so it passes a wall
  sweep, the spawn tests and every rotation-count check. What it breaks is every kick
  with a vertical component: the S/Z tuck that slides a piece into a slot, the drop
  that lands a T-spin triple, and the lift that finishes a rotation flush on the
  floor. Verified after the fix: 168 rotations in the air and 168 on the floor, all
  336 accepted, and the first fitting offset in the published row is the one used in
  every case.
- **SRS+ reorders the I piece's rows**: `0 → R` tries `(+1, 0)` where plain SRS
  tries `(-2, 0)`, and several other I rows swap their first two tests too. The
  J/L/S/T/Z rows are identical to plain SRS [HD]. This is why "the SRS table" is not
  a safe substitute for the SRS+ one.
- **Kick offsets are box offsets**: J/L/S/T/Z live in a 3×3 box, I in a 4×4, O in a
  2×2, and every rotation state of a piece sits in the same box. An implementation
  whose states drift inside their box (or that mirrors the matrices vertically —
  which swaps clockwise and counter-clockwise) has kick rows that are self-consistent
  and wrong, which is the failure mode to watch for.
- **Tests per turn**: five per quarter turn (the in-place rotation, then four
  published offsets), six per 180 for J/L/S/T/Z, two for the I's 180s, and the O
  never kicks at all. The index of the offset that fitted (0 = in place) is what the
  T-spin rule below calls the kick index.

**T-spin detection** (guideline rule, which is what the preset uses): the last
successful movement was a rotation, and at least three of the four corners of the
piece's box are filled — walls, the floor and anything below it count as filled, the
space above the board does not. It is a **mini** when the two corners the nub points
at are not both filled, unless the rotation only fitted on the fifth test (kick index
4), which promotes it to a full spin. Triangle's `cornerTable` lists the same four
corner offsets per state [T].

**Preset deviation, stated plainly.** TETR.IO's current multiplayer default is
`spinbonuses = all-mini+` (All-Mini+, Beta 1.5.0), under which *every* piece can spin
(non-T spins count as Mini-Spins) and the T may also use **immobile detection**. This
project implements the `T-spins` preset instead: the classic rotation-plus-corners
rule, T only [T] `spinbonusRules`, [W]. There is no All-Spin, no immobile detection,
and no mini classification for other pieces; a spin quad/penta beyond the table is
paid as a full spin.

## 8. Contrast with other games (only what clarifies)

Jstris/PPT/T99 use a **flat combo table** (+0,0,1,1,2,2,3…) and flat +1 B2B; TETR.IO's combo is multiplicative on the base attack and its B2B is a charging Surge that dumps a 3-chunk spike [W]. Jstris/PPT have no "Surge", no Clutch Clear, and no messiness/`nosame`/`timeout` knobs.

## Implementation checklist (priority order)

1. Attack table + multiplier combo + rounding mode (down/rng).
2. B2B charging (+1 per attack) and Surge (threshold 4, base 4 non-QP/1 QP, 3-chunk split).
3. Queue with 20-frame travel, `confirm`-based activation, cancel-oldest-first, send remainder.
4. Tank only on non-clearing locks; `garbagecap` 8 (dynamic cap/clamp), absolute cap.
5. Messiness engine: change/within/nosame/timeout/center; clean when change=within=0.
6. No lock out + Clutch Clear; top-out = block out or active-piece dragged into buffer.
7. Garbage multiplier growth (+0.008/s after 180 s), margin time for gravity.
8. Opener Phase (14 pieces, double cancel), special bonus, PC garbage per preset.
9. Match structure (FT3/5/7, stock, forfeit/disconnect), FFA targeting + target bonus, Windup (QP only).
10. Rotation: SRS+ tables copied verbatim, 180 rows included, T-spin detection
    (rotation + 3-of-4 box corners, mini unless both front corners or a far kick),
    spin attacks wired to the detection. Done; the remaining gap is All-Mini+ /
    immobile detection for the current default preset.
