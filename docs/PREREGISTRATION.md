# Pre-registration: candidates and selection rule

Written and committed before any candidate below is run. The commit that adds
this file is the timestamp. Nothing here changes after results are seen; any
deviation is recorded in a "Deviations" section at the bottom with the reason.

Goal: maximize next-season predictiveness of Stuff+ without overfitting.

## What has already been seen

- **Fold 1, variant A only** (Phase 2): stickiness 0.867, next wOBA 0.227, next
  RV/100 0.227 at 100 pitches, plus A's sanity controls (seeds, bagging, y=50
  release point, mirroring removed). No fold 1 result for G, H, J or any
  combination has been looked at.
- **Fold 2: nothing.** An earlier all-variant run was stopped when this plan
  arrived. It had finished fold 1 and computed two variants on fold 2 (A and
  B_plus_center). Its log was deleted without the fold 2 numbers, or the fold 1
  numbers for G/H/J, being read, and it wrote no results file.
- 2026 pitch data has been used only for the descriptive data-quality report.

## Data splits

| Name | Train | Score | Validate against | Use |
|---|---|---|---|---|
| Inner CV | 2022–23, 4/5 of pitchers | 2022, held-out pitchers | their 2023 | development |
| Fold 1 | 2022–23 | 2024 | 2025 | development check |
| Fold 2 | 2022–24 | 2025 | 2026 | **locked**: one-shot confirmation |

- **Inner CV**: 5-fold GroupKFold by pitcher on 2022–23. Each model is fit on
  four folds' pitchers (both seasons) and predicts the fifth fold's pitchers.
  Pooling the five held-out sets gives out-of-fold Stuff+ for every 2022 and
  2023 pitch. Metrics: 2022 Stuff+ against the same pitchers' 2023 Stuff+
  (stickiness) and 2023 results (wOBA, K−BB%, RV/100).
- **Fold 2 lock**: `run_ablation.py` refuses fold 2 without `--unlock-fold2`.
  It is run exactly once, for A and the frozen candidate, after `FROZEN.md`
  (candidate, hyperparameters, shrinkage decision) has been committed.

## Candidates (fixed; nothing is added)

All use TJ's LightGBM settings by default (1000 trees, learning rate 0.01,
31 leaves, min_child_samples 20, colsample 0.8, reg_alpha 0.1, reg_lambda 0.2)
and average 3 seeds (0, 1, 2).

| # | Name | Features | Target |
|---|---|---|---|
| 1 | A (baseline) | tjStuff+ v3.0: velo, spin, extension, az, ax, release x/z, and speed/az/ax differences from the primary fastball | TJ count-specific run value |
| 2 | J | A's | Balls in play valued by xwOBA converted to runs for the count (`target_x`); everything else as in A |
| 3 | G | A + `same_side` (batter stands on the pitcher's throwing side) | as A |
| 4 | H | A + `ssw_dev` (observed spin axis minus the axis implied by pfx_x/pfx_z, mirrored for LHP) | as A |
| 5 | J+G+H | A + `same_side` + `ssw_dev` | `target_x` |
| 6 | Component | same as 5 | four models combined, below |

Dropped: the grid variants (B, B9, C, C1, D, D1). The 18 grid angles span two
dimensions and are implied by A's features. E, F, I and A50 are also not run.

### Candidate 6: component model

Pitch outcomes from Statcast `description`:

- swing: swinging_strike, swinging_strike_blocked, foul_tip, bunt_foul_tip,
  missed_bunt, foul, foul_bunt, foul_pitchout, hit_into_play
- whiff (within swings): swinging_strike, swinging_strike_blocked, foul_tip,
  bunt_foul_tip, missed_bunt (the outcomes TJ values as a swinging strike)
- take: ball, blocked_ball, pitchout, called_strike, hit_by_pitch

Four LightGBM models on candidate 5's features:

1. P(swing), binary, all pitches
2. P(whiff | swing), binary, swings only
3. P(called strike | take), binary, takes only
4. E[runs | contact], regression of `target_x` on swings that are not whiffs
   (fouls keep their table value; balls in play use xwOBA-runs)

Combined per pitch:

    xRV = P(swing) * [ P(whiff) * v_whiff + (1 - P(whiff)) * E[runs | contact] ]
        + (1 - P(swing)) * [ P(cs) * v_cs + (1 - P(cs)) * v_ball ]

where v_whiff, v_cs and v_ball are the mean TJ run value of whiffs, called
strikes, and other takes in the training years (constants, so the score is
count-neutral like A's). Pitches with no target are excluded, as for A.

### Scoring conventions (all candidates)

- Stuff+ = 100 − 10·z(xRV), z from the scored season's pitch-level mean and sd.
- G, J+G+H and the component model are scored on the batter side actually
  faced. A platoon-neutral version is not run.
- Next-season results are always actual outcomes (wOBA, K−BB%, and RV/100 on
  TJ's run values), including for candidates trained on `target_x`.

### Null control (not a candidate)

A′ = A refit with seeds 3, 4, 5. The A′ − A differences show how large a
"gain" pure seed noise produces. A′ cannot be selected.

## Regularization

Stage 1 compares the six candidates at TJ's settings. Tuning happens only in
stage 2 and only for two models: A and the stage 1 winner.

- Optuna (TPE, seed 0), **30 trials per model**, same search space for both.
- Objective: out-of-fold loss on the model's own training target under 3-fold
  GroupKFold by pitcher on the training years, with early stopping (patience
  50, at most 3000 trees). Training years only; fold 1's 2024–25 are never used.
- Search space, regularizing parameters only:
  min_child_samples 20–2000 (log), num_leaves 7–127 (log), reg_lambda 0.01–100
  (log), learning_rate 0.01–0.1 (log) with the number of trees set by early
  stopping. Everything else stays at TJ's values.
- For the component model, one shared parameter set is tuned for all four
  models; the objective is the mean of each component's loss relative to its
  loss at TJ's settings.
- A model adopts its tuned settings only if they raise its selection score
  (below) over TJ's settings; otherwise it keeps TJ's. A and the winner make
  that choice independently, by the same criterion.

## Shrinkage

Empirical-Bayes shrinkage toward the league mean by pitch count:

    shrunk = league_mean + n / (n + k) * (raw_mean − league_mean)

- Pitcher level: league_mean is the scored season's pitch-weighted mean Stuff+.
  Pitch-type level: the league mean for that pitch type.
- k comes from split-half reliability on the **training years only**, using the
  inner-CV out-of-fold Stuff+ (so no pitcher's score comes from a model that
  saw them). Each pitcher-season with at least 100 pitches (pitch-type level:
  50 of that type) is split into alternating games by date; r is the
  correlation of the two half means and m the harmonic mean half size;
  k = m·(1 − r)/r. One k per level, estimated separately for each model by the
  same procedure.
- Every table is reported with and without shrinkage, for A and the candidate
  alike.
- The **unshrunk** numbers are primary for selection and confirmation.
  Shrinkage ships in the final leaderboards if it does not lower the frozen
  model's fold 1 score; that decision is recorded in `FROZEN.md` before fold 2.

## Selection rule

Selection score of a model = mean of four correlations: next-season wOBA and
next-season RV/100 (sign-flipped so higher is better), in the inner CV and in
fold 1, unshrunk, minimum 100 pitches in both seasons.

1. **Stage 1**: the candidate with the highest selection score at TJ's settings
   is the winner. If it is A, A ships and fold 2 is run once for reporting only.
2. **Stage 2**: tune A and the winner as above; each keeps tuned or TJ settings.
   Commit `FROZEN.md`.
3. **Confirmation**: run frozen A and the frozen winner once on fold 2. The
   winner is confirmed only if, at minimum 100 pitches, unshrunk:
   - next wOBA **and** next RV/100 are both higher than A's, each with a paired
     bootstrap 95% CI (2000 replicates over pitchers) that excludes 0, and
   - stickiness is no more than 0.01 below A's.
4. If confirmation fails, that is reported as a failure and the simplest model,
   A at TJ's settings, ships.

## Reporting

- Every candidate that was run, in every split, not only the winner; stage 2
  tuned and untuned results for both tuned models.
- Minimum 100, 300 and 500 pitches for every table.
- Paired bootstrap 95% CIs against A in inner CV and fold 1.
- How many candidates would beat A by chance: with five non-baseline
  candidates and no true differences, about 2.5 are expected to beat A on any
  one point estimate, and the best of five is ahead of A with probability near
  0.97 if they were independent, so "the best candidate beat A in development"
  is not evidence. About 0.125 of the five are expected to show a development
  CI above 0 on a given metric. The empirical size of chance differences is
  reported from A′ − A. The fold 2 test is a single comparison of one frozen
  candidate, with a false-positive rate of at most 2.5%.

## Deviations

### 1. Candidate S6 added after stage 1 results (owner's request)

Added after the stage 1 results for the six original candidates had been
seen, so it does not have the protection the original list has. Written and
committed before S6 was run on anything.

| # | Name | Features (6) | Target |
|---|---|---|---|
| 7 | S6 | velocity, extension, location-neutral VAA, location-neutral HAA, VAA difference from the primary fastball, absolute HAA difference from the primary fastball | TJ count-specific run value (as A) |

- VAA/HAA are the centre-of-zone point of the fixed 3×3 grid (`vaa_ln`,
  `haa_ln`). The 18 grid angles span two dimensions (first two principal
  components explain 100.00% of their variance), so the centre pair carries the
  whole grid.
- The fastball differences use the same primary-fastball reference as A:
  `vaa_diff` is signed, `haa_diff` is absolute (as A's `ax_diff` is).
- No spin rate, release position, or acceleration terms.
- TJ's LightGBM settings, seeds 0–2, same scoring conventions as every other
  candidate.

What was already known when S6 was added: stage 1 results for A, J, G, H,
J+G+H and the component model. No result for any approach-angle model has been
seen (the earlier all-variant run that included similar variants was stopped
and its log deleted unread).

S6 is run on the inner CV and fold 1 and reported next to the others. It does
not change the stage 1 winner (J), which was fixed by the original rule. Whether
and how S6 is run on fold 2 is decided and recorded here before fold 2 is
unlocked.


### 2. Stage 2 tuning skipped (owner's decision, after stage 1, before fold 2)

No hyperparameters are tuned. A, J and S6 are frozen at TJ's LightGBM
settings. Reason: limiting overtuning is a stated goal, stage 1 separated no
candidate from A, and the tuned-versus-untuned adoption step would have been
decided by a score whose sampling noise (about ±0.04) dwarfs any plausible
tuning gain. This removes a choice; it cannot favour any candidate. `tune.py`
stays in the repo, unrun.

### 3. Two candidates on fold 2: J and S6 (owner's decision, before fold 2)

Fold 2 is run once for A, J (the stage 1 winner under the original rule) and
S6 (deviation 1). Stage 1 development results for all three were known when
this was decided; no fold 2 result was.

- The original confirmation rule is applied to J and to S6 separately and
  unchanged: at minimum 100 pitches, unshrunk, next wOBA and next RV/100 both
  above A with paired-bootstrap 95% CIs excluding 0, and stickiness no more
  than 0.01 below A.
- Two comparisons instead of one: the chance that at least one passes with no
  true difference is at most 5%, up from at most 2.5%. No multiplicity
  correction is applied; both verdicts are reported.
- What ships: if exactly one is confirmed, that one. If both, the one with the
  higher mean of its fold 2 next-wOBA and next-RV/100 correlations. If neither,
  A at TJ's settings.
- S6's development stickiness was 0.02–0.05 below A's, so on development
  evidence it is expected to fail the stickiness condition.

### 4. Shrinkage decision

Recorded in `FROZEN.md`: shrinkage ships in the leaderboards for whichever
model ships, since it did not lower the fold 1 score of A, J or S6.
