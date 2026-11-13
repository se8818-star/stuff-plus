# Pre-registration: pre-contract 2025 Stuff+ (junior paper)

Written and committed before any candidate below is run on this design. The
commit that adds this file is the timestamp. Nothing here changes after results
are seen; any deviation goes in a "Deviations" section at the bottom.

## Purpose

A Stuff+ score for every pitcher's 2025 regular season that uses only
information MLB teams had before the 2025–26 free-agent signing period, for an
economics paper on free-agent pitcher AAV. The score is an explanatory
variable, so the rule is: **nothing from 2025 or 2026 outcomes is used to choose,
tune or train the model.**

| Season | Role |
|---|---|
| 2022–2024 | training, feature selection and validation |
| 2025 | scored only (pitch physics in, Stuff+ out); its outcomes are never read |
| 2026 | not loaded at all |

Code: `pipeline/precontract_2025.py`. The selection step only opens the 2022,
2023 and 2024 data files and asserts that no other season is present.

## What was already known

Earlier work in this repository (`docs/PREREGISTRATION.md`,
`reports/confirmation.md`) compared feature sets using 2025 and 2026 outcomes.
Those results are **not** used here: none of their numbers enter the selection
below. The candidate list is not chosen from them either. It is every
location-neutral VAA/HAA specification already in the code
(`stuffplus/features.py` variants B–D and `stuffplus/models.py` S6), plus the
tjStuff+ v3.0 baseline. The author of this design had seen those earlier
results, which is stated in the methodology as a limitation.

## Candidates (fixed)

All are a single LightGBM regression with tjStuff+ v3.0's settings (1000 trees,
learning rate 0.01, 31 leaves, min_child_samples 20, subsample 0.8,
colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 0.2), averaged over seeds 0, 1
and 2. Target: tjStuff+'s count-specific run value per pitch. No tuning.

| Name | Features |
|---|---|
| A (baseline) | tjStuff+ v3.0: velocity, spin rate, extension, az, ax, release x, release z, and velocity / az / ax differences from the primary fastball |
| B_plus_center | A + location-neutral VAA and HAA at the zone centre |
| B9_plus_grid9 | A + location-neutral VAA and HAA at all 9 points of the fixed 3×3 grid |
| C1_center_no_accel | A with az and ax replaced by centre VAA and HAA |
| C_grid9_no_accel | A with az and ax replaced by the 9-point VAA and HAA grid |
| D1_center_angles_only | velocity, spin, extension, velocity difference, VAA and HAA differences from the fastball, centre VAA and HAA (no movement, no release point) |
| D_grid9_angles_only | as D1 with the 9-point grid |
| S6 | velocity, extension, centre VAA and HAA, VAA and HAA differences from the fastball |
| A_reseeded | null control, not a candidate: A with seeds 3, 4, 5 |

## Validation (all inside 2022–2024)

Two designs, three season pairs. In each pair, a pitcher's Stuff+ in season N
is compared with **his own** results in season N+1.

1. **Forward split** (mirrors how the final model is used): train on 2022,
   score 2023 and 2024, compare 2023 Stuff+ with 2024 results.
2. **Cross-validation by pitcher**: 5-fold GroupKFold over all 2022–2024
   pitches, grouped by pitcher. Each pitcher's pitches in every season are
   scored by a model that never saw him. Pairs: 2022 → 2023 and 2023 → 2024.

Stuff+ = 100 − 10 × (xRV − mean) / sd, mean and sd from the scored season's
pitches. Pitcher-season Stuff+ = mean over his pitches.

Metrics per pair (pitchers with at least 100 pitches in both seasons; result
correlations sign-flipped so higher is better):

- next-season wOBA allowed, next-season run value per 100 pitches (selection);
- stickiness (Stuff+ vs next-season Stuff+), next-season K−BB%, pitch-level
  out-of-sample R² (reported, not used for selection).

## Selection rule

- **Selection score** = mean of the next-wOBA and next-RV/100 correlations,
  averaged over the three pairs (forward 2023 → 2024, CV 2022 → 2023, CV
  2023 → 2024), minimum 100 pitches, unshrunk.
- The **challenger** is the candidate (other than A and the null control) with
  the highest selection score.
- The challenger replaces A only if its selection score is higher than A's
  **and** the paired-bootstrap interval for (challenger − A) excludes zero. The
  interval is Bonferroni-adjusted for the seven challengers: 99.29% two-sided
  (percentiles 0.357 and 99.643), 5,000 replicates, seed 0, pitchers resampled
  jointly across the three pairs.
- Otherwise A is selected. A is the default because it is the published,
  established specification; a challenger has to show a difference that is
  unlikely to be noise.

## Final model and output

- Fit the selected specification on all 2022–2024 pitches (seeds 0, 1, 2).
- Score every 2025 regular-season pitch. Outcome columns are dropped from the
  2025 data before scoring.
- Stuff+ scale: mean and sd of the 2025 xRV predictions (2025 league-average
  pitch = 100, one sd = 10). This uses 2025 pitch physics only, not outcomes.
- Pitcher Stuff+ = pitch-weighted mean, shrunk toward the league mean by
  n / (n + k), with k from split-half reliability of the cross-validated
  2022–2024 Stuff+ (the repository's existing method). The raw mean is also
  reported.
- Output CSV: MLBAM ID, name, total 2025 pitches, pitches scored, Stuff+ (shrunk
  and raw), throwing hand and SP/RP role. Every 2025 pitcher is included; a
  minimum-pitch filter is left to the regression.
