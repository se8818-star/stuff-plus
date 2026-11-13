# Pre-contract 2025 Stuff+: methodology

This note covers how the Stuff+ variable in `leaderboards/precontract_2025_pitchers.csv`
was built. It is meant as source material for a methodology section. The plan
and the rule for choosing the model were written down and committed before any
model was run (`docs/PRECONTRACT_2025_PREREGISTRATION.md`). The full results are in
`reports/precontract_2025/`.

## What Stuff+ measures

Stuff+ grades how hard a pitch is to hit, using only the pitch's physical
traits: how fast it is, how it spins, how it moves, where it is released, and
how it differs from the pitcher's fastball. It ignores where the pitch was
thrown, the count, and the batter. A score of 100 is a league-average 2025
pitch. Each 10 points is one standard deviation, and higher is better for the
pitcher.

## No future data

The score is meant to reflect what teams could have known before the 2025–26
free-agent signing period.

| Season | How it is used |
|---|---|
| 2022–2024 | training data, plus all model comparison and validation |
| 2025 | scored only. Pitch physics go in and Stuff+ comes out. Outcome columns (pitch result, events, batted-ball data, run values) are deleted from the 2025 data before scoring. |
| 2026 | never loaded |

The model-comparison code only opens the 2022, 2023 and 2024 files, and it
checks that no other season is present. The 2025 regular season ended before
free agency opened, so its pitch-tracking data were available to teams when
they made offers.

## Data

- Statcast pitch-by-pitch data for MLB regular seasons, downloaded from
  Baseball Savant with pybaseball 2.2.7. This is real tracking data, not
  simulated. The files' SHA-256 hashes are recorded in `models/stuffplus_precontract_2025.json`.
- 2022: 710,210 pitches. 2023: 720,684. 2024: 711,899. 2025: 712,528. Each
  season has 2,429–2,430 games.
- Excluded pitch types: knuckleballs, eephuses, pitchouts, intentional balls,
  position-player lobs and similar (about 0.3% of pitches).
- Training rows: 2,114,948 pitches from 2022–2024 that have every feature and
  a target.

## Target variable

The model predicts each pitch's **run value**: how much the pitch changed the
expected number of runs the batting team would score. It uses tjStuff+ v3.0's
table of the average run value of each outcome (ball, called strike, whiff,
foul, single, home run, and so on) in each ball–strike count. The table was
estimated from 2024 data, inside the training window. If a pitch ends the plate
appearance, it gets the value of that event. Otherwise it gets the value of
the pitch result. A few rare outcomes the table does not cover are dropped
(about 0.17% of pitches).

The model's prediction for a pitch is its **expected run value (xRV)**: the
average run value of pitches with the same physical traits.

## Inputs (final model: tjStuff+ v3.0 feature set)

| Feature | Meaning |
|---|---|
| Velocity | release speed (mph) |
| Spin rate | rpm |
| Extension | how far in front of the rubber the pitcher releases the ball (ft) |
| Vertical acceleration (az) | vertical movement, gravity included |
| Horizontal acceleration (ax) | horizontal movement, mirrored so left- and right-handers share one scale |
| Release side (x0) | horizontal release position, mirrored the same way |
| Release height (z0) | vertical release position |
| Velocity difference | this pitch's velocity minus the pitcher's primary fastball's average |
| Vertical-movement difference | az minus the primary fastball's average az |
| Horizontal-movement difference | absolute difference from the primary fastball's average ax |

The primary fastball is the pitcher's most-used four-seamer, sinker or cutter
that season. A pitcher with none is compared with his hardest pitch type.

## LightGBM

The model is a LightGBM gradient-boosted regression. It builds 1,000 small
decision trees one after another, and each tree corrects the errors of the
trees before it. This lets the model capture non-linear effects and
interactions, for example that the same amount of rise matters more on a
fast fastball than a slow one. Settings follow tjStuff+ v3.0 and were not
tuned: learning rate 0.01, 31 leaves per tree, at least 20 pitches per leaf,
80% of rows and columns sampled per tree, and L1 and L2 penalties of 0.1 and
0.2. Three copies are trained with random seeds 0, 1 and 2 and averaged, so
the result does not depend on one seed.

## Location-neutral VAA and HAA (the alternatives tested)

Vertical and horizontal approach angles (VAA, HAA) describe the angle at which
a pitch crosses the plate. A flat fastball, with a VAA close to 0, is harder
to hit. The raw angle mostly reflects where the pitch was thrown, though: high
pitches arrive flatter. To isolate the pitch's shape, the "location-neutral"
version takes each pitch's fitted flight path from Statcast, keeps its release
point, release speed and accelerations, and re-aims it at fixed spots in the
strike zone. It then computes the angle the pitch would have at each spot:

1. Rewind the fitted trajectory to the release point.
2. Keep the release position, the speed toward the plate and all three
   accelerations unchanged.
3. Solve for the initial sideways and vertical velocity that would make the
   pitch cross the front of the plate at a target point.
4. Compute VAA = −atan(v_z / v_y) and HAA = −atan(v_x / v_y) at the plate,
   mirrored for left-handers.

The "centre" version uses the middle of a fixed 1.5–3.5 ft, 17-inch zone. The
"grid" version uses the nine centres of a 3×3 grid. Every pitcher's pitches
are aimed at the same spots, so the angles contain no information about where
the pitch actually went.

## Model selection and validation

Eight specifications were compared: the tjStuff+ v3.0 baseline (A) and seven
that add location-neutral VAA/HAA or use them in place of movement or release
point. A ninth run, A with different seeds, measures how much the numbers move
by chance alone.

| Model | Features |
|---|---|
| A (tjStuff+ v3.0) | the ten above |
| B_plus_center / B9_plus_grid9 | A + centre angles / A + 9-point angles |
| C1_center_no_accel / C_grid9_no_accel | A with movement (az, ax) replaced by centre / 9-point angles |
| D1_center_angles_only / D_grid9_angles_only | velocity, spin, extension and angles only (no movement or release point) |
| S6 | velocity, extension, centre VAA and HAA, and their differences from the fastball |

**Validation idea.** A good Stuff+ in one season should predict a pitcher's
results in the next season. Each model was tested on three season pairs, all
inside 2022–2024:

- **Forward split**: train on 2022, score 2023, and compare each pitcher's
  2023 Stuff+ with his 2024 results. This matches how the final model is used
  (train on earlier seasons, score a later one).
- **Cross-validation by pitcher**: split all pitchers into five groups. Each
  group is scored by a model trained only on the other four, so no pitcher is
  graded by a model that saw him. This gives two more pairs: 2022 → 2023 and
  2023 → 2024.

**Selection score.** For each pair: the correlation between a pitcher's
Stuff+ and his next-season wOBA allowed, and the correlation with his
next-season run value per 100 pitches (signs flipped so higher is better).
The score averages the two over all three pairs, for pitchers with at least
100 pitches in both seasons (507–517 pitchers per pair).

**Rule (fixed in advance).** An alternative replaces A only if it beats A's
score **and** a paired bootstrap confidence interval for the difference
excludes zero. The interval uses 5,000 resamples of pitchers and is adjusted
for testing seven alternatives (Bonferroni, 99.29%). Otherwise A is kept.

**Result: A was selected.** Every alternative scored below A, so the interval
question never came up.

| Model | Selection score | Difference from A | Stickiness |
|---|---:|---:|---:|
| **A (tjStuff+ v3.0)** | **0.267** | – | 0.845 |
| A with different seeds (chance) | 0.263 | −0.004 | 0.846 |
| B_plus_center | 0.254 | −0.013 | 0.838 |
| B9_plus_grid9 | 0.253 | −0.014 | 0.839 |
| D1_center_angles_only | 0.235 | −0.032 | 0.860 |
| D_grid9_angles_only | 0.234 | −0.033 | 0.853 |
| C1_center_no_accel | 0.234 | −0.033 | 0.850 |
| C_grid9_no_accel | 0.232 | −0.035 | 0.846 |
| S6 | 0.222 | −0.045 | 0.831 |

Stickiness is the correlation of a pitcher's Stuff+ with his own Stuff+ the
next season. The angle-only models are slightly stickier but worse at
predicting results. The likely reason: once velocity, movement and release
point are known, the approach angle is almost fully determined, so it adds
little new information.

For the selected model, the next-season correlations are 0.27 with wOBA and
0.31 with K−BB% at 100+ pitches. At 500+ pitches they rise to 0.31 and 0.36.

## Final model and scoring

1. Train A on all 2022–2024 pitches (three seeds, averaged).
2. Predict xRV for every 2025 regular-season pitch (705,658 of 712,528 could
   be scored; the rest are excluded pitch types or are missing tracking data).
3. **Normalize**: Stuff+ = 100 − 10 × (xRV − mean) / SD, where the mean and SD
   come from the 2025 predictions (mean −0.00073 runs, SD 0.00612 runs). A
   lower expected run value means a better pitch, hence the minus sign.
   This uses 2025 pitch physics only, not outcomes.
4. **Pitcher score**: the average Stuff+ of a pitcher's 2025 pitches, shrunk
   slightly toward the league average for small samples:
   league + n / (n + k) × (raw − league), with k = 12.9 pitches. k comes from
   how well a pitcher's Stuff+ in alternate games agrees with itself
   (split-half reliability 0.95 on the cross-validated 2022–2024 scores).
   At 100 pitches this keeps 89% of the raw distance from average; at 500
   pitches, 97.5%.

## Output

`leaderboards/precontract_2025_pitchers.csv`, one row per pitcher who threw a
2025 regular-season pitch (873 pitchers):

| Column | Meaning |
|---|---|
| `mlbam_id` | MLB Advanced Media player ID (merges with Baseball Reference/FanGraphs ID maps) |
| `name`, `name_last_first` | player name |
| `p_throws`, `role` | throwing hand; SP if he started at least half of his games, else RP |
| `g`, `gs` | games pitched and started |
| `total_pitches` | all 2025 regular-season pitches in Statcast |
| `pitches_scored` | pitches the model could score |
| `stuff_plus` | **the variable**: shrunk pitcher Stuff+ |
| `stuff_plus_raw` | unshrunk mean, for robustness checks |

50 rows have no score: position players whose pitches are all excluded types.
A handful of other position players have extreme scores (as low as −69) from a
few slow pitches. Limit the sample to real pitchers, for example 100+ pitches,
before using the variable.

## Descriptive statistics (2025, pitcher level)

| Sample | N | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| Pitchers with 100+ pitches | 723 | 100.29 | 4.55 | 87.03 | 118.36 |
| Pitchers with 500+ pitches | 479 | 100.54 | 4.72 | 87.90 | 118.36 |
| Every scored pitcher | 823 | 99.18 | 11.38 | −69.03 | 118.36 |

More cuts are in `reports/precontract_2025/descriptive_stats_2025.md`.

## Reproducibility

```bash
python -m pipeline.precontract_2025 select   # ~1 hour on 4 cores
python -m pipeline.precontract_2025 final    # ~2 minutes
```

Seeds are fixed (LightGBM 0, 1, 2; null control 3, 4, 5; bootstrap 0), the
cross-validation split is deterministic, and LightGBM runs in deterministic
mode. Versions: Python 3.9.25, LightGBM 4.6.0, pandas 2.3.3, NumPy 2.0.2,
scikit-learn 1.6.1, pybaseball 2.2.7. They are also logged in the results files.

## Credit and limitations

- The feature set, target and LightGBM settings replicate tjStuff+ v3.0 by
  Thomas Nestico (https://github.com/tnestico/tjstuff_plus, MIT License). This
  is an independent re-implementation on Statcast data, not tjStuff+ itself.
- Earlier work in this repository had compared some of these specifications
  using 2025 and 2026 results. None of those numbers were used here, and the
  candidate list is every location-neutral specification already in the code,
  not a subset chosen from those results. The person who set up this design
  had seen them, though, so the advance commitment cannot fully rule out
  that knowledge.
- Because the final model is trained on 2022–2024, a 2025 score for a pitcher
  who pitched in those seasons can partly reflect how his similar pitches fared
  then. That information was also available to teams before signing. The
  forward split tests exactly this setup.
- Stuff+ ignores command and location, so it is one part of pitcher quality,
  not all of it.
