# Stuff+ v1.0.0 — model card

A pitch-quality model in the style of tjStuff+ v3.0, trained on 2022–2026
Statcast data, with a pre-registered attempt to improve on tjStuff+'s feature
set. **That attempt did not succeed: none of the seven alternatives tested
beat the tjStuff+ v3.0 feature set on held-out data, so v1.0.0 ships that
feature set.** What this repository adds is the reproducible pipeline, the
validation evidence, shrinkage, and the 2026 leaderboards.

## Credit

The feature set, the count-specific run-value target, and the LightGBM settings
replicate **tjStuff+ v3.0 by Thomas Nestico**
(https://github.com/tnestico/tjstuff_plus, MIT License). `run_values.csv` is
copied unmodified from that repository, © 2024 tnestico; the full license text
is in `LICENSES/tjstuff_plus-MIT.txt`. This is an independent re-implementation
on Statcast columns and is not tjStuff+ itself: on 2024 pitchers its scores
correlate 0.91 with his published tjStuff+.

## Data

- Statcast pitch-by-pitch, regular seasons 2022–2026, via pybaseball 2.2.7:
  3.57 million pitches, 2,429–2,430 games per season, no duplicate pitch keys.
- Excluded: pitches with no pitch type, and knuckleballs, eephuses, pitchouts,
  position-player lobs and similar (about 0.3% a season).
- Data-quality report: `reports/data_quality.md`. One anomaly: 2026 `plate_x` /
  `plate_z` are reported at the middle of the plate rather than the front. No
  model feature uses plate location.

## Target

Run value per pitch from tjStuff+'s count-specific table (average 2024 run
value of each outcome in each count). Plate-appearance-ending events take the
event's value; other pitches take the pitch result's value. Outcomes the table
does not map (errors, strikeout double plays, catcher's interference; about
0.17% of pitches) are dropped.

## Features (10)

| Feature | Source |
|---|---|
| Velocity | `release_speed` |
| Spin rate | `release_spin_rate` |
| Extension | `release_extension` |
| Vertical acceleration | `az` |
| Horizontal acceleration, mirrored for left-handers | `ax` |
| Horizontal release position, mirrored for left-handers | `release_pos_x` |
| Release height | `release_pos_z` |
| Velocity difference from the primary fastball | derived |
| Vertical-acceleration difference from the primary fastball | derived |
| Horizontal-acceleration difference from the primary fastball (absolute) | derived |

The primary fastball is the pitcher's most-used FF/SI/FC that season; a
pitcher with none is referenced to their hardest pitch type. Not used: pitch
location, count, pitch-type label, batter, arm angle, spin axis.

## Model

- LightGBM regression, tjStuff+ v3.0 settings (1000 trees, learning rate 0.01,
  31 leaves), three seeds averaged. No hyperparameter tuning.
- Stuff+ = 100 − 10 × (xRV − mean) / sd, with mean and sd from 2026 pitches.
- Pitcher and pitch-type scores are shrunk toward the league mean:
  league + n / (n + k) × (raw − league), k = 12.7 pitches at pitcher level and
  3.7 at pitch-type level, from split-half reliability (0.95) of out-of-fold
  scores. Shrinkage matters only for very small samples.
- Artifacts: `models/stuffplus_v1.0.0.joblib` (not in git) and
  `models/stuffplus_v1.0.0.json` (features, scaling constants, shrinkage
  constants, library versions).

## Validation

Stuff+ in season N against the same pitchers in season N+1. Correlations with
results are sign-flipped so higher is better.

### The shipped feature set

| Split | Min pitches | Pitchers | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|---:|---:|
| Train 2022–23, score 2024 → 2025 | 100 | 517 | 0.867 | 0.222 | 0.239 | 0.222 |
| | 300 | 376 | 0.876 | 0.253 | 0.338 | 0.265 |
| | 500 | 311 | 0.885 | 0.278 | 0.385 | 0.287 |
| Train 2022–24, score 2025 → 2026 | 100 | 513 | 0.865 | 0.280 | 0.305 | 0.285 |
| | 300 | 396 | 0.869 | 0.318 | 0.397 | 0.327 |
| | 500 | 323 | 0.864 | 0.354 | 0.469 | 0.363 |

tjStuff+ v3.0 reports about 0.85 year-over-year stickiness (2023 → 2024,
trained on 2020–22).

### Alternatives tested against it

The candidate list, selection rule and confirmation rule were committed before
any candidate was run (`PREREGISTRATION.md`, which also records four
deviations). Development used 5-fold cross-validation by pitcher inside
2022–23 plus the 2024 → 2025 split. The 2025 → 2026 split was locked and run
once.

Development selection score (mean of next-wOBA and next-RV/100 correlations
over both development splits, minimum 100 pitches):

| Model | What changes from the baseline | Score | vs baseline |
|---|---|---:|---:|
| S6 | six variables: velocity, extension, location-neutral VAA and HAA, and their differences from the fastball | 0.2526 | +0.016 |
| J | balls in play valued by xwOBA instead of the outcome | 0.2493 | +0.012 |
| J+G+H | J's target plus both extra features | 0.2423 | +0.005 |
| **Baseline (shipped)** | – | 0.2369 | – |
| H | + seam-shifted-wake proxy (spin axis minus movement-implied axis) | 0.2362 | −0.001 |
| Baseline, different seeds | null control | 0.2358 | −0.001 |
| G | + batter on the pitcher's throwing side | 0.2331 | −0.004 |
| Component | swing, whiff, called-strike and contact models combined | 0.2230 | −0.014 |

No candidate's 95% interval against the baseline excluded zero. S6 was added
after the first six results were seen, at the owner's request.

One-shot confirmation on the locked split (minimum 100 pitches; difference from
the baseline with paired-bootstrap 95% CI):

| Model | Next wOBA | Next RV/100 | Stickiness | Confirmed |
|---|---:|---:|---:|---|
| J | −0.016 [−0.038, +0.007] | −0.014 [−0.036, +0.009] | +0.010 [−0.002, +0.023] | no |
| S6 | −0.003 [−0.056, +0.048] | 0.000 [−0.053, +0.049] | +0.021 [−0.002, +0.045] | no |

The rule required both result correlations to beat the baseline with intervals
excluding zero. Neither candidate did; J's development lead reversed. At 300
and 500 pitches S6 trails the baseline by 0.04–0.05 on both result
correlations (intervals include zero).

Also examined and dropped before pre-registration: approach angles on a 3×3
zone grid. The 18 grid angles span two dimensions (two principal components
explain 100.00% of their variance), so the grid collapses to its centre point.

Full tables: `reports/development_stage1.md`, `reports/confirmation.md`.

**What the evidence supports.** With about 500 pitchers per season, differences
between models of less than roughly ±0.02 to ±0.05 in these correlations cannot
be detected in one validation season. The tests here show that none of the
alternatives is clearly better. They do not show the alternatives are worse or
equivalent.

### Overfitting check

`overfit_check.py` trains on 2022–23 and compares scores for the training
pitches (in-sample) with scores for the same pitches from a model that never
saw that pitcher (out-of-pitcher), and with later seasons.

| | In-sample | Out-of-pitcher | 2024 | 2025 |
|---|---:|---:|---:|---:|
| Pitch-level R² | 0.0045 | 0.0007 | 0.0007 | 0.0007 |
| 2023 Stuff+ vs same-season RV/100 | 0.437 | 0.277 | | |
| 2023 Stuff+ vs 2024 RV/100 | 0.299 | 0.193 | | |

- The model **does memorise the pitchers it trains on**: in-sample pitch-level
  fit is about seven times the out-of-sample fit, and in-sample Stuff+ tracks a
  pitcher's own same-season results much more closely (0.44 against 0.28).
  In-sample scores are therefore inflated and are not used anywhere in the
  validation tables or leaderboards.
- Out-of-sample fit is stable: R² is the same for unseen pitchers and for the
  next two seasons, so nothing degrades over time.
- Memorised pitchers are predicted **better** the following season (0.30
  against 0.19), because a pitcher's past results carry real information
  (command, deception) that the ten physical inputs do not. A score for a
  pitcher the model has trained on is therefore partly a record of how that
  pitcher has performed, not purely a grade of the pitch's physical traits.
  The next-season validation tables above are of this kind, since most
  pitchers in the scored season were also in the training seasons.

## 2026 outputs

- `leaderboards/pitchers_2026.csv`: 720 pitchers with at least 100 pitches.
- `leaderboards/pitch_types_2026.csv`: 2,557 pitcher × pitch-type rows with at
  least 50 pitches of the type.
- Scores are **out-of-fold**: each pitcher is scored by a model trained without
  that pitcher (5 folds by pitcher across 2022–2026), so a pitcher's own
  results cannot leak into their score. Out-of-fold and in-sample pitcher
  scores correlate 0.95.
- Grades are on the 20–80 scale: 50 + 10 z among qualified pitchers (overall
  and within role), or among qualified pitches of the same type, clipped.
- The pitcher-level sd of Stuff+ is 4.7, so 110 is about two standard
  deviations above average.

### Starters and relievers

A pitcher is a starter if he threw his team's first pitch in at least half of
his games (so openers count as starters).

| Role | Pitchers | Pitches | Stuff+ (pitch-weighted) | Stuff+ (mean) | sd |
|---|---:|---:|---:|---:|---:|
| RP | 491 | 308,767 | 100.9 | 100.6 | 4.8 |
| SP | 229 | 395,226 | 99.3 | 99.1 | 4.3 |

`grade_in_role` in the pitcher leaderboard grades each pitcher against his own
role.

### Park check

Same pitchers' Stuff+ at a park minus everywhere else (at least 50 pitches in
both), 2026:

| Park | Stuff+ difference | SE | Pitchers |
|---|---:|---:|---:|
| COL | −5.4 | 0.3 | 106 |
| ATH | −1.1 | 0.2 | 114 |
| DET | −1.1 | 0.2 | 110 |
| KC | −1.0 | 0.2 | 107 |
| … 22 parks between −0.8 and +1.1 | | | |
| MIA | +1.7 | 0.2 | 102 |
| SD | +1.7 | 0.2 | 113 |
| HOU | +2.4 | 0.2 | 113 |
| TB | +2.5 | 0.2 | 108 |

Coors Field lowers Stuff+ by about 5.4 points, more than one pitcher-level
standard deviation; pooled over 2022–2026 it is −5.7. Scores are **not** park
adjusted. All parks: `reports/park_check_2026.csv`,
`reports/park_check_2022_2026.csv`.

## Limitations

- **Did not beat the baseline.** This model's feature set is tjStuff+ v3.0's.
  Its value over using tjStuff+ directly is the open pipeline and evidence, not
  accuracy.
- **No location or sequencing.** Command, pitch mix, tunnelling and count
  leverage are invisible to it. It grades the pitch, not the pitcher.
- **Reliever velocity bias.** Relievers throw harder in short outings and score
  1.6 points higher on average. A reliever's Stuff+ would not carry over to a
  starting role unchanged; use `grade_in_role` for within-role comparisons.
- **Handedness gap.** Left-handers average 98.0 and right-handers 101.0. The
  model has no batter-side feature, so this reflects velocity and shape
  differences, not platoon effects.
- **Unusual arm slots.** Sidearm and submarine pitchers are rare in training
  and are judged mostly by how their release point and movement compare with
  conventional pitchers'; expect larger errors for them.
- **Pitch classification noise.** Fastball references and pitch-type
  leaderboards depend on Statcast labels. The sweeper share nearly doubled from
  2022 (4.5%) to 2026 (8.3%) as sliders fell; part of that is relabelling. A
  reclassified pitch moves between pitch-type leaderboards.
- **Park and tracking calibration.** Coors costs about 5 points and several
  other parks ±1–2.5. Rockies pitchers are understated and scores are not
  adjusted. Differences between parks may be partly Hawk-Eye calibration
  rather than air.
- **League drift.** On the 2026 scale the league average was 98.5 in 2022 and
  100 in 2026. Compare pitchers within a season, not across seasons.
- **2022 is the only season before the 2023 rule changes** (pitch clock, shift
  limits, larger bases), and the run-value table is from 2024.
- **Short training window.** Five seasons. The published tjStuff+ model trains
  on earlier seasons this project did not use.
- **Small samples at the top.** A 100-pitch minimum admits pitchers with a
  handful of outings; the 2026 number one has 103 pitches. Shrinkage moves such
  scores by only a point or two. Filter on `pitches` for a stricter board.
- **Memorisation of known pitchers.** The model fits its training pitchers
  about seven times better than unseen ones (see the overfitting check). The
  saved model was trained on every 2022–2026 pitcher, so its score for a
  returning pitcher partly reflects that pitcher's past results; the
  leaderboards use out-of-fold scores to avoid this. Scores from the saved
  model and from the leaderboards are not interchangeable.
- **Predictive power is modest.** Correlation with next-season wOBA is 0.22–0.35
  depending on the pitch minimum. Stuff+ is one input, not a projection.

## Reproducing

`README.md` lists the commands. Every run writes its arguments, seeds and
library versions to a JSON under `results/`. Python 3.9, versions pinned in
`requirements.txt`.
