# Stuff+ v1.1.0 — model card

A pitch-quality model in the style of tjStuff+ v3.0, trained on 2022–2026
Statcast data, with a pre-registered attempt to improve on tjStuff+'s feature
set. **That attempt did not succeed: none of the seven alternatives tested
beat the tjStuff+ v3.0 feature set on held-out data, so this release ships that
feature set.** What this repository adds is the reproducible pipeline, the
validation evidence, shrinkage, the 2026 leaderboards, and a second model that
cannot remember the pitcher it is grading.

## Two models

Both use the same ten inputs and settings. They differ in whether the model was
trained on the pitcher it is scoring.

| | Stuff+ | Stuff+ (no remembering) |
|---|---|---|
| What it is | one model trained on every pitcher | five fold models, each trained without a fifth of the pitchers |
| Scoring a pitcher | the one model | only a fold model that never saw him; a new pitcher gets the mean of all five |
| What the score means | physical traits, plus what the model has learned about how that pitcher's pitches fare | physical traits only |
| Next-season wOBA / RV/100 correlation (four forward splits, min 100 pitches) | 0.271 / 0.272 | 0.236 / 0.235 |
| Stickiness | 0.869 | 0.862 |
| Safe to use on | seasons after its training data only | any season |
| 2026 leaderboard from | a model trained on 2022–2025 | the 2022–2026 fold models |
| Files | `models/stuffplus_v1.1.0.*`, `leaderboards/*_2026.csv` | `models/stuffplus_no_remembering_v1.1.0.*`, `leaderboards/*_2026_no_remembering.csv` |

Use Stuff+ when you want the better forecast, and Stuff+ (no remembering) when
you want a grade that cannot be influenced by the pitcher's past results.
`leaderboards/both_models_2026.csv` has both scores for every pitcher.

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
- Stuff+ = 100 − 10 × (xRV − mean) / sd, with mean and sd from 2026 pitches
  (each model has its own constants, in its JSON).
- Pitcher and pitch-type scores are shrunk toward the league mean:
  league + n / (n + k) × (raw − league), k = 12.7 pitches at pitcher level and
  3.7 at pitch-type level, from split-half reliability (0.95) of
  out-of-fold scores. Shrinkage matters only for very small samples.
- `python -m pipeline.score file.parquet out.csv` scores new Statcast pitches with
  Stuff+; add `--no-remembering` for the other model. The `.joblib` files are
  not in git; `python -m pipeline.final` rebuilds them.

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
any candidate was run (`docs/PREREGISTRATION.md`, which also records four
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

The locked confirmation scored pitchers with a model trained on their earlier
seasons, which can favor features that identify a pitcher (A's release point
and extension); E1's out-of-pitcher comparison, below, is the check that A
still leads without that.

Also examined and dropped before pre-registration: approach angles on a 3×3
zone grid. The 18 grid angles span two dimensions (two principal components
explain 100.00% of their variance), so the grid collapses to its centre point.

Full tables: `reports/development_stage1.md`, `reports/confirmation.md`.

**What the evidence supports.** With about 500 pitchers per season, differences
between models of less than roughly ±0.02 to ±0.05 in these correlations cannot
be detected in one validation season. The tests here show that none of the
alternatives is clearly better. They do not show the alternatives are worse or
equivalent.

### What remembering is worth (exploratory)

`experiments/crossfit_test.py` (plan E2 in `docs/EXPLORATORY.md`) trains both ways on the same
seasons and scores the next one, for four forward splits (scoring 2022–2025,
training from 2020). Stuff+ (no remembering) minus Stuff+:

| Min pitches | Next wOBA | Next RV/100 | Next K−BB% | Stickiness |
|---|---:|---:|---:|---:|
| 100 | −0.036 [−0.053, −0.020] | −0.037 [−0.055, −0.020] | −0.034 [−0.048, −0.020] | −0.007 [−0.016, +0.002] |
| 300 | −0.036 [−0.059, −0.017] | −0.038 [−0.063, −0.018] | −0.032 [−0.049, −0.016] | −0.008 [−0.016, +0.001] |
| 500 | −0.029 [−0.053, −0.008] | −0.032 [−0.059, −0.010] | −0.023 [−0.039, −0.007] | −0.010 [−0.019, −0.001] |

Not remembering the pitcher costs about 0.037 in next-season correlation
(0.272 → 0.235), in every split, with intervals that exclude zero. The
remembered part is persistent, pitcher-specific signal the ten inputs cannot
see. It may include command and deception, but also park, team defense and
catcher framing, so it is not all the pitcher's own skill.
The two scores correlate 0.93–0.97 across pitchers.

### Overfitting check

`experiments/overfit_check.py` trains on 2022–23 and compares scores for the training
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
  validation tables or leaderboards: Stuff+ only scores seasons after its
  training data, and Stuff+ (no remembering) never scores a pitcher it saw.
- Out-of-sample fit is stable: R² is the same for unseen pitchers and for the
  next two seasons, so nothing degrades over time.
- Memorised pitchers are predicted **better** the following season (0.30
  against 0.19), because a pitcher's past results carry persistent,
  pitcher-specific signal the ten physical inputs do not (possibly command and
  deception, but also park, team defense and catcher framing). A score for a
  pitcher the model has trained on is therefore partly a record of how that
  pitcher has performed, not purely a grade of the pitch's physical traits.
  The next-season validation tables above are of this kind, since most
  pitchers in the scored season were also in the training seasons.

## 2026 outputs

| File | Contents |
|---|---|
| `leaderboards/pitchers_2026.csv` | Stuff+, 720 pitchers with at least 100 pitches |
| `leaderboards/pitch_types_2026.csv` | Stuff+, 2,557 pitcher × pitch-type rows with at least 50 pitches |
| `leaderboards/pitchers_2026_no_remembering.csv` | Stuff+ (no remembering), 720 pitchers |
| `leaderboards/pitch_types_2026_no_remembering.csv` | Stuff+ (no remembering), 2,557 rows |
| `leaderboards/both_models_2026.csv` | both scores and ranks side by side |

- Grades are on the 20–80 scale: 50 + 10 z among qualified pitchers (overall
  and within role), or among qualified pitches of the same type, clipped.
- The pitcher-level sd is 4.6 for Stuff+ and 4.7 for Stuff+ (no remembering), so
  110 is about two standard deviations above average.

### How the two leaderboards differ

- Pitcher scores correlate 0.95 (0.95 for the 596 pitchers with earlier seasons,
  0.97 for the 124 who debuted in 2026). The median difference is 0.6 points and
  3% of pitchers differ by more than 3. 19 of the top 25 overlap (at least 500 pitches).
- The largest gap overall is Danny Coulombe (385 pitches): 82.6 without
  remembering, 102.6 with, a difference of 20 points. The next largest is under 8.
- Among pitchers with at least 500 pitches, Stuff+ is furthest above the
  physical grade for Hoby Milner (89.4 → 96.5), Devin Williams (99.8 → 106.0)
  and Tyler Wells (89.2 → 95.2).
- It is furthest below for Caleb Thielbar (106.4 → 100.9), Tanner Bibee
  (96.2 → 92.4) and Mason Miller (116.4 → 112.7).

Differences reflect what Stuff+ has learned about a pitcher from earlier
seasons and, to a lesser extent, that its leaderboard model was trained without
2026.

Top five with at least 500 pitches:

| Stuff+ | Role | Pitches | Score |
|---|---|---:|---:|
| Trevor Megill | RP | 873 | 115.2 |
| Tyler Rogers | RP | 1,013 | 114.9 |
| Jacob Misiorowski | SP | 2,644 | 113.5 |
| Chase Burns | SP | 2,563 | 113.1 |
| Jhoan Duran | RP | 947 | 113.1 |

| Stuff+ (no remembering) | Role | Pitches | Score |
|---|---|---:|---:|
| Mason Miller | RP | 1,148 | 116.4 |
| Chase Burns | SP | 2,563 | 115.6 |
| Jacob Misiorowski | SP | 2,644 | 114.4 |
| Trevor Megill | RP | 873 | 114.4 |
| Edgardo Henriquez | RP | 968 | 112.7 |

### Starters and relievers

A pitcher is a starter if he threw his team's first pitch in at least half of
his games (so openers count as starters). Mean Stuff+ of qualified pitchers:

| Role | Pitchers | Stuff+ | Stuff+ (no remembering) |
|---|---:|---:|---:|
| RP | 491 | 100.6 | 100.6 |
| SP | 229 | 98.9 | 99.0 |

`grade_in_role` in the pitcher leaderboards grades each pitcher against his own
role.

### Park check

Same pitchers' score at a park minus everywhere else (at least 50 pitches in
both), 2026:

| Park | Stuff+ | Stuff+ (no remembering) |
|---|---:|---:|
| COL | -5.3 | -5.4 |
| ATH | -1.0 | -1.1 |
| DET | -1.1 | -1.1 |
| KC | -1.0 | -1.0 |
| … 22 parks between -0.8 and +1.1 (no remembering) | | |
| MIA | +1.8 | +1.7 |
| SD | +1.7 | +1.7 |
| HOU | +2.4 | +2.4 |
| TB | +2.5 | +2.5 |

Coors Field lowers both scores by about 5.4 points, more than one pitcher-level
standard deviation. Scores are **not** park adjusted. All parks:
`reports/park_check_2026.csv`, `reports/park_check_2026_no_remembering.csv`,
and pooled over 2022–2026 in `reports/park_check_2022_2026_no_remembering.csv`.

## Six-variable comparison model (S6)

For side-by-side comparison, S6 is also trained on 2022–2026, scored without
remembering the pitcher, and published next to Stuff+ (no remembering): `models/stuffplus_s6_v1.0.0.json`,
`leaderboards/pitchers_2026_s6.csv`, `leaderboards/pitch_types_2026_s6.csv`,
`leaderboards/comparison_2026.csv`, and `reports/comparison_a_vs_s6.md`. It is
**not** the recommended model: it was not confirmed as better than A, and it
trails A at 300 and 500 pitches.

- Pitcher scores from S6 and Stuff+ (no remembering) correlate 0.80; the median difference is
  1.7 points and 9% of pitchers differ by more than 5.
- They disagree most on unusual arm slots and cutters. S6 scores submarine and
  sidearm relievers far lower (Tyler Rogers 109.7 → 90.1), because it has no
  release-position input.
- S6 is less affected by Coors (−2.4 against −5.4) and slightly more tied to
  velocity (r = 0.69 against 0.63).

### E1: out-of-pitcher comparison on 2020–2026 (exploratory)

`experiments/pitcher_holdout.py` (plan in `docs/EXPLORATORY.md`) scores every 2020–2026 pitch
with a model that never saw that pitcher (5 folds by pitcher, 4.5 million
pitches, 1,863 pitchers) and averages six season pairs. S6 minus A:

| Min pitches | Next wOBA | Next RV/100 | Stickiness |
|---|---:|---:|---:|
| 100 | −0.022 [−0.048, +0.003] | −0.020 [−0.046, +0.005] | +0.011 [−0.002, +0.025] |
| 300 | −0.049 [−0.080, −0.022] | −0.046 [−0.077, −0.019] | +0.009 [−0.005, +0.022] |
| 500 | −0.030 [−0.063, −0.000] | −0.028 [−0.060, +0.002] | +0.013 [−0.001, +0.027] |

With memory of the pitcher removed, A predicts next-season results better than
S6: the gap is borderline at 100 pitches and excludes zero at 300. S6's apparent
lead on 87 newly debuted pitchers did not hold up on the larger sample. S6 is
about 0.01 stickier. This was run on data already examined, so it is supporting
evidence for shipping A, not a confirmation.

## Limitations

- **Did not beat the baseline.** This model's feature set is tjStuff+ v3.0's.
  Its value over using tjStuff+ directly is the open pipeline and evidence, not
  accuracy.
- **No location or sequencing.** Command, pitch mix, tunnelling and count
  leverage are invisible to it. It grades the pitch, not the pitcher.
- **Reliever velocity bias.** Relievers throw harder in short outings and score
  about 1.6 points higher on average. A reliever's Stuff+ would not carry over to a
  starting role unchanged; use `grade_in_role` for within-role comparisons.
- **Handedness gap.** Left-handers average 98.0 and right-handers 100.9
  (the same in both models). The
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
  handful of outings; one pitcher near the top of both boards has 103 pitches. Shrinkage moves such
  scores by only a point or two. Filter on `pitches` for a stricter board.
- **Stuff+ remembers pitchers.** It fits its training pitchers about seven times
  better than unseen ones, so it must only score seasons after its training
  data; `pipeline/score.py` warns otherwise. A returning pitcher's Stuff+ partly reflects
  his past results. Use Stuff+ (no remembering) for a grade free of that, at a
  cost of about 0.037 in next-season correlation.
- **The Stuff+ leaderboard and the saved Stuff+ model differ in training data.**
  The 2026 leaderboard comes from a model trained on 2022–2025; the saved model
  adds 2026 and is meant for 2027.
- **Predictive power is modest.** Correlation with next-season wOBA is 0.22–0.35
  depending on the pitch minimum. Stuff+ is one input, not a projection.

## Reproducing

`README.md` lists the commands. Every run writes its arguments, seeds and
library versions to a JSON under `results/`. Python 3.9, versions pinned in
`requirements.txt`.
