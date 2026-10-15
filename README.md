# Stuff+ (v1.1.0)

A reproducible Stuff+ pipeline on 2022–2026 Statcast data, built around a
pre-registered test of whether anything beats the tjStuff+ v3.0 feature set on
next-season prediction. **Nothing did**, so this release ships that feature set,
as two models:

- **Stuff+**: one model trained on every pitcher. More predictive, because it
  remembers how each pitcher's pitches have fared; only used on seasons after
  its training data.
- **Stuff+ (no remembering)**: a pitcher is only scored by a model that never
  saw him. A pure grade of the physical pitch, about 0.037 lower in next-season
  correlation.

See `MODEL_CARD.md` for the models, the validation tables and the limitations.

| Split | Train | Score | Validate against | Role |
|---|---|---|---|---|
| Inner CV | 2022–23, by pitcher | 2022 | 2023 | development |
| Fold 1 | 2022–23 | 2024 | 2025 | development |
| Fold 2 | 2022–24 | 2025 | 2026 | one-shot confirmation |

## Layout

| Folder | What's in it |
|---|---|
| `stuffplus/` | the model code: physics, features, candidate models, the ablation harness |
| `pipeline/` | the pre-registered steps, from fetching data to the shipped models and scoring |
| `experiments/` | exploratory analyses outside the pre-registered comparison |
| `webapp/` | the local leaderboard website (`python serve.py`) |
| `docs/` | pre-registration, frozen candidates, exploratory plans |
| `models/`, `leaderboards/`, `reports/` | shipped outputs |

Run every command from the repo root. Scripts in the three code folders run as
modules (`python -m pipeline.final`), not by path. The documents in `docs/` were
committed before the files moved, so they name scripts without their folder.

## Run it

```bash
bash setup.sh                 # .venv + pinned requirements.txt + synthetic smoke test
source .venv/bin/activate
pytest -q
python -m pipeline.fetch_data           # 2022-2026 Statcast, cached to data/ (resumable, ~15 min)
python -m pipeline.data_quality         # reports/data_quality.md
python -m pipeline.baseline_checks      # sanity checks on the tjStuff+ v3.0 baseline
python -m pipeline.develop              # all candidates: inner CV + fold 1 -> reports/development_stage1.md
python -m pipeline.confirm --unlock-fold2 && python -m pipeline.report_confirmation   # ran once; refuses to rerun
python -m pipeline.final                # both models, leaderboards/, park checks
python -m pipeline.score new_season.parquet scores.csv                   # Stuff+ on a later season
python -m pipeline.score any_season.parquet scores.csv --no-remembering  # Stuff+ (no remembering)
python -m experiments.compare_models    # S6 comparison: leaderboards/comparison_2026.csv
python -m experiments.overfit_check     # reports/overfit_check.md
python -m experiments.holdout_check     # reports/holdout_2025_2026.md
python -m pipeline.fetch_data 2020 2021 && python -m experiments.pitcher_holdout   # reports/pitcher_holdout.md (exploratory)
python -m experiments.crossfit_test     # reports/crossfit_test.md: what remembering is worth (exploratory)
python -m experiments.tjstats_comparison.fetch   # then .build and .analyze: reports/tjstats_comparison.md (exploratory)
python -m stuffplus.run_ablation --synthetic   # smoke test, no data needed
```

`stuffplus/run_ablation.py` is the original feature-set ablation harness (variants A–J,
grid variants); the pre-registered comparison lives in `stuffplus/models.py`,
`pipeline/develop.py` and `pipeline/confirm.py`. `pipeline/tune.py` implements the regularization search
that was planned and then deliberately not run.

Outputs: `leaderboards/pitchers_2026.csv` and `pitch_types_2026.csv` (Stuff+),
the same with `_no_remembering`, `both_models_2026.csv`, and
`models/stuffplus_v1.1.0.json` / `stuffplus_no_remembering_v1.1.0.json` (the
`.joblib` files beside them are not in git; `pipeline/final.py` rebuilds them). The S6
comparison model's files (v1.0.0) were built by an earlier `pipeline/final.py --model S6`.

If LightGBM isn't installed `stuffplus/run_ablation.py` falls back to sklearn's histogram
GBM. Every run records its arguments, seeds and library versions under
`results/`. Versions are pinned for Python 3.9; on Apple Silicon LightGBM needs
`brew install libomp` (`setup.sh` does this). CI runs only `pytest` and the
synthetic smoke test — it never downloads data.

## Leaderboard website

```bash
pip install -r requirements.txt
python serve.py               # http://127.0.0.1:8000
```

The first run builds the leaderboards (`webapp/build.py`) and caches them in
`models/stuffplus_site_*.joblib`; run `python serve.py --rebuild` after new data or a
new model.

- **With `data/` and the shipped models** (`python -m pipeline.final`): the site shows
  **Stuff+ (no remembering)**. 2026 shows the published
  `leaderboards/*_2026_no_remembering.csv` numbers exactly, and every other season in
  `data/` is scored the same way (each pitcher by a fold model that never saw him, same
  shrinkage); the grader uses those fold models too. They come straight from
  `models/stuffplus_no_remembering_v1.1.0.joblib`, so `python -m webapp.folds` is only
  needed for a v1.0.0 model.
  The **Model** menu in the header switches the leaderboards, the grader and My pitchers
  between every frozen model in `models/` (the newest version of each):
  Stuff+ (no remembering), the default; Stuff+ (remembering), `models/stuffplus_v1.1.0`,
  whose 2026 board is its published `leaderboards/*_2026.csv` and whose other seasons are
  in-sample; and S6, `models/stuffplus_s6_v1.0.0`, with `leaderboards/*_2026_s6.csv`.
  Each gets its own cached bundle, built on first run (a few minutes each). The choice is
  remembered in the browser; the API takes `?model=<id>`, and `/api/models` lists them.
- **With `data/` only**, or with `--variant B_plus_center` / `--train 2022 2023 2024`,
  it trains that feature-set variant instead.
- **Without `data/`** it serves a synthetic demo, with a banner saying so.

Displayed Stuff+ is the model's own scale, as tjStuff+ shows it: each pitch is 100 ± 10,
and a pitcher's score is the average of his pitches, so pitcher scores have an SD of about 5.
`python serve.py --rebuild --spread keep` instead stretches each pitch type, and the pitcher
board, to an SD of 10 among qualified rows each season (min 100 pitches for pitchers, 50 for
pitch types), keeping each group's own mean; `--spread center` also centres every pitch type
at 100. Grades and percentiles are the same either way, and `stuff_plus_model` in the API
always keeps the model's / published number.

- **Pitchers**: pitcher-season Stuff+, 20-80 grade, Stuff+ per pitch type, filters
  for hand, SP/RP (same rule as `pipeline/final.py`), min pitches, CSV export.
- **Pitches**: one row per pitcher × pitch type with velo, spin, IVB/HB, extension.
- **Explore**: scatter any two pitch characteristics, coloured by Stuff+.
- **Grade a pitcher**: type in an arsenal (velo, spin, IVB, HB, release, extension,
  usage) and get Stuff+, a 20-80 grade, and a percentile per pitch as you type. You can
  start from any MLB pitcher's arsenal and tweak it.
- **My pitchers**: name an arsenal in the grader and save it. Saved pitchers are listed
  with their live Stuff+, grade, percentile and per-pitch scores, and open back in the
  grader for editing. They're stored in `custom_pitchers.json` at the repo root
  (gitignored), so they survive restarts and rebuilds.

The grader builds Statcast-shaped rows from the inputs and runs them through
`features.add_features`, so it uses exactly the same features as the model. An arsenal
loaded from an MLB pitcher (and a saved pitcher started from one) is scored by the fold
model that never saw him, like his leaderboard number; anything else by the average of
the five fold models. Movement
uses Savant conventions (inches, y = 50 ft to the plate, matching Statcast pfx, HB positive to the arm side).
A pitch's grade is the model's grade for the *average* pitch you describe, which can
differ slightly from a leaderboard number (the mean of every individual pitch's grade).

## How the 9-point angles work (`stuffplus/physics.py`)

1. Rewind Statcast's 9-parameter fit to the release plane (60.5 − extension).
2. Hold release position, release y-velocity, and all three accelerations fixed.
3. Re-aim the pitch at each of 9 fixed targets (centers of zones 1–9 on a fixed
   1.5–3.5 ft, 17" zone; mirrored so +x is arm side for both hands).
4. Compute VAA/HAA where it crosses the front of the plate.

Re-aiming a pitch at its own actual location reproduces its real VAA/HAA
exactly, so the physics is checked against Statcast's own fit.

## Ablation variants (`stuffplus/features.py`, used by `stuffplus/run_ablation.py`)

| Name | Features |
|---|---|
| A_tj_v3 | tjStuff+ v3.0 notebook set: velo, spin, extension, az, ax, release x/z, speed/az/ax diffs vs primary FB |
| B_plus_center | A + VAA/HAA at zone center |
| B9_plus_grid9 | A + all 9 VAA + 9 HAA |
| C_grid9_no_accel | B9 without az/ax |
| D_grid9_angles_only | Your proposal: velo, spin, extension, speed diff, VAA/HAA diffs + 9-point grid. No az/ax, no release, no arm angle |
| D1_center_angles_only | D with one center point instead of 9 |
| E_plus_arm_angle | A + arm angle |
| F_plus_spin_axis | A + spin axis, mirrored for LHP (in the v3.0 article, not in the notebook) |
| G_plus_same_side | A + batter handedness as same-side vs opposite-side |
| H_plus_ssw | A + seam-shifted-wake proxy: observed spin axis minus the axis implied by pfx_x/pfx_z |
| I_recency_weighted | A's features; each training season weighted 0.5 ** (seasons before the latest) |
| J_expected_target | A's features; balls in play valued by xwOBA converted to runs for the count |
| C1_center_no_accel | C with the grid collapsed to its centre point |
| A50_tj_v3_y50 | Reference: A with release x/z from the fit at y=50 (TJ's MLB-API definition) |

## Reading the output

- **stickiness**: r(year N Stuff+, year N+1 Stuff+). TJ reports 0.85 for 2023→2024
  (trained on 2020–22), so treat that as a rough reference, not an exact target.
- **next_woba / next_rv100**: correlation with next-season results, sign-flipped so higher = better.
- **next_k_bb**: correlation with next-season K−BB%.
- **CI vs A**: paired bootstrap over pitchers, per fold and for the mean of the
  folds (a pitcher in both folds is resampled into both together). If the
  interval contains 0, the variant isn't distinguishable from the baseline.
- **pitch-type stickiness**: the same year-over-year correlation per pitcher ×
  pitch type (FF, SI, FC, SL, ST, CU, KC, CH, FS).
- **grid PCA**: cumulative variance of the 18 grid angles. Under the
  constant-acceleration fit, VAA depends only on the target's height and HAA
  only on its width, so expect ~2–3 components to explain essentially all of it.

## Notes on fidelity to tjStuff+ v3.0

- Target: same count-specific run-value table (`run_values.csv`, from his repo).
- Release x/z use Statcast `release_pos_*`; his notebook uses MLB API `x0/z0`
  (the fit at y=50). `A50_tj_v3_y50` is that version; it tracks his published
  2024 scores more closely (r = 0.94 vs 0.91) but validates slightly worse, so
  the release-point version is the baseline to beat.
- RobustScaler is omitted (no effect on tree models).
- Outcomes his `event_dict` maps to None (`field_error`, `strikeout_double_play`,
  `catcher_interf`, ...) have no target and are dropped, as in his notebook.
- His notebook inner-joins the primary-fastball table, which drops pitcher-seasons
  with no FF/SI/FC (the fallback code after it never runs). Here those rows are
  kept, referenced to the pitcher's hardest pitch type, and flagged `fb_fallback`.
- His notebook passes `subsample=0.8` without `subsample_freq`, so LightGBM
  doesn't actually bag rows; kept as-is for a faithful baseline.

`run_values.csv` © 2024 tnestico, MIT License (github.com/tnestico/tjstuff_plus).
