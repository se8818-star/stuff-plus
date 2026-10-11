# Stuff+ (v1.0.0)

A reproducible Stuff+ pipeline on 2022–2026 Statcast data, built around a
pre-registered test of whether anything beats the tjStuff+ v3.0 feature set on
next-season prediction. **Nothing did**, so v1.0.0 ships that feature set. See
`MODEL_CARD.md` for the model, the validation tables and the limitations.

| Split | Train | Score | Validate against | Role |
|---|---|---|---|---|
| Inner CV | 2022–23, by pitcher | 2022 | 2023 | development |
| Fold 1 | 2022–23 | 2024 | 2025 | development |
| Fold 2 | 2022–24 | 2025 | 2026 | one-shot confirmation |

## Run it

```bash
bash setup.sh                 # .venv + pinned requirements.txt + synthetic smoke test
source .venv/bin/activate
pytest -q
python fetch_data.py          # 2022-2026 Statcast, cached to data/ (resumable, ~15 min)
python data_quality.py        # reports/data_quality.md
python baseline_checks.py     # sanity checks on the tjStuff+ v3.0 baseline
python develop.py             # all candidates: inner CV + fold 1 -> reports/development_stage1.md
python confirm.py --unlock-fold2 && python report_confirmation.py   # ran once; refuses to rerun
python final.py               # models/, leaderboards/, park check
python run_ablation.py --synthetic   # smoke test, no data needed
```

`run_ablation.py` is the original feature-set ablation harness (variants A–J,
grid variants); the pre-registered comparison lives in `models.py`,
`develop.py` and `confirm.py`. `tune.py` implements the regularization search
that was planned and then deliberately not run.

Outputs: `leaderboards/pitchers_2026.csv`, `leaderboards/pitch_types_2026.csv`,
`models/stuffplus_v1.0.0.json` (the `.joblib` beside it is not in git).

If LightGBM isn't installed `run_ablation.py` falls back to sklearn's histogram
GBM. Every run records its arguments, seeds and library versions under
`results/`. Versions are pinned for Python 3.9; on Apple Silicon LightGBM needs
`brew install libomp` (`setup.sh` does this). CI runs only `pytest` and the
synthetic smoke test — it never downloads data.

## How the 9-point angles work (`physics.py`)

1. Rewind Statcast's 9-parameter fit to the release plane (60.5 − extension).
2. Hold release position, release y-velocity, and all three accelerations fixed.
3. Re-aim the pitch at each of 9 fixed targets (centers of zones 1–9 on a fixed
   1.5–3.5 ft, 17" zone; mirrored so +x is arm side for both hands).
4. Compute VAA/HAA where it crosses the front of the plate.

Re-aiming a pitch at its own actual location reproduces its real VAA/HAA
exactly, so the physics is checked against Statcast's own fit.

## Ablation variants (`features.py`, used by `run_ablation.py`)

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
