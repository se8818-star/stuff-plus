# Stuff+ ablation: location-neutral 9-point VAA/HAA vs tjStuff+ v3.0

Tests whether approach angles computed on a fixed 3×3 zone grid improve on
tjStuff+ v3.0's feature set, using his validation design on 2022–2026 data
(the post-2023-rule-change era plus one prior season):

| Fold | Train | Score | Validate against |
|---|---|---|---|
| 1 (default) | 2022–23 | 2024 | 2025 |
| 2 | 2022–24 | 2025 | 2026 |

## Run it

```bash
pip install pandas numpy scikit-learn pyarrow pybaseball lightgbm
python fetch_data.py          # 2022-2026 Statcast, cached to data/ (slow, once)
python run_ablation.py        # all variants, ~a few minutes each with LightGBM
python run_ablation.py --seeds 3   # average 3 model seeds (recommended for final numbers)
python run_ablation.py --train 2022 2023 2024 --eval 2025   # fold 2
python run_ablation.py --synthetic # smoke test, no data needed
```

If LightGBM isn't installed it falls back to sklearn's histogram GBM with
matched settings. Results land in `results/ablation_<eval year>.json`.

## How the 9-point angles work (`physics.py`)

1. Rewind Statcast's 9-parameter fit to the release plane (60.5 − extension).
2. Hold release position, release y-velocity, and all three accelerations fixed.
3. Re-aim the pitch at each of 9 fixed targets (centers of zones 1–9 on a fixed
   1.5–3.5 ft, 17" zone; mirrored so +x is arm side for both hands).
4. Compute VAA/HAA where it crosses the front of the plate.

Re-aiming a pitch at its own actual location reproduces its real VAA/HAA
exactly, so the physics is checked against Statcast's own fit.

## Variants (`features.py`)

| Name | Features |
|---|---|
| A_tj_v3 | tjStuff+ v3.0 notebook set: velo, spin, extension, az, ax, release x/z, speed/az/ax diffs vs primary FB |
| B_plus_center | A + VAA/HAA at zone center |
| B9_plus_grid9 | A + all 9 VAA + 9 HAA |
| C_grid9_no_accel | B9 without az/ax |
| D_grid9_angles_only | Your proposal: velo, spin, extension, speed diff, VAA/HAA diffs + 9-point grid. No az/ax, no release, no arm angle |
| D1_center_angles_only | D with one center point instead of 9 |
| E_plus_arm_angle | A + arm angle |
| F_plus_spin_axis | A + spin axis (in the v3.0 article, not in the notebook) |

## Reading the output

- **stickiness**: r(year N Stuff+, year N+1 Stuff+). TJ reports 0.85 for 2023→2024
  (trained on 2020–22), so treat that as a rough reference, not an exact target.
- **next_woba / next_rv100**: correlation with next-season results, sign-flipped so higher = better.
- **next_k_bb**: correlation with next-season K−BB%.
- **CI vs A**: paired bootstrap over pitchers. If the interval contains 0, the
  variant isn't distinguishable from the baseline.
- **grid PCA**: cumulative variance of the 18 grid angles. Under the
  constant-acceleration fit, VAA depends only on the target's height and HAA
  only on its width, so expect ~2–3 components to explain essentially all of it.

## Notes on fidelity to tjStuff+ v3.0

- Target: same count-specific run-value table (`run_values.csv`, from his repo).
- Release x/z use Statcast `release_pos_*`; his notebook uses MLB API `x0/z0`.
- RobustScaler is omitted (no effect on tree models).
- His notebook passes `subsample=0.8` without `subsample_freq`, so LightGBM
  doesn't actually bag rows; kept as-is for a faithful baseline.

`run_values.csv` © 2024 tnestico, MIT License (github.com/tnestico/tjstuff_plus).
