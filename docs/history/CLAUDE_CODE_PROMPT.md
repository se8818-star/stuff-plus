I'm building a public Stuff+ pitch-quality model to rival tjStuff+ v3.0 (Thomas Nestico, github.com/tnestico/tjstuff_plus, MIT license). This folder holds an ablation harness I started:
- physics.py: location-neutral VAA/HAA on a fixed 3x3 zone grid, computed by re-aiming each pitch's Statcast trajectory with release, velocity, and accelerations held fixed
- features.py: tjStuff+ v3.0 feature replication plus variants A–F
- run_ablation.py: train, score a season, validate against the next season
- fetch_data.py: pybaseball Statcast download, 2022–2026 by default
- synthetic.py: fake data for smoke tests
- setup.sh: venv + dependencies + smoke test
Read README.md and all the code first. Only the synthetic path has ever been run. Expect bugs in the real-data path.

DATA WINDOW: use 2022–2026 regular seasons only. 2025 and 2026 are complete.

GOAL: a Stuff+ model that measurably beats tjStuff+ v3.0's feature set on out-of-sample stickiness and next-season prediction, built as a reproducible pipeline that also produces leaderboards.

WORKING RULES
- If this folder isn't a git repo yet, initialize one, make an initial commit, and help me create a private GitHub repo with `gh` (ask me before creating it). Then work on a branch, commit after each phase with a clear message, and open a PR at the end.
- After each phase, give me a short results table and STOP for my go-ahead.
- If a step will take more than ~30 minutes, give me the time estimate before starting it.
- Fix random seeds, log library versions, and save a results JSON for every run.
- Never commit raw data or large files (.gitignore already covers data/, results/, models/*.joblib, .venv/).

PHASE 0 — Setup
- Run `bash setup.sh`. Fix anything that breaks (on Apple Silicon, LightGBM needs `brew install libomp`). Pin working versions in requirements.txt.
- Write pytest tests:
  - re-aiming a pitch at its own actual plate location reproduces its real VAA/HAA (to ~1e-6)
  - left- and right-handed pitches mirror symmetrically
  - the run-value target mapping matches run_values.csv
  - feature engineering works on a tiny fixture
- Add a GitHub Actions workflow that runs only pytest and `python run_ablation.py --synthetic`. No data downloads in CI.

PHASE 1 — Data
- Run `python fetch_data.py` (2022–2026). Make it resumable: chunked, cached, retries with backoff. Run it in the background and log progress.
- Before any modeling, write a data-quality report:
  - rows per season
  - null rates per feature per season, especially arm_angle, spin_axis, extension
  - pitch-type mix by season
  - duplicate pitches on (game_pk, at_bat_number, pitch_number)
  - a physics sanity check on real rows
  Show me the report and stop.

PHASE 2 — Baseline sanity check
- Run variant A (tjStuff+ v3.0 features) on fold 1: train 2022–23, score 2024, validate against 2025.
- TJ published ~0.85 year-over-year stickiness for 2023→2024 trained on 2020–22. My split differs, so don't expect an exact match, but A should land in roughly the same range (about 0.80–0.88 at minimum 100 pitches). If it's far off, find out why before moving on: check the handedness mirroring, the target join, the no-fastball fallback, and the LightGBM settings. Report what you found.

PHASE 3 — Ablation
- Use rolling-origin validation:
  - fold 1: train 2022–23, score 2024, validate on 2025
  - fold 2: train 2022–24, score 2025, validate on 2026
  Report each fold plus the mean, with paired bootstrap 95% CIs against A. Fold 1 trains on only two seasons, so weight fold 2 when they disagree.
- Run the existing variants A–F, plus these new ones:
  - G: batter handedness, encoded as same-side vs opposite-side
  - H: seam-shifted-wake proxy (observed spin_axis minus the axis implied by pfx_x/pfx_z)
  - I: recency-weighted training seasons
  - J: expected-outcome target. Value balls in play with estimated_woba_using_speedangle converted to runs; keep non-contact outcomes as they are.
- Report a PCA of the 18 grid angles. Under the constant-acceleration fit I expect only 2–3 independent dimensions. If that's confirmed, say so plainly and collapse the grid.
- Also report stickiness for each pitch type: FF, SI, FC, SL, ST, CU, KC, CH, FS.
- Never tune on the evaluation seasons. If you tune at all, use GroupKFold by pitcher within the training years, Optuna, and early stopping, with a modest budget.

PHASE 4 — Final model
- Pick the winner by this pre-set rule: it must improve the mean next-season wOBA and RV/100 correlations over A across both folds, with CIs excluding 0, without lowering stickiness by more than 0.01. If nothing beats A, say so. Don't cherry-pick.
- Train the final model on 2022–2026. Save the model, feature list, and scaling constants (xRV mean/sd) under models/ with a version tag.
- Produce 2026 outputs:
  - pitcher-level and pitch-type Stuff+ leaderboards as CSVs, with minimum-pitch filters
  - 20–80 pitch grades
  - an SP/RP split
  - a home/away park check (Coors especially)
- Write MODEL_CARD.md covering:
  - data (2022–2026), target, and features
  - the validation table against the tjStuff+ v3.0 feature set
  - limitations: no location or sequencing, reliever velocity bias, unusual arm slots, pitch classification noise, park/Hawk-Eye calibration, 2022 being the only pre-2023-rule-change season, and a short training window
- Credit tjStuff+ and keep the MIT notice for run_values.csv.
