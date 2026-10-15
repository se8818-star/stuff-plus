# Our Stuff+ vs TJStats' tjStuff+

Seasons 2022–2026, MLB regular season. TJStats numbers are his per-pitch tjStuff+ from his public Hugging Face dataset (TJStatsApps/mlb_data), averaged per pitcher-season. Outcomes come from our Statcast files. Pitchers are counted only if both systems grade them, and the pitch minimum applies to every season used.

"Ours (website)" is Stuff+ (no remembering): each season is graded by fold models that never saw that pitcher. "Ours (remembering)" is Stuff+ graded by a model trained only on earlier seasons. Both come from `results/preds/crossfit_*.parquet` for 2022–2025 and from the committed 2026 leaderboards.

## Short answer

1. **For full seasons, the two systems are not very different.** Pitcher-level correlation with TJ is 0.91–0.94 for the website model and 0.93–0.96 for the remembering model (2023–2025, at least 100 pitches). The spread is the same (sd about 5 for both).
2. **What makes them look different:**
   - **TJ's scale isn't re-centred each season.** His MLB average drifts up from 100.2 (2022) to 101.8 (2025) and 102.5 (2026). Ours is re-centred to 100 every season, so in 2025 TJ reads about 1.4 points higher across the board.
   - **The website grades pitchers without remembering them, but TJ's model has seen them.** TJ's numbers line up more closely with our remembering model than with the website model. Almost every big gap is an unusual pitcher whose results the remembering model has learned: Devin Williams (website 102.6, remembering 113.1, TJ 114.4), Tyler Rogers (95.9 / 109.5 / 106.9), Hoby Milner (88.0 / 99.2 / 97.8), Brent Suter and Edwin Díaz. A model that has never seen a submariner or Williams' changeup can't tell those pitches are good.
   - **TJStats' public 2026 file stops on April 4** (about 7% of the season). If you compared against his 2026 board, the gap is mostly small samples on his side: correlation is only 0.81 for 2026.
   - Smaller technical differences: he uses MLB API release points measured at 50 ft, where we use Statcast release points, and the website model rates hard throwers slightly higher than TJ does (the gap correlates 0.15 with fastball velocity).

## Predictiveness

Correlations are sign-flipped so that higher is better. Each number is the mean over season pairs, with a paired bootstrap 95% CI on the difference from TJ.

### Clean seasons: 2024→2025 and 2025→2026

These are the seasons where TJ's grades cannot have been trained on the outcomes being predicted. TJ retrained with 2024 data only in 2025.

Next-season prediction, at least 300 pitches in both seasons (376 and 397 pitchers):

| | RV/100 | wOBA | xwOBA | FIP | K−BB% |
|---|---:|---:|---:|---:|---:|
| TJ | 0.307 | 0.299 | 0.334 | 0.285 | 0.383 |
| Ours (remembering) | 0.316 (+0.009 [−0.010, +0.028]) | 0.301 (+0.002) | 0.344 (+0.010) | 0.310 (+0.025 [+0.004, +0.046]) | 0.372 (−0.012) |
| Ours (website) | 0.283 (−0.024 [−0.055, +0.005]) | 0.271 (−0.028) | 0.311 (−0.023) | 0.278 (−0.007) | 0.347 (−0.036 [−0.064, −0.006]) |

Next-season prediction, at least 100 pitches (517 and 513 pitchers): TJ 0.262 RV/100, remembering 0.272 (+0.010 [−0.007, +0.027]), website 0.234 (−0.028 [−0.053, −0.006]).

Year-over-year stability, 2024→2025:

| Min pitches | n | TJ | Remembering | Website |
|---:|---:|---:|---:|---:|
| 100 | 517 | 0.872 | 0.870 (−0.002 [−0.015, +0.010]) | 0.831 (−0.041 [−0.067, −0.016]) |
| 300 | 376 | 0.886 | 0.880 (−0.006) | 0.831 (−0.054 [−0.084, −0.026]) |
| 500 | 311 | 0.895 | 0.887 (−0.007) | 0.851 (−0.043 [−0.069, −0.021]) |

Same-season fit, at least 300 pitches: TJ 0.346 RV/100, remembering 0.338, website 0.294. FIP: TJ 0.305, remembering 0.325, website 0.287.

### All four pairs, 2022→2023 through 2025→2026

Next-season RV/100, at least 100 pitches: TJ 0.288, remembering 0.272 (−0.015 [−0.031, −0.000]), website 0.235 (−0.052 [−0.072, −0.033]). TJ's whole lead comes from 2022→23 and 2023→24, where he leads by 0.04 (0.313 vs 0.272/0.273). In 2024→25 and 2025→26 he does not lead (0.253 vs 0.258, and 0.271 vs 0.286).

Our likely explanation, which is inferred and not confirmed: his public 2022–2023 grades were produced by a later model whose training included 2023–2024, so those seasons are partly in-sample for him. His same-season fit also falls from 0.32–0.35 (2022–2024) to 0.25 in 2025, and his agreement with our forward-only model rises every year (0.91 → 0.96). Both fit that explanation. Full per-pair numbers are in `tjstats_comparison.json`, and the clean-seasons-only run is in `tjstats_comparison_clean.json`.

## Bottom line

- With the same information (only past seasons), our remembering Stuff+ predicts as well as tjStuff+. It is level on RV/100, wOBA and xwOBA, slightly better on FIP and slightly worse on K−BB%. None of the gaps is large enough to call either system better, except FIP at 300 pitches.
- The website's no-remembering grade is about 0.02–0.04 less predictive and about 0.04–0.05 less stable than both. That is the cost of grading physical traits alone, as the model card already measured (0.272 → 0.235).
- ERA isn't included because FanGraphs couldn't be reached. FIP here is computed from Statcast events without the league constant, which doesn't affect correlations.

## Reproduce

Exploratory, not part of the pre-registered comparison. Needs `data/statcast_2021.parquet` through 2026 (`python -m pipeline.fetch_data 2021`) and the per-pitch predictions from `python -m experiments.crossfit_test`.

```bash
python -m experiments.tjstats_comparison.fetch           # TJ's per-pitch tjStuff+ -> data/external/ (not in git)
python -m experiments.tjstats_comparison.build           # outcomes and our scores -> results/tjstats/
python -m experiments.tjstats_comparison.analyze         # all pairs, 2000 draws -> reports/tjstats_comparison.json
python -m experiments.tjstats_comparison.analyze --clean # 2024->25 and 2025->26, 1000 draws -> _clean.json
```
