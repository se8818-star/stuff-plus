# Exploratory analyses (after the confirmation)

These were planned after every 2022–2026 result had been seen, so none of them
is a confirmation. Each plan is written and committed before it is run.

## E1. Out-of-pitcher comparison of A and S6 on 2020–2026

**Question.** When a model cannot remember the pitcher it is scoring, which
feature set grades stuff better: A (tjStuff+ v3.0's ten inputs) or S6 (six
approach-angle inputs)?

**Why.** `overfit_check.py` showed both models memorise their training
pitchers. In `holdout_check.py`, S6 predicted next-season results better than A
for pitchers who debuted after training (0.295 against 0.198 on RV/100), but on
only 87 pitchers. That lead is what motivates this analysis, which is why the
result cannot count as independent confirmation of it.

**Design.**
- Data: regular seasons 2020–2026 (2020–21 newly downloaded; 2020 was a
  60-game season, and spin rates changed with the mid-2021 sticky-substance
  enforcement).
- 5-fold GroupKFold by pitcher over all seven seasons. Every pitch is scored by
  a model trained on the other four folds' pitchers, so no pitcher is ever
  scored by a model that saw any of his pitches. This replaces the single
  60/40 split first proposed: every pitcher is held out once, and it does not
  depend on one random draw.
- Models: A and S6 only, exactly as frozen (TJ's LightGBM settings, seeds 0–2,
  no tuning).
- Stuff+ is scaled within each scored season from that season's out-of-fold
  xRV.
- Season pairs: 2020→21, 2021→22, 2022→23, 2023→24, 2024→25, 2025→26.

**Primary comparison (fixed before running).** S6 minus A, mean over the six
season pairs, minimum 100 pitches in both seasons, unshrunk, for next-season
wOBA and next-season RV/100 correlations, with a paired bootstrap 95% CI
(2000 replicates; pitchers resampled jointly across season pairs). Stickiness
and next-season K−BB% are reported alongside.

**Also reported.** Each season pair separately; minimums of 300 and 500
pitches; the mean excluding the 2020→21 pair (short season); same-season
correlations; out-of-pitcher pitch-level R².

**Reading the result.** An interval excluding zero would be the first evidence
in this project that one feature set grades unseen pitchers' stuff better. It
would still be exploratory, on data already examined, and would say nothing
about forecasting returning pitchers, where memory helps A more than S6. An
interval including zero means the two remain indistinguishable. Nothing about
the shipped model changes automatically either way.
