# Frozen models for the fold 2 confirmation

Committed before fold 2 is unlocked. Machine-readable copy: `FROZEN.json`.

| Model | Role | Features | Target | Settings |
|---|---|---|---|---|
| A | baseline | tjStuff+ v3.0 set (10) | TJ run value | TJ's LightGBM settings, seeds 0–2 |
| J | stage 1 winner (original rule) | same as A | balls in play valued by xwOBA-runs | same |
| S6 | added after stage 1 (deviation 1) | velocity, extension, location-neutral VAA and HAA, VAA and HAA difference from the primary fastball | TJ run value | same |

- No tuning (deviation 2). Nothing about any model changes after this commit.
- Fold 2: train 2022–24, score 2025, validate against 2026. Run once with
  `python confirm.py --unlock-fold2`.
- Rule, applied to J and S6 separately at minimum 100 pitches, unshrunk: next
  wOBA and next RV/100 both above A with 95% CIs excluding 0, and stickiness no
  more than 0.01 below A. One confirmed: it ships. Both: the higher fold 2 mean
  of the two correlations. Neither: A ships.

## Development selection scores (inner CV + fold 1, minimum 100 pitches)

| Model | Unshrunk | Shrunk |
|---|---:|---:|
| A | 0.2369 | 0.2369 |
| J | 0.2493 | 0.2495 |
| S6 | 0.2526 | 0.2528 |

Shrinkage does not lower any model's score, so it ships in the leaderboards for
whichever model ships. Selection and confirmation use the unshrunk numbers.
