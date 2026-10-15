# E1: out-of-pitcher comparison of A and S6, 2020–2026

Exploratory (plan in `EXPLORATORY.md`, committed before this was run). 5-fold GroupKFold by pitcher over 4,511,161 pitches and 1,863 pitchers: every score comes from a model that never saw that pitcher. 3 seeds, TJ's settings, no tuning. Correlations are sign-flipped so higher is better; CIs are paired bootstraps over pitchers, resampled jointly across season pairs.

## Primary comparison: S6 minus A, mean of the six season pairs

| Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| 100 | +0.011 [-0.002, +0.025] | -0.022 [-0.048, +0.003] | +0.005 [-0.024, +0.036] | -0.020 [-0.046, +0.005] |
| 300 | +0.009 [-0.005, +0.022] | -0.049 [-0.080, -0.022] | -0.004 [-0.038, +0.028] | -0.046 [-0.077, -0.019] |
| 500 | +0.013 [-0.001, +0.027] | -0.030 [-0.063, -0.000] | -0.002 [-0.041, +0.032] | -0.028 [-0.060, +0.002] |

Excluding the 2020 → 2021 pair (60-game season):

| Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| 100 | +0.010 [-0.005, +0.023] | -0.028 [-0.055, -0.000] | +0.006 [-0.026, +0.037] | -0.024 [-0.052, +0.004] |
| 300 | +0.006 [-0.009, +0.020] | -0.059 [-0.091, -0.028] | -0.003 [-0.040, +0.032] | -0.057 [-0.088, -0.025] |
| 500 | +0.007 [-0.008, +0.020] | -0.056 [-0.091, -0.022] | -0.012 [-0.054, +0.026] | -0.054 [-0.089, -0.020] |

## Mean of the six season pairs, each model

| Model | Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|---:|
| A | 100 | 0.867 | 0.258 | 0.308 | 0.259 |
| S6 | 100 | 0.878 | 0.236 | 0.313 | 0.239 |
| A | 300 | 0.878 | 0.323 | 0.383 | 0.328 |
| S6 | 300 | 0.888 | 0.276 | 0.382 | 0.283 |
| A | 500 | 0.883 | 0.357 | 0.441 | 0.363 |
| S6 | 500 | 0.895 | 0.327 | 0.439 | 0.335 |

## Each season pair, minimum 100 pitches

| Pair | Pitchers | A next wOBA | S6 next wOBA | A next RV/100 | S6 next RV/100 | S6 − A, RV/100 | A stickiness | S6 stickiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 → 2021 | 479 | 0.267 | 0.275 | 0.269 | 0.272 | +0.002 [-0.056, +0.055] | 0.855 | 0.872 |
| 2021 → 2022 | 514 | 0.217 | 0.141 | 0.220 | 0.154 | -0.066 [-0.131, -0.003] | 0.857 | 0.878 |
| 2022 → 2023 | 517 | 0.302 | 0.292 | 0.302 | 0.293 | -0.010 [-0.064, +0.042] | 0.883 | 0.875 |
| 2023 → 2024 | 507 | 0.252 | 0.206 | 0.252 | 0.209 | -0.043 [-0.093, +0.007] | 0.857 | 0.871 |
| 2024 → 2025 | 517 | 0.245 | 0.241 | 0.242 | 0.243 | +0.001 [-0.053, +0.056] | 0.870 | 0.876 |
| 2025 → 2026 | 513 | 0.267 | 0.262 | 0.270 | 0.266 | -0.004 [-0.054, +0.047] | 0.883 | 0.898 |

## Same-season fit and pitch-level R² (out-of-pitcher)

| Season | Pitchers | A vs same-season RV/100 | S6 vs same-season RV/100 | A pitch R² | S6 pitch R² |
|---|---:|---:|---:|---:|---:|
| 2020 | 578 | 0.225 | 0.159 | 0.00098 | 0.00063 |
| 2021 | 736 | 0.346 | 0.280 | 0.00106 | 0.00067 |
| 2022 | 710 | 0.269 | 0.205 | 0.00083 | 0.00047 |
| 2023 | 709 | 0.296 | 0.279 | 0.00099 | 0.00054 |
| 2024 | 712 | 0.287 | 0.247 | 0.00072 | 0.00046 |
| 2025 | 722 | 0.249 | 0.236 | 0.00081 | 0.00054 |
| 2026 | 720 | 0.299 | 0.251 | 0.00084 | 0.00046 |

Library versions: python 3.9.6, platform macOS-15.7.3-arm64-arm-64bit, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.6.0, pyarrow 21.0.0, pybaseball 2.2.7, optuna 5.0.0
