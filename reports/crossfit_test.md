# E2: cross-fitted scoring against the remembering model, forward in time

Exploratory (plan in `EXPLORATORY.md`, committed before this was run). Feature set A, TJ's settings, 3 seeds. Each split trains on every season from 2020 up to the year before the scored one. *Remembering* is one model on all training pitchers; *cross-fitted* scores each pitcher with a fold model that never saw him. Differences are cross-fitted minus remembering with paired-bootstrap 95% CIs.

## Primary comparison: mean of the four splits

| Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| 100 | -0.007 [-0.016, +0.002] | -0.036 [-0.053, -0.020] | -0.034 [-0.048, -0.020] | -0.037 [-0.055, -0.020] |
| 300 | -0.008 [-0.016, +0.001] | -0.036 [-0.059, -0.017] | -0.032 [-0.049, -0.016] | -0.038 [-0.063, -0.018] |
| 500 | -0.010 [-0.019, -0.001] | -0.029 [-0.053, -0.008] | -0.023 [-0.039, -0.007] | -0.032 [-0.059, -0.010] |

Pre-set reading at 100 pitches: close (both lower bounds above −0.03): **no**; as predictive (both intervals include zero): **no**.

Only pitchers who were in the training seasons:

| Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| 100 | -0.008 [-0.018, +0.001] | -0.039 [-0.059, -0.021] | -0.038 [-0.054, -0.022] | -0.041 [-0.061, -0.022] |
| 300 | -0.009 [-0.018, +0.001] | -0.038 [-0.062, -0.016] | -0.033 [-0.052, -0.014] | -0.040 [-0.067, -0.016] |
| 500 | -0.011 [-0.020, -0.002] | -0.029 [-0.055, -0.007] | -0.024 [-0.041, -0.006] | -0.032 [-0.061, -0.007] |

## Each method, mean of the four splits

| Method | Min pitches | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|---:|
| remember | 100 | 0.869 | 0.271 | 0.287 | 0.272 |
| crossfit | 100 | 0.862 | 0.236 | 0.254 | 0.235 |
| remember | 300 | 0.880 | 0.321 | 0.367 | 0.332 |
| crossfit | 300 | 0.872 | 0.285 | 0.336 | 0.293 |
| remember | 500 | 0.886 | 0.342 | 0.414 | 0.353 |
| crossfit | 500 | 0.876 | 0.313 | 0.391 | 0.321 |

## Each split, minimum 100 pitches

| Scored → validated | Trained on | Pitchers (known / new) | Remembering RV/100 | Cross-fitted RV/100 | Difference | Remembering stickiness | Cross-fitted stickiness | r between the two scores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 → 2023 | 2020–2021 | 421 / 96 | 0.272 | 0.233 | -0.039 [-0.075, -0.005] | 0.868 | 0.852 | 0.933 |
| 2023 → 2024 | 2020–2022 | 423 / 84 | 0.273 | 0.240 | -0.033 [-0.053, -0.012] | 0.853 | 0.845 | 0.962 |
| 2024 → 2025 | 2020–2023 | 425 / 92 | 0.258 | 0.207 | -0.051 [-0.084, -0.022] | 0.872 | 0.870 | 0.951 |
| 2025 → 2026 | 2020–2024 | 429 / 84 | 0.286 | 0.261 | -0.025 [-0.048, -0.004] | 0.886 | 0.882 | 0.971 |

Library versions: python 3.9.6, platform macOS-15.7.3-arm64-arm-64bit, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.6.0, pyarrow 21.0.0, pybaseball 2.2.7, optuna 5.0.0
