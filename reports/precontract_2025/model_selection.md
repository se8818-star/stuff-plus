# Pre-contract 2025 Stuff+: model selection on 2022–2024

Pre-registered in `docs/PRECONTRACT_2025_PREREGISTRATION.md`. Only 2022–2024 data were loaded. Correlations are sign-flipped so higher is better; pitchers need 100 pitches in both seasons unless stated.

**Selected: A.** best challenger B_plus_center: not above A; A kept.

## Selection score

Mean of next-season wOBA and RV/100 correlations over the three season pairs. Interval: paired bootstrap over pitchers, 5000 replicates, Bonferroni-adjusted 99.29% (7 challengers).

| Model | Score | vs A | Adjusted interval | Share of replicates > 0 |
|---|---:|---:|---:|---:|
| A | 0.2669 | – | – | – |
| A_reseeded | 0.2626 | -0.0043 | [-0.0137, +0.0040] | 0.09 |
| B_plus_center | 0.2537 | -0.0132 | [-0.0310, +0.0025] | 0.01 |
| B9_plus_grid9 | 0.2526 | -0.0144 | [-0.0335, +0.0037] | 0.02 |
| D1_center_angles_only | 0.2348 | -0.0321 | [-0.0884, +0.0211] | 0.06 |
| D_grid9_angles_only | 0.2343 | -0.0326 | [-0.0867, +0.0185] | 0.05 |
| C1_center_no_accel | 0.2340 | -0.0329 | [-0.0744, +0.0030] | 0.01 |
| C_grid9_no_accel | 0.2323 | -0.0347 | [-0.0754, -0.0000] | 0.00 |
| S6 | 0.2216 | -0.0453 | [-0.1144, +0.0202] | 0.03 |

## Mean over the three pairs, minimum 100 pitches

| Model | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| A | 0.845 | 0.268 | 0.308 | 0.266 |
| A_reseeded | 0.846 | 0.264 | 0.306 | 0.262 |
| B_plus_center | 0.838 | 0.255 | 0.294 | 0.253 |
| B9_plus_grid9 | 0.839 | 0.254 | 0.293 | 0.251 |
| D1_center_angles_only | 0.860 | 0.236 | 0.292 | 0.233 |
| D_grid9_angles_only | 0.853 | 0.236 | 0.292 | 0.233 |
| C1_center_no_accel | 0.850 | 0.235 | 0.284 | 0.233 |
| C_grid9_no_accel | 0.846 | 0.233 | 0.282 | 0.231 |
| S6 | 0.831 | 0.222 | 0.275 | 0.222 |

## Mean over the three pairs, minimum 300 pitches

| Model | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| A | 0.857 | 0.309 | 0.338 | 0.316 |
| A_reseeded | 0.858 | 0.303 | 0.335 | 0.310 |
| B_plus_center | 0.862 | 0.296 | 0.327 | 0.302 |
| B9_plus_grid9 | 0.862 | 0.296 | 0.326 | 0.301 |
| D1_center_angles_only | 0.875 | 0.245 | 0.312 | 0.248 |
| D_grid9_angles_only | 0.871 | 0.246 | 0.312 | 0.249 |
| C1_center_no_accel | 0.866 | 0.256 | 0.312 | 0.260 |
| C_grid9_no_accel | 0.865 | 0.253 | 0.308 | 0.258 |
| S6 | 0.840 | 0.222 | 0.296 | 0.229 |

## Mean over the three pairs, minimum 500 pitches

| Model | Stickiness | Next wOBA | Next K−BB% | Next RV/100 |
|---|---:|---:|---:|---:|
| A | 0.869 | 0.314 | 0.364 | 0.324 |
| A_reseeded | 0.869 | 0.306 | 0.359 | 0.315 |
| B_plus_center | 0.873 | 0.308 | 0.356 | 0.315 |
| B9_plus_grid9 | 0.873 | 0.308 | 0.355 | 0.315 |
| D1_center_angles_only | 0.886 | 0.271 | 0.338 | 0.276 |
| D_grid9_angles_only | 0.881 | 0.272 | 0.340 | 0.278 |
| C1_center_no_accel | 0.879 | 0.274 | 0.338 | 0.281 |
| C_grid9_no_accel | 0.878 | 0.274 | 0.336 | 0.281 |
| S6 | 0.862 | 0.244 | 0.325 | 0.253 |

## forward_2023_2024, minimum 100 pitches

| Model | Pitchers | Stickiness | Next wOBA | Next K−BB% | Next RV/100 | Pitch R² next season |
|---|---:|---:|---:|---:|---:|---:|
| A | 507 | 0.808 | 0.247 | 0.303 | 0.245 | 0.00038 |
| A_reseeded | 507 | 0.808 | 0.245 | 0.298 | 0.241 | 0.00035 |
| B_plus_center | 507 | 0.786 | 0.228 | 0.280 | 0.227 | 0.00037 |
| B9_plus_grid9 | 507 | 0.785 | 0.221 | 0.277 | 0.220 | 0.00035 |
| D1_center_angles_only | 507 | 0.820 | 0.221 | 0.276 | 0.218 | 0.00019 |
| D_grid9_angles_only | 507 | 0.806 | 0.218 | 0.272 | 0.215 | 0.00017 |
| C1_center_no_accel | 507 | 0.803 | 0.217 | 0.271 | 0.213 | 0.00023 |
| C_grid9_no_accel | 507 | 0.795 | 0.212 | 0.270 | 0.209 | 0.00022 |
| S6 | 507 | 0.780 | 0.191 | 0.239 | 0.192 | 0.00011 |

## cv_2022_2023, minimum 100 pitches

| Model | Pitchers | Stickiness | Next wOBA | Next K−BB% | Next RV/100 | Pitch R² next season |
|---|---:|---:|---:|---:|---:|---:|
| A | 517 | 0.879 | 0.286 | 0.312 | 0.285 | 0.00088 |
| A_reseeded | 517 | 0.878 | 0.283 | 0.314 | 0.283 | 0.00087 |
| B_plus_center | 517 | 0.872 | 0.288 | 0.308 | 0.286 | 0.00090 |
| B9_plus_grid9 | 517 | 0.871 | 0.294 | 0.309 | 0.292 | 0.00089 |
| D1_center_angles_only | 517 | 0.884 | 0.269 | 0.307 | 0.266 | 0.00052 |
| D_grid9_angles_only | 517 | 0.883 | 0.269 | 0.307 | 0.266 | 0.00051 |
| C1_center_no_accel | 517 | 0.879 | 0.264 | 0.295 | 0.262 | 0.00062 |
| C_grid9_no_accel | 517 | 0.876 | 0.264 | 0.293 | 0.263 | 0.00062 |
| S6 | 517 | 0.852 | 0.279 | 0.309 | 0.278 | 0.00044 |

## cv_2023_2024, minimum 100 pitches

| Model | Pitchers | Stickiness | Next wOBA | Next K−BB% | Next RV/100 | Pitch R² next season |
|---|---:|---:|---:|---:|---:|---:|
| A | 507 | 0.847 | 0.270 | 0.310 | 0.268 | 0.00065 |
| A_reseeded | 507 | 0.851 | 0.263 | 0.306 | 0.261 | 0.00060 |
| B_plus_center | 507 | 0.857 | 0.248 | 0.295 | 0.246 | 0.00064 |
| B9_plus_grid9 | 507 | 0.859 | 0.246 | 0.293 | 0.243 | 0.00064 |
| D1_center_angles_only | 507 | 0.876 | 0.219 | 0.293 | 0.216 | 0.00041 |
| D_grid9_angles_only | 507 | 0.871 | 0.221 | 0.297 | 0.218 | 0.00040 |
| C1_center_no_accel | 507 | 0.868 | 0.226 | 0.285 | 0.223 | 0.00048 |
| C_grid9_no_accel | 507 | 0.868 | 0.224 | 0.281 | 0.222 | 0.00050 |
| S6 | 507 | 0.860 | 0.195 | 0.277 | 0.195 | 0.00039 |

Library versions: python 3.9.25, platform Linux-6.18.44-fc-v445-x86_64-with-glibc2.39, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.6.0, pyarrow 21.0.0, pybaseball 2.2.7, optuna 5.0.0
