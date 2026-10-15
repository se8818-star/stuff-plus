# Overfitting check

Models trained on 2022–23. *In-sample* scores the training pitches; *out-of-pitcher* scores the same pitches with a model that never saw that pitcher; 2024 and 2025 are later seasons. Pitcher-level columns use 2023 pitchers with at least 100 pitches and are correlations of Stuff+ with RV/100 (sign-flipped), in the same season and in 2024.

| Model | Pitch R² in-sample | Pitch R² out-of-pitcher | Pitch R² 2024 | Pitch R² 2025 | Same-season RV/100, in-sample | Same-season RV/100, out-of-pitcher | Next-season RV/100, in-sample | Next-season RV/100, out-of-pitcher | r(in-sample, out-of-pitcher) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00452 | 0.00066 | 0.00069 | 0.00068 | 0.437 | 0.277 | 0.299 | 0.193 | 0.857 |
| S6 | 0.00344 | 0.00032 | 0.00043 | 0.00041 | 0.378 | 0.253 | 0.241 | 0.169 | 0.880 |
