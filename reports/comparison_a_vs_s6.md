# Stuff+ (no remembering) against the six-variable model (S6), 2026

Both trained on 2022–2026 with the same settings; both are scored without remembering the pitcher (out-of-fold) and shrunk. A below is Stuff+ (no remembering). A uses ten inputs (velocity, spin, extension, vertical and horizontal acceleration, release x/z, and three fastball differences). S6 uses six (velocity, extension, location-neutral VAA and HAA, and their differences from the fastball). S6 was not confirmed as better or worse than A (`reports/confirmation.md`).

|  | A | S6 |
|---|---:|---:|
| Inputs | 10 | 6 |
| Qualified pitchers (≥100 pitches) | 720 | 720 |
| Pitcher-level sd of Stuff+ | 4.72 | 4.93 |
| Split-half reliability | 0.948 | 0.959 |
| Correlation with velocity | 0.634 | 0.690 |
| RP minus SP (mean) | +1.58 | +1.42 |
| RHP minus LHP (mean) | +3.07 | +2.96 |
| Coors effect | -5.38 | -2.38 |

## Agreement

- Pitcher Stuff+ correlation: 0.797 (Spearman 0.783) over 720 pitchers; 0.798 among the 468 with at least 500 pitches.
- Median absolute difference: 1.7 points; 9% of pitchers differ by more than 5.
- Top 25 overlap (≥500 pitches): 14 of 25; top 50: 32 of 50.

## By pitch type (pitcher × pitch, ≥50 pitches)

| Pitch | Pairs | Correlation A vs S6 | Mean A | Mean S6 |
|---|---:|---:|---:|---:|
| CH | 324 | 0.714 | 97.8 | 98.5 |
| CU | 213 | 0.814 | 97.8 | 97.4 |
| FC | 241 | 0.478 | 98.4 | 97.4 |
| FF | 597 | 0.721 | 98.7 | 100.2 |
| FS | 80 | 0.852 | 99.1 | 102.1 |
| KC | 34 | 0.970 | 99.9 | 100.8 |
| SI | 425 | 0.566 | 99.7 | 96.9 |
| SL | 387 | 0.801 | 101.8 | 102.6 |
| ST | 256 | 0.659 | 102.9 | 100.6 |

## Pitchers S6 likes more (≥500 pitches)

| Pitcher | Role | Pitches | A | S6 | S6 − A |
|---|---:|---:|---:|---:|---:|
| Milner, Hoby | RP | 673 | 89.4 | 98.0 | +8.6 |
| Eisert, Brandon | RP | 783 | 93.1 | 101.5 | +8.4 |
| Fermin, José | RP | 958 | 99.3 | 106.9 | +7.6 |
| Sommers, Drew | RP | 639 | 97.8 | 104.6 | +6.8 |
| Cruz, Steven | RP | 928 | 103.5 | 110.1 | +6.6 |
| Guerrero, Tyron | RP | 607 | 112.1 | 118.5 | +6.4 |
| Alvarado, José | RP | 1,127 | 104.3 | 110.7 | +6.4 |
| Fuentes, Didier | RP | 1,230 | 104.1 | 110.5 | +6.4 |
| Soriano, George | RP | 1,149 | 100.6 | 106.5 | +5.9 |
| Muñoz, Andrés | RP | 952 | 108.0 | 113.7 | +5.7 |

## Pitchers S6 likes less (≥500 pitches)

| Pitcher | Role | Pitches | A | S6 | S6 − A |
|---|---:|---:|---:|---:|---:|
| Rogers, Tyler | RP | 1,013 | 109.7 | 90.1 | -19.6 |
| Kelly, Kevin | RP | 977 | 108.0 | 93.4 | -14.6 |
| Jansen, Kenley | RP | 725 | 110.1 | 95.6 | -14.5 |
| Slaten, Justin | RP | 507 | 108.2 | 96.2 | -12.0 |
| Faucher, Calvin | RP | 1,166 | 108.4 | 97.7 | -10.7 |
| Thompson, Ryan | RP | 653 | 108.1 | 99.6 | -8.5 |
| Watson, Ryan | RP | 1,138 | 98.6 | 90.4 | -8.2 |
| Civale, Aaron | RP | 1,893 | 98.0 | 89.9 | -8.1 |
| Criswell, Cooper | RP | 641 | 102.3 | 94.3 | -8.0 |
| Vesia, Alex | RP | 1,008 | 106.9 | 99.3 | -7.6 |

Full table: `leaderboards/comparison_2026.csv`.
