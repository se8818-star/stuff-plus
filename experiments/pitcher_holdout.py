"""
Exploratory analysis E1 (EXPLORATORY.md): out-of-pitcher comparison of A and S6
on 2020-2026. 5-fold GroupKFold by pitcher over all seasons, so every pitch is
scored by a model that never saw that pitcher.

    python pitcher_holdout.py   # results/pitcher_holdout.json, reports/pitcher_holdout.md
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import run_ablation as ra
from develop import KEY, MIN_PITCHES, N_BOOT, pitcher_table, stuff_plus, table
from features import load
from models import AMENDED, CANDIDATES, add_outcome_flags, fit_predict
from overfit_check import r2

HERE = Path(__file__).parent
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
MODELS = ("A", "S6")


def main():
    t0 = time.time()
    df = add_outcome_flags(load(HERE / "data", YEARS))
    df = df[df["target"].notna()].reset_index(drop=True)
    print(f"loaded {len(df):,} pitches, {df['pitcher'].nunique():,} pitchers ({time.time() - t0:.0f}s)", flush=True)
    folds = list(GroupKFold(n_splits=5).split(df, groups=df["pitcher"]))
    specs = {**CANDIDATES, **AMENDED}
    pdir = HERE / "results" / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    xrv = {}
    for name in MODELS:
        cache = pdir / f"holdout2020_{name}.parquet"
        if cache.exists():
            xrv[name] = df[KEY].merge(pd.read_parquet(cache), on=KEY, how="left", validate="1:1")["xrv"].to_numpy()
            continue
        oof = np.full(len(df), np.nan)
        for k, (i_fit, i_hold) in enumerate(folds):
            (p,), _ = fit_predict(specs[name], df.iloc[i_fit], [df.iloc[i_hold]])
            oof[i_hold] = p
            print(f"{name}: fold {k + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
        df[KEY].assign(xrv=oof).to_parquet(cache, index=False)
        xrv[name] = oof

    year = df["year"].to_numpy()
    outc = {y: ra.outcomes(df[year == y]) for y in YEARS}
    pairs = [(y, y + 1) for y in YEARS[:-1]]
    res = {n: {"pairs": {}, "same_season": {}} for n in MODELS}
    tables = {}
    for name in MODELS:
        res[name]["pitch_r2"] = {int(y): r2(df.loc[year == y, "target"].to_numpy(), xrv[name][year == y])
                                 for y in YEARS}
        tabs = {}
        for y in YEARS:
            m = year == y
            tabs[y] = pitcher_table(df[m], stuff_plus(xrv[name][m], xrv[name][m]))
            j = tabs[y][tabs[y]["size"] >= 100].join(outc[y], how="inner")
            res[name]["same_season"][y] = {"rv100": float(-j["mean"].corr(j["rv100"])),
                                           "woba": float(-j["mean"].corr(j["woba"])), "n": len(j)}
        for a, b in pairs:
            for mn in MIN_PITCHES:
                j = ra.corr_table(tabs[a], tabs[b], outc[b], mn)
                tables[(name, a, mn)] = j
                res[name]["pairs"][f"{a}_{mn}"] = {**ra.metrics(j), "n": len(j)}
        for mn in MIN_PITCHES:
            res[name][f"mean_{mn}"] = {k: float(np.mean([res[name]["pairs"][f"{a}_{mn}"][k] for a, _ in pairs]))
                                       for k in ra.KEYS}

    vs = {}
    for mn in MIN_PITCHES:
        for label, use in (("all", pairs), ("ex2020", pairs[1:])):
            bs = ra.bootstrap_folds([tables[("A", a, mn)] for a, _ in use],
                                    [tables[("S6", a, mn)] for a, _ in use], n=N_BOOT)
            vs[f"{label}_{mn}"] = {"mean": {"diff": bs["mean_diff"], "ci": bs["mean_ci"]},
                                   "pairs": {str(a): {"diff": bs["diff"][i], "ci": bs["ci"][i]}
                                             for i, (a, _) in enumerate(use)}}
    out = {"years": YEARS, "models": list(MODELS), "results": res, "s6_minus_a": vs,
           "n_pitches": int(len(df)), "n_pitchers": int(df["pitcher"].nunique()),
           "env": ra.env_info(), "seconds": time.time() - t0}
    (HERE / "results" / "pitcher_holdout.json").write_text(json.dumps(out, indent=1, default=float))

    H = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%", "next_rv100": "Next RV/100"}
    ci = lambda d, k: f"{d['diff'][k]:+.3f} [{d['ci'][k][0]:+.3f}, {d['ci'][k][1]:+.3f}]"  # noqa: E731
    md = ["# E1: out-of-pitcher comparison of A and S6, 2020–2026", "",
          "Exploratory (plan in `EXPLORATORY.md`, committed before this was run). 5-fold GroupKFold by pitcher "
          f"over {len(df):,} pitches and {df['pitcher'].nunique():,} pitchers: every score comes from a model "
          "that never saw that pitcher. 3 seeds, TJ's settings, no tuning. Correlations are sign-flipped so "
          "higher is better; CIs are paired bootstraps over pitchers, resampled jointly across season pairs.", "",
          "## Primary comparison: S6 minus A, mean of the six season pairs", ""]
    md += [table(["Min pitches"] + [H[k] for k in ra.KEYS],
                 [[str(mn)] + [ci(vs[f"all_{mn}"]["mean"], k) for k in ra.KEYS] for mn in MIN_PITCHES]), "",
           "Excluding the 2020 → 2021 pair (60-game season):", "",
           table(["Min pitches"] + [H[k] for k in ra.KEYS],
                 [[str(mn)] + [ci(vs[f"ex2020_{mn}"]["mean"], k) for k in ra.KEYS] for mn in MIN_PITCHES]), "",
           "## Mean of the six season pairs, each model", ""]
    md += [table(["Model", "Min pitches"] + [H[k] for k in ra.KEYS],
                 [[n, str(mn)] + [f"{res[n][f'mean_{mn}'][k]:.3f}" for k in ra.KEYS]
                  for mn in MIN_PITCHES for n in MODELS]), "",
           "## Each season pair, minimum 100 pitches", "",
           table(["Pair", "Pitchers", "A next wOBA", "S6 next wOBA", "A next RV/100", "S6 next RV/100",
                  "S6 − A, RV/100", "A stickiness", "S6 stickiness"],
                 [[f"{a} → {b}", str(res["A"]["pairs"][f"{a}_100"]["n"]),
                   f"{res['A']['pairs'][f'{a}_100']['next_woba']:.3f}", f"{res['S6']['pairs'][f'{a}_100']['next_woba']:.3f}",
                   f"{res['A']['pairs'][f'{a}_100']['next_rv100']:.3f}", f"{res['S6']['pairs'][f'{a}_100']['next_rv100']:.3f}",
                   ci(vs["all_100"]["pairs"][str(a)], "next_rv100"),
                   f"{res['A']['pairs'][f'{a}_100']['stickiness']:.3f}", f"{res['S6']['pairs'][f'{a}_100']['stickiness']:.3f}"]
                  for a, b in pairs]), "",
           "## Same-season fit and pitch-level R² (out-of-pitcher)", "",
           table(["Season", "Pitchers", "A vs same-season RV/100", "S6 vs same-season RV/100", "A pitch R²", "S6 pitch R²"],
                 [[str(y), str(res["A"]["same_season"][y]["n"]), f"{res['A']['same_season'][y]['rv100']:.3f}",
                   f"{res['S6']['same_season'][y]['rv100']:.3f}", f"{res['A']['pitch_r2'][y]:.5f}",
                   f"{res['S6']['pitch_r2'][y]:.5f}"] for y in YEARS]), "",
           "Library versions: " + ", ".join(f"{k} {v}" for k, v in out["env"].items() if v), ""]
    (HERE / "reports" / "pitcher_holdout.md").write_text("\n".join(md))
    print("\n".join(md[4:22]))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
