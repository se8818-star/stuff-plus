"""
Head-to-head ablation of Stuff+ feature sets on tjStuff+ v3.0's validation design.

  Default fold: train 2022-23  ->  score 2024, validate against 2025
  Second fold:  --train 2022 2023 2024 --eval 2025  (validates against 2026)
  Stuff+ = 100 - 10 * z(xRV), scaled with the eval-season pitch-level mean/sd
  Pitcher-season mean Stuff+ (min 100 pitches in both years), then:
    stickiness      r(Stuff+ year N, Stuff+ year N+1)
    predictiveness  r(Stuff+ year N, year N+1 wOBA / K-BB% / RV per 100)
  Paired bootstrap over pitchers gives a 95% CI on each variant minus A.

Usage
  python fetch_data.py                     # once, downloads 2022-2026
  python run_ablation.py                   # all variants
  python run_ablation.py --variants A_tj_v3 D_grid9_angles_only --seeds 3
  python run_ablation.py --synthetic       # smoke test without data
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import VARIANTS, GRID, add_features, load

HERE = Path(__file__).parent


def make_model(seed):
    """LightGBM with tjStuff+ v3.0 hyperparameters if installed, otherwise
    sklearn's histogram GBM (same algorithm family) with matched settings."""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31,
                             max_depth=-1, min_child_samples=20, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2,
                             random_state=seed, force_row_wise=True, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01,
                                             max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=0.2, early_stopping=False,
                                             random_state=seed)


def outcomes(df_year):
    """Pitcher-season results computed straight from Statcast."""
    pa = df_year[df_year["events"].notna()]
    g = pa.groupby("pitcher")
    res = pd.DataFrame({
        "pa": g.size(),
        "k": g["events"].apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).sum()),
        "bb": g["events"].apply(lambda e: (e == "walk").sum()),
    })
    res["k_bb"] = (res["k"] - res["bb"]) / res["pa"]
    w = df_year.groupby("pitcher")[["woba_value", "woba_denom"]].sum()
    res["woba"] = w["woba_value"] / w["woba_denom"]
    rv = df_year.groupby("pitcher")["target"].agg(["sum", "size"])
    res["rv100"] = rv["sum"] / rv["size"] * 100
    return res


def pitcher_stuff(df, pred, mu, sd):
    s = 100 - 10 * (pred - mu) / sd
    return pd.DataFrame({"pitcher": df["pitcher"].to_numpy(), "s": s}) \
             .groupby("pitcher")["s"].agg(["mean", "size"])


def evaluate(train, ev, nxt, feats, seeds):
    tr = train.dropna(subset=feats + ["target"])
    p_ev = np.zeros(len(ev)); p_nx = np.zeros(len(nxt))
    for sd in seeds:
        m = make_model(sd).fit(tr[feats].to_numpy(), tr["target"].to_numpy())
        p_ev += m.predict(ev[feats].to_numpy()) / len(seeds)
        p_nx += m.predict(nxt[feats].to_numpy()) / len(seeds)
    mu, sdv = p_ev.mean(), p_ev.std()
    a = pitcher_stuff(ev, p_ev, mu, sdv)
    b = pitcher_stuff(nxt, p_nx, mu, sdv)
    # pitch-level out-of-sample R^2 on the next season (tiny but comparable)
    y = nxt["target"].to_numpy()
    r2 = 1 - ((y - p_nx) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2, m


def corr_table(a, b, out_nx, min_n):
    j = a.join(b, lsuffix="_y1", rsuffix="_y2", how="inner").join(out_nx, how="inner")
    j = j[(j["size_y1"] >= min_n) & (j["size_y2"] >= min_n)]
    return j


def metrics(j):
    return {
        "stickiness": j["mean_y1"].corr(j["mean_y2"]),
        "next_woba": -j["mean_y1"].corr(j["woba"]),     # sign-flipped: higher = better
        "next_k_bb": j["mean_y1"].corr(j["k_bb"]),
        "next_rv100": -j["mean_y1"].corr(j["rv100"]),
    }


def bootstrap_diff(j_base, j_var, n=1000, seed=0):
    ids = j_base.index.intersection(j_var.index).to_numpy()
    rng = np.random.default_rng(seed)
    out = {k: [] for k in metrics(j_base)}
    for _ in range(n):
        s = rng.choice(ids, len(ids), replace=True)
        mb, mv = metrics(j_base.loc[s]), metrics(j_var.loc[s])
        for k in out:
            out[k].append(mv[k] - mb[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}


def redundancy_report(df):
    """How many independent dimensions do the 18 grid angles really have?"""
    X = df[GRID].dropna().sample(min(200_000, len(df)), random_state=0).to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    ev = np.linalg.svd(X, full_matrices=False, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return np.cumsum(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--train", nargs="*", type=int, default=[2022, 2023])
    ap.add_argument("--eval", type=int, default=2024)
    ap.add_argument("--min-pitches", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1, help="average N model seeds")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    years = args.train + [args.eval, args.eval + 1]
    if args.synthetic:
        from synthetic import make
        df = add_features(make(n_pitchers=150, pitches_per_season=400, years=tuple(years)))
    else:
        df = load(HERE / "data", years)
    print(f"loaded {len(df):,} pitches in {time.time()-t0:.0f}s")

    cum = redundancy_report(df)
    print("\n18 grid angles -> cumulative variance by principal component:")
    print("  " + "  ".join(f"PC{i+1}:{c:.4f}" for i, c in enumerate(cum[:6])))

    train = df[df["year"].isin(args.train)]
    ev = df[df["year"] == args.eval].dropna(subset=["target"])
    nxt = df[df["year"] == args.eval + 1].dropna(subset=["target"])
    out_nx = outcomes(nxt)

    results, tables = {}, {}
    seeds = list(range(args.seeds))
    for name in args.variants:
        feats = VARIANTS[name]
        if df[feats].isna().all().any():
            print(f"\n{name}: skipped (missing column: "
                  f"{[f for f in feats if df[f].isna().all()]})")
            continue
        ev_v, nx_v = ev.dropna(subset=feats), nxt.dropna(subset=feats)
        t1 = time.time()
        a, b, r2, model = evaluate(train, ev_v, nx_v, feats, seeds)
        j = corr_table(a, b, out_nx, args.min_pitches)
        tables[name] = j
        m = metrics(j); m["pitch_r2_next"] = r2; m["n_pitchers"] = len(j)
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            m["top_features"] = [f for _, f in sorted(zip(imp, feats), reverse=True)[:6]]
        results[name] = m
        print(f"\n{name}  ({len(feats)} features, {time.time()-t1:.0f}s)")
        for k in ["stickiness", "next_woba", "next_k_bb", "next_rv100", "pitch_r2_next"]:
            print(f"  {k:<14}{m[k]:+.4f}")

    base = "A_tj_v3"
    if base in tables:
        print(f"\nDifference vs {base} (paired bootstrap 95% CI, {args.boot} reps):")
        for name, j in tables.items():
            if name == base:
                continue
            ci = bootstrap_diff(tables[base], j, n=args.boot)
            results[name]["ci_vs_A"] = ci
            line = "  ".join(
                f"{k}: {results[name][k]-results[base][k]:+.3f} [{lo:+.3f},{hi:+.3f}]"
                for k, (lo, hi) in ci.items())
            print(f"  {name:<24}{line}")

    (HERE / "results").mkdir(exist_ok=True)
    tag = "synthetic" if args.synthetic else f"{args.eval}"
    with open(HERE / "results" / f"ablation_{tag}.json", "w") as f:
        json.dump({"grid_pca_cumvar": cum.tolist(), "results": results}, f, indent=2, default=float)
    print(f"\nsaved results/ablation_{tag}.json  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
