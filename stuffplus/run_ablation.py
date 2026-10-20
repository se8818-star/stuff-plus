"""
Head-to-head ablation of Stuff+ feature sets on tjStuff+ v3.0's validation
design, with rolling-origin folds:

  fold 1: train 2022-23     ->  score 2024, validate against 2025
  fold 2: train 2022-24     ->  score 2025, validate against 2026

  Stuff+ = 100 - 10 * z(xRV), scaled with the scored season's pitch-level mean/sd
  Pitcher-season mean Stuff+ (min 100 pitches in both years), then:
    stickiness      r(Stuff+ year N, Stuff+ year N+1)
    predictiveness  r(Stuff+ year N, year N+1 wOBA / K-BB% / RV per 100)
  Paired bootstrap over pitchers gives a 95% CI on each variant minus A, per
  fold and for the mean of the folds (pitchers resampled jointly across folds).

Usage
  python -m pipeline.fetch_data                     # once, downloads 2022-2026
  python -m stuffplus.run_ablation --seeds 3         # all variants, both folds
  python -m stuffplus.run_ablation --folds 1 --variants A_tj_v3 D_grid9_angles_only
  python -m stuffplus.run_ablation --synthetic       # smoke test without data
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stuffplus.features import SPECS, VARIANTS, GRID, add_expected_target, add_features, load

ROOT = Path(__file__).resolve().parent.parent
BASE = "A_tj_v3"
FOLDS = {1: {"train": [2022, 2023], "eval": 2024},
         2: {"train": [2022, 2023, 2024], "eval": 2025}}
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"]
KEYS = ["stickiness", "next_woba", "next_k_bb", "next_rv100"]


def env_info():
    """Library versions, logged with every results JSON."""
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "pybaseball", "optuna"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def make_model(seed):
    """LightGBM with tjStuff+ v3.0 hyperparameters if installed, otherwise
    sklearn's histogram GBM (same algorithm family) with matched settings."""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31,
                             max_depth=-1, min_child_samples=20, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2,
                             random_state=seed, force_row_wise=True, deterministic=True,
                             verbose=-1)
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


def season_weights(years, recency):
    """Weight of a training pitch = recency ** (seasons before the latest one)."""
    years = np.asarray(years)
    return np.power(float(recency), years.max() - years)


def evaluate(train, ev, nxt, feats, seeds, target="target", recency=None):
    """Fit on `train`, predict xRV for the scored season and the one after.
    Returns pitcher tables for both, pitch-level R^2 on next season's actual
    run values, the last fitted model, and the two pitch-level predictions."""
    tr = train.dropna(subset=feats + [target])
    w = season_weights(tr["year"], recency) if recency else None
    p_ev = np.zeros(len(ev)); p_nx = np.zeros(len(nxt))
    for sd in seeds:
        m = make_model(sd).fit(tr[feats], tr[target].to_numpy(), sample_weight=w)
        p_ev += m.predict(ev[feats]) / len(seeds)
        p_nx += m.predict(nxt[feats]) / len(seeds)
    mu, sdv = p_ev.mean(), p_ev.std()
    a = pitcher_stuff(ev, p_ev, mu, sdv)
    b = pitcher_stuff(nxt, p_nx, mu, sdv)
    # pitch-level out-of-sample R^2 on the next season (tiny but comparable)
    y = nxt["target"].to_numpy()
    r2 = 1 - ((y - p_nx) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2, m, (100 - 10 * (p_ev - mu) / sdv, 100 - 10 * (p_nx - mu) / sdv)


def corr_table(a, b, out_nx, min_n):
    j = a.join(b, lsuffix="_y1", rsuffix="_y2", how="inner").join(out_nx, how="inner")
    j = j[(j["size_y1"] >= min_n) & (j["size_y2"] >= min_n)]
    return j.dropna(subset=["woba", "k_bb", "rv100"])


def metrics(j):
    return {
        "stickiness": j["mean_y1"].corr(j["mean_y2"]),
        "next_woba": -j["mean_y1"].corr(j["woba"]),     # sign-flipped: higher = better
        "next_k_bb": j["mean_y1"].corr(j["k_bb"]),
        "next_rv100": -j["mean_y1"].corr(j["rv100"]),
    }


def pitch_type_stickiness(ev, nxt, s_ev, s_nx, min_n):
    """r(Stuff+ year N, year N+1) per pitcher x pitch type, min_n of that type
    in both seasons."""
    def agg(df, s):
        return (pd.DataFrame({"pitcher": df["pitcher"].to_numpy(),
                              "pitch_type": df["pitch_type"].to_numpy(), "s": s})
                  .groupby(["pitcher", "pitch_type"])["s"].agg(["mean", "size"]))
    j = agg(ev, s_ev).join(agg(nxt, s_nx), lsuffix="_y1", rsuffix="_y2", how="inner")
    j = j[(j["size_y1"] >= min_n) & (j["size_y2"] >= min_n)].reset_index()
    out = {}
    for pt in PITCH_TYPES:
        g = j[j["pitch_type"] == pt]
        out[pt] = {"r": g["mean_y1"].corr(g["mean_y2"]) if len(g) > 2 else np.nan, "n": len(g)}
    return out


# ------------------------------------------------------------------ bootstrap
def _arrays(j):
    return [j[c].to_numpy(dtype=float) for c in ("mean_y1", "mean_y2", "woba", "k_bb", "rv100")]


def _corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return (x * y).sum() / d if d > 0 else np.nan


def _metric_vec(arr, idx):
    y1 = arr[0][idx]
    return np.array([_corr(y1, arr[1][idx]), -_corr(y1, arr[2][idx]),
                     _corr(y1, arr[3][idx]), -_corr(y1, arr[4][idx])])


def bootstrap_folds(base_tables, var_tables, n=1000, seed=0):
    """Paired bootstrap of (variant - base) for each metric, per fold and for
    the mean of the folds. Pitchers are the resampling unit and are drawn once
    per replicate across all folds, so a pitcher who appears in both folds is
    in or out of both. Returns point differences (on the pitchers common to
    variant and base) and 95% percentile intervals."""
    folds, ids_all = [], []
    for jb, jv in zip(base_tables, var_tables):
        ids = jb.index.intersection(jv.index)
        folds.append((ids, _arrays(jb.loc[ids]), _arrays(jv.loc[ids])))
        ids_all.append(ids.to_numpy())
    union = np.unique(np.concatenate(ids_all))
    pos = [ids.get_indexer(union) for ids, _, _ in folds]
    point = np.array([_metric_vec(av, np.arange(len(ids))) - _metric_vec(ab, np.arange(len(ids)))
                      for ids, ab, av in folds])
    rng = np.random.default_rng(seed)
    reps = np.empty((n, len(folds), len(KEYS)))
    for r in range(n):
        draw = rng.integers(0, len(union), len(union))
        for f, (_, ab, av) in enumerate(folds):
            p = pos[f][draw]
            p = p[p >= 0]
            reps[r, f] = _metric_vec(av, p) - _metric_vec(ab, p)

    def ci(x):
        lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=0)
        return {k: (float(lo[i]), float(hi[i])) for i, k in enumerate(KEYS)}

    return {"diff": [dict(zip(KEYS, map(float, p))) for p in point],
            "ci": [ci(reps[:, f]) for f in range(len(folds))],
            "mean_diff": dict(zip(KEYS, map(float, point.mean(0)))),
            "mean_ci": ci(reps.mean(1))}


def bootstrap_diff(j_base, j_var, n=1000, seed=0):
    """Single-fold paired bootstrap: {metric: (lo, hi)} for variant - base."""
    return bootstrap_folds([j_base], [j_var], n=n, seed=seed)["ci"][0]


# ------------------------------------------------------------------- grid PCA
def redundancy_report(df):
    """How many independent dimensions do the 18 grid angles really have?
    Returns the cumulative variance by principal component (standardized) and,
    for a direct reading in degrees, how much of each grid angle is left after
    a linear fit on the two centre-of-zone angles."""
    cols = GRID + ["release_extension"] if "release_extension" in df else GRID
    X = df[cols].dropna()
    X = X.sample(min(200_000, len(X)), random_state=0)
    G = X[GRID].to_numpy()
    Z = (G - G.mean(0)) / G.std(0)
    ev = np.linalg.svd(Z, full_matrices=False, compute_uv=False) ** 2
    cum = np.cumsum(ev / ev.sum())

    def resid_sd(extra):
        A = np.column_stack([X["vaa_mid_mid"], X["haa_mid_mid"], np.ones(len(X))] + extra)
        res = G - A @ np.linalg.lstsq(A, G, rcond=None)[0]
        return res.std(0)

    r2 = resid_sd([])
    detail = {
        "n": len(X),
        "grid_angle_sd_deg": [float(G.std(0).min()), float(G.std(0).max())],
        "max_resid_sd_deg_after_center_pair": float(r2.max()),
        "min_r2_after_center_pair": float((1 - (r2 / G.std(0)) ** 2).min()),
    }
    if "release_extension" in X:
        detail["max_resid_sd_deg_after_center_pair_plus_extension"] = float(
            resid_sd([X["release_extension"]]).max())
    return cum, detail


# ----------------------------------------------------------------------- main
def run_fold(df, fold, train_years, ev_year, variants, seeds, min_n):
    train = df[df["year"].isin(train_years)]
    ev = df[df["year"] == ev_year].dropna(subset=["target"])
    nxt = df[df["year"] == ev_year + 1].dropna(subset=["target"])
    out_nx = outcomes(nxt)
    results, tables = {}, {}
    print(f"\n=== fold {fold}: train {train_years} -> score {ev_year}, validate {ev_year + 1} ===")
    for name in variants:
        feats, spec = VARIANTS[name], SPECS.get(name, {})
        need = feats + [spec.get("target", "target")]
        missing = [f for f in need if f not in df or df[f].isna().all()]
        if missing:
            print(f"\n{name}: skipped (missing column: {missing})")
            continue
        ev_v, nx_v = ev.dropna(subset=feats), nxt.dropna(subset=feats)
        t1 = time.time()
        a, b, r2, model, (s_ev, s_nx) = evaluate(train, ev_v, nx_v, feats, seeds, **spec)
        j = corr_table(a, b, out_nx, min_n)
        tables[name] = j
        m = metrics(j); m["pitch_r2_next"] = r2; m["n_pitchers"] = len(j)
        m["n_train"] = int(train[need].notna().all(axis=1).sum())
        m["pitch_type_stickiness"] = pitch_type_stickiness(ev_v, nx_v, s_ev, s_nx, min_n)
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            m["top_features"] = [f for _, f in sorted(zip(imp, feats), reverse=True)[:6]]
        results[name] = m
        print(f"\n{name}  ({len(feats)} features, {time.time()-t1:.0f}s)", flush=True)
        for k in KEYS + ["pitch_r2_next"]:
            print(f"  {k:<14}{m[k]:+.4f}")
    return results, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--folds", nargs="*", type=int, default=[1], choices=list(FOLDS))
    ap.add_argument("--unlock-fold2", action="store_true",
                    help="fold 2 is the one-shot confirmation test (see docs/PREREGISTRATION.md)")
    ap.add_argument("--min-pitches", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1, help="average N model seeds")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if 2 in args.folds and not (args.synthetic or args.unlock_fold2):
        sys.exit("Fold 2 is locked until the final candidate is frozen (docs/PREREGISTRATION.md).")

    t0 = time.time()
    years = sorted({y for f in args.folds for y in FOLDS[f]["train"] + [FOLDS[f]["eval"], FOLDS[f]["eval"] + 1]})
    if args.synthetic:
        from stuffplus.synthetic import make
        df = add_features(add_expected_target(
            make(n_pitchers=150, pitches_per_season=400, years=tuple(years))))
    else:
        df = load(ROOT / "data", years)
    print(f"loaded {len(df):,} pitches in {time.time()-t0:.0f}s")
    print("env: " + ", ".join(f"{k} {v}" for k, v in env_info().items() if v))

    cum, pca = redundancy_report(df)
    print("\n18 grid angles -> cumulative variance by principal component:")
    print("  " + "  ".join(f"PC{i+1}:{c:.6f}" for i, c in enumerate(cum[:6])))
    print(f"  largest residual sd of any grid angle after a linear fit on the centre "
          f"VAA/HAA: {pca['max_resid_sd_deg_after_center_pair']:.4f} deg "
          f"(grid angle sd {pca['grid_angle_sd_deg'][0]:.2f}-{pca['grid_angle_sd_deg'][1]:.2f} deg)")

    seeds = list(range(args.seeds))
    fold_results, fold_tables = {}, {}
    for f in args.folds:
        fold_results[f], fold_tables[f] = run_fold(
            df, f, FOLDS[f]["train"], FOLDS[f]["eval"], args.variants, seeds, args.min_pitches)

    # variant minus baseline: per fold and mean of folds
    comparison = {}
    done = [n for n in args.variants if all(n in fold_tables[f] for f in args.folds)]
    if BASE in done:
        print(f"\nDifference vs {BASE} (paired bootstrap 95% CI, {args.boot} reps), "
              f"{'mean of folds ' + str(args.folds) if len(args.folds) > 1 else 'fold ' + str(args.folds[0])}:")
        for name in done:
            if name == BASE:
                continue
            bs = bootstrap_folds([fold_tables[f][BASE] for f in args.folds],
                                 [fold_tables[f][name] for f in args.folds], n=args.boot)
            comparison[name] = {"folds": {f: {"diff": bs["diff"][i], "ci": bs["ci"][i]}
                                          for i, f in enumerate(args.folds)},
                                "mean": {"diff": bs["mean_diff"], "ci": bs["mean_ci"]}}
            line = "  ".join(f"{k}: {bs['mean_diff'][k]:+.3f} [{lo:+.3f},{hi:+.3f}]"
                             for k, (lo, hi) in bs["mean_ci"].items())
            print(f"  {name:<24}{line}")

    (ROOT / "results").mkdir(exist_ok=True)
    tag = "synthetic" if args.synthetic else "folds" + "".join(map(str, args.folds))
    path = ROOT / "results" / f"ablation_{tag}.json"
    with open(path, "w") as f:
        json.dump({"args": vars(args), "seeds": seeds, "env": env_info(), "baseline": BASE,
                   "folds": {k: FOLDS[k] for k in args.folds}, "specs": SPECS,
                   "variants": {n: VARIANTS[n] for n in args.variants},
                   "grid_pca_cumvar": cum.tolist(), "grid_pca": pca,
                   "results": fold_results, "vs_baseline": comparison},
                  f, indent=2, default=float)
    print(f"\nsaved {path.relative_to(ROOT)}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
