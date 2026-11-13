"""
Pre-contract 2025 Stuff+ for the free-agent AAV paper
(docs/PRECONTRACT_2025_PREREGISTRATION.md).

Model selection uses 2022-2024 only; the selected specification is then fit on
2022-2024 and scores 2025. No 2025 or 2026 outcome is read anywhere.

    python -m pipeline.precontract_2025 select    # candidates on 2022-24 -> reports/precontract_2025/model_selection.*
    python -m pipeline.precontract_2025 final     # fit the winner on 2022-24, score 2025 -> leaderboards/, reports/

Pitch-level predictions are cached in results/precontract_2025/ so the analysis
can be redone without refitting (delete the folder to refit).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipeline.develop import pitcher_table, shrink, split_half_k, stuff_plus
from pipeline.final import roles
from stuffplus import run_ablation as ra
from stuffplus.features import EXCLUDED_PITCH_TYPES, VARIANTS, add_features, load, plain_dtypes
from stuffplus.models import AMENDED, SEEDS, TJ_PARAMS, fit_predict, fit_seed_models, predict_seed_models

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_YEARS, SCORE_YEAR = [2022, 2023, 2024], 2025
KEY = ["game_pk", "at_bat_number", "pitch_number"]
MIN_PITCHES = [100, 300, 500]
N_CV, N_BOOT, BOOT_SEED = 5, 5000, 0
SEL_METRICS = ["next_woba", "next_rv100"]
BASE, CONTROL = "A", "A_reseeded"

# The candidates, all direct LightGBM on TJ's run value (pre-registered).
CANDIDATES = {
    "A": VARIANTS["A_tj_v3"],
    "B_plus_center": VARIANTS["B_plus_center"],
    "B9_plus_grid9": VARIANTS["B9_plus_grid9"],
    "C1_center_no_accel": VARIANTS["C1_center_no_accel"],
    "C_grid9_no_accel": VARIANTS["C_grid9_no_accel"],
    "D1_center_angles_only": VARIANTS["D1_center_angles_only"],
    "D_grid9_angles_only": VARIANTS["D_grid9_angles_only"],
    "S6": AMENDED["S6"]["features"],
}
SPECS = {n: dict(kind="direct", features=list(f), target="target") for n, f in CANDIDATES.items()}
SPECS[CONTROL] = dict(kind="direct", features=list(VARIANTS["A_tj_v3"]), target="target", seeds=(3, 4, 5))
CHALLENGERS = [n for n in CANDIDATES if n != BASE]
ALPHA = 0.05 / len(CHALLENGERS)          # Bonferroni over the challengers

# The three season pairs: (name, design, season N). Season N Stuff+ vs N+1 results.
PAIRS = [("forward_2023_2024", "forward", 2023),
         ("cv_2022_2023", "cv", 2022),
         ("cv_2023_2024", "cv", 2023)]

# 2025 columns that describe what happened to the pitch; dropped before scoring
OUTCOME_COLS = ["description", "events", "type", "bb_type", "zone", "plate_x", "plate_z",
                "delta_run_exp", "delta_pitcher_run_exp", "woba_value", "woba_denom",
                "estimated_woba_using_speedangle", "launch_speed", "launch_angle", "hit_distance_sc"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(years):
    out = {}
    for y in years:
        p = DATA / f"statcast_{y}.parquet"
        raw = pd.read_parquet(p, columns=["game_pk", "game_type", "pitcher"])
        out[str(y)] = {"file": p.name, "sha256": sha256(p), "rows": len(raw),
                       "games": int(raw["game_pk"].nunique()), "pitchers": int(raw["pitcher"].nunique()),
                       "game_types": sorted(raw["game_type"].astype(str).unique().tolist())}
    return out


def load_window():
    """2022-2024 pitches with a target. Opens no other season's file."""
    df = load(DATA, TRAIN_YEARS)
    assert set(df["year"].unique()) <= set(TRAIN_YEARS), "selection must only see 2022-2024"
    return df[df["target"].notna()].reset_index(drop=True)


# -------------------------------------------------------------------- fitting
def fit_candidate(name, df, out):
    """Forward xRV (train 2022 -> 2023, 2024) and cross-validated xRV (every
    2022-24 pitch, scored by a model that never saw the pitcher)."""
    t0, spec = time.time(), SPECS[name]
    tr = df[df["year"] == 2022]
    ev = df[df["year"].isin([2023, 2024])]
    (p_fwd,), info = fit_predict(spec, tr, [ev])
    oof = np.full(len(df), np.nan)
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=N_CV).split(df, groups=df["pitcher"])):
        (p,), _ = fit_predict(spec, df.iloc[i_fit], [df.iloc[i_hold]])
        oof[i_hold] = p
        print(f"  {name}: CV fold {k + 1}/{N_CV} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat([ev[KEY].assign(design="forward", xrv=p_fwd), df[KEY].assign(design="cv", xrv=oof)])
    preds.to_parquet(out, index=False)
    info = {**info, "seconds": time.time() - t0}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float))
    print(f"  {name}: saved ({info['seconds']:.0f}s)", flush=True)


# ------------------------------------------------------------------- analysis
def pair_tables(df, preds, design, y1):
    """Pitcher tables for season y1 and y1+1 (Stuff+ scaled on y1's pitches)
    and the pitch-level R^2 of the y1+1 predictions on y1+1 run values."""
    p = preds[preds["design"] == design]
    d = df[df["year"].isin([y1, y1 + 1])].merge(p[KEY + ["xrv"]], on=KEY, how="left", validate="1:1")
    first, nxt = d[d["year"] == y1], d[d["year"] == y1 + 1]
    ref = first["xrv"].to_numpy()
    a = pitcher_table(first, stuff_plus(ref, ref))
    b = pitcher_table(nxt, stuff_plus(nxt["xrv"].to_numpy(), ref))
    y, px = nxt["target"].to_numpy(), nxt["xrv"].to_numpy()
    ok = np.isfinite(px)
    r2 = float(1 - ((y[ok] - px[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())
    return a, b, r2, float(ok.mean())


def bootstrap_score(base_tables, var_tables, n=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """Paired bootstrap of (variant - base) for each metric and for the
    selection score (mean of next wOBA and next RV/100 over all pairs).
    Pitchers are resampled once per replicate across all pairs."""
    folds, ids_all = [], []
    for jb, jv in zip(base_tables, var_tables):
        ids = jb.index.intersection(jv.index)
        folds.append((ids, ra._arrays(jb.loc[ids]), ra._arrays(jv.loc[ids])))
        ids_all.append(ids.to_numpy())
    union = np.unique(np.concatenate(ids_all))
    pos = [ids.get_indexer(union) for ids, _, _ in folds]
    point = np.array([ra._metric_vec(av, np.arange(len(ids))) - ra._metric_vec(ab, np.arange(len(ids)))
                      for ids, ab, av in folds])
    rng = np.random.default_rng(seed)
    reps = np.empty((n, len(folds), len(ra.KEYS)))
    for r in range(n):
        draw = rng.integers(0, len(union), len(union))
        for f, (_, ab, av) in enumerate(folds):
            q = pos[f][draw]
            q = q[q >= 0]
            reps[r, f] = ra._metric_vec(av, q) - ra._metric_vec(ab, q)
    sel = [ra.KEYS.index(m) for m in SEL_METRICS]
    score_reps = reps[:, :, sel].mean(axis=(1, 2))
    lo, hi = np.nanpercentile(score_reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lo95, hi95 = np.nanpercentile(score_reps, [2.5, 97.5])
    per = {k: {"diff": float(point[:, i].mean()),
               "ci95": [float(x) for x in np.nanpercentile(reps[:, :, i].mean(1), [2.5, 97.5])]}
           for i, k in enumerate(ra.KEYS)}
    return {"score_diff": float(point[:, sel].mean()), "score_ci_adjusted": [float(lo), float(hi)],
            "score_ci95_unadjusted": [float(lo95), float(hi95)],
            "share_of_replicates_above_zero": float((score_reps > 0).mean()), "per_metric_mean_of_pairs": per,
            "per_pair": {PAIRS[f][0]: dict(zip(ra.KEYS, map(float, point[f]))) for f in range(len(folds))}}


def analyse(df, names):
    outs = {y: ra.outcomes(df[df["year"] == y]) for y in (2023, 2024)}
    res, tables = {}, {}
    for name in names:
        preds = pd.read_parquet(ROOT / "results" / "precontract_2025" / f"{name}.parquet")
        res[name], tables[name] = {"pairs": {}}, {}
        for pair, design, y1 in PAIRS:
            a, b, r2, cov = pair_tables(df, preds, design, y1)
            r = {"pitch_r2_next_season": r2, "coverage_next_season": cov}
            for mn in MIN_PITCHES:
                j = ra.corr_table(a, b, outs[y1 + 1], mn)
                tables[name][(pair, mn)] = j
                r[f"min_{mn}"] = {**{k: float(v) for k, v in ra.metrics(j).items()}, "pitchers": len(j)}
            res[name]["pairs"][pair] = r
        res[name]["selection_score"] = float(np.mean(
            [res[name]["pairs"][p]["min_100"][m] for p, _, _ in PAIRS for m in SEL_METRICS]))
        for mn in MIN_PITCHES:
            res[name][f"mean_of_pairs_min_{mn}"] = {
                k: float(np.mean([res[name]["pairs"][p][f"min_{mn}"][k] for p, _, _ in PAIRS])) for k in ra.KEYS}
    vs = {}
    for name in names:
        if name == BASE:
            continue
        vs[name] = bootstrap_score([tables[BASE][(p, 100)] for p, _, _ in PAIRS],
                                   [tables[name][(p, 100)] for p, _, _ in PAIRS])
        print(f"  bootstrap {name} done", flush=True)
    return res, vs


def decide(res, vs):
    ch = max(CHALLENGERS, key=lambda n: res[n]["selection_score"])
    lo, hi = vs[ch]["score_ci_adjusted"]
    adopt = res[ch]["selection_score"] > res[BASE]["selection_score"] and lo > 0
    return {"challenger": ch, "challenger_score": res[ch]["selection_score"],
            "baseline_score": res[BASE]["selection_score"], "difference": vs[ch]["score_diff"],
            "adjusted_interval": [lo, hi], "selected": ch if adopt else BASE,
            "reason": (f"{ch} beat A and its {100 * (1 - ALPHA):.2f}% interval excludes zero" if adopt else
                       f"best challenger {ch}: {'interval includes zero' if lo <= 0 else 'not above A'}; A kept")}


# --------------------------------------------------------------------- report
def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_selection_report(R, path):
    res, vs, dec = R["results"], R["vs_A"], R["decision"]
    H = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%", "next_rv100": "Next RV/100"}
    order = sorted(R["names"], key=lambda n: -res[n]["selection_score"])
    md = ["# Pre-contract 2025 Stuff+: model selection on 2022–2024", "",
          "Pre-registered in `docs/PRECONTRACT_2025_PREREGISTRATION.md`. Only 2022–2024 data were loaded. "
          "Correlations are sign-flipped so higher is better; pitchers need 100 pitches in both seasons "
          "unless stated.", "",
          f"**Selected: {dec['selected']}.** {dec['reason']}.", "",
          "## Selection score", "",
          "Mean of next-season wOBA and RV/100 correlations over the three season pairs. Interval: paired "
          f"bootstrap over pitchers, {N_BOOT} replicates, Bonferroni-adjusted "
          f"{100 * (1 - ALPHA):.2f}% (7 challengers).", ""]
    rows = []
    for n in order:
        v = vs.get(n)
        rows.append([n, f"{res[n]['selection_score']:.4f}",
                     f"{v['score_diff']:+.4f}" if v else "–",
                     f"[{v['score_ci_adjusted'][0]:+.4f}, {v['score_ci_adjusted'][1]:+.4f}]" if v else "–",
                     f"{v['share_of_replicates_above_zero']:.2f}" if v else "–"])
    md += [md_table(["Model", "Score", "vs A", "Adjusted interval", "Share of replicates > 0"], rows), ""]
    for mn in MIN_PITCHES:
        md += [f"## Mean over the three pairs, minimum {mn} pitches", "",
               md_table(["Model"] + [H[k] for k in ra.KEYS],
                        [[n] + [f"{res[n][f'mean_of_pairs_min_{mn}'][k]:.3f}" for k in ra.KEYS] for n in order]), ""]
    for pair, _, _ in PAIRS:
        md += [f"## {pair}, minimum 100 pitches", "",
               md_table(["Model", "Pitchers"] + [H[k] for k in ra.KEYS] + ["Pitch R² next season"],
                        [[n, str(res[n]["pairs"][pair]["min_100"]["pitchers"])]
                         + [f"{res[n]['pairs'][pair]['min_100'][k]:.3f}" for k in ra.KEYS]
                         + [f"{res[n]['pairs'][pair]['pitch_r2_next_season']:.5f}"] for n in order]), ""]
    md += ["Library versions: " + ", ".join(f"{k} {v}" for k, v in R["env"].items() if v), ""]
    path.write_text("\n".join(md))


# -------------------------------------------------------------------- commands
def cmd_select():
    t0 = time.time()
    pdir = ROOT / "results" / "precontract_2025"
    pdir.mkdir(parents=True, exist_ok=True)
    df = load_window()
    print(f"loaded {len(df):,} pitches with a target, seasons {sorted(df['year'].unique())}", flush=True)
    names = list(SPECS)
    for n in names:
        out = pdir / f"{n}.parquet"
        if out.exists():
            print(f"{n}: cached")
            continue
        print(f"{n}: fitting", flush=True)
        fit_candidate(n, df, out)
    res, vs = analyse(df, names)
    dec = decide(res, vs)
    R = {"design": "docs/PRECONTRACT_2025_PREREGISTRATION.md", "seasons_loaded": TRAIN_YEARS,
         "pairs": [{"name": p, "design": d, "season": y, "validated_against": y + 1} for p, d, y in PAIRS],
         "selection_metrics": SEL_METRICS, "min_pitches_for_selection": 100,
         "bootstrap": {"replicates": N_BOOT, "seed": BOOT_SEED, "alpha_per_comparison": ALPHA},
         "lightgbm_params": TJ_PARAMS, "seeds": list(SEEDS), "cv_folds": N_CV,
         "names": names, "specs": {n: {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in s.items()}
                                   for n, s in SPECS.items()},
         "fit_info": {n: json.loads((pdir / f"{n}.json").read_text()) for n in names},
         "results": res, "vs_A": vs, "decision": dec,
         "data": provenance(TRAIN_YEARS), "env": ra.env_info(), "seconds": time.time() - t0}
    rdir = ROOT / "reports" / "precontract_2025"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "model_selection.json").write_text(json.dumps(R, indent=1, default=float))
    write_selection_report(R, rdir / "model_selection.md")
    for n in sorted(names, key=lambda n: -res[n]["selection_score"]):
        print(f"{n:<24} {res[n]['selection_score']:.4f}")
    print(f"selected: {dec['selected']} ({dec['reason']}) ({time.time() - t0:.0f}s)")


def load_2025_without_outcomes():
    raw = plain_dtypes(pd.read_parquet(DATA / f"statcast_{SCORE_YEAR}.parquet"))
    raw = raw[raw["game_type"] == "R"]
    total = raw.groupby("pitcher").size().rename("total_pitches")
    df = raw.drop(columns=[c for c in OUTCOME_COLS if c in raw.columns])
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    df = add_features(df)
    assert (df["year"] == SCORE_YEAR).all()
    return df, total, raw


def describe(x):
    return {"N": int(x.notna().sum()), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "min": float(x.min()), "p25": float(x.quantile(0.25)), "median": float(x.median()),
            "p75": float(x.quantile(0.75)), "max": float(x.max())}


def cmd_final():
    t0 = time.time()
    sel = json.loads((ROOT / "reports" / "precontract_2025" / "model_selection.json").read_text())
    name = sel["decision"]["selected"]
    spec = SPECS[name]
    feats = spec["features"]
    df = load_window()
    print(f"selected {name}; loaded {len(df):,} training pitches", flush=True)

    # shrinkage constant from the cross-validated 2022-24 Stuff+ of the selected model
    preds = pd.read_parquet(ROOT / "results" / "precontract_2025" / f"{name}.parquet")
    cv = df.merge(preds[preds["design"] == "cv"][KEY + ["xrv"]], on=KEY, how="left", validate="1:1")
    s_cv = np.full(len(cv), np.nan)
    for y in TRAIN_YEARS:
        m = (cv["year"] == y).to_numpy()
        s_cv[m] = stuff_plus(cv.loc[m, "xrv"].to_numpy(), cv.loc[m, "xrv"].to_numpy())
    k_p = split_half_k(cv, s_cv, ["pitcher", "year"], 100)

    models = fit_seed_models(spec, df)
    print(f"fitted on 2022-2024 ({time.time() - t0:.0f}s)", flush=True)
    d25, total, raw25 = load_2025_without_outcomes()
    xrv = predict_seed_models(models, d25, feats)
    scale = {"xrv_mean": float(np.nanmean(xrv)), "xrv_sd": float(np.nanstd(xrv))}
    s = 100 - 10 * (xrv - scale["xrv_mean"]) / scale["xrv_sd"]

    pit = pitcher_table(d25, s)
    pit["stuff_plus"] = shrink(pit, k_p["k"])["mean"]
    pit = pit.rename(columns={"mean": "stuff_plus_raw", "size": "pitches_scored"})
    info = raw25.groupby("pitcher")[["player_name", "p_throws"]].first()
    out = (info.join(total).join(pit, how="left").join(roles(raw25)[["role", "g", "gs"]]))
    out["pitches_scored"] = out["pitches_scored"].fillna(0).astype(int)
    last_first = out["player_name"].astype(str)
    out["name"] = last_first.str.split(", ", n=1).map(lambda p: f"{p[1]} {p[0]}" if len(p) == 2 else p[0])
    out = out.reset_index().rename(columns={"pitcher": "mlbam_id", "player_name": "name_last_first"})
    cols = ["mlbam_id", "name", "name_last_first", "p_throws", "role", "g", "gs", "total_pitches",
            "pitches_scored", "stuff_plus", "stuff_plus_raw"]
    out = out[cols].sort_values(["stuff_plus", "total_pitches"], ascending=False)
    lb = ROOT / "leaderboards"
    out.round({"stuff_plus": 2, "stuff_plus_raw": 2}).to_csv(lb / "precontract_2025_pitchers.csv", index=False)

    rdir = ROOT / "reports" / "precontract_2025"
    scored = out[out["pitches_scored"] > 0]
    stats = {"season": SCORE_YEAR, "model": name, "variable": "stuff_plus (shrunk pitcher mean)",
             "all_pitchers": describe(scored["stuff_plus"]),
             **{f"min_{mn}_pitches": describe(scored.loc[scored["total_pitches"] >= mn, "stuff_plus"])
                for mn in (100, 300, 500, 1000)},
             "raw_all_pitchers": describe(scored["stuff_plus_raw"]),
             "pitchers_in_2025_data": int(len(out)), "pitchers_with_no_scorable_pitch": int((out["pitches_scored"] == 0).sum()),
             "pitches_2025_regular_season": int(total.sum()), "pitches_scored": int(np.isfinite(s).sum()),
             "league_pitch_mean_check": float(np.nanmean(s)), "league_pitch_sd_check": float(np.nanstd(s))}
    (rdir / "descriptive_stats_2025.json").write_text(json.dumps(stats, indent=1))
    rows = [[k.replace("_", " "), str(v["N"])] + [f"{v[c]:.2f}" for c in ("mean", "sd", "min", "p25", "median", "p75", "max")]
            for k, v in stats.items() if isinstance(v, dict)]
    (rdir / "descriptive_stats_2025.md").write_text("\n".join([
        "# 2025 Stuff+: descriptive statistics", "",
        f"Model {name}, trained on 2022–2024, scoring the 2025 regular season. Pitcher-level Stuff+ "
        f"(shrunk pitch-weighted mean, k = {k_p['k']:.1f} pitches). Subsets filter on total 2025 pitches. "
        "SD is the sample standard deviation across pitchers.", "",
        md_table(["Sample", "N", "Mean", "SD", "Min", "P25", "Median", "P75", "Max"], rows), "",
        f"{stats['pitches_2025_regular_season']:,} regular-season pitches in 2025; {stats['pitches_scored']:,} "
        "scored (the rest are excluded pitch types or have missing tracking data).", ""]))

    (ROOT / "models").mkdir(exist_ok=True)
    tag = "stuffplus_precontract_2025"
    joblib.dump({"models": models, "features": feats}, ROOT / "models" / f"{tag}.joblib")
    meta = {"name": "Stuff+ (pre-contract 2025)", "file": f"models/{tag}.joblib (not in git)",
            "feature_set": name, "features": feats, "target": spec["target"], "lightgbm_params": TJ_PARAMS,
            "seeds": list(spec.get("seeds", SEEDS)), "train_years": TRAIN_YEARS, "scored_season": SCORE_YEAR,
            "n_train": int(df.dropna(subset=feats + ["target"]).shape[0]),
            "scaling": {**scale, "season": SCORE_YEAR, "formula": "stuff_plus = 100 - 10 * (xRV - xrv_mean) / xrv_sd"},
            "shrinkage": {"pitcher_k": k_p, "formula": "league_mean + n / (n + k) * (raw_mean - league_mean)"},
            "outcome_columns_dropped_from_2025": OUTCOME_COLS,
            "data": provenance(TRAIN_YEARS + [SCORE_YEAR]), "env": ra.env_info()}
    (ROOT / "models" / f"{tag}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(stats["all_pitchers"], indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["select", "final"])
    args = ap.parse_args()
    cmd_select() if args.step == "select" else cmd_final()


if __name__ == "__main__":
    main()
