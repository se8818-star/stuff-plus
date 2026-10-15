"""
Phase 4: train the shipped models on 2022-2026 and produce the 2026 outputs.

    python -m pipeline.final            # fits, saves models/, writes leaderboards/ and reports/

Two models, same feature set (A) and settings:

  Stuff+                   one model on every pitcher. It can remember the
                           pitchers it trained on, which makes it more
                           predictive, so it is only ever used on seasons after
                           its training data: the 2026 leaderboard comes from a
                           model trained on 2022-2025, and the saved model
                           (trained on 2022-2026) is for 2027 onward.
  Stuff+ (no remembering)  five fold models, each trained without a fifth of the
                           pitchers; a pitcher is only scored by a model that
                           never saw him. A pure grade of the physical pitch.

See reports/crossfit_test.md for what remembering is worth (about 0.037 in
next-season correlation) and reports/confirmation.md for why feature set A.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from stuffplus import run_ablation as ra
from pipeline.develop import pitcher_table, shrink, split_half_k, stuff_plus
from stuffplus.features import load
from stuffplus.models import (AMENDED, CANDIDATES, SEEDS, TJ_PARAMS, add_outcome_flags, crossfit_xrv,
                    fit_seed_models, predict_seed_models)

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.1.0"
MODEL = "A"
YEARS, SEASON = [2022, 2023, 2024, 2025, 2026], 2026
MIN_PITCHER, MIN_PITCH_TYPE = 100, 50
NAMES = {"": "Stuff+", "_no_remembering": "Stuff+ (no remembering)"}
PARK_MIN = 50


def grade(x):
    """20-80 scouting grade: 50 + 10 z across the qualified group, clipped."""
    z = (x - x.mean()) / x.std(ddof=0)
    return (50 + 10 * z).clip(20, 80)


def roles(df):
    """SP/RP per pitcher: a pitcher starts a game if he threw his team's first
    pitch of it; SP if he started at least half of his games."""
    d = df[["game_pk", "inning_topbot", "at_bat_number", "pitch_number", "pitcher"]]
    first = (d.sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
               .drop_duplicates(["game_pk", "inning_topbot"]))
    gs = first.groupby("pitcher").size().rename("gs")
    g = d.drop_duplicates(["game_pk", "pitcher"]).groupby("pitcher").size().rename("g")
    out = pd.concat([g, gs], axis=1).fillna({"gs": 0}).astype(int)
    out["role"] = np.where(out["gs"] / out["g"] >= 0.5, "SP", "RP")
    return out


def park_check(df, s, min_n=PARK_MIN):
    """For each park: the same pitchers' Stuff+ there minus elsewhere, averaged
    over pitchers with at least min_n pitches in both, weighted by the smaller
    of the two samples."""
    d = pd.DataFrame({"pitcher": df["pitcher"].to_numpy(), "park": df["home_team"].to_numpy(), "s": s}).dropna()
    tot = d.groupby("pitcher")["s"].agg(["sum", "size"])
    here = d.groupby(["park", "pitcher"])["s"].agg(["sum", "size"]).reset_index().join(tot, on="pitcher", rsuffix="_all")
    here["n_else"] = here["size_all"] - here["size"]
    here = here[(here["size"] >= min_n) & (here["n_else"] >= min_n)].copy()
    here["diff"] = here["sum"] / here["size"] - (here["sum_all"] - here["sum"]) / here["n_else"]
    here["w"] = here[["size", "n_else"]].min(axis=1)

    def agg(g):
        m = np.average(g["diff"], weights=g["w"])
        se = np.sqrt(np.average((g["diff"] - m) ** 2, weights=g["w"]) / max(len(g) - 1, 1))
        return pd.Series({"stuff_plus_diff": m, "se": se, "pitchers": len(g), "pitches_at_park": g["size"].sum()})

    return here.groupby("park").apply(agg).sort_values("stuff_plus_diff")


def leaderboards(d, s, k_p, k_pt, sfx, out):
    """Write the pitcher and pitch-type leaderboards for one score vector and
    return the checks that go in the model card."""
    role = roles(d)
    name = d.groupby("pitcher")[["player_name", "p_throws"]].first()
    pit = pitcher_table(d, s)
    pit["stuff_plus"] = shrink(pit, k_p)["mean"]
    pit = pit.rename(columns={"mean": "stuff_plus_raw", "size": "pitches"}).join(name).join(role)
    pit["velo"] = d.groupby("pitcher")["release_speed"].mean()
    q = pit[pit["pitches"] >= MIN_PITCHER].copy()
    q["grade"] = grade(q["stuff_plus"]).round(0).astype(int)
    q["grade_in_role"] = q.groupby("role")["stuff_plus"].transform(grade).round(0).astype(int)
    q = q.sort_values("stuff_plus", ascending=False).reset_index()
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    cols = ["rank", "pitcher", "player_name", "p_throws", "role", "g", "gs", "pitches", "velo",
            "stuff_plus", "stuff_plus_raw", "grade", "grade_in_role"]
    q[cols].round({"velo": 1, "stuff_plus": 1, "stuff_plus_raw": 1}).to_csv(out / f"pitchers_{SEASON}{sfx}.csv", index=False)

    dd = d.assign(_s=s)
    pt = pitcher_table(dd, s, ("pitcher", "pitch_type"))
    pt["stuff_plus"] = shrink(pt, k_pt, "pitch_type")["mean"]
    pt = pt.rename(columns={"mean": "stuff_plus_raw", "size": "pitches"})
    pt["velo"] = d.groupby(["pitcher", "pitch_type"])["release_speed"].mean()
    pt["usage_pct"] = pt["pitches"] / pt.groupby(level="pitcher")["pitches"].transform("sum") * 100
    pt = pt.reset_index().join(name, on="pitcher").join(role[["role"]], on="pitcher")
    qt = pt[(pt["pitches"] >= MIN_PITCH_TYPE) & pt["pitch_type"].isin(ra.PITCH_TYPES)].copy()
    qt["grade"] = qt.groupby("pitch_type")["stuff_plus"].transform(grade).round(0).astype(int)
    qt = qt.sort_values(["pitch_type", "stuff_plus"], ascending=[True, False])
    qt["rank_in_type"] = qt.groupby("pitch_type").cumcount() + 1
    qt[["pitch_type", "rank_in_type", "pitcher", "player_name", "p_throws", "role", "pitches", "usage_pct",
        "velo", "stuff_plus", "stuff_plus_raw", "grade"]].round(
        {"usage_pct": 1, "velo": 1, "stuff_plus": 1, "stuff_plus_raw": 1}).to_csv(
        out / f"pitch_types_{SEASON}{sfx}.csv", index=False)

    sp_rp = {r: {"pitchers": len(x), "pitches": int(x["pitches"].sum()),
                 "stuff_plus_pitch_weighted": float(np.average(x["stuff_plus"], weights=x["pitches"])),
                 "stuff_plus_mean": float(x["stuff_plus"].mean()), "stuff_plus_sd": float(x["stuff_plus"].std())}
             for r, x in q.groupby("role")}
    hand = {h: float(x["stuff_plus"].mean()) for h, x in q.groupby("p_throws")}
    by_type = qt.groupby("pitch_type").agg(pairs=("stuff_plus", "size"), mean=("stuff_plus", "mean"),
                                           sd=("stuff_plus", "std")).reindex(ra.PITCH_TYPES)
    parks = park_check(d, s)
    parks.round(3).to_csv(ROOT / "reports" / f"park_check_{SEASON}{sfx}.csv")
    return {"qualified_pitchers": len(q), "qualified_pitch_types": len(qt),
            "pitcher_stuff_plus_sd": float(q["stuff_plus"].std()), "velo_r": float(q["stuff_plus"].corr(q["velo"])),
            "sp_rp": sp_rp, "by_hand": hand, "by_pitch_type": by_type.to_dict("index"),
            "parks": parks.to_dict("index"),
            "top10": q.head(10)[["player_name", "role", "pitches", "stuff_plus", "grade"]].to_dict("records"),
            "top10_500": q[q["pitches"] >= 500].head(10)[["player_name", "role", "pitches", "stuff_plus", "grade"]].to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", help="JSON of LightGBM overrides (testing only)")
    args = ap.parse_args()
    params = json.loads(args.params) if args.params else None
    t0 = time.time()
    spec = CANDIDATES[MODEL]
    feats = spec["features"]
    df = add_outcome_flags(load(ROOT / "data", YEARS))
    df = df[df["target"].notna()].reset_index(drop=True)
    cur = (df["year"] == SEASON).to_numpy()
    print(f"loaded {len(df):,} pitches ({time.time() - t0:.0f}s)", flush=True)

    # 1. Stuff+ (no remembering): five fold models, each without a fifth of the pitchers
    fold_models, pitcher_fold = [], {}
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["pitcher"])):
        fold_models.append(fit_seed_models(spec, df.iloc[i_fit], params))
        pitcher_fold.update(dict.fromkeys((int(p) for p in df["pitcher"].iloc[i_hold].unique()), k))
        print(f"fold model {k + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
    oof = crossfit_xrv(fold_models, pitcher_fold, df, feats)

    # 2. Stuff+: prior seasons only for the leaderboard season, all seasons for the saved model
    forward = fit_seed_models(spec, df[~cur], params)
    fwd = predict_seed_models(forward, df[cur], feats)
    print(f"Stuff+ leaderboard model ({YEARS[0]}-{SEASON - 1}) fitted ({time.time() - t0:.0f}s)", flush=True)
    models = fit_seed_models(spec, df, params)
    print(f"Stuff+ saved model ({YEARS[0]}-{SEASON}) fitted ({time.time() - t0:.0f}s)", flush=True)

    s_all = stuff_plus(oof, oof)
    k_p = split_half_k(df, s_all, ["pitcher", "year"], 100)
    k_pt = split_half_k(df, s_all, ["pitcher", "year", "pitch_type"], 50)
    scale = {"": {"xrv_mean": float(np.nanmean(fwd)), "xrv_sd": float(np.nanstd(fwd))},
             "_no_remembering": {"xrv_mean": float(np.nanmean(oof[cur])), "xrv_sd": float(np.nanstd(oof[cur]))}}

    (ROOT / "models").mkdir(exist_ok=True)
    common = {
        "version": VERSION, "feature_set": MODEL, "features": feats, "target": spec["target"],
        "lightgbm_params": {**TJ_PARAMS, **(params or {})}, "seeds": list(SEEDS), "train_years": YEARS,
        "shrinkage": {"pitcher_k": k_p, "pitch_type_k": k_pt,
                      "formula": "league_mean + n / (n + k) * (raw_mean - league_mean)"},
        "feature_notes": {"ax, x0": "mirrored so both hands share a frame (see features.add_features)",
                          "*_diff": "difference from the pitcher-season's primary fastball (FF/SI/FC by usage)"},
        "env": ra.env_info()}
    for sfx, artifact, extra in (
        ("", {"models": models, "features": feats},
         {"scoring": "one model trained on every pitcher; it remembers them, so use it only on seasons after "
                     f"{SEASON}. The {SEASON} leaderboard was scored by a model trained on {YEARS[0]}-{SEASON - 1}.",
          "artifact": f"dict with models ({len(SEEDS)} LGBMRegressor, average them) and features"}),
        ("_no_remembering", {"fold_models": fold_models, "pitcher_fold": pitcher_fold, "features": feats},
         {"scoring": "cross-fitted: a pitcher in pitcher_fold is scored by the fold model that held him out; any "
                     "other pitcher by the mean of the five fold models (models.crossfit_xrv)",
          "n_pitchers": len(pitcher_fold),
          "artifact": f"dict with fold_models (5 folds x {len(SEEDS)} LGBMRegressor), pitcher_fold "
                      "(MLBAM id -> held-out fold) and features"})):
        tag = f"stuffplus{sfx}_v{VERSION}"
        joblib.dump(artifact, ROOT / "models" / f"{tag}.joblib")
        meta = {"name": NAMES[sfx], "file": f"models/{tag}.joblib (not in git)", **common, **extra,
                "scaling": {**scale[sfx], "season": SEASON,
                            "formula": "stuff_plus = 100 - 10 * (xRV - xrv_mean) / xrv_sd"}}
        (ROOT / "models" / f"{tag}.json").write_text(json.dumps(meta, indent=1, default=float))

    # 3. 2026 leaderboards and checks for both
    d = df[cur].copy()
    out = ROOT / "leaderboards"
    out.mkdir(exist_ok=True)
    scores = {"": stuff_plus(fwd, fwd), "_no_remembering": stuff_plus(oof[cur], oof[cur])}
    summary = {"version": VERSION, "season": SEASON, "min_pitcher": MIN_PITCHER, "min_pitch_type": MIN_PITCH_TYPE,
               "scaling": scale, "shrinkage": {"pitcher": k_p, "pitch_type": k_pt}, "models": {}}
    for sfx, s in scores.items():
        summary["models"][NAMES[sfx]] = leaderboards(d, s, k_p["k"], k_pt["k"], sfx, out)
    a = pd.read_csv(out / f"pitchers_{SEASON}.csv")
    b = pd.read_csv(out / f"pitchers_{SEASON}_no_remembering.csv")
    c = a[["pitcher", "player_name", "p_throws", "role", "pitches", "velo", "rank", "stuff_plus", "grade"]].merge(
        b[["pitcher", "rank", "stuff_plus", "grade"]], on="pitcher", suffixes=("", "_no_remembering"))
    c["difference"] = (c["stuff_plus"] - c["stuff_plus_no_remembering"]).round(1)
    c["known_before_2026"] = c["pitcher"].isin(set(df.loc[~cur, "pitcher"]))
    c.to_csv(out / f"both_models_{SEASON}.csv", index=False)
    big = c[c["pitches"] >= 500]
    summary["agreement"] = {
        "r": float(c["stuff_plus"].corr(c["stuff_plus_no_remembering"])),
        "r_known": float(c[c["known_before_2026"]][["stuff_plus", "stuff_plus_no_remembering"]].corr().iloc[0, 1]),
        "r_new": float(c[~c["known_before_2026"]][["stuff_plus", "stuff_plus_no_remembering"]].corr().iloc[0, 1]),
        "n_known": int(c["known_before_2026"].sum()), "n_new": int((~c["known_before_2026"]).sum()),
        "median_abs_diff": float(c["difference"].abs().median()),
        "share_over_3": float((c["difference"].abs() > 3).mean()),
        "top25_overlap_500": len(set(big.nsmallest(25, "rank")["pitcher"]) & set(big.nsmallest(25, "rank_no_remembering")["pitcher"])),
        "remembering_higher": big.nlargest(8, "difference")[["player_name", "pitches", "stuff_plus", "stuff_plus_no_remembering", "difference"]].to_dict("records"),
        "remembering_lower": big.nsmallest(8, "difference")[["player_name", "pitches", "stuff_plus", "stuff_plus_no_remembering", "difference"]].to_dict("records")}
    all_s = stuff_plus(oof, oof[cur])
    park_check(df, all_s).round(3).to_csv(ROOT / "reports" / "park_check_2022_2026_no_remembering.csv")
    summary["league_no_remembering_by_year"] = {int(y): float(np.nanmean(all_s[(df["year"] == y).to_numpy()])) for y in YEARS}
    summary["env"], summary["seconds"] = ra.env_info(), time.time() - t0
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "final_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({"agreement": {k: v for k, v in summary["agreement"].items() if not isinstance(v, list)},
                      "scaling": scale}, indent=1, default=float))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
