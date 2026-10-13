"""
Phase 4: train the shipped model on 2022-2026 and produce the 2026 outputs.

    python final.py            # fits, saves models/, writes leaderboards/ and reports/
    python final.py --model S6 # the six-variable comparison model, files suffixed _s6

The shipped model is A (tjStuff+ v3.0 feature set, TJ's LightGBM settings, 3
seeds averaged); see reports/confirmation.md for why. Since v1.1.0 the model is
cross-fitted: five fold models, each trained without a fifth of the pitchers
(GroupKFold by pitcher over 2022-2026), and a pitcher is only ever scored by a
model that never saw him. The saved bundle and the leaderboards use the same
scores.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import run_ablation as ra
from develop import pitcher_table, shrink, split_half_k, stuff_plus
from features import load
from models import (AMENDED, CANDIDATES, SEEDS, TJ_PARAMS, add_outcome_flags, crossfit_xrv,
                    fit_seed_models, predict_seed_models)

HERE = Path(__file__).parent
VERSION = "1.1.0"
MODEL = "A"
YEARS, SEASON = [2022, 2023, 2024, 2025, 2026], 2026
MIN_PITCHER, MIN_PITCH_TYPE = 100, 50
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", help="JSON of LightGBM overrides (testing only)")
    ap.add_argument("--model", default=MODEL, choices=[MODEL, "S6"],
                    help="A is the shipped model; S6 is the six-variable comparison model")
    args = ap.parse_args()
    primary = args.model == MODEL
    sfx = "" if primary else f"_{args.model.lower()}"
    params = json.loads(args.params) if args.params else None
    t0 = time.time()
    spec = {**CANDIDATES, **AMENDED}[args.model]
    feats = spec["features"]
    df = add_outcome_flags(load(HERE / "data", YEARS))
    df = df[df["target"].notna()].reset_index(drop=True)
    print(f"loaded {len(df):,} pitches ({time.time() - t0:.0f}s)", flush=True)

    # 1. five fold models, each trained without a fifth of the pitchers. Every
    #    pitch is scored by the model that never saw its pitcher (cross-fitted).
    fold_models, pitcher_fold = [], {}
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["pitcher"])):
        fold_models.append(fit_seed_models(spec, df.iloc[i_fit], params))
        pitcher_fold.update(dict.fromkeys((int(p) for p in df["pitcher"].iloc[i_hold].unique()), k))
        print(f"fold model {k + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
    oof = crossfit_xrv(fold_models, pitcher_fold, df, feats)

    # 2. for reference only: one model on all pitchers, which can remember them
    tr = df.dropna(subset=feats + [spec["target"]])
    models = fit_seed_models(spec, df, params)
    full = predict_seed_models(models, df, feats)
    print(f"all-pitcher reference model fitted ({time.time() - t0:.0f}s)", flush=True)

    cur = (df["year"] == SEASON).to_numpy()
    s_all = stuff_plus(oof, oof)                                  # for k only (scale-free)
    k_p = split_half_k(df, s_all, ["pitcher", "year"], 100)
    k_pt = split_half_k(df, s_all, ["pitcher", "year", "pitch_type"], 50)
    scale = {"model_xrv_mean": float(np.nanmean(full[cur])), "model_xrv_sd": float(np.nanstd(full[cur])),
             "oof_xrv_mean": float(np.nanmean(oof[cur])), "oof_xrv_sd": float(np.nanstd(oof[cur]))}

    (HERE / "models").mkdir(exist_ok=True)
    tag = f"stuffplus{sfx}_v{VERSION}"
    joblib.dump({"fold_models": fold_models, "pitcher_fold": pitcher_fold, "features": feats},
                HERE / "models" / f"{tag}.joblib")
    meta = {
        "name": f"stuffplus{sfx}", "version": VERSION, "model": args.model,
        "status": "shipped model" if primary else
                  "comparison model: not confirmed better than the shipped model (reports/confirmation.md)", "features": feats, "target": spec["target"],
        "lightgbm_params": {**TJ_PARAMS, **(params or {})}, "seeds": list(SEEDS),
        "train_years": YEARS, "n_train": int(len(tr)),
        "scoring": "cross-fitted: a pitcher in pitcher_fold is scored by the fold model that held him out; "
                   "any other pitcher by the mean of the five fold models (models.crossfit_xrv, score.py)",
        "n_pitchers": len(pitcher_fold),
        "scaling": {"xrv_mean": scale["oof_xrv_mean"], "xrv_sd": scale["oof_xrv_sd"], "season": SEASON,
                    "formula": "stuff_plus = 100 - 10 * (xRV - xrv_mean) / xrv_sd",
                    "all_pitcher_reference_model": {"xrv_mean": scale["model_xrv_mean"],
                                                    "xrv_sd": scale["model_xrv_sd"]}},
        "shrinkage": {"pitcher_k": k_p, "pitch_type_k": k_pt,
                      "formula": "league_mean + n / (n + k) * (raw_mean - league_mean)"},
        "feature_notes": {"ax, x0, haa_ln": "mirrored so both hands share a frame (see features.add_features)",
                          "vaa_ln, haa_ln": "approach angles re-aimed at the centre of the zone (physics.py)",
                          "*_diff": "difference from the pitcher-season's primary fastball (FF/SI/FC by usage)"},
        "artifact": f"models/{tag}.joblib: dict with fold_models (5 folds x {len(SEEDS)} LGBMRegressor), "
                    "pitcher_fold (MLBAM id -> held-out fold) and features",
        "env": ra.env_info()}
    (HERE / "models" / f"{tag}.json").write_text(json.dumps(meta, indent=1, default=float))

    # 3. 2026 leaderboards from out-of-fold scores
    d = df[cur].copy()
    d["s"] = stuff_plus(oof[cur], oof[cur])
    d["s_model"] = stuff_plus(full[cur], full[cur])
    role = roles(d)
    name = d.groupby("pitcher")[["player_name", "p_throws"]].first()

    pit = pitcher_table(d, d["s"].to_numpy())
    pit["stuff_plus"] = shrink(pit, k_p["k"])["mean"]
    pit = pit.rename(columns={"mean": "stuff_plus_raw", "size": "pitches"}).join(name).join(role)
    pit["velo"] = d.groupby("pitcher")["release_speed"].mean()
    q = pit[pit["pitches"] >= MIN_PITCHER].copy()
    q["grade"] = grade(q["stuff_plus"]).round(0).astype(int)
    q["grade_in_role"] = q.groupby("role")["stuff_plus"].transform(grade).round(0).astype(int)
    q = q.sort_values("stuff_plus", ascending=False).reset_index()
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    cols = ["rank", "pitcher", "player_name", "p_throws", "role", "g", "gs", "pitches", "velo",
            "stuff_plus", "stuff_plus_raw", "grade", "grade_in_role"]
    out = HERE / "leaderboards"
    out.mkdir(exist_ok=True)
    q[cols].round({"velo": 1, "stuff_plus": 1, "stuff_plus_raw": 1}).to_csv(out / f"pitchers_{SEASON}{sfx}.csv", index=False)

    pt = pitcher_table(d, d["s"].to_numpy(), ("pitcher", "pitch_type"))
    pt["stuff_plus"] = shrink(pt, k_pt["k"], "pitch_type")["mean"]
    pt = pt.rename(columns={"mean": "stuff_plus_raw", "size": "pitches"})
    g = d.groupby(["pitcher", "pitch_type"])
    pt["velo"] = g["release_speed"].mean()
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

    # 4. checks that go in the model card
    sp_rp = q.groupby("role").apply(lambda x: pd.Series({
        "pitchers": len(x), "pitches": x["pitches"].sum(),
        "stuff_plus_pitch_weighted": np.average(x["stuff_plus"], weights=x["pitches"]),
        "stuff_plus_mean": x["stuff_plus"].mean(), "stuff_plus_sd": x["stuff_plus"].std(),
        "velo": np.average(x["velo"], weights=x["pitches"])}))
    by_type = qt.groupby("pitch_type").agg(pairs=("stuff_plus", "size"), mean=("stuff_plus", "mean"),
                                           sd=("stuff_plus", "std")).reindex(ra.PITCH_TYPES)
    parks = park_check(d, d["s"].to_numpy())
    all_s = stuff_plus(oof, oof[cur])
    parks_all = park_check(df, all_s)
    parks.round(3).to_csv(HERE / "reports" / f"park_check_{SEASON}{sfx}.csv")
    parks_all.round(3).to_csv(HERE / "reports" / f"park_check_2022_2026{sfx}.csv")
    pm = pitcher_table(d, d["s_model"].to_numpy())["mean"]
    both = pit.join(pm.rename("in_sample"))
    both = both[both["pitches"] >= MIN_PITCHER]
    summary = {
        "version": VERSION, "model": args.model, "season": SEASON, "min_pitcher": MIN_PITCHER, "min_pitch_type": MIN_PITCH_TYPE,
        "qualified_pitchers": len(q), "qualified_pitch_types": len(qt),
        "scaling": scale, "shrinkage": {"pitcher": k_p, "pitch_type": k_pt},
        "pitcher_stuff_plus_sd": float(q["stuff_plus"].std()),
        "oof_vs_in_sample_r": float(both["stuff_plus_raw"].corr(both["in_sample"])),
        "in_sample_sd": float(both["in_sample"].std()), "oof_sd": float(both["stuff_plus_raw"].std()),
        "sp_rp": sp_rp.to_dict("index"), "by_pitch_type": by_type.to_dict("index"),
        "parks_2026": parks.to_dict("index"), "parks_2022_2026": parks_all.to_dict("index"),
        "top10": q.head(10)[["player_name", "role", "pitches", "stuff_plus", "grade"]].to_dict("records"),
        "league_stuff_plus_by_year": {int(y): float(np.nanmean(all_s[(df["year"] == y).to_numpy()])) for y in YEARS},
        "env": ra.env_info(), "seconds": time.time() - t0}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"final_summary{sfx}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("qualified_pitchers", "scaling", "oof_vs_in_sample_r", "sp_rp")},
                     indent=1, default=float))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
