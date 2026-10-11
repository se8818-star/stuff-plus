"""
Train a Stuff+ model and precompute everything the website shows.

    python -m webapp.build                       # real data in data/, all seasons
    python -m webapp.build --synthetic           # demo bundle, no data needed
    python -m webapp.build --variant B_plus_center --train 2022 2023 2024 2025

Trains one feature-set variant from features.py with run_ablation.py's model
settings, scores every season, and writes a single joblib bundle under
models/ (gitignored) with the model, scaling constants, leaderboards, and
20-80 grade constants. The existing ablation code is imported, not changed.

Leaderboards for seasons inside the training window are in-sample.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features import VARIANTS, load  # noqa: E402
from run_ablation import make_model  # noqa: E402
from webapp.grading import AveragedModel, accel_to_movement  # noqa: E402

MODELS = ROOT / "models"
MIN_PITCHER_GRADE = 100   # pitchers used for overall 20-80 constants
MIN_PITCH_GRADE = 50      # pitcher-pitch-types used for per-type constants


def bundle_path(variant, synthetic):
    return MODELS / f"stuffplus_site_{variant}{'_synthetic' if synthetic else ''}.joblib"


def load_synthetic(seasons, n_pitchers=220, pitches_per_season=500):
    from synthetic import make
    from features import add_features
    df = make(n_pitchers=n_pitchers, pitches_per_season=pitches_per_season,
              years=tuple(seasons), seed=7)
    df["player_name"] = "Synthetic Pitcher " + df["pitcher"].astype(str)
    return add_features(df)


def load_real(seasons):
    df = load(ROOT / "data", seasons)
    if "player_name" in df:
        df["player_name"] = flip_name(df["player_name"])
    return df


def infer_roles(df):
    """SP/RP per pitcher-season: a starter is the first pitcher seen from each
    side of a game. Needs game_pk / at_bat_number (real data only)."""
    if not {"game_pk", "at_bat_number"} <= set(df.columns):
        return None
    g = df[["game_pk", "at_bat_number", "pitch_number", "pitcher", "year"]] \
        .sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first = g.drop_duplicates(["game_pk", "pitcher"])
    sp1 = first.groupby("game_pk").nth(0)
    sp2 = first.groupby("game_pk").nth(1)
    starts = pd.concat([sp1, sp2]).groupby(["pitcher", "year"]).size()
    games = first.groupby(["pitcher", "year"]).size()
    gs = starts.reindex(games.index, fill_value=0)
    role = np.where(gs / games >= 0.5, "SP", "RP")
    return pd.DataFrame({"games": games, "gs": gs, "role": role})


def frozen_meta():
    """The shipped model's metadata (models/stuffplus_v<version>.json), if any."""
    metas = sorted(MODELS.glob("stuffplus_v*.json"))
    metas = [m for m in metas if (MODELS / f"{m.stem}.joblib").exists()]
    return metas[-1] if metas else None


def roles_by_season(df):
    """SP/RP per pitcher-season, same rule as final.roles: a pitcher starts a
    game if he threw his team's first pitch of it; SP if he started at least
    half of his games. Falls back to infer_roles without inning_topbot."""
    if "inning_topbot" not in df:
        return infer_roles(df)
    d = df[["game_pk", "inning_topbot", "at_bat_number", "pitch_number", "pitcher", "year"]]
    first = (d.sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
               .drop_duplicates(["game_pk", "inning_topbot"]))
    gs = first.groupby(["pitcher", "year"]).size()
    games = d.drop_duplicates(["game_pk", "pitcher"]).groupby(["pitcher", "year"]).size()
    gs = gs.reindex(games.index, fill_value=0)
    return pd.DataFrame({"games": games, "gs": gs, "role": np.where(gs / games >= 0.5, "SP", "RP")})


def shrink(raw, n, k, groups):
    """Empirical-Bayes shrinkage toward the (pitch-weighted) mean of each group,
    as in develop.shrink: mu + n/(n+k) (raw - mu)."""
    w = pd.DataFrame({"r": raw * n, "n": n})
    g = w.groupby([pd.Series(x).to_numpy() for x in groups])
    mu = g["r"].transform("sum") / g["n"].transform("sum")
    return mu + n / (n + k) * (raw - mu)


def flip_name(s):
    """Savant's "Last, First" -> "First Last"."""
    parts = s.fillna("").str.split(", ", n=1)
    return parts.map(lambda p: f"{p[1]} {p[0]}" if len(p) == 2 else p[0])


def to_grades(x, mean, sd):
    """20-80 scale like final.grade: 50 + 10 z, clipped, rounded."""
    return (50 + 10 * (x - mean) / sd).clip(20, 80).round()


def build(variant="A_tj_v3", synthetic=False, train=None, seed=0, out=None,
          synth_size=(220, 500), frozen=None):
    """Train `variant` (or load the frozen model whose metadata JSON is
    `frozen`), score every season and save the website bundle."""
    t0 = time.time()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in redirected logs
    if synthetic:
        seasons = [2024, 2025, 2026]
        df = load_synthetic(seasons, *synth_size)
    else:
        seasons = sorted(int(p.stem.split("_")[1]) for p in (ROOT / "data").glob("statcast_*.parquet"))
        if not seasons:
            raise SystemExit("No data/statcast_*.parquet found. Run `python fetch_data.py`, "
                             "or use --synthetic for a demo.")
        df = load_real(seasons)
    print(f"loaded {len(df):,} pitches ({seasons[0]}-{seasons[-1]}) in {time.time()-t0:.0f}s")

    meta, official = None, {}
    if frozen:
        meta = json.loads(Path(frozen).read_text())
        feats, train = meta["features"], meta["train_years"]
        model = AveragedModel(joblib.load(Path(frozen).with_suffix(".joblib")), feats)
        label = f"stuffplus v{meta['version']} (frozen model {meta['model']})"
        k_pitcher = meta["shrinkage"]["pitcher_k"]["k"]
        k_type = meta["shrinkage"]["pitch_type_k"]["k"]
        print(f"using {label}: {Path(frozen).name}")
    else:
        feats = VARIANTS[variant]
        train = train or seasons
        label = f"{variant} · trained {train[0]}–{train[-1]}"
        k_pitcher = k_type = None
        tr = df[df["year"].isin(train)].dropna(subset=feats + ["target"])
        print(f"training {variant} on {train}: {len(tr):,} pitches, {len(feats)} features ...")
        t1 = time.time()
        model = AveragedModel([make_model(seed).fit(tr[feats], tr["target"].to_numpy())], feats)
        print(f"  trained in {time.time()-t1:.0f}s")

    sc = df.dropna(subset=feats).copy()
    sc["pred"] = model.predict(sc[feats])
    scale = {int(y): {"mu": float(g["pred"].mean()), "sd": float(g["pred"].std(ddof=0))}
             for y, g in sc.groupby("year")}
    if meta:  # the frozen model's own published scaling for its season
        s = meta["scaling"]
        scale[int(s["season"])] = {"mu": s["model_xrv_mean"], "sd": s["model_xrv_sd"]}
    sc["stuff"] = 100 - 10 * (sc["pred"] - sc["year"].map(lambda y: scale[y]["mu"])) \
        / sc["year"].map(lambda y: scale[y]["sd"])

    # ay vs velocity, used to turn entered movement back into accelerations
    fit = sc.sample(min(len(sc), 500_000), random_state=0)
    ay_coef = np.polyfit(fit["start_speed"], fit["ay"], 1).tolist()
    ay = np.polyval(ay_coef, sc["start_speed"])
    sc["ivb"], sc["hb"] = accel_to_movement(sc["az"], sc["ax"], sc["start_speed"],
                                            sc["extension"], ay)
    if {"pfx_x", "pfx_z"} <= set(sc.columns):
        hb_sc = np.where(sc["p_throws"] == "R", -sc["pfx_x"], sc["pfx_x"]) * 12
        print("  movement check vs Statcast pfx (median abs diff, in): "
              f"IVB {(sc['ivb'] - sc['pfx_z'] * 12).abs().median():.2f}, "
              f"HB {(sc['hb'] - hb_sc).abs().median():.2f}")
    # features.py mirrors LHP spin axis; show and pre-fill the raw Savant value
    sc["spin_axis_raw"] = np.where(sc["p_throws"] == "L", (360 - sc["spin_axis"]) % 360, sc["spin_axis"])

    arm_coef = None
    a = sc[["arm_angle", "z0", "x0"]].dropna()
    if len(a) > 100:
        X = np.column_stack([np.ones(len(a)), a["z0"], a["x0"].abs()])
        arm_coef = np.linalg.lstsq(X, a["arm_angle"], rcond=None)[0].tolist()

    keys = ["pitcher", "year"]
    pt = (sc.groupby(keys + ["pitch_type"])
            .agg(n=("stuff", "size"), stuff_plus=("stuff", "mean"),
                 velo=("start_speed", "mean"), spin=("spin_rate", "mean"),
                 ext=("extension", "mean"), rel_side=("x0", "mean"),
                 rel_height=("z0", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"),
                 spin_axis=("spin_axis_raw", "mean"), arm_angle=("arm_angle", "mean"))
            .reset_index())
    tot = sc.groupby(keys).agg(n=("stuff", "size"), stuff_plus=("stuff", "mean"),
                               hand=("p_throws", lambda s: s.mode().iat[0]),
                               name=("player_name", "first")).reset_index()
    pt = pt.merge(tot[keys + ["n", "name", "hand"]].rename(columns={"n": "n_total"}), on=keys)
    pt["usage"] = 100 * pt["n"] / pt["n_total"]
    if k_pitcher is not None:  # shrink like the shipped leaderboards
        tot["stuff_plus_raw"], pt["stuff_plus_raw"] = tot["stuff_plus"], pt["stuff_plus"]
        tot["stuff_plus"] = shrink(tot["stuff_plus"], tot["n"], k_pitcher, [tot["year"]])
        pt["stuff_plus"] = shrink(pt["stuff_plus"], pt["n"], k_type, [pt["year"], pt["pitch_type"]])

    fb = pt[pt["pitch_type"].isin(["FF", "SI", "FC"])].sort_values("n", ascending=False) \
        .drop_duplicates(keys)[keys + ["velo"]].rename(columns={"velo": "fb_velo"})
    tot = tot.merge(fb, on=keys, how="left")
    roles = roles_by_season(sc)
    if roles is not None:
        tot = tot.merge(roles.reset_index(), on=keys, how="left")

    # The shipped model's published leaderboards (out-of-fold, shrunk) replace
    # the website's own numbers for that season, so the two always agree.
    if meta:
        y = int(meta["scaling"]["season"])
        fp, ft = ROOT / "leaderboards" / f"pitchers_{y}.csv", ROOT / "leaderboards" / f"pitch_types_{y}.csv"
        if fp.exists() and ft.exists():
            op = pd.read_csv(fp).assign(year=y)
            ot = pd.read_csv(ft).assign(year=y)
            cur = tot[tot["year"] == y].drop(columns=["stuff_plus", "stuff_plus_raw", "role", "games", "gs"],
                                             errors="ignore")
            cur = cur.drop(columns="n").merge(
                op[keys + ["rank", "pitches", "stuff_plus", "stuff_plus_raw", "grade", "role", "g", "gs"]]
                .rename(columns={"g": "games", "pitches": "n"}), on=keys, how="inner")
            cur = cur.sort_values("rank").drop(columns="rank")  # published order breaks ties
            tot = pd.concat([tot[tot["year"] != y], cur], ignore_index=True)
            curt = pt[pt["year"] == y].drop(columns=["stuff_plus", "stuff_plus_raw"], errors="ignore")
            curt = curt.merge(ot[keys + ["pitch_type", "rank_in_type", "pitches", "stuff_plus",
                                         "stuff_plus_raw", "grade"]],
                              on=keys + ["pitch_type"], how="left")
            curt["n"] = curt["pitches"].fillna(curt["n"]).astype(int)
            curt = curt.sort_values(["pitch_type", "rank_in_type"]).drop(columns=["pitches", "rank_in_type"])
            pt = pd.concat([pt[pt["year"] != y], curt], ignore_index=True)
            official = {y: f"{fp.relative_to(ROOT)}, {ft.relative_to(ROOT)}"}
            print(f"  {y}: using the published leaderboards ({official[y]})")

    grades, pct = {}, {}
    for y in scale:
        ty, py = tot[tot["year"] == y], pt[pt["year"] == y]
        pq = ty[ty["n"] >= MIN_PITCHER_GRADE]["stuff_plus"].dropna()
        qq = py[(py["n"] >= MIN_PITCH_GRADE) & py["stuff_plus"].notna()]
        by_type = {k: {"mean": float(g["stuff_plus"].mean()), "sd": float(g["stuff_plus"].std(ddof=0))}
                   for k, g in qq.groupby("pitch_type") if len(g) >= 5}
        grades[y] = {"pitcher": {"mean": float(pq.mean()), "sd": float(pq.std(ddof=0))},
                     "pitch_type": by_type,
                     "all_pitches": {"mean": float(qq["stuff_plus"].mean()),
                                     "sd": float(qq["stuff_plus"].std(ddof=0))}}
        pct[y] = {"pitcher": np.sort(pq.to_numpy()),
                  "pitch_type": {k: np.sort(g["stuff_plus"].to_numpy())
                                 for k, g in qq.groupby("pitch_type")}}

    def grade_col(rows, which):
        c = [grades[y]["pitcher"] if which == "pitcher" else
             grades[y]["pitch_type"].get(t, grades[y]["all_pitches"])
             for y, t in zip(rows["year"], rows.get("pitch_type", pd.Series([None] * len(rows))))]
        return to_grades(rows["stuff_plus"].to_numpy(dtype=float),
                         np.array([x["mean"] for x in c]), np.array([x["sd"] for x in c]))

    for rows, which in ((tot, "pitcher"), (pt, "pitch_type")):
        g = grade_col(rows, which)
        keep = rows["year"].isin(official) & rows["grade"].notna() if "grade" in rows else False
        rows["grade"] = np.where(keep, rows.get("grade", g), g)

    notes = {}
    for y in scale:
        if y in official:
            notes[y] = "Published leaderboard: out-of-fold scores, shrunk (" + official[y] + ")"
        elif y in train:
            notes[y] = "In-sample: the model was trained on this season" + (", scores shrunk" if meta else "")
        else:
            notes[y] = "Out-of-sample: the model never saw this season"

    import sklearn
    try:
        import lightgbm
        lgb = lightgbm.__version__
    except ImportError:
        lgb = None
    bundle = {
        "variant": meta["name"] + " v" + meta["version"] if meta else variant, "label": label,
        "frozen": bool(meta), "features": feats, "model": model,
        "train_years": list(train), "seasons": [int(s) for s in scale], "season_notes": notes,
        "latest_season": int(max(scale)), "scale": scale, "ay_coef": ay_coef,
        "arm_coef": arm_coef, "grades": grades, "percentiles": pct,
        "pitchers": tot, "pitch_types": pt, "synthetic": synthetic, "seed": seed,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lgb},
    }
    MODELS.mkdir(exist_ok=True)
    out = Path(out or bundle_path(Path(frozen).stem if frozen else variant, synthetic))
    joblib.dump(bundle, out, compress=3)
    print(f"saved {out}  (total {time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A_tj_v3", choices=list(VARIANTS))
    ap.add_argument("--train", nargs="*", type=int, help="training seasons (default: all)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                    help="train --variant even when a frozen model exists in models/")
    a = ap.parse_args()
    frozen = None if (a.synthetic or a.retrain) else frozen_meta()
    build(a.variant, a.synthetic, a.train, a.seed, frozen=frozen)


if __name__ == "__main__":
    main()
