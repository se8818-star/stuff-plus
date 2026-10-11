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
from webapp.grading import accel_to_movement  # noqa: E402

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
        # Savant gives "Last, First"
        parts = df["player_name"].fillna("").str.split(", ", n=1)
        df["player_name"] = parts.map(lambda p: f"{p[1]} {p[0]}" if len(p) == 2 else p[0])
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


def build(variant="A_tj_v3", synthetic=False, train=None, seed=0, out=None, synth_size=(220, 500)):
    t0 = time.time()
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

    feats = VARIANTS[variant]
    train = train or seasons
    tr = df[df["year"].isin(train)].dropna(subset=feats + ["target"])
    print(f"training {variant} on {train}: {len(tr):,} pitches, {len(feats)} features ...")
    t1 = time.time()
    model = make_model(seed).fit(tr[feats].to_numpy(dtype=float), tr["target"].to_numpy())
    print(f"  trained in {time.time()-t1:.0f}s")

    sc = df.dropna(subset=feats).copy()
    sc["pred"] = model.predict(sc[feats].to_numpy(dtype=float))
    scale = {int(y): {"mu": float(g["pred"].mean()), "sd": float(g["pred"].std(ddof=0))}
             for y, g in sc.groupby("year")}
    sc["stuff"] = 100 - 10 * (sc["pred"] - sc["year"].map(lambda y: scale[y]["mu"])) \
        / sc["year"].map(lambda y: scale[y]["sd"])

    # ay vs velocity, used to turn entered movement back into accelerations
    ay_coef = np.polyfit(tr["start_speed"], tr["ay"], 1).tolist()
    ay = np.polyval(ay_coef, sc["start_speed"])
    sc["ivb"], sc["hb"] = accel_to_movement(sc["az"], sc["ax"], sc["start_speed"],
                                            sc["extension"], ay)
    if {"pfx_x", "pfx_z"} <= set(sc.columns):
        hb_sc = np.where(sc["p_throws"] == "R", -sc["pfx_x"], sc["pfx_x"]) * 12
        print("  movement check vs Statcast pfx (median abs diff, in): "
              f"IVB {(sc['ivb'] - sc['pfx_z'] * 12).abs().median():.2f}, "
              f"HB {(sc['hb'] - hb_sc).abs().median():.2f}")

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
                 spin_axis=("spin_axis", "mean"), arm_angle=("arm_angle", "mean"))
            .reset_index())
    tot = sc.groupby(keys).agg(n=("stuff", "size"), stuff_plus=("stuff", "mean"),
                               hand=("p_throws", lambda s: s.mode().iat[0]),
                               name=("player_name", "first")).reset_index()
    pt = pt.merge(tot[keys + ["n", "name", "hand"]].rename(columns={"n": "n_total"}), on=keys)
    pt["usage"] = 100 * pt["n"] / pt["n_total"]

    fb = pt[pt["pitch_type"].isin(["FF", "SI", "FC"])].sort_values("n", ascending=False) \
        .drop_duplicates(keys)[keys + ["velo"]].rename(columns={"velo": "fb_velo"})
    tot = tot.merge(fb, on=keys, how="left")
    roles = infer_roles(sc)
    if roles is not None:
        tot = tot.merge(roles.reset_index(), on=keys, how="left")

    grades, pct = {}, {}
    for y in scale:
        ty, py = tot[tot["year"] == y], pt[pt["year"] == y]
        pq = ty[ty["n"] >= MIN_PITCHER_GRADE]["stuff_plus"]
        qq = py[py["n"] >= MIN_PITCH_GRADE]
        by_type = {k: {"mean": float(g["stuff_plus"].mean()), "sd": float(g["stuff_plus"].std())}
                   for k, g in qq.groupby("pitch_type") if len(g) >= 5}
        grades[y] = {"pitcher": {"mean": float(pq.mean()), "sd": float(pq.std())},
                     "pitch_type": by_type,
                     "all_pitches": {"mean": float(qq["stuff_plus"].mean()),
                                     "sd": float(qq["stuff_plus"].std())}}
        pct[y] = {"pitcher": np.sort(pq.to_numpy()),
                  "pitch_type": {k: np.sort(g["stuff_plus"].to_numpy())
                                 for k, g in qq.groupby("pitch_type")}}

    def grade_col(rows, which):
        out = []
        for y, s, t in zip(rows["year"], rows["stuff_plus"],
                           rows.get("pitch_type", pd.Series([None] * len(rows)))):
            c = grades[y]["pitcher"] if which == "pitcher" else \
                grades[y]["pitch_type"].get(t, grades[y]["all_pitches"])
            g = 50 + 10 * (s - c["mean"]) / c["sd"] if c["sd"] else np.nan
            out.append(int(np.clip(5 * round(g / 5), 20, 80)) if np.isfinite(g) else None)
        return out

    tot["grade"] = grade_col(tot, "pitcher")
    pt["grade"] = grade_col(pt, "pitch_type")

    import sklearn
    try:
        import lightgbm
        lgb = lightgbm.__version__
    except ImportError:
        lgb = None
    bundle = {
        "variant": variant, "features": feats, "model": model,
        "train_years": list(train), "seasons": [int(s) for s in scale],
        "latest_season": int(max(scale)), "scale": scale, "ay_coef": ay_coef,
        "arm_coef": arm_coef, "grades": grades, "percentiles": pct,
        "pitchers": tot, "pitch_types": pt, "synthetic": synthetic, "seed": seed,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lgb},
    }
    MODELS.mkdir(exist_ok=True)
    out = Path(out or bundle_path(variant, synthetic))
    joblib.dump(bundle, out, compress=3)
    print(f"saved {out}  (total {time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A_tj_v3", choices=list(VARIANTS))
    ap.add_argument("--train", nargs="*", type=int, help="training seasons (default: all)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.variant, a.synthetic, a.train, a.seed)


if __name__ == "__main__":
    main()
