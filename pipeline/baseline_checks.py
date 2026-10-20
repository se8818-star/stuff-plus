"""
Phase 2 sanity checks on the tjStuff+ v3.0 baseline (variant A, fold 1):
no-fastball handling, seeds, LightGBM settings, handedness mirroring, and a
comparison with TJ's published 2024 scores (data/external/, from his repo).

    python -m pipeline.baseline_checks        # writes results/phase2_baseline_checks.json
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from stuffplus import run_ablation as ra
from stuffplus.features import VARIANTS, load
df = load(Path("data"), [2022, 2023, 2024, 2025])
feats = VARIANTS["A_tj_v3"]
out = {}
def run(df, label, seeds=(0,), min_n=100, model_kw=None, keep=False):
    train = df[df.year.isin([2022, 2023])]
    ev = df[df.year == 2024].dropna(subset=["target"] + feats)
    nx = df[df.year == 2025].dropna(subset=["target"] + feats)
    if model_kw is not None:
        orig = ra.make_model
        from lightgbm import LGBMRegressor
        ra.make_model = lambda s: LGBMRegressor(random_state=s, verbose=-1, **model_kw)
    a, b, r2, m = ra.evaluate(train, ev, nx, feats, list(seeds))
    if model_kw is not None: ra.make_model = orig
    res = {}
    for mn in ([min_n] if isinstance(min_n, int) else min_n):
        j = ra.corr_table(a, b, ra.outcomes(df[df.year == 2025].dropna(subset=["target"])), mn)
        r = ra.metrics(j); r["n"] = len(j); res[mn] = r
        print(f"{label:<44} min{mn:<4} n={len(j):<4} " + "  ".join(f"{k}={v:+.3f}" for k, v in r.items() if k != "n"), flush=True)
    out[label] = res
    return a, b, ev, nx, m
a, b, ev, nx, m = run(df, "A as-is (1 seed)", min_n=[100, 300, 500, 1000])
print("rows: train", len(df[df.year.isin([2022,2023])].dropna(subset=feats+["target"])), "fb_fallback share by year:", df.groupby("year").fb_fallback.mean().round(4).to_dict(),
      "fallback pitcher-seasons:", df[df.fb_fallback].groupby("year").pitcher.nunique().to_dict())
print("feature importance:", dict(zip(feats, m.feature_importances_)))
run(df[~df.fb_fallback], "A, no-fastball pitchers dropped (TJ join)")
run(df, "A, 3 seeds averaged", seeds=(0, 1, 2))
run(df, "A, seed 42 (TJ's)", seeds=(42,))
run(df, "A, real bagging (subsample_freq=1)", model_kw=dict(n_estimators=1000, learning_rate=0.01, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2, force_row_wise=True))
run(df, "A, LightGBM defaults (100 trees, lr 0.1)", model_kw=dict())
# handedness
h = df.drop_duplicates("pitcher").set_index("pitcher").p_throws
j = a.join(b, lsuffix="_y1", rsuffix="_y2", how="inner"); j = j[(j.size_y1 >= 100) & (j.size_y2 >= 100)]; j["hand"] = h
for hd, g in j.groupby("hand"):
    print(f"hand {hd}: n={len(g)} stickiness={g.mean_y1.corr(g.mean_y2):+.3f} mean Stuff+ 2024={g.mean_y1.mean():.2f}")
# mirrored features by hand
print(df[df.pitch_type == "FF"].groupby("p_throws")[["ax", "x0", "az", "start_speed"]].mean().round(2))
# un-mirrored control: what if handedness mirroring were broken
d2 = df.copy(); l = d2.p_throws == "L"; d2.loc[l, "ax"] = -d2.loc[l, "ax"]; d2.loc[l, "x0"] = -d2.loc[l, "x0"]
d2["ax_diff"] = (d2.ax - np.where(l, -d2.fb_ax, d2.fb_ax)).abs()
run(d2, "control: mirroring removed")
# TJ's x0/z0 come from the MLB API, i.e. the fit at y=50 rather than at release
from stuffplus import physics
d3 = df.copy()
t_r = physics._time_to_y(physics.Y50, d3.vy0.to_numpy(), d3.ay.to_numpy(), 60.5 - d3.release_extension.to_numpy())
raw_ax = np.where(d3.p_throws == "L", -d3.ax, d3.ax)
x50 = d3.release_pos_x - d3.vx0 * t_r - 0.5 * raw_ax * t_r ** 2
d3["x0"] = np.where(d3.p_throws == "L", x50, -x50)
d3["z0"] = d3.release_pos_z - d3.vz0 * t_r - 0.5 * d3.az * t_r ** 2
a50, *_ = run(d3, "A, x0/z0 at y=50 (MLB API definition)")
# external: TJ's published 2024 tjStuff+
tj = pd.read_csv("data/external/tjstuff_plus_pitch_data_2024.csv"); tj = tj[tj.pitch_type == "All"].set_index("pitcher_id")
for lab, aa in (("release-point x0/z0", a), ("y=50 x0/z0", a50)):
  k = aa.join(tj[["tj_stuff_plus", "pitches"]], how="inner")
  for mn in (100, 500):
    kk = k[(k["size"] >= mn) & (k.pitches >= mn)]
    out[f"vs_tj_published_{lab}_min{mn}"] = {"n": len(kk), "pearson": kk["mean"].corr(kk.tj_stuff_plus)}
    print(f"[{lab}] vs TJ published 2024 tjStuff+ (min {mn}): n={len(kk)} pearson={kk['mean'].corr(kk.tj_stuff_plus):.3f} spearman={kk['mean'].corr(kk.tj_stuff_plus, method='spearman'):.3f} sd mine={kk['mean'].std():.2f} sd TJ={kk.tj_stuff_plus.std():.2f} pitches ratio={ (kk['size']/kk.pitches).median():.3f}")
out["env"] = ra.env_info()
Path("results").mkdir(exist_ok=True); json.dump(out, open("results/phase2_baseline_checks.json", "w"), indent=1, default=float)
