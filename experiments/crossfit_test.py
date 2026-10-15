"""
Exploratory analysis E2 (docs/EXPLORATORY.md): cross-fitted scoring (no memory of
the pitcher) against the remembering model, forward in time.

    python -m experiments.crossfit_test     # results/crossfit_test.json, reports/crossfit_test.md
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from stuffplus import run_ablation as ra
from pipeline.develop import KEY, MIN_PITCHES, N_BOOT, pitcher_table, stuff_plus, table
from stuffplus.features import load
from stuffplus.models import CANDIDATES, add_outcome_flags, fit_predict

ROOT = Path(__file__).resolve().parent.parent
FIRST, EVAL_YEARS, N_FOLDS = 2020, [2022, 2023, 2024, 2025], 5
METHODS = ("remember", "crossfit")


def crossfit_predict(spec, train, frames, n_folds=N_FOLDS, params=None, log=lambda m: None):
    """xRV for each frame from models that never saw the pitcher: a pitcher in
    `train` is scored by the fold model that held him out, anyone else by the
    mean of all fold models."""
    fold_of = {}
    preds = [np.zeros((n_folds, len(f))) for f in frames]
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=n_folds).split(train, groups=train["pitcher"])):
        fold_of.update(dict.fromkeys(train["pitcher"].iloc[i_hold].unique(), k))
        out, _ = fit_predict(spec, train.iloc[i_fit], frames, params)
        for p, o in zip(preds, out):
            p[k] = o
        log(f"fold {k + 1}/{n_folds}")
    res = []
    for f, p in zip(frames, preds):
        k = f["pitcher"].map(fold_of).to_numpy(dtype=float)
        known = np.isfinite(k)
        x = p.mean(axis=0)
        x[known] = p[k[known].astype(int), np.flatnonzero(known)]
        res.append(x)
    return res, fold_of


def main():
    t0 = time.time()
    years = list(range(FIRST, EVAL_YEARS[-1] + 2))
    df = add_outcome_flags(load(ROOT / "data", years))
    df = df[df["target"].notna()].reset_index(drop=True)
    spec = CANDIDATES["A"]
    pdir = ROOT / "results" / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    res, tables, agree = {}, {}, {}
    for y in EVAL_YEARS:
        tr, ev, nx = df[df["year"] < y], df[df["year"] == y], df[df["year"] == y + 1]
        cache = pdir / f"crossfit_{y}.parquet"
        if cache.exists():
            c = pd.read_parquet(cache)
        else:
            log = lambda m: print(f"{y}: {m} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
            (m1, m2), _ = fit_predict(spec, tr, [ev, nx])
            log("remembering model fitted")
            (c1, c2), _ = crossfit_predict(spec, tr, [ev, nx], log=log)
            c = pd.concat([ev[KEY].assign(remember=m1, crossfit=c1), nx[KEY].assign(remember=m2, crossfit=c2)])
            c.to_parquet(cache, index=False)
        ev = ev.merge(c, on=KEY, validate="1:1"); nx = nx.merge(c, on=KEY, validate="1:1")
        seen = set(tr["pitcher"])
        out_next = ra.outcomes(nx)
        res[y] = {"train": [FIRST, y - 1], "n_train_pitchers": len(seen)}
        tabs = {}
        for m in METHODS:
            a = pitcher_table(ev, stuff_plus(ev[m].to_numpy(), ev[m].to_numpy()))
            b = pitcher_table(nx, stuff_plus(nx[m].to_numpy(), ev[m].to_numpy()))
            tabs[m] = a
            for mn in MIN_PITCHES:
                j = ra.corr_table(a, b, out_next, mn)
                for grp, jj in (("all", j), ("known", j[j.index.isin(seen)]), ("new", j[~j.index.isin(seen)])):
                    tables[(m, y, mn, grp)] = jj
                    res[y][f"{m}_{grp}_{mn}"] = {**ra.metrics(jj), "n": len(jj)}
        q = tabs["remember"].join(tabs["crossfit"], lsuffix="_r", rsuffix="_c")
        q = q[q["size_r"] >= 100]
        agree[y] = {"r": float(q["mean_r"].corr(q["mean_c"])),
                    "median_abs_diff": float((q["mean_r"] - q["mean_c"]).abs().median()),
                    "sd_remember": float(q["mean_r"].std()), "sd_crossfit": float(q["mean_c"].std())}
        print(f"{y}: done ({time.time() - t0:.0f}s)", flush=True)

    vs = {}
    for grp in ("all", "known"):
        for mn in MIN_PITCHES:
            bs = ra.bootstrap_folds([tables[("remember", y, mn, grp)] for y in EVAL_YEARS],
                                    [tables[("crossfit", y, mn, grp)] for y in EVAL_YEARS], n=N_BOOT)
            vs[f"{grp}_{mn}"] = {"mean": {"diff": bs["mean_diff"], "ci": bs["mean_ci"]},
                                 "splits": {str(y): {"diff": bs["diff"][i], "ci": bs["ci"][i]}
                                            for i, y in enumerate(EVAL_YEARS)}}
    p = vs["all_100"]["mean"]
    lows = [p["ci"][k][0] for k in ("next_woba", "next_rv100")]
    highs = [p["ci"][k][1] for k in ("next_woba", "next_rv100")]
    verdict = {"close": bool(min(lows) > -0.03),
               "as_predictive": bool(all(lo <= 0 <= hi for lo, hi in zip(lows, highs)))}
    means = {m: {mn: {k: float(np.mean([res[y][f"{m}_all_{mn}"][k] for y in EVAL_YEARS])) for k in ra.KEYS}
                 for mn in MIN_PITCHES} for m in METHODS}
    out = {"eval_years": EVAL_YEARS, "results": res, "means": means, "crossfit_minus_remember": vs,
           "agreement": agree, "verdict": verdict, "env": ra.env_info(), "seconds": time.time() - t0}
    (ROOT / "results" / "crossfit_test.json").write_text(json.dumps(out, indent=1, default=float))

    H = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%", "next_rv100": "Next RV/100"}
    ci = lambda d, k: f"{d['diff'][k]:+.3f} [{d['ci'][k][0]:+.3f}, {d['ci'][k][1]:+.3f}]"  # noqa: E731
    md = ["# E2: cross-fitted scoring against the remembering model, forward in time", "",
          "Exploratory (plan in `docs/EXPLORATORY.md`, committed before this was run). Feature set A, TJ's settings, "
          "3 seeds. Each split trains on every season from 2020 up to the year before the scored one. "
          "*Remembering* is one model on all training pitchers; *cross-fitted* scores each pitcher with a fold "
          "model that never saw him. Differences are cross-fitted minus remembering with paired-bootstrap 95% CIs.", "",
          "## Primary comparison: mean of the four splits", "",
          table(["Min pitches"] + [H[k] for k in ra.KEYS],
                [[str(mn)] + [ci(vs[f"all_{mn}"]["mean"], k) for k in ra.KEYS] for mn in MIN_PITCHES]), "",
          f"Pre-set reading at 100 pitches: close (both lower bounds above −0.03): **{'yes' if verdict['close'] else 'no'}**; "
          f"as predictive (both intervals include zero): **{'yes' if verdict['as_predictive'] else 'no'}**.", "",
          "Only pitchers who were in the training seasons:", "",
          table(["Min pitches"] + [H[k] for k in ra.KEYS],
                [[str(mn)] + [ci(vs[f"known_{mn}"]["mean"], k) for k in ra.KEYS] for mn in MIN_PITCHES]), "",
          "## Each method, mean of the four splits", "",
          table(["Method", "Min pitches"] + [H[k] for k in ra.KEYS],
                [[m, str(mn)] + [f"{means[m][mn][k]:.3f}" for k in ra.KEYS] for mn in MIN_PITCHES for m in METHODS]), "",
          "## Each split, minimum 100 pitches", "",
          table(["Scored → validated", "Trained on", "Pitchers (known / new)", "Remembering RV/100", "Cross-fitted RV/100",
                 "Difference", "Remembering stickiness", "Cross-fitted stickiness", "r between the two scores"],
                [[f"{y} → {y + 1}", f"{FIRST}–{y - 1}",
                  f"{res[y]['remember_known_100']['n']} / {res[y]['remember_new_100']['n']}",
                  f"{res[y]['remember_all_100']['next_rv100']:.3f}", f"{res[y]['crossfit_all_100']['next_rv100']:.3f}",
                  ci(vs["all_100"]["splits"][str(y)], "next_rv100"),
                  f"{res[y]['remember_all_100']['stickiness']:.3f}", f"{res[y]['crossfit_all_100']['stickiness']:.3f}",
                  f"{agree[y]['r']:.3f}"] for y in EVAL_YEARS]), "",
          "Library versions: " + ", ".join(f"{k} {v}" for k, v in out["env"].items() if v), ""]
    (ROOT / "reports" / "crossfit_test.md").write_text("\n".join(md))
    print("\n".join(md[4:]))


if __name__ == "__main__":
    main()
