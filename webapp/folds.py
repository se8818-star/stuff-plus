"""
Save a v1.0.0 leaderboard's out-of-fold models so the website can use them.

    python -m webapp.folds --model models/stuffplus_v1.0.0.json
                                # writes models/stuffplus_v1.0.0_folds.joblib

Only for v1.0.0 models. From v1.1.0, final.py saves the fold models itself in
models/stuffplus_no_remembering_v<version>.joblib and the website reads them
from there, so this step is not needed and exits straight away.

v1.0.0's final.py scores the published leaderboards out-of-fold: 5-fold GroupKFold by
pitcher over 2022-2026, each fold three LightGBM seeds averaged, so no pitcher
is scored by a model that saw his pitches. It keeps only the predictions. This
refits exactly those fold models (same pitches, folds, settings and seeds),
saves them with each pitcher's fold, and checks that they reproduce the
published leaderboard. final.py, the shipped model and leaderboards/ are not
changed. Run `python serve.py --rebuild` afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.grading import AveragedModel  # noqa: E402

N_SPLITS = 5


def folds_path(frozen):
    """Where the fold models of the frozen model `frozen` (its metadata JSON) live."""
    return Path(frozen).with_name(f"{Path(frozen).stem}_folds.joblib")


def fit_folds(df, feats, target, make, seeds, n_splits=N_SPLITS, log=print):
    """final.py's out-of-fold models: GroupKFold by pitcher over `df`'s rows in
    order; fold k is `make(seed)` fitted on the other folds' pitches, once per
    seed. Returns the fitted models (one list of seeds per fold) and the fold
    each pitcher was held out of."""
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    models, pitcher_fold = [], {}
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df["pitcher"])):
        tr = df.iloc[i_fit].dropna(subset=list(feats) + [target])
        models.append([make(s).fit(tr[feats], tr[target].to_numpy()) for s in seeds])
        pitcher_fold.update({int(p): k for p in df["pitcher"].iloc[i_hold].unique()})
        log(f"  fold {k + 1}/{n_splits} fitted ({time.time() - t0:.0f}s)")
    return models, pitcher_fold


def oof_predict(folds, pitcher_fold, X, pitchers, fallback):
    """Score each row with the fold model that never saw its pitcher. Rows of
    pitchers who are in no fold get `fallback` (the average of all folds)."""
    k = pd.Series(np.asarray(pitchers)).map(pitcher_fold).to_numpy(dtype=float)
    out = np.full(len(X), np.nan)
    for i, m in enumerate(folds):
        rows = np.flatnonzero(k == i)
        if len(rows):
            out[rows] = m.predict(X.iloc[rows])
    rest = np.flatnonzero(np.isnan(k))
    if len(rest):
        out[rest] = fallback.predict(X.iloc[rest])
    return out


def check_published(models, pitcher_fold, df, meta):
    """Re-score the published season out-of-fold, exactly as final.py does, and
    compare with leaderboards/pitchers_<season>.csv."""
    from develop import pitcher_table, stuff_plus
    feats, season = meta["features"], int(meta["scaling"]["season"])
    d = df[df["year"] == season]
    ok = d[feats].notna().all(axis=1).to_numpy()
    folds = [AveragedModel(ms, feats) for ms in models]
    ens = AveragedModel([m for ms in models for m in ms], feats)
    oof = np.full(len(d), np.nan)
    oof[ok] = oof_predict(folds, pitcher_fold, d.loc[ok, feats], d.loc[ok, "pitcher"], ens)
    out = {"season": season,
           "oof_xrv_mean": float(np.nanmean(oof)), "published_oof_xrv_mean": meta["scaling"]["oof_xrv_mean"],
           "oof_xrv_sd": float(np.nanstd(oof)), "published_oof_xrv_sd": meta["scaling"]["oof_xrv_sd"]}
    csv = ROOT / "leaderboards" / f"pitchers_{season}.csv"
    if csv.exists():
        pub = pd.read_csv(csv).set_index("pitcher")["stuff_plus_raw"]
        mine = pitcher_table(d, stuff_plus(oof, oof))["mean"].reindex(pub.index)
        diff = (mine - pub).abs()
        out.update(pitchers=int(diff.notna().sum()), max_abs_diff=float(diff.max()),
                   median_abs_diff=float(diff.median()),
                   reproduces=bool(diff.notna().all() and diff.max() <= 0.051))
    return out


def already_has_folds(meta, meta_path):
    """True for a v1.1.0+ model: its metadata has `xrv_mean` scaling rather than
    `oof_xrv_mean`, or its joblib is a dict holding `fold_models`."""
    if "xrv_mean" in meta.get("scaling", {}):
        return True
    artifact = Path(meta_path).with_suffix(".joblib")
    if artifact.exists():
        saved = joblib.load(artifact)
        return isinstance(saved, dict) and "fold_models" in saved
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path,
                    help="frozen model metadata JSON (default: the model the website would use)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    from webapp.build import frozen_meta
    meta_path = a.model or frozen_meta()
    if meta_path is None:
        raise SystemExit("No frozen model in models/. Run `python final.py` first.")
    meta = json.loads(Path(meta_path).read_text())
    if already_has_folds(meta, meta_path):          # before any data loading or fitting
        raise SystemExit(
            f"{Path(meta_path).name} is a v{meta.get('version', '1.1.0+')} model: the fold models are already in "
            "the no-remembering bundle (models/stuffplus_no_remembering_v<version>.joblib) and the website "
            "reads them from there. This step isn't needed; it is only for v1.0.0 models "
            "(`python -m webapp.folds --model models/stuffplus_v1.0.0.json`).")
    feats, target = meta["features"], meta["target"]

    # the same pitches, in the same order, as final.py
    from features import load
    from models import add_outcome_flags
    t0 = time.time()
    df = add_outcome_flags(load(ROOT / "data", meta["train_years"]))
    df = df[df["target"].notna()].reset_index(drop=True)
    print(f"loaded {len(df):,} pitches ({time.time() - t0:.0f}s); fitting {N_SPLITS} folds x "
          f"{len(meta['seeds'])} seeds of {Path(meta_path).stem} ...")

    from lightgbm import LGBMRegressor
    params = meta["lightgbm_params"]
    models, pitcher_fold = fit_folds(df, feats, target,
                                     lambda s: LGBMRegressor(**{**params, "random_state": s}), meta["seeds"])
    check = check_published(models, pitcher_fold, df, meta)
    print(json.dumps(check, indent=1))
    if not check.get("reproduces"):
        print("WARNING: these fold models do not reproduce the published leaderboard exactly. "
              "Has data/ or a library version changed since final.py ran?")

    import lightgbm
    import sklearn
    out = folds_path(meta_path)
    joblib.dump({"model": Path(meta_path).stem, "features": feats, "target": target,
                 "seeds": list(meta["seeds"]), "n_splits": N_SPLITS, "models": models,
                 "pitcher_fold": pitcher_fold, "check": check,
                 "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "versions": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}},
                out, compress=3)
    print(f"saved {out} ({time.time() - t0:.0f}s). Now run `python serve.py --rebuild`.")


if __name__ == "__main__":
    main()
