"""
The pre-registered candidates (docs/PREREGISTRATION.md) and how each one turns
training pitches into expected run value (xRV) per pitch.

  direct     one LightGBM regression of the target on the features
  component  P(swing), P(whiff | swing), P(called strike | take) and
             E[runs | contact], combined into expected runs per pitch
"""
import numpy as np
import pandas as pd

from stuffplus.features import TJ_V3

TJ_PARAMS = dict(n_estimators=1000, learning_rate=0.01, num_leaves=31, max_depth=-1,
                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                 reg_lambda=0.2, force_row_wise=True, deterministic=True, verbose=-1)
SEEDS = (0, 1, 2)
JGH = TJ_V3 + ["same_side", "ssw_dev"]

CANDIDATES = {
    "A": dict(kind="direct", features=TJ_V3, target="target"),
    "J": dict(kind="direct", features=TJ_V3, target="target_x"),
    "G": dict(kind="direct", features=TJ_V3 + ["same_side"], target="target"),
    "H": dict(kind="direct", features=TJ_V3 + ["ssw_dev"], target="target"),
    "JGH": dict(kind="direct", features=JGH, target="target_x"),
    "COMP": dict(kind="component", features=JGH, target="target_x"),
}
# Added after stage 1 at the owner's request (docs/PREREGISTRATION.md, Deviations):
# a six-variable model on location-neutral approach angles.
AMENDED = {
    "S6": dict(kind="direct", target="target",
               features=["start_speed", "extension", "vaa_ln", "haa_ln", "vaa_diff", "haa_diff"]),
}
# Null control, not a candidate: A with different seeds
CONTROLS = {"A_reseeded": dict(kind="direct", features=TJ_V3, target="target", seeds=(3, 4, 5))}

WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip", "missed_bunt"}
SWING = WHIFF | {"foul", "foul_bunt", "foul_pitchout", "hit_into_play"}
TAKE = {"ball", "blocked_ball", "pitchout", "called_strike", "hit_by_pitch"}


def add_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
    """is_swing / is_whiff / is_cs as floats; NaN where the description is
    neither a swing nor a take (e.g. automatic balls)."""
    d = df["description"]
    known = d.isin(SWING | TAKE)
    df["is_swing"] = np.where(known, d.isin(SWING) * 1.0, np.nan)
    df["is_whiff"] = np.where(d.isin(SWING), d.isin(WHIFF) * 1.0, np.nan)
    df["is_cs"] = np.where(d.isin(TAKE), (d == "called_strike") * 1.0, np.nan)
    return df


def _params(params, seed, component=None):
    """TJ's settings with any overrides. Tuned component models carry one tree
    count per component in `n_estimators_by_component`."""
    p = {**TJ_PARAMS, **(params or {}), "random_state": seed}
    by_component = p.pop("n_estimators_by_component", None)
    if by_component and component:
        p["n_estimators"] = by_component[component]
    return p


def _fit_direct(spec, train, frames, seeds, params):
    from lightgbm import LGBMRegressor
    feats = spec["features"]
    tr = train.dropna(subset=feats + [spec["target"]])
    preds = [np.zeros(len(f)) for f in frames]
    for sd in seeds:
        m = LGBMRegressor(**_params(params, sd)).fit(tr[feats], tr[spec["target"]].to_numpy())
        for p, f in zip(preds, frames):
            p += m.predict(f[feats]) / len(seeds)
    return preds, {"n_train": len(tr)}


def component_values(tr):
    """Count-neutral run values of the non-contact outcomes: training-year
    means of TJ's run value for whiffs, called strikes and other takes."""
    return {"v_whiff": float(tr.loc[tr["is_whiff"] == 1, "target"].mean()),
            "v_cs": float(tr.loc[tr["is_cs"] == 1, "target"].mean()),
            "v_ball": float(tr.loc[tr["is_cs"] == 0, "target"].mean())}


def combine_components(p_swing, p_whiff, p_cs, contact, v):
    return (p_swing * (p_whiff * v["v_whiff"] + (1 - p_whiff) * contact)
            + (1 - p_swing) * (p_cs * v["v_cs"] + (1 - p_cs) * v["v_ball"]))


def component_tasks(tr):
    """(name, training rows, label column, is_classifier) for the four models."""
    return [("swing", tr, "is_swing", True),
            ("whiff", tr[tr["is_swing"] == 1], "is_whiff", True),
            ("cs", tr[tr["is_swing"] == 0], "is_cs", True),
            ("contact", tr[(tr["is_whiff"] == 0) & tr["target_x"].notna()], "target_x", False)]


def _fit_component(spec, train, frames, seeds, params):
    from lightgbm import LGBMClassifier, LGBMRegressor
    feats = spec["features"]
    tr = train.dropna(subset=feats + ["target", "is_swing"])
    v = component_values(tr)
    parts = {name: [np.zeros(len(f)) for f in frames] for name in ("swing", "whiff", "cs", "contact")}
    for name, rows, label, clf in component_tasks(tr):
        for sd in seeds:
            m = (LGBMClassifier if clf else LGBMRegressor)(**_params(params, sd, name))
            m.fit(rows[feats], rows[label].to_numpy())
            for p, f in zip(parts[name], frames):
                p += (m.predict_proba(f[feats])[:, 1] if clf else m.predict(f[feats])) / len(seeds)
    preds = [combine_components(parts["swing"][i], parts["whiff"][i], parts["cs"][i],
                                parts["contact"][i], v) for i in range(len(frames))]
    return preds, {"n_train": len(tr), **v}


def fit_predict(spec, train, frames, params=None):
    """Fit one candidate on `train` and return xRV for each frame in `frames`
    (NaN where the candidate's features are missing), plus fit info."""
    seeds = spec.get("seeds", SEEDS)
    feats = spec["features"]
    ok = [f[feats].notna().all(axis=1).to_numpy() for f in frames]
    fit = _fit_component if spec["kind"] == "component" else _fit_direct
    preds, info = fit(spec, train, [f[m] for f, m in zip(frames, ok)], seeds, params)
    out = []
    for f, m, p in zip(frames, ok, preds):
        full = np.full(len(f), np.nan)
        full[m] = p
        out.append(full)
    return out, info


# ------------------------------------------------- cross-fitted (no-memory) bundle
def fit_seed_models(spec, train, params=None):
    """The fitted LightGBM models (one per seed) for a direct candidate."""
    from lightgbm import LGBMRegressor
    assert spec["kind"] == "direct"
    feats = spec["features"]
    tr = train.dropna(subset=feats + [spec["target"]])
    return [LGBMRegressor(**_params(params, sd)).fit(tr[feats], tr[spec["target"]].to_numpy())
            for sd in spec.get("seeds", SEEDS)]


def predict_seed_models(seed_models, frame, feats):
    """Mean prediction of the seed models; NaN where a feature is missing."""
    ok = frame[feats].notna().all(axis=1).to_numpy()
    out = np.full(len(frame), np.nan)
    if ok.any():
        out[ok] = np.mean([m.predict(frame.loc[ok, feats]) for m in seed_models], axis=0)
    return out


def crossfit_xrv(fold_models, pitcher_fold, frame, feats):
    """xRV with no memory of the pitcher: a pitcher the bundle was trained on is
    scored by the fold model that held him out, anyone else by the mean of all
    fold models."""
    preds = np.vstack([predict_seed_models(m, frame, feats) for m in fold_models])
    k = frame["pitcher"].map(pitcher_fold).to_numpy(dtype=float)
    known = np.isfinite(k)
    x = preds.mean(axis=0)
    x[known] = preds[k[known].astype(int), np.flatnonzero(known)]
    return x
