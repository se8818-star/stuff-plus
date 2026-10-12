"""
Turn hand-entered pitch characteristics into Statcast-shaped rows and grade
them with a trained Stuff+ model.

The grader never re-implements feature engineering: it builds fake Statcast
rows (release point, velocity, 9-parameter fit aimed at the middle of the
zone) and runs them through features.add_features, exactly like real pitches.

Movement inputs follow Baseball Savant conventions:
  IVB  induced vertical break, inches (+ = rise vs. a spinless pitch)
  HB   horizontal break, inches, + = pitcher's arm side (mirrored for LHP)
measured from y = 50 ft to the tip of the plate (y = 0), the window that
reproduces Statcast's pfx_x / pfx_z. The leaderboards compute
displayed movement with the same functions, so a leaderboard arsenal loaded
into the grader reproduces its accelerations exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import add_features
from physics import Y50, Y_PLATE, _time_to_y

G = 32.174           # ft/s^2
MPH = 5280 / 3600    # ft/s per mph


def _movement_time(speed, ext, ay):
    """Flight time (s) from y = 50 ft to y = 0 for a pitch released at
    `speed` mph with constant y-acceleration `ay`."""
    y_r = 60.5 - np.asarray(ext, float)
    vy_r = -np.asarray(speed, float) * MPH
    return _time_to_y(y_r, vy_r, ay, 0.0) - _time_to_y(y_r, vy_r, ay, Y50)


def accel_to_movement(az, ax_arm, speed, ext, ay):
    """(az, mirrored ax) in ft/s^2 -> (IVB, arm-side HB) in inches.
    `ax_arm` is features.py's mirrored ax, where negative = arm side."""
    t2 = _movement_time(speed, ext, ay) ** 2
    ivb = 0.5 * (np.asarray(az) + G) * t2 * 12
    hb = -0.5 * np.asarray(ax_arm) * t2 * 12
    return ivb, hb


def movement_to_accel(ivb, hb, speed, ext, ay):
    """Inverse of accel_to_movement: returns (az, mirrored ax)."""
    t2 = _movement_time(speed, ext, ay) ** 2
    az = 2 * (np.asarray(ivb) / 12) / t2 - G
    ax_arm = -2 * (np.asarray(hb) / 12) / t2
    return az, ax_arm


def default_spin_axis(ivb, hb, hand):
    """Savant-style spin axis (deg, 180 = pure backspin) implied by movement
    direction. Only used when the user leaves spin axis blank."""
    pfx_x = -hb if hand == "R" else hb   # catcher's view
    return float((180 - np.degrees(np.arctan2(pfx_x, ivb))) % 360)


def synth_rows(pitches, hand, ay_coef, season=2026, arm_coef=None):
    """Build one Statcast-shaped row per unit of usage for each pitch."""
    rows = []
    flip = -1.0 if hand == "R" else 1.0  # arm side -> catcher-view x
    for i, p in enumerate(pitches):
        v, ext = float(p["velo"]), float(p["ext"])
        ay = float(np.polyval(ay_coef, v))
        az, ax_arm = movement_to_accel(p["ivb"], p["hb"], v, ext, ay)
        ax_raw = float(ax_arm) if hand == "R" else -float(ax_arm)
        xr, zr, yr = flip * float(p["rel_side"]), float(p["rel_height"]), 60.5 - ext
        vy_r = -v * MPH
        # aim at the middle of the zone; location-neutral features ignore it
        T = float(_time_to_y(yr, vy_r, ay, Y_PLATE))
        vx_r = (0.0 - xr - 0.5 * ax_raw * T ** 2) / T
        vz_r = (2.5 - zr - 0.5 * float(az) * T ** 2) / T
        t50 = float(_time_to_y(yr, vy_r, ay, Y50))
        spin_axis = p.get("spin_axis")
        if spin_axis in (None, ""):
            spin_axis = default_spin_axis(float(p["ivb"]), float(p["hb"]), hand)
        arm_angle = p.get("arm_angle")
        if arm_angle in (None, ""):
            arm_angle = (float(arm_coef[0] + arm_coef[1] * zr + arm_coef[2] * abs(xr))
                         if arm_coef is not None else 40.0)
        row = dict(game_date=f"{season}-06-01", game_type="R", pitcher=-1,
                   p_throws=hand, pitch_type=p["pitch_type"], _idx=i,
                   release_speed=v, release_spin_rate=float(p["spin"]),
                   spin_axis=float(spin_axis), release_extension=ext,
                   release_pos_x=xr, release_pos_z=zr, arm_angle=float(arm_angle),
                   vx0=vx_r + ax_raw * t50, vy0=vy_r + ay * t50,
                   vz0=vz_r + float(az) * t50, ax=ax_raw, ay=ay, az=float(az))
        # usage decides the primary fastball, so repeat rows in proportion
        rows += [row] * max(1, int(round(float(p.get("usage") or 1))))
    return pd.DataFrame(rows)


class AveragedModel:
    """Average prediction of several fitted regressors (the frozen model is a
    list of LightGBM fits with different seeds)."""

    def __init__(self, models, features):
        self.models, self.features = list(models), list(features)

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.features) if not isinstance(X, pd.DataFrame) else X[self.features]
        return np.mean([m.predict(X) for m in self.models], axis=0)


def percentile(sorted_vals, x):
    if sorted_vals is None or len(sorted_vals) == 0:
        return None
    return float(100 * np.searchsorted(sorted_vals, x, side="right") / len(sorted_vals))


def to_grade(x, mean, sd):
    """20-80 scouting grade (50 + 10 z, clipped), like final.grade."""
    if mean is None or not sd:
        return None
    return int(round(float(np.clip(50 + 10 * (x - mean) / sd, 20, 80))))


def rescale(x, c, spread):
    """Display scale: stretch so the qualified group (constants `c`) has an SD
    of 10. "keep" keeps the group's own mean, "center" moves it to 100."""
    if not spread or not c.get("sd"):
        return x
    return (c["mean"] if spread == "keep" else 100.0) + 10 * (x - c["mean"]) / c["sd"]


# Typical pitch-to-pitch spread around a pitcher's own average for one pitch
# type (median within-pitcher SD, 2025 Statcast, pitch types with 100+ thrown).
SCATTER = {"velo": 1.15, "ivb": 2.2, "hb": 2.2}   # mph, in, in
N_SIM = 300


def simulate_pitches(pitches, n=N_SIM, seed=0):
    """Expand an arsenal into about `n` individual pitches, split by usage,
    each pitch type scattered around the entered values. The scatter is
    centred, so every pitch type's average stays exactly what was entered,
    and seeded, so the same inputs always give the same grade. Returns the
    pitches (usage 1 each) and the index of the entered pitch each came from."""
    rng = np.random.default_rng(seed)
    w = np.array([max(float(p.get("usage") or 0), 0) for p in pitches])
    w = w / w.sum() if w.sum() > 0 else np.full(len(pitches), 1 / len(pitches))
    # enough pitches that the rarest pitch type still gets a usable sample
    # without changing which fastball is the most used
    if (w > 0).any():
        n = int(min(max(n, 20 / w[w > 0].min()), 3000))
    sim, idx = [], []
    for i, p in enumerate(pitches):
        k = max(20, int(round(n * w[i])))
        z = rng.standard_normal((k, len(SCATTER)))
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        for row in z:
            q = dict(p, usage=1)
            for key, zz in zip(SCATTER, row):
                q[key] = float(p[key]) + SCATTER[key] * zz
            sim.append(q)
            idx.append(i)
    return sim, np.array(idx)


def pick_model(bundle, pitcher=None):
    """The model to grade with, and its fold. An arsenal taken from an MLB
    pitcher is scored by the fold model that never saw him, as on the
    leaderboard; anything else by the bundle's model (with fold models, the
    average of all of them)."""
    k = bundle.get("pitcher_fold", {}).get(pitcher) if pitcher is not None else None
    return (bundle["model"], None) if k is None else (bundle["folds"][k], k)


def grade_arsenal(bundle, hand, pitches, season=None, pitcher=None):
    season = season or bundle["latest_season"]
    model, fold = pick_model(bundle, pitcher)
    scale = bundle["scale"][season]
    df = synth_rows(pitches, hand, bundle["ay_coef"], season, bundle.get("arm_coef"))
    feat = add_features(df.drop(columns="_idx")).assign(_idx=df["_idx"].to_numpy())
    first = feat.drop_duplicates("_idx").sort_values("_idx")
    # Score a simulated sample of each pitch, not the one average pitch: the
    # model was trained on real pitches, which always vary around a pitcher's
    # average, and it has no sensible answer for a fastball that is identical
    # every time (ax_diff exactly 0 only happens for one-pitch junk samples).
    sim, idx = simulate_pitches(pitches)
    sdf = synth_rows(sim, hand, bundle["ay_coef"], season, bundle.get("arm_coef"))
    sfeat = add_features(sdf.drop(columns="_idx"))
    pred = model.predict(sfeat[bundle["features"]].astype(float))
    pred = pd.Series(pred).groupby(idx).mean().reindex(range(len(pitches))).to_numpy()
    stuff = 100 - 10 * (pred - scale["mu"]) / scale["sd"]

    consts = bundle["grades"][season]
    out = []
    for (_, r), s in zip(first.iterrows(), stuff):
        pt = r["pitch_type"]
        c = consts["pitch_type"].get(pt) or consts["all_pitches"]
        out.append({
            "pitch_type": pt,
            "stuff_plus": float(rescale(s, c, bundle.get("spread"))),
            "stuff_plus_model": float(s),
            "grade": to_grade(s, c["mean"], c["sd"]),
            "percentile": percentile(bundle["percentiles"][season]["pitch_type"].get(pt), s),
            "is_primary_fb": bool(np.isclose(r["fb_speed"], r["start_speed"])
                                  and np.isclose(r["fb_az"], r["az"])),
            "features": {f: float(r[f]) for f in bundle["features"]},
        })
    w = np.array([max(float(p.get("usage") or 0), 0) for p in pitches])
    w = w / w.sum() if w.sum() > 0 else np.full(len(pitches), 1 / len(pitches))
    overall = float(np.dot(w, stuff))
    pc = consts["pitcher"]
    return {
        "season_scale": season,
        "model": {"out_of_fold": bool(bundle.get("folds")), "fold": fold,
                  "pitcher": pitcher if fold is not None else None},
        "pitches": out,
        "overall": {
            "stuff_plus": float(rescale(overall, pc, bundle.get("spread"))),
            "stuff_plus_model": overall,
            "grade": to_grade(overall, pc["mean"], pc["sd"]),
            "percentile": percentile(bundle["percentiles"][season]["pitcher"], overall),
        },
    }
