"""
Target + feature engineering, replicating tjStuff+ v3.0 in Statcast column
names, plus the location-neutral approach-angle features.

tjStuff+ v3.0 reference: https://github.com/tnestico/tjstuff_plus (MIT).
run_values.csv is copied from that repo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from physics import grid_angles, VAA_GRID_COLS, HAA_GRID_COLS

HERE = Path(__file__).parent
FASTBALLS = ["FF", "SI", "FC"]

# Statcast `description` -> run-value table outcome (same grouping as TJ's des_dict)
DES = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball", "intent_ball": None,
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike", "swinging_strike_blocked": "swinging_strike",
    "foul_tip": "swinging_strike", "missed_bunt": "swinging_strike",
    "bunt_foul_tip": "swinging_strike",
    "foul": "foul", "foul_bunt": "foul", "foul_pitchout": "foul",
    "hit_by_pitch": "hit_by_pitch",
    "hit_into_play": None,  # resolved by `events`
}
# Statcast `events` -> run-value table outcome (same grouping as TJ's event_dict)
EVT = {
    "single": "single", "double": "double", "triple": "triple", "home_run": "home_run",
    "walk": "walk", "strikeout": "strikeout", "hit_by_pitch": "hit_by_pitch",
    "field_out": "field_out", "force_out": "field_out",
    "grounded_into_double_play": "field_out", "double_play": "field_out",
    "fielders_choice": "field_out", "fielders_choice_out": "field_out",
    "sac_fly": "field_out", "sac_bunt": "field_out", "sac_fly_double_play": "field_out",
    "triple_play": "field_out", "other_out": "field_out",
}

# ---------------------------------------------------------------- feature sets
TJ_V3 = ["start_speed", "spin_rate", "extension", "az", "ax", "x0", "z0",
         "speed_diff", "az_diff", "ax_diff"]
GRID = VAA_GRID_COLS + HAA_GRID_COLS
CENTER = ["vaa_ln", "haa_ln"]

VARIANTS = {
    # A: faithful tjStuff+ v3.0 baseline (features as in his notebook)
    "A_tj_v3": TJ_V3,
    # B: baseline + one location-neutral VAA/HAA (zone center)
    "B_plus_center": TJ_V3 + CENTER,
    # B9: baseline + all 9-point VAA/HAA
    "B9_plus_grid9": TJ_V3 + GRID,
    # C: 9-point angles replace movement (az, ax), release kept
    "C_grid9_no_accel": [f for f in TJ_V3 if f not in ("az", "ax")] + GRID,
    # D: your proposal -- angles replace movement AND release (and arm angle)
    "D_grid9_angles_only": ["start_speed", "spin_rate", "extension", "speed_diff",
                            "vaa_diff", "haa_diff"] + GRID,
    # D1: same as D but a single center point instead of 9 (tests 9 vs 1)
    "D1_center_angles_only": ["start_speed", "spin_rate", "extension", "speed_diff",
                              "vaa_diff", "haa_diff"] + CENTER,
    # E: baseline + arm angle (not in v3.0)
    "E_plus_arm_angle": TJ_V3 + ["arm_angle"],
    # F: baseline + spin axis (listed in the v3.0 article, not in the notebook)
    "F_plus_spin_axis": TJ_V3 + ["spin_axis"],
}


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Count-specific run value per pitch, TJ-style: terminal events use the
    event's value in that count, otherwise the pitch description's value.
    Outcomes TJ leaves unmapped (field_error, strikeout_double_play, ...) get
    NaN and are dropped at train/score time, as in his notebook."""
    rv = pd.read_csv(HERE / "run_values.csv").astype({"balls": float, "strikes": float})
    count = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
             for c in ("balls", "strikes")}

    def lookup(outcome):
        key = pd.DataFrame({"event": outcome.to_numpy(dtype=object), **count})
        return key.merge(rv, how="left", on=["event", "balls", "strikes"])["delta_run_exp"].to_numpy()

    v_evt = lookup(df["events"].map(EVT))
    v_des = lookup(df["description"].map(DES))
    df["target"] = np.where(np.isnan(v_evt), v_des, v_evt)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["game_date"]).dt.year
    lhp = df["p_throws"] == "L"

    df["start_speed"] = df["release_speed"]
    df["spin_rate"] = df["release_spin_rate"]
    df["extension"] = df["release_extension"]
    # Mirror like TJ: flip LHP horizontal accel; flip RHP horizontal release
    df["ax"] = np.where(lhp, -df["ax"], df["ax"])
    df["x0"] = np.where(lhp, df["release_pos_x"], -df["release_pos_x"])
    df["z0"] = df["release_pos_z"]
    # (raw ax is restored for the physics below via a mirrored copy)

    # Location-neutral 3x3 approach angles (physics.py does its own mirroring
    # from the *raw* Statcast columns, so pass the un-mirrored ax)
    raw = df[["p_throws", "vx0", "vy0", "vz0", "ay", "az", "release_extension",
              "release_pos_x", "release_pos_z"]].copy()
    raw["ax"] = np.where(lhp, -df["ax"], df["ax"])  # undo the flip above
    raw = grid_angles(raw)
    for c in GRID + CENTER:
        df[c] = raw[c].to_numpy()

    # Primary fastball per pitcher-season (most-used of FF/SI/FC). Pitcher-seasons
    # with no fastball fall back to their hardest pitch type by mean velocity
    # (TJ's notebook inner-joins here, which silently drops them; fb_fallback
    # flags those rows so they can be excluded for a like-for-like comparison).
    by_type = (df.groupby(["pitcher", "year", "pitch_type"])
                 .agg(fb_speed=("start_speed", "mean"), fb_az=("az", "mean"),
                      fb_ax=("ax", "mean"), fb_vaa=("vaa_ln", "mean"),
                      fb_haa=("haa_ln", "mean"), n=("start_speed", "size"))
                 .reset_index())
    is_fb = by_type["pitch_type"].isin(FASTBALLS)
    primary = by_type[is_fb].sort_values(["n", "fb_speed"], ascending=False, kind="stable")
    hardest = by_type[~is_fb].sort_values("fb_speed", ascending=False, kind="stable")
    ref = (pd.concat([primary.assign(fb_fallback=False), hardest.assign(fb_fallback=True)])
             .drop_duplicates(["pitcher", "year"])
             .drop(columns=["pitch_type", "n"]))
    df = df.merge(ref, on=["pitcher", "year"], how="left")

    df["speed_diff"] = df["start_speed"] - df["fb_speed"]
    df["az_diff"] = df["az"] - df["fb_az"]
    df["ax_diff"] = (df["ax"] - df["fb_ax"]).abs()
    df["vaa_diff"] = df["vaa_ln"] - df["fb_vaa"]
    df["haa_diff"] = (df["haa_ln"] - df["fb_haa"]).abs()
    return df


def load(data_dir: Path, years) -> pd.DataFrame:
    frames = [pd.read_parquet(data_dir / f"statcast_{y}.parquet") for y in years]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pitch_type"].notna()
            & ~df["pitch_type"].isin(["PO", "IN", "EP", "KN", "FA", "CS", "SC"])].copy()
    df = add_target(df)
    return add_features(df)
