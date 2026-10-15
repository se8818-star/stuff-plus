"""
Synthetic Statcast-shaped data for smoke-testing the pipeline end to end
(no network needed). Physics columns are internally consistent; outcomes are
generated from a made-up 'true stuff' function so the harness has signal to
find. Results on this data say nothing about real baseball.
"""
import numpy as np
import pandas as pd

from physics import Y50, Y_PLATE

RV = None


def make(n_pitchers=400, pitches_per_season=900, years=(2022, 2023, 2024, 2025, 2026), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    arsenals = {
        "FF": dict(v=94, az=-14, ax=-8, spin=2300),
        "SL": dict(v=85, az=-30, ax=3, spin=2450),
        "CH": dict(v=86, az=-24, ax=-15, spin=1750),
    }
    for p in range(n_pitchers):
        hand = "L" if rng.random() < 0.28 else "R"
        slot_z = rng.normal(5.9, 0.45)
        slot_x = rng.normal(1.9, 0.35)
        ext = rng.normal(6.4, 0.35)
        talent = {k: dict(dv=rng.normal(0, 1.8), daz=rng.normal(0, 3), dax=rng.normal(0, 3))
                  for k in arsenals}
        for y in years:
            drift = rng.normal(0, 0.5)
            for _ in range(pitches_per_season):
                pt = rng.choice(list(arsenals), p=[0.55, 0.3, 0.15])
                a, t = arsenals[pt], talent[pt]
                v_mph = a["v"] + t["dv"] + drift + rng.normal(0, 0.8)
                az = a["az"] + t["daz"] + rng.normal(0, 1.5)
                ax = a["ax"] + t["dax"] + rng.normal(0, 1.5)
                ay = 0.22 * v_mph + rng.normal(0, 1)
                # catcher view: RHP release at x<0
                xr = -slot_x if hand == "R" else slot_x
                ax_raw = ax if hand == "R" else -ax
                zr = slot_z + rng.normal(0, 0.08)
                yr = 60.5 - ext
                # aim at a random location, solve release velocity
                px, pz = rng.normal(0, 0.8), rng.normal(2.4, 0.8)
                vy_r = -v_mph * 1.467
                T = (-vy_r - np.sqrt(vy_r ** 2 - 2 * ay * (yr - Y_PLATE))) / ay
                vx_r = (px - xr - 0.5 * ax_raw * T ** 2) / T
                vz_r = (pz - zr - 0.5 * az * T ** 2) / T
                # state at y=50
                t50 = (-vy_r - np.sqrt(vy_r ** 2 - 2 * ay * (yr - Y50))) / ay
                vx0, vy0, vz0 = vx_r + ax_raw * t50, vy_r + ay * t50, vz_r + az * t50
                vz_p = vz_r + az * T
                vaa = -np.degrees(np.arctan(vz_p / (vy_r + ay * T)))
                # hidden 'true stuff' -> run value (lower = better for pitcher)
                stuff = (0.04 * (v_mph - a["v"]) + 0.03 * (vaa + 5) * (pt == "FF")
                         + 0.02 * abs(ax - (-8)) * (pt != "FF") / 5 - 0.01 * (slot_z - 5.9))
                loc_pen = 0.03 * (abs(px) > 0.83) + 0.03 * ((pz < 1.5) | (pz > 3.5))
                rv = rng.normal(0.01 - 0.6 * stuff + loc_pen, 0.25)
                rows.append((f"{y}-06-01", "R", p, hand, pt, v_mph, a["spin"] + rng.normal(0, 100),
                             rng.uniform(0, 360), ext + rng.normal(0, 0.1), xr, zr, np.nan,
                             vx0, vy0, vz0, ax_raw, ay, az, px, pz, rv,
                             rng.normal(0.31 + 0.4 * rv, 0.05), 1))
    cols = ["game_date", "game_type", "pitcher", "p_throws", "pitch_type", "release_speed",
            "release_spin_rate", "spin_axis", "release_extension", "release_pos_x",
            "release_pos_z", "arm_angle", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x",
            "plate_z", "target", "woba_value", "woba_denom"]
    df = pd.DataFrame(rows, columns=cols)
    df["events"] = np.where(rng.random(len(df)) < 0.25,
                            rng.choice(["strikeout", "walk", "single", "field_out"], len(df),
                                       p=[0.23, 0.08, 0.2, 0.49]), None)
    df["arm_angle"] = rng.normal(40, 12, len(df))
    # columns needed by variants G (batter side), H (spin axis vs movement) and
    # J (xwOBA-valued balls in play)
    df["stand"] = rng.choice(["L", "R"], len(df), p=[0.42, 0.58])
    df["balls"] = rng.integers(0, 4, len(df))
    df["strikes"] = rng.integers(0, 3, len(df))
    in_play = df["events"].isin(["single", "field_out"])
    df["description"] = np.where(in_play, "hit_into_play", "ball")
    df["estimated_woba_using_speedangle"] = np.where(
        in_play, np.clip(rng.normal(0.35, 0.3, len(df)), 0, 2), np.nan)
    df["pfx_x"] = df["ax"] * 0.08
    df["pfx_z"] = (df["az"] + 32.17) * 0.08
    implied = (np.degrees(np.arctan2(df["pfx_z"], df["pfx_x"])) + 90) % 360
    df["spin_axis"] = (implied + rng.normal(0, 15, len(df))) % 360
    return df
