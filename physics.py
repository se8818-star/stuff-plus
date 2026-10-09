"""
Location-neutral approach angles from the Statcast 9-parameter trajectory fit.

Statcast fits each pitch as constant-acceleration motion anchored at y = 50 ft:
    r(t) = r50 + v50 * t + 0.5 * a * t^2
with columns vx0, vy0, vz0 (velocity at y=50), ax, ay, az (constant accel),
and release_pos_x / release_pos_z / release_extension for the release point.

To grade a pitch's *shape* rather than where it was located, we:
  1. Rewind the fitted trajectory to the release plane (y = 60.5 - extension)
     to get the release position and the y-component of release velocity.
  2. Keep release position, vy at release, and all three accelerations fixed
     (i.e. same velocity, same movement, same release).
  3. Re-aim the pitch: solve for the vx, vz at release that make it cross
     the front of the plate at each target (x_t, z_t).
  4. Compute VAA/HAA where it crosses the plate.

Because accelerations and release are held fixed, the only thing that differs
between the original pitch and the re-aimed copies is the target -- so the
resulting angles contain no information about the pitch's actual location.

Sign conventions (same as Baseball Savant):
  VAA = -atan(vz / vy)  -> negative; closer to 0 = flatter
  HAA = -atan(vx / vy)  -> catcher's view; we mirror LHP so + = arm side
"""
from __future__ import annotations

import numpy as np

Y50 = 50.0               # Statcast fit anchor (ft from tip of plate)
Y_PLATE = 17.0 / 12.0    # front edge of home plate (ft)

# 3x3 grid = centers of Statcast zones 1-9 on a fixed 1.5-3.5 ft, 17" zone.
# Fixed (not batter-specific sz_top/sz_bot) so the grid itself is batter-neutral.
GRID_X = {"glove": -17 / 36, "mid": 0.0, "arm": 17 / 36}   # ft, LHP-mirrored frame
GRID_Z = {"low": 1.5 + 2 / 6, "mid": 2.5, "high": 3.5 - 2 / 6}

GRID = [(f"{zn}_{xn}", xv, zv) for zn, zv in GRID_Z.items() for xn, xv in GRID_X.items()]


def _time_to_y(y_start, vy_start, ay, y_end):
    """Time (s) to travel from y_start to y_end under constant ay (vy < 0)."""
    disc = vy_start ** 2 - 2.0 * ay * (y_start - y_end)
    disc = np.where(disc < 0, np.nan, disc)
    # ay can be ~0 in degenerate rows; fall back to constant-velocity solution
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-vy_start - np.sqrt(disc)) / ay
        t_lin = (y_end - y_start) / vy_start
    return np.where(np.abs(ay) < 1e-6, t_lin, t)


def release_state(vx0, vy0, vz0, ax, ay, az, extension, x50=None, z50=None,
                  release_x=None, release_z=None):
    """Rewind the fit to the release plane.

    Returns (x_r, z_r, y_r, vy_r). If release_x/z are given they are used for
    position (Statcast's release_pos_*); otherwise position comes from the fit
    rewound from (x50, z50).
    """
    y_r = 60.5 - extension
    # time from y=50 back to release (negative)
    t_r = _time_to_y(Y50, vy0, ay, y_r)
    vy_r = vy0 + ay * t_r
    if release_x is None:
        x_r = x50 + vx0 * t_r + 0.5 * ax * t_r ** 2
        z_r = z50 + vz0 * t_r + 0.5 * az * t_r ** 2
    else:
        x_r, z_r = release_x, release_z
    return x_r, z_r, y_r, vy_r


def angles_at_target(x_r, z_r, y_r, vy_r, ax, ay, az, x_t, z_t):
    """VAA/HAA (deg) for a pitch re-aimed to cross the plate at (x_t, z_t)."""
    T = _time_to_y(y_r, vy_r, ay, Y_PLATE)        # flight time release -> plate
    vx_r = (x_t - x_r - 0.5 * ax * T ** 2) / T
    vz_r = (z_t - z_r - 0.5 * az * T ** 2) / T
    vx_p = vx_r + ax * T
    vz_p = vz_r + az * T
    vy_p = vy_r + ay * T
    vaa = -np.degrees(np.arctan(vz_p / vy_p))
    haa = -np.degrees(np.arctan(vx_p / vy_p))
    return vaa, haa


def actual_angles(vx0, vy0, vz0, ax, ay, az):
    """VAA/HAA at the pitch's real plate crossing (location-dependent).
    Used only as a sanity check / for comparison -- never as a Stuff feature."""
    t = _time_to_y(Y50, vy0, ay, Y_PLATE)
    vx, vy, vz = vx0 + ax * t, vy0 + ay * t, vz0 + az * t
    return -np.degrees(np.arctan(vz / vy)), -np.degrees(np.arctan(vx / vy))


def grid_angles(df, mirror_col="p_throws"):
    """Add 9 VAA + 9 HAA columns (vaa_<row>_<col>, haa_<row>_<col>) plus
    vaa_ln / haa_ln (center-of-zone, location-neutral) to a pandas DataFrame
    with Statcast columns. Handedness is mirrored so x>0 = pitcher's arm side.
    """
    lhp = (df[mirror_col] == "L").to_numpy()
    # Statcast x is catcher's view: + toward 1B side. A RHP's arm side is the
    # 3B side (x<0). Mirror so arm side is +x for everyone: RHP flip, LHP keep.
    flip = np.where(lhp, 1.0, -1.0)

    vx0 = df["vx0"].to_numpy() * flip
    ax = df["ax"].to_numpy() * flip
    rel_x = df["release_pos_x"].to_numpy() * flip
    vy0, vz0 = df["vy0"].to_numpy(), df["vz0"].to_numpy()
    ay, az = df["ay"].to_numpy(), df["az"].to_numpy()
    ext = df["release_extension"].to_numpy()
    rel_z = df["release_pos_z"].to_numpy()

    x_r, z_r, y_r, vy_r = release_state(vx0, vy0, vz0, ax, ay, az, ext,
                                        release_x=rel_x, release_z=rel_z)
    out = {}
    for name, xt, zt in GRID:
        v, h = angles_at_target(x_r, z_r, y_r, vy_r, ax, ay, az, xt, zt)
        out[f"vaa_{name}"] = v
        out[f"haa_{name}"] = h
    out["vaa_ln"] = out["vaa_mid_mid"]
    out["haa_ln"] = out["haa_mid_mid"]
    for k, v in out.items():
        df[k] = v
    return df


VAA_GRID_COLS = [f"vaa_{n}" for n, _, _ in GRID]
HAA_GRID_COLS = [f"haa_{n}" for n, _, _ in GRID]
