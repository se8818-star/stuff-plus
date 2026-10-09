import numpy as np
import pandas as pd
import pytest

import physics
from physics import (GRID, HAA_GRID_COLS, VAA_GRID_COLS, Y50, Y_PLATE, actual_angles,
                     angles_at_target, grid_angles, release_state)
from synthetic import make


def random_pitches(n=500, seed=0):
    """Self-consistent constant-acceleration trajectories in Statcast columns."""
    rng = np.random.default_rng(seed)
    ext = rng.normal(6.4, 0.4, n)
    y_r = 60.5 - ext
    x_r, z_r = rng.normal(-1.8, 0.8, n), rng.normal(5.8, 0.5, n)
    vy_r = -rng.uniform(72, 102, n) * 1.467
    ax, ay, az = rng.normal(-5, 10, n), rng.normal(27, 4, n), rng.normal(-22, 9, n)
    vx_r, vz_r = rng.normal(5, 4, n), rng.normal(-4, 3, n)

    def state(y):
        t = (-vy_r - np.sqrt(vy_r ** 2 - 2 * ay * (y_r - y))) / ay
        return (x_r + vx_r * t + 0.5 * ax * t ** 2, z_r + vz_r * t + 0.5 * az * t ** 2,
                vx_r + ax * t, vy_r + ay * t, vz_r + az * t)

    _, _, vx0, vy0, vz0 = state(Y50)
    plate_x, plate_z, *_ = state(Y_PLATE)
    return pd.DataFrame({
        "p_throws": "R", "release_extension": ext, "release_pos_x": x_r, "release_pos_z": z_r,
        "vx0": vx0, "vy0": vy0, "vz0": vz0, "ax": ax, "ay": ay, "az": az,
        "plate_x": plate_x, "plate_z": plate_z, "vy_r": vy_r})


def reaimed_at_own_location(df):
    c = {k: df[k].to_numpy() for k in df.columns if k != "p_throws"}
    x_r, z_r, y_r, vy_r = release_state(
        c["vx0"], c["vy0"], c["vz0"], c["ax"], c["ay"], c["az"], c["release_extension"],
        release_x=c["release_pos_x"], release_z=c["release_pos_z"])
    got = angles_at_target(x_r, z_r, y_r, vy_r, c["ax"], c["ay"], c["az"],
                           c["plate_x"], c["plate_z"])
    want = actual_angles(c["vx0"], c["vy0"], c["vz0"], c["ax"], c["ay"], c["az"])
    return got, want


def test_reaim_at_own_location_reproduces_actual_angles():
    df = random_pitches()
    (vaa, haa), (vaa0, haa0) = reaimed_at_own_location(df)
    np.testing.assert_allclose(vaa, vaa0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(haa, haa0, atol=1e-6, rtol=0)
    assert np.median(vaa) < 0 and np.ptp(vaa) > 1 and np.ptp(haa) > 1   # not trivially constant


def test_reaim_on_synthetic_statcast_rows():
    df = make(n_pitchers=8, pitches_per_season=50, years=(2024,))
    (vaa, haa), (vaa0, haa0) = reaimed_at_own_location(
        df[["p_throws", "release_extension", "release_pos_x", "release_pos_z", "vx0", "vy0",
            "vz0", "ax", "ay", "az", "plate_x", "plate_z"]])
    # synthetic.py jitters per-pitch extension independently of the trajectory's
    # release plane, so this only holds to the size of that jitter
    np.testing.assert_allclose(vaa, vaa0, atol=0.05)
    np.testing.assert_allclose(haa, haa0, atol=0.05)


def test_rewind_recovers_release_velocity():
    df = random_pitches()
    _, _, y_r, vy_r = release_state(*(df[k].to_numpy() for k in
                                      ["vx0", "vy0", "vz0", "ax", "ay", "az", "release_extension"]),
                                    release_x=0, release_z=0)
    np.testing.assert_allclose(vy_r, df["vy_r"], atol=1e-8)
    np.testing.assert_allclose(y_r, 60.5 - df["release_extension"])


def test_left_and_right_handers_mirror():
    rhp = random_pitches()
    lhp = rhp.copy()
    lhp["p_throws"] = "L"
    for c in ("release_pos_x", "vx0", "ax", "plate_x"):
        lhp[c] = -lhp[c]
    r, l = grid_angles(rhp.copy()), grid_angles(lhp.copy())
    cols = VAA_GRID_COLS + HAA_GRID_COLS + ["vaa_ln", "haa_ln"]
    np.testing.assert_allclose(r[cols].to_numpy(), l[cols].to_numpy(), atol=1e-9, rtol=0)


def test_arm_side_is_positive_for_both_hands():
    # A pitch released from the arm side and thrown to the middle travels toward
    # the glove side, so its mirrored HAA must have the same sign for both hands.
    rhp = random_pitches(50)
    rhp["release_pos_x"] = -2.0
    rhp["ax"] = 0.0
    lhp = rhp.assign(p_throws="L", release_pos_x=2.0, vx0=-rhp["vx0"])
    hr, hl = grid_angles(rhp.copy())["haa_ln"], grid_angles(lhp.copy())["haa_ln"]
    assert (np.sign(hr) == np.sign(hl)).all() and (hr != 0).all()


def test_grid_is_fixed_3x3_inside_the_zone():
    assert len(GRID) == 9 and len(set(VAA_GRID_COLS + HAA_GRID_COLS)) == 18
    for _, x, z in GRID:
        assert abs(x) < 17 / 24 and 1.5 < z < 3.5


def test_grid_angles_ignore_actual_location():
    # Same release/velocity/movement aimed somewhere else -> identical features.
    df = random_pitches(100)
    c = {k: df[k].to_numpy() for k in df.columns if k != "p_throws"}
    y_r = 60.5 - c["release_extension"]
    T = physics._time_to_y(y_r, c["vy_r"], c["ay"], Y_PLATE)
    t50 = physics._time_to_y(y_r, c["vy_r"], c["ay"], Y50)
    moved = df.copy()
    for axis, pos, acc, v0, shift in [("x", "release_pos_x", "ax", "vx0", 1.3),
                                      ("z", "release_pos_z", "az", "vz0", -0.9)]:
        tgt = df[f"plate_{axis}"].to_numpy() + shift
        v_r = (tgt - c[pos] - 0.5 * c[acc] * T ** 2) / T
        moved[v0] = v_r + c[acc] * t50
    cols = VAA_GRID_COLS + HAA_GRID_COLS
    np.testing.assert_allclose(grid_angles(df.copy())[cols].to_numpy(),
                               grid_angles(moved)[cols].to_numpy(), atol=1e-9, rtol=0)


@pytest.mark.parametrize("ay", [0.0, 1e-9])
def test_zero_drag_does_not_blow_up(ay):
    t = physics._time_to_y(55.0, -130.0, ay, Y_PLATE)
    assert np.isfinite(t) and t == pytest.approx((55.0 - Y_PLATE) / 130.0)
