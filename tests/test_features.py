import numpy as np
import pandas as pd
import pytest

import features
from features import DES, EVT, VARIANTS, add_features, add_target
from synthetic import make

RV = pd.read_csv(features.HERE / "run_values.csv")
# one Statcast spelling for each outcome in the run-value table
AS_EVENT = {"single", "double", "triple", "home_run", "walk", "strikeout", "field_out",
            "hit_by_pitch"}
DESC_FOR_EVENT = {"walk": "ball", "strikeout": "swinging_strike", "hit_by_pitch": "hit_by_pitch"}


def pitch_rows(rows):
    return pd.DataFrame(rows, columns=["events", "description", "balls", "strikes"])


def test_target_matches_every_row_of_run_values():
    rows = []
    for r in RV.itertuples():
        if r.event in AS_EVENT:
            rows.append((r.event, DESC_FOR_EVENT.get(r.event, "hit_into_play"), r.balls, r.strikes))
        else:
            rows.append((None, r.event, r.balls, r.strikes))
    out = add_target(pitch_rows(rows))
    assert len(out) == len(RV) == 116
    np.testing.assert_allclose(out["target"], RV["delta_run_exp"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("events,description,balls,strikes,expect", [
    ("force_out", "hit_into_play", 1, 1, ("field_out", 1, 1)),
    ("grounded_into_double_play", "hit_into_play", 0, 2, ("field_out", 0, 2)),
    ("sac_fly", "hit_into_play", 2, 0, ("field_out", 2, 0)),
    (None, "blocked_ball", 1, 2, ("ball", 1, 2)),
    (None, "foul_tip", 0, 1, ("swinging_strike", 0, 1)),
    (None, "swinging_strike_blocked", 1, 0, ("swinging_strike", 1, 0)),
    (None, "foul_bunt", 0, 0, ("foul", 0, 0)),
    # terminal event wins over the description
    ("strikeout", "called_strike", 3, 2, ("strikeout", 3, 2)),
    ("strikeout", "foul_tip", 1, 2, ("strikeout", 1, 2)),
    ("walk", "ball", 3, 1, ("walk", 3, 1)),
    # a mid-PA baserunning event falls through to the pitch description
    ("caught_stealing_2b", "ball", 1, 1, ("ball", 1, 1)),
    ("pickoff_1b", "called_strike", 0, 0, ("called_strike", 0, 0)),
])
def test_target_grouping(events, description, balls, strikes, expect):
    want = RV.set_index(["event", "balls", "strikes"]).loc[expect, "delta_run_exp"]
    got = add_target(pitch_rows([(events, description, balls, strikes)]))["target"].iloc[0]
    assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("events,description,balls,strikes", [
    ("field_error", "hit_into_play", 0, 0),          # unmapped by TJ
    ("strikeout_double_play", "swinging_strike", 1, 2),
    ("catcher_interf", "hit_into_play", 1, 1),
    (None, "intent_ball", 3, 0),
    (None, "ball", 4, 2),                            # impossible count
    (None, "not_a_description", 0, 0),
])
def test_target_is_nan_when_unmapped(events, description, balls, strikes):
    assert np.isnan(add_target(pitch_rows([(events, description, balls, strikes)]))["target"].iloc[0])


def test_target_handles_nullable_dtypes_and_keeps_row_order():
    df = pitch_rows([(None, "ball", 0, 0), ("single", "hit_into_play", 1, 1),
                     (None, "foul", 0, 2), (pd.NA, "called_strike", pd.NA, 1)])
    df = df.astype({"balls": "Int64", "strikes": "Int64"})
    df.index = [10, 7, 3, 99]
    out = add_target(df)
    lut = RV.set_index(["event", "balls", "strikes"])["delta_run_exp"]
    assert list(out.index) == [10, 7, 3, 99]
    np.testing.assert_allclose(out["target"].iloc[:3],
                               [lut["ball", 0, 0], lut["single", 1, 1], 0.0], atol=1e-12)
    assert np.isnan(out["target"].iloc[3])


def test_mapped_outcomes_exist_in_the_table():
    outcomes = set(RV["event"])
    assert {v for v in DES.values() if v} <= outcomes
    assert {v for v in EVT.values() if v} <= outcomes


# ------------------------------------------------------------------ features
@pytest.fixture(scope="module")
def fixture():
    raw = make(n_pitchers=6, pitches_per_season=60, years=(2023, 2024), seed=1)
    # pitcher 0 throws no fastball at all in 2024 -> exercises the fallback
    drop = (raw["pitcher"] == 0) & (raw["game_date"].str.startswith("2024")) \
        & (raw["pitch_type"] == "FF")
    raw = raw[~drop].reset_index(drop=True)
    return raw, add_features(raw)


def test_features_present_and_finite(fixture):
    raw, df = fixture
    assert len(df) == len(raw)
    for name, feats in VARIANTS.items():
        assert set(feats) <= set(df.columns), name
        assert np.isfinite(df[feats].to_numpy(dtype=float)).all(), name
    assert set(df["year"]) == {2023, 2024}


def test_input_frame_is_not_mutated(fixture):
    raw, _ = fixture
    before = raw.copy()
    add_features(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_handedness_mirroring(fixture):
    raw, df = fixture
    lhp = (raw["p_throws"] == "L").to_numpy()
    assert lhp.any() and (~lhp).any()
    # release side is positive for everyone; ax flipped for LHP only
    assert (df["x0"] > 0).all()
    np.testing.assert_allclose(df["ax"], np.where(lhp, -raw["ax"], raw["ax"]))
    np.testing.assert_allclose(df["z0"], raw["release_pos_z"])
    # mirroring a whole pitcher leaves every model feature unchanged
    flipped = raw.copy()
    flipped["p_throws"] = np.where(lhp, "R", "L")
    for c in ("release_pos_x", "vx0", "ax", "plate_x"):
        flipped[c] = -flipped[c]
    all_feats = sorted({f for v in VARIANTS.values() for f in v})
    pd.testing.assert_frame_equal(add_features(flipped)[all_feats], df[all_feats],
                                  check_exact=False, atol=1e-9, rtol=0)


def test_primary_fastball_reference(fixture):
    _, df = fixture
    g = df[(df["pitcher"] == 1) & (df["year"] == 2023)]
    counts = g[g["pitch_type"].isin(["FF", "SI", "FC"])]["pitch_type"].value_counts()
    fb = g[g["pitch_type"] == counts.index[0]]
    assert not g["fb_fallback"].any()
    assert g["fb_speed"].nunique() == 1
    assert g["fb_speed"].iloc[0] == pytest.approx(fb["start_speed"].mean())
    assert g["fb_az"].iloc[0] == pytest.approx(fb["az"].mean())
    assert g["fb_ax"].iloc[0] == pytest.approx(fb["ax"].mean())
    np.testing.assert_allclose(g["speed_diff"], g["start_speed"] - fb["start_speed"].mean())
    assert fb["speed_diff"].mean() == pytest.approx(0, abs=1e-9)
    assert (df["ax_diff"] >= 0).all() and (df["haa_diff"] >= 0).all()


def test_no_fastball_falls_back_to_hardest_pitch_type(fixture):
    _, df = fixture
    g = df[(df["pitcher"] == 0) & (df["year"] == 2024)]
    assert len(g) and g["fb_fallback"].all()
    hardest = g.groupby("pitch_type")["start_speed"].mean().idxmax()
    ref = g[g["pitch_type"] == hardest]
    assert g["fb_speed"].iloc[0] == pytest.approx(ref["start_speed"].mean())
    assert g["fb_az"].iloc[0] == pytest.approx(ref["az"].mean())
    # only that one pitcher-season uses the fallback; his 2023 is untouched
    assert df["fb_fallback"].sum() == len(g)


def test_fastball_reference_is_per_season(fixture):
    _, df = fixture
    per = df.groupby(["pitcher", "year"])["fb_speed"].first().unstack()
    assert (per[2023] != per[2024]).all()


def test_plain_dtypes_removes_pandas_nullable_types():
    df = pd.DataFrame({"a": pd.array([1.5, None], dtype="Float64"),
                       "b": pd.array([1, None], dtype="Int64"),
                       "c": pd.array([1, 2], dtype="Int64"), "d": ["x", None]})
    out = features.plain_dtypes(df)
    assert [str(t) for t in out.dtypes] == ["float64", "float64", "int64", "object"]
    assert np.isnan(out["a"].to_numpy()[1]) and np.isnan(out["b"].to_numpy()[1])
    assert str(df["a"].dtype) == "Float64"           # input untouched


def test_features_accept_statcast_nullable_dtypes(fixture):
    raw, df = fixture
    nullable = raw.convert_dtypes()
    assert "Float64" in {str(t) for t in nullable.dtypes}
    feats = sorted({f for v in VARIANTS.values() for f in v})
    got = add_features(features.plain_dtypes(nullable))
    pd.testing.assert_frame_equal(got[feats], df[feats], check_dtype=False)
