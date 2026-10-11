"""Website: model bundle, grader physics, and API, on a small synthetic bundle."""
import joblib
import numpy as np
import pytest

import webapp  # noqa: F401  (repo root on sys.path)
from webapp.build import build
from webapp.grading import accel_to_movement, grade_arsenal, movement_to_accel


@pytest.fixture(scope="module")
def bundle_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "bundle.joblib"
    return build(synthetic=True, out=out, synth_size=(40, 120))


@pytest.fixture(scope="module")
def bundle(bundle_file):
    return joblib.load(bundle_file)


@pytest.fixture(scope="module")
def client(bundle_file):
    from fastapi.testclient import TestClient
    from webapp.app import create_app
    return TestClient(create_app(bundle_file))


def test_movement_round_trip():
    az, ax = np.array([-15.0, -35.0]), np.array([-12.0, 6.0])
    speed, ext, ay = np.array([95.0, 82.0]), np.array([6.5, 6.0]), np.array([28.0, 24.0])
    ivb, hb = accel_to_movement(az, ax, speed, ext, ay)
    az2, ax2 = movement_to_accel(ivb, hb, speed, ext, ay)
    np.testing.assert_allclose(az2, az, atol=1e-9)
    np.testing.assert_allclose(ax2, ax, atol=1e-9)
    assert hb[0] > 0 > hb[1]  # negative mirrored ax = arm side


@pytest.mark.parametrize("hand", ["R", "L"])
def test_grader_reproduces_leaderboard_features(bundle, hand):
    """An arsenal loaded from the leaderboard turns back into the same model inputs."""
    pt = bundle["pitch_types"]
    pid = pt[(pt["hand"] == hand) & (pt["year"] == bundle["latest_season"])]["pitcher"].iat[0]
    rows = pt[(pt["pitcher"] == pid) & (pt["year"] == bundle["latest_season"])]
    pitches = rows[["pitch_type", "usage", "velo", "spin", "ext", "rel_side", "rel_height",
                    "ivb", "hb", "spin_axis", "arm_angle"]].to_dict("records")
    res = grade_arsenal(bundle, hand, pitches)
    for p, r in zip(res["pitches"], pitches):
        f = p["features"]
        assert f["start_speed"] == pytest.approx(r["velo"])
        assert f["x0"] == pytest.approx(r["rel_side"])
        assert f["z0"] == pytest.approx(r["rel_height"])
        ay = np.polyval(bundle["ay_coef"], r["velo"])
        ivb, hb = accel_to_movement(f["az"], f["ax"], r["velo"], r["ext"], ay)
        assert ivb == pytest.approx(r["ivb"], abs=1e-6)
        assert hb == pytest.approx(r["hb"], abs=1e-6)
    assert sum(p["is_primary_fb"] for p in res["pitches"]) == 1
    # grade of the average pitch should sit near the mean of the pitch grades
    board = bundle["pitchers"]
    s = board[(board["pitcher"] == pid) & (board["year"] == bundle["latest_season"])]["stuff_plus"].iat[0]
    assert abs(res["overall"]["stuff_plus"] - s) < 10


def test_api(client, bundle):
    meta = client.get("/api/meta").json()
    assert meta["synthetic"] and meta["seasons"] == [2024, 2025, 2026]
    lb = client.get("/api/leaderboard/2026").json()
    assert len(lb["pitchers"]) == 40 and lb["pitch_types"]
    pid = lb["pitchers"][0]["pitcher"]
    assert len(client.get(f"/api/pitcher/{pid}").json()["seasons"]) == 3
    assert client.get("/api/leaderboard/1999").status_code == 404
    assert client.get("/").status_code == 200

    def grade(velo):
        body = {"hand": "R", "pitches": [
            {"pitch_type": "FF", "usage": 60, "velo": velo, "spin": 2300, "ivb": 10, "hb": 6},
            {"pitch_type": "SL", "usage": 40, "velo": 85, "spin": 2500, "ivb": 1, "hb": -4}]}
        r = client.post("/api/grade", json=body)
        assert r.status_code == 200, r.text
        return r.json()
    slow, fast = grade(90), grade(99)
    assert fast["pitches"][0]["stuff_plus"] > slow["pitches"][0]["stuff_plus"]
    assert 20 <= fast["overall"]["grade"] <= 80

    bad = client.post("/api/grade", json={"hand": "R", "pitches": [{"pitch_type": "FF", "velo": 300}]})
    assert bad.status_code == 422
