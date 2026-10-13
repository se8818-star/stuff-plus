"""Website: model bundle, grader physics, and API, on a small synthetic bundle."""
import joblib
import numpy as np
import pandas as pd
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


def test_simulated_pitches_keep_entered_averages():
    """The grader scores a scattered sample of each pitch; the sample's
    averages are exactly what was entered, it is never one identical pitch
    repeated, and it is deterministic."""
    from webapp.grading import SCATTER, simulate_pitches
    arsenal = [{"pitch_type": "FF", "usage": 60, "velo": 95.0, "ivb": 16.0, "hb": 8.0},
               {"pitch_type": "SL", "usage": 38, "velo": 86.0, "ivb": 2.0, "hb": -4.0},
               {"pitch_type": "CH", "usage": 2, "velo": 87.0, "ivb": 6.0, "hb": 14.0}]
    sim, idx = simulate_pitches(arsenal)
    counts = np.bincount(idx)
    assert counts.min() >= 20 and counts[0] > counts[1] > counts[2]
    for i, p in enumerate(arsenal):
        for k, sd in SCATTER.items():
            v = np.array([q[k] for q, j in zip(sim, idx) if j == i])
            assert v.mean() == pytest.approx(p[k]) and v.std() == pytest.approx(sd)
    assert simulate_pitches(arsenal)[0] == sim


@pytest.mark.parametrize("hand", ["R", "L"])
def test_grader_reproduces_leaderboard_features(bundle, hand):
    """An arsenal loaded from the leaderboard turns back into the same model inputs."""
    pt = bundle["pitch_types"]
    pid = pt[(pt["hand"] == hand) & (pt["year"] == bundle["latest_season"])]["pitcher"].iat[0]
    rows = pt[(pt["pitcher"] == pid) & (pt["year"] == bundle["latest_season"])]
    pitches = rows[["pitch_type", "usage", "velo", "spin", "ext", "rel_side", "rel_height",
                    "ivb", "hb", "spin_axis", "arm_angle"]].to_dict("records")
    res = grade_arsenal(bundle, hand, pitches, pitcher=pid)
    # scored like the leaderboard: by the fold model that never saw him
    assert res["model"] == {"out_of_fold": True, "fold": bundle["pitcher_fold"][pid], "pitcher": pid}
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
    s = board[(board["pitcher"] == pid) & (board["year"] == bundle["latest_season"])]["stuff_plus_model"].iat[0]
    assert abs(res["overall"]["stuff_plus_model"] - s) < 10


def test_out_of_fold_models_never_score_their_own_pitchers():
    """Like final.py's leaderboard: every pitcher is held out of one fold, and
    his pitches are scored only by that fold's model. A pitcher in no fold gets
    the average of all of them."""
    from webapp.folds import fit_folds, oof_predict
    from webapp.grading import AveragedModel

    class Recorder:  # predicts 1 for pitchers it was trained on, else 0
        def __init__(self, seed):
            pass

        def fit(self, X, y):
            self.seen = set(X["pitcher"])
            return self

        def predict(self, X):
            return X["pitcher"].isin(self.seen).to_numpy(float)

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pitcher": rng.integers(0, 40, 2000), "x": rng.normal(size=2000),
                       "target": rng.normal(size=2000)})
    feats = ["pitcher", "x"]
    models, fold = fit_folds(df, feats, "target", Recorder, [0, 1], log=lambda *_: None)
    assert len(models) == 5 and set(fold) == set(df["pitcher"]) and set(fold.values()) == set(range(5))
    folds = [AveragedModel(ms, feats) for ms in models]
    ens = AveragedModel([m for ms in models for m in ms], feats)
    assert (oof_predict(folds, fold, df[feats], df["pitcher"], ens) == 0).all()
    new = pd.DataFrame({"pitcher": [999, 999], "x": [0.0, 1.0]})
    assert (oof_predict(folds, fold, new, new["pitcher"], ens) == 0).all()
    assert ens.predict(df[feats].iloc[:1])[0] == pytest.approx(0.8)  # 4 of 5 folds saw him


def test_fold_models_reproduce_final_oof():
    """The saved fold models give exactly final.py's out-of-fold predictions
    (same folds, rows, settings and seeds), so the website can reuse them."""
    from sklearn.model_selection import GroupKFold
    from lightgbm import LGBMRegressor
    from models import CANDIDATES, SEEDS, _params, fit_predict
    from webapp.build import load_synthetic
    from webapp.folds import fit_folds, oof_predict
    from webapp.grading import AveragedModel
    spec, params = CANDIDATES["A"], {"n_estimators": 30}
    df = load_synthetic([2025, 2026], 30, 80)
    df = df[df["target"].notna()].reset_index(drop=True)
    feats = spec["features"]
    oof = np.full(len(df), np.nan)  # final.py, step 1
    for i_fit, i_hold in GroupKFold(n_splits=5).split(df, groups=df["pitcher"]):
        (p,), _ = fit_predict(spec, df.iloc[i_fit], [df.iloc[i_hold]], params)
        oof[i_hold] = p
    models, fold = fit_folds(df, feats, spec["target"], lambda s: LGBMRegressor(**_params(params, s)),
                             SEEDS, log=lambda *_: None)
    ok = df[feats].notna().all(axis=1).to_numpy()
    mine = oof_predict([AveragedModel(ms, feats) for ms in models], fold, df.loc[ok, feats],
                       df.loc[ok, "pitcher"], None)
    np.testing.assert_allclose(mine, oof[ok], rtol=0, atol=1e-12)


def test_api(client, bundle):
    meta = client.get("/api/meta").json()
    assert meta["synthetic"] and meta["seasons"] == [2024, 2025, 2026]
    assert meta["out_of_fold"] and all(n.startswith("Out-of-fold") for n in meta["season_notes"].values())
    lb = client.get("/api/leaderboard/2026").json()
    assert len(lb["pitchers"]) == 40 and lb["pitch_types"]
    pid = lb["pitchers"][0]["pitcher"]
    assert len(client.get(f"/api/pitcher/{pid}").json()["seasons"]) == 3
    assert client.get("/api/leaderboard/1999").status_code == 404
    assert client.get("/").status_code == 200
    # page and script are always revalidated, so an update never leaves a stale script
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"

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
    assert fast["model"] == {"out_of_fold": True, "fold": None, "pitcher": None}

    # an arsenal loaded from an MLB pitcher is graded by the model that never saw him
    body = {"hand": "R", "pitcher": pid, "pitches": [
        {"pitch_type": "FF", "usage": 60, "velo": 95, "spin": 2300, "ivb": 10, "hb": 6}]}
    assert client.post("/api/grade", json=body).json()["model"]["fold"] == bundle["pitcher_fold"][pid]

    bad = client.post("/api/grade", json={"hand": "R", "pitches": [{"pitch_type": "FF", "velo": 300}]})
    assert bad.status_code == 422


def test_spread_rescale(bundle):
    """Displayed Stuff+ has SD 10 within each pitch type and on the pitcher
    board (qualified rows), keeping each group's mean; grades are unchanged."""
    from webapp.build import MIN_PITCH_GRADE, MIN_PITCHER_GRADE
    assert bundle["spread"] == "keep"
    for y in bundle["seasons"]:
        p = bundle["pitchers"]
        p = p[(p["year"] == y) & (p["n"] >= MIN_PITCHER_GRADE)]
        assert p["stuff_plus"].std(ddof=0) == pytest.approx(10)
        assert p["stuff_plus"].mean() == pytest.approx(p["stuff_plus_model"].mean())
        t = bundle["pitch_types"]
        t = t[(t["year"] == y) & (t["n"] >= MIN_PITCH_GRADE)]
        for k, g in t.groupby("pitch_type"):
            if len(g) >= 5:
                assert g["stuff_plus"].std(ddof=0) == pytest.approx(10)
                assert g["stuff_plus"].mean() == pytest.approx(g["stuff_plus_model"].mean())


def test_custom_pitchers(bundle_file, tmp_path):
    """Saved custom pitchers persist to their JSON file and come back graded."""
    from fastapi.testclient import TestClient
    from webapp.app import create_app
    store = tmp_path / "custom.json"
    c = TestClient(create_app(bundle_file, custom_path=store))
    assert c.get("/api/custom").json() == []
    arsenal = [{"pitch_type": "FF", "usage": 60, "velo": 96, "spin": 2400, "ivb": 17, "hb": 8},
               {"pitch_type": "SL", "usage": 40, "velo": 86, "spin": 2500, "ivb": 2, "hb": -4}]
    made = c.post("/api/custom", json={"name": "Prospect A", "hand": "R", "pitches": arsenal}).json()
    assert made["name"] == "Prospect A" and len(made["result"]["pitches"]) == 2
    assert made["based_on"] is None and made["result"]["model"]["fold"] is None
    # one started from an MLB pitcher keeps being graded by that pitcher's fold model
    pid = next(iter(joblib.load(bundle_file)["pitcher_fold"]))
    src = c.post("/api/custom", json={"name": "Tweaked", "hand": "R", "pitches": arsenal,
                                      "based_on": {"pitcher": pid, "name": "Someone"}}).json()
    assert src["based_on"] == {"pitcher": pid, "name": "Someone"} and src["result"]["model"]["pitcher"] == pid
    assert c.delete(f"/api/custom/{src['id']}").status_code == 200
    arsenal[0]["velo"] = 99
    upd = c.put(f"/api/custom/{made['id']}", json={"name": "Prospect A", "hand": "R", "pitches": arsenal}).json()
    assert upd["result"]["pitches"][0]["stuff_plus"] > made["result"]["pitches"][0]["stuff_plus"]
    # a fresh app on the same file sees it (survives restarts)
    rows = TestClient(create_app(bundle_file, custom_path=store)).get("/api/custom?season=2025").json()
    assert [r["pitches"][0]["velo"] for r in rows] == [99]
    assert c.post("/api/custom", json={"name": "", "pitches": arsenal}).status_code == 422
    assert c.delete(f"/api/custom/{made['id']}").status_code == 200
    assert c.get("/api/custom").json() == []
    assert c.delete(f"/api/custom/{made['id']}").status_code == 404


@pytest.mark.parametrize("kind", ["scaling", "bundle"])
def test_folds_step_exits_early_for_v110_models(tmp_path, monkeypatch, kind):
    """webapp.folds is only for v1.0.0: a v1.1.0+ model already ships its fold
    models, so the step stops before loading any data or fitting anything."""
    import json

    import features
    from webapp import folds

    meta = {"version": "1.1.0", "features": ["start_speed"], "target": "target", "train_years": [2022],
            "seeds": [0], "lightgbm_params": {},
            "scaling": {"xrv_mean": 0.0, "xrv_sd": 1.0, "season": 2026} if kind == "scaling" else
                       {"oof_xrv_mean": 0.0, "oof_xrv_sd": 1.0, "season": 2026}}
    path = tmp_path / "stuffplus_no_remembering_v1.1.0.json"
    path.write_text(json.dumps(meta))
    joblib.dump({"fold_models": [], "pitcher_fold": {}, "features": []} if kind == "bundle" else [],
                path.with_suffix(".joblib"))

    def boom(*a, **k):
        raise AssertionError("data was loaded")

    monkeypatch.setattr(features, "load", boom)
    monkeypatch.setattr(folds, "fit_folds", boom)
    with pytest.raises(SystemExit) as e:
        folds.main(["--model", str(path)])
    assert "isn't needed" in str(e.value) and "v1.0.0" in str(e.value)
    assert not folds.folds_path(path).exists()


def test_folds_step_still_recognises_a_v100_model(tmp_path):
    from webapp import folds
    meta = {"version": "1.0.0", "scaling": {"oof_xrv_mean": 0.0, "oof_xrv_sd": 1.0, "season": 2026}}
    path = tmp_path / "stuffplus_v1.0.0.json"
    joblib.dump([], path.with_suffix(".joblib"))                 # v1.0.0 saved a plain list of models
    assert not folds.already_has_folds(meta, path)
    assert not folds.already_has_folds(meta, tmp_path / "missing.json")


# ------------------------------------------------ v1.1.0 site path (no-remembering bundle)
@pytest.fixture(scope="module")
def v110_site(tmp_path_factory):
    """A tiny Stuff+ (no remembering) v1.1.0 model in a temp models dir, shaped
    like final.py's: fold models over 5 GroupKFold-by-pitcher folds, the fold
    each pitcher was held out of, and metadata with xrv_mean scaling."""
    import json

    from sklearn.model_selection import GroupKFold

    import models as stuff_models
    from webapp import build as wb

    root = tmp_path_factory.mktemp("site_root")
    mdir = root / "models"
    mdir.mkdir()
    size = (40, 120)
    df = wb.load_synthetic([2024, 2025, 2026], *size)
    spec = stuff_models.CANDIDATES["A"]
    feats = spec["features"]
    small = {"n_estimators": 12, "learning_rate": 0.1}
    fold_models, pitcher_fold = [], {}
    for k, (i_fit, i_hold) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["pitcher"])):
        fold_models.append(stuff_models.fit_seed_models(spec, df.iloc[i_fit], small))
        pitcher_fold.update(dict.fromkeys((int(p) for p in df["pitcher"].iloc[i_hold].unique()), k))
    xrv = stuff_models.crossfit_xrv(fold_models, pitcher_fold, df[df["year"] == 2026], feats)
    meta = {"name": "Stuff+ (no remembering)", "version": "1.1.0", "feature_set": "A", "features": feats,
            "target": "target", "train_years": [2024, 2025, 2026], "seeds": [0, 1, 2],
            "shrinkage": {"pitcher_k": {"k": 12.0}, "pitch_type_k": {"k": 4.0}},
            "scaling": {"xrv_mean": float(np.nanmean(xrv)), "xrv_sd": float(np.nanstd(xrv)), "season": 2026}}
    stem = mdir / "stuffplus_no_remembering_v1.1.0"
    joblib.dump({"fold_models": fold_models, "pitcher_fold": pitcher_fold, "features": feats},
                stem.with_suffix(".joblib"))
    stem.with_suffix(".json").write_text(json.dumps(meta))
    # an older remembering-model file next to it must not be picked instead
    (mdir / "stuffplus_v1.1.0.json").write_text(json.dumps({**meta, "name": "Stuff+"}))
    joblib.dump({"models": fold_models[0], "features": feats}, mdir / "stuffplus_v1.1.0.joblib")
    return {"root": root, "models": mdir, "meta": stem.with_suffix(".json"), "size": size,
            "fold_models": fold_models, "pitcher_fold": pitcher_fold, "df": df, "feats": feats}


@pytest.fixture()
def v110_paths(v110_site, monkeypatch):
    from webapp import build as wb
    monkeypatch.setattr(wb, "MODELS", v110_site["models"])
    monkeypatch.setattr(wb, "ROOT", v110_site["root"])       # no leaderboards/ here: nothing published
    return v110_site


def test_site_builds_from_a_v110_no_remembering_bundle(v110_paths):
    from webapp import build as wb
    site = v110_paths
    assert wb.frozen_meta() == site["meta"]
    out = wb.build(synthetic=True, frozen=wb.frozen_meta(), synth_size=site["size"],
                   out=site["models"] / "site.joblib")
    b = joblib.load(out)
    assert b["variant"] == "Stuff+ (no remembering) v1.1.0" and b["frozen"]
    assert b["seasons"] == [2024, 2025, 2026] and len(b["folds"]) == 5
    assert b["pitcher_fold"] == site["pitcher_fold"]
    # every season is scored out-of-fold, none in-sample
    assert all(note.startswith("Out-of-fold") for note in b["season_notes"].values()), b["season_notes"]
    # 2026 uses the model's own published scaling constants
    meta_scale = __import__("json").loads(site["meta"].read_text())["scaling"]
    assert b["scale"][2026] == {"mu": meta_scale["xrv_mean"], "sd": meta_scale["xrv_sd"]}

    # the grader scores a known pitcher with the fold that never saw him
    pt = b["pitch_types"]
    by_fold = {}
    for pid in pt[pt["year"] == 2026]["pitcher"].unique():
        by_fold.setdefault(site["pitcher_fold"][int(pid)], []).append(int(pid))
    (k1, ids1), (k2, ids2) = list(by_fold.items())[:2]
    rows = pt[(pt["pitcher"] == ids1[0]) & (pt["year"] == 2026)]
    hand = rows["hand"].iat[0]
    pitches = rows[["pitch_type", "usage", "velo", "spin", "ext", "rel_side", "rel_height",
                    "ivb", "hb", "spin_axis", "arm_angle"]].to_dict("records")
    own = grade_arsenal(b, hand, pitches, pitcher=ids1[0])
    assert own["model"] == {"out_of_fold": True, "fold": k1, "pitcher": ids1[0]}
    score = lambda r: [p["stuff_plus_model"] for p in r["pitches"]]  # noqa: E731
    # same arsenal, another pitcher of the same fold -> same model, same scores
    assert score(grade_arsenal(b, hand, pitches, pitcher=ids1[1])) == score(own)
    # a pitcher of a different fold, or nobody, is a different model
    other = grade_arsenal(b, hand, pitches, pitcher=ids2[0])
    assert other["model"]["fold"] == k2 and score(other) != score(own)
    anon = grade_arsenal(b, hand, pitches)
    assert anon["model"] == {"out_of_fold": True, "fold": None, "pitcher": None}   # mean of all folds
    assert score(anon) != score(own)
