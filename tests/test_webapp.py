"""Website: model bundle, grader physics, and API, on a small synthetic bundle."""
import json

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
    return build(synthetic=True, out=out, synth_size=(40, 120), spread="keep")


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


def test_default_scale_is_raw(tmp_path):
    """By default the site shows the model's own scale (as tjStuff+ does):
    displayed Stuff+ equals stuff_plus_model for pitchers and pitch types."""
    b = joblib.load(build(synthetic=True, out=tmp_path / "raw.joblib", synth_size=(40, 120)))
    assert b["spread"] is None
    for k in ("pitchers", "pitch_types"):
        assert b[k]["stuff_plus"].equals(b[k]["stuff_plus_model"])


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


def test_published_season_keeps_unqualified_pitchers(v110_paths):
    """The published CSVs only hold qualified pitchers. The site shows their
    published numbers and keeps everyone else with its own scores, so the
    published season lists the same pitchers as any other season."""
    from webapp import build as wb
    site = v110_paths
    plain = joblib.load(wb.build(synthetic=True, frozen=site["meta"], synth_size=site["size"],
                                 out=site["models"] / "plain.joblib", spread=None))
    own = plain["pitchers"][plain["pitchers"]["year"] == 2026].set_index("pitcher")
    own_t = plain["pitch_types"][plain["pitch_types"]["year"] == 2026]
    published = sorted(own.index)[::2]                         # every other pitcher is "qualified"
    lb = site["root"] / "leaderboards"
    lb.mkdir(exist_ok=True)
    try:
        op = own.loc[published].reset_index()
        pd.DataFrame({"rank": range(1, len(op) + 1), "pitcher": op["pitcher"], "pitches": op["n"] + 7,
                      "stuff_plus": 150.0 - np.arange(len(op)), "stuff_plus_raw": 151.0, "grade": 77,
                      "role": "SP", "g": 9, "gs": 9}).to_csv(lb / "pitchers_2026_no_remembering.csv", index=False)
        ot = own_t[own_t["pitcher"].isin(published) & (own_t["pitch_type"] == "FF")]
        pd.DataFrame({"pitch_type": "FF", "rank_in_type": range(1, len(ot) + 1), "pitcher": ot["pitcher"].to_numpy(),
                      "pitches": ot["n"].to_numpy(), "stuff_plus": 140.0, "stuff_plus_raw": 141.0,
                      "grade": 66}).to_csv(lb / "pitch_types_2026_no_remembering.csv", index=False)
        b = joblib.load(wb.build(synthetic=True, frozen=site["meta"], synth_size=site["size"],
                                 out=site["models"] / "published.joblib", spread=None))
    finally:
        for f in lb.glob("*.csv"):
            f.unlink()
        lb.rmdir()
    cur = b["pitchers"][b["pitchers"]["year"] == 2026].set_index("pitcher")
    assert sorted(cur.index) == sorted(own.index)                         # nobody vanished
    assert len(cur) == (b["pitchers"]["year"] == 2025).sum()
    # published pitchers: the published numbers, in published order, first
    assert list(cur.index[:len(published)]) == published
    assert cur.loc[published, "stuff_plus"].tolist() == list(150.0 - np.arange(len(published)))
    assert (cur.loc[published, "n"] == own.loc[published, "n"] + 7).all()
    assert (cur.loc[published, "grade"] == 77).all() and (cur.loc[published, "role"] == "SP").all()
    # everyone else: the site's own scores, with a grade computed by the site
    rest = [p for p in own.index if p not in published]
    np.testing.assert_allclose(cur.loc[rest, "stuff_plus"], own.loc[rest, "stuff_plus"])
    np.testing.assert_allclose(cur.loc[rest, "stuff_plus_raw"], own.loc[rest, "stuff_plus_raw"])
    assert (cur.loc[rest, "n"] == own.loc[rest, "n"]).all() and cur.loc[rest, "grade"].notna().all()
    assert "site's own" in b["season_notes"][2026] and b["season_notes"][2025].startswith("Out-of-fold")
    # pitch types: published where published, the site's own (not NaN) elsewhere
    t = b["pitch_types"][b["pitch_types"]["year"] == 2026]
    pub_rows = t["pitcher"].isin(published) & (t["pitch_type"] == "FF")
    assert (t.loc[pub_rows, "stuff_plus"] == 140.0).all() and len(t) == len(own_t)
    assert t["stuff_plus"].notna().all() and (t.loc[~pub_rows, "stuff_plus"] != 140.0).all()


def test_roles_use_pitches_with_missing_features(v110_paths, monkeypatch):
    """A pitch with a missing feature can't be scored, but it still counts for
    who started the game, as in final.roles."""
    import final
    from webapp import build as wb
    site = v110_paths
    df = site["df"].copy()
    # pitchers 0 and 1 share four games a season: 0 throws the first 30 pitches, 1 the rest
    pair = df["pitcher"].isin([0, 1])
    seq = df.groupby(["pitcher", "year"]).cumcount()
    df["game_pk"] = df["year"] * 1000 + np.where(pair, seq // 30, 100 + df["pitcher"])
    df["inning_topbot"] = "Top"
    df["at_bat_number"] = seq + np.where(df["pitcher"] == 1, 1000, 0)
    df["pitch_number"] = 1
    # the starter's pitches in three of the four games have no spin reading
    df.loc[(df["pitcher"] == 0) & (seq < 90), "spin_rate"] = np.nan
    monkeypatch.setattr(wb, "load_synthetic", lambda *a, **k: df.copy())
    b = joblib.load(wb.build(synthetic=True, frozen=site["meta"], synth_size=site["size"],
                             out=site["models"] / "roles.joblib", spread=None))
    board = b["pitchers"].set_index(["pitcher", "year"])
    want = final.roles(df[df["year"] == 2026])
    for pid in (0, 1):
        row = board.loc[(pid, 2026)]
        assert (row["games"], row["gs"], row["role"]) == tuple(want.loc[pid, ["g", "gs", "role"]])
    assert board.loc[(0, 2026), "role"] == "SP" and board.loc[(0, 2026), "gs"] == 4
    assert board.loc[(1, 2026), "role"] == "RP" and board.loc[(1, 2026), "gs"] == 0
    assert board.loc[(0, 2026), "n"] == 30                 # only the scoreable pitches are graded


def _js_engine():
    import shutil
    from pathlib import Path
    jsc = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc")
    return shutil.which("node") or (str(jsc) if jsc.exists() else None)


def test_grader_dropdown_keeps_pitch_types_without_a_template(tmp_path):
    """An arsenal loaded from the leaderboard can hold FO, SC, KN ...; the row's
    dropdown must offer that type, or it shows (and becomes) the first template."""
    import re
    import subprocess
    from pathlib import Path
    src = (Path(webapp.__file__).parent / "static" / "app.js").read_text()
    fn = re.search(r"function pitchTypeOptions\(types, current\) \{.*?\n\}", src, re.S)
    assert fn, "pitchTypeOptions is missing"
    # the row's <select> is built from it, not from the bare template list
    assert "pitchTypeOptions(types, p.pitch_type).map(" in src
    assert "types.map(t => `<option value=\"${t}\" title" not in src
    engine = _js_engine()
    if engine is None:
        pytest.skip("no JavaScript engine (node or jsc) to run the helper")
    script = tmp_path / "check.js"
    script.write_text(fn.group(0) + """
const out = typeof console !== "undefined" && console.log ? console.log : print;
const T = ["FF", "SI", "SL"];
out(JSON.stringify([pitchTypeOptions(T, "FO"), pitchTypeOptions(T, "SL"), pitchTypeOptions(T, null),
                    pitchTypeOptions(T, "KN").filter(t => t === "KN").length, T.length]));
""")
    res = subprocess.run([engine, str(script)], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    import json
    assert json.loads(res.stdout.strip().splitlines()[-1]) == [
        ["FO", "FF", "SI", "SL"], ["FF", "SI", "SL"], ["FF", "SI", "SL"], 1, 3]


def test_concurrent_saves_do_not_lose_each_other(bundle_file, tmp_path, monkeypatch):
    """Saves arriving at the same time must all land: the read-modify-write of
    the JSON file is serialized, and each write uses its own temp file."""
    import threading
    import time

    from fastapi.testclient import TestClient

    import webapp.app as app_module
    from webapp.app import create_app

    real_dumps = app_module.json.dumps

    def slow_dumps(*a, **k):                      # hold the window between read and write open
        time.sleep(0.02)
        return real_dumps(*a, **k)

    monkeypatch.setattr(app_module.json, "dumps", slow_dumps)
    monkeypatch.setattr(app_module, "grade_arsenal", lambda *a, **k: {})     # grading is not under test
    store = tmp_path / "custom.json"
    c = TestClient(create_app(bundle_file, custom_path=store))
    arsenal = [{"pitch_type": "FF", "usage": 60, "velo": 96, "spin": 2400, "ivb": 17, "hb": 8},
               {"pitch_type": "SL", "usage": 40, "velo": 86, "spin": 2500, "ivb": 2, "hb": -4}]
    codes = []

    def save(i):
        codes.append(c.post("/api/custom", json={"name": f"P{i}", "hand": "R", "pitches": arsenal}).status_code)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200] * 12
    saved = app_module.json.loads(store.read_text())
    assert sorted(r["name"] for r in saved) == sorted(f"P{i}" for i in range(12))
    assert len({r["id"] for r in saved}) == 12
    assert [p.name for p in tmp_path.iterdir()] == ["custom.json"]            # no temp files left behind

    # a failed write leaves the saved file intact and no temp file
    def broken_dumps(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_module.json, "dumps", broken_dumps)
    with pytest.raises(RuntimeError):
        c.post("/api/custom", json={"name": "lost", "hand": "R", "pitches": arsenal})
    assert len(app_module.json.loads(store.read_text())) == 12
    assert [p.name for p in tmp_path.iterdir()] == ["custom.json"]


def test_site_models_pick_newest_of_each_family(v110_paths):
    """The model menu offers each frozen family once (newest version), no-remembering first."""
    from webapp import build as wb
    site = v110_paths
    old = site["models"] / "stuffplus_v1.0.0"
    old.with_suffix(".json").write_text(json.dumps({"name": "stuffplus", "version": "1.0.0"}))
    joblib.dump([], old.with_suffix(".joblib"))
    (site["models"] / "stuffplus_s6_v1.0.0.json").write_text("{}")    # no .joblib: not offered
    assert wb.site_models() == [site["meta"], site["models"] / "stuffplus_v1.1.0.json"]
    assert wb.model_name("stuffplus_v1.1.0") == "Stuff+ (remembering)"
    assert wb.model_name("stuffplus_s6_v1.0.0") == "S6 (approach angles)"


def test_model_menu_switches_leaderboards_and_my_pitchers(v110_paths, tmp_path):
    """Two bundles behind one app: ?model= picks the leaderboard, the grader and
    the grades of saved custom pitchers; no model means the first (default)."""
    from fastapi.testclient import TestClient
    from webapp import build as wb
    from webapp.app import create_app, model_id
    site = v110_paths
    nr_meta, rem_meta = wb.site_models()
    paths = [wb.build(synthetic=True, frozen=m, synth_size=site["size"],
                      out=site["models"] / f"stuffplus_site_{m.stem}.joblib") for m in (nr_meta, rem_meta)]
    rem = joblib.load(paths[1])
    assert rem["model_name"] == "Stuff+ (remembering)" and not rem["folds"]
    assert all(n.startswith("In-sample") for n in rem["season_notes"].values())

    nr_id, rem_id = model_id(paths[0]), model_id(paths[1])
    assert (nr_id, rem_id) == (nr_meta.stem, "stuffplus_v1.1.0")
    c = TestClient(create_app(paths, custom_path=tmp_path / "custom.json"))
    ms = c.get("/api/models").json()
    assert [(m["id"], m["name"], m["default"]) for m in ms] == [
        (nr_id, "Stuff+ (no remembering)", True), (rem_id, "Stuff+ (remembering)", False)]
    assert c.get("/api/meta").json()["model"] == nr_id
    assert c.get(f"/api/meta?model={rem_id}").json()["model_name"] == "Stuff+ (remembering)"
    assert c.get("/api/meta?model=nope").status_code == 404

    board = lambda q: {r["pitcher"]: r["stuff_plus"]  # noqa: E731
                       for r in c.get(f"/api/leaderboard/2026{q}").json()["pitchers"]}
    default, nr, remb = board(""), board(f"?model={nr_id}"), board(f"?model={rem_id}")
    assert default == nr and nr.keys() == remb.keys() and nr != remb

    arsenal = {"name": "Test arm", "hand": "R", "pitches": [
        {"pitch_type": "FF", "usage": 60, "velo": 96, "spin": 2400, "ivb": 17, "hb": 8},
        {"pitch_type": "SL", "usage": 40, "velo": 86, "spin": 2600, "ivb": 1, "hb": -5}]}
    assert c.post("/api/custom", json=arsenal).status_code == 200
    graded = {m["id"]: c.get(f"/api/custom?season=2026&model={m['id']}").json()[0]["result"]["overall"]["stuff_plus"]
              for m in ms}
    assert graded[nr_id] != graded[rem_id]


def test_pitch_types_carry_approach_angles(bundle):
    """Explore's VAA/HAA: Savant-style plate angles per pitcher-pitch type, HAA in
    the arm-side frame (from an arm-side release a pitch angles glove-side, < 0)."""
    pt = bundle["pitch_types"]
    assert pt["vaa"].between(-15, 0).all() and pt["haa"].between(-10, 10).all()
    by_type = pt.groupby("pitch_type")[["vaa", "haa"]].mean()
    assert by_type.loc["FF", "vaa"] > by_type.loc["SL", "vaa"]      # fastballs arrive flatter
    assert by_type.loc["CH", "haa"] > by_type.loc["SL", "haa"]      # changeups run arm side
    for hand, g in pt.groupby("hand"):
        assert g["haa"].mean() < 0, hand
