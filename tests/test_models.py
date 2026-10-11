import numpy as np
import pandas as pd
import pytest

import develop
import models
from features import add_expected_target, add_features
from synthetic import make


@pytest.fixture(scope="module")
def data():
    df = add_features(add_expected_target(make(n_pitchers=12, pitches_per_season=120,
                                               years=(2022, 2023), seed=3)))
    rng = np.random.default_rng(0)
    df["description"] = rng.choice(
        ["ball", "called_strike", "swinging_strike", "foul", "hit_into_play", "foul_tip",
         "hit_by_pitch", "automatic_ball"], len(df), p=[0.34, 0.16, 0.1, 0.18, 0.17, 0.01, 0.01, 0.03])
    df["game_pk"] = rng.integers(0, 30, len(df)) + df["year"] * 100
    df["at_bat_number"] = np.arange(len(df)); df["pitch_number"] = 1
    return models.add_outcome_flags(df)


def test_outcome_flags(data):
    d = data.set_index("description")
    assert d.loc["automatic_ball", "is_swing"].isna().all()
    assert (d.loc["foul", ["is_swing", "is_whiff"]] == [1, 0]).all().all()
    assert (d.loc["foul_tip", "is_whiff"] == 1).all() and (d.loc["hit_into_play", "is_whiff"] == 0).all()
    assert (d.loc["called_strike", ["is_swing", "is_cs"]] == [0, 1]).all().all()
    assert (d.loc["hit_by_pitch", "is_cs"] == 0).all() and d.loc["ball", "is_whiff"].isna().all()
    assert not (models.SWING & models.TAKE)


def test_candidate_list_is_the_preregistered_one():
    assert list(models.CANDIDATES) == ["A", "J", "G", "H", "JGH", "COMP"]
    assert models.CANDIDATES["JGH"]["features"] == models.CANDIDATES["COMP"]["features"]
    assert models.CANDIDATES["A"]["features"] == models.CANDIDATES["J"]["features"]
    assert list(models.AMENDED) == ["S6"] and len(models.AMENDED["S6"]["features"]) == 6
    assert models.TJ_PARAMS["n_estimators"] == 1000 and models.TJ_PARAMS["learning_rate"] == 0.01


def test_combine_components_matches_hand_calculation():
    v = {"v_whiff": -0.1, "v_cs": -0.05, "v_ball": 0.06}
    got = models.combine_components(np.array([0.5]), np.array([0.2]), np.array([0.4]), np.array([0.03]), v)
    assert got[0] == pytest.approx(0.5 * (0.2 * -0.1 + 0.8 * 0.03) + 0.5 * (0.4 * -0.05 + 0.6 * 0.06))
    # always swings and always misses -> the whiff value
    assert models.combine_components(1.0, 1.0, 0.3, 9.9, v) == pytest.approx(-0.1)


@pytest.mark.parametrize("name", ["A", "JGH", "COMP"])
def test_fit_predict_is_seeded_and_marks_missing_features(data, name):
    spec = models.CANDIDATES[name]
    tr, sc = data[data["year"] == 2022], data[data["year"] == 2023].copy()
    sc.loc[sc.index[:5], spec["features"][0]] = np.nan
    small = {"n_estimators": 20, "learning_rate": 0.1}
    (p1,), info = models.fit_predict(spec, tr, [sc], small)
    (p2,), _ = models.fit_predict(spec, tr, [sc], small)
    np.testing.assert_array_equal(p1, p2)
    assert np.isnan(p1[:5]).all() and np.isfinite(p1[5:]).all() and info["n_train"] > 0
    assert np.nanstd(p1) > 0 and np.nanmax(np.abs(p1)) < 1


def test_component_tasks_use_the_right_rows(data):
    tr = data.dropna(subset=["target", "is_swing"])
    tasks = {n: (rows, label) for n, rows, label, _ in models.component_tasks(tr)}
    assert len(tasks["swing"][0]) == len(tr)
    assert set(tasks["whiff"][0]["description"]) <= models.SWING
    assert set(tasks["cs"][0]["description"]) <= models.TAKE
    assert set(tasks["contact"][0]["description"]) == {"foul", "hit_into_play"}


def test_inner_folds_never_share_a_pitcher(data):
    for i_fit, i_hold in develop.inner_folds(data):
        assert not set(data["pitcher"].iloc[i_fit]) & set(data["pitcher"].iloc[i_hold])


def test_shrink_pulls_small_samples_toward_the_league_mean():
    t = pd.DataFrame({"mean": [120.0, 120.0, 90.0], "size": [10, 1000, 1000]}, index=[1, 2, 3])
    mu = (t["mean"] * t["size"]).sum() / t["size"].sum()
    s = develop.shrink(t, k=100)
    assert s.loc[1, "mean"] == pytest.approx(mu + 10 / 110 * (120 - mu))
    assert abs(s.loc[1, "mean"] - mu) < abs(s.loc[2, "mean"] - mu) < abs(120 - mu)
    assert develop.shrink(t, k=0)["mean"].tolist() == t["mean"].tolist()
    assert t["mean"].tolist() == [120.0, 120.0, 90.0]            # input untouched
    # per-pitch-type league means
    idx = pd.MultiIndex.from_tuples([(1, "FF"), (2, "FF"), (1, "SL"), (2, "SL")], names=["pitcher", "pitch_type"])
    pt = pd.DataFrame({"mean": [110.0, 90.0, 60.0, 80.0], "size": [100, 100, 100, 100]}, index=idx)
    assert develop.shrink(pt, 100, "pitch_type")["mean"].tolist() == [105.0, 95.0, 65.0, 75.0]


def test_split_half_k_recovers_a_known_constant():
    # true talent sd 5, pitch noise sd 20 -> k = 400 / 25 = 16
    rng = np.random.default_rng(0)
    rows = []
    for p in range(600):
        t = rng.normal(100, 5)
        n = int(rng.integers(120, 400))
        rows.append(pd.DataFrame({"pitcher": p, "year": 2022, "game_pk": rng.integers(0, 30, n),
                                  "s": t + rng.normal(0, 20, n)}))
    d = pd.concat(rows, ignore_index=True)
    d["game_date"] = pd.Timestamp("2022-04-01") + pd.to_timedelta(d["game_pk"], unit="D")
    out = develop.split_half_k(d, d["s"].to_numpy(), ["pitcher", "year"], 100)
    assert out["k"] == pytest.approx(16, rel=0.25) and 0 < out["r_half"] < 1 and out["groups"] == 600


def test_tuned_component_models_get_their_own_tree_counts():
    p = models._params({"n_estimators_by_component": {"swing": 7, "cs": 9}, "num_leaves": 15}, 1, "cs")
    assert p["n_estimators"] == 9 and p["num_leaves"] == 15 and p["random_state"] == 1
    assert "n_estimators_by_component" not in p
    assert models._params(None, 0)["n_estimators"] == 1000


def test_tuning_search_space_is_regularizing_only(data):
    import optuna
    import tune
    assert (tune.N_TRIALS, tune.N_FOLDS, tune.PATIENCE, tune.MAX_TREES) == (30, 3, 50, 3000)
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)).ask()
    assert set(tune.suggest(trial)) == {"min_child_samples", "num_leaves", "reg_lambda", "learning_rate"}
    spec = models.CANDIDATES["A"]
    (name, rows, label, clf), = tune.tasks_for(spec, data)
    loss, trees = tune.cv_loss(rows, spec["features"], label, clf,
                               {"n_estimators": 40, "learning_rate": 0.1}, early_stopping=True)
    assert loss > 0 and 1 <= trees <= 40
    assert [t[0] for t in tune.tasks_for(models.CANDIDATES["COMP"], data)] == ["swing", "whiff", "cs", "contact"]


def test_confirmation_runs_end_to_end_and_applies_the_rule(data):
    import confirm
    d = pd.concat([data, data[data["year"] == 2023].assign(year=2024, game_pk=lambda x: x["game_pk"] + 1000)],
                  ignore_index=True)
    d["woba_denom"] = 1
    small = {"n_estimators": 10, "learning_rate": 0.1}
    out = confirm.run_confirmation(d, {"winner": "JGH", "params": {"A": small, "JGH": small}},
                                   [2022], 2023, n_boot=50, log=lambda m: None)
    assert set(out["results"]) == {"A", "JGH"} and set(out["vs_A"]) == {
        f"{s}_{m}" for s in ("raw", "shrunk") for m in (100, 300, 500)}
    assert out["results"]["A"]["raw_100"]["n"] == 12 and out["shrinkage"]["A"]["pitcher"]["k"] > 0
    assert isinstance(out["verdict"]["confirmed"], bool)
    solo = confirm.run_confirmation(d, {"winner": "A", "params": {}}, [2022], 2023, n_boot=10,
                                    log=lambda m: None)
    assert "verdict" not in solo and list(solo["results"]) == ["A"]


@pytest.mark.parametrize("woba,rv,stick,want", [
    ((0.02, 0.005), (0.02, 0.004), -0.005, True),
    ((0.02, -0.001), (0.02, 0.004), 0.0, False),      # one CI includes 0
    ((0.02, 0.005), (0.02, 0.004), -0.02, False),     # stickiness fell too far
    ((-0.02, -0.03), (0.02, 0.004), 0.0, False),
])
def test_confirmation_rule(woba, rv, stick, want):
    import confirm
    vs = {"diff": {"next_woba": woba[0], "next_rv100": rv[0], "stickiness": stick},
          "ci": {"next_woba": (woba[1], 0.1), "next_rv100": (rv[1], 0.1), "stickiness": (-0.1, 0.1)}}
    assert confirm.verdict(vs)["confirmed"] is want
