import json

import numpy as np
import pandas as pd
import pytest

from stuffplus import run_ablation as ra
def table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(100, 10, n)
    return pd.DataFrame({"mean_y1": s, "mean_y2": s + rng.normal(0, 4, n),
                         "woba": 0.32 - 0.002 * (s - 100) + rng.normal(0, 0.02, n),
                         "k_bb": 0.12 + 0.003 * (s - 100) + rng.normal(0, 0.03, n),
                         "rv100": -0.05 * (s - 100) + rng.normal(0, 0.5, n)},
                        index=pd.Index(range(n), name="pitcher"))


def test_metrics_are_signed_so_higher_is_better():
    m = ra.metrics(table())
    assert all(m[k] > 0.3 for k in ("stickiness", "next_woba", "next_k_bb", "next_rv100"))


def test_bootstrap_is_seeded_and_zero_for_identical_tables():
    j = table()
    assert ra.bootstrap_diff(j, j, n=50) == {k: (0.0, 0.0) for k in ra.metrics(j)}
    worse = j.assign(mean_y1=np.random.default_rng(1).normal(100, 10, len(j)))
    a, b = ra.bootstrap_diff(j, worse, n=200, seed=3), ra.bootstrap_diff(j, worse, n=200, seed=3)
    assert a == b
    assert a["next_woba"][1] < 0                     # CI excludes 0 for a random score


def test_pitcher_stuff_scale():
    df = pd.DataFrame({"pitcher": [1, 1, 2, 2]})
    out = ra.pitcher_stuff(df, np.array([0.0, 0.0, -0.01, -0.03]), mu=0.0, sd=0.01)
    assert out.loc[1, "mean"] == 100 and out.loc[2, "mean"] == 120 and out.loc[2, "size"] == 2


def test_env_info_reports_versions():
    env = ra.env_info()
    assert env["numpy"] == np.__version__ and env["python"]
    json.dumps(env)


def test_fold_bootstrap_resamples_pitchers_jointly():
    j1, j2 = table(60, 0), table(50, 1)
    j2.index = j2.index + 30                          # 30 pitchers appear in both folds
    rng = np.random.default_rng(5)
    v1 = j1.assign(mean_y1=j1["mean_y1"] + rng.normal(0, 6, len(j1)))
    v2 = j2.assign(mean_y1=j2["mean_y1"] + rng.normal(0, 6, len(j2)))
    bs = ra.bootstrap_folds([j1, j2], [v1, v2], n=300, seed=1)
    assert bs == ra.bootstrap_folds([j1, j2], [v1, v2], n=300, seed=1)
    for i, (jb, jv) in enumerate([(j1, v1), (j2, v2)]):
        for k in ra.KEYS:                             # point diff == metric(var) - metric(base)
            assert bs["diff"][i][k] == pytest.approx(ra.metrics(jv)[k] - ra.metrics(jb)[k])
            lo, hi = bs["ci"][i][k]
            assert lo <= hi
    for k in ra.KEYS:
        assert bs["mean_diff"][k] == pytest.approx((bs["diff"][0][k] + bs["diff"][1][k]) / 2)
        lo, hi = bs["mean_ci"][k]
        assert lo < bs["mean_diff"][k] < hi
    assert bs["mean_ci"]["next_woba"][1] < 0          # noisier score is detectably worse


def test_season_weights():
    w = ra.season_weights([2022, 2023, 2024, 2024], 0.5)
    np.testing.assert_allclose(w, [0.25, 0.5, 1.0, 1.0])


def test_pitch_type_stickiness_uses_per_type_minimum():
    n = 40
    pit = np.repeat(np.arange(8), n)
    df = pd.DataFrame({"pitcher": pit, "pitch_type": np.where(np.arange(len(pit)) % n < 30, "FF", "SL")})
    s = 100 + pit * 2.0 + np.random.default_rng(0).normal(0, 1, len(pit))
    out = ra.pitch_type_stickiness(df, df, s, s, min_n=20)
    assert out["FF"]["n"] == 8 and out["FF"]["r"] == pytest.approx(1.0)
    assert out["SL"]["n"] == 0 and np.isnan(out["SL"]["r"])


def test_grid_is_two_dimensional_on_synthetic_data():
    from stuffplus.features import add_features
    from stuffplus.synthetic import make
    df = add_features(make(n_pitchers=10, pitches_per_season=80, years=(2024,)))
    cum, detail = ra.redundancy_report(df)
    assert cum[1] > 0.999 and cum[0] < 0.95
    assert detail["max_resid_sd_deg_after_center_pair"] < 0.05
