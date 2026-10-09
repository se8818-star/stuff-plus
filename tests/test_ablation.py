import json

import numpy as np
import pandas as pd

import run_ablation as ra


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
