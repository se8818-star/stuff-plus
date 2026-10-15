import numpy as np
import pandas as pd
import pytest

import final


def test_grades_are_20_to_80_centred_on_50():
    x = pd.Series(np.random.default_rng(0).normal(100, 5, 500))
    g = final.grade(x)
    assert g.mean() == pytest.approx(50, abs=0.2) and g.std() == pytest.approx(10, abs=0.3)
    extreme = final.grade(pd.Series([0.0] + [100.0] * 200 + [200.0]))
    assert extreme.iloc[0] == 20 and extreme.iloc[-1] == 80 and extreme.iloc[1] == 50


def test_roles_from_first_pitch_of_each_half_inning_side():
    rows = []
    for game in range(4):                      # pitcher 1 starts every game for the home side
        rows += [(game, "Top", 1, 1, 1), (game, "Top", 30, 1, 2), (game, "Bot", 2, 1, 3 + game), (game, "Bot", 40, 1, 2)]
    df = pd.DataFrame(rows, columns=["game_pk", "inning_topbot", "at_bat_number", "pitch_number", "pitcher"])
    r = final.roles(df)
    assert r.loc[1, ["g", "gs", "role"]].tolist() == [4, 4, "SP"]
    assert r.loc[2, ["g", "gs", "role"]].tolist() == [4, 0, "RP"]
    assert r.loc[3, "role"] == "SP"


def test_park_check_recovers_a_park_effect():
    rng = np.random.default_rng(1)
    rows = []
    for p in range(60):
        talent = rng.normal(100, 5)
        for park in ("COL", "AAA", "BBB"):
            n = 80
            rows.append(pd.DataFrame({"pitcher": p, "home_team": park,
                                      "s": talent + (-4 if park == "COL" else 0) + rng.normal(0, 10, n)}))
    d = pd.concat(rows, ignore_index=True)
    out = final.park_check(d, d["s"].to_numpy())
    assert out.index[0] == "COL" and out.loc["COL", "stuff_plus_diff"] == pytest.approx(-4, abs=0.6)
    assert out.loc["AAA", "stuff_plus_diff"] == pytest.approx(2, abs=0.6)   # 'elsewhere' includes COL
    assert out.loc["COL", "pitchers"] == 60
