"""
Pitcher-season outcomes and our scores for the TJStats comparison.

    python -m experiments.crossfit_test                      # first: results/preds/crossfit_*.parquet
    python -m experiments.tjstats_comparison.build           # -> results/tjstats/

Reads data/statcast_2021-2026.parquet, the forward (remembering) and cross-fitted
(no remembering) per-pitch predictions for 2022-2025, and the committed 2026
leaderboards. Writes results/tjstats/outcomes.parquet and ours.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from stuffplus.features import EXCLUDED_PITCH_TYPES, add_target, plain_dtypes

ROOT = Path(__file__).resolve().parents[2]
D = ROOT
OUT = ROOT / "results" / "tjstats"
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["game_pk", "at_bat_number", "pitch_number"]
OUTS = {"strikeout": 1, "field_out": 1, "force_out": 1, "fielders_choice_out": 1, "sac_fly": 1, "sac_bunt": 1,
        "double_play": 2, "grounded_into_double_play": 2, "strikeout_double_play": 2, "sac_fly_double_play": 2,
        "sac_bunt_double_play": 2, "triple_play": 3, "fielders_choice": 0, "other_out": 1}
outs_all, scores = [], []
for y in range(2021, 2027):
    df = plain_dtypes(pd.read_parquet(f"{D}/data/statcast_{y}.parquet"))
    df = df[(df.game_type == "R") & df.pitch_type.notna() & ~df.pitch_type.isin(EXCLUDED_PITCH_TYPES)].copy()
    df = add_target(df)
    ev = df[df.events.notna()]
    g = ev.groupby("pitcher")
    o = pd.DataFrame({"pa": g.size(),
                      "k": g.events.apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).sum()),
                      "bb": g.events.apply(lambda e: (e == "walk").sum()),
                      "hbp": g.events.apply(lambda e: (e == "hit_by_pitch").sum()),
                      "hr": g.events.apply(lambda e: (e == "home_run").sum()),
                      "outs": g.events.apply(lambda e: e.map(OUTS).fillna(0).sum())})
    o["ip"] = o.outs / 3
    lg = o.sum()
    o["k_bb"] = (o.k - o.bb) / o.pa
    fip_raw = (13 * o.hr + 3 * (o.bb + o.hbp) - 2 * o.k) / o.ip
    o["fip"] = fip_raw  # constant omitted: irrelevant for correlations
    w = df.groupby("pitcher")[["woba_value", "woba_denom"]].sum()
    o["woba"] = w.woba_value / w.woba_denom
    xw = np.where(df.description.eq("hit_into_play") & df.estimated_woba_using_speedangle.notna(),
                  df.estimated_woba_using_speedangle, df.woba_value)
    t = pd.DataFrame({"pitcher": df.pitcher, "xw": np.where(df.woba_denom > 0, xw, 0), "d": df.woba_denom.fillna(0)}).groupby("pitcher").sum()
    o["xwoba"] = t.xw / t.d
    rv = df.dropna(subset=["target"]).groupby("pitcher")["target"].agg(["sum", "size"])
    o["rv100"] = rv["sum"] / rv["size"] * 100
    o["pitches"] = df.groupby("pitcher").size()
    o["velo_fb"] = df[df.pitch_type.isin(["FF", "SI"])].groupby("pitcher").release_speed.mean()
    o["arm_angle"] = df.groupby("pitcher").arm_angle.mean()
    o["p_throws"] = df.groupby("pitcher").p_throws.first()
    o["name"] = df.groupby("pitcher").player_name.first()
    o["season"] = y
    outs_all.append(o.reset_index())
    # our scores for season y from the forward models (trained on seasons before y)
    if 2022 <= y <= 2025:
        c = pd.read_parquet(f"{D}/results/preds/crossfit_{y}.parquet")
        m = df[KEY + ["pitcher"]].merge(c, on=KEY, how="inner")
        for col, lab in (("crossfit", "ours_nr"), ("remember", "ours_rem")):
            x = m[col].to_numpy(); sp = 100 - 10 * (x - np.nanmean(x)) / np.nanstd(x)
            s = pd.DataFrame({"pitcher": m.pitcher, "s": sp}).dropna().groupby("pitcher").s.mean().rename(lab)
            scores.append(s.reset_index().assign(season=y).melt(["pitcher", "season"], var_name="system", value_name="score"))
    print(y, len(df), flush=True)
for f, lab in (("pitchers_2026_no_remembering.csv", "ours_nr"), ("pitchers_2026.csv", "ours_rem")):
    lb = pd.read_csv(ROOT / "leaderboards" / f)
    scores.append(pd.DataFrame({"pitcher": lb.pitcher, "season": 2026, "system": lab, "score": lb.stuff_plus_raw}))
pd.concat(outs_all).to_parquet(f"{OUT}/outcomes.parquet", index=False)
pd.concat(scores).to_parquet(f"{OUT}/ours.parquet", index=False)
