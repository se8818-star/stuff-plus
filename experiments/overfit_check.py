"""
How much does the model memorise the pitchers it was trained on?

Compares three kinds of prediction from the same recipe (train 2022-23):
  in-sample        the pitches the model was trained on
  out-of-pitcher   the same pitches, predicted by a model that never saw that
                   pitcher (5-fold GroupKFold, cached by pipeline/develop.py)
  next seasons     2024 and 2025 pitches

    python -m experiments.overfit_check     # writes results/overfit_check.json, reports/overfit_check.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stuffplus import run_ablation as ra
from pipeline.develop import KEY, TRAIN_YEARS, load_dev, pitcher_table, stuff_plus, table
from stuffplus.models import AMENDED, CANDIDATES, fit_predict

ROOT = Path(__file__).resolve().parent.parent


def r2(y, p):
    ok = np.isfinite(p)
    return float(1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def pitcher_corrs(first, s, out_same, out_next, min_n=100):
    t = pitcher_table(first, s)
    t = t[t["size"] >= min_n]
    res = {}
    for label, o in (("same", out_same), ("next", out_next)):
        j = t.join(o, how="inner").dropna(subset=["woba", "rv100"])
        j = j[j["pa"] >= 50]
        res[label] = {"woba": float(-j["mean"].corr(j["woba"])), "rv100": float(-j["mean"].corr(j["rv100"])),
                      "n": len(j)}
    return res


def main():
    df = load_dev()
    tr = df[df["year"].isin(TRAIN_YEARS)]
    out = {}
    specs = {**CANDIDATES, **AMENDED}
    for name in ("A", "S6"):
        (p_in,), _ = fit_predict(specs[name], tr, [tr])
        cache = pd.read_parquet(ROOT / "results" / "preds" / f"stage1_{name}.parquet")
        d = df.merge(cache[KEY + ["xrv"]], on=KEY, how="left", validate="1:1")
        oof = d.loc[d["year"].isin(TRAIN_YEARS), "xrv"].to_numpy()
        y = tr["target"].to_numpy()
        res = {"pitch_r2": {"in_sample": r2(y, p_in), "out_of_pitcher": r2(y, oof)}}
        for yr in (2024, 2025):
            m = (d["year"] == yr).to_numpy()
            res["pitch_r2"][f"next_{yr}"] = r2(d.loc[m, "target"].to_numpy(), d.loc[m, "xrv"].to_numpy())
        # pitcher level, 2023 pitchers: same-season and next-season results
        m23 = (tr["year"] == 2023).to_numpy()
        first = tr[m23]
        same, nxt = ra.outcomes(first), ra.outcomes(df[df["year"] == 2024])
        s_in, s_oof = stuff_plus(p_in[m23], p_in[m23]), stuff_plus(oof[m23], oof[m23])
        res["pitcher_2023"] = {"in_sample": pitcher_corrs(first, s_in, same, nxt),
                               "out_of_pitcher": pitcher_corrs(first, s_oof, same, nxt)}
        a, b = pitcher_table(first, s_in), pitcher_table(first, s_oof)
        j = a.join(b, lsuffix="_in", rsuffix="_oof")
        j = j[j["size_in"] >= 100]
        res["in_vs_oof_pitcher_r"] = float(j["mean_in"].corr(j["mean_oof"]))
        res["pitcher_sd"] = {"in_sample": float(j["mean_in"].std()), "out_of_pitcher": float(j["mean_oof"].std())}
        out[name] = res
        print(name, json.dumps(res, indent=1))
    out["env"] = ra.env_info()
    (ROOT / "results" / "overfit_check.json").write_text(json.dumps(out, indent=1))
    rows = []
    for n in ("A", "S6"):
        p, q = out[n]["pitch_r2"], out[n]["pitcher_2023"]
        rows.append([n, f"{p['in_sample']:.5f}", f"{p['out_of_pitcher']:.5f}", f"{p['next_2024']:.5f}",
                     f"{p['next_2025']:.5f}",
                     f"{q['in_sample']['same']['rv100']:.3f}", f"{q['out_of_pitcher']['same']['rv100']:.3f}",
                     f"{q['in_sample']['next']['rv100']:.3f}", f"{q['out_of_pitcher']['next']['rv100']:.3f}",
                     f"{out[n]['in_vs_oof_pitcher_r']:.3f}"])
    md = ["# Overfitting check", "",
          "Models trained on 2022–23. *In-sample* scores the training pitches; *out-of-pitcher* scores the same "
          "pitches with a model that never saw that pitcher; 2024 and 2025 are later seasons. Pitcher-level "
          "columns use 2023 pitchers with at least 100 pitches and are correlations of Stuff+ with RV/100 "
          "(sign-flipped), in the same season and in 2024.", "",
          table(["Model", "Pitch R² in-sample", "Pitch R² out-of-pitcher", "Pitch R² 2024", "Pitch R² 2025",
                 "Same-season RV/100, in-sample", "Same-season RV/100, out-of-pitcher",
                 "Next-season RV/100, in-sample", "Next-season RV/100, out-of-pitcher",
                 "r(in-sample, out-of-pitcher)"], rows), ""]
    (ROOT / "reports" / "overfit_check.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
