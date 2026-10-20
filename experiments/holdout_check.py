"""
Exploratory (run after the fold 2 confirmation, so it cannot change that
verdict): train on 2022-24 and look at both held-out seasons, 2025 and 2026,
splitting pitchers by whether the model saw them in training.

    python -m experiments.holdout_check     # results/holdout_2025_2026.json, reports/holdout_2025_2026.md
"""
import json
from pathlib import Path

import numpy as np

from stuffplus import run_ablation as ra
from pipeline.develop import pitcher_table, stuff_plus, table
from stuffplus.features import load
from stuffplus.models import AMENDED, CANDIDATES, add_outcome_flags, fit_predict
from experiments.overfit_check import r2

ROOT = Path(__file__).resolve().parent.parent
TRAIN = [2022, 2023, 2024]


def corr(j, a, b, flip=True):
    return float((-1 if flip else 1) * j[a].corr(j[b])) if len(j) > 2 else float("nan")


def main():
    df = add_outcome_flags(load(ROOT / "data", TRAIN + [2025, 2026]))
    df = df[df["target"].notna()].reset_index(drop=True)
    tr, y25, y26 = df[df["year"].isin(TRAIN)], df[df["year"] == 2025], df[df["year"] == 2026]
    seen = set(tr["pitcher"])
    o25, o26 = ra.outcomes(y25), ra.outcomes(y26)
    out = {}
    for name in ("A", "S6"):
        (p25, p26), _ = fit_predict({**CANDIDATES, **AMENDED}[name], tr, [y25, y26])
        s25, s26 = stuff_plus(p25, p25), stuff_plus(p26, p25)
        a, b = pitcher_table(y25, s25), pitcher_table(y26, s26)
        res = {"pitch_r2": {"2025": r2(y25["target"].to_numpy(), p25), "2026": r2(y26["target"].to_numpy(), p26)}}
        j = a.join(b, lsuffix="_y1", rsuffix="_y2", how="outer")
        j = j.join(o25.add_suffix("_25")).join(o26.add_suffix("_26"))
        j["known"] = j.index.isin(seen)
        for label, g in (("all", j), ("known", j[j["known"]]), ("new", j[~j["known"]])):
            q25, q26 = g[g["size_y1"] >= 100], g[g["size_y2"] >= 100]
            both = g[(g["size_y1"] >= 100) & (g["size_y2"] >= 100)]
            res[label] = {
                "same_2025": {"n": len(q25), "woba": corr(q25, "mean_y1", "woba_25"), "rv100": corr(q25, "mean_y1", "rv100_25")},
                "same_2026": {"n": len(q26), "woba": corr(q26, "mean_y2", "woba_26"), "rv100": corr(q26, "mean_y2", "rv100_26")},
                "next_2025_to_2026": {"n": len(both), "woba": corr(both, "mean_y1", "woba_26"),
                                      "rv100": corr(both, "mean_y1", "rv100_26"),
                                      "k_bb": corr(both, "mean_y1", "k_bb_26", flip=False),
                                      "stickiness": corr(both, "mean_y1", "mean_y2", flip=False)}}
        out[name] = res
        print(name, json.dumps(res, indent=1), flush=True)
    out["env"] = ra.env_info()
    (ROOT / "results" / "holdout_2025_2026.json").write_text(json.dumps(out, indent=1))
    rows = []
    for n in ("A", "S6"):
        for g, lab in (("all", "all pitchers"), ("known", "in 2022–24 training"), ("new", "debuted after 2024")):
            r = out[n][g]
            rows.append([n, lab, str(r["same_2025"]["n"]), f"{r['same_2025']['rv100']:.3f}",
                         str(r["same_2026"]["n"]), f"{r['same_2026']['rv100']:.3f}",
                         str(r["next_2025_to_2026"]["n"]), f"{r['next_2025_to_2026']['woba']:.3f}",
                         f"{r['next_2025_to_2026']['rv100']:.3f}", f"{r['next_2025_to_2026']['stickiness']:.3f}"])
    md = ["# Trained on 2022–24, both later seasons held out", "",
          "Exploratory, run after the fold 2 confirmation. Minimum 100 pitches; correlations of Stuff+ with "
          "results are sign-flipped so higher is better. The all-pitcher 2025 → 2026 row repeats the "
          "confirmation run.", "",
          table(["Model", "Pitchers", "n", "2025 Stuff+ vs 2025 RV/100", "n", "2026 Stuff+ vs 2026 RV/100", "n",
                 "2025 Stuff+ vs 2026 wOBA", "2025 Stuff+ vs 2026 RV/100", "Stickiness 2025 → 2026"], rows), "",
          table(["Model", "Pitch R² 2025", "Pitch R² 2026"],
                [[n, f"{out[n]['pitch_r2']['2025']:.5f}", f"{out[n]['pitch_r2']['2026']:.5f}"] for n in ("A", "S6")]), ""]
    (ROOT / "reports" / "holdout_2025_2026.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
