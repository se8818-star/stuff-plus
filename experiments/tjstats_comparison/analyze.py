"""
Stability and predictiveness of our Stuff+ against TJStats' tjStuff+.

    python -m experiments.tjstats_comparison.fetch      # TJ's per-pitch data -> data/external/
    python -m experiments.tjstats_comparison.build      # outcomes and our scores -> results/tjstats/
    python -m experiments.tjstats_comparison.analyze            # all season pairs, 2000 draws
    python -m experiments.tjstats_comparison.analyze --clean    # 2024->25 and 2025->26 only, 1000 draws

Writes reports/tjstats_comparison.json (or _clean.json). The write-up is
reports/tjstats_comparison.md. Correlations are sign-flipped so higher is better;
CIs are a paired bootstrap of each system minus TJ, averaged over season pairs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--clean", action="store_true", help="only seasons TJ's grades cannot have been trained on")
ap.add_argument("--boot", type=int, help="bootstrap draws (default 2000, 1000 with --clean)")
args = ap.parse_args()
S = ROOT / "results" / "tjstats"
o = pd.read_parquet(S / "outcomes.parquet").set_index(["pitcher", "season"])
ours = pd.read_parquet(S / "ours.parquet").pivot_table(index=["pitcher", "season"], columns="system", values="score")
tj = pd.read_csv(ROOT / "data" / "external" / "tjstuff_pitcher_seasons.csv").rename(columns={"pitcher_id": "pitcher", "tj_stuff_plus": "tj"}).set_index(["pitcher", "season"])["tj"]
W = ours.join(tj, how="outer").join(o)
SYS = ["ours_nr", "ours_rem", "tj"]
OUTC = {"rv100": -1, "woba": -1, "xwoba": -1, "fip": -1, "k_bb": 1}
rng = np.random.default_rng(0)
B = args.boot or (1000 if args.clean else 2000)
PAIRS, STICK = ([2024, 2025], [2024]) if args.clean else ([2022, 2023, 2024, 2025], [2022, 2023, 2024])

def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))

def frames(kind, pairs, mn):
    out = []
    for y in pairs:
        a = W.xs(y, level="season")
        if kind == "same":
            f = a[(a.pitches >= mn)].dropna(subset=SYS + list(OUTC))
            f = pd.DataFrame({**{s: f[s] for s in SYS}, **{k: f[k] for k in OUTC}})
        else:
            b = W.xs(y + 1, level="season")
            j = a.join(b, rsuffix="_n", how="inner")
            j = j[(j.pitches >= mn) & (j.pitches_n >= mn)]
            cols = SYS + (list(OUTC) if kind == "next" else [])
            need = SYS + ([k + "_n" for k in OUTC] if kind == "next" else [s + "_n" for s in SYS])
            j = j.dropna(subset=need)
            f = pd.DataFrame({**{s: j[s] for s in SYS},
                              **({k: j[k + "_n"] for k in OUTC} if kind == "next" else {s + "_n": j[s + "_n"] for s in SYS})})
        out.append(f)
    return out

def metric_vec(f, kind):
    X = f[SYS].to_numpy()
    if kind == "stick":
        return {s: corr(X[:, i], f[s + "_n"].to_numpy()) for i, s in enumerate(SYS)}
    return {(s, k): OUTC[k] * corr(X[:, i], f[k].to_numpy()) for i, s in enumerate(SYS) for k in OUTC}

def run(kind, pairs, mn):
    fs = frames(kind, pairs, mn)
    point = [metric_vec(f, kind) for f in fs]
    keys = list(point[0])
    mean = {k: float(np.mean([p[k] for p in point])) for k in keys}
    boots = {k: [] for k in keys}
    for _ in range(B):
        bp = [metric_vec(f.iloc[rng.integers(0, len(f), len(f))], kind) for f in fs]
        for k in keys:
            boots[k].append(np.mean([p[k] for p in bp]))
    return {"n": [len(f) for f in fs], "per_pair": [{str(k): float(v) for k, v in p.items()} for p in point],
            "mean": {str(k): v for k, v in mean.items()}, "boots": {str(k): np.array(v) for k, v in boots.items()}}

R = {}
for mn in (100, 300, 500):
    R[("same", mn)] = run("same", PAIRS, mn)
    R[("next", mn)] = run("next", PAIRS, mn)
    R[("stick", mn)] = run("stick", STICK, mn)

def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

res = {}
for (kind, mn), r in R.items():
    d = {"n": r["n"], "per_pair": r["per_pair"], "mean": r["mean"], "diff_vs_tj": {}}
    for k in r["mean"]:
        if "'tj'" in k or k == "tj":
            continue
        tk = k.replace("ours_nr", "tj").replace("ours_rem", "tj")
        dd = r["boots"][k] - r["boots"][tk]
        d["diff_vs_tj"][k] = {"diff": r["mean"][k] - r["mean"][tk], "ci": ci(dd)}
    res[f"{kind}_{mn}"] = d
# agreement
agr = []
for y in range(2021, 2027):
    a = W.xs(y, level="season")
    for mn in (100, 500):
        f = a[a.pitches >= mn].dropna(subset=["tj"])
        row = {"season": y, "min": mn, "n_tj": len(f), "sd_tj": f.tj.std(), "mean_tj": f.tj.mean()}
        for s in ("ours_nr", "ours_rem"):
            g = f.dropna(subset=[s])
            if len(g):
                row.update({f"n_{s}": len(g), f"r_{s}": g[s].corr(g.tj), f"rho_{s}": g[s].corr(g.tj, method="spearman"),
                            f"sd_{s}": g[s].std(), f"mean_{s}": g[s].mean()})
        agr.append(row)
res["agreement"] = agr
out = ROOT / "reports" / ("tjstats_comparison_clean.json" if args.clean else "tjstats_comparison.json")
out.write_text(json.dumps(res, indent=1, default=float))
print(f"wrote {out.relative_to(ROOT)}")
pd.set_option("display.width", 250)
print(pd.DataFrame(agr).round(3).to_string())
for key in ("same", "next", "stick"):
    for mn in (100, 300, 500):
        d = res[f"{key}_{mn}"]
        print(f"\n== {key} min{mn} n={d['n']}")
        for k, v in d["mean"].items():
            extra = d["diff_vs_tj"].get(k)
            print(f"  {k:28s} {v:.3f}" + (f"   vs TJ {extra['diff']:+.3f} [{extra['ci'][0]:+.3f}, {extra['ci'][1]:+.3f}]" if extra else ""))
