"""
Development runs for the pre-registered candidates (docs/PREREGISTRATION.md):
inner GroupKFold-by-pitcher CV on 2022-23 and fold 1 (train 2022-23, score
2024, validate on 2025). Never loads 2026.

    python -m pipeline.develop                  # fit everything missing, then analyse
    python -m pipeline.develop --tag tuned --params results/tuned_A.json --only A
    python -m pipeline.develop --analyse-only

Pitch-level xRV is cached in results/preds/ so the analysis can be redone
without refitting. Writes results/development_<tag>.json and
reports/development_<tag>.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from stuffplus import run_ablation as ra
from stuffplus.features import load
from stuffplus.models import AMENDED, CANDIDATES, CONTROLS, add_outcome_flags, fit_predict

ROOT = Path(__file__).resolve().parent.parent
TRAIN_YEARS, EVAL_YEAR = [2022, 2023], 2024
KEY = ["game_pk", "at_bat_number", "pitch_number"]
MIN_PITCHES = [100, 300, 500]
N_INNER, N_BOOT = 5, 2000
SCORE_KEYS = ["next_woba", "next_rv100"]


def load_dev():
    df = load(ROOT / "data", TRAIN_YEARS + [EVAL_YEAR, EVAL_YEAR + 1])
    assert df["year"].max() <= EVAL_YEAR + 1, "development must not touch the locked season"
    df = add_outcome_flags(df)
    return df[df["target"].notna()].reset_index(drop=True)


def inner_folds(tr):
    return list(GroupKFold(n_splits=N_INNER).split(tr, groups=tr["pitcher"]))


def fit_candidate(name, spec, df, params, out):
    """Inner-CV out-of-fold xRV for 2022-23 and fold 1 xRV for 2024-25."""
    t0 = time.time()
    tr = df[df["year"].isin(TRAIN_YEARS)]
    oof = np.full(len(tr), np.nan)
    for k, (i_fit, i_hold) in enumerate(inner_folds(tr)):
        (p,), _ = fit_predict(spec, tr.iloc[i_fit], [tr.iloc[i_hold]], params)
        oof[i_hold] = p
        print(f"  {name}: inner fold {k + 1}/{N_INNER} done ({time.time() - t0:.0f}s)", flush=True)
    ev = df[df["year"].isin([EVAL_YEAR, EVAL_YEAR + 1])]
    (p_ev,), info = fit_predict(spec, tr, [ev], params)
    preds = pd.concat([tr[KEY].assign(split="inner", xrv=oof), ev[KEY].assign(split="fold1", xrv=p_ev)])
    preds.to_parquet(out, index=False)
    info["seconds"] = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float))
    print(f"  {name}: saved ({info['seconds']:.0f}s)", flush=True)


# ------------------------------------------------------------------ analysis
def stuff_plus(xrv, ref):
    """100 - 10 z, with z from the reference (scored) season's pitches."""
    return 100 - 10 * (xrv - np.nanmean(ref)) / np.nanstd(ref)


def pitcher_table(df, s, by=("pitcher",)):
    d = pd.DataFrame({**{c: df[c].to_numpy() for c in by}, "s": s}).dropna(subset=["s"])
    return d.groupby(list(by))["s"].agg(["mean", "size"])


def shrink(t, k, level=None):
    """Empirical-Bayes shrinkage of group means toward the league mean by pitch
    count: mu + n/(n+k) (mean - mu). `level` names an index level (pitch_type)
    whose own league mean is the target; otherwise the overall mean."""
    t = t.copy()
    w = t["mean"] * t["size"]
    if level:
        mu = w.groupby(level=level).sum() / t["size"].groupby(level=level).sum()
        mu = mu.reindex(t.index.get_level_values(level)).to_numpy()
    else:
        mu = w.sum() / t["size"].sum()
    t["mean"] = mu + t["size"] / (t["size"] + k) * (t["mean"] - mu)
    return t


def split_half_k(df, s, by, min_n):
    """Shrinkage constant from split-half reliability: each group (pitcher-season,
    or pitcher-season-pitch type) is split into alternating games by date;
    r = correlation of the half means, m = harmonic mean half size,
    k = m (1 - r) / r."""
    d = pd.DataFrame({**{c: df[c].to_numpy() for c in by}, "game_pk": df["game_pk"].to_numpy(),
                      "game_date": df["game_date"].to_numpy(), "s": s}).dropna(subset=["s"])
    season = [c for c in by if c != "pitch_type"]
    games = d[season + ["game_pk", "game_date"]].drop_duplicates().sort_values(
        season + ["game_date", "game_pk"])
    games["half"] = games.groupby(season).cumcount() % 2
    d = d.merge(games[season + ["game_pk", "half"]], on=season + ["game_pk"])
    g = d.groupby(list(by) + ["half"])["s"].agg(["mean", "size"]).unstack("half")
    g = g[(g["size"].sum(axis=1) >= min_n) & g["size"].notna().all(axis=1) & (g["size"].min(axis=1) > 0)]
    r = g["mean"][0].corr(g["mean"][1])
    half = g["size"].mean(axis=1)
    m = len(half) / (1.0 / half).sum()
    return {"k": float(m * (1 - r) / r), "r_half": float(r), "harmonic_half_size": float(m),
            "groups": int(len(g))}


def split_tables(df, preds, split):
    """Pitcher tables for the scored season and the next one, plus next-season
    outcomes, for the inner CV (2022 -> 2023) or fold 1 (2024 -> 2025)."""
    y1 = TRAIN_YEARS[0] if split == "inner" else EVAL_YEAR
    p = preds[preds["split"] == split]
    d = df[df["year"].isin([y1, y1 + 1])].merge(p[KEY + ["xrv"]], on=KEY, how="left", validate="1:1")
    first, nxt = d[d["year"] == y1], d[d["year"] == y1 + 1]
    ref = first["xrv"].to_numpy()
    s1, s2 = stuff_plus(ref, ref), stuff_plus(nxt["xrv"].to_numpy(), ref)
    return first, nxt, s1, s2


def analyse(df, names, tag):
    tr = df[df["year"].isin(TRAIN_YEARS)]
    res, tables, ks = {}, {}, {}
    out_next = {"inner": ra.outcomes(df[df["year"] == TRAIN_YEARS[0] + 1]),
                "fold1": ra.outcomes(df[df["year"] == EVAL_YEAR + 1])}
    for name in names:
        preds = pd.read_parquet(ROOT / "results" / "preds" / f"{tag}_{name}.parquet")
        # shrinkage constants from training-year out-of-fold Stuff+ only
        oof = tr.merge(preds[preds["split"] == "inner"][KEY + ["xrv"]], on=KEY, how="left", validate="1:1")
        s_oof = stuff_plus(oof["xrv"].to_numpy(), oof["xrv"].to_numpy())
        ks[name] = {"pitcher": split_half_k(oof, s_oof, ["pitcher", "year"], 100),
                    "pitch_type": split_half_k(oof, s_oof, ["pitcher", "year", "pitch_type"], 50)}
        k, k_pt = ks[name]["pitcher"]["k"], ks[name]["pitch_type"]["k"]
        res[name], tables[name] = {}, {}
        for split in ("inner", "fold1"):
            first, nxt, s1, s2 = split_tables(df, preds, split)
            a, b = pitcher_table(first, s1), pitcher_table(nxt, s2)
            y, px = nxt["target"].to_numpy(), nxt["xrv"].to_numpy()
            ok = np.isfinite(px)
            r = {"pitch_r2_next": float(1 - ((y[ok] - px[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()),
                 "coverage": float(ok.mean())}
            for shr, (aa, bb) in {"raw": (a, b), "shrunk": (shrink(a, k), shrink(b, k))}.items():
                for mn in MIN_PITCHES:
                    j = ra.corr_table(aa, bb, out_next[split], mn)
                    tables[name][(split, shr, mn)] = j
                    r[f"{shr}_{mn}"] = {**ra.metrics(j), "n": len(j)}
            if split == "fold1":
                pa = pitcher_table(first, s1, ("pitcher", "pitch_type"))
                pb = pitcher_table(nxt, s2, ("pitcher", "pitch_type"))
                r["pitch_type_stickiness"] = {}
                for shr, (aa, bb) in {"raw": (pa, pb), "shrunk": (shrink(pa, k_pt, "pitch_type"),
                                                                   shrink(pb, k_pt, "pitch_type"))}.items():
                    j = aa.join(bb, lsuffix="_y1", rsuffix="_y2", how="inner")
                    j = j[(j["size_y1"] >= 100) & (j["size_y2"] >= 100)].reset_index()
                    r["pitch_type_stickiness"][shr] = {
                        pt: {"r": float(g["mean_y1"].corr(g["mean_y2"])), "n": len(g)}
                        for pt, g in j[j["pitch_type"].isin(ra.PITCH_TYPES)].groupby("pitch_type")}
            res[name][split] = r
        res[name]["selection_score"] = float(np.mean(
            [res[name][sp]["raw_100"][m] for sp in ("inner", "fold1") for m in SCORE_KEYS]))
        res[name]["selection_score_shrunk"] = float(np.mean(
            [res[name][sp]["shrunk_100"][m] for sp in ("inner", "fold1") for m in SCORE_KEYS]))

    # paired bootstrap against A
    vs = {}
    if "A" in names:
        for name in names:
            if name == "A":
                continue
            vs[name] = {}
            for key in tables[name]:
                bs = ra.bootstrap_folds([tables["A"][key]], [tables[name][key]], n=N_BOOT)
                vs[name]["{}_{}_{}".format(*key)] = {"diff": bs["diff"][0], "ci": bs["ci"][0]}
            # selection score difference, bootstrapped jointly over both splits
            bs = ra.bootstrap_folds([tables["A"][(sp, "raw", 100)] for sp in ("inner", "fold1")],
                                    [tables[name][(sp, "raw", 100)] for sp in ("inner", "fold1")], n=N_BOOT)
            vs[name]["selection_score"] = {
                "diff": float(np.mean([bs["mean_diff"][m] for m in SCORE_KEYS])),
                "per_metric_mean_of_splits": {m: {"diff": bs["mean_diff"][m], "ci": bs["mean_ci"][m]}
                                              for m in ra.KEYS}}
    return res, vs, ks


# -------------------------------------------------------------------- report
def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_report(R, tag):
    res, vs, ks, names = R["results"], R["vs_A"], R["shrinkage"], R["names"]
    H = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%", "next_rv100": "Next RV/100"}
    md = [f"# Development results ({tag})", "",
          "Inner CV: 5-fold GroupKFold by pitcher on 2022–23 (2022 Stuff+ vs the same pitchers' 2023). "
          "Fold 1: train 2022–23, score 2024, validate on 2025. Fold 2 is not touched. "
          "3 seeds averaged; correlations sign-flipped so higher is better; "
          f"95% CIs are paired bootstraps over pitchers against A ({N_BOOT} replicates).", "",
          "## Selection score", "",
          "Mean of next wOBA and next RV/100 correlations over inner CV and fold 1, unshrunk, "
          "minimum 100 pitches.", ""]
    order = sorted(names, key=lambda n: -res[n]["selection_score"])
    rows = []
    for n in order:
        d = vs.get(n, {}).get("selection_score")
        rows.append([n, f"{res[n]['selection_score']:.4f}", f"{d['diff']:+.4f}" if d else "–",
                     f"{res[n]['selection_score_shrunk']:.4f}"]
                    + [f"{res[n][sp]['raw_100'][m]:.3f}" for sp in ("inner", "fold1") for m in SCORE_KEYS])
    md += [table(["Model", "Score", "vs A", "Score (shrunk)", "Inner wOBA", "Inner RV/100",
                  "Fold 1 wOBA", "Fold 1 RV/100"], rows), ""]
    for split, title in (("inner", "Inner CV (2022 → 2023)"), ("fold1", "Fold 1 (2024 → 2025)")):
        for shr in ("raw", "shrunk"):
            md += [f"## {title}, {'with' if shr == 'shrunk' else 'without'} shrinkage", ""]
            for mn in MIN_PITCHES:
                rows = []
                for n in names:
                    m = res[n][split][f"{shr}_{mn}"]
                    cells = []
                    for k in ra.KEYS:
                        c = f"{m[k]:.3f}"
                        if n in vs:
                            v = vs[n][f"{split}_{shr}_{mn}"]
                            c += f" ({v['diff'][k]:+.3f} [{v['ci'][k][0]:+.3f}, {v['ci'][k][1]:+.3f}])"
                        cells.append(c)
                    rows.append([n, str(m["n"])] + cells)
                md += [f"Minimum {mn} pitches:", "", table(["Model", "Pitchers"] + [H[k] for k in ra.KEYS], rows), ""]
    md += ["## Shrinkage constants (training years, split-half by alternating games)", "",
           table(["Model", "k pitcher", "split-half r", "half size", "k pitch type", "split-half r", "half size"],
                 [[n, f"{ks[n]['pitcher']['k']:.1f}", f"{ks[n]['pitcher']['r_half']:.4f}",
                   f"{ks[n]['pitcher']['harmonic_half_size']:.0f}", f"{ks[n]['pitch_type']['k']:.1f}",
                   f"{ks[n]['pitch_type']['r_half']:.4f}", f"{ks[n]['pitch_type']['harmonic_half_size']:.0f}"]
                  for n in names]), "",
           "## Fold 1 stickiness by pitch type (min 100 of the type in both seasons)", ""]
    for shr in ("raw", "shrunk"):
        rows = [[n] + [f"{res[n]['fold1']['pitch_type_stickiness'][shr].get(pt, {}).get('r', float('nan')):.3f}"
                       for pt in ra.PITCH_TYPES] for n in names]
        rows.append(["pairs"] + [str(res[names[0]]["fold1"]["pitch_type_stickiness"][shr].get(pt, {}).get("n", 0))
                                 for pt in ra.PITCH_TYPES])
        md += [f"{'With' if shr == 'shrunk' else 'Without'} shrinkage:", "",
               table(["Model"] + ra.PITCH_TYPES, rows), ""]
    md += ["## Other", "",
           table(["Model", "Inner pitch R²", "Fold 1 pitch R²", "Fold 1 rows scored", "Fit time (s)"],
                 [[n, f"{res[n]['inner']['pitch_r2_next']:.5f}", f"{res[n]['fold1']['pitch_r2_next']:.5f}",
                   f"{res[n]['fold1']['coverage']:.4f}", f"{R['fit_info'][n].get('seconds', 0):.0f}"]
                  for n in names]), "",
           "Library versions: " + ", ".join(f"{k} {v}" for k, v in R["env"].items() if v), ""]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"development_{tag}.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--only", nargs="*", help="subset of model names")
    ap.add_argument("--params", help="JSON file of LightGBM overrides (tuned settings)")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    specs = {**CANDIDATES, **AMENDED, **CONTROLS}
    names = args.only or list(specs)
    params = json.loads(Path(args.params).read_text())["params"] if args.params else None
    pdir = ROOT / "results" / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_dev()
    print(f"loaded {len(df):,} pitches with a target, seasons {sorted(df['year'].unique())} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not args.analyse_only:
        for n in names:
            out = pdir / f"{args.tag}_{n}.parquet"
            if out.exists():
                print(f"{n}: cached")
                continue
            print(f"{n}: fitting", flush=True)
            fit_candidate(n, specs[n], df, params, out)
    res, vs, ks = analyse(df, names, args.tag)
    R = {"tag": args.tag, "names": names, "params": params, "env": ra.env_info(),
         "specs": {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in specs[n].items()} for n in names},
         "fit_info": {n: json.loads((pdir / f"{args.tag}_{n}.json").read_text()) for n in names},
         "results": res, "vs_A": vs, "shrinkage": ks}
    (ROOT / "results" / f"development_{args.tag}.json").write_text(json.dumps(R, indent=1, default=float))
    write_report(R, args.tag)
    for n in sorted(names, key=lambda n: -res[n]["selection_score"]):
        print(f"{n:<12} selection score {res[n]['selection_score']:.4f}")
    print(f"wrote reports/development_{args.tag}.md ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
