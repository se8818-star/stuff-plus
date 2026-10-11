"""
One-shot confirmation on the locked fold 2 (PREREGISTRATION.md):
train 2022-24, score 2025, validate against 2026, for frozen A and the frozen
winner only.

    python confirm.py --unlock-fold2

Refuses to run unless FROZEN.json exists and is committed, and refuses to run a
second time (results/confirmation.json already present).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import run_ablation as ra
from develop import KEY, MIN_PITCHES, N_BOOT, N_INNER, pitcher_table, shrink, split_half_k, stuff_plus
from features import load
from models import CANDIDATES, add_outcome_flags, fit_predict

HERE = Path(__file__).parent
TRAIN_YEARS, EVAL_YEAR = [2022, 2023, 2024], 2025


def verdict(vs100):
    """The pre-set rule at minimum 100 pitches, unshrunk: next wOBA and next
    RV/100 both above A with 95% CIs excluding 0, stickiness within 0.01."""
    gains = {m: bool(vs100["diff"][m] > 0 and vs100["ci"][m][0] > 0) for m in ("next_woba", "next_rv100")}
    sticky = bool(vs100["diff"]["stickiness"] >= -0.01)
    return {"confirmed": all(gains.values()) and sticky, "gains": gains, "stickiness_ok": sticky}


def run_confirmation(df, frozen, train_years, eval_year, n_boot=N_BOOT, log=print):
    """Fit frozen A and the frozen winner on `train_years`, score `eval_year`
    and the next season. Returns metrics (raw and shrunk, at each minimum),
    paired bootstrap against A, and the rule's verdict."""
    names = ["A"] + [n for n in [frozen["winner"]] if n != "A"]
    tr = df[df["year"].isin(train_years)]
    first, nxt = df[df["year"] == eval_year], df[df["year"] == eval_year + 1]
    out_next = ra.outcomes(nxt)
    res, tables, ks = {}, {}, {}
    for name in names:
        spec, params = CANDIDATES[name], frozen["params"].get(name)
        t0 = time.time()
        (p1, p2), info = fit_predict(spec, tr, [first, nxt], params)
        log(f"{name}: fitted ({time.time() - t0:.0f}s)")
        # shrinkage constants: training-year out-of-fold Stuff+ only
        oof = np.full(len(tr), np.nan)
        for i_fit, i_hold in GroupKFold(n_splits=N_INNER).split(tr, groups=tr["pitcher"]):
            (p,), _ = fit_predict(spec, tr.iloc[i_fit], [tr.iloc[i_hold]], params)
            oof[i_hold] = p
        s_oof = stuff_plus(oof, oof)
        ks[name] = {"pitcher": split_half_k(tr, s_oof, ["pitcher", "year"], 100),
                    "pitch_type": split_half_k(tr, s_oof, ["pitcher", "year", "pitch_type"], 50)}
        log(f"{name}: shrinkage constants done ({time.time() - t0:.0f}s)")
        s1, s2 = stuff_plus(p1, p1), stuff_plus(p2, p1)
        a, b = pitcher_table(first, s1), pitcher_table(nxt, s2)
        k, k_pt = ks[name]["pitcher"]["k"], ks[name]["pitch_type"]["k"]
        r = {"xrv_mean": float(np.nanmean(p1)), "xrv_sd": float(np.nanstd(p1)), **info}
        for shr, (aa, bb) in {"raw": (a, b), "shrunk": (shrink(a, k), shrink(b, k))}.items():
            for mn in MIN_PITCHES:
                j = ra.corr_table(aa, bb, out_next, mn)
                tables[(name, shr, mn)] = j
                r[f"{shr}_{mn}"] = {**ra.metrics(j), "n": len(j)}
        pa, pb = (pitcher_table(d, s, ("pitcher", "pitch_type")) for d, s in ((first, s1), (nxt, s2)))
        r["pitch_type_stickiness"] = {}
        for shr, (aa, bb) in {"raw": (pa, pb), "shrunk": (shrink(pa, k_pt, "pitch_type"),
                                                           shrink(pb, k_pt, "pitch_type"))}.items():
            j = aa.join(bb, lsuffix="_y1", rsuffix="_y2", how="inner")
            j = j[(j["size_y1"] >= 100) & (j["size_y2"] >= 100)].reset_index()
            r["pitch_type_stickiness"][shr] = {
                pt: {"r": float(g["mean_y1"].corr(g["mean_y2"])), "n": len(g)}
                for pt, g in j[j["pitch_type"].isin(ra.PITCH_TYPES)].groupby("pitch_type")}
        res[name] = r
    out = {"train_years": train_years, "eval_year": eval_year, "frozen": frozen,
           "results": res, "shrinkage": ks}
    if len(names) == 2:
        w = names[1]
        out["vs_A"] = {}
        for shr in ("raw", "shrunk"):
            for mn in MIN_PITCHES:
                bs = ra.bootstrap_folds([tables[("A", shr, mn)]], [tables[(w, shr, mn)]], n=n_boot)
                out["vs_A"][f"{shr}_{mn}"] = {"diff": bs["diff"][0], "ci": bs["ci"][0]}
        out["verdict"] = verdict(out["vs_A"]["raw_100"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlock-fold2", action="store_true")
    args = ap.parse_args()
    frozen_path, out_path = HERE / "FROZEN.json", HERE / "results" / "confirmation.json"
    if not args.unlock_fold2:
        sys.exit("Fold 2 is the one-shot confirmation test; pass --unlock-fold2 to run it.")
    if not frozen_path.exists():
        sys.exit("FROZEN.json not found: freeze the candidate before touching fold 2.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "FROZEN.json", "FROZEN.md"],
                           cwd=HERE, capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "FROZEN.json"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    if dirty or not tracked:
        sys.exit("FROZEN.json / FROZEN.md must be committed before fold 2 is run.")
    if out_path.exists():
        sys.exit("results/confirmation.json exists: the confirmation has already been run once.")

    t0 = time.time()
    frozen = json.loads(frozen_path.read_text())
    df = add_outcome_flags(load(HERE / "data", TRAIN_YEARS + [EVAL_YEAR, EVAL_YEAR + 1]))
    df = df[df["target"].notna()].reset_index(drop=True)
    out = run_confirmation(df, frozen, TRAIN_YEARS, EVAL_YEAR, log=lambda m: print(m, flush=True))
    out["env"] = ra.env_info()
    out["frozen_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                                          text=True).stdout.strip()
    out["seconds"] = time.time() - t0
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("verdict", "vs_A") if k in out}, indent=1))
    print(f"wrote {out_path.name} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
