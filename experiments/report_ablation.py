"""
Turn results/ablation_folds12.json into reports/ablation.md: per-fold metrics,
mean-of-folds differences vs the baseline with paired-bootstrap 95% CIs, the
pre-set selection rule, per-pitch-type stickiness, and the grid PCA.

    python -m experiments.report_ablation [results/ablation_folds12.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["stickiness", "next_woba", "next_k_bb", "next_rv100"]
HEAD = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%",
        "next_rv100": "Next RV/100"}


def rule(cmp):
    """Pre-set rule: mean-of-folds next wOBA and next RV/100 both improve over
    the baseline with CIs excluding 0, and stickiness drops by no more than 0.01."""
    m = cmp["mean"]
    gains = all(m["diff"][k] > 0 and m["ci"][k][0] > 0 for k in ("next_woba", "next_rv100"))
    sticky = m["diff"]["stickiness"] >= -0.01
    return gains and sticky, gains, sticky


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ablation_folds12.json"
    R = json.loads(path.read_text())
    folds, base, res, cmp = list(R["results"]), R["baseline"], R["results"], R["vs_baseline"]
    names = [n for n in R["variants"] if all(n in res[f] for f in folds)]
    md = ["# Ablation results", "",
          f"Baseline: `{base}` (tjStuff+ v3.0 feature set, release-point x0/z0). "
          f"{len(R['seeds'])} model seeds averaged, minimum {R['args']['min_pitches']} pitches in "
          f"both seasons, {R['args']['boot']} bootstrap replicates. Outcome correlations are "
          "sign-flipped so higher is always better. No hyperparameters were tuned.", ""]
    for f in folds:
        fd = R["folds"][f]
        md += [f"## Fold {f}: train {fd['train'][0]}–{fd['train'][-1]}, score {fd['eval']}, "
               f"validate on {fd['eval'] + 1}", "",
               table(["Variant", "Features", "Pitchers"] + [HEAD[k] for k in KEYS],
                     [[f"`{n}`", str(len(R["variants"][n])), str(res[f][n]["n_pitchers"])]
                      + [f"{res[f][n][k]:.3f}" for k in KEYS] for n in names]), ""]
    md += ["## Mean of folds: variant minus baseline, with 95% CI", "",
           "Rule: next wOBA and next RV/100 must both improve with CIs excluding 0, "
           "and stickiness must not fall by more than 0.01.", ""]
    rows = []
    for n in names:
        if n == base:
            continue
        ok, gains, sticky = rule(cmp[n])
        m = cmp[n]["mean"]
        rows.append([f"`{n}`"] + [f"{m['diff'][k]:+.3f} [{m['ci'][k][0]:+.3f}, {m['ci'][k][1]:+.3f}]"
                                  for k in KEYS] + ["**yes**" if ok else "no"])
    md += [table(["Variant"] + [HEAD[k] for k in KEYS] + ["Passes rule"], rows), ""]
    for f in folds:
        rows = [[f"`{n}`"] + [f"{cmp[n]['folds'][f]['diff'][k]:+.3f} "
                              f"[{cmp[n]['folds'][f]['ci'][k][0]:+.3f}, {cmp[n]['folds'][f]['ci'][k][1]:+.3f}]"
                              for k in KEYS] for n in names if n != base]
        md += [f"### Fold {f} only", "", table(["Variant"] + [HEAD[k] for k in KEYS], rows), ""]

    pts = list(res[folds[0]][base]["pitch_type_stickiness"])
    md += ["## Stickiness by pitch type", "",
           f"Pitcher × pitch type, minimum {R['args']['min_pitches']} pitches of that type in both "
           "seasons; mean of the folds. The last row is the number of pitcher-pitch pairs per fold.", ""]
    rows = []
    for n in names:
        rows.append([f"`{n}`"] + [
            f"{sum(res[f][n]['pitch_type_stickiness'][p]['r'] for f in folds) / len(folds):.3f}"
            for p in pts])
    rows.append(["pairs (" + " / ".join(f"fold {f}" for f in folds) + ")"] + [
        " / ".join(str(res[f][base]["pitch_type_stickiness"][p]["n"]) for f in folds) for p in pts])
    md += [table(["Variant"] + pts, rows), ""]

    p, cum = R["grid_pca"], R["grid_pca_cumvar"]
    md += ["## PCA of the 18 grid angles", "",
           table(["Component"] + [f"PC{i + 1}" for i in range(5)],
                 [["Cumulative variance"] + [f"{c:.6f}" for c in cum[:5]]]), "",
           f"- Sample: {p['n']:,} pitches. Individual grid angles have an sd of "
           f"{p['grid_angle_sd_deg'][0]:.2f}–{p['grid_angle_sd_deg'][1]:.2f}°.",
           f"- After a linear fit on just the two centre-of-zone angles, the largest residual sd of "
           f"any of the 18 angles is {p['max_resid_sd_deg_after_center_pair']:.4f}° "
           f"(minimum R² {p['min_r2_after_center_pair']:.6f})."]
    if "max_resid_sd_deg_after_center_pair_plus_extension" in p:
        md.append(f"- Adding extension to that fit leaves "
                  f"{p['max_resid_sd_deg_after_center_pair_plus_extension']:.4f}°.")
    md += ["", "Library versions: " + ", ".join(f"{k} {v}" for k, v in R["env"].items() if v), ""]
    out = ROOT / "reports" / "ablation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
