"""Turn results/confirmation.json (the one-shot fold 2 run) into reports/confirmation.md."""
import json
from pathlib import Path

import run_ablation as ra
from develop import MIN_PITCHES, table

HERE = Path(__file__).parent
H = {"stickiness": "Stickiness", "next_woba": "Next wOBA", "next_k_bb": "Next K−BB%", "next_rv100": "Next RV/100"}


def main():
    R = json.loads((HERE / "results" / "confirmation.json").read_text())
    res, vs, names = R["results"], R["vs_A"], list(R["results"])
    md = ["# Fold 2 confirmation", "",
          f"Train {R['train_years'][0]}–{R['train_years'][-1]}, score {R['eval_year']}, validate against "
          f"{R['eval_year'] + 1}. Run once, on models frozen at commit `{R['frozen_commit'][:7]}`. "
          "3 seeds averaged, TJ's LightGBM settings, no tuning. Correlations are sign-flipped so higher is "
          "better; differences from A carry paired-bootstrap 95% CIs over pitchers.", "",
          "## Verdict", "",
          "Rule (minimum 100 pitches, unshrunk): next wOBA and next RV/100 both above A with CIs "
          "excluding 0, and stickiness no more than 0.01 below A.", "",
          table(["Model", "wOBA gain clears 0", "RV/100 gain clears 0", "Stickiness within 0.01", "Confirmed"],
                [[n, *("yes" if x else "no" for x in (v["gains"]["next_woba"], v["gains"]["next_rv100"],
                                                      v["stickiness_ok"])),
                  "**yes**" if v["confirmed"] else "no"] for n, v in R["verdict"].items()]), "",
          f"**Ships: {R['ships']}**", ""]
    for shr in ("raw", "shrunk"):
        md += [f"## {'With' if shr == 'shrunk' else 'Without'} shrinkage", ""]
        for mn in MIN_PITCHES:
            rows = []
            for n in names:
                m = res[n][f"{shr}_{mn}"]
                cells = []
                for k in ra.KEYS:
                    c = f"{m[k]:.3f}"
                    if n in vs:
                        d = vs[n][f"{shr}_{mn}"]
                        c += f" ({d['diff'][k]:+.3f} [{d['ci'][k][0]:+.3f}, {d['ci'][k][1]:+.3f}])"
                    cells.append(c)
                rows.append([n, str(m["n"])] + cells)
            md += [f"Minimum {mn} pitches:", "", table(["Model", "Pitchers"] + [H[k] for k in ra.KEYS], rows), ""]
    ks = R["shrinkage"]
    md += ["## Shrinkage constants (2022–24 out-of-fold, split-half by alternating games)", "",
           table(["Model", "k pitcher", "split-half r", "k pitch type", "split-half r"],
                 [[n, f"{ks[n]['pitcher']['k']:.1f}", f"{ks[n]['pitcher']['r_half']:.4f}",
                   f"{ks[n]['pitch_type']['k']:.1f}", f"{ks[n]['pitch_type']['r_half']:.4f}"] for n in names]), "",
           "## Stickiness by pitch type (min 100 of the type in both seasons, unshrunk)", ""]
    rows = [[n] + [f"{res[n]['pitch_type_stickiness']['raw'].get(pt, {}).get('r', float('nan')):.3f}"
                   for pt in ra.PITCH_TYPES] for n in names]
    rows.append(["pairs"] + [str(res["A"]["pitch_type_stickiness"]["raw"].get(pt, {}).get("n", 0))
                             for pt in ra.PITCH_TYPES])
    md += [table(["Model"] + ra.PITCH_TYPES, rows), "",
           "Library versions: " + ", ".join(f"{k} {v}" for k, v in R["env"].items() if v), ""]
    (HERE / "reports" / "confirmation.md").write_text("\n".join(md))
    print("wrote reports/confirmation.md")


if __name__ == "__main__":
    main()
