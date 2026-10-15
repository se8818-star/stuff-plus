"""
Side-by-side 2026 comparison of the shipped model (A) and the six-variable
comparison model (S6), from their leaderboards.

    python compare_models.py    # leaderboards/comparison_2026.csv, reports/comparison_a_vs_s6.md
"""
import json
from pathlib import Path

import pandas as pd

from develop import table

HERE = Path(__file__).parent


def main():
    lb = HERE / "leaderboards"
    a, s = pd.read_csv(lb / "pitchers_2026.csv"), pd.read_csv(lb / "pitchers_2026_s6.csv")
    keep = ["pitcher", "rank", "stuff_plus", "grade"]
    c = a[["pitcher", "player_name", "p_throws", "role", "pitches", "velo", "rank", "stuff_plus", "grade"]].merge(
        s[keep], on="pitcher", suffixes=("_a", "_s6"))
    c["stuff_plus_diff"] = (c["stuff_plus_s6"] - c["stuff_plus_a"]).round(1)
    c["rank_diff"] = c["rank_a"] - c["rank_s6"]            # positive: S6 ranks him higher
    c = c.sort_values("rank_a")
    c.to_csv(lb / "comparison_2026.csv", index=False)

    ta, ts = pd.read_csv(lb / "pitch_types_2026.csv"), pd.read_csv(lb / "pitch_types_2026_s6.csv")
    t = ta.merge(ts[["pitcher", "pitch_type", "stuff_plus"]], on=["pitcher", "pitch_type"], suffixes=("_a", "_s6"))
    by_type = t.groupby("pitch_type").apply(lambda g: pd.Series({
        "pairs": len(g), "r": g["stuff_plus_a"].corr(g["stuff_plus_s6"]),
        "mean_a": g["stuff_plus_a"].mean(), "mean_s6": g["stuff_plus_s6"].mean()}))
    sa = json.loads((HERE / "results" / "final_summary.json").read_text())
    ss = json.loads((HERE / "results" / "final_summary_s6.json").read_text())
    big = c[c["pitches"] >= 500]
    top = lambda n, col: set(big.nsmallest(n, col)["pitcher"])  # noqa: E731
    f = lambda d: [d["player_name"], d["role"], f"{d['pitches']:,}", f"{d['stuff_plus_a']:.1f}",  # noqa: E731
                   f"{d['stuff_plus_s6']:.1f}", f"{d['stuff_plus_diff']:+.1f}"]
    head = ["Pitcher", "Role", "Pitches", "A", "S6", "S6 − A"]
    md = ["# Shipped model (A) against the six-variable model (S6), 2026", "",
          "Both trained on 2022–2026 with the same settings; scores are out-of-fold by pitcher and shrunk. "
          "A uses ten inputs (velocity, spin, extension, vertical and horizontal acceleration, release x/z, and "
          "three fastball differences). S6 uses six (velocity, extension, location-neutral VAA and HAA, and their "
          "differences from the fastball). S6 was not confirmed as better or worse than A "
          "(`reports/confirmation.md`).", "",
          table(["", "A", "S6"], [
              ["Inputs", "10", "6"],
              ["Qualified pitchers (≥100 pitches)", str(len(a)), str(len(s))],
              ["Pitcher-level sd of Stuff+", f"{a['stuff_plus'].std():.2f}", f"{s['stuff_plus'].std():.2f}"],
              ["Split-half reliability", f"{sa['shrinkage']['pitcher']['r_half']:.3f}",
               f"{ss['shrinkage']['pitcher']['r_half']:.3f}"],
              ["Correlation with velocity", f"{c['stuff_plus_a'].corr(c['velo']):.3f}",
               f"{c['stuff_plus_s6'].corr(c['velo']):.3f}"],
              ["RP minus SP (mean)", f"{sa['sp_rp']['RP']['stuff_plus_mean'] - sa['sp_rp']['SP']['stuff_plus_mean']:+.2f}",
               f"{ss['sp_rp']['RP']['stuff_plus_mean'] - ss['sp_rp']['SP']['stuff_plus_mean']:+.2f}"],
              ["RHP minus LHP (mean)",
               f"{c[c.p_throws == 'R']['stuff_plus_a'].mean() - c[c.p_throws == 'L']['stuff_plus_a'].mean():+.2f}",
               f"{c[c.p_throws == 'R']['stuff_plus_s6'].mean() - c[c.p_throws == 'L']['stuff_plus_s6'].mean():+.2f}"],
              ["Coors effect", f"{sa['parks_2026']['COL']['stuff_plus_diff']:+.2f}",
               f"{ss['parks_2026']['COL']['stuff_plus_diff']:+.2f}"]]), "",
          "## Agreement", "",
          f"- Pitcher Stuff+ correlation: {c['stuff_plus_a'].corr(c['stuff_plus_s6']):.3f} "
          f"(Spearman {c['stuff_plus_a'].corr(c['stuff_plus_s6'], method='spearman'):.3f}) over {len(c)} pitchers; "
          f"{big['stuff_plus_a'].corr(big['stuff_plus_s6']):.3f} among the {len(big)} with at least 500 pitches.",
          f"- Median absolute difference: {c['stuff_plus_diff'].abs().median():.1f} points; "
          f"{(c['stuff_plus_diff'].abs() > 5).mean() * 100:.0f}% of pitchers differ by more than 5.",
          f"- Top 25 overlap (≥500 pitches): {len(top(25, 'rank_a') & top(25, 'rank_s6'))} of 25; "
          f"top 50: {len(top(50, 'rank_a') & top(50, 'rank_s6'))} of 50.", "",
          "## By pitch type (pitcher × pitch, ≥50 pitches)", "",
          table(["Pitch", "Pairs", "Correlation A vs S6", "Mean A", "Mean S6"],
                [[pt, f"{int(r['pairs'])}", f"{r['r']:.3f}", f"{r['mean_a']:.1f}", f"{r['mean_s6']:.1f}"]
                 for pt, r in by_type.iterrows()]), "",
          "## Pitchers S6 likes more (≥500 pitches)", "",
          table(head, [f(d) for _, d in big.nlargest(10, "stuff_plus_diff").iterrows()]), "",
          "## Pitchers S6 likes less (≥500 pitches)", "",
          table(head, [f(d) for _, d in big.nsmallest(10, "stuff_plus_diff").iterrows()]), "",
          "Full table: `leaderboards/comparison_2026.csv`.", ""]
    (HERE / "reports" / "comparison_a_vs_s6.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
