"""
Score Statcast pitches with the saved cross-fitted Stuff+ bundle.

    python score.py data/statcast_2026.parquet scores.csv            # shipped model
    python score.py data/statcast_2026.parquet scores.csv --model S6 # six-variable comparison model

Input: a parquet or CSV of Statcast pitches (pybaseball column names) covering
whole pitcher-seasons, because the fastball-difference features are computed per
pitcher-season. A pitcher the bundle was trained on is scored by the fold model
that never saw him; anyone else by the mean of the five fold models.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from develop import pitcher_table, shrink
from features import EXCLUDED_PITCH_TYPES, add_features, plain_dtypes
from final import VERSION
from models import crossfit_xrv

HERE = Path(__file__).parent


def load_bundle(model="A", version=VERSION):
    tag = f"stuffplus{'' if model == 'A' else '_' + model.lower()}_v{version}"
    return (joblib.load(HERE / "models" / f"{tag}.joblib"),
            json.loads((HERE / "models" / f"{tag}.json").read_text()))


def score_pitches(raw, bundle, meta):
    """Per-pitch Stuff+ (column `stuff_plus`) for raw Statcast rows."""
    df = plain_dtypes(raw)
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    df = add_features(df)
    xrv = crossfit_xrv(bundle["fold_models"], bundle["pitcher_fold"], df, bundle["features"])
    sc = meta["scaling"]
    df["xrv"] = xrv
    df["stuff_plus"] = 100 - 10 * (xrv - sc["xrv_mean"]) / sc["xrv_sd"]
    return df


def pitcher_scores(scored, meta, by=("pitcher", "year")):
    """Pitcher-season means, raw and shrunk toward the league mean of the rows scored."""
    t = pitcher_table(scored, scored["stuff_plus"].to_numpy(), by)
    t["stuff_plus"] = np.nan
    for _, idx in t.groupby(level="year").groups.items():
        t.loc[idx, "stuff_plus"] = shrink(t.loc[idx], meta["shrinkage"]["pitcher_k"]["k"])["mean"]
    return t.rename(columns={"mean": "stuff_plus_raw", "size": "pitches"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", default="A", choices=["A", "S6"])
    ap.add_argument("--pitches", action="store_true", help="write one row per pitch instead of per pitcher-season")
    args = ap.parse_args()
    raw = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    bundle, meta = load_bundle(args.model)
    scored = score_pitches(raw, bundle, meta)
    if args.pitches:
        out = scored[["game_pk", "at_bat_number", "pitch_number", "pitcher", "pitch_type", "xrv", "stuff_plus"]]
    else:
        out = pitcher_scores(scored, meta).reset_index()
        if "player_name" in scored:
            out = out.merge(scored.groupby("pitcher")["player_name"].first(), on="pitcher")
        out = out.sort_values("stuff_plus", ascending=False)
    out.round({"stuff_plus": 2, "stuff_plus_raw": 2, "xrv": 5}).to_csv(args.output, index=False)
    known = scored["pitcher"].isin(bundle["pitcher_fold"]).mean()
    print(f"scored {len(scored):,} pitches with {meta['name']} v{meta['version']} "
          f"({known:.1%} from pitchers in the bundle's folds) -> {args.output}")


if __name__ == "__main__":
    main()
