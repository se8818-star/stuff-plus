"""
Download TJStats' public per-pitch tjStuff+ and average it per pitcher-season.

    python -m experiments.tjstats_comparison.fetch            # 2021-2026

Source: the Hugging Face dataset TJStatsApps/mlb_data (Thomas Nestico). The raw
parquet files go to data/external/tjstats/ and the aggregates to data/external/;
both are gitignored like the rest of data/, so rerun this instead of committing
them. Writes:
  tjstuff_pitcher_seasons.csv     pitcher_id, season, pitches, tj_stuff_plus
  tjstuff_pitch_type_seasons.csv  pitcher_id, season, pitch_type, pitches, tj_stuff_plus
tj_stuff_plus is the mean of TJ's per-pitch value on his scale (no rescaling).
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external"
RAW = OUT / "tjstats"
BASE = "https://huggingface.co/datasets/TJStatsApps/mlb_data/resolve/main/stuff"

# the first name present in the file wins
PITCHER = ["pitcher_id", "pitcher"]
PITCH_TYPE = ["pitch_type"]
VALUE = ["tj_stuff_plus"]


def download(name, force=False):
    path = RAW / name
    if force or not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{BASE}/{name}") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def pick(df, names, what):
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"no {what} column (looked for {names}); columns are {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="*", type=int, default=list(range(2021, 2027)))
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    a = ap.parse_args()

    pitchers, types = [], []
    for y in a.seasons:
        df = pd.read_parquet(download(f"stuff_pbp_{y}.parquet", a.force))
        if "game_type" in df.columns:
            df = df[df["game_type"] == "R"]
        pid, pt, val = pick(df, PITCHER, "pitcher id"), pick(df, PITCH_TYPE, "pitch type"), pick(df, VALUE, "tjStuff+")
        df = df.dropna(subset=[val]).rename(columns={pid: "pitcher_id", pt: "pitch_type", val: "tj_stuff_plus"})
        g = df.groupby("pitcher_id")["tj_stuff_plus"]
        pitchers.append(pd.DataFrame({"season": y, "pitches": g.size(), "tj_stuff_plus": g.mean()}).reset_index())
        g = df.groupby(["pitcher_id", "pitch_type"])["tj_stuff_plus"]
        types.append(pd.DataFrame({"season": y, "pitches": g.size(), "tj_stuff_plus": g.mean()}).reset_index())
        print(f"{y}: {len(df)} pitches, {pitchers[-1].shape[0]} pitchers", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["pitcher_id", "season", "pitches", "tj_stuff_plus"]
    pd.concat(pitchers)[cols].round(2).to_csv(OUT / "tjstuff_pitcher_seasons.csv", index=False)
    pd.concat(types)[cols[:2] + ["pitch_type"] + cols[2:]].round(2).to_csv(
        OUT / "tjstuff_pitch_type_seasons.csv", index=False)

    print(f"wrote {OUT.relative_to(ROOT)}/tjstuff_*.csv")


if __name__ == "__main__":
    main()
