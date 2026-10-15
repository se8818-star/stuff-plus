"""
Download regular-season Statcast pitch data (2022-2026 by default) and cache it
as one parquet per season in ./data/.

    python fetch_data.py                 # 2022-2026
    python fetch_data.py 2022 2026       # custom range (inclusive)
    python fetch_data.py --force 2026    # rebuild one season from scratch

Resumable: each season is pulled in 7-day chunks, every chunk is written to
data/chunks/<year>/ as soon as it arrives, and a rerun only requests the chunks
that are missing. Failed requests are retried with exponential backoff.
Progress goes to stdout and data/fetch.log.
"""
import argparse
import logging
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

KEEP = [
    "game_date", "game_type", "game_pk", "at_bat_number", "pitch_number",
    "inning", "inning_topbot", "outs_when_up", "n_thruorder_pitcher",
    "pitcher", "player_name", "p_throws", "batter", "stand", "pitch_type", "pitch_name",
    "balls", "strikes", "description", "events", "type", "bb_type", "zone",
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis", "release_extension",
    "release_pos_x", "release_pos_y", "release_pos_z", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z",
    "api_break_z_with_gravity", "api_break_x_arm",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "delta_run_exp", "delta_pitcher_run_exp", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "launch_speed", "launch_angle", "hit_distance_sc",
    "home_team", "away_team",
]
KEY = ["game_pk", "at_bat_number", "pitch_number"]

# Generous window around every regular season (2022 ran to Oct 5 after the
# lockout); spring training and postseason are removed by game_type == "R".
WINDOW = ("03-15", "10-07")
CHUNK_DAYS = 7
MAX_TRIES = 6

log = logging.getLogger("fetch")


def chunks(year, today=None):
    start = date.fromisoformat(f"{year}-{WINDOW[0]}")
    end = min(date.fromisoformat(f"{year}-{WINDOW[1]}"), today or date.today())
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_chunk(start, stop, fetch=None, sleep=time.sleep):
    """One date range from Savant, retried with exponential backoff."""
    if fetch is None:
        from pybaseball import statcast
        fetch = lambda a, b: statcast(start_dt=a, end_dt=b, verbose=False)  # noqa: E731
    for attempt in range(1, MAX_TRIES + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = fetch(start.isoformat(), stop.isoformat())
            if df is None:
                df = pd.DataFrame()
            return df[[c for c in KEEP if c in df.columns]]
        except Exception as e:  # network errors, Savant 5xx, parser errors on error pages
            if attempt == MAX_TRIES:
                raise
            wait = min(300, 10 * 2 ** (attempt - 1))
            log.warning("%s -> %s failed (%s: %s); retry %d/%d in %ds", start, stop,
                        type(e).__name__, str(e)[:120], attempt, MAX_TRIES - 1, wait)
            sleep(wait)


def fetch_season(year, out_dir, force=False, fetch=None):
    out = out_dir / f"statcast_{year}.parquet"
    cdir = out_dir / "chunks" / str(year)
    if force:
        out.unlink(missing_ok=True)
        for f in cdir.glob("*.parquet"):
            f.unlink()
    if out.exists():
        log.info("%d: cached (%s)", year, out.name)
        return out
    cdir.mkdir(parents=True, exist_ok=True)
    todo = list(chunks(year))
    for i, (start, stop) in enumerate(todo, 1):
        f = cdir / f"{start}_{stop}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        df = fetch_chunk(start, stop, fetch)
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)      # empty chunks are saved too, so they are not refetched
        tmp.rename(f)
        log.info("%d: chunk %d/%d %s -> %s  %s rows  %.0fs", year, i, len(todo), start, stop,
                 f"{len(df):,}", time.time() - t0)

    frames = [pd.read_parquet(cdir / f"{a}_{b}.parquet") for a, b in todo]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    n_all = len(df)
    df = df[df["game_type"] == "R"]
    n_reg = len(df)
    df = (df.drop_duplicates(KEY)
            .sort_values(["game_date"] + KEY, kind="stable")
            .reset_index(drop=True))
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    log.info("%d: saved %s regular-season pitches, %d games (%s non-regular rows and "
             "%s duplicate rows removed)", year, f"{len(df):,}", df["game_pk"].nunique(),
             f"{n_all - n_reg:,}", f"{n_reg - len(df):,}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("years", nargs="*", type=int, help="first and last season (inclusive)")
    ap.add_argument("--force", type=int, nargs="*", default=[], help="seasons to refetch")
    args = ap.parse_args()
    lo, hi = (args.years + args.years)[:2] if args.years else (2022, 2026)

    d = Path(__file__).parent / "data"
    d.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(d / "fetch.log")])
    import pybaseball
    log.info("fetching %d-%d with pybaseball %s, pandas %s", lo, hi,
             pybaseball.__version__, pd.__version__)
    for y in range(lo, hi + 1):
        fetch_season(y, d, force=y in args.force)
    log.info("done")


if __name__ == "__main__":
    main()
