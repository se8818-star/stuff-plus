"""
Download regular-season Statcast pitch data (2022-2026 by default) and cache it
as one parquet per season in ./data/.

    pip install pybaseball pyarrow
    python fetch_data.py                 # 2022-2026
    python fetch_data.py 2022 2026       # custom range (inclusive)

Pulls in ~2-week chunks so Savant's row limit is never hit. A full season is
roughly 700k pitches and takes a while; already-cached seasons are skipped.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

KEEP = [
    "game_date", "game_type", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "p_throws", "stand", "pitch_type",
    "balls", "strikes", "description", "events",
    "release_speed", "release_spin_rate", "spin_axis", "release_extension",
    "release_pos_x", "release_pos_z", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "delta_run_exp", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "launch_speed", "launch_angle",
    "sz_top", "sz_bot", "batter", "home_team",
]

SEASONS = {  # approximate regular-season windows; filtered by game_type anyway
    2022: ("2022-04-07", "2022-10-05"),
}


def season_window(year):
    if year in SEASONS:
        return SEASONS[year]
    return (f"{year}-03-15", f"{year}-10-02")


def fetch_season(year, out_dir):
    from pybaseball import statcast, cache
    cache.enable()
    out = out_dir / f"statcast_{year}.parquet"
    if out.exists():
        print(f"{year}: cached")
        return
    start, end = (date.fromisoformat(d) for d in season_window(year))
    chunks, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=13), end)
        print(f"{year}: {cur} -> {stop}")
        df = statcast(start_dt=cur.isoformat(), end_dt=stop.isoformat(), verbose=False)
        if df is not None and len(df):
            chunks.append(df[[c for c in KEEP if c in df.columns]])
        cur = stop + timedelta(days=1)
    df = pd.concat(chunks, ignore_index=True)
    df = df[df["game_type"] == "R"]
    df.to_parquet(out, index=False)
    print(f"{year}: saved {len(df):,} pitches")


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (2022, 2026)
    d = Path(__file__).parent / "data"
    d.mkdir(exist_ok=True)
    for y in range(lo, hi + 1):
        fetch_season(y, d)
