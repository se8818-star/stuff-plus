from datetime import date

import pandas as pd
import pytest

import fetch_data as fd


def fake_fetch(calls, fail_first=0):
    def fetch(a, b):
        calls.append((a, b))
        if len(calls) <= fail_first:
            raise ConnectionError("boom")
        days = pd.date_range(a, b)
        return pd.DataFrame({"game_date": days, "game_type": "R", "game_pk": [d.toordinal() for d in days],
                             "at_bat_number": 1, "pitch_number": 1, "not_kept": 0})
    return fetch


def test_chunks_cover_the_window_without_gaps_or_overlap():
    c = list(fd.chunks(2023, today=date(2030, 1, 1)))
    assert c[0][0] == date(2023, 3, 15) and c[-1][1] == date(2023, 10, 7)
    assert all(b[0] == a[1] + pd.Timedelta(days=1) for a, b in zip(c, c[1:]))
    assert list(fd.chunks(2026, today=date(2026, 3, 20)))[-1][1] == date(2026, 3, 20)


def test_retry_with_backoff_then_success():
    calls, waits = [], []
    df = fd.fetch_chunk(date(2024, 4, 1), date(2024, 4, 7), fake_fetch(calls, 2), waits.append)
    assert len(calls) == 3 and waits == [10, 20]
    assert len(df) == 7 and "not_kept" not in df.columns


def test_gives_up_after_max_tries():
    calls = []
    with pytest.raises(ConnectionError):
        fd.fetch_chunk(date(2024, 4, 1), date(2024, 4, 7), fake_fetch(calls, 99), lambda s: None)
    assert len(calls) == fd.MAX_TRIES


def test_season_resumes_from_cached_chunks(tmp_path):
    calls = []
    with pytest.raises(ConnectionError):          # dies part-way through the season
        def flaky(a, b):
            if len(calls) >= 5:
                raise ConnectionError("down")
            return fake_fetch(calls)(a, b)
        orig, fd.MAX_TRIES = fd.MAX_TRIES, 1
        try:
            fd.fetch_season(2023, tmp_path, fetch=flaky)
        finally:
            fd.MAX_TRIES = orig
    assert len(calls) == 5 and not (tmp_path / "statcast_2023.parquet").exists()

    calls2 = []
    out = fd.fetch_season(2023, tmp_path, fetch=fake_fetch(calls2))
    n_chunks = len(list(fd.chunks(2023)))
    assert len(calls2) == n_chunks - 5            # only the missing chunks are requested
    df = pd.read_parquet(out)
    assert len(df) == 207 and df["game_date"].is_monotonic_increasing

    calls3 = []
    fd.fetch_season(2023, tmp_path, fetch=fake_fetch(calls3))
    assert calls3 == []                           # finished season is not refetched
