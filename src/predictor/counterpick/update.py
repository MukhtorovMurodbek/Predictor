"""Keep the counter-picker statistics up to date. The bot runs this every hour.

    python -m predictor.counterpick.update             (fetch new games + refresh tables)
    python -m predictor.counterpick.update --rebuild   (only rebuild tables from saved games)

1. Download public games newer than the newest one we have (all ranks).
2. Recount the daily tables (data/counterpick/YYYY-MM-DD.npz) for the days that got new games.
3. Delete raw games older than RAW_KEEP_DAYS from public_matches.parquet. The daily tables
   keep their counts, so the 90-day statistics don't lose anything, and the raw file stays small.
"""

import argparse
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..data import HEROES_PATH, PATCHES_PATH, SOURCES
from ..data.collect import collect_heroes, collect_public
from ..data.opendota import OpenDota
from .stats import Counts, day_path

RAW_KEEP_DAYS = 7
CALLS_PER_UPDATE = 100  # at most ~8,700 games per hourly update
MIN_RANK = 10  # all ranks: the counter-picker is for regular players
KEEP_QUOTA = 400  # leave this many OpenDota calls per day for the pro models and /live

_lock = threading.Lock()  # one update at a time (hourly job vs. /train counterpick)


def _day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")


def rebuild_days(df: pd.DataFrame, days=None) -> list[str]:
    """Recount the daily tables for `days` (default: every day in df)."""
    df = df.assign(day=_day(df["start_time"]))
    days = sorted(set(df["day"]) if days is None else days)
    for day in days:
        Counts.from_matches(df[df["day"] == day]).save(day_path(day))
    return days


def prune_raw(df: pd.DataFrame, keep_days: int = RAW_KEEP_DAYS) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).timestamp()
    return df[df["start_time"] >= cutoff]


def update(max_calls: int = CALLS_PER_UPDATE, download: bool = True) -> str:
    """Fetch new games, refresh the daily tables, prune old raw games. Returns a summary."""
    if not _lock.acquire(blocking=False):
        return "an update is already running"
    try:
        path = SOURCES["public"]
        before = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        newest = int(before["match_id"].max()) if len(before) else None
        calls = 0
        if download:
            api = OpenDota()
            if not HEROES_PATH.exists() or not PATCHES_PATH.exists():
                collect_heroes(api)
            calls = collect_public(api, max_calls, MIN_RANK, save_every=max_calls + 1, stop_at_id=newest, min_quota=KEEP_QUOTA)
        df = pd.read_parquet(path) if path.exists() else before
        new = df[df["match_id"] > newest] if newest is not None else df
        days = rebuild_days(df, set(_day(new["start_time"])) if download else None)
        kept = prune_raw(df)
        if len(kept) < len(df):
            kept.to_parquet(path, index=False)
        return (
            f"counter-picker update: +{len(new):,} new games ({calls} API calls), "
            f"recounted {len(days)} day(s), raw file keeps {len(kept):,} games from the last {RAW_KEEP_DAYS} days"
        )
    finally:
        _lock.release()


def main(argv=None):
    p = argparse.ArgumentParser(description="Update the counter-picker statistics")
    p.add_argument("--rebuild", action="store_true", help="only recount daily tables from saved games, no download")
    p.add_argument("--calls", type=int, default=CALLS_PER_UPDATE)
    args = p.parse_args(argv)
    print(update(args.calls, download=not args.rebuild))


if __name__ == "__main__":
    main()
