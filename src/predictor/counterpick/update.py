"""Keep the counter-picker statistics up to date. The bot runs this every hour.

    python -m predictor.counterpick.update             (fetch games + add them to the tables)
    python -m predictor.counterpick.update --rebuild   (recount tables from the saved raw games)

Each update:
1. Forward: download public games newer than the newest one we have (all ranks).
2. Backfill: recover the last 60 days of raw games, then continue the 90-day daily tables.
   A separate cursor lets existing installations re-fetch games previously pruned from raw.
3. Add only games outside the already-counted ID range to daily tables. The 60-day recovery
   may re-fetch previously counted games; these must not be added to daily tables again.
4. Delete raw games older than RAW_KEEP_DAYS from public_matches.parquet (their counts stay in
   the daily tables), so the raw file stays small.
"""

import argparse
import json
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..data import HEROES_PATH, PATCHES_PATH, SOURCES
from ..data.collect import collect_heroes, collect_public
from ..data.opendota import OpenDota
from ..io import atomic_write
from . import stats
from .stats import Counts, WINDOW_DAYS, day_path

RAW_KEEP_DAYS = 60
CALLS_PER_UPDATE = 100  # at most ~8,700 games per hourly update
MIN_RANK = 10  # all ranks: the counter-picker is for regular players
KEEP_QUOTA = 400  # leave this many OpenDota calls per day for the pro models and /live

_lock = threading.Lock()  # one update at a time (hourly job vs. /train counterpick)


def _day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")


def _state_path():
    return stats.DAILY_DIR / "state.json"


def load_state() -> dict:
    """oldest = lowest match id / time ever fetched (where backfill continues)."""
    path = _state_path()
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def save_state(state: dict) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state), "utf-8")


def rebuild_days(df: pd.DataFrame, days=None) -> list[str]:
    """Recount the daily tables for `days` (default: every day in df) from raw games only."""
    df = df.assign(day=_day(df["start_time"]))
    days = sorted(set(df["day"]) if days is None else days)
    for day in days:
        Counts.from_matches(df[df["day"] == day]).save(day_path(day))
    return days


def add_to_days(new: pd.DataFrame) -> list[str]:
    """Add the counts of games not counted before to their daily tables."""
    new = new.assign(day=_day(new["start_time"]))
    days = sorted(set(new["day"]))
    for day in days:
        path = day_path(day)
        old = Counts.load(path) if path.exists() else Counts.empty()
        (old + Counts.from_matches(new[new["day"] == day])).save(path)
    return days


def prune_raw(df: pd.DataFrame, keep_days: int = RAW_KEEP_DAYS) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).timestamp()
    return df[df["start_time"] >= cutoff]


def update(max_calls: int = CALLS_PER_UPDATE) -> str:
    """Fetch new and older games, add them to the daily tables, prune raw games."""
    if not _lock.acquire(blocking=False):
        return "an update is already running"
    try:
        path = SOURCES["public"]
        before = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["match_id", "start_time"])
        newest = int(before["match_id"].max()) if len(before) else None
        state = load_state()
        oldest = state.get("oldest_id", int(before["match_id"].min()) if len(before) else None)
        # This cursor is independent of the daily tables: an older version pruned raw
        # games even while backfilling the 90-day tables. It is saved even when all
        # fetched rows are ineligible or older than the retention cutoff.
        if "raw_oldest_id" not in state and len(before):
            first_raw = before.loc[before["match_id"].idxmin()]
            state.update(raw_oldest_id=int(first_raw["match_id"]), raw_oldest_time=int(first_raw["start_time"]))
        api = OpenDota()
        if not HEROES_PATH.exists() or not PATCHES_PATH.exists():
            collect_heroes(api)

        calls = collect_public(api, max_calls, MIN_RANK, save_every=max_calls + 1, stop_at_id=newest, min_quota=KEEP_QUOTA)
        now = datetime.now(timezone.utc)
        raw_start = (now - timedelta(days=RAW_KEEP_DAYS)).timestamp()
        window_start = (now - timedelta(days=WINDOW_DAYS)).timestamp()
        raw_cursor = state.get("raw_oldest_id")
        recover_raw = raw_cursor is not None and state.get("raw_oldest_time", float("inf")) > raw_start
        daily_backfill = oldest is not None and state.get("oldest_time", float("inf")) > window_start
        cursor = raw_cursor if recover_raw else (oldest if daily_backfill else None)
        page_cursor = None

        def record_page(batch):
            nonlocal page_cursor
            first = min(batch, key=lambda row: row["match_id"])
            # A missing timestamp cannot prove the boundary has been reached.
            timestamp = first.get("start_time")
            page_cursor = (int(first["match_id"]), int(timestamp) if timestamp else None)

        if cursor is not None and calls < max_calls:
            calls += collect_public(api, max_calls - calls, MIN_RANK, save_every=max_calls + 1,
                                    start_before=cursor, min_quota=KEEP_QUOTA, on_page=record_page)

        df = pd.read_parquet(path) if path.exists() else before
        is_new = (df["match_id"] > newest) if newest is not None else pd.Series(True, index=df.index)
        if oldest is not None:
            is_new |= df["match_id"] < oldest
        new = df[is_new]
        days = add_to_days(new) if len(new) else []
        if len(df):
            first = df.loc[df["match_id"].idxmin()]
            if oldest is None or first["match_id"] < oldest:
                state.update(oldest_id=int(first["match_id"]), oldest_time=int(first["start_time"]))
        if page_cursor is not None:
            page_id, page_time = page_cursor
            if recover_raw:
                state["raw_oldest_id"] = page_id
                if page_time is None:
                    state.pop("raw_oldest_time", None)
                else:
                    state["raw_oldest_time"] = page_time
            # Also advance the daily cursor over pages with no eligible games.
            if oldest is None or page_id < oldest:
                state["oldest_id"] = page_id
                if page_time is None:
                    state.pop("oldest_time", None)
                else:
                    state["oldest_time"] = page_time
        if "raw_oldest_id" not in state and len(df):
            first_raw = df.loc[df["match_id"].idxmin()]
            state.update(raw_oldest_id=int(first_raw["match_id"]), raw_oldest_time=int(first_raw["start_time"]))
        save_state(state)
        kept = prune_raw(df)
        if len(kept) < len(df):
            with atomic_write(path) as tmp:
                kept.to_parquet(tmp, index=False)
        return (
            f"counter-picker update: +{len(new):,} games ({calls} API calls), {len(days)} day table(s) updated, "
            f"raw file keeps {len(kept):,} games from the last {RAW_KEEP_DAYS} days"
        )
    finally:
        _lock.release()


def rebuild() -> str:
    """Recount every day that has raw games. Days whose raw games were already pruned keep
    their tables. Use this after changing how counting works."""
    path = SOURCES["public"]
    df = pd.read_parquet(path)
    days = rebuild_days(df)
    first = df.loc[df["match_id"].idxmin()]
    save_state({"oldest_id": int(first["match_id"]), "oldest_time": int(first["start_time"])})
    return f"recounted {len(days)} day(s) from {len(df):,} raw games"


def main(argv=None):
    p = argparse.ArgumentParser(description="Update the counter-picker statistics")
    p.add_argument("--rebuild", action="store_true", help="only recount daily tables from saved games, no download")
    p.add_argument("--calls", type=int, default=CALLS_PER_UPDATE)
    args = p.parse_args(argv)
    print(rebuild() if args.rebuild else update(args.calls))


if __name__ == "__main__":
    main()
