"""Keep the counter-picker statistics up to date. The bot runs this every hour.

    python -m predictor.counterpick.update             (count new and older games)
    python -m predictor.counterpick.update --blocks 300 (count more older 4-hour blocks)

OpenDota's database does the counting. One SQL query (explorer) returns, for a block of time,
how many games and wins every hero pair had, split by medal. That covers every public game
OpenDota lists (about 550,000 a day after the filters below), not a sample, in a ~1 MB answer.
The explorer stops queries after ~15 s, so we ask for 4-hour blocks and split a block in two
when it still times out.

Each update:
1. Forward: count the blocks between the last counted one and 3 hours ago (OpenDota lists a
   game when it ends, so the last few hours are still filling up).
2. Backfill: count older blocks, newest first, until the 90-day window is full.
3. Add each block's counts to its day table (data/counterpick/YYYY-MM-DD.npz).
   state.json keeps the counted time range [counted_from, counted_until), and blocks never
   overlap, so no game is counted twice.
"""

import argparse
import json
import shutil
import threading
import time
from datetime import datetime, timezone

import numpy as np

from ..data import HEROES_PATH, PATCHES_PATH
from ..data.collect import MIN_DURATION, NORMAL_GAME_MODES, collect_heroes
from ..data.opendota import ExplorerError, OpenDota
from . import stats
from .stats import DayTable, WINDOW_DAYS, day_path

BLOCK = 4 * 3600  # seconds of games per query (a busy 6 hours takes ~11 s of the 15 s limit)
MIN_BLOCK = 1800  # give up splitting below this
LAG = 3 * 3600  # only count games that started at least this long ago
BACKFILL_BLOCKS = 200  # older blocks per update (~10 s each): about 33 days of history an hour
MIN_RANK = 10  # all ranks: the counter-picker is for regular players
KEEP_QUOTA = 400  # leave this many OpenDota calls per day for the pro models and /live
DAY = 86400

# t=0: radiant hero a vs dire heroes b (g games, v radiant wins), per medal k
# t=1/2: radiant/dire teammates a < b (g games, v wins of that team)
# t=3: per medal k: b = first start_time, g = games;  t=4: b = last start_time
SQL = """WITH m AS (
  SELECT start_time AS s, radiant_win AS w, radiant_team AS r, dire_team AS d,
         CASE WHEN avg_rank_tier::int = 80 OR (avg_rank_tier::int < 80 AND avg_rank_tier::int % 10 <= 5)
              THEN avg_rank_tier::int / 10 ELSE 0 END AS k
  FROM public_matches
  WHERE start_time >= {t0} AND start_time < {t1}
    AND game_mode IN ({modes}) AND duration >= {min_duration} AND avg_rank_tier >= {min_rank}
    AND radiant_win IS NOT NULL AND array_length(radiant_team, 1) = 5 AND array_length(dire_team, 1) = 5
    AND (SELECT count(DISTINCT x) FROM unnest(radiant_team || dire_team) x WHERE x > 0) = 10
)
SELECT t, k, a, array_agg(b::int) AS b, array_agg(g::int) AS g, array_agg(v::int) AS v FROM (
  SELECT 0 AS t, k, a, b, count(*) AS g, sum(w::int) AS v FROM m, unnest(r) a, unnest(d) b GROUP BY k, a, b
  UNION ALL SELECT 1, 0, a, b, count(*), sum(w::int) FROM m, unnest(r) a, unnest(r) b WHERE a < b GROUP BY a, b
  UNION ALL SELECT 2, 0, a, b, count(*), sum((NOT w)::int) FROM m, unnest(d) a, unnest(d) b WHERE a < b GROUP BY a, b
  UNION ALL SELECT 3, k, 0, min(s), count(*), 0 FROM m GROUP BY k
  UNION ALL SELECT 4, k, 0, max(s), 0, 0 FROM m GROUP BY k
) x GROUP BY t, k, a"""

_lock = threading.Lock()  # one update at a time (hourly job vs. /train counterpick)


def _state_path():
    return stats.DAILY_DIR / "state.json"


def load_state() -> dict:
    """counted_from / counted_until: unix times of the counted range."""
    path = _state_path()
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def save_state(state: dict) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state), "utf-8")


def _day(t: int) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def table_from_rows(rows: list[dict], patch: str) -> DayTable:
    """Turn the SQL answer into counts (see the t= legend above SQL)."""
    table, H = DayTable.empty(), stats.N_HEROES
    for r in rows:
        t, k, a = int(r["t"]), int(r["k"]), int(r["a"])
        b, g, v = (np.asarray(r[c], dtype=np.int64) for c in "bgv")
        if t in (0, 1, 2):
            ok = (b > 0) & (b < H)
            if not 0 < a < H or not ok.any():
                continue  # a hero id we have no room for (a brand-new hero): skip it
            b, g, v = b[ok], g[ok], v[ok]
        if t == 0:
            s = table.slice((patch, k))
            s.games[a, b] += g
            s.wins[a, b] += v
            s.games[b, a] += g
            s.wins[b, a] += g - v
        elif t in (1, 2):
            for x, y in ((a, b), (b, a)):
                table.with_games[x, y] += g
                table.with_wins[x, y] += v
        elif t == 3:
            s = table.slice((patch, k))
            s.n, s.first = int(g[0]), int(b[0])
        elif t == 4:
            table.slice((patch, k)).last = int(b[0])
    return table


def _patch_pieces(t0: int, t1: int) -> list[tuple[int, int, str]]:
    """[t0, t1) cut where a new patch starts, each piece labelled with its patch."""
    patches = json.loads(PATCHES_PATH.read_text("utf-8")) if PATCHES_PATH.exists() else []
    starts = [(int(datetime.fromisoformat(p["date"].replace("Z", "+00:00")).timestamp()), p["name"]) for p in patches]
    cuts = sorted({t0, t1} | {s for s, _ in starts if t0 < s < t1})
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        names = [name for s, name in starts if s <= a]
        pieces.append((a, b, names[-1] if names else ""))
    return pieces


def _count(api: OpenDota, t0: int, t1: int, patch: str) -> tuple[DayTable, int]:
    """Count [t0, t1); split in two when the explorer times out. Returns (table, queries)."""
    sql = SQL.format(t0=t0, t1=t1, modes=",".join(map(str, sorted(NORMAL_GAME_MODES))),
                     min_duration=MIN_DURATION, min_rank=MIN_RANK)
    try:
        return table_from_rows(api.explorer(sql, tries=1), patch), 1
    except ExplorerError:
        if t1 - t0 <= MIN_BLOCK:
            raise
        mid = t0 + (t1 - t0) // 2
        left, n1 = _count(api, t0, mid, patch)
        right, n2 = _count(api, mid, t1, patch)
        return left + right, n1 + n2 + 1


def count_block(api: OpenDota, t0: int, t1: int) -> tuple[DayTable, int]:
    """Count the games that started in [t0, t1) (inside one UTC day)."""
    total, queries = DayTable.empty(), 0
    for a, b, patch in _patch_pieces(t0, t1):
        table, n = _count(api, a, b, patch)
        total, queries = total + table, queries + n
    return total, queries


def add_to_day(table: DayTable, day: str) -> None:
    path = day_path(day)
    old = DayTable.load(path) if path.exists() else DayTable.empty()
    (old + table).save(path)


def _set_aside_old_tables() -> int:
    """Tables from before the switch to OpenDota's database (a sample of games, another file
    format) go to data/counterpick/old/ so they aren't mixed in."""
    old = [p for p in stats.DAILY_DIR.glob("*.npz")] + [p for p in [_state_path()] if p.exists()]
    if not old:
        return 0
    target = stats.DAILY_DIR / "old"
    target.mkdir(parents=True, exist_ok=True)
    for p in old:
        shutil.move(str(p), target / p.name)
    return len(old)


def update(backfill_blocks: int = BACKFILL_BLOCKS, api: OpenDota | None = None, now: float | None = None) -> str:
    """Count new blocks, then older ones until the 90-day window is full."""
    if not _lock.acquire(blocking=False):
        return "an update is already running"
    try:
        state = load_state()
        notes = []
        if "counted_until" not in state:
            if moved := _set_aside_old_tables():
                notes.append(f"moved {moved} old file(s) to data/counterpick/old/")
            state = {}
        api = api or OpenDota()
        if not HEROES_PATH.exists() or not PATCHES_PATH.exists():
            collect_heroes(api)
        now = int(now or time.time())
        ready = (now - LAG) // BLOCK * BLOCK
        window_start = (now // DAY - WINDOW_DAYS) * DAY
        if not state:
            state = {"counted_from": ready, "counted_until": ready}
        games = queries = 0
        days = set()

        def quota_ok() -> bool:
            return api.remaining_day is None or api.remaining_day > KEEP_QUOTA

        def count(t0, t1):
            nonlocal games, queries
            table, n = count_block(api, t0, t1)
            add_to_day(table, _day(t0))
            games, queries = games + table.n_matches, queries + n
            days.add(_day(t0))

        while state["counted_until"] < ready and quota_ok():  # 1. new games
            t0 = state["counted_until"]
            t1 = min(ready, (t0 // BLOCK + 1) * BLOCK)
            count(t0, t1)
            state["counted_until"] = t1
            save_state(state)
        backfilled = 0
        while state["counted_from"] > window_start and backfilled < backfill_blocks and quota_ok():  # 2. older games
            t1 = state["counted_from"]
            t0 = max(window_start, (t1 - 1) // BLOCK * BLOCK)
            count(t0, t1)
            state["counted_from"] = t0
            save_state(state)
            backfilled += 1
            if t0 % DAY == 0:
                print(f"  counter-picker: counted back to {_day(t0)} ({games:,} games, {queries} queries so far)", flush=True)
        for p in stats.DAILY_DIR.glob("*.npz"):  # days long out of the window
            if p.stem < _day(window_start - 7 * DAY):
                p.unlink()
        full = state["counted_from"] <= window_start
        notes.append(
            f"counted {games:,} games in {queries} queries ({len(days)} day table(s) updated); "
            f"tables cover {_day(state['counted_from'])} to {_day(state['counted_until'] - 1)}"
            + ("" if full else f", still filling the {WINDOW_DAYS}-day window")
        )
        if not quota_ok():
            notes.append(f"stopped early to keep {KEEP_QUOTA} OpenDota calls for the other models")
        return "counter-picker update: " + "; ".join(notes)
    finally:
        _lock.release()


def main(argv=None):
    p = argparse.ArgumentParser(description="Update the counter-picker statistics")
    p.add_argument("--blocks", type=int, default=BACKFILL_BLOCKS, help="older 4-hour blocks to count")
    args = p.parse_args(argv)
    print(update(args.blocks))


if __name__ == "__main__":
    main()
