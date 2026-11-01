"""Data collectors.

    python -m predictor.collect heroes
    python -m predictor.collect pro --since 2025-01-01
    python -m predictor.collect public --calls 500 --min-rank 70
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR, RAW_DIR
from .data import D_COLS, HEROES_PATH, PATCHES_PATH, R_COLS, SOURCES, patch_for_times, to_unix
from .opendota import OpenDota, QuotaExhausted

# All Pick, Captains Mode, Random Draft, Single Draft, Captains Draft, Ranked All Pick.
# Leaves out Turbo (23), Ability Draft (18), All Random (5) and event modes.
NORMAL_GAME_MODES = {1, 2, 3, 4, 16, 22}
MIN_DURATION = 600  # seconds; shorter games are mostly abandons

PRO_SQL = """
SELECT m.match_id, m.start_time, m.duration, m.radiant_win, m.leagueid,
       m.radiant_team_id, m.dire_team_id, mp.patch, pm.player_slot, pm.hero_id
FROM matches m
JOIN match_patch mp USING (match_id)
JOIN player_matches pm USING (match_id)
WHERE m.start_time >= {since} AND m.match_id < {cursor}
ORDER BY m.match_id DESC
LIMIT {limit}
"""
PRO_META = ["start_time", "duration", "radiant_win", "leagueid", "radiant_team_id", "dire_team_id", "patch"]


def save_matches(df: pd.DataFrame, path: Path) -> int:
    """Merge into an existing parquet file, dropping duplicate match_ids. Returns total rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates("match_id", keep="last").sort_values("match_id", ignore_index=True)
    df.to_parquet(path, index=False)
    return len(df)


def collect_heroes(api: OpenDota) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    heroes = api.heroes()
    HEROES_PATH.write_text(json.dumps(heroes, indent=1), "utf-8")
    PATCHES_PATH.write_text(json.dumps(api.patches(), indent=1), "utf-8")
    print(f"saved {len(heroes)} heroes and patch dates to {DATA_DIR}")


def _pro_rows_to_wide(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[df["hero_id"] > 0].sort_values(["match_id", "player_slot"])
    side = np.where(df["player_slot"] < 128, "r", "d")
    df["col"] = side + df.groupby(["match_id", side]).cumcount().astype(str)
    heroes = df.pivot(index="match_id", columns="col", values="hero_id")
    heroes = heroes.reindex(columns=R_COLS + D_COLS).dropna().astype("int16")
    meta = df.groupby("match_id")[PRO_META].first()
    return heroes.join(meta, how="inner").reset_index()


def collect_pro(api: OpenDota, since: str, page_rows: int = 50_000) -> None:
    """Pull every pro match since a date via explorer SQL, newest first, in pages."""
    cursor, total = 10**12, 0
    while True:
        rows = api.explorer(PRO_SQL.format(since=to_unix(since), cursor=cursor, limit=page_rows))
        if not rows:
            break
        full_page = len(rows) == page_rows
        if full_page:
            # The oldest match on a full page may be cut off; fetch it again on the next page.
            oldest = min(r["match_id"] for r in rows)
            rows = [r for r in rows if r["match_id"] != oldest]
            cursor = oldest + 1
        wide = _pro_rows_to_wide(rows)
        total = save_matches(wide, SOURCES["pro"])
        print(f"  +{len(wide)} matches (oldest {pd.to_datetime(wide.start_time.min(), unit='s'):%Y-%m-%d}), file has {total}")
        if not full_page:
            break
    print(f"pro matches saved: {total} -> {SOURCES['pro']}")


def _public_batch_to_rows(batch: list[dict]) -> list[dict]:
    rows = []
    for m in batch:
        r, d = m.get("radiant_team") or [], m.get("dire_team") or []
        if len(r) != 5 or len(d) != 5 or 0 in r or 0 in d:
            continue
        if m.get("game_mode") not in NORMAL_GAME_MODES or (m.get("duration") or 0) < MIN_DURATION:
            continue
        rows.append(
            {
                "match_id": m["match_id"],
                **{c: h for c, h in zip(R_COLS, r)},
                **{c: h for c, h in zip(D_COLS, d)},
                "start_time": m["start_time"],
                "duration": m["duration"],
                "radiant_win": m["radiant_win"],
                "game_mode": m["game_mode"],
                "avg_rank_tier": m.get("avg_rank_tier"),
            }
        )
    return rows


def collect_public(api: OpenDota, calls: int, min_rank: int | None, save_every: int = 50) -> None:
    """Page backwards through /publicMatches. Each call returns up to 100 matches."""
    cursor, rows, total = None, [], 0

    def flush():
        nonlocal rows, total
        if rows:
            df = pd.DataFrame(rows)
            df["patch"] = patch_for_times(df["start_time"])
            df[R_COLS + D_COLS] = df[R_COLS + D_COLS].astype("int16")
            total = save_matches(df, SOURCES["public"])
            rows = []

    try:
        for i in range(1, calls + 1):
            batch = api.public_matches(less_than_match_id=cursor, min_rank=min_rank)
            if not batch:
                break
            cursor = min(m["match_id"] for m in batch)
            rows.extend(_public_batch_to_rows(batch))
            if i % save_every == 0:
                flush()
                print(f"  call {i}/{calls}: file has {total} matches, quota left today: {api.remaining_day}")
    except (QuotaExhausted, KeyboardInterrupt) as e:
        print(f"stopping early: {e or 'interrupted'}")
    finally:
        flush()
    print(f"public matches saved: {total} -> {SOURCES['public']}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect Dota 2 match data from OpenDota")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("heroes", help="hero list + patch dates")
    pro = sub.add_parser("pro", help="pro matches via explorer SQL")
    pro.add_argument("--since", default="2025-01-01", help="YYYY-MM-DD")
    pub = sub.add_parser("public", help="public ranked matches")
    pub.add_argument("--calls", type=int, default=500, help="API calls to spend (100 matches each)")
    pub.add_argument("--min-rank", type=int, default=70, help="rank tier, e.g. 70=Divine, 80=Immortal")
    args = p.parse_args(argv)

    api = OpenDota()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "heroes":
        collect_heroes(api)
    elif args.cmd == "pro":
        collect_pro(api, args.since)
    else:
        if not PATCHES_PATH.exists():
            collect_heroes(api)
        collect_public(api, args.calls, args.min_rank)


if __name__ == "__main__":
    main()
