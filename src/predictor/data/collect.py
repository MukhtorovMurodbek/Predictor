"""Data collectors.

    python -m predictor.data.collect heroes
    python -m predictor.data.collect pro              (new matches only; --since 2023-01-01 to rebuild)
    python -m predictor.data.collect public --calls 500 --min-rank 70
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DATA_DIR, RAW_DIR
from . import (
    D_COLS, HEROES_PATH, LEAGUES_PATH, PATCHES_PATH, PLAYER_COLS, PLAYERS_PATH, R_COLS, SOURCES, TEAMS_PATH,
    patch_for_times, to_unix,
)
from .opendota import OpenDota, QuotaExhausted

# All Pick, Captains Mode, Random Draft, Single Draft, Captains Draft, Ranked All Pick.
# Leaves out Turbo (23), Ability Draft (18), All Random (5) and event modes.
NORMAL_GAME_MODES = {1, 2, 3, 4, 16, 22}
MIN_DURATION = 600  # seconds; shorter games are mostly abandons

PRO_SQL = """
SELECT m.match_id, m.start_time, m.duration, m.radiant_win, m.leagueid, l.tier,
       m.radiant_team_id, m.dire_team_id, mp.patch, pm.player_slot, pm.hero_id, pm.account_id
FROM matches m
JOIN match_patch mp USING (match_id)
JOIN player_matches pm USING (match_id)
LEFT JOIN leagues l USING (leagueid)
WHERE m.start_time >= {since} AND m.match_id < {cursor}
ORDER BY m.match_id DESC
LIMIT {limit}
"""
PRO_META = ["start_time", "duration", "radiant_win", "leagueid", "tier", "radiant_team_id", "dire_team_id", "patch"]
DEFAULT_PRO_SINCE = "2023-01-01"  # older history warms up team and player ratings


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
    df["pcol"] = side + "p" + df.groupby(["match_id", side]).cumcount().astype(str)
    df["account_id"] = df["account_id"].fillna(0)
    heroes = df.pivot(index="match_id", columns="col", values="hero_id")
    heroes = heroes.reindex(columns=R_COLS + D_COLS).dropna().astype("int16")
    players = df.pivot(index="match_id", columns="pcol", values="account_id")
    players = players.reindex(columns=PLAYER_COLS).fillna(0).astype("int64")
    meta = df.groupby("match_id")[PRO_META].first()
    return heroes.join(players, how="inner").join(meta, how="inner").reset_index()


def collect_teams_and_leagues(api: OpenDota) -> None:
    teams = pd.DataFrame(api.explorer("SELECT team_id, name, tag FROM teams"))
    teams.to_parquet(TEAMS_PATH, index=False)
    leagues = pd.DataFrame(api.explorer("SELECT leagueid, name, tier FROM leagues"))
    leagues.to_parquet(LEAGUES_PATH, index=False)
    players = pd.DataFrame(api.explorer("SELECT account_id, name FROM notable_players WHERE name IS NOT NULL"))
    players.to_parquet(PLAYERS_PATH, index=False)
    print(f"saved {len(teams)} teams, {len(leagues)} leagues, {len(players)} named players")


def _pro_since(since: str | None) -> int:
    """Explicit date, or continue from the newest saved match, or start from DEFAULT_PRO_SINCE."""
    if since:
        return to_unix(since)
    path = SOURCES["pro"]
    if path.exists():
        old = pd.read_parquet(path)
        if set(PLAYER_COLS) <= set(old.columns):
            return int(old["start_time"].max()) - 3 * 86400
        path.unlink()  # saved by an older version without player columns: rebuild it
    return to_unix(DEFAULT_PRO_SINCE)


def collect_pro(api: OpenDota, since: str | None = None, page_rows: int = 50_000) -> None:
    """Pull pro matches via explorer SQL, newest first, in pages. Without `since` it only
    fetches matches newer than the ones already saved."""
    cursor, total, start = 10**12, 0, _pro_since(since)
    print(f"fetching pro matches since {pd.to_datetime(start, unit='s'):%Y-%m-%d}")
    while True:
        rows = api.explorer(PRO_SQL.format(since=start, cursor=cursor, limit=page_rows))
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
    collect_teams_and_leagues(api)


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
    pro.add_argument("--since", help="YYYY-MM-DD (default: continue from the newest saved match, or 2023-01-01)")
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
