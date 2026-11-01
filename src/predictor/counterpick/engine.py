"""Asilbek's counter-picker (the website) in Python, so the bot can host it.

Following Asilbek's code exactly:
  pair_counts()     the same numbers as build_snapshot() in statistics.py (directional wins and
                    games per hero pair, same filters and data checks), but vectorised
  window_counts()   the same numbers again, but from the daily tables of the last N days
                    (every public game OpenDota lists, ~550k a day), which /apick and the
                    website use; rank filters work in whole medals
  snapshot()        window_counts() in the website's /api/sample format (build_snapshot's)
  rank_candidates() a line-by-line port of rankCandidates() in web/static/engine.js

If the website's rules change, change them here too (tests/test_counterpick_engine.py compares
pair_counts() and window_counts() with build_snapshot()).
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data import D_COLS, R_COLS
from . import stats
from .stats import N_HEROES, DayTable, _pair_counts, valid_rows

MIN_MATCHES_PER_OPPONENT = 20  # engine.js: below this a candidate is "limited confidence"
APICK_DAYS = 30  # Asilbek: about a month of games (he asked for "from October 1" on November 1)


@dataclass
class PairCounts:
    wins: np.ndarray  # (H, H): wins of row hero against column hero
    games: np.ndarray  # (H, H)
    eligible: int  # games that passed the filters
    oldest: int | None  # start_time range of those games
    newest: int | None
    scanned: int = 0  # games looked at before the rank and patch filters


def pair_counts(matches: pd.DataFrame, rank_min: int | None = None, rank_max: int | None = None, patch: str | None = None) -> PairCounts:
    """Same counts as statistics.build_snapshot(matches, ..., rank_min, rank_max, patch)."""
    empty = PairCounts(np.zeros((N_HEROES, N_HEROES)), np.zeros((N_HEROES, N_HEROES)), 0, None, None, len(matches))
    if matches.empty:
        return empty
    df = matches[valid_rows(matches)]
    df = df.drop_duplicates("match_id", keep="first")  # build_snapshot counts a match id once
    if rank_min is not None or rank_max is not None:
        rank = pd.to_numeric(df["avg_rank_tier"], errors="coerce")
        keep = rank.notna() & (rank > 0) & (rank.fillna(0) % 1 == 0)
        if rank_min is not None:
            keep &= rank >= rank_min
        if rank_max is not None:
            keep &= rank <= rank_max
        df = df[keep]
    if patch is not None:
        df = df[df["patch"].map(lambda v: isinstance(v, str) and v == patch)]
    if df.empty:
        return empty
    R = df[R_COLS].to_numpy(np.int64)
    D = df[D_COLS].to_numpy(np.int64)
    y = df["radiant_win"].to_numpy(dtype=float)
    g1, w1 = _pair_counts(R, D, y)
    g2, w2 = _pair_counts(D, R, 1.0 - y)
    times = pd.to_numeric(df.get("start_time"), errors="coerce") if "start_time" in df else pd.Series(dtype=float)
    times = times[(times > 0) & (times % 1 == 0)]
    return PairCounts(
        wins=w1 + w2,
        games=g1 + g2,
        eligible=len(df),
        oldest=int(times.min()) if len(times) else None,
        newest=int(times.max()) if len(times) else None,
        scanned=len(matches),
    )


def medals_for(rank_min: int | None, rank_max: int | None) -> set[int] | None:
    """The medals a rank range covers (None = every game, incl. unranked). The tables keep
    medals, not single tiers, so the range must hold whole medals: 40-45 is Archon (tiers
    41-45; tier 40 doesn't exist), 30-55 is Crusader to Legend, 80 is Immortal."""
    if rank_min is None and rank_max is None:
        return None
    lo, hi = 10 if rank_min is None else rank_min, 80 if rank_max is None else rank_max
    medals = set()
    for m in range(1, 9):
        first, last = (80, 80) if m == 8 else (m * 10 + 1, m * 10 + 5)  # tiers that exist
        if lo <= first and last <= hi:
            medals.add(m)
        elif first <= hi and lo <= last:
            raise ValueError("rank filters work in whole medals, e.g. 40-45 for Archon or 30-55 for Crusader to Legend")
    return medals


def window_counts(rank_min: int | None = None, rank_max: int | None = None, patch: str | None = None,
                  days: int = APICK_DAYS, today: datetime | None = None) -> PairCounts:
    """Same counts as pair_counts() over the games of the last `days` days, from the daily
    tables (data/counterpick/YYYY-MM-DD.npz) instead of a table of games."""
    medals = medals_for(rank_min, rank_max)
    picked, scanned = stats.Slice.empty(), 0
    for path in stats.daily_files(days, today):
        table = DayTable.load(path)
        scanned += table.n_matches
        picked = picked + table.select(medals, patch)
    return PairCounts(wins=picked.wins, games=picked.games, eligible=picked.n,
                      oldest=picked.first, newest=picked.last, scanned=scanned)


def snapshot(hero_names: dict[int, str], rank_min: int | None = None, rank_max: int | None = None,
             patch: str | None = None, days: int = APICK_DAYS) -> dict:
    """The website's /api/sample answer (the fields of statistics.build_snapshot) from the
    daily tables. Raises FileNotFoundError when there are no tables yet."""
    if not stats.daily_files(days):
        raise FileNotFoundError("no counter-picker tables yet, run: python -m predictor.counterpick.update")
    c = window_counts(rank_min, rank_max, patch, days)
    pairs = np.argwhere(c.games > 0)
    return {
        "heroes": [{"id": int(h), "name": name} for h, name in sorted(hero_names.items())],
        "pairwise": [{"hero_id": int(a), "enemy_id": int(b), "wins": int(c.wins[a, b]), "matches": int(c.games[a, b])}
                     for a, b in pairs],
        "gamesScanned": c.scanned,
        "gamesEligible": c.eligible,
        "oldestTimestamp": c.oldest,
        "newestTimestamp": c.newest,
        "rankMin": rank_min,
        "rankMax": rank_max,
        "patch": patch,
        "warnings": [
            f"Every ranked public match OpenDota lists from the last {days} days (normal modes, 10+ minutes), "
            "counted by OpenDota's database and refreshed every hour; the newest 3-7 hours are not in yet.",
            "Rank is the match average, not an individual player's rank.",
        ] + ([] if c.eligible else ["No valid matches met the selected filters."])
          + (["Patch labels come from the patch release dates."] if patch is not None else []),
        "fetchedAt": None,
        "aggregatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "OpenDota public matches, counted into daily tables by the bot",
    }


def rank_candidates(hero_names: dict[int, str], counts: PairCounts, enemies, allies=(), bans=()) -> list[dict]:
    """Port of rankCandidates() in web/static/engine.js.

    Score = average of the candidate's win rate (%) against each enemy it has games against;
    an unseen matchup is never treated as 50%. Order: reliable candidates first (games against
    every enemy, at least 20 each), then more enemies covered, then (if not reliable) more
    games, then higher score, then more games, then name.
    """
    enemy_ids = list(dict.fromkeys(e for e in enemies if e))
    if not enemy_ids:
        return []
    unavailable = set(enemy_ids) | set(allies) | set(bans)
    out = []
    for hero_id, name in sorted(hero_names.items(), key=lambda kv: kv[1]):  # website sorts heroes by name
        if hero_id in unavailable:
            continue
        per_enemy = []
        for e in enemy_ids:
            m = int(counts.games[hero_id, e])
            w = int(counts.wins[hero_id, e])
            per_enemy.append({"enemy_id": e, "wins": w, "matches": m, "win_rate": w / m * 100 if m else None})
        observed = [p for p in per_enemy if p["matches"] > 0]
        if not observed:
            continue
        total = sum(p["matches"] for p in observed)
        reliable = len(observed) == len(enemy_ids) and all(p["matches"] >= MIN_MATCHES_PER_OPPONENT for p in per_enemy)
        out.append({
            "hero_id": hero_id,
            "name": name,
            "score": sum(p["win_rate"] for p in observed) / len(observed),
            "per_enemy": per_enemy,
            "coverage": len(observed),
            "enemy_count": len(enemy_ids),
            "total_matches": total,
            "reliable": reliable,
        })
    out.sort(key=lambda c: (
        not c["reliable"],
        -c["coverage"],
        0 if c["reliable"] else -c["total_matches"],
        -c["score"],
        -c["total_matches"],
        c["name"],
    ))
    return out
