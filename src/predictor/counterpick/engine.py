"""Asilbek's counter-picker (the website) in Python, so the bot can host it.

Two parts, both following Asilbek's code exactly:
  pair_counts()     the same numbers as build_snapshot() in statistics.py (directional wins and
                    games per hero pair, same filters and data checks), but vectorised, so a
                    week of games (~1M) takes seconds instead of minutes
  rank_candidates() a line-by-line port of rankCandidates() in web/static/engine.js

If the website's rules change, change them here too (tests/test_counterpick_engine.py compares
pair_counts() with build_snapshot()).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import D_COLS, R_COLS
from .stats import N_HEROES, _pair_counts

MIN_MATCHES_PER_OPPONENT = 20  # engine.js: below this a candidate is "limited confidence"


@dataclass
class PairCounts:
    wins: np.ndarray  # (H, H): wins of row hero against column hero
    games: np.ndarray  # (H, H)
    eligible: int  # games that passed the filters
    oldest: int | None  # start_time range of those games
    newest: int | None


def _valid(df: pd.DataFrame) -> pd.Series:
    """Rows build_snapshot() would count: positive match id, 10 different positive hero ids,
    a True/False result."""
    heroes = df[R_COLS + D_COLS].apply(pd.to_numeric, errors="coerce")
    ints = heroes.notna().all(axis=1) & (heroes.fillna(0) % 1 == 0).all(axis=1) & (heroes.fillna(0) > 0).all(axis=1)
    distinct = heroes.nunique(axis=1) == 10
    mid = pd.to_numeric(df["match_id"], errors="coerce")
    good_id = mid.notna() & (mid > 0) & (mid.fillna(0) % 1 == 0)
    result_ok = df["radiant_win"].map(lambda v: isinstance(v, (bool, np.bool_)))
    return ints & distinct & good_id & result_ok


def pair_counts(matches: pd.DataFrame, rank_min: int | None = None, rank_max: int | None = None, patch: str | None = None) -> PairCounts:
    """Same counts as statistics.build_snapshot(matches, ..., rank_min, rank_max, patch)."""
    empty = PairCounts(np.zeros((N_HEROES, N_HEROES)), np.zeros((N_HEROES, N_HEROES)), 0, None, None)
    if matches.empty:
        return empty
    df = matches[_valid(matches)]
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
    )


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
