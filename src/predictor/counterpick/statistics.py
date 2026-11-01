"""Build observable hero matchup counts from locally collected public matches.

This is an aggregation, not a fitted model. Matchup rates describe the matches in
the selected dataset; they do not predict a full draft's chance of winning.
"""

from collections import defaultdict
from datetime import datetime, timezone
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ..data import D_COLS, R_COLS


def _positive_integer(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    if not math.isfinite(value) or value <= 0 or int(value) != value:
        return None
    return int(value)


def _rank_bound(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def build_snapshot(
    matches: pd.DataFrame,
    hero_names: dict[int, str],
    *,
    rank_min: int | None = None,
    rank_max: int | None = None,
    patch: str | None = None,
) -> dict:
    """Aggregate directional wins and games, filtering on stored match metadata.

    Rank refers to a match's average rank; patch values are those already stored
    with matches. Missing observations are omitted rather than set to 50%.
    Duplicate match IDs count once. An empty input can have no columns.
    """
    low = _rank_bound(rank_min, "rank_min")
    high = _rank_bound(rank_max, "rank_max")
    if low is not None and high is not None and low > high:
        raise ValueError("rank_min must not exceed rank_max")
    if patch is not None and (not isinstance(patch, str) or not patch.strip()):
        raise ValueError("patch must be a non-empty string or None")
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")

    result = {
        "heroes": [{"id": int(h), "name": name} for h, name in sorted(hero_names.items())],
        "pairwise": [],
        "gamesScanned": int(len(matches)),
        "gamesEligible": 0,
        "oldestTimestamp": None,
        "newestTimestamp": None,
        "rankMin": low,
        "rankMax": high,
        "patch": patch,
        "warnings": [
            "OpenDota public matches are sampled; the stored dataset may include multiple game modes and lobby types.",
            "Rank is the match average, not an individual player's rank.",
            "The source dataset's collection time is not available; aggregation time is shown separately.",
        ],
        "fetchedAt": None,
        "aggregatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "OpenDota locally collected public-match dataset",
    }
    if matches.empty:
        result["warnings"].append("The dataset is empty.")
        return result

    required = {"match_id", "radiant_win", *R_COLS, *D_COLS}
    if low is not None or high is not None:
        required.add("avg_rank_tier")
    if patch is not None:
        required.add("patch")
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError("Missing required match columns: " + ", ".join(missing))
    if "start_time" not in matches.columns:
        result["warnings"].append("Match start times are unavailable.")

    counts = defaultdict(lambda: [0, 0])
    seen = set()
    columns = ["match_id", *R_COLS, *D_COLS, "radiant_win", "avg_rank_tier", "patch", "start_time"]
    for row in matches.reindex(columns=columns).itertuples(index=False, name=None):
        match_id = _positive_integer(row[0])
        radiant = [_positive_integer(value) for value in row[1:6]]
        dire = [_positive_integer(value) for value in row[6:11]]
        winner = row[11]
        if (match_id is None or match_id in seen or any(h is None for h in radiant + dire)
                or len(set(radiant + dire)) != 10 or not isinstance(winner, (bool, np.bool_))):
            continue
        # Deduplicate before selecting metadata to avoid counting conflicting records twice.
        seen.add(match_id)
        rank = _positive_integer(row[12])
        if (low is not None or high is not None) and (
            rank is None or (low is not None and rank < low) or (high is not None and rank > high)
        ):
            continue
        if patch is not None and (not isinstance(row[13], str) or row[13] != patch):
            continue
        result["gamesEligible"] += 1
        timestamp = _positive_integer(row[14])
        if timestamp is not None:
            oldest, newest = result["oldestTimestamp"], result["newestTimestamp"]
            result["oldestTimestamp"] = timestamp if oldest is None else min(oldest, timestamp)
            result["newestTimestamp"] = timestamp if newest is None else max(newest, timestamp)
        for r in radiant:
            for d in dire:
                for key, won in (((r, d), winner), ((d, r), not winner)):
                    counts[key][0] += int(won)
                    counts[key][1] += 1
    result["pairwise"] = [
        {"hero_id": a, "enemy_id": b, "wins": wins, "matches": games}
        for (a, b), (wins, games) in sorted(counts.items())
    ]
    if not result["gamesEligible"]:
        result["warnings"].append("No valid matches met the selected filters.")
    if patch is not None:
        result["warnings"].append("Patch labels come from the stored dataset and depend on its patch-date metadata.")
    return result
