import json

import pandas as pd
import pytest

from predictor.counterpick.statistics import build_snapshot


NAMES = {i: f"Hero {i}" for i in range(1, 13)}


def match(match_id, *, won=True, rank=70, patch="7.41", start=1000):
    return {
        "match_id": match_id,
        **{f"r{i}": i + 1 for i in range(5)},
        **{f"d{i}": i + 6 for i in range(5)},
        "radiant_win": won,
        "avg_rank_tier": rank,
        "patch": patch,
        "start_time": start,
    }


def pairs(snapshot):
    return {(p["hero_id"], p["enemy_id"]): p for p in snapshot["pairwise"]}


def test_directional_counts_filters_duplicates_and_evidence():
    rows = [
        match(1, won=True, start=1000),
        match(2, won=False, start=2000),
        match(2, won=False, start=2000),  # same game is not a new observation
        match(3, rank=40),
        match(4, rank=None),
        match(5, patch="7.40"),
    ]
    snapshot = build_snapshot(pd.DataFrame(rows), NAMES, rank_min=70, rank_max=75, patch="7.41")
    assert snapshot["gamesScanned"] == 6
    assert snapshot["gamesEligible"] == 2
    assert (snapshot["oldestTimestamp"], snapshot["newestTimestamp"]) == (1000, 2000)
    assert (snapshot["rankMin"], snapshot["rankMax"], snapshot["patch"]) == (70, 75, "7.41")
    lookup = pairs(snapshot)
    assert lookup[1, 6] == {"hero_id": 1, "enemy_id": 6, "wins": 1, "matches": 2}
    assert lookup[6, 1] == {"hero_id": 6, "enemy_id": 1, "wins": 1, "matches": 2}
    assert len(lookup) == 50
    assert (1, 2) not in lookup  # allies are not opponents
    assert (1, 11) not in lookup  # never observed: do not invent a 50% matchup
    json.dumps(snapshot)  # transport should contain only JSON-serializable values


def test_direction_changes_with_winner():
    lookup = pairs(build_snapshot(pd.DataFrame([match(1), match(2)]), NAMES))
    assert lookup[1, 6]["wins"] == 2
    assert lookup[6, 1]["wins"] == 0


def test_incomplete_duplicated_and_invalid_games_are_excluded():
    incomplete = match(1)
    incomplete["r0"] = None
    duplicate_hero = match(2)
    duplicate_hero["d0"] = 1
    bad_winner = match(3)
    bad_winner["radiant_win"] = None
    snapshot = build_snapshot(pd.DataFrame([incomplete, duplicate_hero, bad_winner]), NAMES)
    assert snapshot["gamesEligible"] == 0
    assert snapshot["pairwise"] == []
    assert snapshot["oldestTimestamp"] is None
    assert any("No valid matches" in warning for warning in snapshot["warnings"])


def test_unknown_rank_is_allowed_only_without_rank_filter():
    matches = pd.DataFrame([match(1, rank=None)])
    assert build_snapshot(matches, NAMES)["gamesEligible"] == 1
    assert build_snapshot(matches, NAMES, rank_min=10)["gamesEligible"] == 0


def test_missing_metadata_and_empty_dataset():
    assert build_snapshot(pd.DataFrame(), NAMES)["gamesEligible"] == 0
    rows = pd.DataFrame([match(1)]).drop(columns=["avg_rank_tier", "patch", "start_time"])
    snapshot = build_snapshot(rows, NAMES)
    assert snapshot["gamesEligible"] == 1 and snapshot["oldestTimestamp"] is None
    assert any("start times" in warning for warning in snapshot["warnings"])
    with pytest.raises(ValueError, match="avg_rank_tier"):
        build_snapshot(rows, NAMES, rank_min=70)
    with pytest.raises(ValueError, match="patch"):
        build_snapshot(rows, NAMES, patch="7.41")
    with pytest.raises(ValueError, match="rank_min"):
        build_snapshot(rows, NAMES, rank_min=80, rank_max=75)
