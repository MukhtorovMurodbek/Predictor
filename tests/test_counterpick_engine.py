import numpy as np
import pandas as pd
import pytest

from predictor.counterpick.engine import PairCounts, pair_counts, rank_candidates
from predictor.counterpick.statistics import build_snapshot
from predictor.data import D_COLS, R_COLS

HEROES = {i: f"Hero {i:02d}" for i in range(1, 21)}


def messy_matches(n=3000, seed=0) -> pd.DataFrame:
    """Random games incl. the cases build_snapshot() skips: duplicates, bad heroes, missing
    results and ranks, several patches."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        heroes = rng.choice(np.arange(1, 21), 10, replace=False)
        row = dict(zip(R_COLS + D_COLS, (int(h) for h in heroes)))
        row.update(match_id=1000 + i, radiant_win=bool(rng.random() < 0.5 + 0.2 * (heroes[0] == 1)),
                   avg_rank_tier=float(rng.choice([11, 25, 34, 45, 52, 61, 75, 80, np.nan])),
                   patch=str(rng.choice(["7.40", "7.41"])), start_time=1_790_000_000 + i)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)  # duplicate match ids
    df.loc[3, "d4"] = df.loc[3, "r0"]  # same hero twice in one game
    df["radiant_win"] = df["radiant_win"].astype(object)
    df.loc[7, "radiant_win"] = None  # missing result
    return df


def snapshot_arrays(snap: dict) -> tuple[np.ndarray, np.ndarray]:
    wins, games = np.zeros((160, 160)), np.zeros((160, 160))
    for p in snap["pairwise"]:
        wins[p["hero_id"], p["enemy_id"]] = p["wins"]
        games[p["hero_id"], p["enemy_id"]] = p["matches"]
    return wins, games


@pytest.mark.parametrize("rank_min,rank_max,patch", [(None, None, None), (40, 55, None), (None, 35, "7.41"), (80, 80, "7.40")])
def test_same_numbers_as_build_snapshot(rank_min, rank_max, patch):
    df = messy_matches()
    snap = build_snapshot(df, HEROES, rank_min=rank_min, rank_max=rank_max, patch=patch)
    ours = pair_counts(df, rank_min, rank_max, patch)
    wins, games = snapshot_arrays(snap)
    assert ours.eligible == snap["gamesEligible"]
    assert np.array_equal(ours.games, games) and np.array_equal(ours.wins, wins)
    assert (ours.oldest, ours.newest) == (snap["oldestTimestamp"], snap["newestTimestamp"])


def test_ranking_follows_engine_js_rules():
    wins, games = np.zeros((160, 160)), np.zeros((160, 160))
    # Hero 3: reliable, 60% vs both enemies. Hero 4: reliable, 55%. Hero 5: 90% but only 5 games
    # vs enemy 1 (not reliable). Hero 6: games vs only one enemy (less coverage).
    for h, rate, n in ((3, 0.60, 100), (4, 0.55, 100), (5, 0.90, 5)):
        for e in (1, 2):
            games[h, e], wins[h, e] = n, round(n * rate)
    games[6, 1], wins[6, 1] = 100, 80
    counts = PairCounts(wins, games, 1000, None, None)
    ranked = rank_candidates(HEROES, counts, enemies=[1, 2], allies=[4])
    names = [c["name"] for c in ranked]
    assert names == ["Hero 03", "Hero 05", "Hero 06"]  # 4 is an ally; reliable first, then coverage
    assert ranked[0]["reliable"] and not ranked[1]["reliable"]
    assert ranked[0]["score"] == pytest.approx(60.0)
    assert rank_candidates(HEROES, counts, enemies=[]) == []
