import time

import numpy as np
import pandas as pd

from predictor.counterpick import Counts, MatchupStats
from predictor.counterpick import stats as stats_mod
from predictor.counterpick import update as update_mod
from predictor.data import D_COLS, R_COLS, resolve_heroes

HEROES = {i: f"Hero {i}" for i in range(1, 21)}


def synthetic_matches(n=20_000, seed=0):
    """Random 5v5 drafts where hero 1 hard-counters hero 2 and heroes 3+4 combo well."""
    rng = np.random.default_rng(seed)
    drafts = np.array([rng.choice(np.arange(1, 21), 10, replace=False) for _ in range(n)])
    R, D = drafts[:, :5], drafts[:, 5:]
    logit = np.zeros(n)
    logit += 1.5 * ((R == 1).any(1) & (D == 2).any(1))
    logit -= 1.5 * ((D == 1).any(1) & (R == 2).any(1))
    logit += 1.0 * ((R == 3).any(1) & (R == 4).any(1))
    logit -= 1.0 * ((D == 3).any(1) & (D == 4).any(1))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return R, D, y


def as_frame(R, D, y, start=None) -> pd.DataFrame:
    df = pd.DataFrame(np.c_[R, D], columns=R_COLS + D_COLS)
    df["radiant_win"] = y.astype(bool)
    df["start_time"] = (start or int(time.time()) - 3600) + np.arange(len(y)) % 600
    df["match_id"] = np.arange(len(y)) + (start or 0)
    return df


def make_stats(n=20_000) -> MatchupStats:
    return MatchupStats(Counts.from_matches(as_frame(*synthetic_matches(n))), HEROES)


def test_finds_planted_counter_and_synergy():
    s = make_stats()
    assert s.vs_edge[1, 2] > 0.1
    assert s.vs_edge[1, 2] == np.nanmax(s.vs_edge[1:21, 2])
    assert s.with_edge[3, 4] > 0.1
    assert s.counters_for(2, top=1)["hero"][0] == "Hero 1"
    assert set(s.recommend(allies=[3], enemies=[2], top=2)["hero"]) == {"Hero 1", "Hero 4"}


def test_raw_counts_are_exact():
    R, D, y = np.array([[1, 2, 3, 4, 5]]), np.array([[6, 7, 8, 9, 10]]), np.array([1.0])
    c = Counts.from_matches(as_frame(R, D, y))
    assert c.vs_games[1, 6] == c.vs_games[6, 1] == 1
    assert c.vs_wins[1, 6] == 1 and c.vs_wins[6, 1] == 0
    assert c.with_games[1, 2] == 1 and c.with_games[1, 1] == 0
    assert c.games[1] == 1 and c.wins[6] == 0


def test_counts_add_up_and_roundtrip(tmp_path):
    df = as_frame(*synthetic_matches(2000))
    whole = Counts.from_matches(df)
    parts = Counts.from_matches(df.iloc[:700]) + Counts.from_matches(df.iloc[700:])
    assert np.allclose(parts.vs_games, whole.vs_games) and np.allclose(parts.with_wins, whole.with_wins)
    assert parts.n_matches == 2000
    whole.save(tmp_path / "x.npz")
    assert np.allclose(Counts.load(tmp_path / "x.npz").vs_wins, whole.vs_wins)


def test_daily_tables_window_and_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(stats_mod, "DAILY_DIR", tmp_path)
    now = int(time.time())
    recent = as_frame(*synthetic_matches(300), start=now - 3600)
    old = as_frame(*synthetic_matches(200, seed=1), start=now - 120 * 86400)  # outside the 90-day window
    df = pd.concat([recent, old], ignore_index=True)
    days = update_mod.rebuild_days(df)
    assert len(days) >= 2 and all((tmp_path / f"{d}.npz").exists() for d in days)
    stats = MatchupStats.from_daily(HEROES)
    assert stats.n_matches == 300
    assert len(update_mod.prune_raw(df)) == 300


def test_resolve_heroes():
    names = {1: "Anti-Mage", 44: "Phantom Assassin", 12: "Phantom Lancer", 5: "Crystal Maiden", 120: "Pangolier"}
    assert resolve_heroes("pa, crystal, anti-mage", names) == [44, 5, 1]
    assert resolve_heroes("am", names) == [1]
    try:
        resolve_heroes("phantom", names)
    except ValueError as e:
        assert "ambiguous" in str(e)
    else:
        raise AssertionError("expected ambiguity error")
