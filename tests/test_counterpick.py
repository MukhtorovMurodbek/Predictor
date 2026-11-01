import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from predictor.counterpick import Counts, MatchupStats
from predictor.counterpick import stats as stats_mod
from predictor.counterpick import update as update_mod
from predictor.data import D_COLS, R_COLS, resolve_heroes
from predictor.data.collect import MIN_DURATION, NORMAL_GAME_MODES
from predictor.data.opendota import ExplorerError

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


def sql_answer(df: pd.DataFrame, t0: int, t1: int) -> list[dict]:
    """What OpenDota's explorer answers to update.SQL for these games (a test double, written
    straight from the SQL, so it checks table_from_rows and the update loop)."""
    acc, span = defaultdict(lambda: [0, 0]), {}
    for m in df[(df.start_time >= t0) & (df.start_time < t1)].itertuples():
        r, d = [getattr(m, c) for c in R_COLS], [getattr(m, c) for c in D_COLS]
        if (m.game_mode not in NORMAL_GAME_MODES or m.duration < MIN_DURATION or not m.avg_rank_tier >= 10
                or m.radiant_win is None or len(set(r + d)) != 10 or min(r + d) <= 0):
            continue
        tier = int(round(m.avg_rank_tier))
        k = tier // 10 if tier == 80 or (tier < 80 and tier % 10 <= 5) else 0
        w = bool(m.radiant_win)
        for a in r:
            for b in d:
                acc[(0, k, a, b)][0] += 1
                acc[(0, k, a, b)][1] += w
        for team, won, t in ((r, w, 1), (d, not w, 2)):
            for a in team:
                for b in team:
                    if a < b:
                        acc[(t, 0, a, b)][0] += 1
                        acc[(t, 0, a, b)][1] += won
        lo, hi, n = span.get(k, (m.start_time, m.start_time, 0))
        span[k] = (min(lo, m.start_time), max(hi, m.start_time), n + 1)
    grouped = defaultdict(lambda: ([], [], []))
    for (t, k, a, b), (g, v) in acc.items():
        for values, x in zip(grouped[(t, k, a)], (b, g, v)):
            values.append(int(x))
    out = [{"t": t, "k": k, "a": int(a), "b": b, "g": g, "v": v} for (t, k, a), (b, g, v) in grouped.items()]
    for k, (lo, hi, n) in span.items():
        out += [{"t": 3, "k": k, "a": 0, "b": [int(lo)], "g": [n], "v": [0]},
                {"t": 4, "k": k, "a": 0, "b": [int(hi)], "g": [0], "v": [0]}]
    return out


class FakeExplorer:
    """Answers update.SQL from a table of games; windows longer than `max_window` time out."""

    def __init__(self, df: pd.DataFrame, max_window: int | None = None):
        self.df, self.max_window, self.queries, self.remaining_day = df, max_window, [], 3000

    def explorer(self, sql: str, tries: int = 3) -> list[dict]:
        t0, t1 = map(int, re.search(r"start_time >= (\d+) AND start_time < (\d+)", sql).groups())
        self.queries.append((t0, t1))
        if self.max_window and t1 - t0 > self.max_window:
            raise ExplorerError("explorer error: Error: Query read timeout")
        return sql_answer(self.df, t0, t1)


def public_games(n: int, start: int, end: int, seed: int = 0) -> pd.DataFrame:
    """Random public games between two times, incl. some the counting must skip."""
    rng = np.random.default_rng(seed)
    df = as_frame(*synthetic_matches(n, seed))
    df["start_time"] = rng.integers(start, end, n)
    df["match_id"] = 5000 + np.arange(n)
    df["game_mode"] = rng.choice([22, 22, 22, 2, 23], n)  # 23 = turbo: skipped
    df["duration"] = rng.choice([1800, 2400, 300], n, p=[0.5, 0.45, 0.05])  # 300 s: skipped
    df["avg_rank_tier"] = rng.choice([11.0, 25.0, 34.0, 45.0, 52.0, 61.0, 75.0, 80.0, np.nan, 5.0], n)
    df.loc[:4, "d4"] = df.loc[:4, "r0"]  # broken games with the same hero twice: skipped
    return df


def test_update_counts_every_game_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr(stats_mod, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(update_mod, "WINDOW_DAYS", 4)
    patch_start = int(time.time()) - 2 * 86400 + 5 * 3600  # a new patch, mid-window
    patches = tmp_path / "patches.json"
    patches.write_text(json.dumps([{"name": "7.40", "date": "2026-01-01T00:00:00Z"},
                                   {"name": "7.41", "date": datetime.fromtimestamp(patch_start, timezone.utc).isoformat()}]))
    monkeypatch.setattr(update_mod, "PATCHES_PATH", patches)
    monkeypatch.setattr(update_mod, "HEROES_PATH", patches)
    old = tmp_path / "2026-10-31.npz"  # a table from the old sampled collector
    Counts.from_matches(as_frame(*synthetic_matches(10))).save(old)
    update_mod.save_state({"oldest_id": 1, "oldest_time": 1})

    now = int(time.time())
    games = public_games(4000, now - 6 * 86400, now)
    api = FakeExplorer(games, max_window=2 * 3600)  # 4-hour blocks time out: split in two
    first = update_mod.update(backfill_blocks=8, api=api, now=now)
    assert "moved 2 old file(s)" in first and (tmp_path / "old" / old.name).exists()
    later = now + 5 * 3600
    update_mod.update(backfill_blocks=10_000, api=api, now=later)
    asked = len(api.queries)
    update_mod.update(backfill_blocks=10_000, api=api, now=later)
    assert len(api.queries) == asked  # nothing left to count
    state = update_mod.load_state()
    # The bot is stopped after a table was saved but before state.json: the cursors are a
    # block behind. The next run must skip those blocks, not count them twice.
    update_mod.save_state({"counted_from": state["counted_from"] + update_mod.BLOCK,
                           "counted_until": state["counted_until"] - update_mod.BLOCK})
    update_mod.update(backfill_blocks=10_000, api=api, now=later)
    assert len(api.queries) == asked and update_mod.load_state() == state
    window_start = (later // 86400 - 4) * 86400
    ready = (later - update_mod.LAG) // update_mod.BLOCK * update_mod.BLOCK
    assert (state["counted_from"], state["counted_until"]) == (window_start, ready)
    counted = sorted(q for q in api.queries if q[1] - q[0] <= 2 * 3600)
    assert all(a[1] <= b[0] for a, b in zip(counted, counted[1:]))  # counted pieces never overlap

    window = games[(games.start_time >= window_start) & (games.start_time < ready)]
    ok = window[window.game_mode.isin(NORMAL_GAME_MODES) & (window.duration >= MIN_DURATION) & (window.avg_rank_tier >= 10)]
    ok = ok[stats_mod.valid_rows(ok)]
    expected = Counts.from_matches(ok)
    got = MatchupStats.from_daily(HEROES).counts
    assert got.n_matches == len(ok) == expected.n_matches
    for field in ("games", "wins", "vs_games", "vs_wins", "with_games", "with_wins"):
        assert np.array_equal(getattr(got, field), getattr(expected, field)), field
    tables = [stats_mod.DayTable.load(p) for p in tmp_path.glob("*.npz")]
    assert sum(t.select(patch="7.41").n for t in tables) == (ok.start_time >= patch_start).sum() > 0


def test_day_table_adds_up_and_roundtrips(tmp_path):
    df = public_games(1500, 1_790_000_000, 1_790_080_000)
    df["patch"] = "7.41"
    whole = stats_mod.DayTable.from_matches(df)
    parts = stats_mod.DayTable.from_matches(df.iloc[:600]) + stats_mod.DayTable.from_matches(df.iloc[600:])
    whole.save(tmp_path / "d.npz")
    back = stats_mod.DayTable.load(tmp_path / "d.npz")
    for t in (parts, back):
        assert set(t.slices) == set(whole.slices)
        for key, s in whole.slices.items():
            assert np.array_equal(t.slices[key].games, s.games)
            assert (t.slices[key].n, t.slices[key].first, t.slices[key].last) == (s.n, s.first, s.last)
        assert np.array_equal(t.with_wins, whole.with_wins)
    c = whole.counts()
    valid = df[stats_mod.valid_rows(df)]
    assert c.n_matches == whole.n_matches == len(valid)
    assert np.array_equal(c.games, Counts.from_matches(valid).games)


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
