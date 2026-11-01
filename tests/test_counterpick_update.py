"""Backfilling raw history must not add the same game to daily totals twice."""

import time

import pandas as pd

from predictor.counterpick import stats, update
from predictor.data import R_COLS, D_COLS
from predictor.data import collect


def public_match(match_id, age_days, valid=True):
    return {
        "match_id": match_id,
        "start_time": int(time.time() - age_days * 86400),
        "duration": 1800,
        "radiant_win": True,
        "radiant_team": [1, 2, 3, 4, 5],
        "dire_team": [6, 7, 8, 9, 10],
        "game_mode": 1 if valid else 23,
        "avg_rank_tier": 40,
    }


def test_recover_pruned_history_resume_without_recount(tmp_path, monkeypatch):
    public_path = tmp_path / "public.parquet"
    monkeypatch.setitem(update.SOURCES, "public", public_path)
    monkeypatch.setattr(stats, "DAILY_DIR", tmp_path / "daily")
    heroes, patches = tmp_path / "heroes.json", tmp_path / "patches.json"
    heroes.touch()
    patches.touch()
    monkeypatch.setattr(update, "HEROES_PATH", heroes)
    monkeypatch.setattr(update, "PATCHES_PATH", patches)
    monkeypatch.setattr(collect, "patch_for_times", lambda times: pd.Series(["7.40"] * len(times), index=times.index))

    recent = public_match(1000, 1)
    saved = pd.DataFrame(collect._public_batch_to_rows([recent]))
    saved["patch"] = "7.40"
    collect.save_matches(saved, public_path)
    # Historical row 990 has already been included in these daily tables, but was
    # discarded from raw storage by the previous seven-day retention policy.
    recovered = pd.DataFrame(collect._public_batch_to_rows([public_match(990, 15)]))
    update.add_to_days(pd.concat([saved, recovered], ignore_index=True))
    update.save_state({"oldest_id": 800, "oldest_time": public_match(800, 85)["start_time"]})

    class FakeAPI:
        remaining_day = 1000
        requests = []
        pages = {
            1000: [public_match(990, 15)],
            990: [public_match(980, 30, valid=False)],  # no eligible rows
            980: [public_match(970, 61)],  # crossing boundary is pruned
            800: [public_match(790, 95)],  # regular daily backfill resumes
        }

        def public_matches(self, less_than_match_id=None, min_rank=None):
            self.requests.append(less_than_match_id)
            return self.pages.get(less_than_match_id, [])

    monkeypatch.setattr(update, "OpenDota", FakeAPI)

    # Forward uses one call, leaving exactly one call for backfill each hour.
    for expected_cursor in (990, 980, 970):
        before_calls = len(FakeAPI.requests)
        update.update(max_calls=2)
        assert len(FakeAPI.requests) - before_calls <= 2
        assert update.load_state()["raw_oldest_id"] == expected_cursor
    assert FakeAPI.requests == [None, 1000, None, 990, None, 980]
    stored = pd.read_parquet(public_path)
    assert sorted(stored["match_id"].tolist()) == [990, 1000]
    assert stored["match_id"].is_unique
    assert sum(stats.Counts.load(f).n_matches for f in stats.DAILY_DIR.glob("*.npz")) == 2

    update.update(max_calls=2)
    assert FakeAPI.requests[-2:] == [None, 800]
    assert update.load_state()["oldest_id"] == 790
    assert sum(stats.Counts.load(f).n_matches for f in stats.DAILY_DIR.glob("*.npz")) == 3


def test_backfill_waits_when_forward_uses_full_budget(tmp_path, monkeypatch):
    path = tmp_path / "public.parquet"
    monkeypatch.setitem(update.SOURCES, "public", path)
    monkeypatch.setattr(stats, "DAILY_DIR", tmp_path / "daily")
    heroes, patches = tmp_path / "heroes.json", tmp_path / "patches.json"
    heroes.touch()
    patches.touch()
    monkeypatch.setattr(update, "HEROES_PATH", heroes)
    monkeypatch.setattr(update, "PATCHES_PATH", patches)
    monkeypatch.setattr(collect, "patch_for_times", lambda times: pd.Series(["7.40"] * len(times), index=times.index))
    saved = pd.DataFrame(collect._public_batch_to_rows([public_match(1000, 1)]))
    saved["patch"] = "7.40"
    collect.save_matches(saved, path)
    update.add_to_days(saved)

    class FakeAPI:
        remaining_day = 1000
        requests = []

        def public_matches(self, less_than_match_id=None, min_rank=None):
            self.requests.append(less_than_match_id)
            return [public_match(1001, 0.5)]

    monkeypatch.setattr(update, "OpenDota", FakeAPI)
    update.update(max_calls=1)
    assert FakeAPI.requests == [None]
    assert update.load_state()["raw_oldest_id"] == 1000

