import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predictor.data import D_COLS, PLAYER_COLS, R_COLS
from predictor.pro import ProModel, series_win
from predictor.pro import bot as pro_bot
from predictor.pro.dataset import replay, training_mask
from predictor.pro.model import fit_classifiers

from test_bot import FakeMessage, assert_telegram_html

TEAMS = {1: ("Team Alpha", "ALP"), 2: ("Beta Squad", "BETA"), 3: ("Gamma", "GAM"), 4: ("Delta Force", "DF")}
STRENGTH = {1: 1.2, 2: 0.4, 3: -0.4, 4: -1.2}
HEROES = {i: f"Hero {i}" for i in range(1, 31)}


def synthetic_pro(n=2500, seed=0) -> pd.DataFrame:
    """A small league where team 1 is clearly best and team 4 clearly worst."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rt, dt = rng.choice(list(TEAMS), 2, replace=False)
        heroes = rng.choice(np.arange(1, 31), 10, replace=False)
        p = 1 / (1 + math.exp(-(STRENGTH[rt] - STRENGTH[dt])))
        row = {"match_id": i, "start_time": 1_600_000_000 + i * 6 * 3600, "radiant_team_id": float(rt),
               "dire_team_id": float(dt), "radiant_win": bool(rng.random() < p), "tier": "professional"}
        row.update(zip(R_COLS + D_COLS, heroes))
        row.update(zip(PLAYER_COLS, [rt * 10 + k for k in range(5)] + [dt * 10 + k for k in range(5)]))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model() -> ProModel:
    df = synthetic_pro()
    X, state = replay(df)
    use = training_mask(df)
    pregame, draft = fit_classifiers(X[use], df.loc[use, "radiant_win"].to_numpy().astype(int))
    return ProModel(state, pregame, draft, TEAMS, {10: "alpha_carry"}, HEROES, len(df))


def test_series_win():
    assert series_win(0.5, 3) == pytest.approx(0.5)
    assert series_win(0.6, 3) == pytest.approx(0.648)
    assert series_win(0.7, 1) == pytest.approx(0.7)
    assert series_win(0.6, 5) > series_win(0.6, 3) > 0.6


def test_features_have_no_gaps_and_ratings_order_teams(model):
    X, _ = replay(synthetic_pro(500))
    assert not X.isna().any().any()
    elos = [model.elo(t) for t in (1, 2, 3, 4)]
    assert elos == sorted(elos, reverse=True)


def test_first_match_features_know_nothing():
    X, _ = replay(synthetic_pro(50))
    assert X.iloc[0]["team_elo_diff"] == 0 and X.iloc[0]["h2h_games"] == 0


def test_predict_favours_stronger_team(model):
    p = model.predict(1, 4)
    assert p > 0.7
    assert model.predict(4, 1) == pytest.approx(1 - p, abs=0.02)
    heroes_a, heroes_b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert model.predict(1, 4, heroes_a, heroes_b) > 0.6
    with pytest.raises(ValueError):
        model.predict(1, 4, heroes_a, None)


def test_find_team(model):
    assert model.find_team("alpha") == 1
    assert model.find_team("BETA") == 2
    assert model.find_team("delta force") == 4
    with pytest.raises(ValueError):
        model.find_team("zzz")


def run(handler, text):
    msg = FakeMessage()
    asyncio.run(handler(SimpleNamespace(effective_message=msg), SimpleNamespace(args=text.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies


def test_bot_commands(model, monkeypatch):
    monkeypatch.setattr(pro_bot, "_model", lambda: model)
    reply = run(pro_bot.predict, "alpha vs delta")[0]
    assert "Map:" in reply and "Bo3" in reply and "Favourite: <b>Team Alpha</b>" in reply
    reply = run(pro_bot.predict, "alpha: hero 1, hero 2, hero 3, hero 4, hero 5 vs gamma: hero 6, hero 7, hero 8, hero 9, hero 10")[0]
    assert "After-draft" in reply
    assert "exactly 5" in run(pro_bot.predict, "alpha: hero 1 vs gamma: hero 6")[0]
    assert "Write it like" in run(pro_bot.predict, "alpha")[0]
    assert "Team Alpha" in run(pro_bot.team, "alpha")[0]
    assert "1. <b>Team Alpha</b>" in run(pro_bot.top, "")[0]
    assert_telegram_html(pro_bot.help_text())
