import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from predictor.pro import ProModel
from predictor.pro import bot as pro_bot
from predictor.pro.dataset import replay, training_mask
from predictor.pro.model import fit_classifiers
from predictor.tournament import bot as tournament_bot
from predictor.tournament.sim import _double_elim, _swiss, map_matrix, title_odds

from test_bot import FakeMessage, assert_telegram_html
from test_pro import HEROES, TEAMS, synthetic_pro


def ordered_matrix(n=16, spread=3.0):
    s = np.linspace(spread / 2, -spread / 2, n)
    return 1 / (1 + np.exp(-(s[:, None] - s[None, :])))


def test_odds_sum_to_one_and_follow_strength():
    odds = title_odds(ordered_matrix(), 3000)
    assert odds.sum() == pytest.approx(1.0)
    assert odds[0] > odds[7] > odds[15] and odds[0] < 0.9


def test_equal_teams_get_equal_chances():
    odds = title_odds(np.full((16, 16), 0.5), 6000, seed=1)
    assert np.allclose(odds, 1 / 16, atol=0.02)


def test_swiss_and_bracket_shapes():
    rng = np.random.default_rng(0)
    S = ordered_matrix()
    for n in (8, 12, 16, 20):
        q = _swiss(n, ordered_matrix(n), rng)
        assert len(q) == len(set(q)) == 8
    assert _double_elim(list(range(8)), np.ones((16, 16)) * np.triu(np.ones((16, 16)), 1), S, rng) == 0  # lower id always wins


@pytest.fixture(scope="module")
def pro():
    df = synthetic_pro(1500)
    X, state = replay(df)
    use = training_mask(df)
    pregame, draft = fit_classifiers(X[use], df.loc[use, "radiant_win"].to_numpy().astype(int))
    return ProModel(state, pregame, draft, TEAMS, {}, HEROES, len(df))


def test_needs_eight_teams():
    with pytest.raises(ValueError):
        title_odds(ordered_matrix(4), 10)


def test_map_matrix(pro):
    P = map_matrix(pro, [1, 2, 3, 4])
    assert np.allclose(P + P.T, 1.0)
    assert P[0, 3] > 0.6  # Alpha is the strongest synthetic team, Delta the weakest


def run(text):
    msg = FakeMessage()
    asyncio.run(tournament_bot.odds(SimpleNamespace(effective_message=msg), SimpleNamespace(args=text.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies[0]


def test_bot(pro, monkeypatch):
    monkeypatch.setattr(pro_bot, "_model", lambda: pro)
    monkeypatch.setattr(tournament_bot, "N_SIMS", 500)
    assert "at least 8" in run("alpha, beta, gamma, delta")
    assert "listed twice" in run("alpha, alpha, beta, gamma")
    assert "not enough active teams" in run("")  # the synthetic league has only 4 teams
    assert_telegram_html(tournament_bot.help_text())
