import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predictor.live import LiveModel
from predictor.live import bot as live_bot
from predictor.live.features import parse_events, state_at
from predictor.live.model import build_rows, hide_inputs, new_classifier
from predictor.pro import ProModel
from predictor.pro import bot as pro_bot
from predictor.pro.dataset import replay, training_mask
from predictor.pro.model import fit_classifiers

from test_bot import FakeMessage, assert_telegram_html
from test_pro import HEROES, TEAMS, synthetic_pro


def synthetic_timelines(df: pd.DataFrame, seed=0) -> pd.DataFrame:
    """The winner slowly pulls ahead in gold and takes towers."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in df.itertuples():
        s = 1 if m.radiant_win else -1
        n = int(rng.integers(25, 45))
        gold = [int(s * k * rng.uniform(50, 300) + rng.normal(0, 1500)) for k in range(n)]
        towers = ";".join(f"{int(rng.integers(600, n * 60))}:npc_dota_{'badguys' if s > 0 else 'goodguys'}_tower1_mid" for _ in range(3))
        rows.append({"match_id": m.match_id, "start_time": m.start_time, "duration": n * 60, "gold": gold, "xp": gold, "events": towers})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def models():
    df = synthetic_pro(1500)
    X, state = replay(df)
    use = training_mask(df)
    pregame, draft = fit_classifiers(X[use], df.loc[use, "radiant_win"].to_numpy().astype(int))
    pro = ProModel(state, pregame, draft, TEAMS, {}, HEROES, len(df))
    rows, y, _ = build_rows(df, X, synthetic_timelines(df))
    live = LiveModel(new_classifier().fit(hide_inputs(rows), y), len(df), "report", {"stats": {"acc_0": 0.6, "acc_10": 0.7, "acc_20": 0.8, "acc_30": 0.9}})
    return live, pro


def test_parse_events():
    evs = parse_events("805:npc_dota_badguys_tower1_top;900:npc_dota_goodguys_melee_rax_bot;1326:rosh3;96:fb130;x:junk")
    assert evs == [(805, "tower", 1), (900, "rax", -1), (1326, "rosh", -1), (96, "fb", -1)]
    s = state_at([0, 100, 200, 300, 400, 500, 600, 700], evs, 7)
    assert s["gold_adv"] == 700 and s["gold_adv_5min"] == 500 and s["first_blood"] == -1 and s["towers_diff"] == 0
    s = state_at(list(range(0, 3000, 100)), evs, 25)
    assert s["towers_diff"] == 1 and s["rax_diff"] == -1 and s["roshan_diff"] == -1


def test_gold_lead_matters(models):
    live, _ = models
    ahead = live.predict_radiant(None, {"minute": 30, "gold_adv": 10_000})
    behind = live.predict_radiant(None, {"minute": 30, "gold_adv": -10_000})
    assert ahead > 0.7 > 0.3 > behind


def run(handler, text):
    msg = FakeMessage()
    asyncio.run(handler(SimpleNamespace(effective_message=msg), SimpleNamespace(args=text.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies[0]


def test_bot(models, monkeypatch):
    live, pro = models
    monkeypatch.setattr(live_bot, "_model", lambda: live)
    monkeypatch.setattr(pro_bot, "_model", lambda: pro)
    ahead = run(live_bot.ingame, "alpha vs delta 30 +10k towers 3-1 rosh 1-0")
    assert "Win chance" in ahead and "towers +2" in ahead
    p_ahead = int(ahead.split("Win chance: <b>")[1].split("%")[0])
    p_behind = int(run(live_bot.ingame, "alpha vs delta 30 -10k").split("Win chance: <b>")[1].split("%")[0])
    assert p_ahead > p_behind
    assert "Write it like" in run(live_bot.ingame, "alpha vs delta")

    game = {"league_id": 1, "team_id_radiant": 1, "team_id_dire": 4, "team_name_radiant": "Team Alpha",
            "team_name_dire": "Delta Force", "game_time": 1500, "radiant_lead": 8000, "radiant_score": 20, "dire_score": 9,
            "players": [{"team": t, "account_id": 10 * (1 if t == 0 else 4) + k, "hero_id": 1 + k + 5 * t} for t in (0, 1) for k in range(5)]}
    monkeypatch.setattr(live_bot, "_fetch_live", lambda: [game, {"league_id": 0}])
    reply = run(live_bot.live_games, "")
    assert "Team Alpha" in reply and "25:00" in reply
    monkeypatch.setattr(live_bot, "_fetch_live", lambda: [])
    assert "No pro league games" in run(live_bot.live_games, "")
    assert_telegram_html(live_bot.help_text())
