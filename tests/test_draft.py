import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from predictor.draft import DraftModel
from predictor.draft import bot as draft_bot
from predictor.draft.features import CM_ORDER, parse_order
from predictor.draft.model import MatchRecord, fit_classifier, replay, topk_report

from test_bot import FakeMessage, assert_telegram_html

TEAMS = {1: ("Team Alpha", "ALP"), 2: ("Beta Squad", "BETA"), 3: ("Gamma", "GAM"), 4: ("Delta Force", "DF")}
VALID = np.arange(1, 51)
HEROES = {int(h): f"Hero {h}" for h in VALID}
ORDER = parse_order(CM_ORDER)


def pool(team):
    return list(range(10 * team - 9, 10 * team + 1))  # team 1: heroes 1-10, team 2: 11-20, ...


def synthetic_records(n=500, seed=0):
    """Each team picks from its own 10-hero pool and bans from the opponent's pool."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        a, b = (int(x) for x in rng.choice(list(TEAMS), 2, replace=False))
        first = int(rng.integers(2))  # side of team A: 0 radiant, 1 dire
        team_of = {"A": a, "B": b}
        used, actions, picks = set(), [], {0: [], 1: []}
        for is_pick, who in ORDER:
            source = pool(team_of[who]) if is_pick else pool(team_of["B" if who == "A" else "A"])
            free = [h for h in source if h not in used] or [h for h in VALID if h not in used]
            hero = int(rng.choice(free))
            used.add(hero)
            side = first if who == "A" else 1 - first
            actions.append((is_pick, side, hero))
            if is_pick:
                picks[side].append(hero)
        teams = (a, b) if first == 0 else (b, a)
        players = tuple(tuple(t * 100 + k for k in range(5)) for t in teams)
        records.append(MatchRecord(
            t=1_600_000_000 + i * 6 * 3600, teams=teams, players=players,
            picks_by_player=[(p, h) for s in (0, 1) for p, h in zip(players[s], picks[s])],
            radiant_win=bool(rng.random() < 0.5), actions=actions, cm=True,
            order=[(p, "A" if s == first else "B") for p, s, _ in actions],
        ))
    return records


@pytest.fixture(scope="module")
def model() -> DraftModel:
    records = synthetic_records()
    state, rows, test = replay(records, VALID, train_from=100, test_from=400, test_every=5)
    clf = fit_classifier(rows["X"], rows["y"])
    report, stats = topk_report(clf, test, 20)
    return DraftModel(state, clf, VALID, ORDER, TEAMS, HEROES, len(records), report, {"stats": stats})


def test_learns_team_pools(model):
    first_ban = model.predict_next(1, 2, [])  # Alpha bans first: should target Beta's heroes
    assert first_ban[0][0] in pool(2)
    bans = [11, 12, 1, 2, 13, 3, 4]  # the 7 bans before Alpha's first pick
    first_pick = model.predict_next(1, 2, bans)
    assert first_pick[0][0] in pool(1)
    assert sum(share for _, share in model.predict_next(1, 2, bans, top=50)) == pytest.approx(1.0)


def test_report_beats_meta(model):
    s = model.extra["stats"]
    assert s["model_picks"][5] > s["most popular in the meta_picks"][5]


def test_draft_errors(model):
    with pytest.raises(ValueError):
        model.predict_next(1, 2, [1, 1])
    with pytest.raises(ValueError):
        model.predict_next(1, 2, list(range(1, 25)))


def run(text):
    msg = FakeMessage()
    asyncio.run(draft_bot.next_action(SimpleNamespace(effective_message=msg), SimpleNamespace(args=text.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies[0]


def test_bot(model, monkeypatch):
    monkeypatch.setattr(draft_bot, "_model", lambda: model)
    reply = run("alpha vs beta")
    assert "Step 1 of 24" in reply and "Team Alpha bans" in reply
    reply = run("alpha vs beta: hero 11, hero 12, hero 1, hero 2, hero 13, hero 3, hero 4")
    assert "Step 8 of 24" in reply and "Team Alpha picks" in reply and "Bans Beta Squad" in reply
    assert "Write it like" in run("alpha")
    assert "same team" in run("alpha vs alpha")
    assert_telegram_html(draft_bot.help_text())
