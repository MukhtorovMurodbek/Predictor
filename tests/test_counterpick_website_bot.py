import asyncio
from types import SimpleNamespace

import pytest

from predictor.counterpick import website_bot
from predictor.counterpick.engine import pair_counts

from test_bot import FakeMessage, assert_telegram_html
from test_counterpick_engine import HEROES, messy_matches


@pytest.mark.parametrize("text,expected", [
    ("pa rank archon", (40, 45, "Archon")),
    ("pa rank crusader-legend", (30, 55, "Crusader–Legend")),
    ("pa rank imm", (80, 80, "Immortal")),
    ("pa rank 52", (52, 52, "52")),
    ("pa, pudge", (None, None, "all ranks")),
])
def test_parse_rank(text, expected):
    assert website_bot.parse_rank(text) == expected


def test_parse_rank_errors():
    for text in ("pa rank wizard", "pa rank 99"):
        with pytest.raises(ValueError):
            website_bot.parse_rank(text)


def run(text):
    msg = FakeMessage()
    asyncio.run(website_bot.apick(SimpleNamespace(effective_message=msg), SimpleNamespace(args=text.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies[0]


def test_apick(monkeypatch):
    df = messy_matches()
    monkeypatch.setattr(website_bot, "load_heroes", lambda: HEROES)
    monkeypatch.setattr(website_bot, "_counts", lambda lo, hi, patch: pair_counts(df, lo, hi, patch))
    reply = run("hero 02, hero 03 | hero 04")
    assert "Best picks" in reply and "Left out: Hero 04" in reply and "vs Hero 02" in reply
    first = reply.split("Best picks")[1].split("\n")[1]
    assert first.startswith("1. <b>Hero ") and "% · " in first and "games" in first
    assert "Archon" in run("hero 02 rank archon patch 7.41")
    assert "unknown rank" in run("hero 02 rank wizard")
    assert "Tell me the enemy heroes" in run("rank archon")
    assert_telegram_html(website_bot.help_text())
