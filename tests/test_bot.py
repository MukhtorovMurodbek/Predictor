import asyncio
import re
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from predictor.bot import auth as auth_mod
from predictor.bot.app import all_commands, help_text, start_text
from predictor.bot.auth import Auth
from predictor.bot.config import load_config
from predictor.bot.registry import MODELS
from predictor.counterpick import CounterModel
from predictor.counterpick import bot as cp_bot

from test_counterpick import HEROES, synthetic_matches

TELEGRAM_TAGS = {"b", "i", "code", "pre"}


class TagChecker(HTMLParser):
    """Telegram rejects messages whose HTML tags are unknown or unbalanced."""

    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        assert tag in TELEGRAM_TAGS, f"unsupported tag <{tag}>"
        self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag, f"unbalanced </{tag}>"


def assert_telegram_html(text):
    checker = TagChecker()
    checker.feed(text)
    assert not checker.stack, f"unclosed tags {checker.stack}"
    assert len(text) < 4096


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def run(handler, args):
    msg = FakeMessage()
    update = SimpleNamespace(effective_message=msg, effective_user=None, effective_chat=None)
    asyncio.run(handler(update, SimpleNamespace(args=args)))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies


@pytest.fixture
def model(monkeypatch):
    R, D, y = synthetic_matches(5000)
    m = CounterModel.fit(R, D, y, HEROES)
    monkeypatch.setattr(cp_bot, "_model", lambda: m)
    return m


def test_config_parses_env(tmp_path, monkeypatch):
    for key in ("BOT_API", "USERNAMES", "BOT_USERNAME"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("bot_api = 123:abc\nbot_username = @MyBot\nusernames = Alice, @bob ,carol\n")
    cfg = load_config(env)
    assert cfg.token == "123:abc"
    assert cfg.username == "MyBot"
    assert cfg.allowed == {"alice", "bob", "carol"}


def test_auth_locks_username_to_first_id(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_mod, "USERS_PATH", tmp_path / "users.json")
    auth = Auth(frozenset({"alice"}))
    alice = SimpleNamespace(id=1, username="Alice")
    assert auth.check(alice)
    assert not auth.check(SimpleNamespace(id=2, username="alice"))  # someone grabbed the name
    assert auth.check(SimpleNamespace(id=1, username="alice_new"))  # alice renamed herself
    assert not auth.check(SimpleNamespace(id=3, username="mallory"))
    assert not auth.check(None)
    assert Auth(frozenset({"alice"})).check(alice)  # remembered across restarts


def test_help_text_and_command_names():
    assert_telegram_html(help_text())
    assert_telegram_html(start_text("Tester <3"))
    names = [name for name, _, _ in all_commands()]
    assert len(names) == len(set(names))
    for name, _, desc in all_commands():
        assert re.fullmatch(r"[a-z0-9_]{1,32}", name)
        assert 3 <= len(desc) <= 256
    for m in MODELS.values():
        assert m.NAME in names
        assert_telegram_html(m.help_text())


def test_pick(model):
    (reply,) = run(cp_bot.pick, "hero 2 | hero 3".split())
    assert "Best picks" in reply
    assert "Hero 1" in reply and "Hero 4" in reply  # the planted counter and synergy


def test_pick_errors(model):
    assert "Tell me the enemy heroes" in run(cp_bot.pick, [])[0]
    assert "unknown hero" in run(cp_bot.pick, ["nobody"])[0]
    assert "twice" in run(cp_bot.pick, "hero 2 | hero 2".split())[0]


def test_counters_and_hero(model):
    assert "Hero 1" in run(cp_bot.counters, ["hero", "2"])[0]
    reply = run(cp_bot.hero, ["hero", "1"])[0]
    assert "Win rate" in reply and "Strong vs" in reply


def test_no_model_message(monkeypatch):
    monkeypatch.setattr(cp_bot, "_model", lambda: None)
    assert "/train counterpick" in run(cp_bot.pick, ["pa"])[0]
