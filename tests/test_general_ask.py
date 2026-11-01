import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from predictor.bot.registry import MODELS
from predictor.general import ask as ask_mod
from predictor.general import bot as general_bot
from predictor.general import llm as llm_mod

from test_bot import FakeMessage, assert_telegram_html, model  # noqa: F401  (model is a fixture)


class ScriptedLLM:
    """Plays back fixed answers instead of calling Groq/Ollama, and records what it was sent."""

    label = "test model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        return self.replies.pop(0)


def tool_call(name, args, id_="call_1"):
    return {"content": None, "tool_calls": [{"id": id_, "type": "function",
                                            "function": {"name": name, "arguments": json.dumps({"args": args})}}]}


def ask(question, monkeypatch, llm):
    monkeypatch.setattr(general_bot, "_llm", lambda: llm)
    msg = FakeMessage()
    asyncio.run(general_bot.ask(SimpleNamespace(effective_message=msg), SimpleNamespace(args=question.split())))
    for r in msg.replies:
        assert_telegram_html(r)
    return msg.replies[0]


def test_every_model_command_is_a_tool_and_admin_is_not():
    names = [t["function"]["name"] for t in ask_mod.tools()]
    commands = {c for m in MODELS.values() if m.NAME != "general" for c, _, _ in m.COMMANDS}
    assert set(names) == commands | {"hero_abilities"} and len(names) == len(set(names))
    assert not {"collect", "train", "evaluate", "refresh", "replay", "ask"} & set(names)
    for t in ask_mod.tools():
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", t["function"]["name"])
        assert t["function"]["parameters"]["required"] == ["args"]
    pick = next(t for t in ask_mod.tools() if t["function"]["name"] == "pick")
    assert "/pick pa, pudge" in pick["function"]["description"]  # examples come from /counterpick's help


def test_ask_runs_the_command_the_model_picks(monkeypatch, model):
    llm = ScriptedLLM(tool_call("pick", "hero 2 | hero 3"), {"content": "Pick Hero 1: it beats Hero 2."})
    reply = ask("what should I pick against hero 2 with hero 3 on my team?", monkeypatch, llm)
    assert reply.startswith("Pick Hero 1: it beats Hero 2.")
    assert "<code>/pick hero 2 | hero 3</code>" in reply and "test model" in reply
    messages, tools = llm.seen[1]
    tool_result = messages[-1]
    assert tool_result["role"] == "tool" and "Best picks" in tool_result["content"] and "<b>" not in tool_result["content"]
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "pick"


def test_ask_must_answer_after_a_few_tool_rounds(monkeypatch, model):
    loops = [tool_call("counters", "hero 2", id_=f"c{i}") for i in range(ask_mod.MAX_ROUNDS)]
    llm = ScriptedLLM(*loops, {"content": "Hero 1 counters Hero 2."})
    assert ask("who counters hero 2", monkeypatch, llm).startswith("Hero 1 counters Hero 2.")
    assert llm.seen[-1][1] is None  # the last round offers no tools


def test_ask_without_a_language_model(monkeypatch):
    assert "groq_api" in ask("who counters pa", monkeypatch, None)
    assert_telegram_html(general_bot.help_text())


def test_ask_reports_an_unreachable_model(monkeypatch):
    class Down:
        label = "down"

        def chat(self, messages, tools=None):
            raise llm_mod.LLMError("can't reach the server")

    assert "didn't answer" in ask("who counters pa", monkeypatch, Down())


def test_llm_settings_from_env(tmp_path, monkeypatch):
    for key in ("GROQ_API", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_URL"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("bot_api = 1:x\n")
    assert llm_mod.from_env(env) is None
    env.write_text("groq_api = gsk_test\ngroq_model = some-model\n")
    llm = llm_mod.from_env(env)
    assert (llm.base_url, llm.model, llm.api_key) == (llm_mod.GROQ_URL, "some-model", "gsk_test")
    env.write_text("ollama_model = qwen3:4b\n")
    llm = llm_mod.from_env(env)
    assert (llm.base_url, llm.model, llm.api_key) == (llm_mod.OLLAMA_URL, "qwen3:4b", None)


def test_llm_sends_openai_style_requests(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json, headers=headers)
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    llm = llm_mod.LLM(llm_mod.GROQ_URL, "m", "key")
    assert llm.chat([{"role": "user", "content": "hey"}], tools=[{"type": "function"}]) == {"content": "hi"}
    assert sent["url"] == llm_mod.GROQ_URL + "/chat/completions" and sent["headers"]["Authorization"] == "Bearer key"
    assert sent["json"]["tool_choice"] == "auto" and sent["json"]["temperature"] == 0
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: SimpleNamespace(status_code=401, text="bad key"))
    with pytest.raises(llm_mod.LLMError):
        llm.chat([{"role": "user", "content": "hey"}])
