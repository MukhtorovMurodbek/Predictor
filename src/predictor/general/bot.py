"""Telegram commands for the general predictor (phase 6): /ask in plain English.

A language model (Groq in the cloud, or Ollama on this laptop; see llm.py) reads the question
and picks the right command of the other models; the numbers always come from those models
(ask.py). Registered in predictor/bot/registry.py.
"""

import asyncio
import html

from telegram import Update
from telegram.ext import ContextTypes

from .. import ROOT
from .ask import answer
from .llm import LLMError, describe, from_env

NAME = "general"
TITLE = "General predictor"
SUMMARY = "ask anything in plain English; it picks the right model"

SETUP = """The general predictor needs a language model. Pick one and add it to <code>.env</code> on the bot's laptop:
• <b>Groq</b> (free, in the cloud): make a key at console.groq.com, then add <code>groq_api = gsk_...</code>
• <b>Ollama</b> (runs on the laptop): install it, run <code>ollama pull qwen3:4b</code>, then add <code>ollama_model = qwen3:4b</code>
No restart needed."""

HELP = """<b>🧠 General predictor</b> (phase 6, plain English)
/ask <i>question</i> — ask anything; a language model picks the right model and answers with its numbers
   <code>/ask what should I pick against pa and pudge if we have cm?</code>
   <code>/ask who wins spirit vs liquid?</code>
   <code>/ask why is axe good against phantom lancer?</code>
   <code>/ask who will win the next TI?</code>
The language model only reads the question and words the answer: every number comes from the other models, and the answer says which command it used."""


def help_text() -> str:
    return HELP


_cache: dict = {"key": None, "llm": None}


def _llm():
    """The language model from .env, re-read when .env changes."""
    env = ROOT / ".env"
    key = env.stat().st_mtime_ns if env.exists() else None
    if _cache["key"] != key or _cache["llm"] is None:
        _cache.update(key=key, llm=from_env(env))
    return _cache["llm"]


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args).strip()
    if not question:
        await update.effective_message.reply_text("Ask me anything, e.g.\n<code>/ask who counters phantom lancer?</code>")
        return
    llm = await asyncio.to_thread(_llm)
    if llm is None:
        await update.effective_message.reply_text(SETUP)
        return
    try:
        text, used = await answer(question, llm, lambda messages, tools: asyncio.to_thread(llm.chat, messages, tools))
    except LLMError as e:
        await update.effective_message.reply_text(f"The language model didn't answer: {html.escape(str(e))}")
        return
    lines = [html.escape(text[:3300] or "I couldn't find an answer to that.")]
    if used:
        lines += ["", "<i>Used:</i> " + " · ".join(f"<code>{html.escape(u[:120])}</code>" for u in used[:4])]
    lines.append(f"<i>Worded by {html.escape(llm.label)}</i>")
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [("ask", ask, "Ask anything in plain English")]

# questions and the commands that can answer them, for /evaluate general
EVAL_QUESTIONS = [
    ("who counters phantom lancer?", {"counters", "apick", "pick"}),
    ("what should I pick against pa and pudge? my team has cm", {"pick", "apick"}),
    ("best heroes against anti-mage in archon games", {"apick"}),
    ("tell me about crystal maiden", {"hero"}),
    ("who wins spirit vs liquid?", {"predict"}),
    ("chance that falcons beat tundra in a bo3", {"predict"}),
    ("how strong is team falcons right now?", {"team"}),
    ("which pro teams are the best right now?", {"top"}),
    ("who will win TI if it was played today?", {"odds"}),
    ("are there pro games being played right now?", {"live"}),
    ("spirit vs liquid draft, axe and pudge are banned, what comes next?", {"next"}),
    ("how have your pro predictions done lately?", {"journal"}),
    ("why is axe good against phantom lancer?", {"hero_abilities", "counters", "hero"}),
]


def status() -> str:
    label = describe(ROOT / ".env")
    return f"{TITLE}: {html.escape(label) if label else '<i>off: no language model in .env (see /general)</i>'}"


def train_model() -> str:
    return "Nothing to train: the general predictor uses a ready-made language model and the other models."


def evaluate_model() -> str:
    """How often the language model picks a right command for EVAL_QUESTIONS (it runs them)."""
    llm = _llm()
    if llm is None:
        return "No language model in .env yet (see /general)."

    async def first_tool(question):
        picked = []

        async def chat(messages, tools):
            reply = await asyncio.to_thread(llm.chat, messages, tools)
            calls = reply.get("tool_calls") or []
            if calls and not picked:
                picked.append(calls[0]["function"]["name"])
            return {"content": "ok"} if calls else reply  # stop after the first choice

        await answer(question, llm, chat)
        return picked[0] if picked else None

    lines, right = [], 0
    for q, ok in EVAL_QUESTIONS:
        try:
            got = asyncio.run(first_tool(q))
        except LLMError as e:
            got = f"error: {e}"[:60]
        right += got in ok
        lines.append(f"{'✓' if got in ok else '✗'} {q} -> {got}")
    lines.append(f"{right} of {len(EVAL_QUESTIONS)} questions went to a right command ({llm.label})")
    return "\n".join(lines)
