"""/ask: a question in plain English -> the right model -> a short answer.

The language model never predicts anything itself. It reads the question, picks one of the
bot's commands as a "tool" (with the arguments a person would type), the bot runs that
command like any other, and the language model turns the reply into a short answer. Every
number in the answer comes from our models. Admin commands are never offered as tools.

    question --LLM--> pick("pa, pudge | cm") --bot--> /pick reply --LLM--> answer
"""

import html
import json
import re
from types import SimpleNamespace

from ..data import load_heroes, resolve_heroes
from .llm import LLM

MAX_ROUNDS = 3  # tool rounds before the model must answer
MAX_TOOL_TEXT = 3000  # characters of a command's reply passed back to the model

SYSTEM = """You answer questions for Dota 2 Lab, a Telegram bot with Dota 2 models built by a small team of students.
Answer by calling the bot's commands as tools. Write the tool's "args" exactly as a person would type them after the command.
- Picks and counters in normal (public) games: pick, counters, hero, apick.
- Pro matches: predict (who wins, chances), team, top (best teams), journal, next (pro drafts: next pick/ban), live and ingame (games being played), odds (tournament winner).
- "Why" questions about heroes: hero_abilities gives a hero's skills; combine it with counters or hero.
Hero names can be short: pa = Phantom Assassin, pl = Phantom Lancer, am = Anti-Mage, cm = Crystal Maiden, wk = Wraith King.
After the tools answer, reply in 1 to 6 short sentences of plain text (no markdown), using only the numbers the tools gave you.
If the tools can't answer the question, say so in one sentence."""

_HTML_TAG = re.compile(r"<[^>]+>")


def plain(text: str) -> str:
    """Telegram HTML -> plain text for the language model."""
    return html.unescape(_HTML_TAG.sub("", text))


class CaptureMessage:
    """Stands in for a Telegram message: keeps the replies instead of sending them."""

    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def commands() -> dict:
    """{command: (handler, model)} for every model command (no admin commands)."""
    from ..bot.registry import MODELS  # here, not at the top: the registry imports this package

    return {name: (handler, m) for m in MODELS.values() if m.NAME != "general" for name, handler, _ in m.COMMANDS}


def tools() -> list[dict]:
    """Every model command as a tool, described by its menu text and the examples in its help."""
    from ..bot.registry import MODELS

    out = []
    for m in MODELS.values():
        if m.NAME == "general":
            continue
        examples = re.findall(r"<code>/(\w+)([^<]*)</code>", m.help_text())
        for name, _, desc in m.COMMANDS:
            uses = [f"/{name}{plain(a)}" for c, a in examples if c == name][:4]
            text = f"{m.TITLE}: {desc}." + (" Examples: " + " ; ".join(uses) if uses else "")
            out.append(_tool(name, text, "what to type after the command, e.g. 'pa, pudge | cm'"))
    out.append(_tool("hero_abilities", "A hero's abilities as described in the game (to explain why a hero counters another).",
                     "one hero name, e.g. 'axe'"))
    return out


def _tool(name: str, description: str, args_help: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description[:1000],
        "parameters": {"type": "object", "properties": {"args": {"type": "string", "description": args_help}}, "required": ["args"]},
    }}


def hero_abilities(args: str) -> str:
    from .skills import load_constants

    names = load_heroes()
    hero_ids = resolve_heroes(args, names)[:1]
    if not hero_ids:
        return "Name one hero."
    heroes, hero_abilities_, abilities = load_constants()
    hero = heroes[str(hero_ids[0])]
    lines = [f"{hero['localized_name']} ({hero.get('attack_type', '')}, {hero.get('primary_attr', '')}; roles: {', '.join(hero.get('roles') or [])})"]
    for item in hero_abilities_.get(hero["name"], {}).get("abilities", []):
        for name in item if isinstance(item, list) else [item]:
            a = abilities.get(name) or {}
            if a.get("dname") and name != "generic_hidden":
                facts = ", ".join(f"{k}: {a[k]}" for k in ("dmg_type", "bkbpierce", "dispellable") if a.get(k))
                lines.append(f"- {a['dname']}: {a.get('desc', '')}" + (f" ({facts})" if facts else ""))
    return "\n".join(lines)


async def run_tool(name: str, args: str) -> str:
    """Run a bot command (or hero_abilities) and return its reply as plain text."""
    if name == "hero_abilities":
        try:
            return hero_abilities(args)
        except (ValueError, FileNotFoundError) as e:
            return str(e)
    table = commands()
    if name not in table:
        return f"There is no command /{name}."
    handler, _ = table[name]
    msg = CaptureMessage()
    await handler(SimpleNamespace(effective_message=msg), SimpleNamespace(args=args.split()))
    return plain("\n\n".join(msg.replies))[:MAX_TOOL_TEXT]


async def answer(question: str, llm: LLM, chat) -> tuple[str, list[str]]:
    """Answer one question. `chat(messages, tools)` runs llm.chat off the event loop.
    Returns (answer text, commands used)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    used = []
    available = tools()
    for round_ in range(MAX_ROUNDS + 1):
        reply = await chat(messages, available if round_ < MAX_ROUNDS else None)
        calls = reply.get("tool_calls") or []
        if not calls:
            return (reply.get("content") or "").strip(), used
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = str(json.loads(call["function"].get("arguments") or "{}").get("args", ""))
            except (json.JSONDecodeError, AttributeError):
                args = ""
            used.append(f"/{name} {args}".strip())
            result = await run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})
    return "", used
