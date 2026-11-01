"""Telegram commands for the live in-game predictor (see predictor/bot/registry.py)."""

import asyncio
import html
import re
import time
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes

from ..data.opendota import OpenDota
from ..pro import ProModel
from ..pro import bot as pro_bot
from .model import MODEL_PATH, LiveModel
from .train import train

NAME = "livepro"
TITLE = "Live pro match predictor"
SUMMARY = "win % during a pro game, from the scoreboard"

_cache: dict = {"mtime": None, "model": None}
_feed: dict = {"time": 0.0, "games": []}
FEED_SECONDS = 60  # reuse the live feed for a minute (it counts against the OpenDota quota)


def _model() -> LiveModel | None:
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"], _cache["mtime"] = LiveModel.load(MODEL_PATH), mtime
    return _cache["model"]


def help_text() -> str:
    text = """<b>📺 Live pro match predictor</b> (learned from pro matches only)
/live — pro games being played right now, with live win %
/ingame <i>team</i> vs <i>team</i> <i>minute</i> <i>gold lead</i> — win % from a scoreboard you type
   <code>/ingame spirit vs liquid 25 +5k</code>  (Spirit 5,000 gold ahead at 25:00)
   <code>/ingame spirit vs liquid 31 -2.5k towers 3-5 rosh 1-0</code>
Gold lead is for the <b>first</b> team (minus = behind). Optional: <code>towers A-B</code> (towers each team destroyed), <code>rax A-B</code>, <code>rosh A-B</code>."""
    model = _model()
    if model and (s := model.extra.get("stats")):
        text += (
            f"\n\n<i>Tested on newer pro matches: right {s['acc_0']:.0%} of the time at the start, "
            f"{s['acc_10']:.0%} at 10 min, {s['acc_20']:.0%} at 20 min, {s['acc_30']:.0%} at 30 min. "
            "The live feed is delayed by a few minutes, like the broadcast.</i>"
        )
    return text


async def _need_models(update: Update) -> tuple[LiveModel, ProModel] | None:
    live, pro = _model(), pro_bot._model()
    if live is None or pro is None:
        await update.effective_message.reply_text("The live model needs both /train pro and /train livepro first.")
        return None
    return live, pro


def flip(state: dict) -> dict:
    """The same scoreboard seen from the other team's side."""
    return {k: (v if k == "minute" or v is None else -v) for k, v in state.items()}


def team_chance(live: LiveModel, pro: ProModel, a: int, b: int, state_a: dict) -> float:
    """Chance that team a wins, given the scoreboard from a's side. Sides are averaged."""
    pre_a = pro.state.features(a, b, pro.roster(a), pro.roster(b))
    pre_b = pro.state.features(b, a, pro.roster(b), pro.roster(a))
    return (live.predict_radiant(pre_a, state_a) + 1 - live.predict_radiant(pre_b, flip(state_a))) / 2


def _gold(text: str) -> float:
    t = text.lower().replace(",", "")
    return float(t[:-1]) * 1000 if t.endswith("k") else float(t)


def _extra_feature(kind: str) -> str:
    k = kind.lower()
    return "towers_diff" if k.startswith("tower") else "roshan_diff" if k.startswith("rosh") else "rax_diff"


INGAME = re.compile(r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)\s+(?P<min>\d+)(?::\d\d)?\s*(?:min)?\s+(?P<gold>[+-]?\d+(?:\.\d+)?k?)(?P<rest>.*)$", re.IGNORECASE)
EXTRA = re.compile(r"(towers?|rax|barracks|rosh(?:an)?)\s+(\d+)\s*[-:]\s*(\d+)", re.IGNORECASE)


async def ingame(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    m = INGAME.match(" ".join(context.args).strip())
    if not m:
        await update.effective_message.reply_text(
            "Write it like <code>/ingame spirit vs liquid 25 +5k</code>\n"
            "(minute, then the first team's gold lead; optional <code>towers 3-1 rosh 1-0 rax 0-0</code>)"
        )
        return
    if not (models := await _need_models(update)):
        return
    live, pro = models
    try:
        a, b = pro.find_team(m["a"]), pro.find_team(m["b"])
        if a == b:
            raise ValueError("that's the same team twice")
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    minute, gold = int(m["min"]), _gold(m["gold"])
    state = {"minute": float(minute), "gold_adv": gold}
    for kind, x, y in EXTRA.findall(m["rest"]):
        state[_extra_feature(kind)] = float(int(x) - int(y))
    p = team_chance(live, pro, a, b, state)
    na, nb = html.escape(pro.team_name(a)), html.escape(pro.team_name(b))
    lead = f"{na} {'+' if gold >= 0 else '−'}{abs(gold) / 1000:.1f}k gold"
    lines = [
        f"<b>{na}</b> vs <b>{nb}</b> at {minute}:00, {lead}",
        f"Win chance: <b>{p:.0%}</b> – <b>{1 - p:.0%}</b>",
    ]
    extras = [f"{k.split('_')[0]} {v:+.0f}" for k, v in state.items() if k not in ("minute", "gold_adv")]
    if extras:
        lines.append(f"<i>Also counted ({na}'s side): {', '.join(extras)}</i>")
    lines.append(f"<i>Before the game it was {pro.predict(a, b):.0%} for {na}.</i>")
    await update.effective_message.reply_text("\n".join(lines))


def _fetch_live() -> list[dict]:
    if time.time() - _feed["time"] > FEED_SECONDS:
        _feed["games"], _feed["time"] = OpenDota().live(), time.time()
    return _feed["games"]


def _team_id(pro: ProModel, tid) -> int | None:
    return int(tid) if tid and int(tid) in pro.state.team_roster else None


async def live_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (models := await _need_models(update)):
        return
    live, pro = models
    games = await asyncio.to_thread(_fetch_live)
    league = [g for g in games if g.get("league_id") and (g.get("team_name_radiant") or g.get("team_id_radiant"))]
    if not league:
        await update.effective_message.reply_text(
            "No pro league games live right now (OpenDota shows the top ~100 live games). Try again later, "
            "or use <code>/ingame</code> with a scoreboard."
        )
        return
    lines = ["<b>Live pro games</b> <i>(broadcast delay applies)</i>", ""]
    for g in sorted(league, key=lambda g: -(g.get("spectators") or 0))[:8]:
        rt, dt = _team_id(pro, g.get("team_id_radiant")), _team_id(pro, g.get("team_id_dire"))
        players = g.get("players") or []
        side = {s: [p for p in players if p.get("team") == s] for s in (0, 1)}
        rp = [p.get("account_id") or 0 for p in side[0]] or list(pro.roster(rt) if rt else [0])
        dp = [p.get("account_id") or 0 for p in side[1]] or list(pro.roster(dt) if dt else [0])
        rh = [p.get("hero_id") or 0 for p in side[0]]
        dh = [p.get("hero_id") or 0 for p in side[1]]
        drafted = len(rh) == 5 and len(dh) == 5 and all(rh + dh)
        pre = pro.state.features(rt, dt, rp, dp, rh if drafted else None, dh if drafted else None)
        minute = max(0, (g.get("game_time") or 0)) / 60
        p = live.predict_radiant(pre, {"minute": minute, "gold_adv": float(g.get("radiant_lead") or 0)})
        rn = html.escape(g.get("team_name_radiant") or (pro.team_name(rt) if rt else "Radiant"))
        dn = html.escape(g.get("team_name_dire") or (pro.team_name(dt) if dt else "Dire"))
        clock = f"{int(minute)}:{int((minute % 1) * 60):02d}" if minute else "drafting"
        lead = g.get("radiant_lead") or 0
        lines.append(f"<b>{rn}</b> {p:.0%} – {1 - p:.0%} <b>{dn}</b>")
        lines.append(f"   {clock} · kills {g.get('radiant_score', 0)}–{g.get('dire_score', 0)} · gold {'+' if lead >= 0 else '−'}{abs(lead) / 1000:.1f}k for {rn}")
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [
    ("live", live_games, "Pro games live now, with win %"),
    ("ingame", ingame, "Win % from a scoreboard you type"),
]


def status() -> str:
    if not MODEL_PATH.exists():
        return f"{TITLE}: <i>not trained yet</i>"
    model = _model()
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{TITLE}: {model.n_matches:,} pro matches, trained {trained}"


def train_model() -> str:
    model = train(log=lambda *a: None)
    return f"{model.report}\n\nLive predictor retrained on {model.n_matches:,} matches."


def evaluate_model() -> str:
    model = _model()
    if model is None:
        return "No live model yet. Run /train livepro (it tests itself while training)."
    return f"{model.report}\n\n(from the last /train livepro)"
