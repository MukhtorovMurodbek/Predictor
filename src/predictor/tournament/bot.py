"""Telegram commands for the tournament (TI) predictor (see predictor/bot/registry.py)."""

import asyncio
import html
import json

from telegram import Update
from telegram.ext import ContextTypes

from ..pro import bot as pro_bot
from .backtest import REPORT_PATH, backtest
from .sim import map_matrix, title_odds

NAME = "tournament"
TITLE = "Tournament (TI) predictor"
SUMMARY = "chance each team wins a TI-style event"
N_SIMS = 10_000


def _backtest_rows() -> list[str]:
    """Plain-text lines, one per past TI."""
    if not REPORT_PATH.exists():
        return []
    return [
        f"{r['name']}: winner {r['champion']} got {r['champion_odds']:.0%} "
        f"(#{r['champion_rank']} of {r['teams']}; a random guess gives {r['uniform_odds']:.0%})"
        for r in json.loads(REPORT_PATH.read_text("utf-8"))
        if r.get("champion_odds") is not None
    ]


def _backtest_lines() -> list[str]:
    rows = _backtest_rows()
    if not rows:
        return []
    return ["", "<b>Tested on past TIs</b> (model frozen the day each event started):", *(f"• {html.escape(r)}" for r in rows)]


def help_text() -> str:
    return "\n".join([
        "<b>🏆 Tournament (TI) predictor</b> (uses the pro match predictor)",
        "/odds — who wins if the 16 strongest active teams played a TI today",
        "/odds <i>team, team, ...</i> — title chances for the teams you list (8 or more)",
        "   <code>/odds spirit, yandex, parivision, liquid, navi, falcons, tundra, xtreme</code>",
        "Format like recent TIs: Swiss stage (Bo3, 3 wins through / 3 losses out) → 8-team double elimination, grand final Bo5. "
        f"Each event is simulated {N_SIMS:,} times.",
        *_backtest_lines(),
    ])


async def odds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pro = pro_bot._model()
    if pro is None:
        await update.effective_message.reply_text("Needs the pro model first: /train pro")
        return
    text = " ".join(context.args).strip()
    try:
        if text:
            teams = [pro.find_team(t) for t in text.split(",") if t.strip()]
            if len(set(teams)) != len(teams):
                raise ValueError("a team is listed twice")
            if len(teams) < 8:
                raise ValueError("list at least 8 teams, separated by commas (the playoffs have 8 teams)")
        else:
            teams = [t for t, _ in pro.top_teams(16)]
            if len(teams) < 8:
                raise ValueError("not enough active teams in the data for a tournament (needs 8)")
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    probs = await asyncio.to_thread(lambda: title_odds(map_matrix(pro, teams), N_SIMS))
    ranked = sorted(zip(teams, probs), key=lambda x: -x[1])
    head = "Your tournament" if text else "If the 16 strongest active teams played a TI today"
    lines = [f"<b>{head}</b> <i>(chance to win the title)</i>"]
    for i, (t, p) in enumerate(ranked):
        lines.append(f"{i + 1}. <b>{html.escape(pro.team_name(t))}</b> {p:.1%}")
    lines.append(f"\n<i>{N_SIMS:,} simulated events. Upsets are common: the favourite usually wins well under half the time.</i>")
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [("odds", odds, "Title chances in a TI-style tournament")]


def status() -> str:
    tested = "backtested on past TIs" if REPORT_PATH.exists() else "not backtested yet"
    return f"{TITLE}: uses the pro model, {tested}"


def train_model() -> str:
    """No model of its own: /train tournament re-runs the backtest (a few minutes)."""
    lines = []
    backtest(log=lines.append)
    return "\n".join(lines)


def evaluate_model() -> str:
    return "\n".join(_backtest_rows()) or "No backtest yet: /train tournament"
