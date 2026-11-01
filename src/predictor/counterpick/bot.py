"""Telegram commands for the counter-picker (see predictor/bot/registry.py)."""

import html
from datetime import datetime

import numpy as np
from telegram import Update
from telegram.ext import ContextTypes

from ..data import resolve_heroes
from .evaluate import evaluate
from .model import CounterModel, model_path
from .train import train

NAME = "counterpick"
TITLE = "Counter-picker"
SUMMARY = "best heroes to pick in your own (non-pro) games"
SOURCE = "public"  # counter-picks are for regular ranked games, not pro play

HELP = """<b>🛡 Counter-picker</b> (learned from public ranked games)
/pick <i>enemies | allies | bans</i> — best heroes to pick
   <code>/pick pa, pudge, lion</code>
   <code>/pick pa, pudge | cm, axe | invoker</code>
/counters <i>hero</i> — heroes that beat this hero
   <code>/counters phantom assassin</code>
/hero <i>hero</i> — win rate, best and worst matchups, best allies
   <code>/hero cm</code>
Hero names can be full or short: <code>Phantom Assassin</code>, <code>phantom a</code>, <code>pa</code>, <code>cm</code>.
Numbers are like Dotabuff's: how many win-rate points better (or worse) than usual a hero does in that matchup. Few games = less reliable."""


def help_text() -> str:
    return HELP

_cache: dict = {"mtime": None, "model": None}


def _model() -> CounterModel | None:
    """Load the model, and load it again whenever /train rewrites the file."""
    path = model_path(SOURCE)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"], _cache["mtime"] = CounterModel.load(path), mtime
    return _cache["model"]


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _names(model: CounterModel, ids) -> str:
    return html.escape(", ".join(model.hero_names[h] for h in ids)) or "—"


async def _need_model(update: Update) -> CounterModel | None:
    model = _model()
    if model is None:
        await update.effective_message.reply_text(
            "No counter-pick model yet. Run /collect public and then /train counterpick."
        )
    return model


async def pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args)
    if not text.strip():
        await update.effective_message.reply_text(
            "Tell me the enemy heroes, e.g.\n<code>/pick pa, pudge, lion</code>\n"
            "Add your team and bans after <code>|</code>:\n<code>/pick pa, pudge | cm | invoker</code>"
        )
        return
    if not (model := await _need_model(update)):
        return
    parts = (text.split("|") + ["", ""])[:3]
    try:
        enemies, allies, bans = (resolve_heroes(p, model.hero_names) for p in parts)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    picked = enemies + allies + bans
    if len(set(picked)) != len(picked):
        await update.effective_message.reply_text("The same hero is listed twice.")
        return
    if len(enemies) > 5 or len(allies) > 4:
        await update.effective_message.reply_text("At most 5 enemies and 4 allies.")
        return

    recs = model.recommend(allies, enemies, bans, top=10)
    lines = [
        f"<b>Enemies:</b> {_names(model, enemies)}",
        f"<b>Allies:</b> {_names(model, allies)}",
    ]
    if bans:
        lines.append(f"<b>Banned:</b> {_names(model, bans)}")
    lines += ["", "<b>Best picks</b> <i>(edge vs enemies, + with allies)</i>"]
    for i, r in recs.iterrows():
        line = f"{i + 1}. <b>{html.escape(r.hero)}</b> {_pct(r.score)}"
        if enemies and len(enemies) > 1 and r.best_vs_adv > 0:
            line += f" · best vs {html.escape(r.best_vs)}"
        if enemies:
            line += f" <i>({int(r.games_vs):,} games)</i>"
        lines.append(line)
    lines += ["", f"<i>From {model.n_matches:,} public games. Edge = win rate in these matchups minus what the heroes' usual win rates predict.</i>"]
    await update.effective_message.reply_text("\n".join(lines))


async def counters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Which hero? e.g. <code>/counters pa</code>")
        return
    if not (model := await _need_model(update)):
        return
    try:
        (enemy,) = resolve_heroes(" ".join(context.args).replace(",", " "), model.hero_names)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    table = model.counters_for(enemy, top=10)
    lines = [f"<b>Best counters to {html.escape(model.hero_names[enemy])}</b>"]
    for i, r in table.iterrows():
        lines.append(f"{i + 1}. <b>{html.escape(r.hero)}</b> {_pct(r.advantage)} <i>({r.games} games)</i>")
    await update.effective_message.reply_text("\n".join(lines))


def _top(model: CounterModel, row: np.ndarray, games: np.ndarray, h: int, best: bool, n=4) -> str:
    cand = [k for k in model.hero_names if k != h and games[k] >= 30]
    cand.sort(key=lambda k: row[k], reverse=best)
    return ", ".join(f"{html.escape(model.hero_names[k])} {_pct(row[k])}" for k in cand[:n]) or "—"


async def hero(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Which hero? e.g. <code>/hero cm</code>")
        return
    if not (model := await _need_model(update)):
        return
    try:
        (h,) = resolve_heroes(" ".join(context.args).replace(",", " "), model.hero_names)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    vs, with_ = model.edges()
    lines = [
        f"<b>{html.escape(model.hero_names[h])}</b>",
        f"Win rate: {model.win_rates()[h]:.1%} over {int(model.hero_games[h]):,} games",
        f"<b>Strong vs:</b> {_top(model, vs[h], model.vs_games[h], h, best=True)}",
        f"<b>Weak vs:</b> {_top(model, vs[h], model.vs_games[h], h, best=False)}",
        f"<b>Best allies:</b> {_top(model, with_[h], model.with_games[h], h, best=True)}",
    ]
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [
    ("pick", pick, "Best heroes to pick vs an enemy draft"),
    ("counters", counters, "Heroes that beat a given hero"),
    ("hero", hero, "A hero's win rate, matchups and allies"),
]


def status() -> str:
    path = model_path(SOURCE)
    if not path.exists():
        return f"{TITLE}: <i>not trained yet</i>"
    model = _model()
    trained = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{TITLE}: {model.n_matches:,} public games, trained {trained}"


def train_model() -> str:
    model, _ = train(SOURCE)
    w = model.weights
    return (
        f"Counter-picker retrained on {model.n_matches:,} public games.\n"
        f"weights: base {w[1]:.2f}, counter {w[2]:.2f}, synergy {w[3]:.2f}"
    )


def evaluate_model() -> str:
    return evaluate(SOURCE)
