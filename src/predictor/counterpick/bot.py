"""Telegram commands for the counter-picker (see predictor/bot/registry.py).

Pure statistics (counterpick/stats.py), kept fresh by an hourly background update.
"""

import asyncio
import html
import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..data import load_heroes, resolve_heroes
from .stats import DAILY_DIR, MIN_GAMES, WINDOW_DAYS, MatchupStats
from .update import update

NAME = "counterpick"
TITLE = "Counter-picker"
SUMMARY = "best heroes to pick in your own (non-pro) games"
UPDATE_MINUTES = 60

log = logging.getLogger(__name__)

HELP = f"""<b>🛡 Counter-picker</b> (statistics from public games of all ranks, last {WINDOW_DAYS} days, updated every hour)
/pick <i>enemies | allies | bans</i> — best heroes to pick
   <code>/pick pa, pudge, lion</code>
   <code>/pick pa, pudge | cm, axe | invoker</code>
/counters <i>hero</i> — heroes that beat this hero
   <code>/counters phantom assassin</code>
/hero <i>hero</i> — win rate, best and worst matchups, best allies
   <code>/hero cm</code>
Hero names can be full or short: <code>Phantom Assassin</code>, <code>phantom a</code>, <code>pa</code>, <code>cm</code>.
<b>Edge</b> (like Dotabuff's advantage): the real win rate in a matchup minus what the two heroes' usual win rates predict. Matchups with under {MIN_GAMES} games are left out."""


def help_text() -> str:
    return HELP


_cache: dict = {"key": None, "stats": None}


def _stats() -> MatchupStats | None:
    """Sum of the daily tables, recomputed whenever a table changes."""
    files = sorted(DAILY_DIR.glob("*.npz")) if DAILY_DIR.exists() else []
    if not files:
        return None
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    if _cache["key"] != key:
        _cache["stats"], _cache["key"] = MatchupStats.from_daily(load_heroes()), key
    return _cache["stats"]


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _names(stats: MatchupStats, ids) -> str:
    return html.escape(", ".join(stats.hero_names[h] for h in ids)) or "—"


def _footer(stats: MatchupStats) -> str:
    return f"<i>{stats.n_matches:,} public games, {stats.first_day} to {stats.last_day}.</i>"


async def _need_stats(update: Update) -> MatchupStats | None:
    stats = _stats()
    if stats is None:
        await update.effective_message.reply_text("No statistics yet. Run /train counterpick to download games now.")
    return stats


async def pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args)
    if not text.strip():
        await update.effective_message.reply_text(
            "Tell me the enemy heroes, e.g.\n<code>/pick pa, pudge, lion</code>\n"
            "Add your team and bans after <code>|</code>:\n<code>/pick pa, pudge | cm | invoker</code>"
        )
        return
    if not (stats := await _need_stats(update)):
        return
    parts = (text.split("|") + ["", ""])[:3]
    try:
        enemies, allies, bans = (resolve_heroes(p, stats.hero_names) for p in parts)
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

    recs = stats.recommend(allies, enemies, bans, top=10)
    lines = [f"<b>Enemies:</b> {_names(stats, enemies)}", f"<b>Allies:</b> {_names(stats, allies)}"]
    if bans:
        lines.append(f"<b>Banned:</b> {_names(stats, bans)}")
    lines += ["", "<b>Best picks</b> <i>(edge vs enemies + with allies)</i>"]
    for i, r in recs.iterrows():
        line = f"{i + 1}. <b>{html.escape(r.hero)}</b> {_pct(r.score)}"
        if len(enemies) > 1 and r.best_vs_edge > 0:
            line += f" · best vs {html.escape(r.best_vs)}"
        line += f" <i>({int(r.games):,} games)</i>"
        if r.unknown:
            line += " <i>⚠ some matchups have too few games</i>"
        lines.append(line)
    lines += ["", _footer(stats)]
    await update.effective_message.reply_text("\n".join(lines))


async def counters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Which hero? e.g. <code>/counters pa</code>")
        return
    if not (stats := await _need_stats(update)):
        return
    try:
        (enemy,) = resolve_heroes(" ".join(context.args).replace(",", " "), stats.hero_names)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    name = html.escape(stats.hero_names[enemy])
    table = stats.counters_for(enemy, top=10)
    lines = [f"<b>Best counters to {name}</b> <i>(edge · their win rate vs {name} · games)</i>"]
    for i, r in table.iterrows():
        lines.append(f"{i + 1}. <b>{html.escape(r.hero)}</b> {_pct(r.edge)} · {r.win_rate:.1%} · <i>{r.games:,}</i>")
    if table.empty:
        lines.append(f"<i>No matchup has {MIN_GAMES}+ games yet.</i>")
    lines += ["", _footer(stats)]
    await update.effective_message.reply_text("\n".join(lines))


async def hero(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Which hero? e.g. <code>/hero cm</code>")
        return
    if not (stats := await _need_stats(update)):
        return
    try:
        (h,) = resolve_heroes(" ".join(context.args).replace(",", " "), stats.hero_names)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    card = stats.hero_card(h)

    def fmt(items):
        return ", ".join(f"{html.escape(n)} {_pct(e)}" for n, e in items) or "—"

    lines = [
        f"<b>{html.escape(stats.hero_names[h])}</b>",
        f"Win rate: {card['win_rate']:.1%} over {card['games']:,} games",
        f"<b>Strong vs:</b> {fmt(card['strong_vs'])}",
        f"<b>Weak vs:</b> {fmt(card['weak_vs'])}",
        f"<b>Best allies:</b> {fmt(card['allies'])}",
        "",
        _footer(stats),
    ]
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [
    ("pick", pick, "Best heroes to pick vs an enemy draft"),
    ("counters", counters, "Heroes that beat a given hero"),
    ("hero", hero, "A hero's win rate, matchups and allies"),
]


def status() -> str:
    stats = _stats()
    if stats is None:
        return f"{TITLE}: <i>no statistics yet</i>"
    return f"{TITLE}: {stats.n_matches:,} public games ({stats.first_day} to {stats.last_day}), auto-updates hourly"


def train_model() -> str:
    """/train counterpick = update now (there is no model to train)."""
    return update()


def evaluate_model() -> str:
    stats = _stats()
    if stats is None:
        return "No statistics yet."
    c = stats.counts
    pairs = c.vs_games[c.vs_games > 0]
    return (
        f"{stats.n_matches:,} public games from {stats.first_day} to {stats.last_day}\n"
        f"hero pairs with {MIN_GAMES}+ games: {(pairs >= MIN_GAMES).sum():,} of {len(pairs):,}\n"
        f"median games per pair: {int(sorted(pairs)[len(pairs) // 2]) if len(pairs) else 0:,}\n"
        "(statistics, not a model: there is nothing to test, only how much data backs each number)"
    )


async def background() -> None:
    """Started by the bot: fetch new public games and refresh the statistics every hour."""
    await asyncio.sleep(60)
    while True:
        try:
            log.info(await asyncio.to_thread(update))
            await asyncio.to_thread(_stats)  # warm the cache so commands answer instantly
        except Exception:
            log.exception("counter-picker update failed")
        await asyncio.sleep(UPDATE_MINUTES * 60)
