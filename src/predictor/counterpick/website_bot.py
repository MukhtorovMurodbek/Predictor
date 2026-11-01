"""Telegram commands for Asilbek's counter-picker (the website version, see engine.py).

Same data as /counterpick (public games the bot downloads every hour), but Asilbek's rules:
a candidate's score is its average win rate against the enemies, with optional rank and patch
filters. Registered in predictor/bot/registry.py.
"""

import asyncio
import html
import re
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import ContextTypes

from ..data import SOURCES, load_heroes, load_matches, resolve_heroes
from .engine import MIN_MATCHES_PER_OPPONENT, pair_counts, rank_candidates

NAME = "asil"
TITLE = "Asilbek's counter-picker"
SUMMARY = "his website's ranking: best average win rate vs the enemies, rank filters"

# avg_rank_tier = 10 * medal + stars (Herald 1x ... Divine 7x, Immortal 80)
MEDALS = {"herald": 1, "guardian": 2, "crusader": 3, "archon": 4, "legend": 5, "ancient": 6, "divine": 7, "immortal": 8}
RANK = re.compile(r"\brank\s+([a-z]+|\d+)(?:\s*[-–]\s*([a-z]+|\d+))?", re.IGNORECASE)
PATCH = re.compile(r"\bpatch\s+(\d+\.\d+[a-z]?)", re.IGNORECASE)

HELP = f"""<b>🧩 Asilbek's counter-picker</b> (the website version)
/apick <i>enemies | allies | bans</i> — heroes with the best average win rate against those enemies
   <code>/apick pa, pudge, lion</code>
   <code>/apick pa, pudge | cm | invoker rank archon-legend</code>
   <code>/apick pl rank divine patch 7.41</code>
Optional filters: <code>rank herald</code>, <code>rank crusader-legend</code>, <code>rank immortal</code> (the game's average rank), <code>patch 7.41</code>.
Allies and bans are only left out of the suggestions; they don't change the scores.
Heroes with at least {MIN_MATCHES_PER_OPPONENT} games against every enemy come first; the others are marked ⚠.
Uses the public games of the last 7 days. The other counter-picker, /counterpick, uses 90 days and Dotabuff-style edges."""


def help_text() -> str:
    return HELP


_cache: dict = {"key": None, "matches": None, "counts": {}}


def _counts(rank_min, rank_max, patch):
    """Pair counts for these filters, recomputed when new games arrive (hourly)."""
    path = SOURCES["public"]
    if not path.exists():
        return None
    key = path.stat().st_mtime_ns
    if _cache["key"] != key:
        _cache.update(key=key, matches=load_matches("public"), counts={})
    f = (rank_min, rank_max, patch)
    if f not in _cache["counts"]:
        _cache["counts"][f] = pair_counts(_cache["matches"], *f)
    return _cache["counts"][f]


def parse_rank(text: str) -> tuple[int | None, int | None, str]:
    """'rank archon-legend' -> (40, 55, 'Archon–Legend'). Numbers are rank tiers (10-80)."""
    m = RANK.search(text)
    if not m:
        return None, None, "all ranks"

    def bounds(word):
        w = word.lower()
        if w.isdigit():
            tier = int(w)
            if not 10 <= tier <= 80:
                raise ValueError("rank numbers go from 10 (Herald 0) to 80 (Immortal)")
            return tier, tier, str(tier)
        medal = next((v for k, v in MEDALS.items() if k.startswith(w)), None) if len(w) >= 3 else None
        if medal is None:
            raise ValueError(f"unknown rank {word!r}: use herald, guardian, crusader, archon, legend, ancient, divine or immortal")
        return medal * 10, 80 if medal == 8 else medal * 10 + 5, next(k for k, v in MEDALS.items() if v == medal).title()

    lo, hi1, label1 = bounds(m.group(1))
    if m.group(2):
        lo2, hi, label2 = bounds(m.group(2))
        lo, hi = min(lo, lo2), max(hi1, hi)
        return lo, hi, f"{label1}–{label2}"
    return lo, hi1, label1


def _date(ts) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "?"


async def apick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args)
    try:
        rank_min, rank_max, rank_label = parse_rank(text)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    patch_m = PATCH.search(text)
    patch = patch_m.group(1) if patch_m else None
    heroes_text = PATCH.sub("", RANK.sub("", text)).strip()
    if not heroes_text:
        await update.effective_message.reply_text(
            "Tell me the enemy heroes, e.g.\n<code>/apick pa, pudge, lion</code>\n"
            "Optional: allies and bans after <code>|</code>, and <code>rank archon-legend</code> or <code>patch 7.41</code>."
        )
        return
    names = load_heroes()
    parts = (heroes_text.split("|") + ["", ""])[:3]
    try:
        enemies, allies, bans = (resolve_heroes(p, names) for p in parts)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    if not enemies:
        await update.effective_message.reply_text("Name at least one enemy hero, e.g. <code>/apick pa</code>")
        return
    counts = await asyncio.to_thread(_counts, rank_min, rank_max, patch)
    if counts is None or counts.eligible == 0:
        await update.effective_message.reply_text("No public games match those filters yet.")
        return
    ranked = rank_candidates(names, counts, enemies, allies, bans)[:10]
    short = {e: names[e] for e in enemies}
    lines = [
        f"<b>Asilbek's counter-picker</b> · {html.escape(rank_label)}" + (f" · patch {html.escape(patch)}" if patch else ""),
        f"<b>Enemies:</b> {html.escape(', '.join(names[e] for e in enemies))}",
    ]
    if allies or bans:
        lines.append(f"<i>Left out: {html.escape(', '.join(names[h] for h in allies + bans))}</i>")
    lines += ["", "<b>Best picks</b> <i>(average win rate vs the enemies · games)</i>"]
    for i, c in enumerate(ranked):
        flag = "" if c["reliable"] else " ⚠"
        line = f"{i + 1}. <b>{html.escape(c['name'])}</b> {c['score']:.1f}% · {c['total_matches']:,} games{flag}"
        if 1 < len(enemies) <= 3:
            line += "\n    " + " · ".join(
                f"vs {html.escape(short[p['enemy_id']])} {p['win_rate']:.0f}% ({p['matches']})" if p["matches"] else f"vs {html.escape(short[p['enemy_id']])}: no games"
                for p in c["per_enemy"]
            )
        lines.append(line)
    if not ranked:
        lines.append("<i>No hero has games against these enemies yet.</i>")
    lines += [
        "",
        f"<i>{counts.eligible:,} public games ({_date(counts.oldest)} to {_date(counts.newest)}). "
        f"⚠ = fewer than {MIN_MATCHES_PER_OPPONENT} games against some enemy. Rank = the game's average rank.</i>",
    ]
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [("apick", apick, "Asilbek's counter-picks (win rate, rank filter)")]


def status() -> str:
    path = SOURCES["public"]
    if not path.exists():
        return f"{TITLE}: <i>no public games yet</i>"
    counts = _counts(None, None, None)
    return f"{TITLE}: {counts.eligible:,} public games ({_date(counts.oldest)} to {_date(counts.newest)})"


def train_model() -> str:
    return "Nothing to train: Asilbek's counter-picker counts the public games the bot downloads every hour."


def evaluate_model() -> str:
    """How many games back each rank filter (statistics, not a model: nothing to test)."""
    lines = []
    for medal, n in MEDALS.items():
        lo, hi = n * 10, 80 if n == 8 else n * 10 + 5
        lines.append(f"{medal.title():<9} {_counts(lo, hi, None).eligible:>8,} games")
    total = _counts(None, None, None)
    lines.append(f"{'all':<9} {total.eligible:>8,} games ({_date(total.oldest)} to {_date(total.newest)})")
    return "\n".join(lines)
