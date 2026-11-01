"""Telegram commands for the pro match predictor (see predictor/bot/registry.py)."""

import asyncio
import html
import logging
import re
from datetime import datetime, timedelta, timezone

from telegram import Update
from telegram.ext import ContextTypes

from ..data import resolve_heroes
from .evaluate import evaluate
from .model import MODEL_PATH, ProModel, series_win
from .train import train
from ..refresh import journal_summary, refresh

NAME = "pro"
TITLE = "Pro match predictor"
SUMMARY = "who wins a pro match, before or after the draft"

REFRESH_HOUR = 5  # local time of the nightly refresh (new matches, journal, retraining)
log = logging.getLogger(__name__)

_cache: dict = {"mtime": None, "model": None}


def _model() -> ProModel | None:
    """Load the model, and load it again whenever /train pro rewrites the file."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"], _cache["mtime"] = ProModel.load(MODEL_PATH), mtime
    return _cache["model"]


def _date(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def help_text() -> str:
    text = """<b>🏆 Pro match predictor</b> (learned from pro matches only)
/predict <i>team</i> vs <i>team</i> — chance to win a map, Bo3 and Bo5
   <code>/predict spirit vs liquid</code>
After the draft, add each team's 5 heroes after a colon:
   <code>/predict spirit: pa, cm, axe, lion, invoker vs liquid: am, lina, tide, rubick, sven</code>
/team <i>name</i> — rating, recent results, roster, favourite heroes
   <code>/team falcons</code>
/top — strongest active teams right now
Team names can be full, short or the tag: <code>Team Spirit</code>, <code>spirit</code>, <code>NAVI</code>.
Hero names can be full or short: <code>Phantom Assassin</code>, <code>pa</code>, <code>cm</code>."""
    model = _model()
    if model and (s := model.extra.get("stats")):
        text += (
            f"\n\n<i>Tested on {s['test_since']} onwards (tier 1–2 matches): right {s['acc_top']:.0%} of the time after the draft. "
            f"When it says 80%+ it was right {s['acc_sure']:.0%} of the time. "
            f"Data up to {_date(model.state.t)}.</i>"
        )
    return text



async def _need_model(update: Update) -> ProModel | None:
    model = _model()
    if model is None:
        await update.effective_message.reply_text("No pro model yet. Run /collect pro and then /train pro.")
    return model


def _parse_side(model: ProModel, text: str) -> tuple[int, list[int] | None]:
    team_part, _, hero_part = text.partition(":")
    team = model.find_team(team_part)
    if not hero_part.strip():
        return team, None
    heroes = resolve_heroes(hero_part, model.hero_names)
    if len(heroes) != 5 or len(set(heroes)) != 5:
        raise ValueError(f"{model.team_name(team)} needs exactly 5 different heroes")
    return team, heroes


def _record(model: ProModel, team: int, n: int = 10) -> str:
    res = model.recent_results(team, n)
    wins = sum(won for _, _, won in res)
    return f"{wins}-{len(res) - wins}"


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args)
    sides = re.split(r"\s+vs\.?\s+", text, flags=re.IGNORECASE)
    if len(sides) != 2:
        await update.effective_message.reply_text(
            "Write it like <code>/predict spirit vs liquid</code>\n"
            "or with drafts: <code>/predict spirit: pa, cm, axe, lion, invoker vs liquid: am, lina, tide, rubick, sven</code>"
        )
        return
    if not (model := await _need_model(update)):
        return
    try:
        (a, ha), (b, hb) = (_parse_side(model, s) for s in sides)
        if a == b:
            raise ValueError("that's the same team twice")
        if (ha is None) != (hb is None):
            raise ValueError("give heroes for both teams, or for neither")
        if ha and set(ha) & set(hb):
            raise ValueError("a hero can't be on both teams")
        p = model.predict(a, b, ha, hb)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return

    na, nb = html.escape(model.team_name(a)), html.escape(model.team_name(b))
    fav, p_fav = (na, p) if p >= 0.5 else (nb, 1 - p)
    hw, hl = model.head_to_head(a, b)
    lines = [
        f"<b>{na}</b> vs <b>{nb}</b>",
        f"Map: <b>{p:.1%}</b> – <b>{1 - p:.1%}</b>",
        f"Bo3: {series_win(p, 3):.1%} – {series_win(1 - p, 3):.1%}",
        f"Bo5: {series_win(p, 5):.1%} – {series_win(1 - p, 5):.1%}",
        f"Favourite: <b>{fav}</b> ({p_fav:.0%} per map)",
        "",
        f"Rating: {model.elo(a):.0f} vs {model.elo(b):.0f}",
        f"Last 10 maps: {_record(model, a)} vs {_record(model, b)}",
        f"Head-to-head (1 year): {hw}-{hl}",
    ]
    if ha:
        lines.append(f"Draft: {html.escape(', '.join(model.hero_names[h] for h in ha))}")
        lines.append(f"   vs {html.escape(', '.join(model.hero_names[h] for h in hb))}")
        lines.append("<i>After-draft prediction.</i>")
    else:
        lines.append("<i>Before the draft. Add heroes for an after-draft prediction (see /pro).</i>")
    lines.append(f"<i>Rosters from each team's latest match. Data up to {_date(model.state.t)}.</i>")
    await update.effective_message.reply_text("\n".join(lines))


async def team(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Which team? e.g. <code>/team spirit</code>")
        return
    if not (model := await _need_model(update)):
        return
    try:
        t = model.find_team(" ".join(context.args))
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return
    name, tag = model.team_names.get(t, ("?", ""))
    ranking = [tid for tid, _ in model.top_teams(n=10_000)]
    rank = f" · #{ranking.index(t) + 1} of {len(ranking)} active teams" if t in ranking else ""
    last_time, roster = model.state.team_roster[t]
    lines = [
        f"<b>{html.escape(name)}</b>" + (f" [{html.escape(tag)}]" if tag else ""),
        f"Rating {model.elo(t):.0f}{rank}",
        f"Last 10 maps: {_record(model, t)} · last played {_date(last_time)}",
        "",
        "<b>Recent maps</b>",
    ]
    for when, opp, won in reversed(model.recent_results(t, 6)):
        opp_name = model.team_name(opp) if opp is not None else "unknown team"
        lines.append(f"{'✅' if won else '❌'} vs {html.escape(opp_name)} <i>({_date(when)})</i>")
    lines += [
        "",
        f"<b>Roster</b> (latest match): {html.escape(', '.join(model.player_name(p) for p in roster))}",
        *(
            [f"<i>Left since: {html.escape(', '.join(f'{model.player_name(p)} (now {model.team_name(t)})' for p, t in gone))}</i>"]
            if (gone := model.departed(t))
            else []
        ),
        "<b>Most played heroes</b> (recent): "
        + html.escape(", ".join(model.hero_names[h] for h, g in model.signature_heroes(t) if g >= 0.5) or "—"),
    ]
    await update.effective_message.reply_text("\n".join(lines))


async def top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (model := await _need_model(update)):
        return
    lines = [f"<b>Strongest active teams</b> <i>(played in the last 90 days, data up to {_date(model.state.t)})</i>"]
    for i, (t, elo) in enumerate(model.top_teams(15)):
        lines.append(f"{i + 1}. <b>{html.escape(model.team_name(t))}</b> {elo:.0f} · last 10: {_record(model, t)}")
    await update.effective_message.reply_text("\n".join(lines))


async def journal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = await asyncio.to_thread(journal_summary)
    await update.effective_message.reply_text(
        "<b>Prediction journal</b> <i>(new pro maps, predicted before the models saw them)</i>\n"
        f"{html.escape(text)}\n\n"
        f"<i>Filled every night at {REFRESH_HOUR:02d}:00 when new pro matches are downloaded.</i>"
    )


COMMANDS = [
    ("predict", predict, "Who wins a pro match (map, Bo3, Bo5)"),
    ("team", team, "A pro team's rating, results and roster"),
    ("top", top, "Strongest active pro teams"),
    ("journal", journal, "How predictions did on new pro matches"),
]


def status() -> str:
    if not MODEL_PATH.exists():
        return f"{TITLE}: <i>not trained yet</i>"
    model = _model()
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{TITLE}: {model.n_matches:,} pro matches up to {_date(model.state.t)}, trained {trained}"


def train_model() -> str:
    model = train()
    return f"{model.report}\n\nPro predictor retrained on {model.n_matches:,} matches."


def evaluate_model() -> str:
    return evaluate()


async def background() -> None:
    """Every night: new pro matches -> journal -> retrain the pro, draft and live models."""
    while True:
        now = datetime.now()
        nxt = now.replace(hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep((nxt - now).total_seconds())
        try:
            log.info(await asyncio.to_thread(refresh, log=lambda *a: None))
        except Exception:
            log.exception("nightly refresh failed")
