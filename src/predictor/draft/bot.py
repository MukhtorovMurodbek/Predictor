"""Telegram commands for the pro next-pick predictor (see predictor/bot/registry.py)."""

import html
import itertools
import re
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import ContextTypes

from ..data import resolve_heroes
from .model import MODEL_PATH, DraftModel
from .train import train

NAME = "draft"
TITLE = "Pro draft predictor"
SUMMARY = "which hero a pro team picks or bans next"

_cache: dict = {"mtime": None, "model": None}


def _model() -> DraftModel | None:
    """Load the model, and load it again whenever /train draft rewrites the file."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"], _cache["mtime"] = DraftModel.load(MODEL_PATH), mtime
    return _cache["model"]


def _date(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def order_text(order) -> str:
    """'Ban A A B B A B B · Pick A B · ...' from [(is_pick, 'A'/'B'), ...]."""
    parts = []
    for is_pick, group in itertools.groupby(order, key=lambda x: x[0]):
        parts.append(("Pick " if is_pick else "Ban ") + " ".join(who for _, who in group))
    return " · ".join(parts)


def help_text() -> str:
    text = """<b>🧠 Pro draft predictor</b> (learned from pro drafts only)
/next <i>team A</i> vs <i>team B</i>: <i>heroes so far</i> — the most likely next pick or ban
Write the team that acts <b>first</b> (has first pick) first, then every hero picked or banned so far, in draft order:
   <code>/next spirit vs liquid</code>  (who gets banned first?)
   <code>/next spirit vs liquid: axe, pudge, lion, mars, io, tusk</code>
Team names can be full, short or the tag. Hero names can be full or short (<code>pa</code>, <code>cm</code>)."""
    model = _model()
    if model:
        text += f"\n\n<b>Draft order</b> (A = first team): {order_text(model.order)}"
        if s := model.extra.get("stats"):
            picks, bans = s["model_picks"], s["model_bans"]
            text += (
                f"\n\n<i>Tested on newer pro drafts: the real hero is in the top 5 for {picks[5]:.0%} of picks "
                f"and {bans[5]:.0%} of bans (top 10: {picks[10]:.0%} and {bans[10]:.0%}). "
                f"Teams hide their plans, so drafts are hard to predict. Data up to {_date(model.state.t)}.</i>"
            )
    return text


async def _need_model(update: Update) -> DraftModel | None:
    model = _model()
    if model is None:
        await update.effective_message.reply_text("No draft model yet. Run /collect pro and then /train draft.")
    return model


async def next_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args)
    teams_part, _, heroes_part = text.partition(":")
    sides = re.split(r"\s+vs\.?\s+", teams_part.strip(), flags=re.IGNORECASE)
    if len(sides) != 2:
        await update.effective_message.reply_text(
            "Write it like <code>/next spirit vs liquid: axe, pudge, lion</code>\n"
            "(the first team acts first; heroes are everything picked or banned so far, in order)"
        )
        return
    if not (model := await _need_model(update)):
        return
    try:
        a, b = (model.find_team(s) for s in sides)
        if a == b:
            raise ValueError("that's the same team twice")
        done = resolve_heroes(heroes_part, model.hero_names)
        ranked = model.predict_next(a, b, done)
    except ValueError as e:
        await update.effective_message.reply_text(html.escape(str(e)))
        return

    names = {"A": model.team_name(a), "B": model.team_name(b)}
    is_pick, actor = model.next_step(len(done))
    lines = [f"<b>{html.escape(names['A'])}</b> (first) vs <b>{html.escape(names['B'])}</b>"]
    if done:
        for kind, want_pick in (("Bans", False), ("Picks", True)):
            for who in "AB":
                heroes = [model.hero_names[h] for (p, w), h in zip(model.order, done) if p == want_pick and w == who]
                if heroes:
                    lines.append(f"{kind} {html.escape(names[who])}: {html.escape(', '.join(heroes))}")
    lines += [
        "",
        f"Step {len(done) + 1} of {len(model.order)}: <b>{html.escape(names[actor])} {'picks' if is_pick else 'bans'}</b>",
        "<b>Most likely</b>",
    ]
    for i, (hero, share) in enumerate(ranked):
        lines.append(f"{i + 1}. {html.escape(model.hero_names[hero])} <i>{share:.0%}</i>")
    lines.append(f"<i>Rosters and habits from data up to {_date(model.state.t)}.</i>")
    await update.effective_message.reply_text("\n".join(lines))


COMMANDS = [("next", next_action, "Most likely next pick or ban in a pro draft")]


def status() -> str:
    if not MODEL_PATH.exists():
        return f"{TITLE}: <i>not trained yet</i>"
    model = _model()
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{TITLE}: {model.n_matches:,} pro drafts up to {_date(model.state.t)}, trained {trained}"


def train_model() -> str:
    model = train(log=lambda *a: None)
    return f"{model.report}\n\nDraft predictor retrained on {model.n_matches:,} drafts."


def evaluate_model() -> str:
    model = _model()
    if model is None:
        return "No draft model yet. Run /train draft (it tests itself while training)."
    return f"{model.report}\n\n(from the last /train draft. Retrain to re-test on newer data.)"
