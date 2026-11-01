"""Admin commands: collect data, retrain and evaluate models. Every allowed user is an admin.

Jobs take minutes, so they run in a background thread, one at a time, and the bot keeps
answering other commands while they run.
"""

import asyncio
import contextlib
import html
import io
import logging
import time

from telegram import Update
from telegram.ext import ContextTypes

from ..data.collect import collect_heroes, collect_pro, collect_public
from ..data.opendota import OpenDota
from .registry import MODELS

log = logging.getLogger(__name__)
MAX_PUBLIC_CALLS = 2900  # OpenDota free tier: 3000 calls/day

HELP = """<b>⚙️ Admin</b> (everyone on the list can use these)
<b>Daily refresh:</b> /collect pro → /train pro → /train draft → /train livepro. The counter-picker updates itself every hour.
/collect public <i>[calls]</i> — download public games, ~87 per call, 1 call/sec (default 300 ≈ 5 min)
/collect pro — download new pro matches, drafts, timelines (+ teams, leagues)
/collect heroes — refresh hero list and patches
/train <i>[model]</i> — retrain a model on the current data
/evaluate <i>[model]</i> — test a model on the newest games
Models: {models}"""

_lock = asyncio.Lock()
_job: dict = {"label": None, "since": 0.0}


def job_status() -> str:
    if not _job["label"]:
        return "no job running"
    return f"running <b>{html.escape(_job['label'])}</b> for {int(time.time() - _job['since'])} s"


async def _run_job(update: Update, label: str, func, *args) -> None:
    """Run a blocking function in a thread and send back what it printed or returned."""
    msg = update.effective_message
    if _lock.locked():
        await msg.reply_text(f"Busy: {job_status()}. Try again when it's done.")
        return
    async with _lock:
        _job.update(label=label, since=time.time())
        await msg.reply_text(f"Started <b>{html.escape(label)}</b>. I'll reply here when it's done.")
        out = io.StringIO()

        def work():
            with contextlib.redirect_stdout(out):
                return func(*args)

        try:
            result = await asyncio.to_thread(work)
            text = "\n".join(s for s in (out.getvalue().strip(), result if isinstance(result, str) else "") if s)
            took = int(time.time() - _job["since"])
            await msg.reply_text(f"✅ <b>{html.escape(label)}</b> done in {took} s\n<pre>{html.escape(text[-3500:] or 'ok')}</pre>")
        except Exception as e:  # report any failure back to the chat
            log.exception("job %s failed", label)
            await msg.reply_text(f"❌ <b>{html.escape(label)}</b> failed:\n<pre>{html.escape(repr(e))[-3500:]}</pre>")
        finally:
            _job.update(label=None)


async def collect(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = [a.lower() for a in context.args]
    what = args[0] if args else ""
    if what == "heroes":
        await _run_job(update, "collect heroes", collect_heroes, OpenDota())
    elif what == "pro":
        since = args[1] if len(args) > 1 else None
        await _run_job(update, f"collect pro{' since ' + since if since else ''}", collect_pro, OpenDota(), since)
    elif what == "public":
        try:
            calls = min(int(args[1]) if len(args) > 1 else 300, MAX_PUBLIC_CALLS)
        except ValueError:
            await update.effective_message.reply_text("Calls must be a number, e.g. <code>/collect public 500</code>")
            return
        await _run_job(update, f"collect public ({calls} calls)", collect_public, OpenDota(), calls, 70)
    else:
        await update.effective_message.reply_text(
            "What should I collect?\n<code>/collect public 300</code>\n<code>/collect pro</code>\n<code>/collect heroes</code>"
        )


def _pick_model(context: ContextTypes.DEFAULT_TYPE):
    return MODELS.get(context.args[0].lower()) if context.args else None


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (model := _pick_model(context)):
        await update.effective_message.reply_text(f"Which model? " + " or ".join(f"<code>/train {m}</code>" for m in MODELS))
        return
    await _run_job(update, f"train {model.NAME}", model.train_model)


async def evaluate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (model := _pick_model(context)):
        await update.effective_message.reply_text(f"Which model? " + " or ".join(f"<code>/evaluate {m}</code>" for m in MODELS))
        return
    await _run_job(update, f"evaluate {model.NAME}", model.evaluate_model)


COMMANDS = [
    ("collect", collect, "Admin: download match data"),
    ("train", train, "Admin: retrain a model"),
    ("evaluate", evaluate, "Admin: test a model's accuracy"),
]
