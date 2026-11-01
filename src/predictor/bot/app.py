"""The Telegram bot: access control, /start /help /status, and every model's commands."""

import functools
import html
import logging
import time

import pyarrow.parquet as pq
from telegram import BotCommand, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, Defaults, MessageHandler, filters

from ..data import SOURCES
from . import admin
from .auth import Auth
from .config import BotConfig, load_config
from .registry import MODELS

log = logging.getLogger("predictor.bot")
STARTED = time.time()

INTRO = """<b>Dota 2 Predictor</b> 🎯
Hero names can be full or short: <code>Phantom Assassin</code>, <code>phantom a</code>, <code>pa</code>, <code>cm</code>.
Separate heroes with commas."""

GENERAL_HELP = """<b>ℹ️ General</b>
/status — models, data and running jobs
/help — this message"""


def help_text() -> str:
    sections = [INTRO, *(m.HELP for m in MODELS.values()), admin.HELP.format(models=", ".join(MODELS)), GENERAL_HELP]
    return "\n\n".join(sections)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = html.escape(update.effective_user.first_name or "")
    await update.effective_message.reply_text(f"Hi {name}! Here's what I can do.\n\n{help_text()}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(help_text())


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    hours, rest = divmod(int(time.time() - STARTED), 3600)
    lines = [f"<b>Status</b> · up {hours} h {rest // 60} min", "", "<b>Models</b>"]
    lines += [f"• {m.status()}" for m in MODELS.values()]
    lines += ["", "<b>Data</b>"]
    for name, path in SOURCES.items():
        rows = f"{pq.ParquetFile(path).metadata.num_rows:,} matches" if path.exists() else "<i>none yet</i>"
        lines.append(f"• {name}: {rows}")
    lines += ["", f"<b>Jobs:</b> {admin.job_status()}"]
    await update.effective_message.reply_text("\n".join(lines))


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("I don't know that command. Send /help to see what I can do.")


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("Send /help to see what I can do.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("error while handling an update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"Something went wrong: <code>{html.escape(repr(context.error))}</code>")


def build_app(config: BotConfig) -> Application:
    auth = Auth(config.allowed)

    def guarded(func):
        @functools.wraps(func)
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            user = update.effective_user
            if not auth.check(user):
                log.warning("ignored message from %s (id %s)", user and user.username, user and user.id)
                if update.effective_chat and update.effective_chat.type == ChatType.PRIVATE:
                    await update.effective_message.reply_text("Sorry, this is a private bot.")
                return
            await func(update, context)

        return handler

    commands = [
        ("start", start, "Welcome and instructions"),
        ("help", help_cmd, "All commands with examples"),
        ("status", status, "Models, data and running jobs"),
        *(cmd for m in MODELS.values() for cmd in m.COMMANDS),
        *admin.COMMANDS,
    ]

    async def post_init(app: Application) -> None:
        await app.bot.set_my_commands([BotCommand(name, desc) for name, _, desc in commands])
        me = await app.bot.get_me()
        if config.username and me.username.lower() != config.username.lower():
            log.warning("bot_username in .env is %s but the token belongs to @%s", config.username, me.username)
        log.info("@%s is running for %d allowed users. Press Ctrl+C to stop.", me.username, len(config.allowed))

    app = (
        Application.builder()
        .token(config.token)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .post_init(post_init)
        .build()
    )
    for name, func, _ in commands:
        app.add_handler(CommandHandler(name, guarded(func)))
    app.add_handler(MessageHandler(filters.COMMAND, guarded(unknown)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, guarded(plain_text)))
    app.add_error_handler(on_error)
    return app


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every poll otherwise
    app = build_app(load_config())
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
