"""Every model the bot hosts. To add one, write <model>/bot.py (see counterpick/bot.py or
pro/bot.py) and list it here.

A model's bot module provides:
    NAME         short id: its intro command (/NAME), and /train NAME, /evaluate NAME
    TITLE        human name
    SUMMARY      one line for the /start menu
    help_text()  HTML intro + commands with examples (shown by /NAME and /help)
    COMMANDS     [(command, handler, menu description), ...]
    status()     one-line HTML status
    train_model() -> str, evaluate_model() -> str   (blocking; run in a thread)
"""

from ..counterpick import bot as counterpick_bot
from ..pro import bot as pro_bot

MODELS = {m.NAME: m for m in [counterpick_bot, pro_bot]}
