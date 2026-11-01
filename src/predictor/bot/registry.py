"""Every model the bot hosts. To add one, write <model>/bot.py (see counterpick/bot.py) and
list it here.

A model's bot module provides:
    NAME      short id used in /train <NAME> and /evaluate <NAME>
    TITLE     human name
    HELP      HTML help section
    COMMANDS  [(command, handler, menu description), ...]
    status()  one-line HTML status
    train_model() -> str, evaluate_model() -> str   (blocking; run in a thread)
"""

from ..counterpick import bot as counterpick_bot

MODELS = {m.NAME: m for m in [counterpick_bot]}
