"""Bot settings from the .env file in the repo root (never committed).

    bot_api = <token from @BotFather>
    bot_username = <the bot's username>
    usernames = friend1, friend2, me      (Telegram usernames, no @)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .. import ROOT


@dataclass(frozen=True)
class BotConfig:
    token: str
    username: str
    allowed: frozenset[str]


def load_config(path: Path = ROOT / ".env") -> BotConfig:
    values = {k.strip().lower(): (v or "").strip() for k, v in dotenv_values(path).items()}

    def get(key: str) -> str:
        return os.environ.get(key.upper()) or values.get(key, "")

    token, usernames = get("bot_api"), get("usernames")
    missing = [k for k, v in (("bot_api", token), ("usernames", usernames)) if not v]
    if missing:
        raise SystemExit(f"{path} is missing: {', '.join(missing)}. See docs/BOT.md.")
    allowed = frozenset(u.strip().lstrip("@").lower() for u in usernames.split(",") if u.strip())
    return BotConfig(token=token, username=get("bot_username").lstrip("@"), allowed=allowed)
