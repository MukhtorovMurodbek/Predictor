"""Only the usernames listed in .env may use the bot.

Usernames can be changed, and a freed one can be claimed by a stranger. So the first time an
allowed username writes to the bot, we remember its numeric Telegram ID in
data/bot_users.json. After that only that ID counts for that username, even if the username
later changes hands.
"""

import json
import logging

from .. import DATA_DIR

USERS_PATH = DATA_DIR / "bot_users.json"
log = logging.getLogger(__name__)


class Auth:
    def __init__(self, allowed: frozenset[str]):
        self.allowed = allowed
        self.known: dict[str, int] = json.loads(USERS_PATH.read_text("utf-8")) if USERS_PATH.exists() else {}

    def check(self, user) -> bool:
        if user is None:
            return False
        if any(uid == user.id for name, uid in self.known.items() if name in self.allowed):
            return True
        name = (user.username or "").lower()
        if name in self.allowed and name not in self.known:
            self.known[name] = user.id
            USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
            USERS_PATH.write_text(json.dumps(self.known, indent=1), "utf-8")
            log.info("linked @%s to Telegram id %s", name, user.id)
            return True
        return False
