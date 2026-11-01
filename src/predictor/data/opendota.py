"""Thin OpenDota API client with rate limiting and retries.

Free tier: 60 calls/min, 3,000 calls/day per IP. Set OPENDOTA_API_KEY to use a paid key.
"""

import os
import time

import requests

BASE_URL = "https://api.opendota.com/api"


class QuotaExhausted(RuntimeError):
    pass


class OpenDota:
    def __init__(self, api_key: str | None = None, min_interval: float = 1.05):
        self.api_key = api_key or os.environ.get("OPENDOTA_API_KEY")
        self.min_interval = min_interval  # seconds between calls (~57/min)
        self.session = requests.Session()
        self.remaining_day: int | None = None
        self._last_call = 0.0

    def get(self, path: str, **params):
        if self.remaining_day is not None and self.remaining_day <= 0:
            raise QuotaExhausted("OpenDota daily quota used up, try again tomorrow")
        if self.api_key:
            params["api_key"] = self.api_key
        params = {k: v for k, v in params.items() if v is not None}

        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", params=params, timeout=90)

            remaining = resp.headers.get("X-Rate-Limit-Remaining-Day")
            if remaining is not None:
                self.remaining_day = int(remaining)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(5 * 2**attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def explorer(self, sql: str) -> list[dict]:
        """Run SQL against OpenDota's pro-match database."""
        for attempt in range(3):
            try:
                data = self.get("explorer", sql=sql)
                break
            except requests.HTTPError as e:
                # Big queries sometimes fail with a 400 once (server-side timeout), then work.
                if e.response is None or e.response.status_code != 400 or attempt == 2:
                    raise
                time.sleep(10)
        if data.get("err"):
            raise RuntimeError(f"explorer error: {data['err']}")
        return data["rows"]

    def heroes(self) -> list[dict]:
        return self.get("heroes")

    def patches(self) -> list[dict]:
        return self.get("constants/patch")

    def public_matches(self, less_than_match_id: int | None = None, min_rank: int | None = None):
        return self.get("publicMatches", less_than_match_id=less_than_match_id, min_rank=min_rank)

    def match(self, match_id: int) -> dict:
        return self.get(f"matches/{match_id}")
