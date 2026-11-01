"""Serve the counter-picker and local match statistics on loopback only.

Run from the repository root: python -m web.server
Data: the counter-picker's daily tables (every public game OpenDota lists, last 30 days),
filled every hour by the bot or by: python -m predictor.counterpick.update
"""

import argparse
import json
import re
import threading
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from predictor.counterpick import stats
from predictor.counterpick.engine import APICK_DAYS, snapshot
from predictor.data import HEROES_PATH, load_heroes

STATIC_DIR = Path(__file__).resolve().parent / "static"
_cache = OrderedDict()
_lock = threading.Lock()


def _revision(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _tables_revision():
    files = stats.daily_files(APICK_DAYS)
    if not files:
        raise FileNotFoundError("no counter-picker tables yet")
    return tuple((f.name, _revision(f)) for f in files)


def _heroes():
    names = load_heroes()
    # Original OpenDota hero names provide the Steam portrait filename. These
    # are optional decorations; unknown/missing assets never affect statistics.
    try:
        rows = json.loads(HEROES_PATH.read_text("utf-8"))
    except (ValueError, OSError):
        rows = []
    portraits = {}
    for row in rows:
        internal = row.get("name", "")
        if isinstance(internal, str) and re.fullmatch(r"npc_dota_hero_[a-z0-9_]+", internal):
            slug = internal.removeprefix("npc_dota_hero_")
            portraits[row.get("id")] = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png"
    return [{"id": int(key), "name": value, "portrait": portraits.get(key, "")} for key, value in names.items()]


def _snapshot(rank_min, rank_max, patch):
    # Cache by data and hero catalogue revisions plus every filter. Only the
    # owner-maintained dataset is read; nothing is fetched on user requests.
    key = (_tables_revision(), _revision(HEROES_PATH), rank_min, rank_max, patch)
    with _lock:
        if key not in _cache:
            _cache[key] = snapshot(load_heroes(), rank_min=rank_min, rank_max=rank_max, patch=patch)
            while len(_cache) > 12:
                _cache.popitem(last=False)
        _cache.move_to_end(key)
        return _cache[key]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, status, obj):
        payload = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path not in ("/api/heroes", "/api/sample"):
            return super().do_GET()
        try:
            if url.path == "/api/heroes":
                return self._json(200, {"heroes": _heroes()})
            query = parse_qs(url.query, keep_blank_values=True)
            if any(len(values) != 1 for values in query.values()) or set(query) - {"rank_min", "rank_max", "patch"}:
                return self._json(400, {"error": "Invalid filter."})
            def rank(key):
                value = query.get(key, [""])[0]
                if not value:
                    return None
                if not value.isdigit() or not 10 <= int(value) <= 80:
                    raise ValueError("Invalid rank filter.")
                return int(value)
            rank_min, rank_max = rank("rank_min"), rank("rank_max")
            if rank_min is not None and rank_max is not None and rank_min > rank_max:
                raise ValueError("Invalid rank range.")
            patch = query.get("patch", [""])[0].strip() or None
            if patch and not re.fullmatch(r"[0-9]+\.[0-9]+[a-z]?", patch):
                raise ValueError("Invalid patch. Use a value such as 7.41.")
            return self._json(200, _snapshot(rank_min, rank_max, patch))
        except FileNotFoundError:
            return self._json(503, {"error": "Local data is missing. Run the counter-picker update first: python -m predictor.counterpick.update"})
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        except Exception:
            return self._json(500, {"error": "Could not read the local dataset. Check that the project dependencies and data are available."})


def main():
    parser = argparse.ArgumentParser(description="Start the local counter-picker")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Counter-picker: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
