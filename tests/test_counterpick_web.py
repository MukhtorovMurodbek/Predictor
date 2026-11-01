"""HTTP integration tests for the local statistical counter-picker."""

import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd
import pytest

import predictor.data as data
from web import server


def _match(match_id, radiant_win, rank, patch, timestamp):
    return {
        "match_id": match_id,
        **{f"r{i}": i + 1 for i in range(5)},
        **{f"d{i}": i + 6 for i in range(5)},
        "radiant_win": radiant_win,
        "avg_rank_tier": rank,
        "patch": patch,
        "start_time": timestamp,
    }


@pytest.fixture
def local_site(tmp_path, monkeypatch):
    heroes_path = tmp_path / "heroes.json"
    heroes_path.write_text(json.dumps([
        {"id": i, "localized_name": f"Hero {i}", "name": f"npc_dota_hero_h{i}"}
        for i in range(1, 11)
    ]), encoding="utf-8")
    public_path = tmp_path / "public.parquet"
    pd.DataFrame([
        _match(1, True, 35, "7.41", 100),
        _match(2, False, 75, "7.41", 200),
        _match(3, False, 35, "7.40", 300),
    ]).to_parquet(public_path, index=False)

    monkeypatch.setattr(data, "HEROES_PATH", heroes_path)
    monkeypatch.setitem(data.SOURCES, "public", public_path)
    monkeypatch.setattr(server, "HEROES_PATH", heroes_path)
    monkeypatch.setattr(server, "PUBLIC_PATH", public_path)
    server._cache.clear()
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", public_path, heroes_path
    finally:
        httpd.shutdown()
        worker.join()
        httpd.server_close()
        server._cache.clear()


def _get(base, path):
    try:
        with urlopen(base + path, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_heroes_and_directional_results_with_rank_and_patch(local_site):
    base, _, _ = local_site
    status, result = _get(base, "/api/heroes")
    assert status == 200
    assert {hero["id"] for hero in result["heroes"]} == set(range(1, 11))
    assert next(hero for hero in result["heroes"] if hero["id"] == 1)["name"] == "Hero 1"

    status, result = _get(base, "/api/sample?rank_min=30&rank_max=35&patch=7.41")
    assert status == 200
    assert result["gamesScanned"] == 3
    assert result["gamesEligible"] == 1
    assert result["oldestTimestamp"] == result["newestTimestamp"] == 100
    pairs = {(row["hero_id"], row["enemy_id"]): row for row in result["pairwise"]}
    assert (pairs[1, 6]["wins"], pairs[1, 6]["matches"]) == (1, 1)
    assert (pairs[6, 1]["wins"], pairs[6, 1]["matches"]) == (0, 1)

    status, all_games = _get(base, "/api/sample")
    assert status == 200 and all_games["gamesEligible"] == 3
    pair = next(row for row in all_games["pairwise"] if row["hero_id"] == 1 and row["enemy_id"] == 6)
    assert (pair["wins"], pair["matches"]) == (1, 3)


@pytest.mark.parametrize("query", [
    "rank_min=81", "rank_min=50&rank_max=30", "rank_min=abc", "patch=not-a-patch",
    "rank_min=30&rank_min=40", "unknown=1",
])
def test_invalid_filters_return_400(local_site, query):
    status, body = _get(local_site[0], f"/api/sample?{query}")
    assert status == 400
    assert body["error"]


def test_missing_dataset_returns_503(local_site):
    base, public_path, _ = local_site
    public_path.unlink()
    status, body = _get(base, "/api/sample")
    assert status == 503
    assert "missing" in body["error"].lower()


def test_sample_cache_refreshes_when_dataset_changes(local_site):
    base, public_path, _ = local_site
    assert _get(base, "/api/sample")[1]["gamesEligible"] == 3
    before = public_path.stat().st_mtime_ns
    pd.DataFrame([_match(4, True, 35, "7.41", 400)]).to_parquet(public_path, index=False)
    os.utime(public_path, ns=(before + 2_000_000_000, before + 2_000_000_000))
    status, body = _get(base, "/api/sample")
    assert status == 200
    assert body["gamesEligible"] == 1
    assert body["oldestTimestamp"] == 400
