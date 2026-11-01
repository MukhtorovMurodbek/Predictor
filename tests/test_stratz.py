import json

import pytest
import requests

from predictor.counterpick import stratz


class Response:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.payload = payload if payload is not None else {"data": {"__typename": "Query"}}

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_query_uses_bearer_header_and_bounded_requests():
    session = Session(Response())
    result = stratz.StratzClient("example-token", session=session).query(stratz.AUTH_QUERY)
    assert result == {"__typename": "Query"}
    url, args = session.calls[0]
    assert url == stratz.ENDPOINT
    assert args["headers"]["Authorization"] == "Bearer example-token"
    assert "example-token" not in json.dumps(args["json"])
    assert args["timeout"] == stratz.TIMEOUT_SECONDS
    assert args["allow_redirects"] is False


@pytest.mark.parametrize("response", [
    Response(403, {"message": "example-token"}),
    Response(429, {"message": "example-token"}),
    Response(payload={"errors": [{"message": "example-token"}]}),
    Response(payload=ValueError("example-token")),
    requests.ConnectionError("example-token"),
])
def test_failures_do_not_expose_token(response):
    session = Session(response)
    with pytest.raises(stratz.StratzError) as exc:
        stratz.StratzClient("example-token", session=session).query(stratz.AUTH_QUERY)
    assert "example-token" not in str(exc.value)


def test_env_precedes_ignored_dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("STRATZ_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("STRATZ_TOKEN", "from-env")
    assert stratz.read_token(path) == "from-env"
    monkeypatch.delenv("STRATZ_TOKEN")
    assert stratz.read_token(path) == "from-file"
    with pytest.raises(stratz.StratzError, match="missing"):
        stratz.read_token(tmp_path / "missing")


def test_inspect_schema_and_report_only_safe_names(capsys, monkeypatch):
    root = Response(payload={"data": {"__schema": {"queryType": {"name": "Query"}}}})
    query_type = Response(payload={"data": {"__type": {"fields": [
        {"name": "heroStats", "type": {"kind": "OBJECT", "name": "HeroStats"}}
    ]}}})
    stats_type = Response(payload={"data": {"__type": {"fields": [
        {"name": "matchUp", "type": {"kind": "LIST", "ofType": {"name": "HeroMatchup"}},
         "args": [{"name": "heroId", "type": {"name": "Short"}},
                  {"name": "bracketBasicIds", "type": {"kind": "LIST", "ofType": {"name": "RankBracketBasicEnum"}}},
                  {"name": "unsafe secret string", "type": {"name": "String"}}]}
    ]}}})
    session = Session(Response(), root, query_type, stats_type)
    client = stratz.StratzClient("example-token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "example-token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main([]) == 0
    out = capsys.readouterr().out
    assert "connection succeeded" in out
    assert "bracketBasicIds: RankBracketBasicEnum" in out
    assert "example-token" not in out
    assert "unsafe secret string" not in out


def test_no_schema_access_is_reported_without_claiming_filter_success(capsys, monkeypatch):
    session = Session(Response(), Response(payload={"errors": [{"message": "example-token"}]}))
    client = stratz.StratzClient("example-token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "example-token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main([]) == 2
    out = capsys.readouterr().out
    assert "connection succeeded" in out and "Schema inspection unavailable" in out
    assert "example-token" not in out


def _matchup_response():
    return Response(payload={"data": {"heroStats": {"matchUp": [
        {"heroId": 2, "vs": [
            {"heroId1": 2, "heroId2": 12, "week": 2965, "bracketBasicIds": "UNCALIBRATED", "winCount": 508, "matchCount": 915},
            {"heroId1": 2, "heroId2": 3, "week": 2965, "bracketBasicIds": "UNCALIBRATED", "winCount": 30, "matchCount": 80},
        ]}
    ]}}})


def test_selected_matchup_preserves_raw_discrepant_bracket(capsys, monkeypatch):
    session = Session(Response(), _matchup_response())
    client = stratz.StratzClient("example-token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "example-token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main(["--hero-id", "2", "--opponent-id", "12", "--bracket", "HERALD_GUARDIAN"]) == 0
    output = capsys.readouterr().out
    assert "bracket=HERALD_GUARDIAN" in output
    assert "bracketBasicIds=UNCALIBRATED" in output
    assert "heroId1=2, heroId2=12, week=2965" in output
    assert "winCount=508, matchCount=915" in output
    assert "unverified" in output and "heroId2=3" not in output
    assert len(session.calls) == 2  # one connection check and one matchup query
    assert session.calls[1][1]["json"]["variables"] == {"hero": 2, "brackets": ["HERALD_GUARDIAN"]}
    assert "take: 200" in session.calls[1][1]["json"]["query"]
    assert "example-token" not in output


def test_matchup_without_bracket_does_not_claim_all_rank(capsys, monkeypatch):
    session = Session(Response(), _matchup_response())
    client = stratz.StratzClient("example-token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "example-token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main(["--hero-id", "2", "--opponent-id", "12"]) == 0
    output = capsys.readouterr().out
    assert "bracket=not supplied" in output
    assert "all-rank" not in output
    assert session.calls[1][1]["json"]["variables"]["brackets"] is None


def test_matchup_no_row_is_explicit(capsys, monkeypatch):
    session = Session(Response(), _matchup_response())
    client = stratz.StratzClient("example-token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "example-token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main(["--hero-id", "2", "--opponent-id", "9"]) == 2
    assert "No matching opponent row" in capsys.readouterr().out


def test_matchup_flags_must_be_complete_and_distinct(monkeypatch):
    monkeypatch.setattr(stratz, "read_token", lambda: pytest.fail("No request should be made"))
    for args in (["--hero-id", "2"], ["--bracket", "DIVINE_IMMORTAL"], ["--hero-id", "2", "--opponent-id", "2"]):
        with pytest.raises(SystemExit) as exc:
            stratz.main(args)
        assert exc.value.code == 2


def test_unexpected_provider_bracket_is_never_printed(capsys, monkeypatch):
    response = _matchup_response()
    response.payload["data"]["heroStats"]["matchUp"][0]["vs"][0]["bracketBasicIds"] = "secret_token"
    session = Session(Response(), response)
    client = stratz.StratzClient("secret_token", session=session)
    monkeypatch.setattr(stratz, "read_token", lambda: "secret_token")
    monkeypatch.setattr(stratz, "StratzClient", lambda token: client)
    assert stratz.main(["--hero-id", "2", "--opponent-id", "12"]) == 1
    captured = capsys.readouterr()
    assert "secret_token" not in captured.out + captured.err
