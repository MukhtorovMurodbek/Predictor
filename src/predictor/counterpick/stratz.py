"""Read-only STRATZ connection and schema diagnostics.

The API credential stays on the server. This module does not download a dataset
or assume that rank and patch filters have the semantics their names suggest.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import get_key

from .. import ROOT

ENDPOINT = "https://api.stratz.com/graphql"
TIMEOUT_SECONDS = 25
NAME = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")
BRACKETS = {"UNCALIBRATED", "HERALD_GUARDIAN", "CRUSADER_ARCHON", "LEGEND_ANCIENT", "DIVINE_IMMORTAL"}

AUTH_QUERY = "query CounterpickConnection { __typename }"
ROOT_QUERY = "query CounterpickRoot { __schema { queryType { name } } }"
TYPE_QUERY = """query CounterpickType($name: String!) {
  __type(name: $name) {
    name
    fields {
      name
      args { name type { kind name ofType { kind name ofType { kind name ofType { kind name } } } } }
      type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
    }
  }
}"""
MATCHUP_QUERY = """query CounterpickMatchup($hero: Short, $brackets: [RankBracketBasicEnum!]) {
  heroStats {
    matchUp(heroId: $hero, bracketBasicIds: $brackets, take: 200) {
      heroId
      vs { heroId1 heroId2 week bracketBasicIds winCount matchCount }
    }
  }
}"""


class StratzError(Exception):
    """A sanitized error safe for terminal output."""


def read_token(env_file: Path | None = None) -> str:
    """Use STRATZ_TOKEN from the environment or the repository's ignored .env."""
    token = os.environ.get("STRATZ_TOKEN", "").strip()
    path = Path(env_file) if env_file is not None else ROOT / ".env"
    if not token and path.is_file():
        token = (get_key(str(path), "STRATZ_TOKEN") or "").strip()
    if not token:
        raise StratzError("STRATZ_TOKEN is missing. Set it privately in the environment or the ignored .env file.")
    if "\n" in token or "\r" in token:
        raise StratzError("STRATZ_TOKEN has an invalid format.")
    return token


class StratzClient:
    """Issue bounded GraphQL queries without exposing credentials or provider errors."""

    def __init__(self, token: str, *, session=None):
        if not isinstance(token, str) or not token.strip() or "\n" in token or "\r" in token:
            raise StratzError("STRATZ_TOKEN is missing or has an invalid format.")
        self._token = token.strip()
        self._session = session if session is not None else requests.Session()

    def query(self, query: str, variables: dict | None = None) -> dict:
        try:
            response = self._session.post(
                ENDPOINT,
                json={"query": query, "variables": variables or {}},
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": "STRATZ_API", "Accept": "application/json"},
                timeout=TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.RequestException:
            raise StratzError("Could not reach STRATZ. Check the network and try again.") from None

        if response.status_code != 200:
            if response.status_code in (401, 403):
                raise StratzError(f"STRATZ returned HTTP {response.status_code}; access was denied or blocked.")
            if response.status_code == 429:
                raise StratzError("STRATZ returned HTTP 429; the rate limit was reached.")
            raise StratzError(f"STRATZ returned HTTP {int(response.status_code)}.")
        try:
            payload = response.json()
        except (ValueError, requests.RequestException):
            raise StratzError("STRATZ returned an invalid JSON response.") from None
        if not isinstance(payload, dict) or payload.get("errors"):
            # Never relay provider messages: they may include request parameters or secrets.
            raise StratzError("STRATZ returned a GraphQL error for this request.")
        if not isinstance(payload.get("data"), dict):
            raise StratzError("STRATZ returned no usable GraphQL data.")
        return payload["data"]


def _name(value):
    return value if isinstance(value, str) and NAME.fullmatch(value) else None


def _base_type(type_info):
    while isinstance(type_info, dict):
        if name := _name(type_info.get("name")):
            return name
        type_info = type_info.get("ofType")
    return None


def _field(data, name):
    type_info = data.get("__type")
    if not isinstance(type_info, dict) or not isinstance(type_info.get("fields"), list):
        return None
    return next((item for item in type_info["fields"] if isinstance(item, dict) and item.get("name") == name), None)


def describe_matchup_schema(client: StratzClient) -> dict:
    """Return only safe type/argument names from live introspection."""
    root_data = client.query(ROOT_QUERY)
    root = _name((root_data.get("__schema") or {}).get("queryType", {}).get("name"))
    if not root:
        raise StratzError("STRATZ did not provide its GraphQL query type.")
    hero_field = _field(client.query(TYPE_QUERY, {"name": root}), "heroStats")
    hero_type = _base_type(hero_field.get("type")) if hero_field else None
    if not hero_type:
        raise StratzError("STRATZ introspection did not expose heroStats.")
    matchup = _field(client.query(TYPE_QUERY, {"name": hero_type}), "matchUp")
    if not matchup:
        raise StratzError("STRATZ introspection did not expose heroStats.matchUp.")
    arguments = []
    for argument in matchup.get("args", []):
        if isinstance(argument, dict) and (name := _name(argument.get("name"))):
            arguments.append({"name": name, "type": _base_type(argument.get("type"))})
    return {"query_type": root, "hero_stats_type": hero_type, "matchup_type": _base_type(matchup.get("type")), "arguments": arguments}


def matchup_rows(client: StratzClient, hero_id: int, opponent_id: int, bracket: str | None = None) -> list[dict]:
    """Return selected provider rows unchanged; do not aggregate or invert them."""
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32767
           for value in (hero_id, opponent_id)) or hero_id == opponent_id:
        raise StratzError("Choose two different positive hero IDs.")
    if bracket not in (None, "HERALD_GUARDIAN", "DIVINE_IMMORTAL"):
        raise StratzError("Unsupported bracket selection.")
    data = client.query(MATCHUP_QUERY, {"hero": hero_id, "brackets": [bracket] if bracket else None})
    stats = data.get("heroStats")
    groups = stats.get("matchUp") if isinstance(stats, dict) else None
    if not isinstance(groups, list):
        raise StratzError("STRATZ returned an invalid matchup response.")
    rows = []
    for group in groups:
        if not isinstance(group, dict) or group.get("heroId") != hero_id:
            continue
        for item in group.get("vs") or []:
            if not isinstance(item, dict) or item.get("heroId2") != opponent_id:
                continue
            returned_bracket = item.get("bracketBasicIds")
            valid_bracket = (returned_bracket is None or
                             (isinstance(returned_bracket, str) and returned_bracket in BRACKETS) or
                             (isinstance(returned_bracket, list) and all(isinstance(v, str) and v in BRACKETS for v in returned_bracket)))
            numbers = [item.get(key) for key in ("heroId1", "heroId2", "week", "winCount", "matchCount")]
            valid_numbers = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in numbers)
            if not valid_bracket or not valid_numbers:
                raise StratzError("STRATZ returned invalid matchup metadata.")
            rows.append({key: item.get(key) for key in ("heroId1", "heroId2", "week", "bracketBasicIds", "winCount", "matchCount")})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check STRATZ access and optionally inspect one matchup (read-only).")
    parser.add_argument("--hero-id", type=int, help="First hero ID as requested from STRATZ")
    parser.add_argument("--opponent-id", type=int, help="Opponent hero ID to display")
    parser.add_argument("--bracket", choices=("HERALD_GUARDIAN", "DIVINE_IMMORTAL"), help="Optional bracket argument")
    args = parser.parse_args(argv)
    if (args.hero_id is None) != (args.opponent_id is None) or (args.bracket and args.hero_id is None):
        parser.error("--hero-id and --opponent-id must be supplied together; --bracket also requires them")
    if args.hero_id is not None and (not 1 <= args.hero_id <= 32767 or not 1 <= args.opponent_id <= 32767 or args.hero_id == args.opponent_id):
        parser.error("choose two different positive hero IDs")
    try:
        client = StratzClient(read_token())
        data = client.query(AUTH_QUERY)
        if not _name(data.get("__typename")):
            raise StratzError("STRATZ returned an unexpected connection response.")
        print("STRATZ connection succeeded.")
        if args.hero_id is not None:
            rows = matchup_rows(client, args.hero_id, args.opponent_id, args.bracket)
            print(f"Requested heroId={args.hero_id}, opponentId={args.opponent_id}, bracket={args.bracket or 'not supplied'}.")
            print("Raw provider rows (rank, patch, time-window and win direction are unverified):")
            for row in rows:
                print(", ".join(f"{key}={value}" for key, value in row.items()))
            if not rows:
                print("No matching opponent row was returned.")
                return 2
            return 0
        try:
            schema = describe_matchup_schema(client)
        except StratzError as error:
            print(f"Schema inspection unavailable: {error}")
            return 2
        print("heroStats.matchUp arguments:")
        for argument in schema["arguments"]:
            print(f"  {argument['name']}: {argument['type'] or 'unknown'}")
        print("Rank, patch, time range, and counts still need a verified live matchup query.")
        return 0
    except StratzError as error:
        print(f"STRATZ connection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
