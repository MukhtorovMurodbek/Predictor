"""Backtest: how much chance did we give each past TI's real champion?

    python -m predictor.tournament.backtest

For every The International in the data: rebuild the pro model from matches played *before*
the event started, simulate the event with its real teams, and compare with the real winner.
Takes a few minutes (one replay of pro history per TI). Saves data/tournament_backtest.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

from .. import DATA_DIR
from ..data import LEAGUES_PATH, PLAYERS_PATH, TEAMS_PATH
from ..pro.dataset import load_pro_matches, replay, training_mask
from ..pro.model import ProModel, fit_classifiers
from .sim import map_matrix, title_odds

REPORT_PATH = DATA_DIR / "tournament_backtest.json"


def find_tis(pro: pd.DataFrame) -> list[dict]:
    """Main events only (premium tier), with teams, start time and champion."""
    leagues = pd.read_parquet(LEAGUES_PATH)
    main = leagues[leagues["name"].str.fullmatch(r"The International 20\d\d", na=False)]
    out = []
    for lg in main.itertuples():
        m = pro[pro["leagueid"] == lg.leagueid].sort_values("start_time")
        m = m[m["radiant_team_id"].notna() & m["dire_team_id"].notna()]
        if len(m) < 20:
            continue
        final = m.iloc[-1]
        champion = int(final["radiant_team_id"] if final["radiant_win"] else final["dire_team_id"])
        teams = sorted({int(t) for t in pd.concat([m["radiant_team_id"], m["dire_team_id"]])})
        out.append({"name": lg.name, "leagueid": int(lg.leagueid), "start": int(m["start_time"].min()), "teams": teams, "champion": champion})
    return sorted(out, key=lambda x: x["start"])


def backtest(n_sims: int = 10_000, log=print) -> dict:
    pro = load_pro_matches()
    names = {int(r.team_id): (r.name or "", r.tag or "") for r in pd.read_parquet(TEAMS_PATH).itertuples()}
    players = pd.read_parquet(PLAYERS_PATH) if PLAYERS_PATH.exists() else pd.DataFrame(columns=["account_id", "name"])
    results = []
    for ti in find_tis(pro):
        before = pro[pro["start_time"] < ti["start"]].reset_index(drop=True)
        X, state = replay(before)
        use = training_mask(before)
        pregame, draft = fit_classifiers(X[use], before.loc[use, "radiant_win"].to_numpy().astype(int))
        model = ProModel(state, pregame, draft, names, dict(zip(players["account_id"], players["name"])), {}, len(before))
        teams = [t for t in ti["teams"] if t in state.team_roster]
        odds = title_odds(map_matrix(model, teams), n_sims)
        order = np.argsort(-odds)
        champ = teams.index(ti["champion"]) if ti["champion"] in teams else None
        res = {
            "name": ti["name"],
            "teams": len(teams),
            "champion": names.get(ti["champion"], ("?", ""))[0],
            "champion_odds": float(odds[champ]) if champ is not None else None,
            "champion_rank": int(np.where(order == champ)[0][0]) + 1 if champ is not None else None,
            "favourite": names.get(teams[order[0]], ("?", ""))[0],
            "favourite_odds": float(odds[order[0]]),
            "uniform_odds": 1 / len(teams),
        }
        results.append(res)
        log(f"{res['name']}: champion {res['champion']} had {res['champion_odds']:.1%} (#{res['champion_rank']} of {res['teams']}); "
            f"favourite {res['favourite']} {res['favourite_odds']:.1%}; random guess {res['uniform_odds']:.1%}")
    REPORT_PATH.write_text(json.dumps(results, indent=1), "utf-8")
    return {"results": results}


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest the tournament predictor on past TIs")
    p.add_argument("--sims", type=int, default=10_000)
    args = p.parse_args(argv)
    backtest(args.sims)


if __name__ == "__main__":
    main()
