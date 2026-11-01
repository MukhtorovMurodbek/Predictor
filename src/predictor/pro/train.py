"""Train the pro match predictor on all pro matches and save models/pro.pkl.

    python -m predictor.pro.train

Also runs the evaluation (older matches -> newest 20%) and stores the report in the model,
so the bot can show it.
"""

import argparse
import json

import pandas as pd

from ..data import HEROES_PATH, PLAYERS_PATH, TEAMS_PATH
from .dataset import load_pro_matches, replay, training_mask
from .evaluate import evaluate_frame
from .model import MODEL_PATH, ProModel, fit_classifiers


def train() -> ProModel:
    df = load_pro_matches()
    X, state = replay(df)
    report, stats = evaluate_frame(df, X)
    use = training_mask(df)
    pregame, draft = fit_classifiers(X[use], df.loc[use, "radiant_win"].to_numpy().astype(int))

    teams = pd.read_parquet(TEAMS_PATH)
    players = pd.read_parquet(PLAYERS_PATH) if PLAYERS_PATH.exists() else pd.DataFrame(columns=["account_id", "name"])
    heroes = json.loads(HEROES_PATH.read_text("utf-8"))
    model = ProModel(
        state=state,
        pregame=pregame,
        draft=draft,
        team_names={int(r.team_id): (r.name or "", r.tag or "") for r in teams.itertuples()},
        player_names={int(r.account_id): r.name for r in players.itertuples()},
        hero_names={h["id"]: h["localized_name"] for h in heroes},
        n_matches=len(df),
        report=report,
        extra={"stats": stats},
    )
    model.save()
    return model


def main(argv=None):
    argparse.ArgumentParser(description="Train the pro match predictor").parse_args(argv)
    model = train()
    print(model.report)
    print(f"\nsaved -> {MODEL_PATH} ({model.n_matches:,} matches)")


if __name__ == "__main__":
    main()
