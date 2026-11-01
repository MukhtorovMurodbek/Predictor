"""Train (and test) the pro next-pick / next-ban predictor -> models/draft.pkl

    python -m predictor.draft.train

Takes a few minutes: it replays every pro draft since 2023 once.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..data import HEROES_PATH, TEAMS_PATH
from .model import MODEL_PATH, DraftModel, fit_classifier, load_records, replay, topk_report

TRAIN_MATCHES = 20_000  # most recent drafts used for training
TEST_MATCHES = 10_000  # newest drafts held out for the accuracy report
TEST_SAMPLE_EVERY = 4  # rank every free hero for every 4th test draft (memory)


def train(log=print) -> DraftModel:
    t0 = time.time()
    records = load_records()
    heroes = json.loads(HEROES_PATH.read_text("utf-8"))
    valid = np.array(sorted(h["id"] for h in heroes))
    cm_idx = [i for i, r in enumerate(records) if r.cm]
    test_from = cm_idx[-TEST_MATCHES]
    eval_train_from = cm_idx[max(0, len(cm_idx) - TEST_MATCHES - TRAIN_MATCHES)]
    final_train_from = cm_idx[-TRAIN_MATCHES]
    log(f"{len(records):,} drafts ({len(cm_idx):,} standard Captains Mode), replaying...")

    state, rows, test = replay(records, valid, train_from=eval_train_from, test_from=test_from, test_every=TEST_SAMPLE_EVERY)
    log(f"replayed in {time.time() - t0:.0f} s, {len(rows['y']):,} training rows")

    # 1) test: train on the drafts before the test period, rank heroes in newer drafts
    before = rows["match"] < test_from
    clf = fit_classifier(rows["X"][before], rows["y"][before])
    report, stats = topk_report(clf, test, len(np.unique(test["match"])))

    # 2) final model: the most recent drafts, including the test period
    recent = rows["match"] >= final_train_from
    clf = fit_classifier(rows["X"][recent], rows["y"][recent])

    teams = pd.read_parquet(TEAMS_PATH)
    model = DraftModel(
        state=state,
        clf=clf,
        valid_heroes=valid,
        order=records[cm_idx[-1]].order,
        team_names={int(r.team_id): (r.name or "", r.tag or "") for r in teams.itertuples()},
        hero_names={h["id"]: h["localized_name"] for h in heroes},
        n_matches=len(cm_idx),
        report=report,
        extra={"stats": stats},
    )
    model.save()
    log(f"done in {time.time() - t0:.0f} s")
    return model


def main(argv=None):
    argparse.ArgumentParser(description="Train the pro next-pick predictor").parse_args(argv)
    model = train()
    print(model.report)
    print(f"\nsaved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
