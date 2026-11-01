"""Train (and test) the live in-game win predictor -> models/live.pkl

    python -m predictor.live.train

Needs pro matches and their timelines: python -m predictor.data.collect pro
"""

import argparse
import time

import numpy as np
import pandas as pd

from ..data import TIMELINES_PATH
from ..pro.dataset import load_pro_matches, replay, training_mask
from ..pro.features import DRAFT_FEATURES
from ..pro.model import fit_classifiers
from .model import MODEL_PATH, LiveModel, build_rows, hide_inputs, minute_report, new_classifier


def train(log=print) -> LiveModel:
    t0 = time.time()
    pro = load_pro_matches()
    if not TIMELINES_PATH.exists():
        raise FileNotFoundError("no timelines yet, run: python -m predictor.data.collect timelines")
    timelines = pd.read_parquet(TIMELINES_PATH)
    pre, _ = replay(pro)
    log(f"replayed {len(pro):,} pro matches in {time.time() - t0:.0f} s")

    usable = training_mask(pro)
    X, y, idx = build_rows(pro, pre, timelines)
    keep = usable[idx]
    X, y, idx = X[keep].reset_index(drop=True), y[keep], idx[keep]
    log(f"{len(X):,} snapshots from {len(np.unique(idx)):,} matches")

    # test on the newest 20% of matches
    match_ids = np.unique(idx)
    split = match_ids[int(len(match_ids) * 0.8)]
    train_rows, test_rows = idx < split, idx >= split
    clf = new_classifier().fit(hide_inputs(X[train_rows]), y[train_rows])

    Xt, yt, minutes = X[test_rows], y[test_rows], X.loc[test_rows, "minute"].to_numpy()
    p_live = clf.predict_proba(Xt)[:, 1]
    # baselines: the pre-game pro model alone, and the gold lead alone
    pro_train = usable & (np.arange(len(pro)) < split)
    _, pre_clf = fit_classifiers(pre[pro_train], pro.loc[pro_train, "radiant_win"].to_numpy().astype(int))
    p_pre = pre_clf.predict_proba(Xt[DRAFT_FEATURES])[:, 1]
    gold_clf = new_classifier().fit(X.loc[train_rows, ["minute", "gold_adv"]], y[train_rows])
    p_gold = gold_clf.predict_proba(Xt[["minute", "gold_adv"]])[:, 1]

    since = pd.to_datetime(pro["start_time"].iloc[split], unit="s").date()
    lines = [f"tested on {test_rows.sum():,} snapshots of {len(np.unique(idx[test_rows])):,} newer pro matches (from {since})", ""]
    for name, p in (("live model", p_live), ("gold lead only", p_gold), ("pre-game model only", p_pre)):
        lines += minute_report(yt, np.clip(p, 1e-6, 1 - 1e-6), minutes, name) + [""]
    stats = {
        f"acc_{m}": _acc(yt[(minutes >= m) & (minutes < m + 5)], p_live[(minutes >= m) & (minutes < m + 5)] > 0.5)
        for m in (0, 10, 20, 30)
    }

    final = new_classifier().fit(hide_inputs(X), y)
    model = LiveModel(clf=final, n_matches=len(match_ids), report="\n".join(lines).strip(), extra={"stats": stats})
    model.save()
    log(f"done in {time.time() - t0:.0f} s")
    return model


def _acc(y, pred) -> float:
    return float((y == pred).mean()) if len(y) else float("nan")


def main(argv=None):
    argparse.ArgumentParser(description="Train the live in-game win predictor").parse_args(argv)
    model = train()
    print(model.report)
    print(f"\nsaved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
