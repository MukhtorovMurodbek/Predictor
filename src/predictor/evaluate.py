"""Does the counter model predict anything? Time-split evaluation vs simple baselines.

    python -m predictor.evaluate --source public

Trains on the oldest (1 - test_frac) of matches and tests on the newest ones.
"""

import argparse

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .counters import CounterModel
from .data import D_COLS, R_COLS, hero_arrays, load_heroes, load_matches


def one_hot_teams(R, D, H) -> sparse.csr_matrix:
    """+1 for each Radiant hero, -1 for each Dire hero."""
    n = len(R)
    rows = np.repeat(np.arange(n), 10)
    cols = np.concatenate([R, D], axis=1).ravel()
    vals = np.tile(np.r_[np.ones(5), -np.ones(5)], n)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, H))


def report(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print(f"{name:<28} acc {accuracy_score(y, p > 0.5):.4f}   log loss {log_loss(y, p, labels=[0, 1]):.4f}   brier {brier_score_loss(y, p):.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate draft models on newer held-out matches")
    p.add_argument("--source", choices=["public", "pro", "all"], default="public")
    p.add_argument("--patch")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--prior", type=float, default=150.0)
    args = p.parse_args(argv)

    df = load_matches(args.source, args.patch)
    split = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split], df.iloc[split:]
    Rtr, Dtr, ytr = hero_arrays(train)
    Rte, Dte, yte = hero_arrays(test)
    print(f"train {len(train)} matches, test {len(test)} (newer)\n")

    report("always Radiant (train rate)", yte, np.full(len(yte), ytr.mean()))

    heroes = load_heroes()
    H = max(max(heroes), int(df[R_COLS + D_COLS].to_numpy().max())) + 1
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(one_hot_teams(Rtr, Dtr, H), ytr)
    report("per-hero logistic reg.", yte, lr.predict_proba(one_hot_teams(Rte, Dte, H))[:, 1])

    model = CounterModel.fit(Rtr, Dtr, ytr, heroes, prior=args.prior)
    report("counter model (all terms)", yte, model.predict_proba(Rte, Dte))
    w = model.weights
    print(f"\ncounter model weights: intercept {w[0]:+.3f}, base {w[1]:.2f}, counter {w[2]:.2f}, synergy {w[3]:.2f}")


if __name__ == "__main__":
    main()
