"""Does the counter model predict anything? Time-split evaluation vs simple baselines.

    python -m predictor.counterpick.evaluate --source public

Trains on the oldest (1 - test_frac) of matches and tests on the newest ones.
"""

import argparse

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ..data import D_COLS, R_COLS, hero_arrays, load_heroes, load_matches
from .model import CounterModel


def one_hot_teams(R, D, H) -> sparse.csr_matrix:
    """+1 for each Radiant hero, -1 for each Dire hero."""
    n = len(R)
    rows = np.repeat(np.arange(n), 10)
    cols = np.concatenate([R, D], axis=1).ravel()
    vals = np.tile(np.r_[np.ones(5), -np.ones(5)], n)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, H))


def score_line(name, y, p) -> str:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (
        f"{name:<15} acc {accuracy_score(y, p > 0.5):.3f}  "
        f"logloss {log_loss(y, p, labels=[0, 1]):.4f}  brier {brier_score_loss(y, p):.4f}"
    )


def evaluate(source: str = "public", patch: str | None = None, test_frac: float = 0.2, prior: float = 150.0) -> str:
    """Return a short text report comparing the counter model with two baselines."""
    df = load_matches(source, patch)
    split = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:split], df.iloc[split:]
    Rtr, Dtr, ytr = hero_arrays(train)
    Rte, Dte, yte = hero_arrays(test)
    lines = [f"train {len(train)} matches, test {len(test)} (newer)", ""]

    lines.append(score_line("always Radiant", yte, np.full(len(yte), ytr.mean())))

    heroes = load_heroes()
    H = max(max(heroes), int(df[R_COLS + D_COLS].to_numpy().max())) + 1
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(one_hot_teams(Rtr, Dtr, H), ytr)
    lines.append(score_line("per-hero only", yte, lr.predict_proba(one_hot_teams(Rte, Dte, H))[:, 1]))

    model = CounterModel.fit(Rtr, Dtr, ytr, heroes, prior=prior)
    lines.append(score_line("counter model", yte, model.predict_proba(Rte, Dte)))
    w = model.weights
    lines += ["", f"weights: intercept {w[0]:+.3f}, base {w[1]:.2f}, counter {w[2]:.2f}, synergy {w[3]:.2f}"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate draft models on newer held-out matches")
    p.add_argument("--source", choices=["public", "pro", "all"], default="public")
    p.add_argument("--patch")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--prior", type=float, default=150.0)
    args = p.parse_args(argv)
    print(evaluate(args.source, args.patch, args.test_frac, args.prior))


if __name__ == "__main__":
    main()
