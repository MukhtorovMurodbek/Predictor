"""How good is the pro predictor? Train on older matches, test on the newest 20%.

    python -m predictor.pro.evaluate
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .dataset import load_pro_matches, replay, training_mask
from .features import DRAFT_FEATURES, PREGAME_FEATURES
from .model import fit_classifiers

TOP_TIERS = ["premium", "professional"]


def _line(name, y, p) -> str:
    return f"{name:<16} acc {accuracy_score(y, p > 0.5):.3f}  logloss {log_loss(y, p, labels=[0, 1]):.4f}  brier {brier_score_loss(y, p):.4f}"


def evaluate_frame(df: pd.DataFrame, X: pd.DataFrame, test_frac: float = 0.2) -> tuple[str, dict]:
    """Text report + headline numbers (for the bot's /pro intro)."""
    y = df["radiant_win"].to_numpy().astype(int)
    usable = training_mask(df)
    idx = np.flatnonzero(usable)
    split = idx[int(len(idx) * (1 - test_frac))]
    train = usable & (np.arange(len(df)) < split)
    test = np.arange(len(df)) >= split
    pregame, draft = fit_classifiers(X[train], y[train])
    p_pre = pregame.predict_proba(X.loc[test, PREGAME_FEATURES])[:, 1]
    p_draft = draft.predict_proba(X.loc[test, DRAFT_FEATURES])[:, 1]
    yt = y[test]
    top = df.loc[test, "tier"].isin(TOP_TIERS).to_numpy()

    since = pd.to_datetime(df["start_time"].iloc[split], unit="s").date()
    lines = [f"train {train.sum():,} matches, test {test.sum():,} (from {since})", ""]
    lines.append("all test matches")
    lines.append(_line("always Radiant", yt, np.full(len(yt), y[train].mean())))
    lines.append(_line("team Elo only", yt, 1 / (1 + 10 ** (-X.loc[test, "team_elo_diff"].to_numpy() / 400))))
    lines.append(_line("before draft", yt, p_pre))
    lines.append(_line("after draft", yt, p_draft))
    lines += ["", f"professional + premium tiers ({top.sum():,} matches)"]
    lines.append(_line("before draft", yt[top], p_pre[top]))
    lines.append(_line("after draft", yt[top], p_draft[top]))

    conf = np.maximum(p_draft, 1 - p_draft)
    sure = conf >= 0.8
    stats = {
        "acc_top": accuracy_score(yt[top], p_draft[top] > 0.5),
        "acc_top_pregame": accuracy_score(yt[top], p_pre[top] > 0.5),
        "acc_sure": accuracy_score(yt[sure], p_draft[sure] > 0.5) if sure.any() else float("nan"),
        "share_sure": sure.mean(),
        "test_since": str(since),
    }
    lines += ["", "when the model says... (after draft)"]
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        s = (conf >= lo) & (conf < hi)
        if s.any():
            lines.append(
                f"{lo:.0%}-{min(hi, 1):.0%}: right {accuracy_score(yt[s], p_draft[s] > 0.5):.0%} of the time ({s.mean():.0%} of matches)"
            )
    return "\n".join(lines), stats


def evaluate(test_frac: float = 0.2) -> str:
    df = load_pro_matches()
    X, _ = replay(df)
    return evaluate_frame(df, X, test_frac)[0]


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the pro match predictor on the newest matches")
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args(argv)
    print(evaluate(args.test_frac))


if __name__ == "__main__":
    main()
