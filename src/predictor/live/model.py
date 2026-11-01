"""Live (in-game) win predictor for pro matches.

Inputs = everything the pro match predictor knows before the game (team and player
ratings, form, draft) + the scoreboard at minute m (gold lead, towers, Roshan...). One
gradient-boosting model covers every minute: early on it leans on team strength, later on
the gold lead and buildings.

Training uses a snapshot every 2 minutes of every pro match since 2023, with some inputs
randomly hidden so the model still works when the live feed or the person asking doesn't
know them (no heroes, no Roshan count...).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from .. import MODELS_DIR
from ..io import atomic_write
from ..pro.features import DRAFT_FEATURES, PREGAME_FEATURES
from .features import STATE_FEATURES, mask_optional, series_from_timeline

MODEL_PATH = MODELS_DIR / "live.pkl"
FEATURES = DRAFT_FEATURES + STATE_FEATURES
HERO_ONLY = [f for f in DRAFT_FEATURES if f not in PREGAME_FEATURES]


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    )


def build_rows(pro: pd.DataFrame, pre: pd.DataFrame, timelines: pd.DataFrame, step: int = 2):
    """Snapshot rows for every match that has both pre-match features and a timeline.
    Returns (X DataFrame, y, match row index into `pro`)."""
    pre_by_id = dict(zip(pro["match_id"].to_numpy(), range(len(pro))))
    pre_arr = pre[DRAFT_FEATURES].to_numpy(dtype=np.float32)
    wins = pro["radiant_win"].to_numpy()
    X, y, idx = [], [], []
    for row in timelines.itertuples(index=False):
        i = pre_by_id.get(row.match_id)
        if i is None or row.gold is None or len(row.gold) < 2:
            continue
        states = series_from_timeline(row.gold, row.events, step)
        block = np.empty((len(states), len(FEATURES)), dtype=np.float32)
        block[:, : len(DRAFT_FEATURES)] = pre_arr[i]
        block[:, len(DRAFT_FEATURES):] = [[s[f] for f in STATE_FEATURES] for s in states]
        X.append(block)
        y.append(np.full(len(states), wins[i]))
        idx.append(np.full(len(states), i))
    return pd.DataFrame(np.vstack(X), columns=FEATURES), np.concatenate(y).astype(int), np.concatenate(idx)


def hide_inputs(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Training-time masking: no heroes in 30% of rows, no objectives in 50%."""
    rng = np.random.default_rng(seed)
    arr = X.to_numpy(copy=True)
    arr = mask_optional(arr, FEATURES, rng, 0.5)
    rows = rng.random(len(arr)) < 0.3
    arr[np.ix_(rows, [FEATURES.index(c) for c in HERO_ONLY])] = np.nan
    return pd.DataFrame(arr, columns=FEATURES)


def minute_report(y, p, minutes, name) -> list[str]:
    lines = []
    for lo, hi in [(0, 5), (10, 15), (20, 25), (30, 35), (40, 999)]:
        s = (minutes >= lo) & (minutes < hi)
        if s.any():
            label = f"min {lo}+" if hi == 999 else f"min {lo}-{hi - 1}"
            lines.append(f"{name:<22} {label:<10} acc {accuracy_score(y[s], p[s] > 0.5):.1%}  logloss {log_loss(y[s], p[s], labels=[0, 1]):.3f}")
    return lines


@dataclass
class LiveModel:
    clf: HistGradientBoostingClassifier
    n_matches: int
    report: str = ""
    extra: dict = field(default_factory=dict)

    def predict_radiant(self, pre: dict | None, state: dict) -> float:
        """P(Radiant wins). `pre` = pro-model features (may lack hero features or be None),
        `state` = scoreboard features (missing keys are treated as unknown)."""
        row = {f: np.nan for f in FEATURES}
        row.update({k: v for k, v in (pre or {}).items() if k in row})
        row.update({k: v for k, v in state.items() if k in row and v is not None})
        return float(self.clf.predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])

    def save(self, path: Path = MODEL_PATH) -> None:
        with atomic_write(path) as tmp, open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "LiveModel":
        # Only load model files you made yourself: pickle can run code from the file.
        with open(path, "rb") as fh:
            return pickle.load(fh)
