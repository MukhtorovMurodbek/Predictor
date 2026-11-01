"""Pro match predictor: who wins a pro map / series.

Two gradient-boosted models on the same chronological features:
    pregame  team + player ratings, form, head-to-head (before the draft)
    draft    the same + the 10 picked heroes (after the draft)

Sides (Radiant/Dire) usually aren't known when someone asks, so every prediction is the
average of both side assignments.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import MODELS_DIR
from ..data.teams import find_team
from .features import DRAFT_FEATURES, PREGAME_FEATURES, ProState

MODEL_PATH = MODELS_DIR / "pro.pkl"
DAY = 86400
ACTIVE_DAYS = 90  # teams that played within this many days of the newest match


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
    )


def series_win(p: float, best_of: int) -> float:
    """Chance to win a best-of-N series if every map is won with probability p."""
    need, q = best_of // 2 + 1, 1 - p
    from math import comb

    return sum(comb(need - 1 + k, k) * p**need * q**k for k in range(need))


@dataclass
class ProModel:
    state: ProState
    pregame: HistGradientBoostingClassifier
    draft: HistGradientBoostingClassifier
    team_names: dict[int, tuple[str, str]]  # team_id -> (name, tag)
    player_names: dict[int, str]
    hero_names: dict[int, str]
    n_matches: int
    report: str = ""  # evaluation summary from training time
    extra: dict = field(default_factory=dict)

    # ---- lookup --------------------------------------------------------------------------

    def team_name(self, team_id: int) -> str:
        return self.team_names.get(team_id, (f"team {team_id}", ""))[0]

    def player_name(self, account_id: int) -> str:
        return self.player_names.get(account_id, f"#{account_id}")

    def active_teams(self, days: int = ACTIVE_DAYS) -> list[int]:
        cutoff = self.state.t - days * DAY
        return [t for t, (when, _) in self.state.team_roster.items() if when >= cutoff]

    def find_team(self, query: str) -> int:
        """Team id for a name or tag like 'spirit', 'Team Liquid', 'GG'. Raises ValueError."""
        activity = {t: len(self.state.team_results[t]) for t in self.active_teams(365)}
        return find_team(query, self.team_names, activity)

    def roster(self, team_id: int) -> tuple[int, ...]:
        return self.state.team_roster[team_id][1]

    # ---- prediction ----------------------------------------------------------------------

    def _p_radiant(self, rt, dt, rh=None, dh=None) -> float:
        f = self.state.features(rt, dt, self.roster(rt), self.roster(dt), rh, dh)
        if rh is None:
            return float(self.pregame.predict_proba(pd.DataFrame([f])[PREGAME_FEATURES])[0, 1])
        return float(self.draft.predict_proba(pd.DataFrame([f])[DRAFT_FEATURES])[0, 1])

    def predict(self, team_a: int, team_b: int, heroes_a=None, heroes_b=None) -> float:
        """Chance that team_a wins one map against team_b (heroes optional, both or neither)."""
        if (heroes_a is None) != (heroes_b is None):
            raise ValueError("give heroes for both teams or for neither")
        p1 = self._p_radiant(team_a, team_b, heroes_a, heroes_b)
        p2 = self._p_radiant(team_b, team_a, heroes_b, heroes_a)
        return (p1 + (1 - p2)) / 2

    def elo(self, team_id: int) -> float:
        return self.state.team_elo.get(team_id, self.state.t)

    def recent_results(self, team_id: int, n: int = 10) -> list[tuple[float, int | None, bool]]:
        return self.state.team_results[team_id][-n:]

    def head_to_head(self, a: int, b: int, days: int = 365) -> tuple[int, int]:
        cutoff = self.state.t - days * DAY
        res = [won for t, opp, won in self.state.team_results[a] if opp == b and t >= cutoff]
        return sum(res), len(res) - sum(res)

    def top_teams(self, n: int = 15, min_games: int = 15) -> list[tuple[int, float]]:
        teams = [t for t in self.active_teams() if self.state.team_elo.games[t] >= min_games]
        return sorted(((t, self.elo(t)) for t in teams), key=lambda x: -x[1])[:n]

    def signature_heroes(self, team_id: int, n: int = 5) -> list[tuple[int, float]]:
        s = 1 / self.state.team_hero.weight(self.state.t)
        games = [(h, self.state.team_hero.get((team_id, h), s)[0]) for h in self.hero_names]
        return sorted(games, key=lambda x: -x[1])[:n]

    # ---- persistence ---------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "ProModel":
        # Only load model files you made yourself: pickle can run code from the file.
        with open(path, "rb") as fh:
            return pickle.load(fh)


def fit_classifiers(X: pd.DataFrame, y: np.ndarray) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingClassifier]:
    return new_classifier().fit(X[PREGAME_FEATURES], y), new_classifier().fit(X[DRAFT_FEATURES], y)
