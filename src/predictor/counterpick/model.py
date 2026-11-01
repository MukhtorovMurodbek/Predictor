"""Hero counter & synergy matrices.

Everything is in log-odds ("logit") space so effects add up:

    base[h]     how strong hero h is on its own (smoothed win rate)
    vs[a, b]    extra log-odds for hero a when hero b is on the enemy team
    with_[a, b] extra log-odds for hero a when hero b is on the same team

"Extra" means compared with what base[a] and base[b] alone would predict. Small samples are
pulled towards zero with a Bayesian prior worth `prior` games, so a 9-1 matchup isn't
treated as a hard counter.

A draft is scored as   intercept + w_base*Σbase + w_vs*Σcounters + w_syn*Σsynergy,
with the four numbers fitted by logistic regression, so the scores are win probabilities.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import MODELS_DIR

CHUNK = 200_000
EDGE_PRIOR = 100.0  # virtual games that pull small matchup samples towards "no edge"


def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pair_counts(A: np.ndarray, B: np.ndarray, a_won: np.ndarray, H: int):
    """games[a, b] and wins-for-a[a, b] over every (hero in A, hero in B) pair of each row."""
    games = np.zeros(H * H)
    wins = np.zeros(H * H)
    for s in range(0, len(A), CHUNK):
        a, b, y = A[s : s + CHUNK], B[s : s + CHUNK], a_won[s : s + CHUNK]
        idx = (a[:, :, None] * H + b[:, None, :]).reshape(len(a), -1)
        games += np.bincount(idx.ravel(), minlength=H * H)
        wins += np.bincount(idx.ravel(), weights=np.repeat(y, idx.shape[1]), minlength=H * H)
    return games.reshape(H, H), wins.reshape(H, H)


def _shrunk_effect(wins, games, p0, prior):
    """Log-odds difference between the smoothed observed rate and the expected rate p0."""
    p = (wins + prior * p0) / (games + prior)
    return logit(p) - logit(p0)


@dataclass
class CounterModel:
    hero_names: dict[int, str]
    hero_games: np.ndarray  # (H,)
    base: np.ndarray  # (H,)
    vs: np.ndarray  # (H, H)
    vs_games: np.ndarray  # (H, H)
    with_: np.ndarray  # (H, H)
    with_games: np.ndarray  # (H, H)
    weights: np.ndarray  # [intercept, w_base, w_vs, w_syn]
    n_matches: int = 0
    hero_wins: np.ndarray | None = None  # raw counts, for the Dotabuff-style edges below
    vs_wins: np.ndarray | None = None
    with_wins: np.ndarray | None = None

    # ---- fitting -------------------------------------------------------------------------

    @classmethod
    def fit(cls, R, D, y, hero_names: dict[int, str], prior: float = 150.0, folds: int = 5) -> "CounterModel":
        H = max(max(hero_names), int(R.max()), int(D.max())) + 1
        model = cls._fit_tables(R, D, y, hero_names, H, prior)

        # Fit the weights on out-of-fold terms. Tables built from the same games would have
        # memorised their noise and get inflated weights.
        terms = np.empty((len(y), 3))
        for idx in np.array_split(np.arange(len(y)), folds):
            keep = np.ones(len(y), dtype=bool)
            keep[idx] = False
            fold_model = cls._fit_tables(R[keep], D[keep], y[keep], hero_names, H, prior)
            terms[idx] = fold_model.draft_terms(R[idx], D[idx])
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(terms, y)
        model.weights = np.concatenate([lr.intercept_, lr.coef_[0]])
        return model

    @classmethod
    def _fit_tables(cls, R, D, y, hero_names, H, prior) -> "CounterModel":
        loss = 1.0 - y

        hero_games = np.bincount(R.ravel(), minlength=H) + np.bincount(D.ravel(), minlength=H)
        hero_wins = np.bincount(R.ravel(), weights=np.repeat(y, 5), minlength=H) + np.bincount(
            D.ravel(), weights=np.repeat(loss, 5), minlength=H
        )
        base = logit((hero_wins + prior * 0.5) / (hero_games + prior))

        g1, w1 = _pair_counts(R, D, y, H)
        g2, w2 = _pair_counts(D, R, loss, H)
        vs_games, vs_wins = g1 + g2, w1 + w2
        vs = _shrunk_effect(vs_wins, vs_games, sigmoid(base[:, None] - base[None, :]), prior)

        g1, w1 = _pair_counts(R, R, y, H)
        g2, w2 = _pair_counts(D, D, loss, H)
        with_games, with_wins = g1 + g2, w1 + w2
        with_ = _shrunk_effect(with_wins, with_games, sigmoid(base[:, None] + base[None, :]), prior)
        np.fill_diagonal(with_, 0.0)
        np.fill_diagonal(with_games, 0.0)

        np.fill_diagonal(with_wins, 0.0)
        return cls(
            hero_names, hero_games, base, vs, vs_games, with_, with_games, np.array([0, 1, 1, 1.0]), len(y),
            hero_wins=hero_wins, vs_wins=vs_wins, with_wins=with_wins,
        )

    # ---- scoring -------------------------------------------------------------------------

    @property
    def advantage(self) -> np.ndarray:
        """adv[a, b]: how much a counters b, using both directions of the matchup."""
        return (self.vs - self.vs.T) / 2

    def draft_terms(self, R, D) -> np.ndarray:
        """(n, 3) array: [Σbase diff, Σcounter advantage, Σsynergy diff] from Radiant's side."""
        base = self.base[R].sum(1) - self.base[D].sum(1)
        counter = self.advantage[R[:, :, None], D[:, None, :]].sum((1, 2))
        syn = (self.with_[R[:, :, None], R[:, None, :]].sum((1, 2)) - self.with_[D[:, :, None], D[:, None, :]].sum((1, 2))) / 2
        return np.stack([base, counter, syn], axis=1)

    def predict_proba(self, R, D) -> np.ndarray:
        """P(Radiant wins) for full 5v5 drafts."""
        return sigmoid(self.weights[0] + self.draft_terms(R, D) @ self.weights[1:])

    def win_rates(self, prior: float = 50.0) -> np.ndarray:
        """Each hero's overall win rate (pulled towards 50% if it has few games)."""
        if self.hero_wins is None:
            raise ValueError("this model file is too old, retrain it: /train counterpick")
        return (self.hero_wins + 0.5 * prior) / (self.hero_games + prior)

    def edges(self, prior: float = EDGE_PRIOR) -> tuple[np.ndarray, np.ndarray]:
        """Dotabuff-style matchup numbers, in win-rate points (0.05 = +5 points):

        vs[a, b]    a's win rate against b, minus what a's and b's usual win rates predict
        with_[a, b] a's win rate with b on its team, minus what their usual win rates predict

        Pairs with few games are pulled towards 0 (`prior` = how many virtual games).
        """
        if self.vs_wins is None:
            raise ValueError("this model file is too old, retrain it: /train counterpick")
        wr = self.win_rates()
        exp_vs = 0.5 + (wr[:, None] - wr[None, :]) / 2
        exp_with = np.clip(wr[:, None] + wr[None, :] - 0.5, 0.01, 0.99)
        vs = (self.vs_wins + prior * exp_vs) / (self.vs_games + prior) - exp_vs
        with_ = (self.with_wins + prior * exp_with) / (self.with_games + prior) - exp_with
        np.fill_diagonal(with_, 0.0)
        return vs, with_

    def recommend(self, allies=(), enemies=(), excluded=(), top: int = 10, min_games: int = 50) -> pd.DataFrame:
        """Rank free heroes by how much better than usual they do against these enemies and
        with these allies (sum of matchup edges). A hero being strong in general doesn't count."""
        taken = set(allies) | set(enemies) | set(excluded)
        cand = np.array([h for h in self.hero_names if h not in taken and self.hero_games[h] >= min_games])
        vs, with_ = self.edges()
        counter = vs[np.ix_(cand, list(enemies))] if enemies else np.zeros((len(cand), 0))
        syn = with_[np.ix_(cand, list(allies))] if allies else np.zeros((len(cand), 0))
        out = pd.DataFrame(
            {
                "hero": [self.hero_names[h] for h in cand],
                "counter": counter.sum(1),
                "synergy": syn.sum(1),
                "win_rate": self.win_rates()[cand],
                "games_vs": self.vs_games[np.ix_(cand, list(enemies))].sum(1) if enemies else 0,
            }
        )
        out["score"] = out["counter"] + out["synergy"]
        if enemies:
            best = counter.argmax(1)
            out["best_vs"] = [self.hero_names[enemies[i]] for i in best]
            out["best_vs_adv"] = counter[np.arange(len(cand)), best]
        return out.sort_values("score", ascending=False).head(top).reset_index(drop=True)

    def counters_for(self, enemy: int, top: int = 10, min_games: int = 30) -> pd.DataFrame:
        """Heroes that do best against `enemy` (edge in win-rate points, and the sample size)."""
        cand = [h for h in self.hero_names if h != enemy and self.vs_games[h, enemy] >= min_games]
        vs, _ = self.edges()
        out = pd.DataFrame(
            {"hero": [self.hero_names[h] for h in cand], "advantage": vs[cand, enemy], "games": self.vs_games[cand, enemy].astype(int)}
        )
        return out.sort_values("advantage", ascending=False).head(top).reset_index(drop=True)

    # ---- persistence ---------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(self.hero_names), dtype=np.int64)
        np.savez_compressed(
            path,
            hero_ids=ids,
            hero_labels=np.array([self.hero_names[i] for i in ids]),
            hero_games=self.hero_games,
            base=self.base,
            vs=self.vs,
            vs_games=self.vs_games,
            with_=self.with_,
            with_games=self.with_games,
            weights=self.weights,
            **{k: v for k, v in (("hero_wins", self.hero_wins), ("vs_wins", self.vs_wins), ("with_wins", self.with_wins)) if v is not None},
            n_matches=self.n_matches,
        )

    @classmethod
    def load(cls, path: Path) -> "CounterModel":
        z = np.load(path)
        names = dict(zip(z["hero_ids"].tolist(), z["hero_labels"].tolist()))
        return cls(
            names, z["hero_games"], z["base"], z["vs"], z["vs_games"], z["with_"], z["with_games"],
            z["weights"], int(z["n_matches"]),
            **{k: z[k] for k in ("hero_wins", "vs_wins", "with_wins") if k in z.files},
        )


def model_path(source: str) -> Path:
    return MODELS_DIR / f"counters_{source}.npz"
