"""Counter-picker = plain statistics over recent public games. No trained model.

For every hero, and every pair of heroes, we count games and wins:
    vs[a, b]    a on one team, b on the other   -> a's win rate against b
    with_[a, b] a and b on the same team        -> a's win rate together with b

Counts are kept per day (data/counterpick/YYYY-MM-DD.npz), so the last WINDOW_DAYS days can
be summed quickly and old days simply fall out of the window.

The "edge" of a matchup works like Dotabuff's advantage: the actual win rate minus what the
two heroes' usual win rates predict. +0.05 = 5 win-rate points better than expected.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DATA_DIR
from ..data import hero_arrays

DAILY_DIR = DATA_DIR / "counterpick"
WINDOW_DAYS = 90  # Asilbek: use the last 3 months
MIN_GAMES = 200  # matchups with fewer games are hidden or flagged (±7 points of noise at 200)
N_HEROES = 160
CHUNK = 200_000


def _pair_counts(A: np.ndarray, B: np.ndarray, a_won: np.ndarray):
    """games[a, b] and wins-for-a[a, b] over every (hero in A, hero in B) pair of each row."""
    H = N_HEROES
    games = np.zeros(H * H)
    wins = np.zeros(H * H)
    for s in range(0, len(A), CHUNK):
        a, b, y = A[s : s + CHUNK], B[s : s + CHUNK], a_won[s : s + CHUNK]
        idx = (a[:, :, None] * H + b[:, None, :]).reshape(len(a), -1)
        games += np.bincount(idx.ravel(), minlength=H * H)
        wins += np.bincount(idx.ravel(), weights=np.repeat(y, idx.shape[1]), minlength=H * H)
    return games.reshape(H, H), wins.reshape(H, H)


@dataclass
class Counts:
    """Raw game and win counts. Adding two Counts gives the counts of both periods."""

    games: np.ndarray  # (H,)
    wins: np.ndarray  # (H,)
    vs_games: np.ndarray  # (H, H)
    vs_wins: np.ndarray  # (H, H)
    with_games: np.ndarray  # (H, H)
    with_wins: np.ndarray  # (H, H)
    n_matches: int

    @classmethod
    def empty(cls) -> "Counts":
        z1, z2 = np.zeros(N_HEROES), np.zeros((N_HEROES, N_HEROES))
        return cls(z1, z1.copy(), z2, z2.copy(), z2.copy(), z2.copy(), 0)

    @classmethod
    def from_matches(cls, df: pd.DataFrame) -> "Counts":
        if df.empty:
            return cls.empty()
        R, D, y = hero_arrays(df)
        lose = 1.0 - y
        games = np.bincount(R.ravel(), minlength=N_HEROES) + np.bincount(D.ravel(), minlength=N_HEROES)
        wins = np.bincount(R.ravel(), weights=np.repeat(y, 5), minlength=N_HEROES) + np.bincount(
            D.ravel(), weights=np.repeat(lose, 5), minlength=N_HEROES
        )
        g1, w1 = _pair_counts(R, D, y)
        g2, w2 = _pair_counts(D, R, lose)
        h1, x1 = _pair_counts(R, R, y)
        h2, x2 = _pair_counts(D, D, lose)
        with_games, with_wins = h1 + h2, x1 + x2
        np.fill_diagonal(with_games, 0)
        np.fill_diagonal(with_wins, 0)
        return cls(games.astype(float), wins, g1 + g2, w1 + w2, with_games, with_wins, len(df))

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.games + other.games, self.wins + other.wins, self.vs_games + other.vs_games,
            self.vs_wins + other.vs_wins, self.with_games + other.with_games,
            self.with_wins + other.with_wins, self.n_matches + other.n_matches,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in ("games", "wins", "vs_games", "vs_wins", "with_games", "with_wins")},
                            n_matches=self.n_matches)

    @classmethod
    def load(cls, path: Path) -> "Counts":
        z = np.load(path)
        return cls(z["games"], z["wins"], z["vs_games"], z["vs_wins"], z["with_games"], z["with_wins"], int(z["n_matches"]))


def day_path(day: str) -> Path:
    return DAILY_DIR / f"{day}.npz"


def daily_files(window_days: int = WINDOW_DAYS, today: datetime | None = None) -> list[Path]:
    today = today or datetime.now(timezone.utc)
    first = (today - timedelta(days=window_days)).strftime("%Y-%m-%d")
    return sorted(p for p in DAILY_DIR.glob("*.npz") if p.stem >= first)


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class MatchupStats:
    counts: Counts
    hero_names: dict[int, str]
    first_day: str = ""
    last_day: str = ""

    @classmethod
    def from_daily(cls, hero_names: dict[int, str], window_days: int = WINDOW_DAYS) -> "MatchupStats":
        files = daily_files(window_days)
        total = Counts.empty()
        for f in files:
            total = total + Counts.load(f)
        return cls(total, hero_names, files[0].stem if files else "", files[-1].stem if files else "")

    def __post_init__(self):
        c = self.counts
        with np.errstate(divide="ignore", invalid="ignore"):
            self.win_rate = np.where(c.games > 0, c.wins / np.maximum(c.games, 1), np.nan)
            lw = _logit(np.clip(np.nan_to_num(self.win_rate, nan=0.5), 0.01, 0.99))
            self.vs_rate = np.where(c.vs_games > 0, c.vs_wins / np.maximum(c.vs_games, 1), np.nan)
            self.with_rate = np.where(c.with_games > 0, c.with_wins / np.maximum(c.with_games, 1), np.nan)
            self.vs_edge = self.vs_rate - _sigmoid(lw[:, None] - lw[None, :])
            self.with_edge = self.with_rate - _sigmoid(lw[:, None] + lw[None, :])

    @property
    def n_matches(self) -> int:
        return self.counts.n_matches

    def _heroes(self, exclude=()) -> list[int]:
        return [h for h in self.hero_names if h not in set(exclude) and self.counts.games[h] > 0]

    def counters_for(self, enemy: int, top: int = 10, min_games: int = MIN_GAMES) -> pd.DataFrame:
        """Heroes with the biggest edge against `enemy`: edge, their win rate vs it, games."""
        cand = [h for h in self._heroes([enemy]) if self.counts.vs_games[h, enemy] >= min_games]
        out = pd.DataFrame({
            "hero": [self.hero_names[h] for h in cand],
            "edge": self.vs_edge[cand, enemy],
            "win_rate": self.vs_rate[cand, enemy],
            "games": self.counts.vs_games[cand, enemy].astype(int),
        })
        return out.sort_values("edge", ascending=False).head(top).reset_index(drop=True)

    def recommend(self, allies=(), enemies=(), excluded=(), top: int = 10, min_games: int = MIN_GAMES) -> pd.DataFrame:
        """Free heroes ranked by summed edge vs the enemies + with the allies. Pairs with fewer
        than `min_games` games count as 0 and are reported in `unknown`."""
        allies, enemies = list(allies), list(enemies)
        cand = np.array(self._heroes([*allies, *enemies, *excluded]))
        c = self.counts
        vs = self.vs_edge[np.ix_(cand, enemies)] if enemies else np.zeros((len(cand), 0))
        vs_ok = c.vs_games[np.ix_(cand, enemies)] >= min_games if enemies else np.zeros((len(cand), 0), bool)
        wi = self.with_edge[np.ix_(cand, allies)] if allies else np.zeros((len(cand), 0))
        wi_ok = c.with_games[np.ix_(cand, allies)] >= min_games if allies else np.zeros((len(cand), 0), bool)
        counter = np.where(vs_ok, vs, 0).sum(1)
        synergy = np.where(wi_ok, wi, 0).sum(1)
        out = pd.DataFrame({
            "hero": [self.hero_names[h] for h in cand],
            "counter": counter,
            "synergy": synergy,
            "score": counter + synergy,
            "games": (c.vs_games[np.ix_(cand, enemies)].sum(1) if enemies else 0) + (c.with_games[np.ix_(cand, allies)].sum(1) if allies else 0),
            "unknown": (~vs_ok).sum(1) + (~wi_ok).sum(1),
        })
        if enemies:
            masked = np.where(vs_ok, vs, -np.inf)
            best = masked.argmax(1)
            out["best_vs"] = [self.hero_names[enemies[i]] for i in best]
            out["best_vs_edge"] = np.where(vs_ok.any(1), vs[np.arange(len(cand)), best], np.nan)
        return out.sort_values("score", ascending=False).head(top).reset_index(drop=True)

    def hero_card(self, h: int, n: int = 4, min_games: int = MIN_GAMES) -> dict:
        """Win rate, best/worst matchups and best allies of one hero."""
        def ranked(edge_row, games_row, best):
            cand = [k for k in self._heroes([h]) if games_row[k] >= min_games]
            cand.sort(key=lambda k: edge_row[k], reverse=best)
            return [(self.hero_names[k], float(edge_row[k])) for k in cand[:n]]

        c = self.counts
        return {
            "win_rate": float(self.win_rate[h]) if c.games[h] else float("nan"),
            "games": int(c.games[h]),
            "strong_vs": ranked(self.vs_edge[h], c.vs_games[h], True),
            "weak_vs": ranked(self.vs_edge[h], c.vs_games[h], False),
            "allies": ranked(self.with_edge[h], c.with_games[h], True),
        }
