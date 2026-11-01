"""Monte Carlo of a TI-style tournament.

Format (close to recent TIs; the exact rules change every year):
  1. Swiss stage, all Bo3: teams with the same record play each other; 3 wins = through,
     3 losses = out, until 8 teams are through (16 teams -> exactly 8).
  2. Playoffs: 8-team double elimination, Bo3, grand final Bo5.

Map win chances come from the pro match predictor (team + player ratings, form, head-to-head),
turned into series chances with series_win(). Simulating the event thousands of times gives
each team's chance to win it.
"""

import numpy as np

from ..pro.model import ProModel, series_win


def map_matrix(pro: ProModel, teams: list[int]) -> np.ndarray:
    """P[i, j] = chance team i wins one map against team j (sides averaged)."""
    n = len(teams)
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = pro.predict(teams[i], teams[j])
            P[j, i] = 1 - P[i, j]
    return P


def _series(P: np.ndarray, best_of: int) -> np.ndarray:
    return np.vectorize(lambda p: series_win(p, best_of))(P)


def _swiss(n: int, S3: np.ndarray, rng) -> list[int]:
    """Swiss stage. Returns 8 qualifiers, best record first."""
    wins, losses = np.zeros(n, int), np.zeros(n, int)
    through, out = [], set()
    while len(through) < min(8, n) and len(through) + len(out) < n:
        active = [t for t in range(n) if t not in out and t not in through]
        groups: dict = {}
        for t in active:
            groups.setdefault((wins[t], losses[t]), []).append(t)
        leftover = []
        for record in sorted(groups, key=lambda r: (-r[0], r[1])):
            pool = leftover + list(rng.permutation(groups[record]))
            leftover = [pool.pop()] if len(pool) % 2 else []
            for a, b in zip(pool[::2], pool[1::2]):
                winner, loser = (a, b) if rng.random() < S3[a, b] else (b, a)
                wins[winner] += 1
                losses[loser] += 1
        for t in leftover:  # odd one out gets a bye (counts as a win)
            wins[t] += 1
        for t in active:
            if wins[t] >= 3:
                through.append(t)
            elif losses[t] >= 3:
                out.add(t)
    if len(through) < 8:  # small fields: fill up with the best of the rest
        rest = sorted((t for t in range(n) if t not in through), key=lambda t: (-wins[t], losses[t], rng.random()))
        through += rest[: 8 - len(through)]
    return sorted(through, key=lambda t: (losses[t], rng.random()))[:8]


def _double_elim(seeds: list[int], S3: np.ndarray, S5: np.ndarray, rng) -> int:
    """8-team double elimination; seeds[0] is the top seed. Returns the champion."""
    def play(a, b, S=S3):
        return (a, b) if rng.random() < S[a, b] else (b, a)

    s = seeds
    w1, l1 = play(s[0], s[7])
    w2, l2 = play(s[3], s[4])
    w3, l3 = play(s[1], s[6])
    w4, l4 = play(s[2], s[5])
    ua, la = play(w1, w2)  # upper semis
    ub, lb = play(w3, w4)
    ubw, ubl = play(ua, ub)  # upper final
    x1, _ = play(l1, l2)  # lower round 1
    x2, _ = play(l3, l4)
    y1, _ = play(x1, lb)  # lower round 2: cross over with upper-semi losers
    y2, _ = play(x2, la)
    z, _ = play(y1, y2)  # lower semi
    lbw, _ = play(z, ubl)  # lower final
    champ, _ = play(ubw, lbw, S5)  # grand final, Bo5
    return champ


def title_odds(P: np.ndarray, n_sims: int = 10_000, seed: int = 0) -> np.ndarray:
    """Chance each team wins the tournament (same order as P)."""
    if len(P) < 8:
        raise ValueError("a TI-style event needs at least 8 teams")
    rng = np.random.default_rng(seed)
    S3, S5 = _series(P, 3), _series(P, 5)
    wins = np.zeros(len(P))
    for _ in range(n_sims):
        wins[_double_elim(_swiss(len(P), S3, rng), S3, S5, rng)] += 1
    return wins / n_sims
