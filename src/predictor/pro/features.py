"""Features for pro matches, computed chronologically.

ProState walks through matches oldest -> newest. For each match it first computes features
from what was known *before* the match, then updates itself with the result. The bot calls
the same `features()` with the current state, so training and live predictions are built
exactly the same way.

All features are "Radiant minus Dire" so a positive number favours Radiant.
"""

import math
from collections import defaultdict

import numpy as np

DAY = 86400.0
ELO_START = 1500.0
ELO_K = 32.0
ELO_HALF_LIFE_DAYS = 180  # an inactive team/player drifts back towards 1500
N_HEROES = 160

PREGAME_FEATURES = [
    "team_elo_diff",
    "player_elo_diff",
    "team_exp_diff",
    "player_exp_diff",
    "min_team_games",
    "form_diff",
    "h2h",
    "h2h_games",
]
DRAFT_FEATURES = PREGAME_FEATURES + [
    "hero_meta_diff",
    "player_hero_games_diff",
    "player_hero_wr_diff",
    "team_hero_games_diff",
    "counter_sum",
]


def logit(p):
    return math.log(p) - math.log1p(-p)


def rate_logit(games: float, wins: float, prior: float) -> float:
    """Win rate pulled towards 50% by `prior` virtual games, in log-odds."""
    return logit((wins + 0.5 * prior) / (games + prior))


class DecayedCounts:
    """Games and wins per key, where older results count less (given half-life).

    Counts are stored multiplied by exp(t / tau), so adding is O(1) and reading just
    multiplies by exp(-t / tau).
    """

    def __init__(self, half_life_days: float, t0: float):
        self.tau = half_life_days * DAY / math.log(2)
        self.t0 = t0
        self.games: dict = defaultdict(float)
        self.wins: dict = defaultdict(float)

    def weight(self, t: float) -> float:
        return math.exp((t - self.t0) / self.tau)

    def add(self, key, won: float, w: float) -> None:
        self.games[key] += w
        self.wins[key] += w * won

    def get(self, key, scale: float) -> tuple[float, float]:
        """(games, wins) at the time whose weight is 1/scale."""
        g = self.games.get(key)
        return (g * scale, self.wins[key] * scale) if g else (0.0, 0.0)


class Elo:
    """Elo ratings that drift back towards 1500 while a team or player is inactive."""

    def __init__(self):
        self.rating: dict = {}
        self.last: dict = {}
        self.games: dict = defaultdict(int)

    def get(self, key, t: float) -> float:
        if key not in self.rating:
            return ELO_START
        decay = 0.5 ** ((t - self.last[key]) / (ELO_HALF_LIFE_DAYS * DAY))
        return ELO_START + (self.rating[key] - ELO_START) * decay

    def k(self, key) -> float:
        return ELO_K * (1 + 2 * math.exp(-self.games[key] / 15))  # move fast while unknown

    def update(self, key, t: float, rating_now: float, error: float) -> None:
        self.rating[key] = rating_now + self.k(key) * error
        self.last[key] = t
        self.games[key] += 1


def expected(diff: float) -> float:
    return 1 / (1 + 10 ** (-diff / 400))


class ProState:
    def __init__(self, t0: float):
        self.t0 = t0
        self.t = t0
        self.team_elo = Elo()
        self.player_elo = Elo()
        self.form = DecayedCounts(45, t0)
        self.h2h = DecayedCounts(180, t0)
        self.hero = DecayedCounts(60, t0)
        self.player_hero = DecayedCounts(180, t0)
        self.team_hero = DecayedCounts(120, t0)
        self.vs = DecayedCounts(120, t0)  # only used for its weight(); matrices below
        self.vs_games = np.zeros((N_HEROES, N_HEROES))
        self.vs_wins = np.zeros((N_HEROES, N_HEROES))
        self.team_roster: dict = {}  # team_id -> (time, 5 account ids) of its latest match
        self.team_results: dict = defaultdict(list)  # team_id -> [(time, opponent, won)]

    # ---- features ------------------------------------------------------------------------

    def features(self, rt, dt, rp, dp, rh=None, dh=None, t: float | None = None) -> dict:
        """Features for Radiant team `rt` (players rp, heroes rh) vs Dire `dt` (dp, dh).

        Team ids may be None (unknown team). Heroes may be None for a pre-draft prediction.
        """
        t = self.t if t is None else t
        f = {}
        te, pe = self.team_elo, self.player_elo
        f["team_elo_diff"] = te.get(rt, t) - te.get(dt, t)
        f["player_elo_diff"] = np.mean([pe.get(p, t) for p in rp]) - np.mean([pe.get(p, t) for p in dp])
        f["team_exp_diff"] = math.log1p(te.games[rt]) - math.log1p(te.games[dt])
        f["player_exp_diff"] = np.mean([math.log1p(pe.games[p]) for p in rp]) - np.mean([math.log1p(pe.games[p]) for p in dp])
        f["min_team_games"] = math.log1p(min(te.games[rt], te.games[dt]))

        s = 1 / self.form.weight(t)
        f["form_diff"] = rate_logit(*self.form.get(rt, s), 5) - rate_logit(*self.form.get(dt, s), 5)

        g, w = self.h2h.get(_pair(rt, dt), 1 / self.h2h.weight(t))
        if rt is not None and dt is not None and rt > dt:
            w = g - w  # stored from the lower team id's side
        f["h2h"] = rate_logit(g, w, 3) if rt is not None and dt is not None else 0.0
        f["h2h_games"] = math.log1p(g)

        if rh is not None and dh is not None:
            f.update(self._draft_features(rt, dt, rp, dp, rh, dh, t))
        return f

    def _draft_features(self, rt, dt, rp, dp, rh, dh, t) -> dict:
        f = {}
        s_hero = 1 / self.hero.weight(t)
        meta = {h: rate_logit(*self.hero.get(h, s_hero), 20) for h in (*rh, *dh)}
        f["hero_meta_diff"] = sum(meta[h] for h in rh) - sum(meta[h] for h in dh)

        def comfort(players, heroes):
            # Who plays which hero isn't known before the game, so use each hero's most
            # experienced player on the team.
            s = 1 / self.player_hero.weight(t)
            games = wr = 0.0
            for h in heroes:
                g, w = max((self.player_hero.get((p, h), s) for p in players), key=lambda gw: gw[0])
                games += math.log1p(g)
                wr += rate_logit(g, w, 5)
            return games, wr

        (rg, rw), (dg, dw) = comfort(rp, rh), comfort(dp, dh)
        f["player_hero_games_diff"] = rg - dg
        f["player_hero_wr_diff"] = rw - dw

        s_team = 1 / self.team_hero.weight(t)
        f["team_hero_games_diff"] = sum(math.log1p(self.team_hero.get((rt, h), s_team)[0]) for h in rh) - sum(
            math.log1p(self.team_hero.get((dt, h), s_team)[0]) for h in dh
        )

        s_vs = 1 / self.vs.weight(t)
        r, d = np.array(rh), np.array(dh)
        g = self.vs_games[np.ix_(r, d)] * s_vs
        w = self.vs_wins[np.ix_(r, d)] * s_vs
        base = np.array([meta[h] for h in rh])[:, None] - np.array([meta[h] for h in dh])[None, :]
        p0 = 1 / (1 + np.exp(-base))
        p = (w + 20 * p0) / (g + 20)
        f["counter_sum"] = float((np.log(p / (1 - p)) - base).sum())
        return f

    # ---- update --------------------------------------------------------------------------

    def update(self, rt, dt, rp, dp, rh, dh, radiant_win: bool, t: float) -> None:
        y = float(radiant_win)
        self.t = max(self.t, t)

        rt_elo, dt_elo = self.team_elo.get(rt, t), self.team_elo.get(dt, t)
        err = y - expected(rt_elo - dt_elo)
        if rt is not None:
            self.team_elo.update(rt, t, rt_elo, err)
        if dt is not None:
            self.team_elo.update(dt, t, dt_elo, -err)

        rp_elo = [self.player_elo.get(p, t) for p in rp]
        dp_elo = [self.player_elo.get(p, t) for p in dp]
        perr = y - expected(np.mean(rp_elo) - np.mean(dp_elo))
        for p, e in zip(rp, rp_elo):
            self.player_elo.update(p, t, e, perr)
        for p, e in zip(dp, dp_elo):
            self.player_elo.update(p, t, e, -perr)

        for team, players, heroes, won, opp in ((rt, rp, rh, y, dt), (dt, dp, dh, 1 - y, rt)):
            for p, h in zip(players, heroes):
                self.player_hero.add((p, h), won, self.player_hero.weight(t))
                self.hero.add(h, won, self.hero.weight(t))
            if team is None:
                continue  # unknown teams get no team stats, so they don't pool together
            self.form.add(team, won, self.form.weight(t))
            for h in heroes:
                self.team_hero.add((team, h), won, self.team_hero.weight(t))
            self.team_roster[team] = (t, tuple(players))
            self.team_results[team].append((t, opp, bool(won)))

        if rt is not None and dt is not None:
            self.h2h.add(_pair(rt, dt), y if rt < dt else 1 - y, self.h2h.weight(t))

        w = self.vs.weight(t)
        r, d = np.array(rh), np.array(dh)
        self.vs_games[np.ix_(r, d)] += w
        self.vs_wins[np.ix_(r, d)] += w * y
        self.vs_games[np.ix_(d, r)] += w
        self.vs_wins[np.ix_(d, r)] += w * (1 - y)


def _pair(a, b):
    if a is None or b is None:
        return None
    return (a, b) if a < b else (b, a)
