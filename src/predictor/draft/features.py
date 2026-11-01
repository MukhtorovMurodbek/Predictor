"""Features for "which hero gets picked or banned next" in a pro Captains Mode draft.

Like pro/features.py, DraftState walks through matches oldest -> newest. Features for a
match only use earlier matches. Everything is a numpy array over hero ids, so scoring all
~120 free heroes at once is fast.

The draft order is fixed in Captains Mode. "A" is the team that acts first (it also picks
first), "B" the other team.
"""

import math
from collections import defaultdict

import numpy as np

N_HEROES = 160
DAY = 86400.0
META_HALF_LIFE = 45  # days: the pro meta moves fast
TEAM_HALF_LIFE = 150  # days: team and player habits move slower

# 24 actions: b = ban, P = pick; A/B = which team acts. Valve changes this order between
# patches (three versions since 2023), so each draft uses its own recorded order and the
# model keeps the order of the newest matches for live predictions. This is the late-2026 one.
CM_ORDER = "bA bA bB bB bA bB bB PA PB bA bA bB PB PA PA PB PB PA bA bB bA bB PA PB".split()


def parse_order(codes: list[str]) -> list[tuple[bool, str]]:
    """["bA", "PB", ...] -> [(is_pick, "A"/"B"), ...]"""
    return [(code[0] == "P", code[1]) for code in codes]

FEATURES = [
    "step",
    "is_pick",
    "meta_pick",
    "meta_ban",
    "meta_slot",
    "team_pick",
    "team_ban",
    "opp_pick",
    "opp_ban",
    "banned_vs_team",
    "banned_vs_opp",
    "team_comfort",
    "opp_comfort",
    "copick_allies",
    "copick_enemies",
    "adv_vs_enemies",
    "adv_vs_allies",
]


def _zeros() -> np.ndarray:
    return np.zeros(N_HEROES)  # a named function, so the state can be pickled


class DraftState:
    def __init__(self, t0: float):
        self.t0 = t0
        self.t = t0
        self.meta_tau = META_HALF_LIFE * DAY / math.log(2)
        self.team_tau = TEAM_HALF_LIFE * DAY / math.log(2)
        H = N_HEROES
        # Counts are stored multiplied by exp(t / tau) so older matches fade (see pro/features.py).
        self.meta_matches = 0.0
        self.meta_pick = np.zeros(H)
        self.meta_ban = np.zeros(H)
        # how often a hero is a team's n-th pick / n-th ban (first-pick heroes vs last picks)
        self.slot_pick = np.zeros((5, H))
        self.slot_ban = np.zeros((7, H))
        self.team_matches: dict = defaultdict(float)
        self.team_pick: dict = defaultdict(_zeros)
        self.team_ban: dict = defaultdict(_zeros)
        self.banned_vs: dict = defaultdict(_zeros)  # bans made against a team
        self.player_pick: dict = defaultdict(_zeros)
        self.copick = np.zeros((H, H))  # picked on the same team
        self.pick_long = np.zeros(H)  # pick counts with the slower decay, for copick rates
        self.vs_games = np.zeros((H, H))
        self.vs_wins = np.zeros((H, H))
        self.team_roster: dict = {}  # team_id -> (time, account ids)
        self.team_matches_n: dict = defaultdict(int)

    def _w(self, t: float, tau: float) -> float:
        return math.exp((t - self.t0) / tau)

    # ---- features ------------------------------------------------------------------------

    def side_context(self, team, opp, players, opp_players, t: float) -> dict:
        """Arrays that stay the same for the whole draft, from the acting team's view."""
        s = 1 / self._w(t, self.team_tau)

        def team_rates(tid):
            if tid is None or tid not in self.team_pick:
                z = np.zeros(N_HEROES)
                return z, z, z
            n = self.team_matches[tid] + 2 * self._w(t, self.team_tau)  # 2 virtual matches of prior
            return self.team_pick[tid] / n, self.team_ban[tid] / n, self.banned_vs[tid] / n

        def comfort(ps):
            arrays = [self.player_pick[p] for p in ps if p in self.player_pick]
            return np.log1p(np.max(arrays, axis=0) * s) if arrays else np.zeros(N_HEROES)

        tp, tb, bvt = team_rates(team)
        op, ob, bvo = team_rates(opp)
        return {
            "team_pick": tp, "team_ban": tb, "banned_vs_team": bvt,
            "opp_pick": op, "opp_ban": ob, "banned_vs_opp": bvo,
            "team_comfort": comfort(players), "opp_comfort": comfort(opp_players),
        }

    def meta_context(self, t: float) -> dict:
        n = self.meta_matches + 5 * self._w(t, self.meta_tau)
        s = 1 / self._w(t, self.team_tau)
        return {"meta_pick": self.meta_pick / n, "meta_ban": self.meta_ban / n,
                "slot_pick": self.slot_pick / n, "slot_ban": self.slot_ban / n, "scale": s}

    def step_features(self, meta: dict, ctx: dict, step: int, is_pick: bool, slot: int, cand: np.ndarray, allies: list, enemies: list) -> np.ndarray:
        """(len(cand), len(FEATURES)) for candidate heroes at draft step `step`, where this is
        the acting team's `slot`-th pick or ban (0-based)."""
        n = len(cand)
        slots = meta["slot_pick"] if is_pick else meta["slot_ban"]
        cols = {
            "step": np.full(n, step, dtype=float),
            "is_pick": np.full(n, float(is_pick)),
            "meta_pick": meta["meta_pick"][cand],
            "meta_ban": meta["meta_ban"][cand],
            "meta_slot": slots[min(slot, len(slots) - 1)][cand],
        }
        for key in ("team_pick", "team_ban", "opp_pick", "opp_ban", "banned_vs_team", "banned_vs_opp", "team_comfort", "opp_comfort"):
            cols[key] = ctx[key][cand]
        cols["copick_allies"] = self._copick(cand, allies)
        cols["copick_enemies"] = self._copick(cand, enemies)
        cols["adv_vs_enemies"] = self._adv(cand, enemies, meta["scale"])
        cols["adv_vs_allies"] = self._adv(cand, allies, meta["scale"])
        return np.column_stack([cols[f] for f in FEATURES])

    def _copick(self, cand, heroes) -> np.ndarray:
        """How often each candidate was picked together with these heroes (as a share of their picks)."""
        if not heroes:
            return np.zeros(len(cand))
        h = np.array(heroes)
        return (self.copick[np.ix_(cand, h)] / (self.pick_long[h] + 1.0)).sum(axis=1)

    def _adv(self, cand, heroes, scale) -> np.ndarray:
        """Candidate's smoothed log-odds win rate against these heroes, summed."""
        if not heroes:
            return np.zeros(len(cand))
        h = np.array(heroes)
        g = self.vs_games[np.ix_(cand, h)] * scale
        w = self.vs_wins[np.ix_(cand, h)] * scale
        p = (w + 5) / (g + 10)
        return np.log(p / (1 - p)).sum(axis=1)

    # ---- update --------------------------------------------------------------------------

    def update(self, actions, teams, players, picks_by_player, radiant_win, t: float) -> None:
        """actions: [(is_pick, side 0=radiant/1=dire, hero)], teams/players indexed by side,
        picks_by_player: [(account_id, hero)], radiant_win: bool or None."""
        self.t = max(self.t, t)
        wm, wt = self._w(t, self.meta_tau), self._w(t, self.team_tau)
        self.meta_matches += wm
        picks = {0: [], 1: []}
        bans = {0: 0, 1: 0}
        for is_pick, side, hero in actions:
            if is_pick:
                self.meta_pick[hero] += wm
                if len(picks[side]) < 5:
                    self.slot_pick[len(picks[side]), hero] += wm
                picks[side].append(hero)
            else:
                self.meta_ban[hero] += wm
                if bans[side] < 7:
                    self.slot_ban[bans[side], hero] += wm
                bans[side] += 1
                if teams[side] is not None:
                    self.team_ban[teams[side]][hero] += wt
                if teams[1 - side] is not None:
                    self.banned_vs[teams[1 - side]][hero] += wt
        for side in (0, 1):
            tid = teams[side]
            if tid is not None:
                self.team_matches[tid] += wt
                self.team_matches_n[tid] += 1
                for hero in picks[side]:
                    self.team_pick[tid][hero] += wt
                self.team_roster[tid] = (t, tuple(players[side]))
            hs = np.array(picks[side], dtype=int)
            if len(hs):
                self.copick[np.ix_(hs, hs)] += wt
                self.copick[hs, hs] -= wt  # not with itself
                self.pick_long[hs] += wt
        for p, hero in picks_by_player:
            self.player_pick[p][hero] += wt
        if radiant_win is not None and len(picks[0]) == 5 and len(picks[1]) == 5:
            r, d, y = np.array(picks[0]), np.array(picks[1]), float(radiant_win)
            self.vs_games[np.ix_(r, d)] += wt
            self.vs_wins[np.ix_(r, d)] += wt * y
            self.vs_games[np.ix_(d, r)] += wt
            self.vs_wins[np.ix_(d, r)] += wt * (1 - y)
