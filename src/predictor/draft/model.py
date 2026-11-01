"""Pro next-pick / next-ban predictor.

A gradient-boosting classifier scores every free hero at the current draft step ("is this
the hero that gets picked/banned now?"). The scores are normalised into shares that add up
to 100%.

Training replays pro history once (oldest -> newest). For recent matches it records the
features of the hero that was really chosen at each of the 24 steps, plus a sample of heroes
that weren't (more often popular ones, which are the hard cases).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import MODELS_DIR
from ..io import atomic_write
from ..data import DRAFTS_PATH, PLAYER_COLS, R_COLS, D_COLS
from ..data.teams import find_team
from ..pro.dataset import load_pro_matches
from .features import FEATURES, N_HEROES, DraftState

MODEL_PATH = MODELS_DIR / "draft.pkl"
DAY = 86400


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    )


def fit_classifier(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    return new_classifier().fit(pd.DataFrame(X, columns=FEATURES), y)


# ---- data --------------------------------------------------------------------------------


@dataclass
class MatchRecord:
    t: float
    teams: tuple  # (radiant team id or None, dire team id or None)
    players: tuple  # (5 radiant account ids, 5 dire account ids)
    picks_by_player: list  # [(account_id, hero)]
    radiant_win: bool
    actions: list  # [(is_pick, side, hero)] in draft order
    cm: bool  # a full Captains Mode draft: 24 actions, 5 picks + 7 bans per team
    order: list  # [(is_pick, "A"/"B")] for each action, "A" = the team that acted first
    match_id: int = 0


def _team(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def load_records() -> list[MatchRecord]:
    """Pro matches joined with their pick/ban order, oldest first."""
    pro = load_pro_matches()
    drafts = pd.read_parquet(DRAFTS_PATH).sort_values(["match_id", "ord"])
    ids, starts = np.unique(drafts["match_id"].to_numpy(), return_index=True)
    bounds = dict(zip(ids, zip(starts, np.r_[starts[1:], len(drafts)])))
    is_pick, side, hero = (drafts[c].to_numpy() for c in ("is_pick", "team", "hero_id"))

    records = []
    heroes = pro[R_COLS + D_COLS].to_numpy()
    players = pro[PLAYER_COLS].to_numpy()
    for i, row in enumerate(pro[["match_id", "start_time", "radiant_team_id", "dire_team_id", "radiant_win"]].itertuples(index=False)):
        if row.match_id not in bounds:
            continue
        a, b = bounds[row.match_id]
        actions = [(bool(is_pick[j]), int(side[j]), int(hero[j])) for j in range(a, b)]
        first = actions[0][1]
        order = [(p, "A" if s == first else "B") for p, s, _ in actions]
        cm = len(actions) == 24 and all(
            sum(1 for p, a in order if p == pick and a == who) == n
            for who in "AB" for pick, n in ((True, 5), (False, 7))
        )
        records.append(
            MatchRecord(
                t=float(row.start_time),
                teams=(_team(row.radiant_team_id), _team(row.dire_team_id)),
                players=(tuple(players[i, :5]), tuple(players[i, 5:])),
                picks_by_player=list(zip(players[i], heroes[i])),
                radiant_win=bool(row.radiant_win),
                actions=actions,
                cm=cm,
                order=order,
                match_id=int(row.match_id),
            )
        )
    return records


def draft_rows(state: DraftState, rec: MatchRecord, valid: np.ndarray, rng, negatives: int | None):
    """Feature rows for one match's 24 steps, computed from `state` (before the match).
    negatives=None keeps every free hero (for evaluation); otherwise samples that many."""
    first = rec.actions[0][1]
    side_of = {"A": first, "B": 1 - first}
    meta = state.meta_context(rec.t)
    ctx = {
        k: state.side_context(rec.teams[s], rec.teams[1 - s], rec.players[s], rec.players[1 - s], rec.t)
        for k, s in side_of.items()
    }
    contest = meta["meta_pick"] + meta["meta_ban"] + 0.01
    used = np.zeros(N_HEROES, dtype=bool)
    picks = {"A": [], "B": []}
    n_bans = {"A": 0, "B": 0}
    X, y, step_ids = [], [], []
    for k, (is_pick, actor) in enumerate(rec.order):
        chosen = rec.actions[k][2]
        free = valid[~used[valid]]
        if negatives is None:
            cand = free
        else:
            others = free[free != chosen]
            pr = contest[others] / contest[others].sum()
            cand = np.r_[chosen, rng.choice(others, size=min(negatives, len(others)), replace=False, p=pr)]
        other = "B" if actor == "A" else "A"
        slot = len(picks[actor]) if is_pick else n_bans[actor]
        X.append(state.step_features(meta, ctx[actor], k, is_pick, slot, cand, picks[actor], picks[other]))
        y.append(cand == chosen)
        step_ids.append(np.full(len(cand), k))
        used[chosen] = True
        if is_pick:
            picks[actor].append(chosen)
        else:
            n_bans[actor] += 1
    return np.vstack(X).astype(np.float32), np.concatenate(y), np.concatenate(step_ids)


def replay(records, valid, train_from: int, test_from: int, test_every: int, negatives: int = 10, seed: int = 0):
    """Replay all records; collect sampled rows for records >= train_from and full rows for
    every `test_every`-th record >= test_from. Returns (state, train, test)."""
    rng = np.random.default_rng(seed)
    state = DraftState(records[0].t)
    train = {"X": [], "y": [], "match": []}
    test = {"X": [], "y": [], "step": [], "match": []}
    for i, rec in enumerate(records):
        if rec.cm and i >= train_from:
            X, y, _ = draft_rows(state, rec, valid, rng, negatives)
            train["X"].append(X), train["y"].append(y), train["match"].append(np.full(len(y), i))
        if rec.cm and i >= test_from and (i - test_from) % test_every == 0:
            X, y, step = draft_rows(state, rec, valid, rng, None)
            test["X"].append(X), test["y"].append(y), test["step"].append(step), test["match"].append(np.full(len(y), i))
        state.update(rec.actions, rec.teams, rec.players, rec.picks_by_player, rec.radiant_win, rec.t)
    stack = lambda d: {k: np.concatenate(v) if v else np.array([]) for k, v in d.items()}  # noqa: E731
    return state, stack(train), stack(test)


# ---- model -------------------------------------------------------------------------------


@dataclass
class DraftModel:
    state: DraftState
    clf: HistGradientBoostingClassifier
    valid_heroes: np.ndarray
    order: list  # draft order of the newest matches: [(is_pick, "A"/"B")] x 24
    team_names: dict[int, tuple[str, str]]
    hero_names: dict[int, str]
    n_matches: int
    report: str = ""
    extra: dict = field(default_factory=dict)

    def team_name(self, team_id: int) -> str:
        return self.team_names.get(team_id, (f"team {team_id}", ""))[0]

    def find_team(self, query: str) -> int:
        cutoff = self.state.t - 365 * DAY
        activity = {t: self.state.team_matches_n[t] for t, (when, _) in self.state.team_roster.items() if when >= cutoff}
        return find_team(query, self.team_names, activity)

    def next_step(self, n_done: int) -> tuple[bool, str]:
        """(is_pick, 'A' or 'B') for the action after n_done actions."""
        return self.order[n_done]

    def predict_next(self, team_a: int, team_b: int, done: list[int], top: int = 8) -> list[tuple[int, float]]:
        """Most likely heroes for the next action. team_a acts first (has first pick);
        done = heroes picked/banned so far, in draft order."""
        if len(done) >= len(self.order):
            raise ValueError("the draft is already complete (24 picks and bans)")
        if len(set(done)) != len(done):
            raise ValueError("a hero appears twice in the draft")
        st, t = self.state, self.state.t
        teams = {"A": team_a, "B": team_b}
        rosters = {k: st.team_roster.get(v, (0, ()))[1] for k, v in teams.items()}
        is_pick, actor = self.order[len(done)]
        other = "B" if actor == "A" else "A"
        picks = {"A": [], "B": []}
        n_bans = {"A": 0, "B": 0}
        for k, hero in enumerate(done):
            if self.order[k][0]:
                picks[self.order[k][1]].append(hero)
            else:
                n_bans[self.order[k][1]] += 1
        meta = st.meta_context(t)
        ctx = st.side_context(teams[actor], teams[other], rosters[actor], rosters[other], t)
        cand = np.array([h for h in self.valid_heroes if h not in set(done)])
        slot = len(picks[actor]) if is_pick else n_bans[actor]
        X = st.step_features(meta, ctx, len(done), is_pick, slot, cand, picks[actor], picks[other])
        scores = self.clf.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]
        share = scores / scores.sum()
        order = np.argsort(-share)[:top]
        return [(int(cand[i]), float(share[i])) for i in order]

    def save(self, path: Path = MODEL_PATH) -> None:
        with atomic_write(path) as tmp, open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "DraftModel":
        # Only load model files you made yourself: pickle can run code from the file.
        with open(path, "rb") as fh:
            return pickle.load(fh)


def topk_report(clf, test: dict, n_test_matches: int) -> tuple[str, dict]:
    """Top-1/5/10 accuracy for picks and bans vs the 'most popular in the meta' baseline."""
    X = pd.DataFrame(test["X"], columns=FEATURES)
    scores = clf.predict_proba(X)[:, 1]
    groups = test["match"] * 100 + test["step"]
    df = pd.DataFrame({"g": groups, "y": test["y"], "score": scores, "is_pick": X["is_pick"].to_numpy(),
                       "meta": np.where(X["is_pick"] == 1, X["meta_pick"], X["meta_ban"])})
    out, lines = {}, [f"tested on {n_test_matches:,} newer pro drafts (every free hero ranked at all 24 steps)", ""]
    for col, name in (("score", "model"), ("meta", "most popular in the meta")):
        df["rank"] = df.groupby("g")[col].rank(ascending=False, method="first")
        true_rank = df[df["y"]].set_index("g")
        for kind, mask in (("picks", true_rank["is_pick"] == 1), ("bans", true_rank["is_pick"] == 0)):
            r = true_rank.loc[mask, "rank"]
            acc = {k: (r <= k).mean() for k in (1, 5, 10)}
            out[f"{name}_{kind}"] = acc
            lines.append(f"{name:<25} {kind:<5} top-1 {acc[1]:.1%}  top-5 {acc[5]:.1%}  top-10 {acc[10]:.1%}")
    return "\n".join(lines), out
