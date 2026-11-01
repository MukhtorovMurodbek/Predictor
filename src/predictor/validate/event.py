"""Replay a real pro event as if the bot had been running during it.

    python -m predictor.validate.event "PARI UNIVERSE"
    python -m predictor.validate.event "BLAST SLAM VIII" --skip-live

Every model is rebuilt from matches played *before* the event started (nothing from the event
leaks in), then asked about each game of the event:
  - pro predictor: winner of each map, before and after the draft; favourite of each series
  - draft predictor: every pick and ban (was the real hero in the top 1/5/10?)
  - live predictor: win % at minutes 10, 20, 30 of each map
  - tournament predictor: title chances of the participants vs the real champion
Takes ~5-8 minutes (it re-trains everything on the older data).
"""

import argparse
import math
import time

import numpy as np
import pandas as pd

from ..data import HEROES_PATH, LEAGUES_PATH, PLAYERS_PATH, TEAMS_PATH, TIMELINES_PATH
from ..draft.model import DraftModel, fit_classifier, load_records, replay as draft_replay
from ..live.features import parse_events, state_at
from ..live.model import LiveModel, build_rows, hide_inputs, new_classifier as live_classifier
from ..pro.dataset import load_pro_matches, replay as pro_replay, training_mask
from ..pro.model import ProModel, fit_classifiers, series_win
from ..tournament.sim import map_matrix, title_odds

DRAFT_TRAIN_MATCHES = 20_000


def find_league(query: str) -> tuple[int, str]:
    leagues = pd.read_parquet(LEAGUES_PATH)
    pro = load_pro_matches()
    counts = pro["leagueid"].value_counts()
    hits = leagues[leagues["name"].str.contains(query, case=False, na=False, regex=False)]
    hits = hits.assign(n=hits["leagueid"].map(counts).fillna(0)).sort_values("n", ascending=False)
    hits = hits[hits["n"] > 0]
    if hits.empty:
        raise ValueError(f"no event with pro matches matches {query!r}")
    return int(hits.iloc[0]["leagueid"]), hits.iloc[0]["name"]


def _team(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)


def _series(event: pd.DataFrame) -> list[pd.DataFrame]:
    """Group maps into series: same two teams, less than 6 hours apart."""
    out, current, key, last = [], [], None, None
    for row in event.itertuples():
        k = frozenset((row.radiant_team_id, row.dire_team_id))
        if current and (k != key or row.start_time - last > 6 * 3600):
            out.append(pd.DataFrame(current))
            current = []
        current.append(row._asdict())
        key, last = k, row.start_time
    if current:
        out.append(pd.DataFrame(current))
    return out


def replay_event(query: str, live: bool = True, log=print) -> str:
    t0 = time.time()
    leagueid, name = find_league(query)
    pro = load_pro_matches()
    event = pro[(pro["leagueid"] == leagueid) & pro["radiant_team_id"].notna() & pro["dire_team_id"].notna()]
    event = event.sort_values("start_time").reset_index(drop=True)
    start = int(event["start_time"].min())
    before = pro[pro["start_time"] < start].reset_index(drop=True)
    names = {int(r.team_id): (r.name or "", r.tag or "") for r in pd.read_parquet(TEAMS_PATH).itertuples()}
    players = pd.read_parquet(PLAYERS_PATH)
    heroes = {h["id"]: h["localized_name"] for h in pd.read_json(HEROES_PATH).to_dict("records")}
    lines = [f"Event replay: {name} ({len(event)} maps, {pd.to_datetime(start, unit='s').date()} onwards)",
             "All models rebuilt from matches before the event started.", ""]

    # ---- pro predictor ---------------------------------------------------------------------
    X, state = pro_replay(before)
    use = training_mask(before)
    pregame, draft = fit_classifiers(X[use], before.loc[use, "radiant_win"].to_numpy().astype(int))
    pm = ProModel(state, pregame, draft, names, dict(zip(players["account_id"], players["name"])), heroes, len(before))
    log(f"pro model rebuilt ({time.time() - t0:.0f} s)")

    y = event["radiant_win"].to_numpy().astype(int)
    p_pre, p_draft, p_elo = [], [], []
    for row in event.itertuples():
        rt, dt = _team(row.radiant_team_id), _team(row.dire_team_id)
        rh = [getattr(row, f"r{i}") for i in range(5)]
        dh = [getattr(row, f"d{i}") for i in range(5)]
        rr = pm.roster(rt) if rt in state.team_roster else tuple(getattr(row, f"rp{i}") for i in range(5))
        dr = pm.roster(dt) if dt in state.team_roster else tuple(getattr(row, f"dp{i}") for i in range(5))
        f = state.features(rt, dt, rr, dr, rh, dh)
        frame = pd.DataFrame([f])
        p_pre.append(float(pm.pregame.predict_proba(frame[pm.pregame.feature_names_in_])[0, 1]))
        p_draft.append(float(pm.draft.predict_proba(frame[pm.draft.feature_names_in_])[0, 1]))
        p_elo.append(1 / (1 + 10 ** (-f["team_elo_diff"] / 400)))
    p_pre, p_draft, p_elo = map(np.array, (p_pre, p_draft, p_elo))

    def score(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return f"{((p > 0.5) == y).mean():.1%} right, log loss {-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)):.3f}"

    lines += ["MAPS (who wins each map)",
              f"  always Radiant:        {(y == 1).mean():.1%} right",
              f"  higher team rating:    {score(p_elo)}",
              f"  bot before the draft:  {score(p_pre)}",
              f"  bot after the draft:   {score(p_draft)}"]
    sure = np.maximum(p_draft, 1 - p_draft) >= 0.7
    if sure.any():
        lines.append(f"  when the bot said 70%+: right {((p_draft[sure] > 0.5) == y[sure]).mean():.0%} of {sure.sum()} maps")

    # series: did the bot's favourite (before the first map) win the series?
    series = _series(event)
    fav_right = 0
    for s in series:
        a = _team(s.iloc[0]["radiant_team_id"])
        b = _team(s.iloc[0]["dire_team_id"])
        p = pm.predict(a, b) if a in state.team_roster and b in state.team_roster else 0.5
        a_maps = sum((r.radiant_win and _team(r.radiant_team_id) == a) or (not r.radiant_win and _team(r.dire_team_id) == a) for r in s.itertuples())
        b_maps = len(s) - a_maps
        if a_maps != b_maps:
            fav_right += (p >= 0.5) == (a_maps > b_maps)
    lines += ["", f"SERIES: the bot's pre-series favourite won {fav_right} of {len(series)} series"]

    # ---- tournament predictor ----------------------------------------------------------------
    teams = sorted({t for t in pd.concat([event["radiant_team_id"], event["dire_team_id"]]).astype(int) if t in state.team_roster})
    final = event.iloc[-1]
    champion = _team(final["radiant_team_id"] if final["radiant_win"] else final["dire_team_id"])
    if len(teams) >= 8 and champion in teams:
        odds = title_odds(map_matrix(pm, teams), 5000)
        order = list(np.argsort(-odds))
        i = teams.index(champion)
        lines += ["", f"TITLE: champion {pm.team_name(champion)} had {odds[i]:.0%} before the event "
                      f"(#{order.index(i) + 1} of {len(teams)}; favourite {pm.team_name(teams[order[0]])} {odds[order[0]]:.0%}; "
                      f"random guess {1 / len(teams):.0%}; TI-style format, the real format may differ)"]

    # ---- draft predictor -------------------------------------------------------------------
    records = load_records()
    rec_before = [r for r in records if r.t < start]
    cm_idx = [i for i, r in enumerate(rec_before) if r.cm]
    valid = np.array(sorted(heroes))
    dstate, rows, _ = draft_replay(rec_before, valid, train_from=cm_idx[-DRAFT_TRAIN_MATCHES], test_from=len(rec_before) + 1, test_every=1)
    dm = DraftModel(dstate, fit_classifier(rows["X"], rows["y"]), valid, rec_before[cm_idx[-1]].order, names, heroes, len(cm_idx))
    log(f"draft model rebuilt ({time.time() - t0:.0f} s)")
    ranks = {"picks": [], "bans": []}
    by_id = {r.match_id: r for r in records if r.t >= start}
    for row in event.itertuples():
        rec = by_id.get(row.match_id)
        if rec is None or not rec.cm:
            continue
        first = rec.actions[0][1]
        team_a, team_b = rec.teams[first], rec.teams[1 - first]
        if team_a not in dstate.team_roster or team_b not in dstate.team_roster:
            continue
        dm.order = rec.order  # the event's own draft order
        done = []
        for is_pick, _, hero in rec.actions:
            ranked = [h for h, _ in dm.predict_next(team_a, team_b, done, top=200)]
            ranks["picks" if is_pick else "bans"].append(ranked.index(hero) + 1 if hero in ranked else 999)
            done.append(hero)
    lines.append("")
    lines.append("DRAFTS (was the real hero in the bot's top N?)")
    for kind, r in ranks.items():
        r = np.array(r)
        if len(r):
            lines.append(f"  {kind}: top-1 {np.mean(r <= 1):.0%}, top-5 {np.mean(r <= 5):.0%}, top-10 {np.mean(r <= 10):.0%} ({len(r)} {kind})")

    # ---- live predictor --------------------------------------------------------------------
    if live:
        timelines = pd.read_parquet(TIMELINES_PATH)
        Xl, yl, idx = build_rows(before, X, timelines[timelines["start_time"] < start])
        keep = use[idx]
        lm = LiveModel(live_classifier().fit(hide_inputs(Xl[keep].reset_index(drop=True)), yl[keep]), len(before))
        log(f"live model rebuilt ({time.time() - t0:.0f} s)")
        tl = timelines.set_index("match_id")
        hits = {10: [], 20: [], 30: []}
        for row in event.itertuples():
            if row.match_id not in tl.index:
                continue
            t = tl.loc[row.match_id]
            evs = parse_events(t["events"])
            rt, dt = _team(row.radiant_team_id), _team(row.dire_team_id)
            if rt not in state.team_roster or dt not in state.team_roster:
                continue
            rh = [getattr(row, f"r{i}") for i in range(5)]
            dh = [getattr(row, f"d{i}") for i in range(5)]
            pre = state.features(rt, dt, pm.roster(rt), pm.roster(dt), rh, dh)
            for m in hits:
                if len(t["gold"]) > m:
                    p = lm.predict_radiant(pre, state_at(list(t["gold"]), evs, m))
                    hits[m].append((p > 0.5) == bool(row.radiant_win))
        lines += ["", "LIVE (win % during the game)"]
        for m, h in hits.items():
            if h:
                lines.append(f"  at {m} min: right {np.mean(h):.0%} of {len(h)} maps")
    lines += ["", f"(took {time.time() - t0:.0f} s)"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay a real pro event with models frozen before it")
    p.add_argument("league", help="part of the league name, e.g. 'PARI UNIVERSE'")
    p.add_argument("--skip-live", action="store_true")
    args = p.parse_args(argv)
    try:
        print(replay_event(args.league, live=not args.skip_live))
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
