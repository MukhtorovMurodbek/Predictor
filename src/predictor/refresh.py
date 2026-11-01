"""Nightly refresh of the pro models, with a prediction journal.

    python -m predictor.refresh

1. Load the pro, draft and live models as they are now.
2. Download new pro matches (and drafts, timelines).
3. Journal: ask the *current* models about every new match before they learn from it, and
   write prediction + real result to data/journal.csv. This is a true real-world test: the
   models never saw these games.
4. Retrain the pro, draft and live models on everything. The bot reloads them by itself.

The bot runs this every night (see pro/bot.py background()). /journal shows the results.
"""

import argparse
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import DATA_DIR
from .data import TIMELINES_PATH
from .data.collect import collect_pro
from .data.opendota import OpenDota
from .draft import DraftModel
from .draft import MODEL_PATH as DRAFT_PATH
from .draft.model import load_records
from .draft.train import train as train_draft
from .live import LiveModel
from .live import MODEL_PATH as LIVE_PATH
from .live.features import parse_events, state_at
from .live.train import train as train_live
from .pro import MODEL_PATH as PRO_PATH
from .pro import ProModel
from .pro.dataset import load_pro_matches
from .pro.train import train as train_pro

JOURNAL_PATH = DATA_DIR / "journal.csv"
_lock = threading.Lock()


def _team(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def journal_rows(new: pd.DataFrame, pro: ProModel, draft: DraftModel | None, live: LiveModel | None) -> list[dict]:
    """What the given (old) models say about each new match, next to the real result."""
    known = pro.state.team_roster
    records = {r.match_id: r for r in load_records() if r.match_id in set(new["match_id"])} if draft else {}
    timelines = pd.read_parquet(TIMELINES_PATH).set_index("match_id") if live and TIMELINES_PATH.exists() else None
    rows = []
    for m in new.itertuples():
        rt, dt = _team(m.radiant_team_id), _team(m.dire_team_id)
        if rt not in known or dt not in known:
            continue  # teams the model has never seen: the bot couldn't answer either
        rh = [getattr(m, f"r{i}") for i in range(5)]
        dh = [getattr(m, f"d{i}") for i in range(5)]
        f = pd.DataFrame([pro.state.features(rt, dt, pro.roster(rt), pro.roster(dt), rh, dh)])
        row = {
            "date": datetime.fromtimestamp(m.start_time, tz=timezone.utc).strftime("%Y-%m-%d"),
            "match_id": m.match_id,
            "leagueid": getattr(m, "leagueid", None),
            "radiant": pro.team_name(rt),
            "dire": pro.team_name(dt),
            "radiant_win": bool(m.radiant_win),
            "p_pregame": float(pro.pregame.predict_proba(f[pro.pregame.feature_names_in_])[0, 1]),
            "p_draft": float(pro.draft.predict_proba(f[pro.draft.feature_names_in_])[0, 1]),
            "draft_top5": np.nan,
            "p_live_20": np.nan,
        }
        rec = records.get(m.match_id)
        if draft and rec is not None and rec.cm:
            first = rec.actions[0][1]
            a, b = rec.teams[first], rec.teams[1 - first]
            if a in draft.state.team_roster and b in draft.state.team_roster:
                draft.order, done, hits = rec.order, [], []
                for _, _, hero in rec.actions:
                    top = [h for h, _ in draft.predict_next(a, b, done, top=5)]
                    hits.append(hero in top)
                    done.append(hero)
                row["draft_top5"] = float(np.mean(hits))
        if timelines is not None and m.match_id in timelines.index:
            t = timelines.loc[m.match_id]
            if len(t["gold"]) > 20:
                state = state_at(list(t["gold"]), parse_events(t["events"]), 20)
                row["p_live_20"] = live.predict_radiant(f.iloc[0].to_dict(), state)
        rows.append(row)
    return rows


def refresh(log=print) -> str:
    if not _lock.acquire(blocking=False):
        return "a refresh is already running"
    try:
        t0 = time.time()
        pro = ProModel.load(PRO_PATH) if PRO_PATH.exists() else None
        draft = DraftModel.load(DRAFT_PATH) if DRAFT_PATH.exists() else None
        live = LiveModel.load(LIVE_PATH) if LIVE_PATH.exists() else None

        collect_pro(OpenDota())
        lines = []
        if pro is not None:
            matches = load_pro_matches()
            new = matches[matches["start_time"] > pro.state.t]
            rows = journal_rows(new, pro, draft, live)
            if rows:
                old = pd.read_csv(JOURNAL_PATH) if JOURNAL_PATH.exists() else pd.DataFrame()
                journal = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("match_id", keep="first")
                journal.to_csv(JOURNAL_PATH, index=False)
            right = sum((r["p_draft"] > 0.5) == r["radiant_win"] for r in rows)
            lines.append(f"journal: {len(new)} new pro maps, {len(rows)} scored, after-draft call right on {right}")
        quiet = lambda *a: None  # noqa: E731
        train_pro()
        train_draft(log=quiet)
        train_live(log=quiet)
        lines.append("retrained the pro, draft and live models")
        lines.append(f"refresh took {time.time() - t0:.0f} s")
        log("\n".join(lines))
        return "\n".join(lines)
    finally:
        _lock.release()


def journal_summary(days: tuple[int, ...] = (7, 30, 3650)) -> str:
    if not JOURNAL_PATH.exists():
        return "The journal is empty: it fills up after the first nightly refresh with new pro matches."
    j = pd.read_csv(JOURNAL_PATH)
    j["date"] = pd.to_datetime(j["date"])
    out = []
    for d in days:
        part = j[j["date"] >= j["date"].max() - pd.Timedelta(days=d)]
        if part.empty:
            continue
        y = part["radiant_win"].astype(bool)
        label = "all time" if d >= 3650 else f"last {d} days"
        line = (f"{label}: {len(part)} maps · before draft {((part.p_pregame > 0.5) == y).mean():.0%} · "
                f"after draft {((part.p_draft > 0.5) == y).mean():.0%}")
        if part["draft_top5"].notna().any():
            line += f" · draft top-5 {part['draft_top5'].mean():.0%}"
        live = part[part["p_live_20"].notna()]
        if len(live):
            line += f" · live at 20 min {((live.p_live_20 > 0.5) == live.radiant_win.astype(bool)).mean():.0%}"
        out.append(line)
    return "\n".join(out)


def main(argv=None):
    argparse.ArgumentParser(description="Download new pro matches, journal predictions, retrain").parse_args(argv)
    refresh()
    print(journal_summary())


if __name__ == "__main__":
    main()
