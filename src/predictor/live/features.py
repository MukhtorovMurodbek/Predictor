"""Game-state features for live win prediction: what the scoreboard says at minute m.

Everything is from Radiant's side: gold lead > 0 means Radiant is ahead, towers_diff > 0
means Radiant has destroyed more towers than it lost, and so on.
"""

import numpy as np

STATE_FEATURES = ["minute", "gold_adv", "gold_adv_5min", "towers_diff", "rax_diff", "roshan_diff", "first_blood"]
# These come from the objectives log and may be unknown live (e.g. the live feed has no
# Roshan info), so the model is trained to cope with them missing.
OPTIONAL_STATE = ["towers_diff", "rax_diff", "roshan_diff", "first_blood"]


def parse_events(events: str) -> list[tuple[int, str, int]]:
    """'805:npc_dota_badguys_tower1_top;1326:rosh2;96:fb0' ->
    [(time, kind, +1 if it's good for Radiant else -1)], kind in tower/rax/rosh/fb."""
    out = []
    for item in filter(None, (events or "").split(";")):
        t, _, what = item.partition(":")
        if not t.lstrip("-").isdigit():
            continue
        t = int(t)
        if what.startswith("npc_dota_"):
            side = 1 if "badguys" in what else -1 if "goodguys" in what else 0  # whose building died
            kind = "tower" if "tower" in what else "rax" if "rax" in what else None
            if kind and side:
                out.append((t, kind, side))
        elif what.startswith("rosh"):
            out.append((t, "rosh", 1 if what == "rosh2" else -1))
        elif what.startswith("fb") and what[2:].isdigit():
            out.append((t, "fb", 1 if int(what[2:]) < 128 else -1))
    return out


def state_at(gold: list, events: list[tuple[int, str, int]], minute: int) -> dict:
    """Scoreboard at the start of `minute` (gold[i] is the lead at minute i)."""
    t = minute * 60
    g = gold[min(minute, len(gold) - 1)]
    g5 = gold[max(0, min(minute, len(gold) - 1) - 5)]
    diff = {"tower": 0, "rax": 0, "rosh": 0}
    fb = 0
    for when, kind, side in events:
        if when > t:
            continue
        if kind == "fb":
            fb = side
        else:
            diff[kind] += side
    return {
        "minute": float(minute),
        "gold_adv": float(g),
        "gold_adv_5min": float(g - g5),
        "towers_diff": float(diff["tower"]),
        "rax_diff": float(diff["rax"]),
        "roshan_diff": float(diff["rosh"]),
        "first_blood": float(fb),
    }


def series_from_timeline(gold, events: str, step: int = 2) -> list[dict]:
    """One state dict every `step` minutes from 0 to the end of the game."""
    gold = list(gold)
    evs = parse_events(events)
    return [state_at(gold, evs, m) for m in range(0, len(gold), step)]


def mask_optional(X: np.ndarray, columns: list[str], rng, frac: float) -> np.ndarray:
    """Hide the optional columns in a random `frac` of rows (training only)."""
    idx = [columns.index(c) for c in OPTIONAL_STATE]
    rows = rng.random(len(X)) < frac
    X[np.ix_(rows, idx)] = np.nan
    return X
