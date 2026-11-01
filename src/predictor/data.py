"""Loading matches and hero names.

Matches are stored "wide": one row per match with hero IDs in r0..r4 (Radiant) and d0..d4
(Dire), plus radiant_win, start_time, patch and a few source-specific columns.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import DATA_DIR, RAW_DIR

R_COLS = [f"r{i}" for i in range(5)]
D_COLS = [f"d{i}" for i in range(5)]
HEROES_PATH = DATA_DIR / "heroes.json"
PATCHES_PATH = DATA_DIR / "patches.json"
SOURCES = {"pro": RAW_DIR / "pro_matches.parquet", "public": RAW_DIR / "public_matches.parquet"}


def load_heroes() -> dict[int, str]:
    if not HEROES_PATH.exists():
        raise FileNotFoundError("data/heroes.json missing, run: python -m predictor.collect heroes")
    return {h["id"]: h["localized_name"] for h in json.loads(HEROES_PATH.read_text("utf-8"))}


def patch_for_times(start_times: pd.Series) -> pd.Series:
    """Map unix start times to patch names using data/patches.json."""
    patches = json.loads(PATCHES_PATH.read_text("utf-8"))
    starts = [datetime.fromisoformat(p["date"].replace("Z", "+00:00")).timestamp() for p in patches]
    idx = np.searchsorted(starts, start_times.to_numpy(), side="right") - 1
    names = np.array([p["name"] for p in patches], dtype=object)
    return pd.Series(np.where(idx >= 0, names[idx.clip(0)], None), index=start_times.index)


def load_matches(source: str = "public", patch: str | None = None) -> pd.DataFrame:
    """Load matches for 'pro', 'public' or 'all', sorted oldest first."""
    names = list(SOURCES) if source == "all" else [source]
    frames = []
    for name in names:
        path = SOURCES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, run: python -m predictor.collect {name}")
        frames.append(pd.read_parquet(path).assign(source=name))
    df = pd.concat(frames, ignore_index=True)
    if patch:
        df = df[df["patch"] == patch]
    return df.sort_values("start_time", ignore_index=True)


def hero_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radiant heroes (n,5), dire heroes (n,5), radiant_win (n,))."""
    return (
        df[R_COLS].to_numpy(np.int64),
        df[D_COLS].to_numpy(np.int64),
        df["radiant_win"].to_numpy(np.float64),
    )


def resolve_heroes(names: str | list[str] | None, heroes: dict[int, str]) -> list[int]:
    """Turn 'Phantom Assassin, pudge, cm' style input into hero IDs."""
    if not names:
        return []
    if isinstance(names, str):
        names = [n for n in names.split(",")]
    by_name = {v.lower(): k for k, v in heroes.items()}
    # "pa" -> Phantom Assassin, "cm" -> Crystal Maiden; only initials shared by one hero count
    initials: dict[str, list[int]] = {}
    for k, v in heroes.items():
        initials.setdefault("".join(w[0] for w in v.lower().replace("-", " ").split()), []).append(k)
    ids = []
    for raw in names:
        q = raw.strip().lower()
        if not q:
            continue
        if q in by_name:
            ids.append(by_name[q])
            continue
        if len(initials.get(q, [])) == 1:
            ids.append(initials[q][0])
            continue
        matches = [k for n, k in by_name.items() if n.startswith(q)] or [
            k for n, k in by_name.items() if q in n
        ]
        if len(matches) == 1:
            ids.append(matches[0])
        elif not matches:
            raise ValueError(f"unknown hero: {raw.strip()!r}")
        else:
            options = ", ".join(sorted(heroes[m] for m in matches))
            raise ValueError(f"{raw.strip()!r} is ambiguous: {options}")
    return ids


def to_unix(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())
