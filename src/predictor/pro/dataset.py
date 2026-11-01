"""Turn the pro match file into a feature table by replaying history in time order."""

import math

import numpy as np
import pandas as pd

from ..data import D_COLS, PLAYER_COLS, R_COLS, SOURCES
from .features import DRAFT_FEATURES, ProState

R_PLAYERS, D_PLAYERS = PLAYER_COLS[:5], PLAYER_COLS[5:]


def load_pro_matches() -> pd.DataFrame:
    df = pd.read_parquet(SOURCES["pro"])
    if not set(PLAYER_COLS) <= set(df.columns):
        raise FileNotFoundError("pro data has no player columns, run: python -m predictor.data.collect pro --since 2023-01-01")
    df = df[df["radiant_win"].notna()].sort_values(["start_time", "match_id"], ignore_index=True)
    df["radiant_win"] = df["radiant_win"].astype(bool)
    return df


def _team(x) -> int | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)


def replay(df: pd.DataFrame) -> tuple[pd.DataFrame, ProState]:
    """Features for every match (from what was known before it) + the final state."""
    state = ProState(t0=float(df["start_time"].iloc[0]))
    heroes = df[R_COLS + D_COLS].to_numpy()
    players = df[R_PLAYERS + D_PLAYERS].to_numpy()
    teams = df[["radiant_team_id", "dire_team_id"]].to_numpy(dtype=object)
    times = df["start_time"].to_numpy(dtype=float)
    wins = df["radiant_win"].to_numpy()

    rows = []
    for i in range(len(df)):
        rt, dt = _team(teams[i, 0]), _team(teams[i, 1])
        rp, dp = tuple(players[i, :5]), tuple(players[i, 5:])
        rh, dh = tuple(heroes[i, :5]), tuple(heroes[i, 5:])
        rows.append(state.features(rt, dt, rp, dp, rh, dh, times[i]))
        state.update(rt, dt, rp, dp, rh, dh, wins[i], times[i])

    X = pd.DataFrame(rows, columns=DRAFT_FEATURES, index=df.index)
    return X, state


def training_mask(df: pd.DataFrame, warmup_days: int = 180) -> np.ndarray:
    """Skip the first months: ratings start from zero knowledge and would be noise."""
    return (df["start_time"] >= df["start_time"].iloc[0] + warmup_days * 86400).to_numpy()
