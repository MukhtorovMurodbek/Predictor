import pandas as pd

from predictor.validate.event import _series


def test_series_grouping():
    """Maps between the same two teams within 6 hours form one series."""
    h = 3600
    df = pd.DataFrame([
        {"start_time": 0, "radiant_team_id": 1.0, "dire_team_id": 2.0, "radiant_win": True},
        {"start_time": 1 * h, "radiant_team_id": 2.0, "dire_team_id": 1.0, "radiant_win": True},
        {"start_time": 2 * h, "radiant_team_id": 1.0, "dire_team_id": 2.0, "radiant_win": False},
        {"start_time": 3 * h, "radiant_team_id": 3.0, "dire_team_id": 4.0, "radiant_win": True},
        {"start_time": 30 * h, "radiant_team_id": 1.0, "dire_team_id": 2.0, "radiant_win": True},
    ])
    assert [len(s) for s in _series(df)] == [3, 1, 1]
