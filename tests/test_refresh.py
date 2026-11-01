import pandas as pd

from predictor import refresh as refresh_mod
from predictor.pro import ProModel
from predictor.pro.dataset import replay, training_mask
from predictor.pro.model import fit_classifiers

from test_pro import HEROES, TEAMS, synthetic_pro


def test_journal_scores_only_new_matches(tmp_path, monkeypatch):
    df = synthetic_pro(1600)
    old, new = df.iloc[:1500].reset_index(drop=True), df.iloc[1500:].reset_index(drop=True)
    X, state = replay(old)
    use = training_mask(old)
    pregame, draft = fit_classifiers(X[use], old.loc[use, "radiant_win"].to_numpy().astype(int))
    pro = ProModel(state, pregame, draft, TEAMS, {}, HEROES, len(old))

    rows = refresh_mod.journal_rows(new, pro, None, None)
    assert len(rows) == 100
    assert all(0 < r["p_draft"] < 1 and 0 < r["p_pregame"] < 1 for r in rows)
    right = sum((r["p_pregame"] > 0.5) == r["radiant_win"] for r in rows)
    assert right > 55  # the synthetic league has clear favourites

    monkeypatch.setattr(refresh_mod, "JOURNAL_PATH", tmp_path / "journal.csv")
    pd.DataFrame(rows).to_csv(tmp_path / "journal.csv", index=False)
    summary = refresh_mod.journal_summary()
    assert "100 maps" in summary and "before draft" in summary


def test_empty_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_mod, "JOURNAL_PATH", tmp_path / "missing.csv")
    assert "empty" in refresh_mod.journal_summary()
