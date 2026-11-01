import numpy as np

from predictor.counters import CounterModel
from predictor.data import resolve_heroes

HEROES = {i: f"Hero {i}" for i in range(1, 21)}


def synthetic_matches(n=20_000, seed=0):
    """Random 5v5 drafts where hero 1 hard-counters hero 2 and heroes 3+4 combo well."""
    rng = np.random.default_rng(seed)
    drafts = np.array([rng.choice(np.arange(1, 21), 10, replace=False) for _ in range(n)])
    R, D = drafts[:, :5], drafts[:, 5:]
    logit = np.zeros(n)
    logit += 1.5 * ((R == 1).any(1) & (D == 2).any(1))
    logit -= 1.5 * ((D == 1).any(1) & (R == 2).any(1))
    logit += 1.0 * ((R == 3).any(1) & (R == 4).any(1))
    logit -= 1.0 * ((D == 3).any(1) & (D == 4).any(1))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return R, D, y


def test_finds_planted_counter_and_synergy():
    R, D, y = synthetic_matches()
    m = CounterModel.fit(R, D, y, HEROES)
    assert m.advantage[1, 2] > 0.3
    assert m.advantage[1, 2] == max(m.advantage[1:, 2])
    assert m.with_[3, 4] > 0.2
    assert m.counters_for(2, top=1)["hero"][0] == "Hero 1"
    assert m.recommend(allies=[3], enemies=[2], top=2)["hero"].tolist()[:2] in (["Hero 1", "Hero 4"], ["Hero 4", "Hero 1"])


def test_predictions_beat_coin_flip():
    R, D, y = synthetic_matches()
    m = CounterModel.fit(R[:15000], D[:15000], y[:15000], HEROES)
    p = m.predict_proba(R[15000:], D[15000:])
    assert ((p > 0.5) == y[15000:]).mean() > 0.55


def test_save_load_roundtrip(tmp_path):
    R, D, y = synthetic_matches(2000)
    m = CounterModel.fit(R, D, y, HEROES)
    m.save(tmp_path / "m.npz")
    m2 = CounterModel.load(tmp_path / "m.npz")
    assert np.allclose(m.predict_proba(R, D), m2.predict_proba(R, D))
    assert m2.hero_names == HEROES


def test_resolve_heroes():
    names = {1: "Anti-Mage", 44: "Phantom Assassin", 12: "Phantom Lancer", 5: "Crystal Maiden", 120: "Pangolier"}
    assert resolve_heroes("pa, crystal, anti-mage", names) == [44, 5, 1]
    assert resolve_heroes("am", names) == [1]
    try:
        resolve_heroes("phantom", names)
    except ValueError as e:
        assert "ambiguous" in str(e)
    else:
        raise AssertionError("expected ambiguity error")
