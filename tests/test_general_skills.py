import numpy as np

from predictor.general import skills

KINDS = ["illusion", "area", "plain"]
DESCRIPTIONS = {
    "illusion": "Creates illusion copies of the hero that attack the target.",
    "area": "Deals magical damage in an area and stuns enemies inside it.",
    "plain": "Gains bonus attack damage and lifesteal for a short time.",
}


def synthetic_constants(n=36):
    """Heroes of three kinds, told apart only by their ability descriptions."""
    heroes, hero_abilities, abilities = {}, {}, {}
    for h in range(1, n + 1):
        kind = KINDS[h % 3]
        npc = f"npc_dota_hero_h{h}"
        heroes[str(h)] = {"name": npc, "primary_attr": "agi", "attack_type": "Melee" if h % 2 else "Ranged",
                          "roles": ["Carry"], "base_armor": float(h % 4), "move_speed": 300.0}
        abilities[f"h{h}_skill"] = {"dname": f"Skill {h}", "desc": DESCRIPTIONS[kind] + f" Variant {h % 5}.",
                                    "behavior": ["Unit Target", "AOE"] if kind == "area" else "No Target",
                                    "dmg_type": "Magical" if kind == "area" else "Physical",
                                    "bkbpierce": "Yes" if kind == "area" else "No",
                                    "attrib": [{"key": "stun_duration", "header": "STUN DURATION:", "value": ["1", "1.5", "2"]}] if kind == "area" else []}
        hero_abilities[npc] = {"abilities": [f"h{h}_skill", "generic_hidden"]}
    return heroes, hero_abilities, abilities


def test_ability_numbers():
    heroes, hero_abilities, abilities = synthetic_constants()
    row, text = skills.hero_skills(heroes["1"], hero_abilities["npc_dota_hero_h1"]["abilities"], abilities)  # an area hero
    assert row["ability_abilities"] == 1 and row["ability_bkb_pierce"] == 1 and row["ability_magical"] == 1
    assert row["max_stun"] == 2 and row["says_stun"] == 2 and row["melee"] == 1  # "stuns" + the "STUN DURATION" header
    assert "Skill 1" in text


def test_hidden_heroes_get_their_kinds_counters():
    """Area beats illusion beats plain beats area. A hero with no statistics should still get
    the right counters from its description alone."""
    constants = synthetic_constants()
    n = len(constants[0])
    kind = [KINDS[h % 3] for h in range(1, n + 1)]
    beats = {("area", "illusion"), ("illusion", "plain"), ("plain", "area")}
    rng = np.random.default_rng(0)
    edge = rng.normal(0, 0.004, (n, n))
    for x in range(n):
        for h in range(n):
            edge[x, h] += 0.03 * ((kind[x], kind[h]) in beats) - 0.03 * ((kind[h], kind[x]) in beats)
    edge = (edge - edge.T) / 2
    np.fill_diagonal(edge, np.nan)
    ids = list(range(1, n + 1))
    r = skills.evaluate(edge, lambda known: skills.hero_features(ids, constants, known, text_dims=4), folds=6)
    assert r["correlation"] > 0.8
    assert r["top5_hit"] > 2 * r["random_top5"]  # ~10 right answers among 29 heroes: ~50% at best
