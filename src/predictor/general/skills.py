"""Can hero skills alone predict counters? (phase 6, the "general predictor" idea)

Murodbek's idea: a model that understands each hero's skills, so when a patch changes a hero it
can name the new counters without waiting for statistics. This checks how far skills get us:

  hero_features()  one row of numbers per hero from OpenDota's hero and ability descriptions:
                   base stats, roles, ability types, damage types, BKB/dispel rules, keyword
                   counts (stun, illusion, heal, ...) and the ability text itself (TF-IDF)
  predict_counters() for a hero with no statistics: its counters = the real counters of the
                   heroes whose skills are most similar (weighted k nearest neighbours)
  evaluate()       leave-heroes-out: hide all statistics of 1/10 of the heroes, predict their
                   counters from skills, compare with the real matchups; repeat for every tenth

    python -m predictor.data.collect constants
    python -m predictor.general.skills          (about a minute)

Result on 2026-11-01 (7.5 M public games): about 3x more real top-5 counters than random
picks, correlation ~0.2 with the real matchup edges. Statistics are far better for heroes that
have games. Details in docs/PLAN.md, phase 6.
"""

import argparse
import json
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data import CONSTANTS_DIR, load_heroes

KEYWORDS = {  # counted in each hero's ability descriptions
    "stun": r"\bstun", "slow": r"\bslow", "silence": r"silenc", "root": r"\broot|ensnare|entangle", "hex": r"\bhex",
    "disarm": r"disarm", "break": r"\bbreak|passives? disabled", "invis": r"invisib", "illusion": r"illusion",
    "summon": r"summon|spawn", "heal": r"\bheal|restore", "lifesteal": r"lifesteal", "evasion": r"evasion|evade",
    "dispel": r"dispel|purge", "mana": r"\bmana\b", "manaburn": r"burn.{0,20}mana|mana.{0,20}(?:burn|drain)",
    "armor": r"\barmor", "magic_res": r"magic resist", "immune": r"immun",
    "mobility": r"blink|teleport|leap|jump|dash|charge", "dot": r"per second|over time|each second",
    "crit": r"critical", "cleave": r"cleave", "bash": r"\bbash", "pull": r"pull|drag|hook", "knockback": r"knock|push",
    "invulnerable": r"invulnerab", "reveal": r"true sight|reveal", "attack_speed": r"attack speed",
    "move_speed": r"movement speed|move speed", "shield": r"shield|barrier|block", "aura": r"\baura",
    "global": r"global|anywhere on the map", "mute": r"\bmute", "fear": r"\bfear|terrif", "taunt": r"taunt|berserker",
    "sleep": r"\bsleep", "lift": r"cyclone|lifts?", "on_attack": r"attacks? (?:deal|cause|apply)|on attack|each attack",
    "channel": r"channel", "creeps": r"\bcreep", "buildings": r"tower|building", "vision": r"vision",
    "reflect": r"reflect|return",
}
BEHAVIORS = ["Passive", "No Target", "Unit Target", "Point Target", "AOE", "Channeled", "Toggle", "Autocast", "Instant Cast", "Aura"]
ROLES = ["Carry", "Support", "Nuker", "Disabler", "Jungler", "Durable", "Escape", "Pusher", "Initiator"]
BASE = ["base_health", "base_health_regen", "base_mana", "base_mana_regen", "base_armor", "base_mr", "base_attack_min",
        "base_attack_max", "base_str", "base_agi", "base_int", "str_gain", "agi_gain", "int_gain", "attack_range",
        "projectile_speed", "attack_rate", "base_attack_time", "attack_point", "move_speed", "day_vision", "night_vision"]


def load_constants() -> tuple[dict, dict, dict]:
    """heroes (by id), hero_abilities (by npc name), abilities (by ability name)."""
    if not (CONSTANTS_DIR / "abilities.json").exists():
        raise FileNotFoundError("no hero/ability descriptions yet, run: python -m predictor.data.collect constants")
    load = lambda name: json.loads((CONSTANTS_DIR / f"{name}.json").read_text("utf-8"))
    return load("heroes"), load("hero_abilities"), load("abilities")


def _largest_number(value) -> float:
    """'-35%' / ['100', '160'] / '3.75' -> the biggest number (max level)."""
    found = [abs(float(m)) for v in (value if isinstance(value, list) else [value]) for m in re.findall(r"-?\d+(?:\.\d+)?", str(v))]
    return max(found) if found else 0.0


def hero_skills(hero: dict, ability_names: list, abilities: dict) -> tuple[dict, str]:
    """Numbers and text describing one hero's base stats and abilities."""
    row = {k: float(hero.get(k) or 0) for k in BASE}
    row.update({f"attr_{a}": float(hero.get("primary_attr") == a) for a in ("str", "agi", "int", "all")})
    row["melee"] = float(hero.get("attack_type") == "Melee")
    row.update({f"role_{r}": float(r in (hero.get("roles") or [])) for r in ROLES})
    counts = dict.fromkeys(["abilities", "innate", "bkb_pierce", "dispel_basic", "dispel_strong", "undispellable",
                            "targets_enemy", "targets_ally", "magical", "physical", "pure"] + BEHAVIORS, 0.0)
    biggest = dict.fromkeys(["stun", "damage", "radius", "duration", "slow", "cooldown", "mana_cost"], 0.0)
    texts = []
    flat = [n for item in ability_names for n in (item if isinstance(item, list) else [item])]
    for name in flat:
        a = abilities.get(name) or {}
        behavior = a.get("behavior") or []
        behavior = [behavior] if isinstance(behavior, str) else behavior
        if name == "generic_hidden" or not a.get("dname") or ("Hidden" in behavior and not a.get("is_innate")):
            continue
        counts["abilities"] += 1
        counts["innate"] += bool(a.get("is_innate"))
        for b in BEHAVIORS:
            counts[b] += b in behavior
        damage_type = (a.get("dmg_type") or "").lower()
        if damage_type in ("magical", "physical", "pure"):
            counts[damage_type] += 1
        counts["bkb_pierce"] += a.get("bkbpierce") == "Yes"
        counts["dispel_basic"] += a.get("dispellable") == "Yes"
        counts["dispel_strong"] += a.get("dispellable") == "Strong Dispels Only"
        counts["undispellable"] += a.get("dispellable") == "No"
        team = a.get("target_team") or []
        team = [team] if isinstance(team, str) else team
        counts["targets_enemy"] += "Enemy" in team
        counts["targets_ally"] += "Friendly" in team
        headers = []
        for attrib in a.get("attrib") or []:
            label = f"{attrib.get('key', '')} {attrib.get('header') or ''}".lower()
            headers.append((attrib.get("header") or "").lower())
            v = _largest_number(attrib.get("value"))
            for key, test in (("stun", "stun" in label and "duration" in label), ("radius", "radius" in label or "aoe" in label),
                              ("duration", "duration" in label), ("slow", "slow" in label or "movement_speed" in label),
                              ("damage", "damage" in label and "%" not in str(attrib.get("value")) and "pct" not in label)):
                if test:
                    biggest[key] = max(biggest[key], v)
        biggest["cooldown"] = max(biggest["cooldown"], _largest_number(a.get("cd") or 0))
        biggest["mana_cost"] = max(biggest["mana_cost"], _largest_number(a.get("mc") or 0))
        texts.append(f"{a['dname']}. {a.get('desc', '')} {' '.join(headers)}")
    text = " ".join(texts)
    row.update({f"ability_{k}": v for k, v in counts.items()})
    row.update({f"max_{k}": v for k, v in biggest.items()})
    row.update({f"says_{k}": float(len(re.findall(p, text.lower()))) for k, p in KEYWORDS.items()})
    return row, text


def hero_features(hero_ids: list[int], constants=None, train=None, text_dims: int = 24) -> np.ndarray:
    """(heroes, features), standardised. The text part (TF-IDF -> SVD) is fitted on the `train`
    heroes only, so a hidden hero's text can't leak into the vocabulary."""
    heroes, hero_abilities, abilities = constants or load_constants()
    rows, texts = [], []
    for h in hero_ids:
        hero = heroes[str(h)]
        row, text = hero_skills(hero, hero_abilities.get(hero["name"], {}).get("abilities", []), abilities)
        rows.append(row)
        texts.append(text)
    X = np.array([[r[k] for k in rows[0]] for r in rows])
    train = np.arange(len(hero_ids)) if train is None else np.asarray(train)
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    tfidf.fit([texts[i] for i in train])
    T = tfidf.transform(texts)
    dims = min(text_dims, T.shape[1] - 1, len(train) - 1)
    if dims > 0:
        X = np.hstack([X, TruncatedSVD(dims, random_state=0).fit(T[train]).transform(T)])
    sd = X[train].std(0)
    return (X - X[train].mean(0)) / np.where(sd > 0, sd, 1)


def predict_counters(F: np.ndarray, edge: np.ndarray, hero: int, known, k: int = 10) -> np.ndarray:
    """How well every hero does against `hero`, which has no statistics: edge[x, hero] is guessed
    as x's real edge against the `k` known heroes with the most similar skills (cosine).
    edge[x, y] > 0 = x wins more against y than their usual win rates predict."""
    known = np.asarray(known)
    unit = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    sim = unit[known] @ unit[hero]
    order = np.argsort(-sim)[:k]
    near, weight = known[order], np.clip(sim[order], 0, None) + 1e-6
    vals = edge[:, near]  # (all heroes, k)
    w = np.where(np.isfinite(vals), weight, 0.0)
    w[near, np.arange(len(near))] = 0  # x vs itself
    return np.nansum(np.where(np.isfinite(vals), vals, 0) * w, axis=1) / np.maximum(w.sum(1), 1e-9)


def evaluate(edge: np.ndarray, F_for, folds: int = 10, k: int = 10) -> dict:
    """Leave-heroes-out. `F_for(train)` gives the features (fitted on the train heroes).
    For each hidden hero h: predicted vs real edge[x, h] over the known heroes x."""
    n = len(edge)
    preds, reals, rhos, top5, top10 = [], [], [], [], []
    for f in range(folds):
        hidden = np.arange(n)[np.arange(n) % folds == f]
        known = np.setdiff1d(np.arange(n), hidden)
        F = F_for(known)
        masked = edge.copy()
        masked[:, hidden] = np.nan
        masked[hidden, :] = np.nan
        for h in hidden:
            p_all = predict_counters(F, masked, h, known, k)
            x = np.array([i for i in known if np.isfinite(edge[i, h])])
            p, real = p_all[x], edge[x, h]
            preds += list(p)
            reals += list(real)
            rhos.append(spearmanr(p, real).correlation)
            top5.append(len(set(x[np.argsort(-real)[:5]]) & set(x[np.argsort(-p)[:5]])) / 5)
            top10.append(len(set(x[np.argsort(-real)[:10]]) & set(x[np.argsort(-p)[:10]])) / 10)
    others = n - n // folds - 1
    return {"correlation": float(np.corrcoef(preds, reals)[0, 1]), "spearman_per_hero": float(np.nanmean(rhos)),
            "top5_hit": float(np.mean(top5)), "top10_hit": float(np.mean(top10)),
            "random_top5": 5 / others, "random_top10": 10 / others}


def main(argv=None):
    from ..counterpick.stats import Counts, DayTable, MatchupStats, daily_files

    p = argparse.ArgumentParser(description="Can hero skills alone predict counters? (leave-heroes-out test)")
    p.add_argument("--min-games", type=int, default=1000, help="matchups with fewer games are ignored")
    args = p.parse_args(argv)
    names = load_heroes()
    constants = load_constants()
    stats = MatchupStats.from_daily(names)
    c = stats.counts
    ids = [h for h in sorted(names) if str(h) in constants[0] and c.games[h] > args.min_games]
    idx = np.array(ids)
    games = c.vs_games[np.ix_(idx, idx)]
    edge = np.where((games >= args.min_games) & (games.T >= args.min_games), stats.vs_edge[np.ix_(idx, idx)], np.nan)
    print(f"{stats.n_matches:,} public games ({stats.first_day} to {stats.last_day}), {len(ids)} heroes, "
          f"matchup edges spread +/-{np.nanstd(edge) * 100:.1f} points")
    halves = [Counts.empty(), Counts.empty()]
    for i, f in enumerate(daily_files()):
        halves[i % 2] = halves[i % 2] + DayTable.load(f).counts()
    e1, e2 = (MatchupStats(h, names).vs_edge[np.ix_(idx, idx)] for h in halves)
    both = np.isfinite(edge) & np.isfinite(e1) & np.isfinite(e2)
    print(f"how repeatable the edges are (odd vs even days): r = {np.corrcoef(e1[both], e2[both])[0, 1]:.2f}")
    setups = {
        "base stats + roles": lambda known: hero_features(ids, constants, known, text_dims=0)[:, :len(BASE) + 5 + len(ROLES)],
        "+ abilities (numbers, keywords, text)": lambda known: hero_features(ids, constants, known),
    }
    for label, F_for in setups.items():
        r = evaluate(edge, F_for)
        print(f"{label:<40} correlation {r['correlation']:+.2f} | real top-5 counters found {r['top5_hit']:.0%} "
              f"(random {r['random_top5']:.0%}) | top-10 {r['top10_hit']:.0%} (random {r['random_top10']:.0%})")


if __name__ == "__main__":
    main()
