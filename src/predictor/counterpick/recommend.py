"""Counter-pick CLI (same statistics as the bot).

    python -m predictor.counterpick.recommend --enemies "Phantom Assassin, Pudge" --allies "Crystal Maiden"
    python -m predictor.counterpick.recommend --counters "Phantom Assassin"

Edges are in win-rate points: +0.05 = this hero wins 5 points more often than the two heroes'
usual win rates predict. Refresh the data with: python -m predictor.counterpick.update
"""

import argparse

import pandas as pd

from ..data import load_heroes, resolve_heroes
from .stats import MatchupStats


def main(argv=None):
    p = argparse.ArgumentParser(description="Suggest counter-picks")
    p.add_argument("--allies", help="comma-separated heroes already on your team")
    p.add_argument("--enemies", help="comma-separated enemy heroes")
    p.add_argument("--banned", help="comma-separated banned heroes")
    p.add_argument("--counters", help="show the best counters to this one hero")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)

    stats = MatchupStats.from_daily(load_heroes())
    if not stats.n_matches:
        raise SystemExit("no statistics yet, run: python -m predictor.counterpick.update")
    names = stats.hero_names
    pd.set_option("display.float_format", "{:+.3f}".format)
    pd.set_option("display.width", 120)
    try:
        if args.counters:
            (enemy,) = resolve_heroes(args.counters, names)
            print(f"Best counters to {names[enemy]} ({stats.n_matches:,} games, {stats.first_day} to {stats.last_day}):")
            print(stats.counters_for(enemy, top=args.top).to_string())
            return
        allies = resolve_heroes(args.allies, names)
        enemies = resolve_heroes(args.enemies, names)
        banned = resolve_heroes(args.banned, names)
    except ValueError as e:
        raise SystemExit(str(e))
    print("Allies: ", ", ".join(names[h] for h in allies) or "-")
    print("Enemies:", ", ".join(names[h] for h in enemies) or "-")
    print(stats.recommend(allies, enemies, banned, top=args.top).to_string())


if __name__ == "__main__":
    main()
