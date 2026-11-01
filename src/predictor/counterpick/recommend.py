"""Counter-pick CLI.

    python -m predictor.counterpick.recommend --enemies "Phantom Assassin, Pudge" --allies "Crystal Maiden"
    python -m predictor.counterpick.recommend --counters "Phantom Assassin"

Scores are in log-odds. Near 50% win chance, +0.1 log-odds is about +2.5 percentage points.
"""

import argparse

import pandas as pd

from .model import CounterModel, model_path
from ..data import resolve_heroes


def main(argv=None):
    p = argparse.ArgumentParser(description="Suggest counter-picks")
    p.add_argument("--source", choices=["public", "pro", "all"], default="public", help="which trained model")
    p.add_argument("--allies", help="comma-separated heroes already on your team")
    p.add_argument("--enemies", help="comma-separated enemy heroes")
    p.add_argument("--banned", help="comma-separated banned heroes")
    p.add_argument("--counters", help="show the best counters to this one hero")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)

    path = model_path(args.source)
    if not path.exists():
        raise SystemExit(f"{path} missing, run: python -m predictor.counterpick.train --source {args.source}")
    model = CounterModel.load(path)
    names = model.hero_names

    pd.set_option("display.float_format", "{:+.3f}".format)
    pd.set_option("display.width", 120)
    try:
        if args.counters:
            (enemy,) = resolve_heroes(args.counters, names)
            print(f"Best counters to {names[enemy]} ({model.n_matches} matches in model):")
            print(model.counters_for(enemy, top=args.top).to_string())
            return
        allies = resolve_heroes(args.allies, names)
        enemies = resolve_heroes(args.enemies, names)
        banned = resolve_heroes(args.banned, names)
    except ValueError as e:
        raise SystemExit(str(e))

    print("Allies: ", ", ".join(names[h] for h in allies) or "-")
    print("Enemies:", ", ".join(names[h] for h in enemies) or "-")
    recs = model.recommend(allies, enemies, banned, top=args.top)
    recs["≈Δwin%"] = (recs["score"] * 25).map("{:+.1f}".format)
    print(recs.to_string())


if __name__ == "__main__":
    main()
