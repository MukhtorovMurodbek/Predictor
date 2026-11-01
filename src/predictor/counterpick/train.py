"""Train the counter-pick model.

    python -m predictor.counterpick.train
    python -m predictor.counterpick.train --source pro --patch 7.41
"""

import argparse
from pathlib import Path

from ..data import hero_arrays, load_heroes, load_matches
from .model import CounterModel, model_path


def train(source: str = "public", patch: str | None = None, prior: float = 150.0) -> tuple[CounterModel, Path]:
    df = load_matches(source, patch)
    if df.empty:
        raise ValueError("no matches for that source/patch")
    model = CounterModel.fit(*hero_arrays(df), load_heroes(), prior=prior)
    path = model_path(source)
    model.save(path)
    return model, path


def main(argv=None):
    p = argparse.ArgumentParser(description="Build hero counter/synergy matrices")
    p.add_argument("--source", choices=["public", "pro", "all"], default="public")
    p.add_argument("--patch", help="only use matches from this patch, e.g. 7.41")
    p.add_argument("--prior", type=float, default=150.0, help="smoothing strength, in games")
    args = p.parse_args(argv)

    try:
        model, path = train(args.source, args.patch, args.prior)
    except ValueError as e:
        raise SystemExit(str(e))
    w = model.weights
    print(f"fit on {model.n_matches} matches -> {path}")
    print(f"weights: intercept {w[0]:+.3f}, base {w[1]:.2f}, counter {w[2]:.2f}, synergy {w[3]:.2f}")


if __name__ == "__main__":
    main()
