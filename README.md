# Predictor

Dota 2 draft & match predictor: a **hero counter-picker** first, then **win prediction**
for drafts, live games and whole tournaments (The International).

See **[docs/PLAN.md](docs/PLAN.md)** for the full plan, hardware notes and roadmap, and
**[CONTRIBUTING.md](CONTRIBUTING.md)** for how we work together.

## Status

Phase 1, counter-picker MVP: data collection, counter/synergy matrices, CLI recommender,
evaluation.

## Quickstart

Needs Python 3.11+.

```bash
git clone https://github.com/MukhtorovMurodbek/Predictor.git
cd Predictor
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. Collect data (OpenDota, free tier: 60 calls/min, 3,000/day)

```bash
python -m predictor.collect heroes                       # hero list + patch dates
python -m predictor.collect pro --since 2025-01-01       # all pro matches (~10 calls)
python -m predictor.collect public --calls 500 --min-rank 70   # ~40k Ancient+ games, ~9 min
```

Data goes to `data/` (not committed). Run `collect public` again on later days to grow the
dataset. Every run adds to the same file and skips duplicates.

### 2. Build the counter model

```bash
python -m predictor.counters --source public             # -> models/counters_public.npz
python -m predictor.counters --source pro --patch 7.41
```

### 3. Ask for picks

```bash
python -m predictor.recommend --enemies "Phantom Assassin, Pudge, Lion" --allies "Crystal Maiden"
python -m predictor.recommend --counters "Phantom Assassin"
```

### 4. Check whether it actually predicts anything

```bash
python -m predictor.evaluate --source public
```

This compares against "always Radiant" and a per-hero-only logistic regression, on
held-out **newer** matches.

## Layout

```
src/predictor/
  opendota.py   rate-limited OpenDota API client
  collect.py    data collectors (pro via explorer SQL, public via /publicMatches)
  data.py       loading matches + hero name lookup
  counters.py   counter & synergy matrices, draft scoring, training CLI
  recommend.py  counter-pick CLI
  evaluate.py   time-split evaluation vs baselines
docs/PLAN.md    plan & roadmap
```
