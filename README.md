# Predictor

[![tests](https://github.com/MukhtorovMurodbek/Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/MukhtorovMurodbek/Predictor/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dota 2 models, all hosted through one Telegram bot: a **hero counter-picker** for regular
games, then **pro match predictors** (draft, next pick, live game, tournament winner), and
finally one **universal model** that does everything.

- **[docs/PLAN.md](docs/PLAN.md)**: roadmap, what's done, what's next
- **[docs/BOT.md](docs/BOT.md)**: using and hosting the Telegram bot
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: how we work together, how to add a model

## Models

| # | Model | Trained on | Status | Bot commands |
|---|---|---|---|---|
| 1 | Counter-picker | public ranked games | ✅ live in the bot | `/pick` `/counters` `/hero` |
| 2 | Draft win predictor | pro matches only | ⬜ next | — |
| 3 | Pro next-pick predictor | pro drafts | ⬜ | — |
| 4 | Live in-game predictor | pro match timelines | ⬜ | — |
| 5 | Tournament (TI) predictor | pro results | ⬜ | — |
| 6 | Universal model | everything | ⬜ last | — |

Each model is built, then hosted in the bot, before the next one starts.

## Setup (everyone)

You need Python 3.11+ and git.

```bash
git clone https://github.com/MukhtorovMurodbek/Predictor.git
cd Predictor
python -m venv .venv
```

Activate the environment every time you open a new terminal:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

```bash
pip install -e ".[dev]"
pytest                # should say "passed"
```

## Command line

The same things the bot does, from a terminal:

```bash
# data (OpenDota free tier: 60 calls/min, 3,000/day per internet connection)
python -m predictor.data.collect heroes
python -m predictor.data.collect pro --since 2025-01-01
python -m predictor.data.collect public --calls 500 --min-rank 70

# model 1: counter-picker
python -m predictor.counterpick.train
python -m predictor.counterpick.evaluate
python -m predictor.counterpick.recommend --enemies "pa, pudge, lion" --allies "cm"
python -m predictor.counterpick.recommend --counters "Phantom Assassin"

# the bot (host only, see docs/BOT.md)
python -m predictor.bot
```

## Layout

```
src/predictor/
  data/            shared: OpenDota client, collectors, loading matches, hero names
  counterpick/     model 1: model.py, train.py, evaluate.py, recommend.py (CLI), bot.py (bot commands)
  bot/             Telegram bot: app.py, auth.py (allowed users), admin.py (/collect /train), registry.py (hosted models)
  draft/ ...       next models follow the same pattern
tests/             pytest tests, run automatically on GitHub for every push
scripts/           run_bot.bat: double-click to start the bot on Windows
docs/              plan and bot guide
data/, models/     downloaded data and trained models (not in git, see below)
```

## Where models and data live

Code is in git. Trained models and datasets are **not**: git keeps every version forever, so
the repo would grow huge. They're shared as **GitHub Release** assets of this repo when we
need to share them. Each person can also rebuild them with the commands above.

## License & data

The code is MIT-licensed, see [LICENSE](LICENSE).

Match data comes from the [OpenDota API](https://docs.opendota.com/). It isn't stored in
this repo, and each user downloads it with their own quota. If you publish a dataset or
results built from it, credit OpenDota and follow their terms. Dota 2 and hero names are
trademarks of Valve Corporation. This is an unofficial fan project, not affiliated with
Valve.
