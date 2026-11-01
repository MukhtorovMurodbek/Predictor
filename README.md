# Predictor

[![tests](https://github.com/MukhtorovMurodbek/Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/MukhtorovMurodbek/Predictor/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dota 2 models, all hosted through one Telegram bot: a **hero counter-picker** for regular
games, then **pro match predictors** (draft, next pick, live game, tournament winner), and
finally a **general predictor** that answers plain English questions with all of them.

- **[docs/PLAN.md](docs/PLAN.md)**: roadmap, what's done, what's next
- **[docs/BOT.md](docs/BOT.md)**: using and hosting the Telegram bot
- **[PROGRESS.md](PROGRESS.md)**: short checklist of what's done
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: how we work together, how to add a model
- **[AGENTS.md](AGENTS.md)**: rules for AI coding agents (Codex, Claude, ...)
- **[docs/COUNTERPICK-WEB.md](docs/COUNTERPICK-WEB.md)**: run the new statistics-based counter-picker website

The website is Asilbek's statistics-based counter-picker (no training). The bot hosts both
counter-pickers: `/counterpick` (90-day matchup edges) and `/asil` (Asilbek's ranking).

## Models

| # | Model | Trained on | Status | Bot commands |
|---|---|---|---|---|
| 1 | Counter-picker (statistics, no model) | public games, all ranks, last 90 days, auto-updated hourly | ✅ live in the bot | `/counterpick` `/pick` `/counters` `/hero` |
| 1b | Asilbek's counter-picker (website + bot) | public games, last 30 days, rank/patch filters | ✅ live in the bot | `/asil` `/apick` |
| 2 | Pro match predictor | pro matches only | ✅ live in the bot | `/pro` `/predict` `/team` `/top` |
| 3 | Pro draft predictor (next pick/ban) | pro drafts only | ✅ live in the bot | `/draft` `/next` |
| 4 | Live in-game predictor | pro match timelines | ✅ live in the bot | `/livepro` `/live` `/ingame` |
| 5 | Tournament (TI) predictor | pro model + simulated TI format | ✅ live in the bot | `/tournament` `/odds` |
| 6 | General predictor (plain English, all models; skills test done) | everything | 🔨 idea tested | — |

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
python -m predictor.data.collect pro                 # since 2023 the first time, then only new matches

# model 1: counter-picker (statistics; the bot runs the update every hour by itself)
python -m predictor.counterpick.update               # count new + older public games into daily tables
python -m predictor.counterpick.recommend --enemies "pa, pudge, lion" --allies "cm"
python -m predictor.counterpick.recommend --counters "Phantom Assassin"

# model 2: pro match predictor
python -m predictor.pro.train                        # ~2 min, prints its accuracy report
python -m predictor.pro.evaluate

# model 3: pro draft predictor
python -m predictor.draft.train                      # ~4 min, prints its top-5 accuracy

# model 4: live in-game predictor
python -m predictor.live.train                       # ~3 min

# model 5: tournament predictor (uses the pro model)
python -m predictor.tournament.backtest             # test on TI 2023-2026, ~5 min

# the bot (host only, see docs/BOT.md)
python -m predictor.bot
```

## Layout

```
src/predictor/
  data/            shared: OpenDota client, collectors, loading matches, hero names
  counterpick/     model 1: stats.py (daily win/game counts), update.py (hourly refresh), recommend.py (CLI), bot.py
  bot/             Telegram bot: app.py, auth.py (allowed users), admin.py (/collect /train), registry.py (hosted models)
  pro/             model 2: features.py (ratings, form, draft), model.py, train.py, evaluate.py, bot.py
  draft/           model 3: features.py (draft habits), model.py, train.py, bot.py
  live/            model 4: features.py (scoreboard), model.py, train.py, bot.py (/live, /ingame)
  tournament/      model 5: sim.py (Swiss + double elimination Monte Carlo), backtest.py, bot.py
  ...              next models follow the same pattern
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
