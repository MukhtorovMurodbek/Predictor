# Instructions for AI coding agents (Codex, Claude, Copilot, ...)

Read this before changing anything. Humans: see README.md and CONTRIBUTING.md.

## What this repo is

Dota 2 models for a small team of students, served through one private Telegram bot
("Dota 2 Lab") that runs on Murodbek's laptop. Python 3.11+, installable package in `src/predictor`.

| Model | Folder | Data | Owner |
|---|---|---|---|
| 1. Counter-picker | `src/predictor/counterpick/` | **public** games, plain statistics (no model), last 90 days, auto-updated hourly | Asilbek (+ Codex), also a website |
| 2. Pro match predictor | `src/predictor/pro/` | **pro matches only** | Murodbek (+ Claude) |
| 3. Pro draft predictor (next pick/ban) | `src/predictor/draft/` | **pro drafts only** | Murodbek (+ Claude) |
| 4. Live in-game predictor | `src/predictor/live/` | **pro** match timelines | Murodbek (+ Claude) |
| 5. Tournament (TI) predictor | `src/predictor/tournament/` | pro model + simulation | Murodbek (+ Claude) |
| 6. Universal model | not started | everything | see `docs/PLAN.md` |

Status lives in `PROGRESS.md`. The plan lives in `docs/PLAN.md`.

## Setup and checks

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest -q                    # must pass before you finish
```

Tests don't need the internet, data files or the bot token. GitHub Actions runs them on
every push and PR. Keep it that way: tests build small synthetic data (see
`tests/test_counterpick.py`, `tests/test_pro.py`).

## Hard rules

1. **Never commit** `.env`, `data/`, `models/`, `*.parquet`, `*.npz`, `*.pkl`. They're
   gitignored. Never print, log or copy the bot token or the `usernames` list.
2. **Don't push to `main` directly.** Work on a branch (`feat/...`, `fix/...`), open a PR,
   and describe what changed and how you tested it. Never force-push `main`.
3. **Stay in your area.** Asilbek's work goes in `counterpick/` and the website.
   Don't rewrite `pro/`, `bot/` or `data/` without saying so in the PR. Shared code
   (`data/`, `bot/`) must keep working for every model.
4. **Pro predictors use pro matches only.** The counter-picker uses public games.
5. **No leakage.** Features for a match may only use matches *before* it. Evaluate with a
   time split (train older, test newer), never a random split. See `pro/features.py`
   (`ProState`) for the pattern.
6. **Data source is OpenDota** (`data/opendota.py`, rate-limited: 60 calls/min, 3,000/day).
   Don't scrape Dotabuff or other sites: it's against their terms, and they block bots.
   STRATZ's API (free token) is an allowed extra source.
7. **Don't add heavy dependencies** without a reason in the PR. The host laptop has 4 GB
   of GPU memory and 16 GB of RAM. Models must load and answer in about a second.
8. **Update `PROGRESS.md`** (✅/🔨/⬜ lines) when you finish a milestone.

## Layout

```
src/predictor/
  __init__.py      ROOT, DATA_DIR, RAW_DIR, MODELS_DIR paths
  data/            shared: OpenDota client, collectors (python -m predictor.data.collect ...),
                   load_matches(), resolve_heroes() (hero names like "pa", "cm")
  counterpick/     model 1: stats.py (Counts per day, MatchupStats), update.py (hourly refresh),
                   recommend.py (CLI), bot.py; Asilbek's version: statistics.py (website data),
                   engine.py (Python port of web/static/engine.js), website_bot.py (/asil, /apick)
  pro/             model 2: features.py (ProState), dataset.py, model.py (ProModel),
                   train.py, evaluate.py, bot.py
  draft/           model 3: features.py (DraftState), model.py (DraftModel, replay), train.py, bot.py
  live/            model 4: features.py (scoreboard), model.py (LiveModel), train.py, bot.py
  tournament/      model 5: sim.py (Monte Carlo of a TI-style event), backtest.py, bot.py
  bot/             Telegram bot: app.py, auth.py, admin.py (/collect /train /evaluate),
                   registry.py (the list of hosted models), config.py (.env)
web/              Asilbek's counter-picker website (python -m web.server, loopback only)
tests/             pytest
docs/              PLAN.md (roadmap), BOT.md (bot usage + hosting)
```

## How a model plugs into the bot

Each model folder has a `bot.py` exposing exactly these names (see `bot/registry.py`):
`NAME`, `TITLE`, `SUMMARY`, `help_text()`, `COMMANDS = [(command, async handler, menu text)]`,
`status()`, `train_model() -> str` and `evaluate_model() -> str`, and optionally an async `background()`
loop the bot starts at launch (the counter-picker uses it for hourly updates). Then add the module to
`MODELS` in `bot/registry.py`. The bot picks up the rest by itself: the `/NAME` intro, the
`/start` menu, `/help`, `/train NAME` and `/evaluate NAME`.

Bot message rules:
- Replies are Telegram **HTML**. Only `<b> <i> <code> <pre>` tags. Escape every dynamic
  string with `html.escape`. Keep messages under 4,096 characters and readable on a phone
  (no wide tables).
- Every command explains itself when called with no arguments, with a copy-pastable example.
- Command names: lowercase `a-z0-9_`, unique across all models (a test checks this).
- Long work (training, downloading) goes through `bot/admin.py` jobs, never directly in a
  handler.
- Test replies with the fake-message helpers in `tests/test_bot.py`, including
  `assert_telegram_html`.

## Style

- Match the surrounding code: type hints, short docstrings that say *why*, plain names,
  no clever one-liners.
- Every model has a `train.py` with a CLI (`python -m predictor.<model>.train`) and an
  `evaluate.py` that compares against simple baselines on newer held-out matches. Put the
  numbers in your PR.
- Keep user-facing text simple. The team is new to ML.

## For the website (Asilbek)

The website lives in `web/` and is also hosted in the bot (`/asil`, `/apick`). When you change
its ranking rules in `web/static/engine.js`, change `src/predictor/counterpick/engine.py` the same
way (`tests/test_counterpick_engine.py` checks the counting against `statistics.py`).

Put new website code in `web/` and import models from the package
(`from predictor.counterpick import MatchupStats`) instead of copying code. Note: the raw file
`data/raw/public_matches.parquet` keeps up to the **last 60 days**; historical recovery may
still be in progress. The 90-day numbers live in
the daily tables `data/counterpick/*.npz` (`MatchupStats.from_daily(...)`). If the site
needs an HTTP API, add a small FastAPI app under `web/` that loads the same model files.
Don't change model file formats without updating `bot.py` too.
