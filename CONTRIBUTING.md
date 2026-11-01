# Working together

We're in different cities and time zones, so the defaults are: **write things down, work
async, keep `main` working.**

Using an AI coding agent (Codex, Claude, ...)? Point it at [AGENTS.md](AGENTS.md) first.

## Flow

1. Pick or create an **Issue**. Say what "done" looks like. Assign yourself.
2. Branch from `main`: `git switch -c feat/short-name` (or `fix/...`, `data/...`).
3. Commit small, push often. Open a **draft PR** early so others can see where you are.
4. Before merging: `pytest` passes (GitHub runs it too and shows ✅/❌ on the PR), and one
   teammate approves.
5. Squash-merge into `main`.

## Rules of thumb

- **Never commit data, models or `.env`.** `data/`, `models/` and `.env` are gitignored.
- **Don't commit API keys or the bot token.** They live in `.env` on the host laptop only.
- **Evaluate with a time split.** Train on older matches, test on newer ones.
- **Pro predictors train on pro matches only.** The counter-picker uses public games
  (it's for regular players).
- New model or change? Put its numbers in the PR description (log loss, Brier, accuracy vs
  baseline).

## Adding a new model

Each model gets its own folder under `src/predictor/`, built the same way as `counterpick/`:

| File | Purpose |
|---|---|
| `model.py` | the model class: fit / predict / save / load |
| `train.py` | `train()` function + command line: `python -m predictor.<model>.train` |
| `evaluate.py` | `evaluate() -> str` report vs baselines, on newer held-out matches |
| `bot.py` | Telegram commands: `NAME`, `TITLE`, `HELP`, `COMMANDS`, `status()`, `train_model()`, `evaluate_model()` |

Then add it to `MODELS` in `src/predictor/bot/registry.py`. The bot picks up its commands,
help section, `/status` line, `/train <name>` and `/evaluate <name>` on its own. Add tests in
`tests/` (see `tests/test_bot.py` for how to test bot replies without Telegram).

Order: **build → evaluate → host in the bot → next model.**

## Data runs

Nothing to do by hand any more. Every hour the bot counts every public game OpenDota lists
into the counter-picker's daily tables (OpenDota's database does the counting, a few calls a
day), and every night at 05:00 it fetches new pro matches and retrains the pro models.
`/train counterpick` and `/refresh` run them right away.
