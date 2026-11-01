# Progress

Quick status. Details are in [docs/PLAN.md](docs/PLAN.md).
✅ done · 🔨 in progress · ⬜ not started · 👤 owner

_Last updated: 2026-11-01_

## Setup
- ✅ Public GitHub repo, MIT license, automatic tests on every push
- ✅ Plan, setup guide, team workflow docs, [AGENTS.md](AGENTS.md) for Codex and other AIs
- ✅ Telegram bot "Dota 2 Lab" on Murodbek's laptop, private to our 4 usernames, `/start`
  menu with one intro command per model → [docs/BOT.md](docs/BOT.md)

## 1. Counter-picker (public games) 👤 Asilbek (Codex + website)
- ✅ Data: 26 k public Divine games
- ✅ Legacy trained model: counter + synergy tables, with previously reported results of 56.5% vs 56.0% for the hero-only baseline
- ✅ Legacy model hosted in the bot: `/counterpick` `/pick` `/counters` `/hero`
- ⬜ 200 k+ games (daily `/collect public 2500`)
- 🔨 Website: statistical matchup explorer based on locally collected public games; no model training needed → [setup and limitations](docs/COUNTERPICK-WEB.md)
- ⬜ More matchup data, validated rank and patch coverage, personal hero pool, and position-aware suggestions
- ⬜ Decide how to bring the bot onto the website's statistical approach; it still uses the legacy trained model

## 2. Pro match predictor (pro matches only) 👤 Murodbek + Claude
- ✅ Data: 106,900 pro matches since 2023, with players, teams and league tiers
- ✅ Model: team/player ratings + form + head-to-head + draft → win %
- ✅ Tested on newer matches: **66% right on tier 1–2 matches**, 90% right when it says 80%+
- ✅ Hosted in the bot: `/pro` `/predict` `/team` `/top`
- ⬜ Tuning (Elo settings, patch meta, stand-ins)

## 3. Pro next-pick predictor
- ⬜ Not started

## 4. Live in-game predictor
- ⬜ Not started

## 5. Tournament (TI) predictor
- ⬜ Not started

## 6. Universal model
- ⬜ Not started (last)
