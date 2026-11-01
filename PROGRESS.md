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
- 🔨 New direction: plain statistics over the last 3 months; STRATZ authentication and raw matchup queries work, but rank, time window and patch semantics remain unverified, so STRATZ is not used for recommendations yet → [connection test](docs/STRATZ-CONNECTION.md)
- ✅ Data: 44 k public Divine games
- ✅ Model: counter + synergy tables, beats the hero-only baseline (56.5% vs 56.0%)
- ✅ Hosted in the bot: `/counterpick` `/pick` `/counters` `/hero`
- ⬜ 200 k+ games (daily `/collect public 2500`)
- ⬜ Rank brackets, personal hero pool, roles
- ⬜ Website

## 2. Pro match predictor (pro matches only) 👤 Murodbek + Claude
- ✅ Data: 106,900 pro matches since 2023, with players, teams and league tiers
- ✅ Model: team/player ratings + form + head-to-head + draft → win %
- ✅ Tested on newer matches: **66% right on tier 1–2 matches**, 90% right when it says 80%+
- ✅ Hosted in the bot: `/pro` `/predict` `/team` `/top`
- ⬜ Tuning (Elo settings, patch meta, stand-ins)
- ⬜ Live in-game version (phase 4)

## 3. Pro draft predictor (next pick / ban) 👤 Murodbek + Claude
- ✅ Data: 105,779 pro drafts since 2023 with pick/ban order
- ✅ Model: meta, team habits, player comfort, hero combos → most likely next hero
- ✅ Tested: real hero in its top 5 for 24% of picks / 29% of bans (meta baseline: 16%)
- ✅ Hosted in the bot: `/draft` `/next`

## Extra: talk to the bot in plain language
- ✅ Evaluated: practical with a free cloud model (Groq). Needs a free API key
- ⬜ Build it (after the key is in `.env`)

## 4. Live in-game predictor
- ⬜ Next

## 5. Tournament (TI) predictor
- ⬜ Not started

## 6. Universal model
- ⬜ Not started (last)
