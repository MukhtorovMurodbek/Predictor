# Progress

Quick status. Details are in [docs/PLAN.md](docs/PLAN.md).
✅ done · 🔨 in progress · ⬜ not started · 👤 owner

_Last updated: 2026-11-01 (evening)_

## Setup
- ✅ Public GitHub repo, MIT license, automatic tests on every push
- ✅ Plan, setup guide, team workflow docs, [AGENTS.md](AGENTS.md) for Codex and other AIs
- ✅ Telegram bot "Dota 2 Lab" on Murodbek's laptop, private to our 4 usernames, `/start`
  menu with one intro command per model → [docs/BOT.md](docs/BOT.md)

## Real-world testing → [docs/VALIDATION.md](docs/VALIDATION.md)
- ✅ All 25 bot commands tested from a real Telegram account
- ✅ Counter-picker vs Dotabuff: same top counters for Phantom Lancer after 1 day of data
- ✅ Team ratings vs Dota2ProTips Elo: same top 6, correlation 0.87
- ✅ Replayed 2 real events with models frozen before them (BLAST SLAM VIII: 72% of maps, champion was our #1 pick)
- ✅ Fixes from testing: transfers/rosters, league-tier filter (+0.5% accuracy), /help length, safe file saves
- ✅ Nightly auto-refresh at 05:00 + prediction journal (`/journal`), `/replay <event>`

## 1. Counter-picker (public games) 👤 Asilbek (Codex + website)
- ✅ Now plain statistics, no model: win rates per hero pair, last 90 days, like Dotabuff
- ✅ Updates itself every hour (new games + backfill); raw retention is 60 days
- ✅ Hosted in the bot: `/counterpick` `/pick` `/counters` `/hero`
- ✅ Phantom Lancer check: 8 of our top 10 counters are in Dotabuff's top 14
- 🔨 Data growing: 79 k games on day 1, up to ~200 k/day
- ✅ Asilbek's website version: average win rate vs the enemies, rank and patch filters → [setup](docs/COUNTERPICK-WEB.md)
- ✅ Asilbek's version also in the bot: `/asil` `/apick` (same data, his ranking rules)
- 🔨 `/apick` uses up to the last 60 days; older raw history backfills when the API budget allows
- 🔨 Asilbek: STRATZ connection test on branch `feat/stratz-connection-latest` (not merged yet)
- ⬜ Rank brackets in the 90-day tables, personal hero pool, roles

## 2. Pro match predictor (pro matches only) 👤 Murodbek + Claude
- ✅ Data: 106,900 pro matches since 2023, with players, teams and league tiers
- ✅ Model: team/player ratings + form + head-to-head + draft → win %
- ✅ Tested on newer matches: **65% right on tier 1–2 matches**, 90% right when it says 80%+
- ✅ Hosted in the bot: `/pro` `/predict` `/team` `/top`
- ⬜ Tuning (Elo settings, patch meta, stand-ins)

## 3. Pro draft predictor (next pick / ban) 👤 Murodbek + Claude
- ✅ Data: 105,779 pro drafts since 2023 with pick/ban order
- ✅ Model: meta, team habits, player comfort, hero combos → most likely next hero
- ✅ Tested: real hero in its top 5 for 26% of picks / 35% of bans (meta baseline: 18–20%)
- ✅ Hosted in the bot: `/draft` `/next`

## Extra: talk to the bot in plain language
- ✅ Evaluated: practical with a free cloud model (Groq). Needs a free API key
- ⬜ Build it (after the key is in `.env`)

## 4. Live in-game predictor 👤 Murodbek + Claude
- ✅ Data: gold/XP per minute + towers, Roshan, first blood for 106,752 pro matches
- ✅ Model: team strength + draft + scoreboard → win %
- ✅ Tested: right 66% at the start, 72% at 10 min, 79% at 20 min, 82% at 30 min
- ✅ Hosted in the bot: `/livepro` `/live` `/ingame`

## 5. Tournament (TI) predictor 👤 Murodbek + Claude
- ✅ Simulates a TI-style event 10,000 times with the pro model's map chances
- ✅ Backtested on TI 2023–2026: winner ranked #1 twice, #3 once, #7 once (avg ~16% vs 6% random)
- ✅ Hosted in the bot: `/tournament` `/odds`

## 6. Universal model
- ⬜ Next (last)
