# Real-world validation (2026-11-01)

How the bot holds up against real results and against existing tools. Every test here uses
data the models had **not** seen (except where noted).

Repeat any of it:
- `python -m predictor.validate.event "<event name>"` (or `/replay <event>` in the bot)
- `python -m predictor.tournament.backtest`
- `/journal` in the bot (fills automatically every night from now on)

## 1. The bot itself, end to end in Telegram

All 25 commands were sent from a real Telegram account (Telegram Web) to @localHermesV0Bot.

- ✅ Every command answered, with correct formatting (bold, lists, emoji) and no errors.
- ✅ Wrong input gets a helpful answer: unknown hero, ambiguous team ("team" → list of
  options), unknown command, plain text.
- ✅ Admin jobs (`/collect`, `/evaluate`) report "Started…" and then "done in N s".
- ✅ Access control: a new allowed account was linked to its Telegram ID on its first message.
  Strangers get "Sorry, this is a private bot".
- ✅ Slowest answer: `/odds` (~6 s, it simulates 10,000 tournaments).
- ⚠️ Found and fixed: `/help` had grown past Telegram's 4,096-character limit, so it is now
  one line per command.
- ⚠️ Found and fixed: `/team falcons` showed a player who has since moved to another team.
  Rosters now drop players who have played for a different team since.
- ℹ️ Telegram Web's own autocomplete once turned `/top` into `/topic` while typing fast. That's
  the client, not the bot (the bot answered "I don't know that command").

## 2. Counter-picker vs Dotabuff

Dotabuff's Phantom Lancer "counters" table (100k+ games per pair) vs our statistics after
**one day** of data (91,617 public games):

| | |
|---|---|
| Dotabuff's top-10 counters, their rank in our list of 101 heroes | 1, 25, 5, 4, 3, 8, 6, 11, 14, 10 |
| Win rate vs PL, ours vs Dotabuff (21 heroes) | correlation 0.75, average gap 1.6 points |
| Ranking agreement (Spearman, 21 heroes) | 0.60 |

Our top PL counters: Tidehunter, Centaur, Sand King, Axe, Underlord, Warlock, Timbersaw, CM:
the same heroes Dotabuff lists. The one big miss (Leshrac, #25 for us) has only 212 games so
far. Numbers will tighten as the hourly updates fill the 90-day window.

## 3. Team ratings vs Dota2ProTips Elo

[Dota2ProTips](https://www.dota2protips.com/rankings/data) publishes an open Elo rating
(CC BY-NC 4.0). Compared on their 21 tier-1 teams:

| | |
|---|---|
| Their top 6 vs our top 6 | **the same six teams** (Yandex, PARIVISION, Spirit, Aurora, NAVI, Liquid) |
| Rating correlation | 0.87 (Pearson), 0.66 (Spearman) |
| Top-10 overlap | 7 of 10 |

e-Rankings' points-based Team World Ranking (Sept 2026) also has Team Yandex at #1, like us.
We rated OG, HULIGANI and GamerLegion higher than Dota2ProTips does. That pointed to matches
from OpenDota's lowest league tier ("excluded") inflating some teams. **Dropping that tier
improved tier-1/2 accuracy from 65.4% to 65.9%** (log loss 0.6128 → 0.6096), so the pro
models now use premium + professional leagues only.

## 4. Replaying real events

All models rebuilt from matches **before** each event, then asked about every game of it.

| | PARI UNIVERSE by FISSURE (81 maps) | BLAST SLAM VIII (87 maps) |
|---|---|---|
| Map winner, "higher rating wins" | 60.5% | 69.0% |
| Map winner, bot before the draft | 61.7% | 69.0% |
| Map winner, bot after the draft | 61.7% | **72.4%** |
| When the bot said 70%+ | right 82% (17 maps) | right 83% (12 maps) |
| Pre-series favourite won | 22 of 36 series | 40 of 58 series |
| Champion's title chance before the event | Team Spirit 14% (#3 of 9) | **Team Yandex 37% (#1 of 16)** |
| Real pick in the bot's top 5 | 21% | 24% |
| Real ban in the bot's top 5 | 27% | 24% |
| Live: right at 10 / 20 / 30 min | 78% / 83% / 83% | 77% / 77% / 86% |

(Run before the league-tier change. Event-sized samples have about ±5 points of noise.)

## 5. Past TIs

See `/tournament` in the bot or `docs/PLAN.md` (phase 5): champion ranked #1 (2023), #3
(2024), #1 (2025), #7 (2026).

## What this changed

- League-tier filter for all pro models (section 3)
- Roster tracking for transfers (section 1)
- Compact `/help` (section 1)
- Atomic file saves: a model being retrained at night can no longer be read half-written
- New tools: `/replay` for any past event, a nightly refresh with a **prediction journal**
  (`/journal`), so every new pro match becomes a real-world test automatically

## Not compared (yet)

- Betting odds (no clean source of historical odds)
- Other live win-probability graphs (STRATZ / Dota Plus): no open data
- No other public tool predicts pro picks and bans, so the "most popular hero" baseline is the
  comparison
