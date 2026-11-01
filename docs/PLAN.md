# Predictor — Plan

Goal: one project that can (1) **recommend picks / counter-picks** and (2) **predict who wins**:
a single draft, a live game, and in the end a whole tournament like The International.

The key idea that lets us do "both": **a counter-picker is a win-probability model asked a
different question.** If a model knows `P(win | my 5 heroes, their 5 heroes)`, the best
counter-pick is just the hero that increases that number the most. So we build the draft
model once and use it in two ways.

```
                 ┌──────────────────────────────┐
  match data ──► │  draft model P(win | draft)  │ ──► "Radiant 57%"   (predictor)
                 └──────────────┬───────────────┘
                                │ try every free hero, rank by ΔP
                                ▼
                      "pick Axe: +3.1%"           (counter-picker)
```

---

## 1. What a 3050 Ti laptop can do

Specs checked: RTX 3050 Ti Laptop (4 GB VRAM), i5-12500H, 16 GB RAM.

**Short answer: compute isn't the problem. Data is.** Dota draft data is tiny compared to
images or text: one match is 10 hero IDs plus a label. Two million matches fit in about 40 MB
of RAM.

| Task | Size | Laptop fine? |
|---|---|---|
| Counter/synergy matrices (stats) | 127×127 tables | ✅ seconds, CPU |
| Logistic regression / LightGBM on drafts | 1–5 M matches | ✅ minutes, CPU |
| Small neural draft model (hero embeddings, MLP or small transformer, ~1–5 M params) | 1–5 M matches | ✅ minutes on GPU, 4 GB is plenty |
| Next-pick prediction for pro teams (sequence model over the draft) | ~50 k pro drafts | ✅ |
| Live win probability from gold/XP/kill timelines | ~50 k pro matches × per-minute rows | ✅ |
| Monte Carlo of a whole TI bracket (100 k simulations) | — | ✅ seconds |
| Parsing raw replays yourself (.dem files, 50–100 MB each) | thousands | ⚠️ slow and takes a lot of disk; skip it, OpenDota already parsed pro games |
| Reading the live broadcast with computer vision | — | ❌ not worth it on this hardware; use Game State Integration (see phase 4) |
| Training or fine-tuning a big LLM "Dota advisor" | — | ❌ out of scope |

**The real limit is the API quota:** OpenDota free tier is **60 calls/min and 3,000
calls/day per IP** (seen in the response headers). Each of us has our own IP, so with three
people we get three quotas. That's a natural way to split collection work.

### What accuracy to expect (be honest with ourselves)

- **Draft-only win prediction** is noisy. Published hero-only models usually land around
  **55–60% accuracy**. Players matter more than heroes. If we beat "always pick Radiant"
  plus a per-hero baseline by a few points, the counter signal is real.
- **Live prediction** gets much better as the game goes on. Gold/XP lead at 20 minutes is a
  strong signal. This is where predictions start to look impressive.
- **First blood / first Roshan** are close to coin flips from the draft alone. They're fun to
  build, but expect small edges.
- **TI winner**: the favourite usually has well under a 50% chance (TIs are famous for upsets).
  The goal is **calibrated probabilities**, not "calling the winner". We judge it with a
  Brier score / log loss on past TIs and compare against betting-market odds.

---

## 2. Data sources

| Source | What we get | Notes |
|---|---|---|
| **OpenDota `/api/explorer`** (SQL over the pro DB) | All pro matches: heroes, picks/bans order, teams, league, patch, duration | ~46 k pro matches since Jan 2025. 60 k rows per query. Our main pro source. |
| **OpenDota `/api/publicMatches`** | Public ranked matches with 5v5 hero IDs and avg rank | 100 per call. Filter by `min_rank`. Volume for counter stats. |
| **OpenDota `/api/matches/{id}`** | Full match: per-minute `radiant_gold_adv`, `radiant_xp_adv`, objectives (first blood, towers, Roshan), players | 1 call per match. Use for the live-prediction dataset (phase 4). |
| **OpenDota `/api/live`**, Steam `GetLiveLeagueGames` | Currently live top / league games | For real-time demos. Check what still works when we get there. |
| **Dota 2 Game State Integration (GSI)** | Live JSON from the Dota client to localhost while you play or **spectate** | The cleanest real-time feed. Spectate a pro game in-client and our app gets draft + game state. |
| **STRATZ GraphQL API** | Alternative stats, free token, matchup data | Backup if OpenDota limits hurt. |
| ~~Dotabuff~~ | — | Suggested by Asilbek, but there's **no public API**. Scraping is against their terms and blocked by Cloudflare. Their numbers come from the same Valve match data OpenDota and STRATZ give us legally. |
| **Patches**: `/api/constants/patch` | Patch dates | Meta shifts every patch: weight recent patches. |

**Patch counts in the pro DB:** 7.39 ≈ 17.9 k, 7.40 ≈ 7.9 k, 7.41 ≈ 8.7 k matches.
A single patch of pro games is **too small for 127×127 matchup tables** (~25 games per hero
pair). So: **learn counters from high-rank public games (volume), then adjust for pros.**

---

## 3. Roadmap

**Team decisions (2026-11-01):**
- One **separate, high-precision model per task** first. A **universal model** comes last
  (phase 6).
- The order for every model is **build → evaluate → host in the Telegram bot → next model**.
  All models run in one bot on the host laptop ([docs/BOT.md](BOT.md)).
- The **counter-picker uses public games** (it's for regular players). **All predictors use
  pro matches only** (Asilbek).

Each phase ends with something that works and that we can demo. Do them in order.

### Phase 1 — Counter-picker (public games) ✅ hosted · 👤 Asilbek

**Now plain statistics, not a model (2026-11-01, Asilbek + Murodbek).** Done:
- `counterpick/stats.py`: raw wins/games per hero and hero pair, kept as **one small table per
  day**. The bot sums the **last 90 days**. Edge = real win rate in a matchup minus what the two
  heroes' usual win rates predict (like Dotabuff's advantage). Pairs under 200 games are hidden.
- `counterpick/update.py`, run by the bot **every hour**: OpenDota's database counts the games
  for us (one SQL query per 4-hour block returns the wins and games of every hero pair, split
  by medal), so we get **every ranked public game OpenDota lists, ~550,000 a day**, not a
  sample, for ~6 calls a day. Older blocks are counted until the 90-day window is full.
  (Until 2026-11-01 it downloaded 100 pages of games an hour, ~110k games a day with gaps, and
  backfill never got a turn. Asilbek noticed /apick only went back to Oct 31.)
- Asilbek's `/apick` and website: the last **30 days** (he asked for data from Oct 1), filters
  by medal and patch; same numbers as his `build_snapshot()` on the same games.
- The old trained counter model was removed.
- Check vs Dotabuff (Phantom Lancer, after one day of data): 8 of our top 10 counters are in
  Dotabuff's top 14 (Tidehunter, Axe, Timbersaw, Underlord, Sand King, Warlock, CM, Sven).
- STRATZ would give far more data at once, but its API needs an account with 100+ public
  games. Murodbek's didn't qualify. Asilbek could create the token instead.

How many games is enough? A matchup win rate from *n* games is accurate to about ±1/√n
(95%): 1,000 games → ±3.1 points, 10,000 → ±1.0, 100,000 → ±0.3. Counter effects are usually
1–5 points, so **1k is too noisy, 10k is OK, 100k+ is great**. From match ids, Dota plays
~1.45 M matches/day; Axe vs Phantom Lancer appears in ~0.6% of games, so roughly
100k–250k Axe-vs-PL games per month exist. A STRATZ number of ~1k is probably filtered
(one rank, one position or a few days).

- [x] Rate-limited OpenDota client
- [x] Collectors: pro matches (explorer SQL) + public high-rank matches
- [x] Counter & synergy matrices with Bayesian smoothing (`predictor.counterpick`)
- [x] CLI: `recommend --enemies ... --allies ...` and `--counters <hero>`
- [x] Evaluation on a time-based holdout vs baselines
- [x] **Hosted in the Telegram bot**: `/pick`, `/counters`, `/hero`, plus admin `/collect`, `/train`, `/evaluate`
- [ ] **Rank brackets**: separate tables for Herald–Crusader, Archon–Legend, Ancient–Divine and
      Immortal. The meta and counters differ a lot by rank. Collect with `min_rank`/`max_rank`;
      the bot asks for or remembers each user's rank.
- [ ] **Personal hero pool**: link your Dota account ID. The bot pulls your per-hero games and
      win rate from OpenDota and only suggests heroes you can actually play (or marks them ⭐).
- [ ] **Role / position**: don't suggest a 4th carry. Tag heroes by position from public data.
- [ ] Each of us collects public matches with our own quota → ~200 k+ matches
- [ ] Sanity check: do the top counters agree with what we know as players?
- [ ] Share the dataset (GitHub Release attachment or Kaggle / Hugging Face dataset)

**Done when:** the recommender gives sensible counters, and the counter model beats the
per-hero-only baseline on the holdout.

**First results (2026-11-01).** Time-split holdout, newest 20% of matches as the test set.
Lower log loss is better.

| Data | Model | Accuracy | Log loss |
|---|---|---|---|
| 26 k public, Divine (1 day, patch 7.41) | always Radiant | 52.8% | 0.6916 |
| | per-hero logistic regression | 56.0% | 0.6816 |
| | counter model | **56.5%** | **0.6800** |
| 46 k pro (2025 → now, all patches) | always Radiant | 52.3% | 0.6921 |
| | per-hero logistic regression | 54.3% | 0.6880 |
| | counter model | 54.1% | **0.6876** |

Takeaways: the counter signal is real but small at this size. Most of the edge comes from
per-hero strength in the current meta. The pair tables need much more data: about 65 games
per hero pair now, and we want hundreds. Pro data is hard to predict from heroes alone, as
expected. Lesson learned: the first version fitted the weights on the same games as the
tables and overfitted badly (log loss 0.87). The weights now come from out-of-fold terms.

Known issues / next steps:
- `/publicMatches?min_rank=70` only returned avg rank tier ≤ 75 (no Immortal-average games).
  Check whether Immortal games need a different query, or go through the explorer.
- Grow public data to 200 k+ matches (all three of us collecting daily) and re-run `evaluate`.
- Role awareness: right now it can suggest a 4th carry.

### Phase 2 — Pro match predictor ✅ hosted
- **Pro matches only.** Inputs: both drafts, both teams' recent form/rating, patch.
  Players and teams matter more than heroes in pro play, so team strength is a key feature.
- Logistic regression → **LightGBM** → small neural model (hero + team embeddings) on the 3050 Ti
- Weight recent patches more. Metrics: log loss, Brier, calibration. Accuracy alone is misleading.
- [x] Data: 106,900 pro matches since 2023 with players, team ids, league tiers; team, league, player names
- [x] Chronological features (`pro/features.py`): team Elo, player Elo, experience, form,
      head-to-head; after the draft also hero meta, players' and team's hero comfort, pro counters
- [x] Two gradient-boosting models: before the draft and after the draft. Sides are
      averaged, so you don't need to know who's Radiant
- [x] Hosted: `/predict`, `/team`, `/top`, `/pro`, plus admin `/train pro`

**Results (2026-11-01).** Trained on matches before 2025-11-25, tested on 18,235 newer ones.

| Test set | Model | Accuracy | Log loss |
|---|---|---|---|
| all tiers | always Radiant | 52.3% | 0.6922 |
| | team Elo only | 56.4% | 0.6776 |
| | before draft | 60.3% | 0.6449 |
| | after draft | **61.0%** | **0.6415** |
| tier 1–2 (professional + premium, 6,879 matches) | before draft | 65.6% | 0.6160 |
| | after draft | **65.7%** | **0.6116** |

Calibrated: when it says 60–70% it's right 67% of the time, at 70–80% it's 77%, and at
80%+ it's 90% (that last group is 6% of matches).

Next ideas: tune the Elo settings, add patch-specific hero meta, model best-of series directly,
and handle roster changes better (stand-ins). Most of the gain is in team/player strength,
not heroes. That's expected in pro play.

### Phase 3 — Pro draft predictor (next pick / ban) ✅ hosted
- [x] Data: pick/ban order of 105,779 pro drafts since 2023 (`collect pro` fetches them too)
- [x] Valve changed the Captains Mode order twice since 2023: every draft uses its own order,
      and the bot uses the newest one
- [x] Features (`draft/features.py`): hero pick/ban rate in the pro meta, how often it's a
      team's n-th pick/ban, both teams' pick/ban habits, bans made against each team, players'
      hero comfort, heroes usually picked together with the ones already picked, matchups
      vs picked heroes
- [x] Model: gradient boosting scores every free hero at each step
- [x] Hosted: `/draft` (intro + draft order) and `/next team A vs team B: heroes so far`

**Results (2026-11-01)**, 2,497 newer drafts, every free hero ranked at all 24 steps:

| | top-1 | top-5 | top-10 |
|---|---|---|---|
| model, picks | 6.8% | **24.3%** | 38.4% |
| model, bans | 8.9% | **29.2%** | 43.9% |
| most popular hero in the meta, picks | 4.6% | 15.6% | 25.5% |
| most popular hero in the meta, bans | 3.8% | 16.0% | 26.9% |

(Random guessing would get ~4% top-5.) Pro drafts are hard to predict on purpose.

Next ideas: hero roles/positions, the team's last few drafts (strategies repeat within an
event), a "live draft" view that also shows win % from model 2 after each pick.

### Phase 4 — Live in-game prediction ✅ hosted
- [x] Data: per-minute gold/XP lead + towers, barracks, Roshan, first blood for 106,752 pro
      matches since 2023 (OpenDota explorer, ~22 calls; `collect pro` keeps it fresh)
- [x] Model (`live/`): one gradient-boosting model on [everything the pro model knows before
      the game] + [minute, gold lead, gold change over 5 min, towers, barracks, Roshan, first
      blood]. A snapshot every 2 minutes, with inputs randomly hidden in training so it still
      works without heroes or objective counts
- [x] Hosted: `/live` (pro games happening now, from OpenDota's live feed) and `/ingame` (type
      a scoreboard: minute, gold lead, towers, Roshan)

**Results (2026-11-01)**, 18,201 newer pro matches:

| minute | live model | gold lead only | pre-game model only |
|---|---|---|---|
| 0–4 | **62.3%** | 56.9% | 61.2% |
| 10–14 | **68.9%** | 67.9% | 61.1% |
| 20–24 | **77.1%** | 76.7% | 60.2% |
| 30–34 | **82.2%** | 81.5% | 58.5% |
| 40+ | **77.1%** | 75.1% | 55.0% |

Next ideas: kill score (in the live feed but needs kill timelines for training), decoding the
live feed's building state, a live line chart, first blood / first Roshan / first tower models,
and Dota's Game State Integration while spectating for a faster, richer feed.

### Phase 5 — Tournament predictor ✅ hosted
- [x] Monte Carlo of a TI-style event (`tournament/sim.py`): Swiss stage (Bo3, 3 wins through,
      3 losses out) → 8-team double elimination, grand final Bo5. Map chances from the pro
      match predictor, 10,000 simulations per question
- [x] Backtest (`tournament/backtest.py`): for each TI in the data, rebuild the pro model from
      matches before the event, simulate it with the real teams, compare with the real winner
- [x] Hosted: `/tournament` (intro + backtest) and `/odds` (top-16 teams today, or your list)

**Backtest (2026-11-01):**

| TI | real champion | our chance for them | rank | our favourite | random guess |
|---|---|---|---|---|---|
| 2023 | Team Spirit | **36%** | #1 of 20 | Team Spirit | 5% |
| 2024 | Team Liquid | **12%** | #3 of 16 | Team Falcons 24% | 6% |
| 2025 | Team Falcons | **29%** | #1 of 16 | Team Falcons | 6% |
| 2026 | Team Spirit | 6% | #7 of 16 | TEAM VISION 28% | 6% |

On average (geometric mean) the champions got ~16% vs ~6% for a random guess.
Next ideas: the exact format of each year, seeding from group results, betting odds as a
comparison, roster changes right before the event.

### Phase 5 — original notes
- **Team ratings**: Elo / Glicko-2 from pro results, with roster changes and decay
- Match model: team rating diff (+ draft model during the series)
- **Monte Carlo the TI format** (groups + double elimination playoffs), 100 k runs →
  P(each team wins TI)
- **Backtest on past TIs (2023, 2024, 2025, 2026)**: freeze data at the tournament start
  date and score our predictions
- Compare with betting odds as a reality check

### Extra — Talk to the bot in plain language (evaluated 2026-11-01: ✅ practical)
Goal: write "who wins spirit or liquid?" or "спирит против ликвид кто победит" and the
bot turns it into `/predict spirit vs liquid`. It always shows the command it understood
("🤖 I read this as: /predict spirit vs liquid") before the answer, so mistakes are obvious.

| Option | Verdict |
|---|---|
| **Free cloud model with tool calling** (Groq free plan: no credit card, ~1,000 requests/day on its open models; Gemini free tier: limits have changed often, sometimes only ~20–250/day) | ✅ **Recommended.** Handles English, Russian and Uzbek mixed together. Nothing to train. 4 users are far below the limits. Needs an API key in `.env` |
| Local model, no training (e.g. a 1.5–3B instruct model via Ollama, JSON output) | ⚠️ Fallback. Fits in 4 GB of GPU memory and works offline, but it's weak at Uzbek and slang, and it's a big download |
| Fine-tune a small local model (T5 / 0.5B with LoRA on synthetic sentence→command pairs) | ❌ Not worth it now. Lots of work (training data in three languages, hero/team aliases) and still worse than a cloud model |
| Keyword rules only | ⚠️ Cheap, but brittle. Good as the first try before calling the model |

Plan: rules first, then the cloud model, then show the translation, then run the command.
The model only *chooses a command and fills in its arguments*; the normal command code
checks everything. Needs: a Groq API key (console.groq.com, free) in `.env` as `groq_api`.

### Real-world validation ✅ (2026-11-01) → [VALIDATION.md](VALIDATION.md)
Tested end to end in Telegram, against Dotabuff and Dota2ProTips, and by replaying real
events with frozen models. Led to: transfer-aware rosters, a league-tier filter, atomic saves,
`/replay`, and a nightly refresh with a prediction journal (`refresh.py`, `/journal`).

### Phase 6 — Universal model (last)
- One model with shared hero/team embeddings and several output heads: counter-pick scores,
  draft win, next pick, live win, tournament.
- It only replaces a separate model in the bot if it matches or beats it on that model's
  evaluation.

---

## 4. Team & workflow (Shanghai / Tashkent / remote)

- **Async first.** Everything goes in GitHub Issues + a GitHub Projects board. One issue = one
  task with a clear "done when".
- **Branches + PRs**, at least one review from someone else. `main` must always run.
- **Weekly call** at a time that works across UTC+8 (Shanghai) and UTC+5 (Tashkent).
- **Data is not committed to git** (it's in `.gitignore`). Share datasets as GitHub Release
  assets or on Kaggle / Hugging Face.
- **Shanghai note:** GitHub and some APIs can be slow or blocked from mainland China without a
  VPN. Check that `git clone` and `api.opendota.com` work early, before it blocks work.

### Suggested split for phase 1
| Person | Owns |
|---|---|
| A | Data collection, dataset sharing, `collect.py` improvements (resume, more fields) |
| B | Counter model + recommender CLI, sanity-checking counters by hand |
| C | Evaluation, baselines, starting the phase 2 learned model |

Everyone runs `collect public` daily with their own quota.

---

## 5. Pitfalls to avoid

- **Leakage:** always split by **time** (train on older, test on newer), never randomly.
  Never use post-game stats (kills, GPM) as draft-model features.
- **Patch drift:** a counter from 7.38 may be wrong in 7.41. Filter or weight by patch.
- **Small-sample illusions:** a 9–1 matchup is noise. That's why counters use smoothing
  (`--prior`) and a minimum-games filter.
- **Turbo / ability draft / abandons** pollute public data. The collector keeps only normal
  modes and games ≥ 10 min.
- **Radiant advantage** exists (~52%). Every model gets a side intercept.
