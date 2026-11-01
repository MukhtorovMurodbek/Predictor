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
| **Patches**: `/api/constants/patch` | Patch dates | Meta shifts every patch: weight recent patches. |

**Patch counts in the pro DB:** 7.39 ≈ 17.9 k, 7.40 ≈ 7.9 k, 7.41 ≈ 8.7 k matches.
A single patch of pro games is **too small for 127×127 matchup tables** (~25 games per hero
pair). So: **learn counters from high-rank public games (volume), then adjust for pros.**

---

## 3. Roadmap

Each phase ends with something that works and that we can demo. Do them in order. Phase 1 is
the weekend.

### Phase 1 — Counter-picker MVP (this weekend) ⬅ we are here
- [x] Rate-limited OpenDota client
- [x] Collectors: pro matches (explorer SQL) + public high-rank matches
- [x] Counter & synergy matrices with Bayesian smoothing (`predictor.counters`)
- [x] CLI: `recommend --enemies ... --allies ...` and `--counters <hero>`
- [x] Evaluation on a time-based holdout vs baselines (`predictor.evaluate`)
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

### Phase 2 — Learned draft model (week 2–3)
- Logistic regression on hero one-hots + pairwise features → then **LightGBM**
- Neural model: hero embeddings (dim ~32) → team pooling or a tiny transformer over 10 heroes
  → P(radiant win). PyTorch, runs on the 3050 Ti.
- Train on public, **fine-tune on pro**, add patch as a feature / sample weight by recency
- Recommender switches to "ΔP when adding hero h" (role-aware: don't suggest a 5th carry)
- Metrics: log loss, Brier, calibration plot. Accuracy alone is misleading.

### Phase 3 — Pro draft assistant (week 3–4)
- Data: pro `picks_bans` **with order** (explorer `picks_bans` table)
- **Next-pick / next-ban prediction**: "what will Team X pick now?" Use the current draft
  state + the team's recent hero pool + player signature heroes. Top-5 accuracy is the
  metric.
- Live draft view: after each pick, show win% and the likely next picks

### Phase 4 — Live in-game prediction (month 2)
- Dataset: `/api/matches/{id}` for pro matches → per-minute gold/XP diff, kills, towers, Roshan
- Models:
  - **Win probability by minute** (draft prior + gold/XP/kills/buildings) → live line chart
  - **First blood team** (draft + team aggression history), **first Roshan**, **first tower**
  - Game-length prediction
- Real-time feed: **GSI while spectating** (best), fall back to OpenDota `/live`
- Small local web UI (Streamlit or FastAPI + simple page) showing the live win% line

### Phase 5 — Tournament predictor (before the next TI)
- **Team ratings**: Elo / Glicko-2 from pro results, with roster changes and decay
- Match model: team rating diff (+ draft model during the series)
- **Monte Carlo the TI format** (groups + double elimination playoffs), 100 k runs →
  P(each team wins TI)
- **Backtest on past TIs (2023, 2024, 2025, 2026)**: freeze data at the tournament start
  date and score our predictions
- Compare with betting odds as a reality check

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
