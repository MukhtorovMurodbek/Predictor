# Counter-picker website

The website explores observed matchups in locally collected OpenDota public games. It counts wins and games for hero pairs and ranks available picks from those statistics; it does not train a machine-learning model or promise the probability that a full draft will win. Matchup observations can overlap: a single game can contribute to more than one opponent pairing.

## Run it locally

Use Python 3.11 or newer and install the project from the repository root:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install -e ".[dev]"
python -m predictor.data.collect heroes
python -m predictor.counterpick.update
python -m web.server
```

`predictor.counterpick.update` counts about a month of games per run (run it again to go back
further, up to 90 days); on the host laptop the bot does this every hour by itself.

Open the local address printed by the server. The website needs the server and the daily tables to be present; it is not yet a hosted service. The update calls OpenDota and is subject to its current quotas and data availability.

The daily tables and hero/patch metadata live under the local `data/` directory, which is not committed to Git. A fresh clone therefore has no dataset; run the update before expecting results. The server notices new tables by itself.

## Reading the results

- The source is **every ranked public game OpenDota lists** in normal modes that lasted 10+ minutes (about 550,000 a day), from the **last 30 days**, counted by OpenDota's database. That is most, but not all, Dota 2 games.
- Match rank is the reported **average rank of the game**, not each player's own rank. Rank filters work in whole medals (e.g. `rank_min=40&rank_max=45` for Archon). OpenDota never reports an Immortal average (tier 80), so Immortal has no data; top games show up as Divine.
- Patch labels are inferred from match start times and OpenDota's patch-date metadata. Their precision and freshness depend on the available patch dates; they may not identify every hotfix.
- Percentages describe the sampled individual hero-versus-hero outcomes, not a full-draft win probability. Small matchup samples, incomplete opponent coverage, hero selection, and changing patches can make results misleading.
- Selected allies and banned heroes can be excluded from available picks, but role/position and allied synergy are not modeled by the website yet.
- The website currently uses OpenDota data. STRATZ is a possible future source after its filters and terms are verified. Dotabuff is not integrated; do not scrape it under this repository's current rules.

## Relationship to the bot

The Telegram bot hosts this counter-picker too: `/asil` explains it and `/apick` ranks heroes
with the same rules as `rankCandidates` in `web/static/engine.js` (ported to Python in
`src/predictor/counterpick/engine.py`), including the rank and patch filters. If you change
the ranking rules here, change `engine.py` as well. The bot's other counter-picker,
`/counterpick`, uses 90-day daily tables and Dotabuff-style edges.

Data: both read the daily tables in `data/counterpick/` (one file per UTC day: hero-vs-hero
games and wins per patch and medal), which the bot fills every hour from OpenDota's database
(`counterpick/update.py`). This website and `/apick` sum the **last 30 days** (`/apick ...
days 90` for longer); `engine.window_counts()` gives exactly the numbers `build_snapshot()`
would give on the same games (`tests/test_counterpick_engine.py`). The newest 3–7 hours are
not in yet, because OpenDota only lists a game once it has ended.
