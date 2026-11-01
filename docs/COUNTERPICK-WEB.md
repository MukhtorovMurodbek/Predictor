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
python -m predictor.data.collect public --calls 500 --min-rank 10
python -m web.server
```

Open the local address printed by the server. The website needs the server and collected data to be present; it is not yet a hosted service. The collector calls OpenDota and is subject to its current quotas and data availability. More API calls can take time and do not guarantee representative coverage.

Downloaded matches and hero/patch metadata live under the local `data/` directory, which is not committed to Git. A fresh clone therefore has no dataset; collect or obtain the data separately before expecting results. If data changes, restart the server to make sure the latest snapshot is loaded.

## Reading the results

- The source is a **sample of public games**, not the complete Dota 2 match population. Older data collected with a particular minimum rank may not cover lower ranks.
- Match rank is the reported **average rank of the game**, not each player's own rank. Missing rank values cannot support a rank-specific result. OpenDota may not return enough Immortal-average games, so some ranks can have little or no data.
- Patch labels are inferred from match start times and OpenDota's patch-date metadata. Their precision and freshness depend on the available patch dates; they may not identify every hotfix.
- Percentages describe the sampled individual hero-versus-hero outcomes, not a full-draft win probability. Small matchup samples, incomplete opponent coverage, hero selection, and changing patches can make results misleading.
- Selected allies and banned heroes can be excluded from available picks, but role/position and allied synergy are not modeled by the website yet.
- The website currently uses OpenDota data. STRATZ is a possible future source after its filters and terms are verified. Dotabuff is not integrated; do not scrape it under this repository's current rules.

## Relationship to the bot

The existing Telegram bot continues to use its legacy trained counter-picker. The website's statistical approach is separate for now; changing it does not automatically update the bot. The professional-match predictor is separate as well.
