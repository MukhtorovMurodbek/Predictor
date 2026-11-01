# STRATZ connection test

The read-only connection test checks a STRATZ token and attempts to inspect the
current GraphQL matchup schema. It does not collect a dataset or integrate STRATZ
into the website yet. Rank, patch, matchup direction, and returned match counts
must be verified against real responses before they are shown as recommendations.

## Configure privately

Sign in at <https://stratz.com/api> and follow STRATZ's instructions for API
access. Do not paste the token into chat or commit it to Git. Put it on the
server in the existing ignored `.env` file in the repository root, on its own
line:

```dotenv
STRATZ_TOKEN=your_token_here
```

Alternatively set the `STRATZ_TOKEN` environment variable. If the `.env` file
already has bot settings, keep them unchanged and add only the new line.
The environment variable takes precedence. Use a secret manager when deploying
the application; browser JavaScript must never receive this credential.

## Test

Install the project dependencies as described in the repository README, then run:

```bash
python -m predictor.counterpick.stratz
```

The diagnostic prints a connection result and, when allowed by STRATZ, the names
of the matchup query's schema arguments. To inspect a single matchup's raw
provider fields, use hero IDs (2 is Axe, 12 is Phantom Lancer):

```bash
python -m predictor.counterpick.stratz --hero-id 2 --opponent-id 12
python -m predictor.counterpick.stratz --hero-id 2 --opponent-id 12 --bracket HERALD_GUARDIAN
python -m predictor.counterpick.stratz --hero-id 2 --opponent-id 12 --week-start 1793232000
```

The diagnostic reports a requested bracket separately from the returned
bracket label and does not print the token. Live tests returned `UNCALIBRATED`
even when a particular bracket was requested. Opposite query directions also
returned different sample sizes. Three tested Unix-second week starts returned
distinct week IDs, but coverage, overlap and exact window semantics remain
unverified. `matchUp` exposes no explicit patch argument. Do not
use them as rank- or patch-specific recommendations until clarified. Stop and
respect the provider's limit if it returns HTTP 429. API availability, token
eligibility, permitted uses, attribution and rate limits should be confirmed
with STRATZ.
