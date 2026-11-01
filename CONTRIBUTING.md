# Working together

We're in different cities and time zones, so the defaults are: **write things down, work
async, keep `main` working.**

## Flow

1. Pick or create an **Issue**. Say what "done" looks like. Assign yourself.
2. Branch from `main`: `git switch -c feat/short-name` (or `fix/...`, `data/...`).
3. Commit small, push often. Open a **draft PR** early so others can see where you are.
4. Before merging: `pytest` passes, and one teammate approves.
5. Squash-merge into `main`.

## Rules of thumb

- **Never commit data or models.** `data/` and `models/` are gitignored. Share datasets as
  GitHub Release assets (or Kaggle / Hugging Face).
- **Don't commit API keys.** Use the `OPENDOTA_API_KEY` env var if you get one.
- **Evaluate with a time split.** Train on older matches, test on newer ones.
- New model? Add its numbers to the PR description (log loss, Brier, accuracy vs baseline).

## Daily data run (everyone)

Each of us has our own OpenDota quota (3,000 calls/day per IP):

```bash
python -m predictor.collect public --calls 2500 --min-rank 70
```

About 45 minutes. Then upload `data/raw/public_matches.parquet`, or merge it with the
others' files. Duplicates are removed by `match_id`.
