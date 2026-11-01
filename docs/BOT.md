# Telegram bot

One bot hosts every model. It runs on the host's laptop. Everyone on the allowed list uses it
from Telegram, wherever they are.

```
 friends' phones ──► Telegram servers ◄── bot on the host laptop (polls for messages)
                                              │
                                              ├─ counter-picker model
                                              ├─ (next models as we build them)
                                              └─ data/ and models/ folders
```

The laptop connects *out* to Telegram, so there's no port forwarding, no public IP and no
router setup. The bot works whenever the laptop is on, online and running it.

## Using it

Send `/start` or `/help` to the bot for the full list with examples. The same list shows up
in Telegram's `/` menu.

| Command | What it does |
|---|---|
| `/start` | Pick a model. `/counterpick`, `/pro`, `/draft` and `/admin` each explain their own commands |
| `/pick pa, pudge, lion` | Best heroes to pick against those enemies |
| `/pick pa, pudge \| cm, axe \| invoker` | Same, plus your team (after the 1st `\|`) and bans (after the 2nd) |
| `/counters phantom assassin` | Heroes that beat this hero |
| `/hero cm` | A hero's win rate, best and worst matchups, best allies |
| `/predict spirit vs liquid` | Pro match: chance to win a map, Bo3 and Bo5 |
| `/predict spirit: pa, cm, axe, lion, invoker vs liquid: am, lina, tide, rubick, sven` | The same after the draft |
| `/team falcons` | Pro team rating, recent maps, roster, favourite heroes |
| `/top` | Strongest active pro teams |
| `/next spirit vs liquid: axe, pudge, lion` | Pro draft: most likely next pick/ban. First team = the one that acts first; heroes = everything picked/banned so far, in order |
| `/status` | Which models are loaded, how much data, running jobs |
| `/collect public 300` | Admin: download ~26 k public games (300 API calls ≈ 5 min) |
| `/collect pro` | Admin: download new pro matches (+ team, league and player names) |
| `/train counterpick` | Admin: retrain the counter-picker on all collected data |
| `/train pro` | Admin: retrain the pro predictor (~2 min) |
| `/train draft` | Admin: retrain the draft predictor (~4 min) |
| `/evaluate counterpick` | Admin: accuracy on the newest games vs simple baselines |

Hero names can be full (`Phantom Assassin`), partial (`phantom a`) or initials (`pa`).

**A good daily routine:** `/collect pro` then `/train pro` and `/train draft` (a few minutes each), and
`/collect public 2500` (about 45 min) then `/train counterpick`.
The model reloads by itself after training, so there's no need to restart the bot.

## Who can use it

Only the Telegram usernames in `.env` → `usernames`. The first time each person messages the
bot, it remembers their Telegram ID in `data/bot_users.json`. After that, a stranger who takes
an old username still can't get in. Anyone else gets "Sorry, this is a private bot".

Everyone on the list can run admin commands. They only start predefined jobs (collect, train,
evaluate). The bot can't run arbitrary commands on the laptop.

People without a Telegram username can't be allowed. They need to set one in Telegram
settings first.

## Hosting it (host laptop only)

Only **one** computer may run the bot at a time. Two copies using the same token fight over
messages (you'll see `Conflict` errors).

### 1. One-time setup

```bash
git clone https://github.com/MukhtorovMurodbek/Predictor.git
cd Predictor
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Create `.env` in the repo folder (it's in `.gitignore`, so it never gets uploaded):

```
bot_api = 123456:ABC...        token from @BotFather
bot_username = YourBotName
usernames = me, friend1, friend2
```

Get some data and a model, either from the command line or later through the bot:

```bash
.venv\Scripts\python -m predictor.data.collect heroes
.venv\Scripts\python -m predictor.data.collect public --calls 300
.venv\Scripts\python -m predictor.counterpick.train
.venv\Scripts\python -m predictor.data.collect pro
.venv\Scripts\python -m predictor.pro.train
```

### 2. Start it

Double-click **`scripts\run_bot.bat`**, or run:

```bash
.venv\Scripts\python -m predictor.bot
```

Keep the window open. You'll see `@YourBotName is running...`. The `.bat` restarts the bot
automatically if it crashes or the internet drops.

### 3. Keep the laptop awake
Windows Settings → System → Power → *Screen and sleep* → "When plugged in, put my device to
sleep after" → **Never**. The screen can still turn off.

### 4. (Optional) Start with Windows
Press Win+R, type `shell:startup` and press Enter. Put a shortcut to `scripts\run_bot.bat`
in that folder.

### Updating to new code

Close the bot window, then:

```bash
git pull
.venv\Scripts\python -m pip install -e ".[dev]"
```

Then start it again.

## Troubleshooting

| Problem | Fix |
|---|---|
| `.env is missing: bot_api` | `.env` must be in the repo root next to `README.md` |
| `Conflict: terminated by other getUpdates request` | The bot is running somewhere else too. Close the other copy |
| Bot says "private bot" to a friend | Check their username in `.env` (no @). Restart the bot after editing `.env` |
| A friend changed their Telegram username | Still works, the bot knows their ID |
| You *replaced* a person in `usernames` | Delete their line from `data/bot_users.json`, or delete the file |
| "No counter-pick model yet" | `/collect public` then `/train counterpick` |
| Bot is slow to answer while a job runs | Jobs run in the background. Only one job runs at a time |
