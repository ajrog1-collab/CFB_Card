# The Card — CFB model site

A phone-friendly site that shows this week's qualified wagers and the running record
of every one it has ever flagged. Updates itself twice a week. Costs nothing.

You do not need to write or run any code. Setup is about ten minutes of clicking.

---

## Setup

### 1. Make a GitHub account
[github.com/signup](https://github.com/signup) — free.

### 2. Create the repository
- Click **+** (top right) → **New repository**
- Name it whatever you like, e.g. `cfb-card`
- Set it to **Public** (required for free GitHub Pages)
- Do **not** add a README — this folder has one
- Click **Create repository**

### 3. Upload these files
On the empty repository page, click **uploading an existing file**, then drag in
*everything* from this folder — including the `cfb`, `docs`, `data`, and
`.github` folders. Drag the folders themselves, not just the files inside them.

Click **Commit changes**.

> If GitHub silently drops the `.github` folder (browsers sometimes hide dot-folders),
> create it manually: **Add file → Create new file**, type
> `.github/workflows/update.yml` as the name, and paste in the contents of that file.

### 4. Add your API key
- **Settings** → **Secrets and variables** → **Actions**
- **New repository secret**
- Name: `CFBD_API_KEY`
- Secret: your key from [collegefootballdata.com/key](https://collegefootballdata.com/key)
- **Add secret**

The key is stored encrypted. It never appears on the site or in the code.

### 5. Turn on the website
- **Settings** → **Pages**
- Source: **Deploy from a branch**
- Branch: **main**, folder: **/docs**
- **Save**

Your URL will be `https://YOURNAME.github.io/cfb-card/`. It takes a minute or two
to go live the first time.

### 6. Run it once
- **Actions** tab → **Update model** (left sidebar) → **Run workflow** → **Run workflow**
- First run takes 5–10 minutes; it downloads ten seasons of history
- Green check = worked. Red X = click it and read the log; the error messages say
  what went wrong in plain language

### 7. Put it on your phone
Open the URL in Safari or Chrome → Share → **Add to Home Screen**. It opens
full-screen like an app.

---

## How it runs after that

| When | What happens |
|---|---|
| Wednesday 15:00 UTC | Pulls new lines, posts this week's card |
| Saturday 13:00 UTC | Refreshes before kickoff |
| Sunday 12:00 UTC | Grades the weekend's results |

Nothing to maintain. History is cached in the repo, so each run only makes a
handful of API calls — comfortably inside the free tier.

To run it early any time: **Actions** → **Update model** → **Run workflow**.

---

## Reading the site

**Bets** — Qualified wagers are games where the model disagrees with the market by
at least the threshold in `config.json` (3 points by default). The thickness of the
brass rail on the left scales with the size of the disagreement. Smaller leans are
listed below for context and are not meant to be bet.

**Record** — Live results come first: every qualified wager, graded at the line
recorded when it first qualified. Backtest numbers are shown separately below and
labeled as such. They are not the same thing and shouldn't be averaged together.

**Avg CLV** is the number that matters most, and it matters sooner than win rate.
It measures whether the market moved toward your number after you logged it.
Positive CLV over 30+ wagers is meaningful evidence of an edge. A good win rate
over 30 wagers is not evidence of anything.

**Ratings** — Points better than an average FBS team on a neutral field.

---

## Changing things

Edit `config.json` in GitHub (pencil icon), commit, and the next run picks it up.

| Setting | Does what |
|---|---|
| `min_edge` | Points of disagreement needed to qualify. Raise for fewer, stronger picks |
| `lookahead_days` | How far ahead to show games |
| `blend_k` | Games played before in-season results outweigh the preseason prior |
| `assumed_price` | Odds used for units and ROI. `-110` is standard |
| `current_season` | Bump each August |

---

## Known limits

- **No transfer portal data.** Incoming transfers aren't credited anywhere, so teams
  that rebuilt through the portal will be misrated, worst in September.
- **No injury or QB availability data.** This is the single largest gap between the
  model and the closing line, and it isn't available for free.
- **Consensus lines, not your book's.** Numbers come from the CFBD consensus feed.
  What you can actually bet will differ, sometimes by half a point that matters.
- **Non-FBS games are excluded** from picks. The pooled rating for lower-division
  opponents is too crude to bet against.

The backtest exists to set expectations, not to justify wagering. Expect the closing
line to be sharper than the model. That's the normal result for a public-data model,
and the site says so on the front page rather than hiding it.
