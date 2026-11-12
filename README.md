# Autotrade: Lightweight IBKR news-sentiment + ML trading bot

A personal trading bot for US stocks: screens US stocks, ingests news and bars, scores text with
FinBERT, predicts with a Tri-Lens ML stack, then trades through **IB Gateway** — all driven from a
localhost web UI. PAPER by default; LIVE is double-locked.
It runs in **paper mode** by default: real market data, real orders, fake money.

## Requirements

| Requirement | Notes |
|---|---|
| Miniconda or Anaconda | Used only to create the interpreter; everything else is `pip` |
| Python 3.11 | `pyproject.toml` requires `>=3.11`. 3.11 is what this project is tested on |
| IB Gateway (or TWS) | Free download from IBKR. The **paper** build is enough to start |
| ~2 GB free disk | CPU PyTorch (~250 MB) + FinBERT weights (~420 MB, fetched on first run) + bar cache |
| Outbound network | yfinance, Google News RSS, Hugging Face (first run only) |
| API keys (optional) | Finnhub and Marketaux free tiers, plus an SEC user agent. Without them the bot still runs on RSS + yfinance |

Nothing else is compiled from source. `TA-Lib` and `shap` are optional — every indicator has a
pure-pandas fallback and explanations are skipped if `shap` is absent.

## Installation

### 1. Create the conda environment

```bash
conda create -n autotrade python=3.11 -y
conda activate autotrade
python -V                       # must print Python 3.11.x
```

Run every command below with this env active. If you use PowerShell or cmd, activate the same
way (`conda activate autotrade`) and swap `/` for `\` in paths.

### 2. Install CPU-only PyTorch first

The bot never uses a GPU, so install the small CPU build before anything can pull the
~2.5 GB CUDA wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
```

On Windows `pip install torch` already resolves to a CPU wheel; on Linux and macOS this step
is what keeps the install small.

### 3. Install the bot itself

```bash
cd autotrade
pip install -e ".[dev]"
```

The editable install (`-e`) matters: the code lives in `src/`, so without it
`python -m tradebot.main` fails with `No module named tradebot`. `[dev]` adds pytest, ruff and
the test helpers; drop it if you only want to run the bot.

### 4. Create `.env`

`.env` is git-ignored, so a fresh clone has none. Create it in the project root:

```dotenv
# ---------- IBKR ----------
IB_HOST=127.0.0.1
# 4002 = paper, 4001 = live
IB_PORT=4002
IB_CLIENT_ID=17
# blank = auto-detect the logged-in account
IB_ACCOUNT=
# 3 = delayed data, 1 = live subscription
IB_MARKET_DATA_TYPE=3
# off | dry_run | paper | live
TRADING_MODE=paper
ALLOW_LIVE=0

# ---------- News / data keys (all optional) ----------
FINNHUB_API_KEY=
MARKETAUX_API_KEY=
SEC_USER_AGENT="autotrade/1.0 (you@example.com)"

# ---------- App ----------
APP_HOST=127.0.0.1
APP_PORT=8000
# set a value only if you expose the app beyond localhost
APP_API_TOKEN=
```

An empty key simply disables that connector — the app starts either way, and
`python scripts/check_env.py` prints which sources are live.

### 5. Point IB Gateway at the app

In IB Gateway: **Configure → Settings → API → Settings** and

- enable **Enable ActiveX and Socket Clients**,
- set **Socket port** to `4002` for paper (`4001` for live),
- add `127.0.0.1` to **Trusted IPs**,
- leave **Read-Only API** unchecked, or orders will be rejected.

Then log in and restart Gateway. Two processes cannot share one `IB_CLIENT_ID`; a stale server
still holding `17` makes the new one fail with **error 326**.

### 6. Verify the install

```bash
pytest tests/unit -q            # offline; no broker needed
python scripts/smoke_ib.py      # broker connection, positions, account values
python scripts/check_env.py     # which API keys and news sources respond
```

`smoke_ib.py --orders` additionally places and cancels a live paper order.

### 7. Install from the exact lock (optional)

`requirements.lock.txt` is a full freeze of the environment this project runs in. Prefer
`pip install -e ".[dev]"` for a new machine; the lock is for reproducing a known-good set:

```bash
pip install -r requirements.lock.txt
```

Two lines in that file are machine-specific and must be removed first: the trailing
`-e c:\users\user\documents\autotrade` and the `packaging @ file:///home/conda/...` entry.
Re-add the project with `pip install -e .` afterwards.

### If the install fails

| Symptom | Cause and fix |
|---|---|
| `No module named tradebot` | The editable install was skipped. Run `pip install -e .` from the project root |
| pip starts downloading a ~2.5 GB `torch` | Step 2 was skipped, so PyPI's CUDA wheel won. Reinstall with the CPU index URL above |
| `conda: command not found` (Windows) | Use **Anaconda Prompt** instead of `cmd`, or run `conda init bash` once |
| `error 326 — client id 17 already in use` | An older `tradebot` process still holds the connection. Close it before restarting |
| Connects, but every order is rejected | **Read-Only API** is still ticked in Gateway's API settings |
| `TA-Lib` fails to build | Optional. Skip it; the pandas fallbacks produce the same indicators |
| `error 10197 — competing live session` | Another TWS/Gateway/mobile session holds the market-data subscription. See *Known limitation* below |

## Quick start

```bash
conda activate autotrade
python -m tradebot.main        # → http://127.0.0.1:8000
```

First launch creates the database, downloads the FinBERT weights into `models/hf_cache`
(~420 MB, once), connects to IB Gateway and starts the 15-minute scheduler. Then open the
dashboard and press **Run pipeline**.

```bash
python scripts/train.py        # train a model (~60–90 s)
```

`smoke_ib.py` and `check_env.py` from step 6 are the quickest way to confirm a fresh machine
is wired up correctly.

## What a trade looks like

Say the bot likes a stock at $100. It buys, then immediately attaches two exit orders:

| Order | Price | What happens |
|---|---|---|
| Buy | $100 | Entry |
| Stop-loss | $97 | Price falls 3 % → sell. Loss capped at $3 per share |
| Target | $104.50 | Price rises 4.5 % → sell. Profit taken |

The distance from entry to stop is **1R** — "one unit of risk". Hitting the target earns
**1.5R**; hitting the stop loses **1R**. The bot only takes trades where the target is at
least 1.2× further away than the stop.

That is the whole idea: many small decisions, losses kept small, wins allowed to run a
little further.

## How a run works

Every 15 minutes during market hours the bot runs one cycle:

| # | Step | Explanation |
|---|---|---|
| 1 | **Preflight** | Check the broker connection, the market clock and today's loss limits. Abort if anything is wrong. |
| 2 | **Universe** | Ask a screener for ~100 US companies big enough and liquid enough to get in and out of. |
| 3 | **Ingest** | Download recent 15-minute price bars and today's news headlines. |
| 4 | **Sentiment** | Read the headlines with a language model and score each company positive / neutral / negative. |
| 5 | **Predict** | Feed price patterns and news score into the model → each stock gets a **win probability**. |
| 6 | **Strategy** | Apply the rules below, rank the survivors, size the position, set stop and target. |
| 7 | **Expand** | If fewer than 5 stocks survived, fetch another batch and repeat steps 3–6. |
| 8 | **Execute** | Send the orders to the broker — or park them for your approval. |
| 9 | **Persist** | Save the result and refresh the dashboard. |

Each step has a time budget and reports itself in the UI, so one slow step cannot stall
the whole cycle.

## The rules a stock must pass

| Rule | Explanation | Setting |
|---|---|---|
| Price & size | Not a penny stock; trades at least $25 M a day so orders can fill | `universe.min_price`, `min_avg_dollar_volume` |
| Model rank | Must be in the **top 10 % of this run's candidates** — a relative cut, not a fixed score | `min_bar_pct`, learned at training time |
| Entry timing | Must not be at a bad moment inside the bar (the model also scores *when* to buy) | `min_entry_pct`, learned at training time |
| News | Must not be in a strongly negative news flow | `strategy.min_ewm_240m` |
| Portfolio | Not already held, a slot is free, max 2 names per sector | `max_positions`, `max_per_sector` |
| Edge | Expected return after costs must be positive — **enforced only in LIVE** | `strategy.edge_gate_mode` |

If fewer than 5 stocks survive, the bot relaxes the model-rank cut one step at a time
(top 10 % → 15 % → 25 % → 35 %) instead of returning an empty book — but it never buys
from the bottom 40 % of the ranking.

## How many shares?

Position size follows from the stop, not from a feeling:

| Input | Value |
|---|---|
| Risk per trade | 0.75 % of the account (`risk_per_trade_pct`) |
| Stop distance | ~1× the stock's normal 15-minute price swing (`risk.sl_atr`) |
| Max one position | 20 % of the account (`max_position_pct`) |
| Max one order | $25,000 (`max_order_notional`) — a fat-finger guard |

So `shares = risk budget ÷ stop distance`, then trimmed by the caps above. On a $250 k
account, a trade risking $1,875 with a 3 % stop would be ~$62 k — cut to $25 k by the
order cap.

## Protection and exits

- The moment the buy fills, the bot places a **stop-loss and a target** at the broker,
  priced off the *actual* fill — so the position is protected even if this app is closed.
- The two are linked (**OCA**): if one fills, the other is cancelled. A stop and a manual
  close can never both fire.
- Protection is cancelled **only after** an exit actually fills, never before.
- A **watchdog** checks every 60 seconds: a position with no live stop gets re-protected;
  if that keeps failing, it is closed.
- The stop **only ever moves up**, never down, as the price rises.

## Approvals

With `approval_mode: manual_approve`, every automatic entry and exit waits for you, with
quantity, limit, stop, target and R:R shown on the Orders page.

| Guard | Why it exists |
|---|---|
| Expires after 30 min | A decision taken at 10:05 is not valid at 11:00 |
| Refused if the price moved > 1.5 % | You would be approving a different trade than the one shown |
| Marketable limit re-priced to the live quote | A frozen limit never fills after a two-minute review |
| A newer run supersedes older pending orders | You should only ever see today's view |

## Orders while the market is closed

Outside market hours an order becomes a **bracket**: a buy limit plus its stop and target,
sent together and marked **GTC** ("good till cancelled") so it survives overnight and
fills at the open with protection already attached.

The queue is reconciled on **every** run, including runs that find nothing:

| Situation | What the bot does |
|---|---|
| The same pick is made again next run | Ignored — one live order per stock and direction |
| Duplicate brackets already at the broker | Keeps the newest, cancels the rest |
| A stock drops out of the decision | Cancels its bracket and closes its record |
| Size or limit changed | Resizes / re-prices the bracket in place |
| Queued value exceeds your cash | Cancels the lowest-ranked brackets until it fits |
| More queued than 5 positions | Cancels the oldest |
| A bracket fills only partially | Records what filled, resizes stop and target to match |

Cash is counted the way the broker counts it: money already reserved by working orders is
not available twice, and a queued order occupies one of the 5 slots. The Orders page shows
`queued $X of $Y cash (Z%)` so you can see the headroom.

## Safety nets

| Control | What it does |
|---|---|
| Modes | `off` · `dry_run` (record only) · `paper` (fake money, port 4002) · `live` (real money) |
| LIVE locks | Needs `ALLOW_LIVE=1`, port 4001 and a confirmation prompt — three independent steps |
| Kill switch | Cancels entries, closes positions, pauses the scheduler |
| Daily loss limit | Stops new entries after −2 % in a day |
| Drawdown limit | Stops after −8 % from the equity peak |
| Trade cap | 30 orders per day (5 entries + 5 exits + rotation room) |
| Local only | Binds `127.0.0.1`, checks Host/Origin, requires a CSRF token on every change |

## Configuration

`config/default.yaml` holds the defaults; edits made in the UI are stored separately,
audited, and never touch the file. All 130 settings are type-checked, and the Settings page
is generated from that schema.

A banner appears if two settings that must agree drift apart — for example a target/stop
ratio below 1.2, or a live stop that no longer matches the one the model was trained on.

## Testing

```bash
pytest tests/unit -q              # 58 tests
python scripts/smoke_ib.py [--orders]
```

These check the things that are easy to get silently wrong: features only use past data,
labels never peek ahead, stops always bracket the price, the position cap holds, stops
never move down, and a sell can never open a short by accident.

## Known limitation

IB Gateway reports **error 10197 — "no market data during competing live session"** on
this machine, so broker quotes and bars are unavailable and the bot uses yfinance instead
(~15 minutes delayed). Orders, positions, account values and protective orders all work
normally.

To restore broker data: log out of TWS, the mobile app and any other Gateway session on
this account, then restart IB Gateway. The dashboard banner clears on the next check.

## Layout

```
autotrade/
├── config/
│   └── default.yaml          defaults (UI edits are stored in the DB, not here)
├── data/                     runtime state
│   ├── tradebot.db           SQLite: runs, orders, positions, trades, settings
│   ├── bars/                 cached 15-minute bars (parquet)
│   ├── backtests/            saved backtest results
│   ├── logs/                 server and run logs
│   └── shots/                UI screenshots
├── models/                   trained artefacts (not code)
│   ├── registry/             model bundles: weights + metrics
│   └── hf_cache/             downloaded FinBERT weights
├── scripts/
│   ├── train.py              train and register a model
│   ├── smoke_ib.py           broker connection and order probes
│   └── check_env.py          which API keys are configured
├── src/tradebot/
│   ├── main.py               entry point
│   ├── app.py                wiring: IB link, scheduler, position monitor
│   ├── settings.py           every config field, with validation
│   ├── core/                 clock, pipeline, scheduler, events, logging, types
│   ├── db/                   SQLAlchemy models and session
│   ├── ingest/
│   │   ├── universe/         screener and scoring
│   │   ├── market/           bars, quotes, indicators, parquet store
│   │   └── news/             RSS and API connectors, dedupe, relevance
│   ├── sentiment/            FinBERT scoring and aggregation
│   ├── features/             feature builder and triple-barrier labels
│   ├── models/               lenses, stacker, cross-validation, prediction
│   ├── strategy/             decide(): gates, ranking, exits, sizing
│   ├── risk/                 position sizing and circuit breakers
│   ├── execution/            IB client, order manager, account, reconcile
│   ├── backtest/             event-driven backtest engine
│   └── web/
│       ├── server.py         FastAPI routes and the SSE stream
│       ├── templates/        Jinja pages (dashboard, orders, models, …)
│       └── static/           css, js, vendor assets
└── tests/
    └── unit/                 58 tests
```

All times are stored in UTC and shown in **Hong Kong time**.
Personal research tool, not investment advice.

## Glossary

| Term | Plain meaning |
|---|---|
| **p(win)** | The model's estimated probability that a trade hits its target before its stop. 0.30 = roughly a 30 % chance |
| **Percentile** | Where a stock ranks among the others in the same run. "Top 10 %" = better than 90 % of them |
| **ATR** | Average price swing over a period. Used to set stops that match how much the stock actually moves |
| **R / R:R** | One unit of risk (entry − stop). R:R 1.5 means the target is 1.5× further than the stop |
| **Stop-loss** | Standing order that sells if the price falls to a set level |
| **Target** | Standing order that sells if the price rises to a set level |
| **OCA** | One-Cancels-All: two orders linked so filling one cancels the other |
| **GTC** | Good-Till-Cancelled: the order survives past today instead of expiring at the close |
| **bps** | Basis points. 1 bps = 0.01 %, so 50 bps = 0.5 % |
| **Slippage** | The gap between the price you expected and the price you got |
| **Kelly** | A formula for how much to bet given your edge and odds. Negative Kelly = the bet is unfavourable |
| **OOF** | Out-of-fold: predictions made on data the model was not trained on, so the score stays honest |
| **Paper mode** | Everything runs for real except the money |
