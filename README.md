# MAGICWAND

**One wish. Zero lies.**

A paper-trading bot that refuses to flatter itself. It tests a strategy
honestly, tries to disprove it, and keeps a tamper-evident ledger of every
claim it makes — then grades those claims against what actually happened.

It cannot trade real money. There is no live execution path in this
repository, and there will not be one until a paper strategy survives the
reality check.

```
  __  __    _    ____ ___ ____        ___        __
 |  \/  |  / \  / ___|_ _/ ___|      / \ \      / /
 | |\/| | / _ \| |  _ | | |         / _ \ \ /\ / /
 | |  | |/ ___ \ |_| || | |___     / ___ \ V  V /
 |_|  |_/_/   \_\____|___\____|   /_/   \_\_/\_/
```

---

## Why this exists

You have seen the videos: a model is "handed the keys" to an exchange
account, and $40 becomes $11,000 in a week. The AI did not predict
anything. The videos are almost always one of:

1. **Leverage** — crypto perpetual futures at 20–100x, where the same move
   that turns $40 into thousands liquidates a hundred other accounts.
2. **A lottery ticket** — a memecoin that 20x'd, screenshotted before it died.
3. **Demo money** — a paper account presented as live.
4. **A funnel** — the video exists to sell you the course, the signals, or
   the referral link.

The AI's real contribution in those videos was *writing the script*. That
part is genuinely useful, and it is what this project does properly.

## What it does

| Command | What happens |
|---|---|
| `python3 wand.py data` | Pull real closed candles (Coinbase, no key needed) |
| `python3 wand.py backtest` | Run a strategy over history, fills delayed, fees charged |
| `python3 wand.py reality` | Try to destroy the strategy: 2x costs, extra lag, shuffled data |
| `python3 wand.py step` | One tick of the paper bot |
| `python3 wand.py loop` | Run it every N seconds against live prices |
| `python3 wand.py status` | Equity, open position, and the wand's own scorecard |
| `python3 wand.py verify` | Recompute the ledger chain to detect tampering |

## The four mechanisms that make it honest

**1. No lookahead.** A decision made on bar *i* fills at the **open of bar
*i + 1***. Filling at the close of the bar that produced the signal is the
most common way backtests lie; this engine physically cannot do it.

**2. Pessimistic fills.** 0.60% fee *per side* (Coinbase's small-account
taker rate) and 0.10% slippage *per side*, charged on entry and exit. When a
bar contains both the stop and the target, the **stop wins**. Costs are not
a rounding error at this size — they are usually the whole story.

**3. A risk engine the strategy cannot argue with.** Risk per trade, maximum
position size, exchange minimum, max open positions, daily loss limit, and a
kill switch. Every refusal is recorded with its reason. The daily-loss halt
blocks *new* risk and never traps you in an open position.

**4. A hash-chained ledger, and a scorecard.** Every claim the wand makes
("this trade should move +8%") is written to `data/ledger.jsonl` alongside
its reason, chained to the previous entry by SHA-256. When the claim
resolves, it is graded: right, wrong, overstated, or flat. Editing history
breaks the chain and `wand verify` says so.

That last part is the "magic wand" idea taken seriously: a wand that grants
wishes but cannot lie about what it granted.

## Try it

```bash
./start.sh                                 # ONE COMMAND: console + browser
python3 -m unittest discover -s tests -v   # 179 tests, no dependencies
python3 wand.py data                        # fetch real SOL/USD hourly bars
python3 wand.py reality                     # the honest verdict
python3 wand.py step                        # one paper tick, writes the ledger
python3 wand.py status                      # scorecard
```

Optional: `pip install -e .` then use `wand` instead of `python3 wand.py`.

Zero dependencies for trading. Rendering charts needs Pillow (`pip install pillow`);
everything else is standard library. Python 3.10+.

## The research layer: four agents, one scoreboard

`wand research` runs the DeepSeek research agents against live market data.

```
                    +-------------------+
   public data ---> | MARKET SNAPSHOT   |  derivatives · news · social · onchain
   (no keys)        +---------+---------+
                              |
        +---------+-----------+-----------+---------+
        |         |                       |         |
   derivatives   news                  social    onchain
        |         |                       |         |
        +---------+----------+------------+---------+
                            |
                    +-------v--------+
                    |  SYNTHESIZER   |  evidence score, veto decision
                    +-------+--------+
                            |
              +-------------+-------------+
              |                           |
      main strategy                  AGENT DESK
      (may be vetoed)             (sandboxed, earn-to-trade)
```

**What the agents can and cannot do.** A reader may `support`, `veto`, or stay
`neutral`. Only `veto` changes behaviour: it can block the strategy's trade.
No agent can open a position, change a side, or touch the main account. `support`
is recorded and graded, never acted on — that is the line that keeps a
confident narrator out of the money path.

**The Agent Desk** is where agents *can* act autonomously, and it is a
separate pot funded once at 10% of total equity. Inside it, walls that are
arithmetic rather than advice:

| Wall | Default |
|---|---|
| Pot | 10% of total equity, funded once, never auto-topped-up |
| Per trade | ≤10% of the desk pot |
| Trades per day | 1 |
| Confidence bar | 0.80 ("when the agent is sure") |
| Symbol | whitelist: BTC/ETH/SOL-USD — a hallucinated ticker is a refused order |
| Stop | mandatory on every entry; no stop, no trade |
| Autonomy | reviewed after 10 graded trades; ≤45% hit rate revokes it automatically |
| Kill switch | the operator's switch stops the desk too |

Every refusal is written to the ledger with its reason. Nothing is silently
dropped.

**Agents earn influence and lose it.** Each verdict is stored as a dated claim
with a horizon, graded against the price when the horizon expires, and tallied
per agent. Influence on the synthesizer scales with that hit rate: an agent
with no record gets a cautious 0.5 weight, a 80% hit rate gets ~1.0, a 20%
hit rate gets ~0.05. The scoreboard is the whole point — without it this is
just an opinionated newsletter.

### Run it

```bash
# with your own key (BYOK — never stored by this project)
export DEEPSEEK_API_KEY=sk-...
python3 wand.py research                       # live sources, DeepSeek agents
python3 wand.py research --desk                # plus the sandbox may act
python3 wand.py desk                            # desk state + autonomy status

# no key needed: deterministic fake model, still exercises the whole pipeline
python3 wand.py research --offline --offline-snapshot
```

### What each source actually gives you

| Source | Real | Notes |
|---|---|---|
| Derivatives | Binance public API | funding, retail + top-trader long/short, OI change, taker buy/sell |
| News | Cointelegraph, Decrypt, Bitcoin Magazine, CoinDesk RSS | 24h window, deduped, timestamped |
| Social | Reddit Atom feed | JSON API is blocked; the feed works with a bot UA. **It rate-limits hard.** |
| On-chain | blockchain.info, DefiLlama, CoinGecko | mempool, stablecoin supply, market-cap flows — **proxies, and labelled as such** |

A source that fails is marked `DEGRADED` and named in the prompt. Absence of
data lowers confidence; it is never quietly treated as a neutral signal.

Known limits, stated plainly:
- True exchange netflow needs a keyed provider (Whale Alert/Glassnode). The
  current on-chain reader sees proxies, and its prompt says so.
- Reddit will 429 this IP for a while after heavy use. The agent reports
  `DEGRADED` and the cycle continues. Don't poll it on a short timer.
- Agent verdicts are graded on price direction over the stated horizon. That is
  a coarse measure of a nuanced claim, and coarse honesty beats fine narrative.

## What `./start.sh` actually starts

**One process.** Not a stack of services. There is no database, no queue, no
worker, no broker bridge, no separate agent server — nothing to orchestrate.

```
./start.sh
   └── python3 wand.py serve            ← the only process
         ├── HTTP server        127.0.0.1:8787   (console + JSON API)
         ├── autopilot thread  (optional clock, same process)
         ├── research agents   (plain HTTPS calls, same process)
         ├── strategy gate     (backtests in-process)
         ├── risk engine + ledger (in-process, files on disk)
         └── chart renderer    (PIL, in-process)
```

Startup does exactly this, in order: preflight → prune old research snapshots →
one HTTP listener → optionally one autopilot thread → tell the browser where to
look.

What it does **not** start, and why:

| Not started | Why |
|---|---|
| database | state is JSON/JSONL files in `data/` |
| redis / queue / worker | all work is a thread inside the one process |
| broker / MT5 / exchange bridge | no live trading exists in this build |
| LLM server | DeepSeek is an HTTPS API, called only when you have a key |
| market-data daemon | fetched on demand from public endpoints |

The only things it can spawn are (a) a detached copy of *itself* when you use
`--bg` or `wand service start`, and (b) a headless browser for page captures —
which is unused, since charts are rendered locally.

Verify it yourself with the console running:

```bash
ss -ltnp | grep 8787        # one listener, one port
python3 wand.py service status
```

### Only ever run one instance

Two instances mean two autopilots writing the same files. The launcher scans
the standard ports **and** the requested range before starting, and refuses with
a pointer to the instance already running. Detection is by asking
`/api/health`, so it works no matter how the first one was started — a
foreground `./start.sh` writes no pidfile.

```bash
python3 wand.py service stop     # background mode
# or Ctrl-C in the terminal that owns it
```

### Destructive commands refuse to run under a live server

Restoring a backup while a console is running means the server keeps writing
while you overwrite — the reliable way to corrupt a ledger. So:

- `wand storage --restore` **refuses while any instance is running** (override
  only with `--allow-running`)
- it **never deletes** current data: existing `data/` is renamed to
  `data.superseded-<timestamp>/`, so a wrong archive is recoverable
- it rejects absolute paths and `..` traversal inside the archive

If you break something badly: stop the server, `ls -d data.superseded-*`, and
rename the one you want back.

## Your data is local, and it is not lost

Nothing is "cached in memory" that disappears when you close the app. Every
decision is written to disk as it happens:

| File | Holds | Pruned? |
|---|---|---|
| `data/ledger.jsonl` | **the record**: every research cycle, claim, grade, trade, veto | **never** |
| `data/desk.json` | sandbox cash, open position, trade history | never |
| `data/policy.json` | the evolved risk policy and tier | never |
| `data/memory.json` | what reflection learned (demoted families, weights) | never |
| `data/strategies.json` | gate-approved strategies | never |
| `data/settings.json` | your API key (mode `0600`) | never |
| `data/charts/*.png` | what the agent saw | newest 60 |
| `data/snapshot_*.json` | research evidence per tick | newest 40 / 14 days |
| `data/autopilot.json`, `server.log` | scheduler state and logs | never |

Closing the terminal stops the *process*; it does not touch a byte of that.
Reopening reads it back and continues — the ledger chain verifies itself on
every load (`wand.py verify`).

### There is no database and no cloud

Not SQLite, not Postgres, not S3. `grep` the source: there is no database
driver anywhere. Data is JSON and JSONL in the project folder, and the only
outbound connections are to the public market-data APIs and to the DeepSeek API
when you have connected a key. Nothing is uploaded, and no account is created.

### Keeping it running after you close the terminal

```bash
./start.sh --bg          # detached: survives closing the terminal
python3 wand.py service status
python3 wand.py service stop
python3 wand.py service log
```

For survival across **reboots**, install a user systemd unit:

```bash
python3 wand.py service autostart
systemctl --user stop magicwand    # to turn it off
```

The background mode uses `setsid`/`start_new_session` and a pidfile, so the
server is in its own session with no controlling terminal — closing the window
cannot send it SIGHUP. (The agent sandbox this project was built in gives each
command its own PID namespace with `--die-with-parent`, so detached survival
could only be verified there by construction; on a normal desktop it behaves as
described.)

### Backups happen automatically

You do not need to remember. The server takes a **verified, rotated, local**
backup on a schedule — every 12 hours by default, keeping the newest 14 — plus
an immediate one the moment you save an API key, plus one shortly after every
start if the last is stale.

Each archive is **read back and checked for the ledger** before it counts. A
backup nobody has opened is a hope, not a backup; if verification fails it is
reported as a failure, not recorded as a success.

```
backups/magicwand-backup-20261112T132058522824.tar.gz   0600
```

Managing it, if you want to:

```bash
python3 wand.py storage                    # sizes, plus backup age
curl -s localhost:8787/api/backup          # schedule, runs, errors, last file
```

In the console: the *Local storage* card shows `backups: automatic · last 2.3h
ago · 14 kept`, with **Back up now**. The Settings dialog has the on/off toggle,
interval (1/6/12/24h) and how many to keep.

What that means in practice: the two things you cannot regenerate — the
hash-chained ledger and your API key — are never more than a few hours old, on
your own disk, without you doing anything.

**Copy one somewhere else occasionally.** Automatic backups protect against
mistakes; they do not protect against losing the disk. The archive is a plain
`tar.gz`:

```bash
cp backups/magicwand-backup-*.tar.gz ~/Dropbox/   # or a USB stick, or both
```

It contains your API key, so treat it like one: it is `0600`, and it is not
committed to git (`backups/` is gitignored).

### Housekeeping and backup commands

Old research snapshots are pruned automatically on every start (they are
evidence, not state — the decision they supported is already in the ledger).
Manual control:

```bash
python3 wand.py storage            # what is on disk, grouped, in KB
python3 wand.py storage --prune    # delete old snapshots now
python3 wand.py storage --backup   # write backups/magicwand-backup-*.tar.gz (0600)
python3 wand.py storage --restore backups/magicwand-backup-XXXX.tar.gz --force
```

The console has the same two buttons — **Back up now** and **Prune old** — in
the *Local storage* card, plus a live size breakdown. A backup archive contains
your API key if one is saved, so it is created `0600` and stays on your disk;
move or encrypt it yourself if you intend to take it anywhere.

## Run it: one command

```bash
cd ~/Music/trd/MAGICWAND
./start.sh
```

That is the whole thing. It checks the environment, starts the server, and opens
the console in your browser (`http://127.0.0.1:8787`, or the next free port).

On first run there is no API key, so the console starts in **offline mode** with
the stand-in model and shows a *"Connect DeepSeek to go live"* banner. Click
**Settings → paste your key → Save → Test connection**. That is the entire
setup; you never edit `.bashrc`.

Saving the key takes effect **immediately** — no restart:

- the terminal prints `key connected (sk-…) — mode is now LIVE`
- `data/service.json` is updated
- one research cycle runs in the background, so the console fills with real
  verdicts within a few seconds instead of showing the old offline ones
- the header badge switches from `offline` to `live model`

If it does not, the most likely cause is a server started before the fix, or
`--offline` in the command line.

```
./start.sh                 console + browser, autopilot every 15 min (live if a key exists)
./start.sh --no-browser    don't open a browser
./start.sh --interval 0    start with the autopilot idle (manual ticks only)
./start.sh --offline       force the stub model even if a key is connected
./start.sh --port 9000     different port
./start.sh --live          refuse to start without a connected key
./start.sh --bg            run detached (survives closing the terminal)
```

### The chart is redrawn every tick — but it is not a live feed

Each tick renders a fresh PNG from the last 120 **closed** bars and saves it to
`data/charts/`. The console reloads it on every render.

On the 1h timeframe that means the picture changes **once an hour**, when a new
candle closes — not every five minutes. That is deliberate: the forming bar is
never traded or drawn, because using a partial bar is lookahead. The panel says
so explicitly under the image.

If you want visible movement more often, run a shorter timeframe (`5m`, `15m`).
The strategy's statistics get worse, because costs are paid per trade and there
are more trades — the chart is not the point.

### Who is allowed to change what

Three different kinds of "the AI changes the dashboard", deliberately separated:

| Change | Who may do it | Why |
|---|---|---|
| Add a market, switch venue, change model, tune limits | **a human** (console) or a human asking the coding agent | configuration, auditable, reversible |
| Propose a market or a strategy | **the research model** | recorded in the ledger as a proposal |
| Add a market autonomously, mid-record | **nobody** | see below |

**Why the model may not retarget what the system watches.** If the agent can add
the market that "looks profitable", it will add whatever just went up. That is
selection bias, and it destroys the one thing this project is for: a track record
that means something. A ledger that says "8 of 12 readers were right" is only
worth reading if the system could not choose which questions it was asked.

So the rule is: **the model judges what it is shown; the operator chooses what it
sees.** An agent may propose a market with a reason, and that proposal is logged
— but the switching happens on a human action.

### One market per instance

The chart, the research and the strategy all follow a single symbol. Today that
is **SOL-USD** by default, and three markets are verified end to end —
`SOL-USD`, `BTC-USD`, `ETH-USD` (each needs a Coinbase pair, a Binance perp and
a subreddit; membership of that list is a claim, not a preference).

**Gold is available as PAXG**, not as XAU: no crypto venue lists spot gold, but
PAXG (1 token ≈ 1 troy oz) is on Coinbase, Kraken and Binance at ~$3,875, so it
works with every reader. It is a *proxy*: it carries token risk and can trade off
the spot price. Real XAU/USD needs a forex broker — see `docs/PLATFORMS.md`.

Markets live in one table (`magicwand/markets.py`), so adding one is a data
change: symbol, Coinbase product, Kraken pair, Binance perp, subreddit. Nothing
else needs editing, and the desk whitelist and console picker follow it
automatically.

**The candle venue is switchable too** — Settings → Market → Coinbase or Kraken.
Derivatives (funding, open interest, positioning) stay on Binance regardless,
because that is where those public endpoints exist; the snapshot labels which
source each number came from.

Switch markets in **Settings → Market**. It applies on the next tick, no restart, and
is refused while the desk holds an open position — changing market mid-trade
would orphan the position. The social reader follows the market automatically
(BTC → r/Bitcoin, ETH → r/ethtrader); without that mapping it would have kept
reading r/solana for a Bitcoin chart, which is the kind of silent mismatch that
makes a reader look confidently wrong.

**Multiple markets at once is not implemented.** The desk holds one position
(`max_open_positions = 1`), the ledger records one symbol per tick, and the
console draws one chart. Running BTC and ETH simultaneously needs portfolio
state, per-symbol risk, and a multi-chart layout — it is on the roadmap, not in
the build. The honest way to run two markets today is two instances with
separate `data/` directories, which the project does not support yet either.

### Offline mode is honest about being offline

With no key connected the console runs a **deterministic stand-in model**. It is
a pipeline placeholder: it counts bullish/bearish words in the prompt and does
nothing else.

Such verdicts are marked `stub` and therefore:

- **never veto a trade** — a keyword counter must not suppress real decisions
- **never enter an agent's graded track record** — otherwise your four readers
  would earn a hit rate built on word counting
- are shown in the console as *"OFFLINE STUB — no real analysis"* rather than a
  confident VETO

If you ever see an identical verdict repeated by all four readers, that is the
telltale: a real model gives four different readings. Connect a key and each
reader gets its own prompt and its own opinion.

A reader showing `DEGRADED` is a failed reader, not an abstention. Two causes
were found and both are fixed: the token budget was too small once thinking mode
consumed it (raised, plus retry with escalation), and models sometimes emit a
literal newline inside a JSON string, which is invalid (repaired, and the repair
is reported in the verdict's warnings so a bad format stays visible).

### What keys do I need? One.

```bash
python3 wand.py keys            # generated from the code, always current
```

| Integration | Key? | What it powers |
|---|---|---|
| **DeepSeek** — the AI | **the only one needed** | the four readers, chart vision, strategy proposals |
| Coinbase Exchange | none | the candles your strategy trades |
| Binance Futures | none | funding, positioning, open interest, taker pressure |
| Kraken | none | fallback candles |
| Crypto news RSS (4 feeds) | none | headlines for the news reader |
| Reddit | none | post volume + titles (rate-limited; often DEGRADED) |
| On-chain proxies | none | mempool, stablecoin supply, market-cap flows |
| Whale Alert / Glassnode | not wired | would replace the on-chain proxies with real netflow |
| Longer history | not wired | would let backtests span years, not ~12 days |
| **Broker / exchange** | **none — do not connect one** | nothing; there is no live order path in this build |

So: **one key, for the AI.** Everything else is public data that needs no
account, and the broker slot is deliberately empty.

The DeepSeek key needs **no special permissions** — it is a plain inference
API, not an exchange login. No scopes, no IP allow-list, no withdrawal rights
to worry about. If live trading is ever added, the rule is already written in
`docs/SAFETY.md`: trade-only key, never withdrawal, IP allow-listed, and gated
behind a human tier change.

### Connecting the APIs (in the UI)

| API | How | Why |
|---|---|---|
| **DeepSeek** | Settings → paste `sk-…` → Save → Test connection | BYOK. Enables the real agents and vision. Stored in `data/settings.json`, mode `0600`. |
| **Models** | Settings → **Fetch models** → pick from the dropdown | The list comes from `GET /models`, never from a hardcoded guess. |
| **Market data** | nothing to connect | Coinbase candles, Binance derivatives, RSS, Reddit, on-chain — all public, no keys |
| **Broker** | not available | there is no live broker in this build; paper only |

**Model fetching is real, not cosmetic.** Pressing *Fetch models* calls
`GET https://api.deepseek.com/models` and builds the picker from the response,
including each model's capabilities: whether it **accepts images** (from
`input_modalities`), its context window, and its thinking-effort levels. The
picker labels a model "sees images" or "text only — vision disabled", so you can
never select a model that silently cannot read charts. A model the API did not
report is refused; only `deepseek-flash` and `deepseek-v4-pro` are accepted as
documented fallbacks when the catalogue has not been fetched yet. The result is
cached in `data/models.json` so the picker still works offline.

*Test connection* also refreshes the catalogue and shows your **remaining
balance**, because discovering an empty account mid-run is unpleasant.

Security notes that are enforced, not promised:

- the key is **never sent to the browser** — `GET /api/settings` returns only a
  masked hint like `sk-te…0000` and the source
- the settings file is created owner-only (`0600`) and `data/` is gitignored
- **Test connection** makes one tiny real call and reports the exact error, so a
  bad key is discovered immediately rather than at 3am
- an environment variable (`DEEPSEEK_API_KEY`) still wins if you prefer it
- saving or removing a key is recorded in the ledger as `settings_change`
  (the fact of the change, never the secret)

## The operator console

```bash
./start.sh                                     # the normal way
python3 wand.py serve                          # console only, idle
python3 wand.py serve --auto                   # console + autonomous scheduler
python3 wand.py serve --auto --interval 300    # ...ticking every 5 minutes
python3 wand.py serve --live --auto            # real DeepSeek API
```

### Autopilot — the part that makes "unattended" true

The console alone does nothing: it renders state and waits. That is why
`--auto` exists. The autopilot is a single daemon thread inside the server that
fires `tick()` **immediately** on start and then every `interval` seconds,
forever, with no human involved:

- each tick: refresh data → grade matured claims → manage the desk → run the
  research agents → reflect → maybe propose → maybe trade (only a gate-approved
  strategy, only if research did not veto)
- it holds the same tick lock as the "Run one tick" button, so a human click
  cannot double-trade with the scheduler
- failures are caught per tick, counted, and written to `data/autopilot.log` —
  an autopilot that dies silently is worse than one that never started
- it **cannot raise the risk tier.** It runs whatever the human-set tier allows
- the log records `by: autopilot` vs `by: human` so machine actions are never
  confused with yours

The interval is deliberately slow by default (15 min): the data moves on hours,
the strategy is hourly, and ticking every minute would burn tokens re-reading
the same candles. Only the first tick is immediate.

A local, dependency-free dashboard (one HTML page served by the same Python)
that reads the same files the bot writes:

- **Research** — every reader's stance, confidence and thesis, plus the
  synthesizer's verdict
- **Desk** — sandbox cash, open position, recent round trips, autonomy status
- **Equity** — the mark-to-market curve
- **Reflection** — what the system learned and what it changed
- **Ledger feed** — the hash chain, live; a broken chain shows in red
- **Risk tiers** — a human-only control, recorded in the ledger
- Buttons: run one tick, run reflection

It binds to `127.0.0.1` and has no authentication: it is an operator console,
not a public service. Use a managed background job to keep it running, or a
systemd unit if you want it persistent. The tick endpoint holds a lock file, so
double-clicking "Run one tick" cannot double-trade.

## Risk tiers — the CHIMERA invariant, ported to money

This project borrows the best idea from CHIMERA, the pentest IDE in
`../chimera`: **risk ceilings come from the template, never from the AI.** There,
a Gatekeeper enforces a tool's risk tier at the command level — an agent could
name itself anything and still never mint a permission. Here, the same rule
applies to capital:

| Tier | Ceiling | Who can set it |
|---|---|---|
| `research` | read-only: analyse, propose, validate, never trade | default |
| `paper_desk` | 10% pot, ≤10% per trade, 1 trade/day | human |
| `scaled_desk` | 25% pot, ≤20% per trade, 3 trades/day | human |
| `live` | refused outright — **no live broker exists** | nobody |

Every order passes `Gatekeeper.check_order()`. It is a pure function of the
policy and the proposed order, it cannot place orders, and there is no bypass
flag. `AgentDesk` calls it before the broker or the order never happens. Agents
may *propose* a tier change; only a human action writes one, and that action is
recorded in the ledger as `tier_change` with `by: human`.

The default tier is **read-only**. You must opt in to letting anything trade.

## Reflection — why this is not a scripted bot

A script executes steps. An autonomous system changes its own behaviour because
of what it learned. `wand.py` reflects on its ledger after every tick:

- per-family expectancy and hit rate → **demote** families that lose after costs,
  **prefer** those that do not
- per-agent calibration → scale each reader's influence on the synthesizer
- stop-hit vs target-hit ratio → widen stops if they are being clipped

Then it adjusts `risk_per_trade`, `stop_atr_mult`, `target_atr_mult` and
`trade_frequency_cap` — **within human-set bounds and nowhere else**
(`TUNABLE_BOUNDS` in `gatekeeper.py`). It cannot raise its tier, cannot change
the fees it pays, and cannot change the gate.

The reflection is deliberately arithmetic, not an LLM opinion. Grading yourself
with the same instrument that produced the belief is how trading bots flatter
themselves into ruin. Demoted families are refused at proposal time, and the
tuning sizes future risk from the families it intends to *keep* trading — not
from dead strategies' history.

## Vision — the agents can look at charts

`deepseek-flash` accepts images, so the chart reader actually *sees*. MAGICWAND
renders a candlestick chart from the same bars the strategy trades — price
candles, two moving averages, volume — and asks the model to read levels,
trend structure, and volume behaviour off the picture.

```
price + MOving averages + volume  ──render──>  PNG  ──base64──>  deepseek-flash
                                                                      │
                                        validated verdict (same schema) ◄┘
```

**Why render rather than screenshot a trading site.** A screenshot of someone
else's page depends on their layout, their bot detection, their login wall, and
their terms; and it can show a chart the strategy never traded. A chart drawn
from our own data has none of those failure modes, is byte-for-byte
reproducible, and shows exactly the series the decision was made on. A real
browser is used only if you ask for a public page — we never log in anywhere.

**The image is untrusted input.** It can contain rendered text, and rendered
text can contain instructions. The prompt tells the model to treat any
instruction inside the image as evidence of manipulation and flag it
(`injected_text_seen`), and — as with every reader — a chart verdict can only
ever **veto**, never open a trade. Offline, the fake model explicitly reports
"the image was NOT actually inspected" rather than pretending.

Every rendered chart is saved to `data/charts/` and the exact filename is
attached to the claim, so a graded claim can be checked against the picture
later. The console serves the latest one at `/api/chart` and shows it beside
the vision verdict.

Cost: an image costs at most 1024 tokens ≈ **$0.0003** at peak `deepseek-flash`
prices.

## Attempts: unlimited, novelty-bounded

CHIMERA's evolution has a `MutationBudget` with `per_engagement_cap: 50`: once
50 mutations are spent, `try_spend()` returns `CapReached` and the system stops
being able to adapt while still appearing to run. That is the failure mode to
avoid.

MAGICWAND has **no attempt cap at all**. Autonomy never switches itself off
after N tries, and there is no backoff that silently stretches into nothing.
What is bounded is *repetition*, not *count*:

- identical proposal → refused
- any parameter differing by <10% of its catalog range → refused as a
  near-duplicate
- anything genuinely new → allowed, forever

So the honest limit is "we ran out of untried ideas", not "we ran out of
budget". The proposer is told what it already tried, so it must go somewhere
new.

## What it found on day one

Run against 300 hourly SOL-USD candles (2026-10-30 → 2026-11-11):

```
  strategy        sma_trend
  return          -2.04%
  buy & hold      -12.06%   <-- the benchmark to beat
  max drawdown    2.04%
  trades          4  (win 25%)
  expectancy      -2.04% per trade   t-stat -1.67

  2x costs        -3.38%   not profitable — costs own it
  1 extra bar lag -1.50%   lag cost -0.53% of performance
  shuffled data   -0.03% average across 5 runs   beat 0/5 shuffles (no sign of skill)

  VERDICT: No evidence of edge. Do not trade this with real money.
```

The control group — `--strategy coin_flip`, random entries with identical
exit rules — lost money too, with a **t-stat of -3.30**, i.e. its losses are
statistically *more* certain than any signal found. That is what paying
1.2% round-trip costs looks like.

This is the honest state of a first attempt: **no edge found yet.** The
tool's first job was to tell you that, and it did.

## The $50 reality check (read before dreaming)

At $50 equity with a 25% position cap, the biggest position the risk engine
will allow is **$12.50 of notional**. On Coinbase that means ~$0.15 of fees
per round trip — small in dollars, but **0.3% of the entire account every
single trade**. The 2% daily-loss limit is $1.00, so roughly **6.7 round
trips exhaust the day's risk budget**, before a single bad trade.

And when the math says the trade would be smaller than the $5 exchange
minimum, the engine **refuses it outright rather than rounding up**:

```bash
python3 -c "
from magicwand.config import Config
from magicwand.risk import RiskEngine
from datetime import datetime, timezone
c = Config(); c.risk.equity = 50.0
d = RiskEngine(c.risk).size_entry(50.0, 100.0, 30.0, 0, datetime.now(timezone.utc))
print(d.allowed, '|', d.reason)
"
# False | trade too small to place: 1.67 < exchange minimum 5.00 ...
#         (binding: risk-per-trade). Account is too small for this stop distance.
```

So at $50 the bot is a **test rig, not an income source**. It can place real
paper trades, prove the pipeline, and grade itself honestly — and the fees
make it brutally clear why "turn $40 into $11,000 in a week" needs leverage
rather than skill. Fund it properly only after a strategy survives the
reality check, and then fund it from savings, not from hope.

## Roadmap

- [x] Data feeds (Coinbase, Kraken, synthetic), cached and fingerprinted
- [x] Indicators, causal and tested
- [x] Pessimistic paper broker, risk engine, hash-chained ledger
- [x] Backtester with delayed fills and a built-in reality check
- [x] Self-grading scorecard
- [x] Four research agents on live public data (derivatives, news, social, on-chain)
- [x] Veto-only synthesizer with per-agent track-record weighting
- [x] Sandboxed Agent Desk: autonomous trading inside hard walls, revocable by record
- [x] BYOK DeepSeek client with daily call and cost budgets
- [x] Self-invented strategies from an audited catalog (no agent-written code)
- [x] Validation gate: realistic costs, 2x costs, sample floor, buy-and-hold
      exposure benchmark, out-of-sample holdout, shuffled-data null test
- [x] Unattended autonomous runner with adaptive revalidation
- [x] Risk tiers + gatekeeper (CHIMERA invariant), human-only escalation
- [x] Reflection engine with bounded self-tuning and family demotion
- [x] Local zero-dependency operator dashboard
- [x] Autopilot: an in-server clock so the system runs unattended
- [x] Equity marks written every tick, so the chart has a real series
- [x] Vision: rendered candlestick charts read by `deepseek-flash`
- [x] Image prompt-injection detection; chart verdicts are veto-only
- [x] No attempt cap — novelty-bounded instead of count-bounded
- [ ] Longer history (the free hourly window is only ~12 days)
- [ ] Walk-forward / out-of-sample split, so tuning cannot cheat
- [ ] Portfolio-level risk across several symbols
- [ ] `Broker` protocol, then paper/demo adapters (Alpaca → OANDA → Binance → MT5)
- [ ] Live execution — only after a paper strategy survives the reality check, at 1% risk, with a kill switch

## Platforms (planned)

Where this could trade, in the order that is safe rather than the order that is
exciting. Full reasoning in [`docs/PLATFORMS.md`](docs/PLATFORMS.md).

| Platform | Markets | First step | Constraint |
|---|---|---|---|
| **Alpaca** | US stocks/ETFs | paper keys | easiest; fractional shares |
| **OANDA v20** | forex, metals | practice account | clean REST |
| **Binance / Bybit** | crypto spot & perps | testnet | funding, leverage |
| **Coinbase Advanced** | crypto spot | trade-only key | matches our candle feed |
| **MetaTrader 5** | forex, metals, CFDs | **demo** account | **Windows-only Python API** → needs a bridge |
| MetaTrader 4 | legacy | — | no official Python API; skip, use MT5 |
| Interactive Brokers | stocks, options, futures | paper account | needs TWS/Gateway running |

All of them are gated behind the `live` tier, which the gatekeeper currently
refuses. Two rules hold for every adapter: **paper/demo before real**, and
**trade-only keys, never withdrawal**.

The honest sequencing note: extracting a `Broker` protocol comes first, and no
live adapter is worth building until a strategy actually survives the validation
gate — otherwise it is plumbing attached to nothing.

## Safety

Read [`docs/SAFETY.md`](docs/SAFETY.md). Short version: no live trading, no
withdrawal capability, no API keys in this repo, paper only.

## License

MIT. It is your money — audit it.
