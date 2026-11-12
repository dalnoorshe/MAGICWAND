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
python3 -m unittest discover -s tests -v   # 105 tests, no dependencies
python3 wand.py data                        # fetch real SOL/USD hourly bars
python3 wand.py reality                     # the honest verdict
python3 wand.py step                        # one paper tick, writes the ledger
python3 wand.py status                      # scorecard
```

Optional: `pip install -e .` then use `wand` instead of `python3 wand.py`.

Zero dependencies. Pure standard library, Python 3.10+.

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

## The operator console

```bash
python3 wand.py serve                  # http://127.0.0.1:8787
python3 wand.py serve --live           # use the real DeepSeek API
```

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
- [ ] Longer history (the free hourly window is only ~12 days)
- [ ] Walk-forward / out-of-sample split, so tuning cannot cheat
- [ ] Portfolio-level risk across several symbols
- [ ] Live execution — only after a paper strategy survives the reality check, at 1% risk, with a kill switch

## Safety

Read [`docs/SAFETY.md`](docs/SAFETY.md). Short version: no live trading, no
withdrawal capability, no API keys in this repo, paper only.

## License

MIT. It is your money — audit it.
