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
python3 -m unittest discover -s tests -v   # 24 tests, no dependencies
python3 wand.py data                        # fetch real SOL/USD hourly bars
python3 wand.py reality                     # the honest verdict
python3 wand.py step                        # one paper tick, writes the ledger
python3 wand.py status                      # scorecard
```

Optional: `pip install -e .` then use `wand` instead of `python3 wand.py`.

Zero dependencies. Pure standard library, Python 3.10+.

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
- [ ] Longer history (the free hourly window is only ~12 days)
- [ ] Walk-forward / out-of-sample split, so tuning cannot cheat
- [ ] Portfolio-level risk across several symbols
- [ ] LLM research layer: news summaries, filing parsing, plain-English trade explanations — *outside* the money decision
- [ ] Live execution — only after a paper strategy survives the reality check, at 1% risk, with a kill switch

## Safety

Read [`docs/SAFETY.md`](docs/SAFETY.md). Short version: no live trading, no
withdrawal capability, no API keys in this repo, paper only.

## License

MIT. It is your money — audit it.
