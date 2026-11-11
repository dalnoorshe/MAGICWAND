# SAFETY

## The hard boundary

**This repository cannot move real money.** There is no live broker class,
no exchange order endpoint, no API-key handling. The only execution path is
`PaperBroker`, which runs against a number in a JSON file.

That is a deliberate design choice, not an unfinished feature.

## If you ever add live trading

Do these in order, and do not skip:

1. **Create the API key with trade permission only.** No withdrawal
   permission, ever. If an exchange offers IP allow-listing, use it.
2. **Start at 1% of the paper-proven size.** Not 100%.
3. **Hard-code a kill switch** that a running process cannot rewrite:
   a file whose presence blocks all orders, or an exchange-level limit.
4. **Keep a manual override** that can flatten everything without the bot's
   cooperation.
5. **Log before you act.** Write intent to disk, then place the order. If
   the process dies mid-order, the ledger should show what was attempted.
6. **Reconcile against the exchange every run.** Positions that exist on the
   exchange but not in local state are the classic way accounts get destroyed.
7. **Never let an LLM place orders directly.** The model proposes; deterministic
   code disposes. An LLM that hallucinates a ticker should produce a rejected
   order and a log line, not a position.

## What this project will not do

- Promise returns, percentages, or timeframes. Nobody can.
- Help chase a screenshot. If the plan is "a small account becomes a large
  one quickly", the mechanism is always leverage or ruin, and both end the
  same way.
- Round a too-small order up to fit an exchange minimum. That converts a
  refused trade into an oversized one.

## The honest expectations

A retail bot with realistic fees competes against:

- **costs**: 0.5–1.2% round trip on small accounts
- **spread and slippage**: worse in thin markets and at bad times
- **counterparties** who are faster and better capitalised
- **its own author's overfitting** — the deadliest one, because it feels like progress

The realistic first goal is **not losing much.** The second is a small,
statistically defensible edge. Anything claiming more, ask what the fees were.

## Data and keys

- No API keys are required or stored. Feeds are public market data.
- `data/` holds candles, the paper portfolio, and the ledger. Treat the
  ledger as the record of truth — `wand verify` detects tampering.
- Nothing in this repo phones home except the public data endpoints.
