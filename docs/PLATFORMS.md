# PLATFORMS — where MAGICWAND could trade, and in what order

Status: **plan only.** Nothing in this document is implemented. There is no
live order path in the repository, and the `live` risk tier exists solely to be
refused.

This file exists so the ambition is written down with its constraints attached,
instead of becoming a surprise later.

---

## The rule that does not change

Every platform below is gated behind the same invariant:

1. **`live` tier is refused by the gatekeeper** until a human raises it, and
   raising it requires an adapter that has already traded a demo account.
2. **Paper first, always.** The adapter must run against the venue's *demo or
   paper* endpoint and produce a ledger record of at least a few hundred
   orders before real money is considered.
3. **Trade-only keys.** No withdrawal permission, ever. IP allow-list where the
   venue offers it.
4. **One execution interface.** Live adapters implement the same `Broker`
   protocol as `PaperBroker`, so the strategy, risk engine and ledger cannot
   tell the difference — which is what makes paper results comparable.
5. **Reconcile every run.** Positions that exist at the venue but not locally
   are the classic way an account dies.
6. **The LLM never places an order.** It proposes strategies; the gate decides;
   code executes.

---

## The one code change every platform needs

Today `PaperBroker` is the only execution implementation, and it is called
directly. Before any live adapter, extract its surface into a protocol:

```
Broker (protocol)
  ├── PaperBroker          exists — fills pessimistically, costs modelled
  ├── AlpacaBroker         paper + live, REST
  ├── OandaBroker          practice + live, REST
  ├── BinanceBroker        testnet + live
  ├── Mt5Broker            demo + live, Windows-bound
  └── IbkrBroker           paper + live, TWS/Gateway
```

Shared concerns that must live above the adapter, not inside each one:

- **symbol mapping** — `SOL-USD` vs `SOLUSDT` vs `SOLUSD.a`; contract sizes,
  tick sizes, minimum quantities
- **order state machine** — accepted → partially filled → filled / cancelled /
  rejected, with a timeout policy
- **reconciliation** — pull venue positions and open orders at start and after
  every fill; refuse to trade on a mismatch
- **rate limits and retries** — each venue's own budget, with backoff that does
  not silently stretch into nothing (the CHIMERA lesson)
- **clock** — venue time vs local time; funding/settlement hours

---

## Platforms, ranked by ease of a *safe* first integration

### 1. Alpaca — US equities & ETFs
**Why first:** free paper trading that behaves like live, a clean REST API, and
fractional shares (which matters at small account sizes).
**Needs:** account + key pair (paper keys are separate from live).
**Constraints:** US market hours; pattern-day-trader rules on margin accounts;
fractional orders are market-only at some venues.
**Adapter work:** small — REST, well documented, no desktop app.

### 2. OANDA v20 — forex & metals
**Why:** practice account mirrors production; REST; clean order model; the
closest thing to a "real broker" that is still easy to test.
**Needs:** practice account + token.
**Constraints:** leverage is the danger, not the API. Position sizing must be
driven by the risk engine, never by available margin.

### 3. Binance / Bybit — crypto spot and perps
**Why:** we already read their public data for the derivatives reader, so the
symbol vocabulary matches.
**Needs:** testnet keys first, then trade-only live keys.
**Constraints:** testnet liquidity differs from production; perpetual funding
changes the economics; leverage must be capped by the tier, and the desk defaults
to spot-like sizing.

### 4. Coinbase Advanced Trade — crypto spot
**Why:** already the candle source, so prices match the strategy's view exactly.
**Needs:** API key with trade permission only.

### 5. MetaTrader 5 — forex, metals, indices, CFDs
**Why it matters here:** this is where most of the "signal group" world lives,
and where the videos the operator saw make their claims. Being able to run a
verified strategy on a *demo* MT5 account is a direct answer to that world.
**Needs:** MT5 terminal + an account (demo first), and the official
`MetaTrader5` Python package.
**Hard constraint — the reason it is not first:** the official Python API is
**Windows-only**. On this Linux box it requires Wine, a Windows VM, or a
separate Windows machine acting as an execution bridge. That is a real
architectural decision, not a packaging detail:

```
  MAGICWAND (Linux)  ──HTTP/queue──>  bridge agent (Windows)  ──>  MT5 terminal
```

**Adapter work:** medium-to-large. Plan for the bridge to be a separate, tiny,
auditable process that accepts only "place/cancel/flatten" with an explicit
symbol allow-list, and holds the credentials — never the main app.

### 6. MetaTrader 4 — legacy
**Why not:** no official Python API. It needs an Expert Advisor bridge (ZeroMQ
or file-based) or a third-party package. More moving parts, more failure modes,
same markets as MT5. **Recommendation: skip MT4 and target MT5.**

### 7. Interactive Brokers — stocks, options, futures, FX
**Why it is valuable:** breadth and a genuine paper account.
**Why it is last:** it requires TWS or IB Gateway running as a desktop service,
has a notoriously fiddly API, and its paper environment differs from live in
subtle ways.
**Adapter work:** large. `ib_insync` helps; the session management does not.

### 8. Others worth noting
- **cTrader Open API** — forex/CFDs, OAuth; good alternative to MT5 on Linux.
- **Tradovate / Rithmic** — futures specialists; more complex clearing model.

---

## Order of work

| Step | What | Gate to pass first |
|---|---|---|
| 1 | `Broker` protocol extracted; `PaperBroker` conforms | existing tests stay green |
| 2 | Alpaca paper adapter | paper-trades N orders with full reconciliation |
| 3 | OANDA practice adapter | same |
| 4 | Binance testnet adapter | same |
| 5 | MT5 Windows bridge + demo adapter | demo account, explicit symbol allow-list |
| 6 | IBKR paper adapter | paper account, session recovery tested |
| 7 | *Any* live adapter | a strategy that survived the validation gate **and** months of paper record |

Step 7 is the only one that can lose money, and it is deliberately last.

---

## What would make this document wrong

- A venue changing its API (Alpaca and OANDA both do occasionally).
- MT5 shipping a Linux-native Python API (it has not, as of this writing).
- The gate never approving a strategy — in which case live adapters would be
  wiring for a machine that has nothing to say.

That last one is the honest risk. Building six adapters before a single strategy
survives the gate would be elaborate plumbing attached to nothing.
