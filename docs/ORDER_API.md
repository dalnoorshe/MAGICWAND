# ORDER APIS — what a buy/sell call actually looks like

Short answer: **yes.** Every venue has an authenticated "place order" endpoint,
and it is the one call this repository deliberately does not implement.

Right now MAGICWAND can read prices from Coinbase, Binance and Kraken with **no
key at all** — market data is public. Trading is the only thing that needs
credentials, and `PaperBroker` is the only execution path in the code.

---

## The endpoints

| Venue | Place an order | Auth | Test/demo |
|---|---|---|---|
| **Binance Spot** | `POST /api/v3/order` | API key + HMAC-SHA256 (or Ed25519) signature over the query string | `https://testnet.binance.vision` |
| **Binance USDⓈ-M futures** | `POST /fapi/v1/order` | same signing | same testnet |
| **Kraken** | `POST /0/private/AddOrder` | `API-Key` + `API-Sign` = HMAC-SHA512 of `path + nonce + postdata` | no public testnet; use a small live account |
| **Coinbase Advanced Trade** | `POST /api/v3/brokerage/orders` | JWT (ES256) or legacy HMAC | Coinbase sandbox exists but is limited |
| **Alpaca** | `POST /v2/orders` | `APCA-API-KEY-ID` + `APCA-API-SECRET-KEY` | `https://paper-api.alpaca.markets` — free and behaves like live |
| **OANDA v20** | `POST /v3/accounts/{id}/orders` | `Authorization: Bearer <token>` | free practice account |
| **MetaTrader 5** | not REST — `MetaTrader5.order_send()` | terminal login | demo account; **Windows Python only** |

References: [Binance trading endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints),
[Kraken AddOrder](https://docs.kraken.com/api-reference/trading/add-order),
[Alpaca orders](https://docs.alpaca.markets/us/docs/working-with-orders),
[Coinbase Advanced Trade](https://docs.cdp.coinbase.com/advanced-trade/), and
[Kraken's authentication algorithm](https://support.kraken.com/articles/360029054811-what-is-the-authentication-algorithm-for-private-endpoints-).

A Binance spot order, minimally, is:

```
POST /api/v3/order
  symbol=BTCUSDT&side=BUY&type=MARKET&quoteOrderQty=10
  &newClientOrderId=mw-20261116-0001
  &timestamp=1763000000000&recvWindow=5000
  &signature=<HMAC-SHA256(secret, the exact query string above)>
Headers: X-MBX-APIKEY: <key>
```

Note what is in there: an **idempotency key** (`newClientOrderId`), a
**timestamp** (the server rejects requests outside `recvWindow`), and a
**signature** that must be computed over the exact bytes sent.

---

## What actually goes wrong in production

These are why an adapter is a week of work, not an afternoon:

1. **Clock skew** — a laptop 2 seconds off gets `-1021 Timestamp for this
   request is outside of the recvWindow`. You must sync with `/api/v3/time`.
2. **Duplicate orders** — a timeout does not mean the order failed. Without a
   client order id you can buy twice. Every order needs one, and every retry
   must reuse it.
3. **Partial fills** — `FILLED` is not the only outcome. `PARTIALLY_FILLED`,
   `NEW`, `CANCELED`, `EXPIRED`, `REJECTED` all exist, and the local position
   must follow the venue's, not the other way round.
4. **Rate limits** — 429 and 418 (IP ban) are real. Backoff must be bounded and
   visible, never silently stretched.
5. **Minimum notional and tick size** — venues reject orders below a minimum or
   with a price off the tick grid. Rounding *up* to fit is how you place an
   order ten times larger than intended.
6. **Symbol translation** — `SOL-USD` (Coinbase) is `SOLUSDT` (Binance) is
   `SOLUSD` (Kraken).
7. **Reconciliation** — on start and after every fill, pull venue balances and
   open orders. A position that exists at the venue but not locally is the
   classic way an account is destroyed.

---

## The order of work, if we build one

1. **`Broker` protocol** — `fetch_equity()`, `place_order()`, `cancel_order()`,
   `open_orders()`, `positions()`. `PaperBroker` already does the accounting
   half; a live adapter implements the same surface so the strategy cannot tell
   them apart.
2. **Read-only first** — connect, fetch equity and positions, reconcile, and log
   for a day. No orders.
3. **Testnet orders** — Binance testnet or Alpaca paper. Same code path, fake
   money.
4. **One instrument, minimum size, hard cap** — a single symbol, the venue
   minimum, and a kill switch that a running process cannot rewrite.
5. **Then, and only then**, the `live` tier stops being refused.

Rules that do not change: **trade-only keys, never withdrawal** · IP allow-list
where offered · the gatekeeper stays the last check before the HTTP call · the
LLM never places an order.

---

## Why it is not implemented yet

Because no strategy has passed the validation gate. Wiring an order endpoint to
a system with nothing worth trading is how accounts die slowly — and it would
make the `live` tier meaningful before there is any evidence to justify it.

The plumbing is ready for it: `account.resolve_account()` already prefers a
broker's own `fetch_equity()`, and the desk sizes from whatever that returns.
