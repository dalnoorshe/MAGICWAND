"""MAGICWAND — the markets registry.

One table describing every market this build can watch, and where each piece of
evidence for it comes from. Adding a market should be a data change, not a code
change — that was the point of asking "can we just add XAU/USD to the chart?".

A market is only listed here if all four of its sources actually exist:

    candles    a Coinbase pair (the default feed) or a Kraken pair
    derivatives a Binance USDT-M perpetual (funding, OI, positioning)
    social      a subreddit to read
    onchain     the shared proxies (not per-market, but stated for honesty)

If a source does not exist, the market is not listed. Membership of this table
is a claim about reality, so it needs to be checked, not assumed.

WHAT THIS FILE DOES *NOT* DO
    It does not let the language model decide what the system watches. An agent
    may propose a market; a human adds it here or approves it in the console.
    The reason is selection bias: if the agent can retarget the system to
    whatever just went up, it will, and the track record becomes a record of
    chasing winners after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    symbol: str                 # canonical, e.g. "PAXG-USD"
    name: str
    coinbase: str               # Coinbase product, or "" if unsupported there
    kraken: str                 # Kraken pair, or "" if unsupported there
    binance_perp: str           # Binance USDT-M symbol for derivatives
    subreddit: str
    min_notional: float = 5.0   # venue minimum this market is verified against
    note: str = ""

    def supports(self, exchange: str) -> bool:
        if exchange == "coinbase":
            return bool(self.coinbase)
        if exchange == "kraken":
            return bool(self.kraken)
        if exchange == "synthetic":
            return True
        return False


# Verified against the public APIs. PAXG is tokenised gold — 1 token is meant to
# be 1 troy ounce, so it tracks XAU/USD closely but is NOT spot gold: it carries
# token risk and can trade at a premium/discount.
MARKETS: dict[str, Market] = {
    "SOL-USD": Market(
        symbol="SOL-USD", name="Solana", coinbase="SOL-USD", kraken="SOLUSD",
        binance_perp="SOLUSDT", subreddit="solana"),
    "BTC-USD": Market(
        symbol="BTC-USD", name="Bitcoin", coinbase="BTC-USD", kraken="XBTUSD",
        binance_perp="BTCUSDT", subreddit="Bitcoin"),
    "ETH-USD": Market(
        symbol="ETH-USD", name="Ethereum", coinbase="ETH-USD", kraken="ETHUSD",
        binance_perp="ETHUSDT", subreddit="ethtrader"),
    "PAXG-USD": Market(
        symbol="PAXG-USD", name="Gold (tokenised, PAXG)",
        coinbase="PAXG-USD", kraken="PAXGUSD",
        binance_perp="PAXGUSDT", subreddit="Gold",
        note="proxy for XAU/USD: token risk, and it can trade off the spot price"),
}

DEFAULT_MARKET = "SOL-USD"


def get(symbol: str) -> Market | None:
    return MARKETS.get((symbol or "").upper())


def symbols() -> list[str]:
    return sorted(MARKETS)


def for_exchange(exchange: str) -> list[str]:
    """Markets this candle feed can actually serve."""
    return [s for s, m in sorted(MARKETS.items()) if m.supports(exchange)]


def coinbase_product(symbol: str) -> str:
    m = get(symbol)
    return m.coinbase if m else symbol


def kraken_pair(symbol: str) -> str:
    """Kraken pair for a canonical symbol, falling back to stripping the dash."""
    m = get(symbol)
    if m and m.kraken:
        return m.kraken
    return (symbol or "").replace("-", "")


def binance_perp(symbol: str) -> str:
    m = get(symbol)
    if m:
        return m.binance_perp
    return (symbol or "").split("-")[0].upper() + "USDT"


def subreddit_for(symbol: str) -> str:
    m = get(symbol)
    return m.subreddit if m else "CryptoCurrency"


def is_supported(symbol: str) -> bool:
    return get(symbol) is not None


def whitelist() -> tuple[str, ...]:
    """Symbols the desk may trade. Same set, by construction."""
    return tuple(sorted(MARKETS))
