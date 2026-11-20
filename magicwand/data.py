"""MAGICWAND — market data feeds.

Three sources:
  * Coinbase  (public, no key, the default)
  * Kraken    (public, no key, fallback)
  * Synthetic (deterministic random walk for tests; no network)

All feeds return a list of closed Bars, oldest first, timezone-aware UTC.
Data is only ever used *after* the bar has closed — no future peeking.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .common import Bar, parse_iso

COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
_UA = {"User-Agent": "magicwand/0.1 (+paper-trading research)"}

# Coinbase granularity is in seconds.
# Coinbase's granularity parameter accepts only this set, so this map is ALSO
# its argument whitelist. Bar length in seconds is a separate question: Binance
# happily serves 4h bars, which Coinbase does not, and using this map for both
# jobs crashed the 4h forecast with a KeyError.
_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
_BAR_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
                "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
                "1d": 86400}
_KRAKEN_INTERVAL = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "6h": 360, "1d": 1440}
# Kraken renames some pairs; BTC/ETH are the classic ones.
_KRAKEN_ALIASES = {"BTC": "XBT"}


class FeedError(RuntimeError):
    pass


def _get_json(url: str, timeout: float = 20.0) -> object:
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise FeedError(f"HTTP {e.code} from {url}") from e
    except urllib.error.URLError as e:
        raise FeedError(f"network error for {url}: {e.reason}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise FeedError(f"bad JSON from {url}") from e


# --------------------------------------------------------------------------
# Coinbase
# --------------------------------------------------------------------------

def coinbase_candles(symbol: str = "SOL-USD", timeframe: str = "1h",
                     candles: int = 300, retries: int = 2) -> list[Bar]:
    gran = _GRANULARITY.get(timeframe)
    if gran is None:
        raise FeedError(f"unsupported timeframe {timeframe!r}")
    # Do NOT clamp to 300 here: 300 is the per-request limit, and the loop
    # below paginates. Clamping made every request for more history silently
    # return a single page.
    candles = max(10, int(candles))

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            # Coinbase allows limit<=300 with `end`; paging backwards with
            # `end` only. (Combining start+limit+end returns HTTP 400.)
            out: dict[int, Bar] = {}
            end = datetime.now(timezone.utc).replace(microsecond=0)
            window = min(300, max(1, candles))
            while len(out) < candles:
                start = end - timedelta(seconds=gran * window)
                url = (f"{COINBASE}/products/{symbol}/candles"
                       f"?granularity={gran}"
                       f"&start={int(start.timestamp())}&end={int(end.timestamp())}")
                rows = _get_json(url)
                if not isinstance(rows, list):
                    raise FeedError(f"unexpected candle payload for {symbol}")
                if not rows:
                    break
                before = len(out)
                for row in rows:
                    # Coinbase row order: [time, low, high, open, close, volume]
                    t, lo, hi, op, cl, vol = row[0], row[1], row[2], row[3], row[4], row[5]
                    ts = datetime.fromtimestamp(int(t), tz=timezone.utc)
                    try:
                        out[int(t)] = Bar(ts=ts, open=float(op), high=float(hi),
                                          low=float(lo), close=float(cl), volume=float(vol))
                    except ValueError:
                        continue  # skip malformed candles rather than crash money code
                if len(out) == before:
                    break  # no progress: stop instead of looping forever
                end = start - timedelta(seconds=gran)
            bars = [out[k] for k in sorted(out)]
            if len(bars) < 5:
                raise FeedError(f"only {len(bars)} candles returned for {symbol}")
            _drop_forming_bar(bars, gran)
            return bars[-candles:]
        except FeedError as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise FeedError(f"Coinbase feed failed after {retries + 1} tries: {last_err}")


BINANCE = "https://api.binance.com"
_BINANCE_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
                     "4h": "4h", "6h": "6h", "1d": "1d"}


def binance_candles(symbol: str, timeframe: str = "1h",
                    candles: int = 1000) -> list[Bar]:
    """Spot klines from Binance: 1000 per request, years of history.

    Used for long backtests and for training the trend model, where the ~300
    bar limit of the Coinbase endpoint is not enough.
    """
    interval = _BINANCE_INTERVAL.get(timeframe)
    if interval is None:
        raise FeedError(f"unsupported timeframe {timeframe!r} for Binance")
    pair = symbol.replace("-", "").upper()
    if not pair.endswith("USDT"):
        pair = pair.replace("USDT", "") + "USDT"
    out: dict[int, Bar] = {}
    end_ms: Optional[int] = None
    while len(out) < candles:
        take = min(1000, candles - len(out))
        url = (f"{BINANCE}/api/v3/klines?symbol={pair}&interval={interval}"
               f"&limit={take}")
        if end_ms is not None:
            url += f"&endTime={end_ms}"
        rows = _get_json(url)
        if not isinstance(rows, list) or not rows:
            break
        before = len(out)
        for row in rows:
            try:
                ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                out[int(row[0])] = Bar(
                    ts=ts, open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]), volume=float(row[5]))
            except (ValueError, IndexError, TypeError):
                continue
        if len(out) == before:
            break
        end_ms = min(out) - 1
        if len(rows) < take:
            break
    bars = [out[k] for k in sorted(out)]
    if len(bars) < 5:
        raise FeedError(f"Binance returned {len(bars)} candles for {pair}")
    _drop_forming_bar(bars, _bar_seconds(timeframe))
    return bars[-candles:]


def _drop_forming_bar(bars: list[Bar], gran_seconds: int) -> None:
    """The newest candle may still be open. Drop it so we never trade a partial bar."""
    if not bars:
        return
    now = datetime.now(timezone.utc)
    if now.timestamp() - bars[-1].ts.timestamp() < gran_seconds:
        bars.pop()


# --------------------------------------------------------------------------
# Kraken
# --------------------------------------------------------------------------

def kraken_candles(symbol: str = "XBTUSD", timeframe: str = "1h",
                   candles: int = 300) -> list[Bar]:
    interval = _KRAKEN_INTERVAL.get(timeframe)
    if interval is None:
        raise FeedError(f"unsupported timeframe {timeframe!r}")
    base, quote = symbol.split("-") if "-" in symbol else (symbol[:3], symbol[3:])
    pair = _KRAKEN_ALIASES.get(base, base) + quote
    payload = _get_json(f"{KRAKEN}/OHLC?pair={pair}&interval={interval}")
    if payload.get("error"):
        raise FeedError(f"Kraken: {payload['error']}")
    result = payload.get("result", {})
    keys = [k for k in result if k != "last"]
    if not keys:
        raise FeedError(f"Kraken returned no OHLC for {pair}")
    rows = result[keys[0]]
    bars: list[Bar] = []
    for row in rows[-candles:]:
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        try:
            bars.append(Bar(ts=ts, open=float(row[1]), high=float(row[2]),
                            low=float(row[3]), close=float(row[4]), volume=float(row[6])))
        except ValueError:
            continue
    _drop_forming_bar(bars, _bar_seconds(timeframe))
    return bars


# --------------------------------------------------------------------------
# Synthetic (tests + offline demos)
# --------------------------------------------------------------------------

def synthetic_bars(n: int = 500, start_price: float = 150.0, seed: int = 7,
                   vol: float = 0.012, drift: float = 0.0,
                   timeframe: str = "1h") -> list[Bar]:
    """Deterministic GBM-ish walk. Same seed => same bars, always.

    `drift` is per-bar fractional drift. With drift=0 this is a fair coin,
    which is the correct null hypothesis for "does the strategy have an edge?"
    """
    rng = random.Random(seed)
    gran = _GRANULARITY[timeframe]
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    px = start_price
    bars: list[Bar] = []
    for i in range(n):
        o = px
        ret = rng.gauss(drift, vol)
        c = max(1e-9, o * math.exp(ret))
        wick = abs(rng.gauss(0, vol * 0.6))
        hi = max(o, c) * math.exp(wick)
        lo = min(o, c) * math.exp(-wick)
        bars.append(Bar(ts=t0 + timedelta(seconds=gran * i), open=o, high=hi,
                        low=lo, close=c, volume=1000 + rng.random() * 500))
        px = c
    return bars


# --------------------------------------------------------------------------
# Cached / offline feed
# --------------------------------------------------------------------------

def cache_path(root: str | Path, exchange: str, symbol: str, timeframe: str) -> Path:
    d = Path(root) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exchange}_{symbol}_{timeframe}.json"


def save_bars(path: Path, bars: list[Bar]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([{
        "ts": b.ts.isoformat(), "open": b.open, "high": b.high,
        "low": b.low, "close": b.close, "volume": b.volume,
    } for b in bars]))
    tmp.replace(path)


def load_bars(path: Path) -> list[Bar]:
    return [Bar(ts=parse_iso(r["ts"]), open=r["open"], high=r["high"],
                low=r["low"], close=r["close"], volume=r["volume"])
            for r in json.loads(Path(path).read_text())]


def fetch(exchange: str, symbol: str, timeframe: str = "1h",
          candles: int = 300, seed: int = 7,
          cache_root: str | Path = ".") -> list[Bar]:
    """One entry point the rest of the app uses.

    Canonical symbols ("PAXG-USD") are translated to each venue's own naming
    ("PAXG-USD" on Coinbase, "PAXGUSD" on Kraken) by the markets registry.
    """
    if exchange == "coinbase":
        from .markets import coinbase_product
        bars = coinbase_candles(coinbase_product(symbol), timeframe, candles)
    elif exchange == "binance":
        from .markets import binance_spot
        bars = binance_candles(binance_spot(symbol), timeframe, candles)
    elif exchange == "kraken":
        from .markets import kraken_pair
        bars = kraken_candles(kraken_pair(symbol), timeframe, candles)
    elif exchange == "synthetic":
        bars = synthetic_bars(candles, seed=seed, timeframe=timeframe)
    else:
        raise FeedError(f"unknown exchange {exchange!r}")
    try:
        save_bars(cache_path(cache_root, exchange, symbol, timeframe), bars)
    except OSError:
        pass  # caching is a nicety, never a failure mode
    return bars


def _bar_seconds(timeframe: str) -> int:
    """Bar length in seconds. Never defaults silently — an unknown timeframe is
    a bug in the caller, not something to paper over with an hour."""
    seconds = _BAR_SECONDS.get(timeframe)
    if seconds is None:
        raise FeedError(f"unsupported timeframe {timeframe!r}")
    return seconds


def timeframe_seconds(timeframe: str) -> int:
    return _BAR_SECONDS.get(timeframe, 3600)


def age_seconds(bars: list[Bar], now=None) -> float:
    """How old the newest closed bar is, in seconds."""
    if not bars:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - bars[-1].ts).total_seconds())


def is_stale(bars: list[Bar], timeframe: str, max_age_bars: float) -> bool:
    """True when the feed has stopped advancing.

    A halted market, a dead endpoint, or a cached file all look like a quiet
    market. Trading them means trading a price that no longer exists.
    """
    return age_seconds(bars) > timeframe_seconds(timeframe) * float(max_age_bars)


def fingerprint(bars: list[Bar]) -> str:
    """Stable hash of a dataset, so a result can be tied to exact data."""
    h = hashlib.sha256()
    for b in bars:
        h.update(f"{b.ts.isoformat()},{b.open:.10f},{b.high:.10f},"
                 f"{b.low:.10f},{b.close:.10f},{b.volume:.10f};".encode())
    return h.hexdigest()[:16]
