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
_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
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
    candles = max(10, min(candles, 300))

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            # Coinbase allows limit<=300 with `end`; paging backwards with
            # `end` only. (Combining start+limit+end returns HTTP 400.)
            out: dict[int, Bar] = {}
            end = datetime.now(timezone.utc).replace(microsecond=0)
            while len(out) < candles:
                take = min(300, candles - len(out))
                url = (f"{COINBASE}/products/{symbol}/candles"
                       f"?granularity={gran}&limit={take}"
                       f"&end={int(end.timestamp())}")
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
                end = datetime.fromtimestamp(min(out), tz=timezone.utc) - timedelta(seconds=gran)
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
    _drop_forming_bar(bars, _GRANULARITY[timeframe])
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
    """One entry point the rest of the app uses."""
    if exchange == "coinbase":
        bars = coinbase_candles(symbol, timeframe, candles)
    elif exchange == "kraken":
        bars = kraken_candles(symbol, timeframe, candles)
    elif exchange == "synthetic":
        bars = synthetic_bars(candles, seed=seed, timeframe=timeframe)
    else:
        raise FeedError(f"unknown exchange {exchange!r}")
    try:
        save_bars(cache_path(cache_root, exchange, symbol, timeframe), bars)
    except OSError:
        pass  # caching is a nicety, never a failure mode
    return bars


def fingerprint(bars: list[Bar]) -> str:
    """Stable hash of a dataset, so a result can be tied to exact data."""
    h = hashlib.sha256()
    for b in bars:
        h.update(f"{b.ts.isoformat()},{b.open:.10f},{b.high:.10f},"
                 f"{b.low:.10f},{b.close:.10f},{b.volume:.10f};".encode())
    return h.hexdigest()[:16]
