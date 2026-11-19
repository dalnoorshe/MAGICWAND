"""Extra causal features: cross-asset, session and funding.

The trend model has always been price-only: nine numbers squeezed out of one
symbol's candles. That throws away most of what moves a crypto bar — BTC leads
alts, funding says how crowded the leverage is, and hour-of-day says who is
awake. This module adds those, under the same rule as everything else:

**a feature at bar i may only use information that existed at bar i.**

That is why every series is aligned by TIMESTAMP rather than by position, why
funding is read backwards (the last rate published at or before the bar), and
why a missing counterpart bar returns None instead of a guess. A feature that
cannot be computed honestly is dropped, not filled.

Groups:

    price    the nine existing price features (see forecast.features_at)
    cross    BTC/ETH returns, BTC-vs-SOL relative strength, rolling correlation
    session  hour-of-day, on a circle
    funding  perpetual funding rate, its z-score, and its change
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import data as data_mod
from .common import Bar, iso
from .markets import binance_perp, binance_spot


CROSS_FEATURES = ("btc_ret6", "btc_ret24", "eth_ret6", "sol_vs_btc_24",
                  "btc_sol_corr48")
SESSION_FEATURES = ("hour_sin", "hour_cos")
FUNDING_FEATURES = ("funding_rate", "funding_z48", "funding_change")

GROUPS: dict[str, tuple[str, ...]] = {
    "cross": CROSS_FEATURES,
    "session": SESSION_FEATURES,
    "funding": FUNDING_FEATURES,
}


def _epoch(bars: list[Bar], i: int) -> int:
    return int(bars[i].ts.replace(tzinfo=timezone.utc).timestamp())


def load_series(symbol: str, timeframe: str, bars: list[Bar]
                ) -> dict[int, float]:
    """Close prices keyed by bar-open epoch, for cross-asset alignment.

    Keyed by timestamp rather than index because two venues do not always
    produce the same number of bars: a missing hour must not shift every
    feature after it.
    """
    if not bars:
        return {}
    out: dict[int, float] = {}
    for b in bars:
        out[int(b.ts.replace(tzinfo=timezone.utc).timestamp())] = b.close
    return out


def load_closes(symbol: str, timeframe: str, count: int) -> dict[int, float]:
    pair = binance_spot(symbol)
    if not pair:
        return {}
    try:
        bars = data_mod.binance_candles(pair, timeframe, count)
    except Exception:  # noqa: BLE001 — a missing series is not fatal
        return {}
    return load_series(symbol, timeframe, bars)


def load_funding(symbol: str, count: int = 2000) -> list[tuple[int, float]]:
    """Funding rate history as (epoch_ms, rate), oldest first.

    Paginated: 1000 records per call, 8h apart, so ~330 days per request.
    Returns [] rather than raising when the venue is unreachable.
    """
    from .research import SourceError, _get_json

    pair = binance_perp(symbol)
    if not pair:
        return []
    out: list[tuple[int, float]] = []
    start = 0
    try:
        while len(out) < count:
            url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}"
                   f"&limit=1000" + (f"&startTime={start}" if start else ""))
            page = _get_json(url)
            if not page:
                break
            for row in page:
                t = int(row["fundingTime"])
                out.append((t, float(row["fundingRate"])))
            if len(page) < 1000:
                break
            start = int(page[-1]["fundingTime"]) + 1
    except (SourceError, ValueError, KeyError, TypeError):
        return sorted(set(out))
    # De-duplicate by timestamp; the venue can repeat an edge record.
    return sorted(dict(out).items())


@dataclass
class Context:
    """Everything a feature needs, fetched once and aligned by timestamp.

    `epochs` and `sol` are precomputed because the alternative — rebuilding the
    timestamp list inside every per-bar feature call — is O(n^2) and turns a
    4,000-bar evaluation from seconds into minutes. That exact mistake cost this
    project 45 seconds per backtest once already.
    """

    epochs: list[int] = field(default_factory=list)
    sol: dict[int, float] = field(default_factory=dict)
    btc: dict[int, float] = field(default_factory=dict)
    eth: dict[int, float] = field(default_factory=dict)
    funding: list[tuple[int, float]] = field(default_factory=list)
    _funding_ts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._funding_ts = [t for t, _ in self.funding]

    def set_funding(self, rows: list[tuple[int, float]]) -> None:
        self.funding = rows
        self._funding_ts = [t for t, _ in rows]

    def funding_at(self, epoch_ms: int) -> Optional[float]:
        """The last rate PUBLISHED at or before this bar. Never the next one."""
        # Rebuild if funding was assigned after construction. Without this the
        # index stayed empty, every funding feature silently vanished, and the
        # "does funding help?" experiment would have measured nothing while
        # reporting a clean answer. Caught by printing the features.
        if len(self._funding_ts) != len(self.funding):
            self._funding_ts = [t for t, _ in self.funding]
        if not self._funding_ts:
            return None
        k = bisect_right(self._funding_ts, epoch_ms) - 1
        return self.funding[k][1] if k >= 0 else None


def build_context(symbol: str, timeframe: str, bars: list[Bar],
                  groups: tuple[str, ...]) -> Context:
    ctx = Context(epochs=[_epoch(bars, i) for i in range(len(bars))],
                  sol=load_series(symbol, timeframe, bars))
    if "cross" in groups:
        ctx.btc = load_closes("BTC-USD", timeframe, len(bars))
        ctx.eth = load_closes("ETH-USD", timeframe, len(bars))
    if "funding" in groups:
        ctx.set_funding(load_funding(symbol))
    return ctx


def _ret(series: dict[int, float], epochs: list[int], i: int,
         nbars: int) -> Optional[float]:
    """Return over `nbars` hourly bars, by timestamp, or None if either is absent."""
    if i - nbars < 0:
        return None
    a = series.get(epochs[i])
    b = series.get(epochs[i - nbars])
    if not a or not b:
        return None
    return a / b - 1.0


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / math.sqrt(vx * vy)


def extras(bars: list[Bar], i: int, ctx: Context,
           groups: tuple[str, ...]) -> dict[str, float]:
    """Extra features for bar i. Omitted keys mean "could not be computed"."""
    out: dict[str, float] = {}
    epochs = ctx.epochs or [_epoch(bars, k) for k in range(len(bars))]

    if "cross" in groups:
        b6 = _ret(ctx.btc, epochs, i, 6)
        b24 = _ret(ctx.btc, epochs, i, 24)
        e6 = _ret(ctx.eth, epochs, i, 6)
        s24 = _ret(ctx.sol or {epochs[k]: bars[k].close
                              for k in range(len(bars))}, epochs, i, 24)
        if b6 is not None:
            out["btc_ret6"] = b6
        if b24 is not None:
            out["btc_ret24"] = b24
        if e6 is not None:
            out["eth_ret6"] = e6
        if b24 is not None and s24 is not None:
            out["sol_vs_btc_24"] = s24 - b24
        if i >= 48:
            sr, br = [], []
            for k in range(i - 47, i + 1):
                pa, pb = bars[k - 1].close, bars[k].close
                ba, bb = ctx.btc.get(epochs[k - 1]), ctx.btc.get(epochs[k])
                if not (pa and pb and ba and bb):
                    break
                sr.append(pb / pa - 1.0)
                br.append(bb / ba - 1.0)
            if len(sr) >= 24:
                out["btc_sol_corr48"] = _corr(sr, br)

    if "session" in groups:
        hour = bars[i].ts.hour
        out["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        out["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)

    if "funding" in groups:
        rate = ctx.funding_at(epochs[i] * 1000)
        if rate is not None:
            out["funding_rate"] = rate
            # Z-score against the last 48 PUBLISHED rates (8h apart), all of
            # which are in the past relative to this bar.
            prior = [r for t, r in ctx.funding
                     if t <= epochs[i] * 1000][-48:]
            if len(prior) >= 12:
                m = sum(prior) / len(prior)
                var = sum((r - m) ** 2 for r in prior) / len(prior)
                sd = math.sqrt(var)
                out["funding_z48"] = (rate - m) / sd if sd > 0 else 0.0
            if len(prior) >= 2:
                out["funding_change"] = prior[-1] - prior[-2]
    return out
