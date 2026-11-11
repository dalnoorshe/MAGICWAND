"""MAGICWAND — indicators.

Small on purpose. Every function takes a plain list of floats and returns
a list of the same length, with None where the value is not yet defined.
That way it is impossible to accidentally read a value from the future:
index i only ever looks at indexes <= i.
"""

from __future__ import annotations

import math
from typing import Optional

from .common import Bar


def sma(xs: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(xs)
    if period <= 0:
        return out
    run = 0.0
    for i, x in enumerate(xs):
        run += x
        if i >= period:
            run -= xs[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def ema(xs: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(xs)
    if period <= 0 or len(xs) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(xs[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(xs)):
        prev = xs[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def true_range(bars: list[Bar]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(bars)
    for i, b in enumerate(bars):
        if i == 0:
            out[i] = b.high - b.low
        else:
            pc = bars[i - 1].close
            out[i] = max(b.high - b.low, abs(b.high - pc), abs(b.low - pc))
    return out


def atr(bars: list[Bar], period: int = 14) -> list[Optional[float]]:
    """Wilder's ATR, seeded with the simple mean of the first `period` TRs."""
    tr = true_range(bars)
    out: list[Optional[float]] = [None] * len(bars)
    if len(bars) < period or period <= 0:
        return out
    first = [x for x in tr[:period] if x is not None]
    if len(first) < period:
        return out
    prev = sum(first) / period
    out[period - 1] = prev
    for i in range(period, len(bars)):
        t = tr[i] if tr[i] is not None else 0.0
        prev = (prev * (period - 1) + t) / period
        out[i] = prev
    return out


def rsi(bars: list[Bar], period: int = 14) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, period + 1):
        d = bars[i].close - bars[i - 1].close
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / period, sum(losses) / period
    out[period] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(period + 1, len(bars)):
        d = bars[i].close - bars[i - 1].close
        ag = (ag * (period - 1) + max(d, 0.0)) / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def zscore(xs: list[float], period: int) -> list[Optional[float]]:
    """Rolling z-score of price vs its own mean — a mean-reversion signal."""
    out: list[Optional[float]] = [None] * len(xs)
    for i in range(period - 1, len(xs)):
        w = xs[i - period + 1: i + 1]
        m = sum(w) / period
        var = sum((x - m) ** 2 for x in w) / period
        sd = math.sqrt(var)
        out[i] = (xs[i] - m) / sd if sd > 0 else 0.0
    return out
