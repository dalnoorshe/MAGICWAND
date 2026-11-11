"""MAGICWAND — shared types and small statistics helpers.

Zero dependencies on purpose: every line here is auditable without
installing anything. Money code should be readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, pstdev
from typing import Optional


# --------------------------------------------------------------------------
# Enums / core value types
# --------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ExitReason(str, Enum):
    STOP = "stop"          # ATR stop hit (adverse)
    TARGET = "target"      # take-profit hit (favorable)
    TIME = "time"          # maximum holding period reached
    SIGNAL = "signal"      # strategy flipped / asked to exit
    KILL = "kill"          # risk engine or operator killed the position
    END = "end"            # backtest window ended mid-trade


@dataclass(frozen=True)
class Bar:
    """A single OHLCV candle. `ts` is the OPEN time, UTC, ISO-8601."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError("Bar.ts must be timezone-aware (UTC)")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"inconsistent OHLC bar at {self.ts}")
        if self.low <= 0 or self.high <= 0:
            raise ValueError(f"non-positive price at {self.ts}")


@dataclass(frozen=True)
class Order:
    """A request to trade. Never fills at this price — that's the broker's job."""
    symbol: str
    side: Side
    qty: float
    reason: str
    kind: str = "entry"  # entry | exit | stop | target
    limit: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    """What actually happened, warts included (fees + slippage)."""
    symbol: str
    side: Side
    qty: float
    price: float          # actual execution price, worse than the quote
    fee: float            # quote currency
    ts: datetime
    reason: str
    kind: str = "entry"


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_ts: datetime
    stop: Optional[float] = None
    target: Optional[float] = None
    entry_fee: float = 0.0
    bars_held: int = 0
    entry_reason: str = ""
    expected_move: Optional[float] = None  # the wand's *claim*, in fractional terms

    def notional(self, price: float) -> float:
        return self.qty * price

    def unrealized(self, price: float) -> float:
        if self.side is Side.BUY:
            return (price - self.entry_price) * self.qty
        return (self.entry_price - price) * self.qty


@dataclass
class Trade:
    """A completed round trip, fees included. This is the only honest unit."""
    symbol: str
    side: Side
    qty: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    fees: float
    reason: ExitReason
    entry_reason: str = ""
    expected_move: Optional[float] = None

    @property
    def gross(self) -> float:
        if self.side is Side.BUY:
            return (self.exit_price - self.entry_price) * self.qty
        return (self.entry_price - self.exit_price) * self.qty

    @property
    def net(self) -> float:
        return self.gross - self.fees

    @property
    def return_pct(self) -> float:
        """Net return on notional at entry (fractional)."""
        base = self.entry_price * self.qty
        return self.net / base if base else 0.0

    @property
    def holding_seconds(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Statistics — deliberately plain. No scipy, no black boxes.
# --------------------------------------------------------------------------

def safe_mean(xs: list[float]) -> float:
    return mean(xs) if xs else 0.0


def safe_std(xs: list[float]) -> float:
    return pstdev(xs) if len(xs) > 1 else 0.0


def stderr(xs: list[float]) -> float:
    """Standard error of the mean. Used to ask: is this edge real or noise?"""
    if len(xs) < 2:
        return math.inf
    return safe_std(xs) / math.sqrt(len(xs))


def t_stat(xs: list[float]) -> float:
    """How many standard errors the mean is away from zero.

    Rule of thumb: |t| < 2 over a real sample means you have not shown an edge.
    """
    se = stderr(xs)
    if se == 0 or se == math.inf:
        return 0.0
    return safe_mean(xs) / se


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0, 100]."""
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def max_drawdown(equity: list[float]) -> tuple[float, int, int]:
    """Peak-to-trough drawdown as a fraction, plus the indices involved."""
    peak = -math.inf
    peak_i = 0
    worst = 0.0
    worst_peak_i = worst_i = 0
    for i, v in enumerate(equity):
        if v > peak:
            peak, peak_i = v, i
        if peak > 0:
            dd = (peak - v) / peak
            if dd > worst:
                worst, worst_peak_i, worst_i = dd, peak_i, i
    return worst, worst_peak_i, worst_i


def summarize_returns(rs: list[float]) -> dict:
    """One honest paragraph's worth of numbers about a list of trade returns."""
    if not rs:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    return {
        "n": len(rs),
        "mean": safe_mean(rs),
        "median": percentile(rs, 50),
        "std": safe_std(rs),
        "win_rate": len(wins) / len(rs),
        "avg_win": safe_mean(wins),
        "avg_loss": safe_mean(losses),
        "best": max(rs),
        "worst": min(rs),
        "t_stat": t_stat(rs),
        "profit_factor": (sum(wins) / abs(sum(losses))) if sum(losses) else math.inf,
    }
