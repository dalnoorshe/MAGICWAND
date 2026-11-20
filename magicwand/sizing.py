"""MAGICWAND — sizing from a measured edge, not from confidence.

Every other part of this project asks "are we right?". This module asks the
question that actually decides whether money is made or lost: **how much?** — and
it answers it in the only way that survives contact with uncertainty.

The machinery is Kelly, with two corrections that matter:

  * the edge is taken from the LOWER confidence bound of its estimate, not the
    point estimate. A backtest that showed +2 bps per trade with a standard
    error of 8 bps is not a +2 bps edge; it is "somewhere between −14 and +18",
    and you bet on the part you can defend;
  * costs come off first. A 38 bps round trip does not care how confident you
    are.

When the lower bound minus costs is zero or negative, the size is **exactly
zero**. Not small — zero. That is the whole point: a system that cannot measure
its own edge does not get to have a position. This is why a strategy with no
signal sizes to nothing rather than to "a cautious 10%".

Deliberately conservative choices, stated so nobody "improves" them away:

  * half-Kelly. Full Kelly maximises growth if the edge is exactly known; it
    is brutal when the edge is estimated, because estimation error turns the
    bet into a coin flip you overpay for.
  * a hard cap on the fraction of equity, whatever Kelly says.
  * the estimate is per TRADE RETURN, so n is the number of independent trades,
    not the number of bars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Two standard errors is roughly a 97.5% one-sided bound. One is too brave for
# a quantity that decides position size.
DEFAULT_SIGMA = 2.0
# Never risk more than this fraction of equity on one trade, whatever Kelly says.
MAX_FRACTION = 0.10
# Half-Kelly: growth is close to optimal, and the drawdowns are survivable.
KELLY_HAIRCUT = 0.5


@dataclass(frozen=True)
class EdgeEstimate:
    """A measured per-trade edge, with how well it is measured."""

    mean_bps: float          # average return per trade, in bps of notional
    sd_bps: float            # standard deviation of per-trade returns
    n: int                   # number of INDEPENDENT trades
    label: str = ""

    @property
    def se_bps(self) -> float:
        if self.n <= 0 or not math.isfinite(self.sd_bps):
            return float("inf")
        return self.sd_bps / math.sqrt(self.n)

    def lower_bound_bps(self, sigma: float = DEFAULT_SIGMA) -> float:
        """The pessimistic edge: what we can defend, not what we hope."""
        if not math.isfinite(self.mean_bps) or not math.isfinite(self.se_bps):
            return float("-inf")
        return self.mean_bps - sigma * self.se_bps

    def upper_bound_bps(self, sigma: float = DEFAULT_SIGMA) -> float:
        if not math.isfinite(self.mean_bps) or not math.isfinite(self.se_bps):
            return float("inf")
        return self.mean_bps + sigma * self.se_bps

    def to_dict(self) -> dict:
        return {"mean_bps": round(self.mean_bps, 3),
                "sd_bps": round(self.sd_bps, 3), "n": self.n,
                "se_bps": round(self.se_bps, 3) if math.isfinite(self.se_bps) else None,
                "lower_bps": (round(self.lower_bound_bps(), 3)
                              if math.isfinite(self.lower_bound_bps()) else None),
                "label": self.label}


@dataclass(frozen=True)
class Sizing:
    """A decision, with the arithmetic that produced it."""

    fraction: float          # of equity, 0 .. MAX_FRACTION
    ok: bool
    reason: str
    lower_bps: float
    per_trade_cost_bps: float

    def to_dict(self) -> dict:
        return {"fraction": round(self.fraction, 6), "ok": self.ok,
                "reason": self.reason, "lower_bps": round(self.lower_bps, 3),
                "per_trade_cost_bps": round(self.per_trade_cost_bps, 3)}


def size_from_edge(edge: Optional[EdgeEstimate], per_trade_cost_bps: float,
                   sigma: float = DEFAULT_SIGMA,
                   max_fraction: float = MAX_FRACTION) -> Sizing:
    """Kelly on the pessimistic edge, net of costs. Zero when unproven."""
    if edge is None:
        return Sizing(0.0, False,
                      "no edge estimate: refusing to size on confidence alone",
                      0.0, per_trade_cost_bps)
    if edge.n <= 0:
        return Sizing(0.0, False, "no trades to estimate an edge from",
                      0.0, per_trade_cost_bps)
    if not math.isfinite(edge.mean_bps) or not math.isfinite(edge.sd_bps):
        return Sizing(0.0, False, "non-finite edge estimate", 0.0,
                      per_trade_cost_bps)
    lower = edge.lower_bound_bps(sigma)
    net = lower - per_trade_cost_bps
    if net <= 0:
        return Sizing(
            0.0, False,
            f"no bet: the conservative edge ({lower:+.1f} bps) does not cover "
            f"costs ({per_trade_cost_bps:.1f} bps). Measured {edge.mean_bps:+.1f} "
            f"± {edge.se_bps:.1f} bps over {edge.n} trades.",
            lower, per_trade_cost_bps)
    sigma_frac = edge.sd_bps / 10_000.0
    if sigma_frac <= 0:
        return Sizing(0.0, False, "zero variance: an edge cannot be inferred",
                      lower, per_trade_cost_bps)
    mu = net / 10_000.0
    raw = mu / (sigma_frac ** 2)                  # Kelly, continuous returns
    fraction = min(max_fraction, max(0.0, raw * KELLY_HAIRCUT))
    if fraction <= 0:
        return Sizing(0.0, False, "Kelly says zero", lower, per_trade_cost_bps)
    return Sizing(
        fraction, True,
        f"half-Kelly on the conservative edge: {fraction:.2%} of equity "
        f"(net {net:+.1f} bps/trade, σ {edge.sd_bps:.1f} bps, n={edge.n})",
        lower, per_trade_cost_bps)


def edge_from_returns(returns_bps: list[float], label: str = "") -> EdgeEstimate:
    """Build an estimate from per-trade returns. Sample sd, n = trade count.

    A single trade is not an estimate, and this says so rather than dividing
    by zero: n=1 gives an infinite standard error, which sizes to nothing.
    """
    values = [float(r) for r in returns_bps if math.isfinite(float(r))]
    n = len(values)
    if n == 0:
        return EdgeEstimate(0.0, float("inf"), 0, label)
    mean = sum(values) / n
    if n == 1:
        return EdgeEstimate(mean, float("inf"), 1, label)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return EdgeEstimate(mean, math.sqrt(var), n, label)


def edge_from_trades(trades, label: str = "") -> EdgeEstimate:
    """Per-trade edge in bps from backtest trades.

    `Trade.return_pct` is FRACTIONAL (0.01 = 1%), so the conversion is 10,000,
    not 100. Getting this wrong understates a losing strategy by a factor of a
    hundred — it made a −114 bps/trade result look like −1.14 bps and would have
    argued for a position.
    """
    return edge_from_returns([t.return_pct * 10_000.0 for t in trades], label)
