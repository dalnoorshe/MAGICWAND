"""MAGICWAND — the risk engine.

This module is deliberately boring and deliberately stubborn.

The strategy proposes. The risk engine disposes. It can shrink an order,
refuse an order, or flatten the book, and it always says why in plain
English so the ledger records a reason rather than a mystery.

Rules enforced here (all hard, none overridable by a strategy):
  1. risk-per-trade ceiling
  2. max position size as a fraction of equity
  3. min notional (below exchange minimum, skip — do not round up)
  4. max open positions
  5. max daily loss — stops opening new trades for the day
  6. kill switch — flattens and blocks everything until reset
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from .common import Position, Side
from .config import RiskConfig


@dataclass
class RiskDecision:
    allowed: bool
    qty: float = 0.0
    reason: str = ""


@dataclass
class RiskState:
    day: Optional[date]          # None = the engine has not seen equity yet
    day_start_equity: float
    realized_today: float = 0.0
    halted_today: bool = False
    halt_reason: str = ""


class RiskEngine:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        # Deliberately uninitialised: the day's baseline must come from the
        # first real equity observation, not from a config default at import
        # time. Seeding it early would silently neuter the daily loss limit.
        self.state = RiskState(day=None, day_start_equity=cfg.equity)

    # ---- daily bookkeeping ----

    def roll_day(self, today: date, equity_now: float) -> None:
        """Start a new day only once. Never rewrite an existing day's baseline.

        Getting this wrong silently disables the daily loss limit, which is
        the single most important rule in the engine — so it is set here,
        once, from the first observed equity of the day.
        """
        if today != self.state.day:
            self.state = RiskState(day=today, day_start_equity=equity_now)

    def update_equity(self, equity_now: float, when: datetime) -> None:
        """Mark-to-market observation. Call this every bar / every tick.

        Without this, the daily loss limit only sees equity at trade close,
        so a position bleeding out intraday would never trip it.
        """
        self.roll_day(when.date(), equity_now)
        self._check_daily_loss(equity_now, when)

    def record_realized(self, pnl: float, equity_now: float, when: datetime) -> None:
        # The day's baseline is equity *before* this trade's result.
        self.roll_day(when.date(), equity_now - pnl)
        self.state.realized_today += pnl
        self._check_daily_loss(equity_now, when)

    def _check_daily_loss(self, equity_now: float, when: datetime) -> None:
        if self.state.day_start_equity <= 0:
            return
        drawdown = (self.state.day_start_equity - equity_now) / self.state.day_start_equity
        if drawdown >= self.cfg.max_daily_loss_pct and not self.state.halted_today:
            self.state.halted_today = True
            self.state.halt_reason = (
                f"daily loss limit hit: {drawdown:.2%} down on the day "
                f"(limit {self.cfg.max_daily_loss_pct:.2%}). No new trades today."
            )

    # ---- the gate every entry must pass ----

    def size_entry(self, equity: float, price: float, stop_distance: float,
                   open_positions: int, when: datetime,
                   edge=None, per_trade_cost_bps: float = 0.0) -> RiskDecision:
        """Sizing with an optional measured edge.

        Without `edge` the engine sizes off the fixed ceilings alone — which is
        what every version of this project did until now, and the reason those
        ceilings had to be conservative: they had to stand in for evidence. With
        an edge, the position is capped by Kelly on its PESSIMISTIC bound, so a
        strategy that cannot demonstrate an edge sizes to zero instead of to a
        cautious fraction.
        """
        cfg = self.cfg
        if cfg.kill_switch:
            return RiskDecision(False, 0.0, "kill switch is enabled — all trading blocked")
        self.roll_day(when.date(), equity)
        if self.state.halted_today:
            return RiskDecision(False, 0.0, self.state.halt_reason)
        if open_positions >= cfg.max_open_positions:
            return RiskDecision(
                False, 0.0,
                f"max open positions reached ({open_positions}/{cfg.max_open_positions})",
            )
        if not (math.isfinite(equity) and math.isfinite(price)
                and math.isfinite(stop_distance)):
            return RiskDecision(False, 0.0,
                                "non-finite price/equity/stop; refusing to size")
        if price <= 0 or stop_distance <= 0:
            return RiskDecision(False, 0.0, "invalid price or stop distance; refusing to size")

        # Rule 1: never risk more than risk_per_trade of equity.
        risk_budget = equity * cfg.risk_per_trade
        qty_risk = risk_budget / stop_distance

        # Rule 2: never hold more than max_position_pct of equity.
        qty_cap = (equity * cfg.max_position_pct) / price

        qty = min(qty_risk, qty_cap)
        binding = "risk-per-trade" if qty_risk <= qty_cap else "max-position-size"

        # Rule 0: if an edge has been measured, it CAPS the size. A strategy
        # with no demonstrable edge must not open a position on the strength of
        # a fixed ceiling alone.
        edge_note = ""
        if edge is not None:
            from .sizing import size_from_edge
            sizing = size_from_edge(edge, per_trade_cost_bps)
            if not sizing.ok:
                return RiskDecision(False, 0.0, sizing.reason)
            qty_edge = equity * sizing.fraction / price
            if qty_edge < qty:
                qty = qty_edge
                binding = "kelly-on-conservative-edge"
            edge_note = f"; {sizing.reason}"
        notional = qty * price

        # Rule 3: below exchange minimum, skip. Never silently round up.
        if notional < cfg.min_notional:
            return RiskDecision(
                False, 0.0,
                f"trade too small to place: {notional:.2f} < exchange minimum "
                f"{cfg.min_notional:.2f} (binding: {binding}). Account is too small "
                f"for this stop distance.",
            )
        if equity <= 0:
            return RiskDecision(False, 0.0, "equity is zero or negative")

        return RiskDecision(
            True, qty,
            f"size {qty:.6f} (~{qty * price:.2f}) — risk {qty * stop_distance:.2f} "
            f"({(qty * stop_distance) / equity:.2%} of equity), binding: {binding}"
            + edge_note,
        )

    def can_trade(self, when: datetime, equity_now: float) -> RiskDecision:
        """For exits we are more permissive — but the kill switch still wins."""
        if self.cfg.kill_switch:
            return RiskDecision(False, 0.0, "kill switch is enabled")
        self.roll_day(when.date(), equity_now)
        if self.state.halted_today:
            return RiskDecision(True, 0.0, "daily loss halt: exits allowed, entries blocked")
        return RiskDecision(True, 0.0, "ok")
