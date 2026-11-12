"""MAGICWAND — paper broker and portfolio.

A paper broker is only useful if it is *pessimistic*. This one:
  * fills market orders worse than the quoted price (slippage)
  * charges a fee on both sides
  * refuses orders below a minimum notional
  * never has access to real funds

There is no live broker in this build. That is a feature, not an omission:
nothing here can move real money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .common import Bar, ExitReason, Fill, Order, Position, Side, Trade
from .config import CostConfig, RiskConfig


class BrokerError(RuntimeError):
    pass


@dataclass
class Portfolio:
    cash: float
    position: Optional[Position] = None
    equity: float = 0.0
    peak_equity: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)

    def mark(self, price: float) -> float:
        """Equity = cash + the position's value at market.

        Cash was reduced by the full notional at entry, so the position's
        cost basis has to be added back before marking to market. Omitting
        that (only adding unrealized P&L) deletes the cost basis every tick
        and fabricates catastrophic equity drops — a bug this project had,
        caught by test_equity_is_continuous_while_holding.
        """
        pos = self.position
        eq = self.cash
        if pos is not None:
            eq += pos.entry_price * pos.qty + pos.unrealized(price)
        self.equity = eq
        self.peak_equity = max(self.peak_equity, eq)
        return eq


class PaperBroker:
    def __init__(self, costs: CostConfig, risk_cfg: RiskConfig) -> None:
        self.costs = costs
        self.risk = risk_cfg

    # ---- pricing ----

    def _fill_price(self, side: Side, quote: float) -> float:
        slip = quote * (self.costs.slippage_bps / 10_000.0)
        return quote + slip if side is Side.BUY else quote - slip

    def _fee(self, notional: float) -> float:
        return notional * (self.costs.fee_bps / 10_000.0)

    # ---- execution ----

    def buy(self, pf: Portfolio, quote: float, qty: float, ts: datetime,
            reason: str, kind: str = "entry",
            stop: Optional[float] = None, target: Optional[float] = None,
            expected_move: Optional[float] = None,
            max_hold_bars: int = 0) -> Optional[Fill]:
        if qty <= 0:
            raise BrokerError("qty must be positive")
        price = self._fill_price(Side.BUY, quote)
        notional = price * qty
        fee = self._fee(notional)
        # Slippage can move the notional a hair below the minimum; only reject
        # when it is genuinely undersized. Callers size off the worst-case fill.
        if notional < self.risk.min_notional * 0.98:
            return None  # risk engine should have caught this; broker refuses too
        if notional + fee > pf.cash + 1e-9:
            raise BrokerError(
                f"insufficient paper cash: need {notional + fee:.2f}, have {pf.cash:.2f}"
            )
        pf.cash -= notional + fee
        pf.position = Position(
            symbol="", side=Side.BUY, qty=qty, entry_price=price, entry_ts=ts,
            stop=stop, target=target, entry_fee=fee, entry_reason=reason,
            expected_move=expected_move,
        )
        fill = Fill("", Side.BUY, qty, price, fee, ts, reason, kind)
        pf.fills.append(fill)
        return fill

    def sell(self, pf: Portfolio, quote: float, ts: datetime, reason: str,
             kind: str = "exit") -> Trade:
        pos = pf.position
        if pos is None:
            raise BrokerError("sell with no open position")
        price = self._fill_price(Side.SELL, quote)
        notional = price * pos.qty
        fee = self._fee(notional)
        pf.cash += notional - fee
        try:
            reason_enum = ExitReason(kind)
        except ValueError:
            reason_enum = ExitReason.SIGNAL
        trade = Trade(
            symbol=pos.symbol, side=pos.side, qty=pos.qty,
            entry_ts=pos.entry_ts, entry_price=pos.entry_price,
            exit_ts=ts, exit_price=price, fees=pos.entry_fee + fee,
            reason=reason_enum, entry_reason=pos.entry_reason,
            expected_move=pos.expected_move,
        )
        pf.trades.append(trade)
        pf.fills.append(Fill(pos.symbol, Side.SELL, pos.qty, price, fee, ts, reason, kind))
        pf.position = None
        return trade


# --------------------------------------------------------------------------
# Exits: stops and targets are checked against the bar's own range, not
# just the close, because in reality they would have triggered intrabar.
# --------------------------------------------------------------------------

@dataclass
class ExitCheck:
    reason: ExitReason
    price: float
    note: str


def check_exit(pos: Position, bar: Bar) -> Optional[ExitCheck]:
    """Did this bar hit the stop or the target? Stop wins ties (worse for us)."""
    if pos.side is Side.BUY:
        if pos.stop is not None and bar.low <= pos.stop:
            return ExitCheck(ExitReason.STOP, pos.stop,
                             f"stop hit intrabar (low {bar.low:.4f} <= stop {pos.stop:.4f})")
        if pos.target is not None and bar.high >= pos.target:
            return ExitCheck(ExitReason.TARGET, pos.target,
                             f"target hit intrabar (high {bar.high:.4f} >= target {pos.target:.4f})")
    return None
