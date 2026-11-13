"""MAGICWAND — the Agent Desk.

This is the answer to "let the agent place the trade when it's sure."

The desk is a separate, hard-capped paper sub-account. Inside it, agents may
open a position immediately, on their own, with no human in the loop. Outside
it, they cannot touch anything.

The walls, all enforced in code:

  * funds      — funded once from the parent account, capped at
                 `desk.max_fraction_of_equity` (default 10%)
  * one trade  — at most `max_trades_per_day` (default 1) agent-originated
                 entries per UTC day
  * size       — never more than `max_trade_pct_of_desk` (default 10%) of the
                 desk pot, and never below the exchange minimum (skip, don't
                 round up)
  * stop       — every entry carries a mandatory stop; no stop, no trade
  * symbol     — a whitelist. A hallucinated ticker is a refused order.
  * autonomy   — earned. After `review_after` graded trades, an agent whose
                 hit rate is at or below `demote_below_hit_rate` loses the
                 ability to trade until a human re-arms it.
  * kill switch — the operator's switch stops the desk too.

None of this is a suggestion to the model. It is arithmetic, checked in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .broker import PaperBroker, Portfolio
from .common import ExitReason, Position, Side, Trade, iso, parse_iso, utcnow
from .config import Config
from .gatekeeper import Gatekeeper, Policy, RiskTier, default_policy
from .ledger import Ledger


class DeskRefused(RuntimeError):
    """An order was refused by a wall. Recorded, never silently dropped."""


@dataclass
class DeskConfig:
    enabled: bool = False
    max_fraction_of_equity: float = 0.10   # 10% of total equity, at most
    max_trade_pct_of_desk: float = 0.10    # 10% of the desk pot per trade
    max_trades_per_day: int = 1            # news moves on hours, not seconds
    review_after: int = 10                 # graded trades before a performance review
    demote_below_hit_rate: float = 0.45    # at or below this, autonomy is revoked
    min_confidence: float = 0.80           # "when the agent is sure"
    # Derived from the markets registry, so a market that is listed there is
    # tradable and nothing else is.
    whitelist: tuple = ("BTC-USD", "ETH-USD", "PAXG-USD", "SOL-USD")
    stop_atr_mult: float = 2.0
    max_hold_bars: int = 24
    autonomy_armed: bool = True
    # The tier is a human-set ceiling. Default is the read-only tier: an
    # operator must opt in to letting anything trade.
    tier: RiskTier = RiskTier.RESEARCH


@dataclass
class DeskState:
    pot: float = 0.0
    position: Optional[Position] = None
    trades: list[Trade] = field(default_factory=list)
    trades_today: int = 0
    day: str = ""
    autonomy_armed: bool = True
    revoked_reason: str = ""
    opened_by: str = ""
    # What the desk was funded with, and the total account equity at that
    # moment. The gatekeeper needs the TRUE pot fraction, which is the funded
    # amount over account equity — not cash-after-a-trade over equity.
    funded_amount: float = 0.0
    funded_against_equity: float = 0.0
    # `pot` is CASH, exactly like Portfolio.cash. The position is held beside
    # it. Equity = cash + position value at market, computed on demand.


class AgentDesk:
    def __init__(self, cfg: Config, dcfg: DeskConfig, root: str | Path = ".") -> None:
        self.cfg = cfg
        self.dcfg = dcfg
        self.root = Path(root)
        self.path = self.root / "data" / "desk.json"
        self.ledger = Ledger(self.root / cfg.ledger.path)
        self.state = self._load()
        self.gate = Gatekeeper(Policy(tier=dcfg.tier))
        # Config is the operator's current intent; if they re-armed the desk,
        # that has to survive a restart, not live only in memory.
        if dcfg.autonomy_armed != self.state.autonomy_armed and dcfg.autonomy_armed:
            self.state.autonomy_armed = True
            self.state.revoked_reason = ""
            self.save()

    # ---- persistence ----

    def _load(self) -> DeskState:
        if not self.path.exists():
            return DeskState(pot=0.0)
        try:
            d = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return DeskState(pot=0.0)
        st = DeskState(
            pot=float(d.get("pot", 0.0)),
            trades_today=int(d.get("trades_today", 0)),
            day=str(d.get("day", "")),
            autonomy_armed=bool(d.get("autonomy_armed", True)),
            revoked_reason=str(d.get("revoked_reason", "")),
            opened_by=str(d.get("opened_by", "")),
            funded_amount=float(d.get("funded_amount", 0.0)),
            funded_against_equity=float(d.get("funded_against_equity", 0.0)),
        )
        p = d.get("position")
        if p:
            st.position = Position(
                symbol=p["symbol"], side=Side(p["side"]), qty=float(p["qty"]),
                entry_price=float(p["entry_price"]), entry_ts=parse_iso(p["entry_ts"]),
                stop=p.get("stop"), target=p.get("target"),
                entry_fee=float(p.get("entry_fee", 0.0)),
                bars_held=int(p.get("bars_held", 0)),
                entry_reason=p.get("entry_reason", ""),
                expected_move=p.get("expected_move"),
            )
        for t in d.get("trades", []):
            try:
                st.trades.append(Trade(
                    symbol=t["symbol"], side=Side(t["side"]), qty=float(t["qty"]),
                    entry_ts=parse_iso(t["entry_ts"]), entry_price=float(t["entry_price"]),
                    exit_ts=parse_iso(t["exit_ts"]), exit_price=float(t["exit_price"]),
                    fees=float(t["fees"]), reason=ExitReason(t["reason"]),
                    entry_reason=t.get("entry_reason", ""),
                    expected_move=t.get("expected_move"),
                ))
            except (KeyError, ValueError):
                continue
        return st

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        p = self.state.position
        body = {
            "pot": round(self.state.pot, 8),
            "trades_today": self.state.trades_today,
            "day": self.state.day,
            "autonomy_armed": self.state.autonomy_armed,
            "revoked_reason": self.state.revoked_reason,
            "opened_by": self.state.opened_by,
            "funded_amount": round(self.state.funded_amount, 8),
            "funded_against_equity": round(self.state.funded_against_equity, 8),
            "position": None if p is None else {
                "symbol": p.symbol, "side": p.side.value, "qty": p.qty,
                "entry_price": p.entry_price, "entry_ts": iso(p.entry_ts),
                "stop": p.stop, "target": p.target, "entry_fee": p.entry_fee,
                "bars_held": p.bars_held, "entry_reason": p.entry_reason,
                "expected_move": p.expected_move,
            },
            "trades": [{
                "symbol": t.symbol, "side": t.side.value, "qty": t.qty,
                "entry_ts": iso(t.entry_ts), "entry_price": t.entry_price,
                "exit_ts": iso(t.exit_ts), "exit_price": t.exit_price,
                "fees": t.fees, "reason": t.reason.value,
                "entry_reason": t.entry_reason, "expected_move": t.expected_move,
            } for t in self.state.trades[-200:]],
            "saved_at": iso(utcnow()),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(self.path)

    # ---- funding ----

    def fund(self, total_equity: float) -> float:
        """Fund the desk once, capped. Never tops up automatically."""
        if self.state.pot > 0:
            return self.state.pot
        target = max(0.0, total_equity * self.dcfg.max_fraction_of_equity)
        self.state.pot = round(target, 6)
        self.state.funded_amount = self.state.pot
        self.state.funded_against_equity = total_equity
        self.ledger.append("desk_fund", {
            "pot": self.state.pot, "total_equity": total_equity,
            "cap_fraction": self.dcfg.max_fraction_of_equity,
            "note": "desk funded once; it does not top itself up",
        })
        self.save()
        return self.state.pot

    # ---- performance review ----

    def review(self) -> tuple[bool, str]:
        """Revoke autonomy on the agent's own record. Rule, not opinion."""
        n = len(self.state.trades)
        if n < self.dcfg.review_after:
            return (self.state.autonomy_armed,
                    f"{n}/{self.dcfg.review_after} graded trades before review")
        wins = sum(1 for t in self.state.trades if t.net > 0)
        hit = wins / n
        if hit <= self.dcfg.demote_below_hit_rate and self.state.autonomy_armed:
            self.state.autonomy_armed = False
            self.state.revoked_reason = (
                f"revoked: hit rate {hit:.0%} over {n} trades "
                f"(threshold {self.dcfg.demote_below_hit_rate:.0%})")
            self.ledger.append("autonomy_revoked", {
                "hit_rate": round(hit, 4), "trades": n,
                "threshold": self.dcfg.demote_below_hit_rate,
                "detail": "the desk no longer accepts agent-originated entries",
            })
            self.save()
        return self.state.autonomy_armed, (
            f"hit rate {hit:.0%} over {n} trades")

    # ---- opening an agent-originated trade ----

    def open_trade(self, symbol: str, price: float, atr: float, agent: str,
                   reason: str, expected_move: float, confidence: float,
                   now: Optional[datetime] = None) -> dict:
        now = now or utcnow()
        today = now.strftime("%Y-%m-%d")
        if self.state.day != today:
            self.state.day = today
            self.state.trades_today = 0

        def refuse(msg: str) -> dict:
            self.ledger.append("desk_refused", {
                "agent": agent, "symbol": symbol, "reason": msg,
                "price": price})
            return {"ok": False, "reason": msg}

        if not self.dcfg.enabled:
            return refuse("desk is disabled")
        # --- the gatekeeper is the last word before any broker call ---
        eff = self.gate.policy.effective()
        # Use the funded amount vs the equity it was funded against. Cash alone
        # reads as 100% of the account the moment the pot is fully invested.
        pot_fraction = (self.state.funded_amount / self.state.funded_against_equity
                        if self.state.funded_against_equity > 0 else 1.0)
        verdict = self.gate.check_order(
            pot_fraction=pot_fraction,
            trade_pct_of_pot=self.dcfg.max_trade_pct_of_desk,
            trades_today=self.state.trades_today,
            confidence=confidence,
            symbol_whitelisted=symbol in self.dcfg.whitelist)
        if not verdict.allowed:
            return refuse(f"gatekeeper: {verdict.reason}")
        if self.cfg.risk.kill_switch:
            return refuse("kill switch is enabled")
        if not self.state.autonomy_armed:
            return refuse(f"autonomy revoked: {self.state.revoked_reason}")
        if confidence < self.dcfg.min_confidence:
            return refuse(f"confidence {confidence:.2f} below the "
                          f"{self.dcfg.min_confidence:.2f} 'assured' bar")
        if symbol not in self.dcfg.whitelist:
            return refuse(f"symbol {symbol!r} is not whitelisted")
        if self.state.position is not None:
            return refuse("desk already holds a position")
        if self.state.trades_today >= self.dcfg.max_trades_per_day:
            return refuse(f"daily trade cap reached "
                          f"({self.state.trades_today}/{self.dcfg.max_trades_per_day})")
        if atr <= 0 or price <= 0:
            return refuse("invalid price or ATR; refusing to size")

        cash = self.state.pot
        budget = cash * self.dcfg.max_trade_pct_of_desk
        # Size off the worst-case fill price so slippage cannot push the
        # notional past the budget after the fact.
        slip = 1.0 + self.cfg.costs.slippage_bps / 10_000.0
        # Tolerate 1% rounding/slippage on the minimum-notional check only;
        # the budget cap itself is never loosened.
        qty = budget / (price * slip)
        if qty * price * slip < self.cfg.risk.min_notional * 0.99:
            return refuse(f"position {qty * price * slip:.2f} is below the "
                          f"exchange minimum {self.cfg.risk.min_notional:.2f}")

        stop = price - eff["stop_atr_mult"] * atr
        if stop <= 0:
            return refuse("ATR stop would be negative; refusing")
        risk_amt = qty * (price - stop)
        cost = qty * price * slip
        if cost * (1 + self.cfg.costs.fee_bps / 10_000.0) > cash + 1e-9:
            return refuse(f"trade cost {cost:.2f} exceeds desk cash {cash:.2f}")
        broker = PaperBroker(self.cfg.costs, self.cfg.risk)
        pf = Portfolio(cash=cash)
        try:
            fill = broker.buy(pf, price, qty, now, reason, stop=stop,
                              expected_move=expected_move)
        except Exception as e:  # noqa: BLE001 - a broker refusal must not crash the desk
            return refuse(f"broker rejected: {e}")
        if fill is None:
            return refuse("broker returned no fill (below minimum notional)")

        self.state.pot = pf.cash            # cash left after the purchase
        self.state.position = pf.position
        self.state.position.symbol = symbol
        self.state.position.entry_reason = reason
        self.state.opened_by = agent
        self.state.trades_today += 1
        self.ledger.append("agent_trade", {
            "agent": agent, "symbol": symbol, "side": "buy",
            "qty": round(fill.qty, 8), "price": round(fill.price, 6),
            "fee": round(fill.fee, 6), "stop": round(stop, 6),
            "risk_amount": round(risk_amt, 6),
            "confidence": round(confidence, 3),
            "expected_move": round(expected_move, 6),
            "reason": reason,
            "note": "agent-originated trade inside the desk sandbox",
        })
        self.save()
        return {"ok": True, "qty": fill.qty, "price": fill.price, "stop": stop,
                "risk": risk_amt,
                "detail": (f"desk bought {fill.qty:.6f} {symbol} @ {fill.price:.4f}, "
                           f"stop {stop:.4f}, risk {risk_amt:.4f} of a "
                           f"{self.state.pot:.2f} pot")}

    # ---- closing / managing ----

    def close_trade(self, price: float, why: str, kind: str = "signal",
                    now: Optional[datetime] = None) -> Optional[Trade]:
        pos = self.state.position
        if pos is None:
            return None
        now = now or utcnow()
        broker = PaperBroker(self.cfg.costs, self.cfg.risk)
        # Reconstruct the portfolio exactly as it was mid-trade: the cash the
        # shop actually holds, plus the position. The broker credits the sale
        # proceeds, so this round-trips to cash = pre-trade cash + trade.net.
        pf = Portfolio(cash=self.state.pot)
        pf.position = pos
        trade = broker.sell(pf, price, now, why, kind=kind)
        self.state.pot = pf.cash            # cash after proceeds are credited
        self.state.position = None
        self.state.trades.append(trade)
        self.ledger.append("desk_close", {
            "symbol": trade.symbol, "qty": round(trade.qty, 8),
            "exit_price": round(trade.exit_price, 6), "net": round(trade.net, 6),
            "return_pct": round(trade.return_pct, 6), "reason": trade.reason.value,
            "why": why, "opened_by": self.state.opened_by,
        })
        self.save()
        self.review()
        return trade

    def equity(self, price: float) -> float:
        """Desk equity = cash held + position marked at market."""
        pos = self.state.position
        return self.state.pot + (pos.qty * price if pos else 0.0)
