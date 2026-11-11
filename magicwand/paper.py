"""MAGICWAND — the live paper loop.

Each run:
  1. fetch the latest closed bars
  2. load the paper portfolio from disk
  3. manage the open position (stop / target / time / signal exit)
  4. optionally open one new position, if the risk engine allows
  5. write every decision to the hash-chained ledger
  6. print what happened and why, in plain English

It cannot trade real money. There is no live broker in this repository.

Run it repeatedly (cron, or `wand loop`) to accumulate a track record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import data as data_mod
from .broker import BrokerError, PaperBroker, Portfolio, check_exit
from .common import Bar, ExitReason, Position, Side, iso, utcnow
from .config import Config
from .ledger import Ledger, grade_claim, make_claim, scorecard
from .risk import RiskEngine
from .strategy import Strategy, build as build_strategy


class StateError(RuntimeError):
    pass


def load_portfolio(path: str | Path, cfg: Config) -> Portfolio:
    p = Path(path)
    if not p.exists():
        return Portfolio(cash=cfg.risk.equity, equity=cfg.risk.equity,
                         peak_equity=cfg.risk.equity)
    d = json.loads(p.read_text())
    pf = Portfolio(cash=float(d.get("cash", cfg.risk.equity)),
                   equity=float(d.get("equity", cfg.risk.equity)),
                   peak_equity=float(d.get("peak_equity", cfg.risk.equity)))
    pos = d.get("position")
    if pos:
        from .common import parse_iso
        pf.position = Position(
            symbol=pos["symbol"], side=Side(pos["side"]), qty=float(pos["qty"]),
            entry_price=float(pos["entry_price"]), entry_ts=parse_iso(pos["entry_ts"]),
            stop=pos.get("stop"), target=pos.get("target"),
            entry_fee=float(pos.get("entry_fee", 0.0)),
            bars_held=int(pos.get("bars_held", 0)),
            entry_reason=pos.get("entry_reason", ""),
            expected_move=pos.get("expected_move"),
        )
    return pf


def save_portfolio(path: str | Path, pf: Portfolio) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pos = None
    if pf.position is not None:
        q = pf.position
        pos = {
            "symbol": q.symbol, "side": q.side.value, "qty": q.qty,
            "entry_price": q.entry_price, "entry_ts": iso(q.entry_ts),
            "stop": q.stop, "target": q.target, "entry_fee": q.entry_fee,
            "bars_held": q.bars_held, "entry_reason": q.entry_reason,
            "expected_move": q.expected_move,
        }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "cash": pf.cash, "equity": pf.equity, "peak_equity": pf.peak_equity,
        "position": pos, "saved_at": iso(utcnow()),
    }, indent=2))
    tmp.replace(p)


def _unanswered_claims(entries: list[dict], cfg: Config) -> list[dict]:
    graded = {e["payload"].get("claim_id") for e in entries if e.get("kind") == "grade"}
    return [e["payload"] for e in entries
            if e.get("kind") == "claim" and e["payload"].get("claim_id") not in graded]


def step(cfg: Config, strategy: Optional[Strategy] = None,
         bars: Optional[list[Bar]] = None, root: str | Path = ".") -> dict:
    """One tick of the paper bot. Returns a human-readable report dict."""
    strategy = strategy or build_strategy(cfg.strategy, seed=cfg.seed)
    ledger = Ledger(Path(root) / cfg.ledger.path)
    pf = load_portfolio(Path(root) / cfg.ledger.state_path, cfg)
    broker = PaperBroker(cfg.costs, cfg.risk)
    risk = RiskEngine(cfg.risk)

    if bars is None:
        bars = data_mod.fetch(cfg.market.exchange, cfg.market.symbol,
                              cfg.market.timeframe, cfg.market.candles,
                              seed=cfg.seed, cache_root=root)
    if len(bars) <= strategy.warmup():
        return {"status": "warmup",
                "message": f"need more than {strategy.warmup()} bars, have {len(bars)}"}

    last = bars[-1]
    pf.mark(last.close)
    risk.update_equity(pf.equity, last.ts)
    report: dict = {"ts": iso(last.ts), "price": last.close, "equity": pf.equity,
                    "actions": [], "messages": []}

    # ---- 1. manage an existing position ----
    pos = pf.position
    if pos is not None:
        pos.bars_held += 1
        hit = check_exit(pos, last)
        if hit is not None:
            kind = "stop" if hit.reason is ExitReason.STOP else "target"
            t = broker.sell(pf, hit.price, last.ts, hit.note, kind=kind)
            risk.record_realized(t.net, pf.mark(last.close), last.ts)
            _finish_claim(cfg, ledger, pos, last, t.net / (pos.entry_price * pos.qty),
                          pos.bars_held, kind)
            report["actions"].append(
                f"CLOSED {kind.upper()}: {hit.note} — net {t.net:+.4f} "
                f"({t.return_pct:+.2%}, fees {t.fees:.4f})")
            save_portfolio(Path(root) / cfg.ledger.state_path, pf)
        elif cfg.strategy.max_hold_bars and pos.bars_held >= cfg.strategy.max_hold_bars:
            t = broker.sell(pf, last.close, last.ts, "time stop", kind="time")
            risk.record_realized(t.net, pf.mark(last.close), last.ts)
            _finish_claim(cfg, ledger, pos, last, t.net / (pos.entry_price * pos.qty),
                          pos.bars_held, "time")
            report["actions"].append(f"CLOSED TIME STOP — net {t.net:+.4f} ({t.return_pct:+.2%})")
            save_portfolio(Path(root) / cfg.ledger.state_path, pf)
            pos = pf.position = None
        else:
            intent = strategy.decide(bars, len(bars) - 1, in_position=True)
            if intent is not None and intent.action == "exit":
                t = broker.sell(pf, last.close, last.ts, intent.reason, kind="signal")
                risk.record_realized(t.net, pf.mark(last.close), last.ts)
                _finish_claim(cfg, ledger, pos, last, t.net / (pos.entry_price * pos.qty),
                              pos.bars_held, "signal")
                report["actions"].append(f"CLOSED ON SIGNAL: {intent.reason} — net {t.net:+.4f}")

    # ---- 2. consider a new entry ----
    pf.mark(last.close)
    if pf.position is None:
        intent = strategy.decide(bars, len(bars) - 1, in_position=False)
        if intent is not None and intent.action == "enter":
            dec = risk.size_entry(pf.equity, last.close, intent.stop_distance,
                                  open_positions=0, when=last.ts)
            if dec.allowed:
                stop = last.close - intent.stop_distance
                target = last.close + intent.target_distance if intent.target_distance > 0 else None
                try:
                    fill = broker.buy(pf, last.close, dec.qty, last.ts, intent.reason,
                                      stop=stop, target=target,
                                      expected_move=intent.expected_move)
                except BrokerError as e:
                    fill = None
                    report["messages"].append(f"broker refused: {e}")
                if fill is not None:
                    pf.position.symbol = cfg.market.symbol
                    claim = make_claim(
                        trade_id=f"claim-{ledger._seq}",
                        symbol=cfg.market.symbol,
                        expected_move=intent.expected_move,
                        entry_price=fill.price,
                        horizon_bars=cfg.ledger.claim_horizon_bars,
                        reason=intent.reason, when=last.ts)
                    ledger.append("claim", claim, when=last.ts)
                    ledger.append("fill", {
                        "symbol": cfg.market.symbol, "side": "buy",
                        "qty": round(fill.qty, 8), "price": round(fill.price, 6),
                        "fee": round(fill.fee, 6), "reason": intent.reason,
                        "stop": round(stop, 6) if stop else None,
                        "target": round(target, 6) if target else None,
                    }, when=last.ts)
                    report["actions"].append(
                        f"BOUGHT {fill.qty:.6f} {cfg.market.symbol} @ {fill.price:.4f} "
                        f"— {intent.reason}; {dec.reason}; stop {stop:.4f}, "
                        f"target {target if target else 'none'}")
                    save_portfolio(Path(root) / cfg.ledger.state_path, pf)
            else:
                ledger.append("veto", {"reason": dec.reason, "price": last.close,
                                       "strategy": strategy.name}, when=last.ts)
                report["messages"].append(f"risk engine vetoed entry: {dec.reason}")

    # ---- 3. grade any claims whose horizon has passed ----
    _grade_matured(cfg, ledger, bars)
    ledger.append("mark", {"equity": round(pf.equity, 6), "price": round(last.close, 6),
                           "cash": round(pf.cash, 6)}, when=last.ts)
    save_portfolio(Path(root) / cfg.ledger.state_path, pf)
    return report


def _finish_claim(cfg: Config, ledger: Ledger, pos: Position, bar: Bar,
                  realized_move: float, bars_taken: int, exit_kind: str) -> None:
    if pos.expected_move is None:
        return
    entries = ledger.entries()
    for claim in reversed(_unanswered_claims(entries, cfg)):
        if abs(claim.get("expected_move", 0.0) - pos.expected_move) < 1e-9:
            ledger.append("grade", grade_claim(claim, realized_move, bars_taken, exit_kind),
                          when=bar.ts)
            return


def _grade_matured(cfg: Config, ledger: Ledger, bars: list[Bar]) -> None:
    """Mature untraded claims (e.g. vetoed entries) get graded by price alone."""
    from .common import parse_iso
    entries = ledger.entries()
    now = bars[-1]
    for claim in _unanswered_claims(entries, cfg):
        claim_ts = parse_iso(claim["ts"])
        age = (now.ts - claim_ts).total_seconds() / 3600.0
        if age < cfg.ledger.claim_horizon_bars:
            continue
        entry_price = float(claim["entry_price"])
        realized = (now.close - entry_price) / entry_price
        ledger.append("grade", grade_claim(claim, realized,
                                           int(age), "horizon"), when=now.ts)


def status(cfg: Config, root: str | Path = ".") -> dict:
    ledger = Ledger(Path(root) / cfg.ledger.path)
    ok, msg = ledger.verify()
    pf = load_portfolio(Path(root) / cfg.ledger.state_path, cfg)
    return {
        "ledger_ok": ok, "ledger_message": msg,
        "equity": pf.equity, "cash": pf.cash,
        "position": None if pf.position is None else {
            "symbol": pf.position.symbol, "qty": pf.position.qty,
            "entry": pf.position.entry_price, "stop": pf.position.stop,
            "target": pf.position.target, "bars_held": pf.position.bars_held,
        },
        "scorecard": scorecard(ledger.entries()),
        "entries": len(ledger.entries()),
    }
