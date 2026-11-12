"""MAGICWAND — the backtest engine.

The rules that make this honest:

  1. A decision made on bar i can only use bars 0..i.
  2. That decision fills at the OPEN of bar i+1 — never at the bar it saw.
  3. Stops and targets trigger intrabar, and when a bar hits both, the
     stop wins. That is the pessimistic assumption.
  4. Fees and slippage are charged on both sides, every time.
  5. Buy-and-hold is the benchmark. If the strategy cannot beat it after
     costs, it has no reason to exist.

Point 2 is the single most common way backtests lie. A signal that fills at
the close of the bar that produced it is looking at the future.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from . import data as data_mod
from .broker import PaperBroker, Portfolio, check_exit
from .common import (Bar, ExitReason, Side, Trade, iso, max_drawdown,
                     summarize_returns, safe_mean, safe_std, utcnow)
from .config import Config
from .risk import RiskEngine
from .strategy import Intent, Strategy


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    timeframe: str
    bars: int
    data_fingerprint: str
    start_equity: float
    final_equity: float
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    trades: list[Trade]
    equity_curve: list[float]
    stats: dict
    fees_paid: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def beats_hold(self) -> bool:
        return self.total_return > self.buy_hold_return


@dataclass
class BacktestOptions:
    fee_mult: float = 1.0       # stress: 2.0 = double the assumed costs
    signal_delay: int = 1       # bars between decision and fill (1 = no lookahead)
    shuffle: bool = False       # null-hypothesis test: destroy time structure
    seed: int = 7


def _prepare_bars(bars: list[Bar], opts: BacktestOptions) -> list[Bar]:
    if not opts.shuffle:
        return bars
    rng = random.Random(opts.seed)
    closes = [b.close for b in bars]
    rng.shuffle(closes)
    out: list[Bar] = []
    for orig, c in zip(bars, closes):
        o = orig.open
        hi = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
        out.append(Bar(ts=orig.ts, open=o, high=hi, low=lo, close=c, volume=orig.volume))
    return out


def _note(notes: list[str], msg: str) -> None:
    """Append a note, but only once per distinct message.

    A halted risk engine can emit the same veto on every bar; a hundred
    identical lines is noise, not information.
    """
    if not notes or notes[-1] != msg:
        notes.append(msg)


def run_backtest(bars: list[Bar], cfg: Config, strategy: Strategy,
                 opts: Optional[BacktestOptions] = None) -> BacktestResult:
    opts = opts or BacktestOptions(seed=cfg.seed)
    if len(bars) < 10:
        raise ValueError("need at least 10 bars to backtest anything")
    bars = _prepare_bars(bars, opts)

    costs = type(cfg.costs)(fee_bps=cfg.costs.fee_bps * opts.fee_mult,
                            slippage_bps=cfg.costs.slippage_bps * opts.fee_mult)
    broker = PaperBroker(costs, cfg.risk)
    risk = RiskEngine(cfg.risk)
    pf = Portfolio(cash=cfg.risk.equity, peak_equity=cfg.risk.equity)
    pf.mark(bars[0].close)

    equity_curve: list[float] = []
    notes: list[str] = []
    # A signal decided on bar i fills at the open of bar i + signal_delay.
    # signal_delay=1 is the honest baseline: no information from the future.
    queue: list[tuple[int, Intent]] = []
    warm = strategy.warmup()
    strategy.prepare(bars)   # compute indicator series once, causally

    def due(i: int) -> Optional[Intent]:
        """Pop the oldest signal whose execution bar has arrived."""
        while queue and queue[0][0] <= i:
            _, intent = queue.pop(0)
            return intent
        return None

    for i, bar in enumerate(bars):
        # ---- 1. execute the previous bar's decision at this bar's open ----
        pending = due(i)
        if pending is not None and pending.action == "enter" and pf.position is None:
            px = bar.open
            equity = pf.mark(px)
            dec = risk.size_entry(equity, px, pending.stop_distance,
                                  open_positions=0, when=bar.ts)
            if dec.allowed and dec.qty * px >= cfg.risk.min_notional:
                stop = px - pending.stop_distance if pending.side is Side.BUY else None
                target = (px + pending.target_distance
                          if pending.target_distance > 0 else None)
                try:
                    broker.buy(pf, px, dec.qty, bar.ts, pending.reason,
                               stop=stop, target=target,
                               expected_move=pending.expected_move)
                except Exception as e:  # never let one order kill the run
                    _note(notes, f"buy rejected by broker: {e}")
            else:
                _note(notes, f"entry vetoed: {dec.reason}")
        elif pending is not None and pending.action == "exit" and pf.position is not None:
            # Exits are NEVER gated by the risk engine. A daily-loss halt
            # blocks new risk; it must not trap you in an open position.
            t = broker.sell(pf, bar.open, bar.ts, pending.reason, kind="signal")
            risk.record_realized(t.net, pf.mark(bar.close), bar.ts)

        # ---- 2. intrabar stop / target ----
        pos = pf.position
        if pos is not None:
            pos.bars_held += 1
            hit = check_exit(pos, bar)
            if hit is not None:
                kind = "stop" if hit.reason is ExitReason.STOP else "target"
                try:
                    t = broker.sell(pf, hit.price, bar.ts, hit.note, kind=kind)
                    risk.record_realized(t.net, pf.mark(bar.close), bar.ts)
                except Exception as e:
                    _note(notes, f"exit failed: {e}")
                pos = pf.position = None

        # ---- 3. time stop ----
        pos = pf.position
        if pos is not None:
            strat_max = cfg.strategy.max_hold_bars
            if strat_max and pos.bars_held >= strat_max:
                t = broker.sell(pf, bar.close, bar.ts,
                                f"time stop: held {pos.bars_held} bars (limit {strat_max})",
                                kind="time")
                risk.record_realized(t.net, pf.mark(bar.close), bar.ts)
                pos = pf.position = None

        # ---- 4. ask the strategy, for execution after the configured delay ----
        if i >= warm:
            intent = strategy.decide(bars[: i + 1], i, in_position=pf.position is not None)
            if intent is not None:
                enter_ok = intent.action == "enter" and pf.position is None
                exit_ok = intent.action == "exit" and pf.position is not None
                if enter_ok or exit_ok:
                    queue.append((i + max(1, opts.signal_delay), intent))

        equity_curve.append(pf.mark(bar.close))
        # Feed mark-to-market equity to the risk engine so the daily loss
        # limit sees intraday bleeding, not just realized closes.
        risk.update_equity(equity_curve[-1], bar.ts)
    # ---- close any open position at the last close, so P&L is complete ----
    if pf.position is not None:
        t = broker.sell(pf, bars[-1].close, bars[-1].ts,
                        "end of backtest window", kind="end")
        risk.record_realized(t.net, pf.mark(bars[-1].close), bars[-1].ts)
        equity_curve[-1] = pf.mark(bars[-1].close)

    start = cfg.risk.equity
    final = pf.equity
    total_return = (final / start - 1.0) if start else 0.0
    bh = (bars[-1].close / bars[0].open - 1.0) if bars[0].open else 0.0
    dd, _, _ = max_drawdown(equity_curve)
    stats = _stats(pf.trades, equity_curve)

    if total_return <= bh and total_return <= 0:
        notes.append("did not beat buy-and-hold and did not make money: no edge shown.")
    elif total_return <= bh:
        notes.append("made money but trailed buy-and-hold after costs: the benchmark wins.")

    return BacktestResult(
        strategy=strategy.name, symbol=cfg.market.symbol,
        timeframe=cfg.market.timeframe, bars=len(bars),
        data_fingerprint=data_mod.fingerprint(bars),
        start_equity=start, final_equity=final,
        total_return=total_return, buy_hold_return=bh,
        max_drawdown=dd, trades=pf.trades, equity_curve=equity_curve,
        stats=stats, fees_paid=sum(f.fee for f in pf.fills), notes=notes,
    )


def _stats(trades: list[Trade], equity_curve: list[float]) -> dict:
    rs = [t.return_pct for t in trades]
    s = summarize_returns(rs)
    # Equity-based Sharpe (per-bar), annualized by the bar count is not
    # meaningful across timeframes, so report the raw per-bar ratio and say so.
    bar_rets = []
    for a, b in zip(equity_curve, equity_curve[1:]):
        if a:
            bar_rets.append(b / a - 1.0)
    sd = safe_std(bar_rets)
    s["sharpe_per_bar"] = (safe_mean(bar_rets) / sd) if sd else 0.0
    s["exposure"] = (sum(1 for t in trades) * safe_mean([t.holding_seconds for t in trades])
                     / (len(equity_curve) * 3600)) if trades else 0.0
    s["avg_hold_bars"] = safe_mean([t.holding_seconds / 3600.0 for t in trades])
    return s


def reality_check(bars: list[Bar], cfg: Config, strategy: Strategy) -> dict:
    """Try to kill the strategy. If it survives, it is worth more attention.

    Three attacks:
      * 2x costs  — the assumption was too kind
      * +1 bar delay — the signal was a touch late anyway
      * shuffled series — serial structure is destroyed; only luck survives
    """
    base = run_backtest(bars, cfg, strategy)
    doubled = run_backtest(bars, cfg, strategy, BacktestOptions(fee_mult=2.0, seed=cfg.seed))
    delayed = run_backtest(bars, cfg, strategy, BacktestOptions(signal_delay=2, seed=cfg.seed))
    shuffled_runs = []
    for k in range(5):
        r = run_backtest(bars, cfg, strategy,
                         BacktestOptions(shuffle=True, seed=cfg.seed + k))
        shuffled_runs.append(r.total_return)
    null_mean = safe_mean(shuffled_runs)
    beats_null = sum(1 for r in shuffled_runs if base.total_return > r)
    # "Survives" means the edge is still there — i.e. still profitable — after
    # the attack. Losing less when handicapped is not survival: on a strategy
    # that already loses, a worse fill cannot make it "die".
    lag_penalty = base.total_return - delayed.total_return
    return {
        "base": base,
        "doubled": doubled,
        "delayed": delayed,
        "shuffled_returns": shuffled_runs,
        "null_mean": null_mean,
        "beats_null": beats_null,
        "lag_penalty": lag_penalty,
        "survives_costs": doubled.total_return > 0,
        "survives_delay": delayed.total_return > 0,
        "survives_shuffle": base.total_return > null_mean,
    }
