"""MAGICWAND — command line interface.

    wand data      pull bars and show what we have
    wand backtest  run the strategy over history, honestly
    wand reality   try to kill the strategy (2x costs, delay, shuffle)
    wand step      one tick of the paper bot
    wand loop      ask the exchange again every N seconds
    wand status    equity, open position, and the wand's own scorecard
    wand verify    check the ledger has not been tampered with
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import backtest as bt
from . import data as data_mod
from . import paper
from .config import Config
from .ledger import Ledger
from .strategy import build as build_strategy


def _load_config(path: str | None) -> Config:
    if not path:
        return Config()
    return Config.from_dict(json.loads(Path(path).read_text()))


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _print_result(r: bt.BacktestResult) -> None:
    s = r.stats
    print(f"\n  strategy        {r.strategy}")
    print(f"  market          {r.symbol} @ {r.timeframe}   bars={r.bars}  data={r.data_fingerprint}")
    print(f"  start / final   {r.start_equity:.2f} -> {r.final_equity:.2f}")
    print(f"  return          {_fmt_pct(r.total_return)}")
    print(f"  buy & hold      {_fmt_pct(r.buy_hold_return)}   <-- the benchmark to beat")
    print(f"  max drawdown    {r.max_drawdown:.2%}")
    print(f"  fees paid       {r.fees_paid:.4f}")
    if s.get("n"):
        print(f"  trades          {s['n']}  (win {s['win_rate']:.0%}, "
              f"avg win {_fmt_pct(s['avg_win'])}, avg loss {_fmt_pct(s['avg_loss'])})")
        print(f"  expectancy      {_fmt_pct(s['mean'])} per trade   "
              f"t-stat {s['t_stat']:.2f}   "
              f"{'<-- not significant, could be noise' if abs(s['t_stat']) < 2 else ''}")
        pf = s.get("profit_factor", 0)
        print(f"  profit factor   {'inf' if pf == float('inf') else f'{pf:.2f}'}")
        print(f"  avg hold        {s.get('avg_hold_bars', 0):.1f} bars")
    else:
        print("  trades          0  <-- the strategy never fired")
    for n in r.notes:
        print(f"  note            {n}")
    print()


def cmd_data(cfg: Config, args) -> int:
    bars = data_mod.fetch(cfg.market.exchange, args.symbol or cfg.market.symbol,
                          cfg.market.timeframe, args.candles or cfg.market.candles,
                          seed=cfg.seed)
    print(f"{len(bars)} bars  {bars[0].ts.isoformat()} -> {bars[-1].ts.isoformat()}")
    print(f"fingerprint {data_mod.fingerprint(bars)}")
    print(f"last close  {bars[-1].close:.4f}")
    return 0


def cmd_backtest(cfg: Config, args) -> int:
    if args.strategy:
        cfg.strategy.name = args.strategy
    cfg.market.symbol = args.symbol or cfg.market.symbol
    bars = data_mod.fetch(cfg.market.exchange, cfg.market.symbol, cfg.market.timeframe,
                          args.candles or cfg.market.candles, seed=cfg.seed)
    strat = build_strategy(cfg.strategy, seed=cfg.seed)
    opts = bt.BacktestOptions(fee_mult=args.fee_mult, signal_delay=1,
                              seed=cfg.seed)
    r = bt.run_backtest(bars, cfg, strat, opts)
    _print_result(r)
    return 0


def cmd_reality(cfg: Config, args) -> int:
    if args.strategy:
        cfg.strategy.name = args.strategy
    cfg.market.symbol = args.symbol or cfg.market.symbol
    bars = data_mod.fetch(cfg.market.exchange, cfg.market.symbol, cfg.market.timeframe,
                          args.candles or cfg.market.candles, seed=cfg.seed)
    strat = build_strategy(cfg.strategy, seed=cfg.seed)
    rc = bt.reality_check(bars, cfg, strat)
    print(f"\n  REALITY CHECK — {strat.name} on {cfg.market.symbol} @ {cfg.market.timeframe}")
    _print_result(rc["base"])
    print(f"  2x costs        {_fmt_pct(rc['doubled'].total_return)}   "
          f"{'still profitable' if rc['survives_costs'] else 'not profitable — costs own it'}")
    print(f"  1 extra bar lag {_fmt_pct(rc['delayed'].total_return)}   "
          f"lag cost {_fmt_pct(rc['lag_penalty'])} of performance")
    print(f"  shuffled data   {_fmt_pct(rc['null_mean'])} average across 5 runs   "
          f"strategy beat {rc['beats_null']}/5 shuffles "
          f"({'no sign of skill' if rc['beats_null'] <= 3 else 'weak sign'})")
    verdict = (
        "No evidence of edge. Do not trade this with real money."
        if not (rc["survives_costs"] and rc["survives_delay"] and rc["survives_shuffle"]
                and rc["base"].total_return > rc["base"].buy_hold_return)
        else "Survived all three attacks and beat holding. Worth a paper run — not live money yet."
    )
    print(f"\n  VERDICT: {verdict}\n")
    return 0


def cmd_step(cfg: Config, args) -> int:
    rep = paper.step(cfg)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def cmd_loop(cfg: Config, args) -> int:
    print(f"paper loop: {cfg.market.symbol} @ {cfg.market.timeframe}, "
          f"every {args.interval}s. Ctrl-C to stop. No real money involved.")
    try:
        while True:
            rep = paper.step(cfg)
            for a in rep.get("actions", []):
                print(f"  {rep['ts']}  {a}")
            for m in rep.get("messages", []):
                print(f"  {rep['ts']}  {m}")
            if not rep.get("actions") and not rep.get("messages"):
                print(f"  {rep['ts']}  price {rep['price']:.4f}  equity {rep['equity']:.2f}  "
                      f"(no action)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def cmd_status(cfg: Config, args) -> int:
    st = paper.status(cfg)
    print(json.dumps(st, indent=2, default=str))
    return 0


def cmd_verify(cfg: Config, args) -> int:
    led = Ledger(cfg.ledger.path)
    ok, msg = led.verify()
    print(("OK   " if ok else "FAIL ") + msg)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wand", description="MAGICWAND paper trading bot")
    p.add_argument("--config", help="path to a JSON config")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="fetch bars")
    d.add_argument("--symbol"); d.add_argument("--candles", type=int)
    d.set_defaults(fn=cmd_data)

    b = sub.add_parser("backtest", help="run a backtest")
    b.add_argument("--strategy"); b.add_argument("--symbol")
    b.add_argument("--candles", type=int); b.add_argument("--fee-mult", type=float, default=1.0)
    b.set_defaults(fn=cmd_backtest)

    r = sub.add_parser("reality", help="attempt to disprove the strategy")
    r.add_argument("--strategy"); r.add_argument("--symbol"); r.add_argument("--candles", type=int)
    r.set_defaults(fn=cmd_reality)

    s = sub.add_parser("step", help="one tick of the paper bot")
    s.set_defaults(fn=cmd_step)

    l = sub.add_parser("loop", help="run the paper bot repeatedly")
    l.add_argument("--interval", type=int, default=60)
    l.set_defaults(fn=cmd_loop)

    st = sub.add_parser("status", help="portfolio + scorecard")
    st.set_defaults(fn=cmd_status)

    v = sub.add_parser("verify", help="check ledger integrity")
    v.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    cfg = _load_config(args.config)
    return args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
