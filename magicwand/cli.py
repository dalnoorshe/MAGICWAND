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
from . import research as research_mod
from . import propose as propose_mod
from . import research_loop
from . import runner
from .config import Config
from . import service as service_mod
from . import storage as storage_mod
from .dashboard import serve as serve_dashboard
from .desk import AgentDesk, DeskConfig
from .ledger import Ledger, scorecard
from .llm import DeepSeekClient, fake_canned, fake_proposal
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


def cmd_research(cfg: Config, args) -> int:
    """One research cycle: read the market, get verdicts, maybe act in the desk."""
    client = DeepSeekClient(
        model=args.model,
        fake=fake_canned if args.offline else None,
        max_calls_per_day=args.max_calls,
        max_cost_per_day_usd=args.max_cost,
    )
    if not client.available():
        print("No API key found.\n"
              "  Set it in your shell (BYOK):  export DEEPSEEK_API_KEY=sk-...\n"
              "  or run offline:               python3 wand.py research --offline",
              file=sys.stderr)
        return 2

    symbol = args.symbol or cfg.market.symbol
    price, atr = args.price, args.atr
    if price is None or atr is None:
        bars = data_mod.fetch(cfg.market.exchange, symbol, cfg.market.timeframe,
                              cfg.market.candles, seed=cfg.seed)
        price = bars[-1].close if price is None else price
        from .indicators import atr as _atr
        a = _atr(bars, cfg.strategy.atr_period)
        atr = (a[-1] or price * 0.01) if atr is None else atr

    dcfg = DeskConfig(enabled=args.desk,
                      max_fraction_of_equity=args.desk_fraction,
                      min_confidence=args.min_confidence)
    snapshot = None
    if args.offline_snapshot:
        snapshot = research_mod.MarketSnapshot(
            ts=research_mod.iso(research_mod.utcnow()), symbol=symbol, price=price)
        snapshot.sources["derivatives"] = research_mod.Source(
            "derivatives", True, snapshot.ts,
            {"funding_rate_pct": 0.01, "funding_mean_pct": 0.008,
             "long_short_ratio": 2.7, "top_long_short_ratio": 2.3,
             "open_interest_change_pct": -1.8, "taker_buy_sell_ratio": 0.89})
        snapshot.sources["news"] = research_mod.Source(
            "news", True, snapshot.ts, {"count": 1, "items": [
                {"title": "Exchange reports large customer inflows",
                 "published": snapshot.ts, "source": "fixture", "link": ""}]})
        snapshot.sources["social"] = research_mod.Source(
            "social", False, snapshot.ts, error="offline fixture: feed unavailable")
        snapshot.sources["onchain"] = research_mod.Source(
            "onchain", True, snapshot.ts,
            {"mempool_tx": 61000, "mempool_z": 0.03,
             "stablecoin_supply_change_pct": 0.02,
             "total_mcap_change_pct": 1.1, "caveats": ["fixture"]})

    cycle = research_loop.run_cycle(
        cfg, client, price=price, atr=atr, symbol=symbol,
        subreddit=args.subreddit, dcfg=dcfg, snapshot=snapshot)

    rep = cycle.report
    print(f"\n  RESEARCH CYCLE  {rep.ts}  {symbol} @ {price:.4f}")
    print(f"  decision        {rep.decision.upper()}"
          + (f"  (vetoed by {', '.join(rep.vetoed_by)})" if rep.vetoed_by else ""))
    print(f"  evidence score  {rep.evidence_score:+.3f}")
    print(f"  summary         {rep.summary}")
    for v in rep.agents:
        if v.error:
            print(f"  - {v.agent:12s} DEGRADED  {v.error}")
            continue
        print(f"  - {v.agent:12s} {v.stance:8s} conf {v.confidence:.2f} "
              f"evidence {v.evidence_score:+.2f}")
        if v.thesis:
            print(f"      {v.thesis[:200]}")
    print(f"  claims recorded {len(cycle.recorded_claims)}   graded this run {cycle.graded}")
    if cycle.desk_outcome is not None:
        o = cycle.desk_outcome
        print(f"  desk            " + (("TRADE PLACED: " + o.get("detail", ""))
                                       if o.get("ok") else
                                       ("no trade: " + str(o.get("reason")))))
    if cycle.track_record:
        print("  agent record    " + ", ".join(
            f"{a} {s['right']}/{s['n']}" for a, s in cycle.track_record.items()))
    print(f"  cost            ${rep.cost_usd:.5f} across {rep.calls} calls")
    print()
    return 0


def cmd_desk(cfg: Config, args) -> int:
    dcfg = DeskConfig(enabled=True)
    desk = AgentDesk(cfg, dcfg)
    armed, note = desk.review()
    print(json.dumps({
        "cash": round(desk.state.pot, 6),
        "equity_at_entry": round(desk.equity(desk.state.position.entry_price), 6)
        if desk.state.position else None,
        "position": None if desk.state.position is None else {
            "symbol": desk.state.position.symbol,
            "qty": desk.state.position.qty,
            "entry": desk.state.position.entry_price,
            "stop": desk.state.position.stop,
        },
        "trades": len(desk.state.trades),
        "autonomy_armed": armed,
        "note": note,
        "revoked_reason": desk.state.revoked_reason,
    }, indent=2))
    return 0


def cmd_propose(cfg: Config, args) -> int:
    """Ask the agent for a strategy, then put it through the gate."""
    client = DeepSeekClient(model=args.model,
                            fake=fake_canned if args.offline else None)
    if not client.available():
        print("No API key. Set DEEPSEEK_API_KEY (BYOK) or use --offline.",
              file=sys.stderr)
        return 2
    symbol = args.symbol or cfg.market.symbol
    bars = data_mod.fetch(cfg.market.exchange, symbol, cfg.market.timeframe,
                          max(600, cfg.market.candles), seed=cfg.seed)
    from .research import MarketSnapshot, Source, iso, utcnow
    snap = MarketSnapshot(ts=iso(utcnow()), symbol=symbol, price=bars[-1].close)
    snap.sources["derivatives"] = Source("derivatives", True, snap.ts,
                                         {"funding_rate_pct": 0.01})
    spec = propose_mod.propose(client, symbol=symbol,
                               market_context=snap.to_prompt(),
                               proposer_fake=fake_proposal if args.offline else None)
    print(f"\n  PROPOSED: {spec.name}  ({spec.family})")
    print(f"  hypothesis: {spec.hypothesis}")
    print(f"  params: {spec.params}")
    result = propose_mod.validate(spec, bars, cfg)
    print(f"\n  {result.summary}")
    for gate, (ok, detail) in result.gates.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {gate:15s} {detail}")
    if result.passed:
        reg = propose_mod.StrategyRegistry(Path("data") / "strategies.json")
        reg.add(spec, result)
        print("\n  registered. The runner may now trade it inside the desk.")
    return 0 if result.passed else 1


def cmd_run(cfg: Config, args) -> int:
    """The unattended loop."""
    client = DeepSeekClient(model=args.model,
                            fake=fake_canned if args.offline else None,
                            max_calls_per_day=args.max_calls,
                            max_cost_per_day_usd=args.max_cost)
    if not client.available():
        print("No API key. Set DEEPSEEK_API_KEY (BYOK) or use --offline.",
              file=sys.stderr)
        return 2
    dcfg = DeskConfig(enabled=not args.no_desk,
                      min_confidence=args.min_confidence)
    return runner.run(cfg, client, dcfg, interval=args.interval,
                      max_ticks=args.ticks, propose_enabled=not args.no_propose,
                      subreddit=args.subreddit, offline=args.offline)


def cmd_strategies(cfg: Config, args) -> int:
    reg = propose_mod.StrategyRegistry(Path("data") / "strategies.json")
    items = reg.load()
    if not items:
        print("no strategy has passed the gate yet.")
        return 0
    for it in items:
        sp = it.get("spec", {})
        m = it.get("validation", {}).get("metrics", {})
        print(f"  {sp.get('name'):24s} {sp.get('family'):12s} "
              f"holdout {m.get('holdout_return', 0):+.2%}  "
              f"train {m.get('train_return', 0):+.2%}  "
              f"{'DEPLOYED' if it.get('deployed') else '-'}")
    return 0


def cmd_serve(cfg: Config, args) -> int:
    return serve_dashboard(root=".", host=args.host, port=args.port,
                           cfg=cfg, offline=not args.live,
                           autostart=args.auto, interval=args.interval)


def cmd_service(cfg: Config, args) -> int:
    action = args.action
    if action == "start":
        out = service_mod.start(port=args.port, interval=args.interval,
                                mode="live" if args.live else "offline",
                                autostart=(args.interval > 0))
        if out.get("ok"):
            print(f"started  {out['url']}   pid {out['status']['pid']}")
            print(f"log      {out['status']['log']}")
            print("survives closing this terminal. stop with: wand service stop")
        else:
            print(json.dumps(out, indent=2))
        return 0 if out.get("ok") else 1
    if action == "stop":
        out = service_mod.stop()
        print(json.dumps(out, indent=2))
        return 0 if out.get("ok") else 1
    if action == "restart":
        service_mod.stop()
        time.sleep(1)
        return cmd_service(cfg, argparse.Namespace(action="start", port=args.port,
                                                   interval=args.interval,
                                                   live=args.live))
    if action == "log":
        print(service_mod.tail_log(lines=args.lines))
        return 0
    if action == "autostart":
        out = service_mod.autostart_install(port=args.port, interval=args.interval)
        print(json.dumps(out, indent=2))
        return 0 if out.get("ok") else 1
    print(json.dumps(service_mod.status(), indent=2))
    return 0


def cmd_storage(cfg: Config, args) -> int:
    if args.prune:
        print(json.dumps(storage_mod.housekeep("."), indent=2))
        return 0
    if args.backup:
        p = storage_mod.backup(".")
        print(f"backup written: {p}")
        print(f"integrity: {p.stat().st_size} bytes")
        return 0
    if args.restore:
        out = storage_mod.restore(".", args.restore, overwrite=args.force)
        print(json.dumps(out, indent=2))
        return 0 if out.get("ok") else 1
    print(json.dumps(storage_mod.storage_report("."), indent=2))
    return 0


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

    rs = sub.add_parser("research", help="run the DeepSeek research agents")
    rs.add_argument("--symbol")
    rs.add_argument("--model", default="deepseek-flash")
    rs.add_argument("--offline", action="store_true",
                    help="use the deterministic fake model (no key, no network)")
    rs.add_argument("--offline-snapshot", action="store_true",
                    help="use a fixture snapshot instead of live sources")
    rs.add_argument("--desk", action="store_true",
                    help="let the agent desk place an assured trade")
    rs.add_argument("--desk-fraction", type=float, default=0.10)
    rs.add_argument("--min-confidence", type=float, default=0.80)
    rs.add_argument("--subreddit", default="solana")
    rs.add_argument("--price", type=float)
    rs.add_argument("--atr", type=float)
    rs.add_argument("--max-calls", type=int, default=400)
    rs.add_argument("--max-cost", type=float, default=2.0)
    rs.set_defaults(fn=cmd_research)

    dk = sub.add_parser("desk", help="show the agent desk sandbox state")
    dk.set_defaults(fn=cmd_desk)

    pr = sub.add_parser("propose", help="agent invents a strategy; the gate judges it")
    pr.add_argument("--symbol")
    pr.add_argument("--model", default="deepseek-flash")
    pr.add_argument("--offline", action="store_true")
    pr.set_defaults(fn=cmd_propose)

    rn = sub.add_parser("run", help="unattended autonomous loop")
    rn.add_argument("--interval", type=int, default=300)
    rn.add_argument("--ticks", type=int, default=0, help="0 = run forever")
    rn.add_argument("--model", default="deepseek-flash")
    rn.add_argument("--offline", action="store_true")
    rn.add_argument("--no-desk", action="store_true")
    rn.add_argument("--no-propose", action="store_true")
    rn.add_argument("--min-confidence", type=float, default=0.80)
    rn.add_argument("--subreddit", default="solana")
    rn.add_argument("--max-calls", type=int, default=400)
    rn.add_argument("--max-cost", type=float, default=2.0)
    rn.set_defaults(fn=cmd_run)

    sg = sub.add_parser("strategies", help="list gate-approved strategies")
    sg.set_defaults(fn=cmd_strategies)

    sc = sub.add_parser("service", help="run the console in the background")
    sc.add_argument("action", nargs="?", default="status",
                    choices=["start", "stop", "restart", "status", "log", "autostart"])
    sc.add_argument("--port", type=int, default=8787)
    sc.add_argument("--interval", type=int, default=900)
    sc.add_argument("--live", action="store_true")
    sc.add_argument("--lines", type=int, default=30)
    sc.set_defaults(fn=cmd_service)

    stg = sub.add_parser("storage", help="local data: size, prune, backup, restore")
    stg.add_argument("--prune", action="store_true", help="delete old snapshots")
    stg.add_argument("--backup", action="store_true", help="write a local archive")
    stg.add_argument("--restore", metavar="ARCHIVE")
    stg.add_argument("--force", action="store_true", help="overwrite data/ on restore")
    stg.set_defaults(fn=cmd_storage)

    sv = sub.add_parser("serve", help="local web dashboard")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--live", action="store_true",
                    help="call the real DeepSeek API (needs DEEPSEEK_API_KEY)")
    sv.add_argument("--auto", action="store_true",
                    help="start the autonomous scheduler immediately")
    sv.add_argument("--interval", type=int, default=900,
                    help="autopilot seconds between ticks (default 900)")
    sv.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    cfg = _load_config(args.config)
    return args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
