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
from . import audit as audit_mod
from . import compose as compose_mod
from datetime import datetime, timezone

from . import forecast as forecast_mod
from . import integrations as integrations_mod
from . import service as service_mod
from . import storage as storage_mod
from .dashboard import serve as serve_dashboard
from . import doctor as doctor_mod
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


def _apply_equity(cfg: Config, args) -> None:
    """Apply an explicit paper balance, loudly and with a ledger note."""
    equity = float(getattr(args, "equity", 0.0) or 0.0)
    if equity <= 0:
        return
    from .ledger import Ledger
    cfg.risk.equity = equity
    try:
        Ledger(Path(cfg.ledger.path)).append("note", {
            "text": f"paper balance set to {equity:.2f} from the command line "
                    f"(--equity); no broker is connected, so this is an "
                    f"assumption, not a measurement"})
    except Exception:  # noqa: BLE001 — a note must never block startup
        pass
    print(f"paper equity: {equity:.2f} (from --equity)")


def cmd_loop(cfg: Config, args) -> int:
    _apply_equity(cfg, args)
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
    _apply_equity(cfg, args)
    return serve_dashboard(root=".", host=args.host, port=args.port,
                           cfg=cfg, offline=args.offline,
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
        out = storage_mod.restore(".", args.restore, overwrite=args.force,
                                  allow_running=args.allow_running)
        print(json.dumps(out, indent=2))
        return 0 if out.get("ok") else 1
    print(json.dumps(storage_mod.storage_report("."), indent=2))
    return 0


def cmd_sources(cfg: Config, args) -> int:
    """Probe every data source and report what actually came back."""
    report = doctor_mod.run(cfg, symbol=args.symbol or cfg.market.symbol)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print()
        print(doctor_mod.format_report(report))
        print()
    return 0 if report["all_ok"] else 1


def cmd_compose(cfg: Config, args) -> int:
    """Have the model compose a strategy, then judge it with the same gate."""
    from .llm import DeepSeekClient, fake_composition, fake_canned
    symbol = args.symbol or cfg.market.symbol
    exchange = args.exchange or cfg.market.exchange
    client = DeepSeekClient(model=args.model,
                            fake=(fake_composition if args.offline else None))
    if not client.available():
        print("No API key. Set DEEPSEEK_API_KEY or use --offline.", file=sys.stderr)
        return 2
    bars = data_mod.fetch(exchange, symbol, cfg.market.timeframe,
                          max(args.candles, 400), seed=cfg.seed)

    try:
        comp = compose_mod.propose(client, symbol=symbol,
                                   market_context="",
                                   proposer_fake=fake_composition if args.offline else None)
    except compose_mod.CompositionError as e:
        print(f"  proposed composition refused: {e}")
        return 1

    print()
    print(f"  COMPOSED STRATEGY — {symbol} @ {cfg.market.timeframe}")
    print("  " + "-" * 74)
    print(f"  expression : {comp.describe()}")
    print(f"  hypothesis : {comp.hypothesis}")
    print(f"  risk       : stop {comp.stop_atr:g} ATR · target {comp.target_atr:g} ATR"
          f" · time stop {comp.max_hold_bars} bars")
    print(f"  fingerprint: {comp.fingerprint()}")
    result = compose_mod.validate_composition(comp, bars, cfg)
    print()
    print(f"  {result.summary}")
    for gate, (ok, detail) in result.gates.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {gate:15s} {detail}")
    print()
    return 0 if result.passed else 1


def _forecast_sweep(cfg: Config, args, bars, symbol: str) -> int:
    """Label x feature-group matrix, on one command, one page.

    This is the experiment that decides whether better INPUTS buy skill. Every
    cell is scored the same way — Brier against its own label's climatology —
    because a triple-barrier label has a different base rate from a direction
    label, and comparing them to each other would be meaningless.
    """
    from . import forecast as fc, features as feat_mod

    all_groups = ("cross", "session", "funding")
    ladder = getattr(args, "ladder", "cumulative")
    if ladder == "full":
        combos: list[tuple[str, ...]] = [all_groups]
    elif ladder == "ends":
        combos = [(), all_groups]
    else:
        combos = [()]
        for g in all_groups:
            combos.append(tuple(list(all_groups)[:list(all_groups).index(g) + 1]))
    labels = [x.strip() for x in getattr(args, "labels", "direction,triple").split(",")
              if x.strip()]
    # Contexts are shared per group so nothing is fetched twice.
    ctx_cache: dict[tuple[str, ...], object] = {}
    for groups in combos:
        if groups:
            print(f"  loading context for {groups} …", flush=True)
            ctx_cache[groups] = feat_mod.build_context(
                symbol, args.timeframe or cfg.market.timeframe, bars, groups)

    print()
    print(f"  FEATURE / LABEL SWEEP — {symbol} @ {args.timeframe or cfg.market.timeframe}, "
          f"{len(bars)} bars, horizon {args.horizon}")
    print("  " + "-" * 84)
    print(f"  {'label':10s} {'features':22s} {'n':>5s} {'acc':>7s} "
          f"{'base':>7s} {'brier':>8s} {'climo':>8s}  verdict")
    rows_out = []
    for label in labels:
        for groups in combos:
            ev, _tail = fc.walk_forward(
                bars, horizon=args.horizon, min_train=args.min_train,
                block=args.block, label=label, groups=groups,
                ctx=ctx_cache.get(groups),
                barrier_atr=args.barrier_atr)
            name = ",".join(groups) if groups else "price only"
            if ev is None:
                print(f"  {label:10s} {name:22s}   — not enough data")
                continue
            verdict = "BEATS" if ev.beats_base else ("climo+" if ev.brier <
                                                    ev.base_brier else "no skill")
            print(f"  {label:10s} {name:22s} {ev.n:5d} {ev.accuracy:7.1%} "
                  f"{max(ev.base_rate, 1 - ev.base_rate):7.1%} "
                  f"{ev.brier:8.4f} {ev.base_brier:8.4f}  {verdict}")
            rows_out.append({"label": label, "groups": name, "n": ev.n,
                             "accuracy": ev.accuracy, "brier": ev.brier,
                             "base_brier": ev.base_brier,
                             "beats": ev.beats_base})
    print()
    best = min(rows_out, key=lambda r: r["brier"] - r["base_brier"],
               default=None)
    if best:
        print(f"  closest to skill: {best['label']} / {best['groups']} "
              f"(Brier {best['brier']:.4f} vs {best['base_brier']:.4f})")
    print()
    return 0


def cmd_forecast(cfg: Config, args) -> int:
    """Train a trend model walk-forward and report whether it beats the base rate."""
    symbol = args.symbol or cfg.market.symbol
    exchange = args.exchange or cfg.market.exchange
    timeframe = getattr(args, "timeframe", "") or cfg.market.timeframe
    bars = data_mod.fetch(exchange, symbol, timeframe,
                          max(args.candles, 400), seed=cfg.seed)
    groups = tuple(g.strip() for g in getattr(args, "groups", "").split(",")
                   if g.strip())
    for g in groups:
        if g not in forecast_mod.feature_names(("cross", "session", "funding")) and \
                g not in ("cross", "session", "funding"):
            print(f"unknown feature group {g!r}")
            return 2
    ctx = None
    if groups:
        from . import features as feat_mod
        print(f"  loading context for {groups} …", flush=True)
        ctx = feat_mod.build_context(symbol, timeframe, bars, groups)

    if getattr(args, "sweep", False):
        return _forecast_sweep(cfg, args, bars, symbol)

    stats: dict = {}
    ev, tail = forecast_mod.walk_forward(
        bars, horizon=args.horizon, min_train=args.min_train, block=args.block,
        label=args.label, groups=groups, ctx=ctx,
        barrier_atr=args.barrier_atr, stats=stats)
    print()
    print(f"  TREND FORECAST — {symbol} @ {timeframe}, "
          f"horizon {args.horizon} bars")
    print("  " + "-" * 74)
    if ev is None:
        print(f"  not enough history: {len(bars)} bars "
              f"(need {args.min_train + args.block}+)")
        return 2
    print(f"  {ev.summary()}")
    if ev.groups != "price" or ev.label != "direction":
        print(f"  label {ev.label} · features {ev.groups}")
    if stats:
        extra = ", ".join(f"{k}={v}" for k, v in sorted(stats.items())
                          if k not in ("n",))
        if extra:
            print(f"  samples {stats.get('n')} · {extra}")
    print(f"  log loss {ev.log_loss:.4f} · base rate {ev.base_rate:.1%}")
    print("  calibration (predicted vs actual):")
    for row in ev.calibration:
        print(f"    p {row['bin']:9s} n={row['n']:4d}  predicted {row['predicted']:.2f}"
              f"  actual {row['actual']:.2f}")
    if ev.by_regime:
        print("  by regime:")
        for r, v in sorted(ev.by_regime.items()):
            print(f"    {r:9s} n={v['n']:4d}  accuracy {v['accuracy']:.1%}"
                  f"  avg p {v['avg_p']:.2f}")
    if getattr(args, "by_regime", False):
        reg_ev, reg_tail = forecast_mod.walk_forward(
            bars, horizon=args.horizon, min_train=args.min_train,
            block=args.block, by_regime=True,
            min_regime_train=args.min_regime_train)
        print()
        print("  PER-REGIME MODELS (one fit per regime, pooled fallback)")
        print("  " + "-" * 74)
        if reg_ev is None:
            print("  not enough history to compare")
        else:
            print(f"  {reg_ev.summary()}")
            print(f"  {'regime':10s} {'n':>5s} {'acc':>7s} {'climo':>7s} "
                  f"{'brier':>8s} {'climo':>8s} {'verdict':>10s}")
            for r, v in sorted(reg_ev.by_regime.items()):
                beats = v["brier"] < v["base_brier"]
                note = "BEATS" if beats else "no skill"
                if v.get("fallback"):
                    note += " (fallback)"
                print(f"  {r:10s} {v['n']:5d} {v['accuracy']:7.1%} "
                      f"{max(v['base_rate'], 1 - v['base_rate']):7.1%} "
                      f"{v['brier']:8.4f} {v['base_brier']:8.4f} {note:>10s}")
            delta = reg_ev.brier - ev.brier
            print(f"  overall Brier {reg_ev.brier:.4f} vs pooled {ev.brier:.4f} "
                  f"({delta:+.4f}) -> "
                  f"{'per-regime is better' if delta < 0 else 'pooled is better'}")
            if reg_tail:
                ft = reg_tail[0]
                print(f"  next-bar forecast: p_up={ft.p_up:.3f} "
                      f"trend={ft.trend:+.2f} ({ft.regime})")
    if tail:
        f = tail[0]
        print(f"  next-bar forecast: p_up={f.p_up:.3f} trend={f.trend:+.2f} "
              f"({f.regime})")
        print(f"    claim: {f.claim_text()}")
    print()
    return 0 if ev.beats_base else 1


def _binom_p(k: int, n: int, p0: float) -> float:
    """One-sided p-value that k successes out of n came from rate p0.

    Normal approximation, which is fine at n in the thousands and keeps this
    dependency-free. Used to ask whether a cell's accuracy is better than its
    base rate by more than chance.
    """
    if n <= 0:
        return 1.0
    phat = k / n
    se = (p0 * (1.0 - p0) / n) ** 0.5
    if se <= 0:
        return 1.0
    z = (phat - p0) / se
    # Abramowitz-style logistic approximation to the normal CDF tail.
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * pow(2.718281828, -z * z / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t
                                   * (-1.821256 + t * 1.330274))))
    return prob if z > 0 else 1.0 - prob


def cmd_journal(cfg: Config, args) -> int:
    """The 30-day paper scorecard: did it trade, and if not, why exactly?"""
    from . import journal as journal_mod

    j = journal_mod.build_journal(".", cfg, days=args.days)
    print()
    print(journal_mod.format_journal(j))
    print()
    if args.save:
        out = Path("data") / "paper-journal.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(j, indent=2))
        print(f"  saved to {out}")
        print()
    return 0


def cmd_edge(cfg: Config, args) -> int:
    """Measure the current strategy's edge, then ask what size it justifies."""
    from .sizing import edge_from_trades, size_from_edge

    cfg.market.symbol = args.symbol or cfg.market.symbol
    bars = data_mod.fetch(args.exchange or cfg.market.exchange, cfg.market.symbol,
                          args.timeframe or cfg.market.timeframe,
                          args.candles or cfg.market.candles, seed=cfg.seed)
    strat = build_strategy(cfg.strategy, seed=cfg.seed)
    r = bt.run_backtest(bars, cfg, strat,
                        bt.BacktestOptions(fee_mult=1.0, signal_delay=1,
                                           seed=cfg.seed))
    edge = edge_from_trades(r.trades, label=f"{strat.name} {cfg.market.symbol}")
    # ONE instrument, two fills: fee+slippage on entry and on exit. (carry.py
    # uses four legs because it holds two instruments.)
    round_trip = 2 * (cfg.costs.fee_bps + cfg.costs.slippage_bps)
    sizing = size_from_edge(edge, per_trade_cost_bps=round_trip)

    print()
    print(f"  EDGE — {strat.name} on {cfg.market.symbol} @ "
          f"{cfg.market.timeframe}, {len(r.trades)} trades")
    print("  " + "-" * 74)
    print(f"  per-trade mean   {edge.mean_bps:+8.2f} bps")
    print(f"  per-trade sd     {edge.sd_bps:8.2f} bps")
    print(f"  standard error   {edge.se_bps:8.2f} bps   (n={edge.n})")
    print(f"  conservative     {edge.lower_bound_bps():+8.2f} bps   "
          f"(mean - 2 se)")
    print(f"  round-trip cost  {round_trip:8.2f} bps")
    print(f"  -> size          {sizing.fraction:8.2%} of equity")
    print(f"  verdict          {sizing.reason}")
    print()
    if edge.n < 30:
        print(f"  {edge.n} trades is too few to size on. This is a refusal, "
              f"not a result.")
        print()
    return 0 if sizing.ok else 1


def cmd_scan(cfg: Config, args) -> int:
    """Breadth, not cleverness: many markets x horizons, chance-corrected.

    One hypothesis at a time has produced seven negatives and no usable power:
    each cell rests on a few hundred semi-independent outcomes, so a "win" is
    indistinguishable from luck. This runs the same honest evaluation across
    every market and horizon the registry knows, then asks the only question a
    grid can answer — does ANY cell beat chance by more than testing this many
    cells would produce by accident?
    """
    from .markets import symbols as market_symbols
    from . import forecast as fc

    syms = ([x.strip() for x in args.symbols.split(",") if x.strip()]
            if args.symbols else market_symbols())
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print()
    print(f"  SCAN — {len(syms)} markets x {len(horizons)} horizons, "
          f"label {args.label}, {args.candles} bars")
    print("  " + "-" * 88)
    print(f"  {'market':10s} {'h':>4s} {'n':>6s} {'acc':>7s} {'base':>7s} "
          f"{'edge':>7s} {'p':>7s} {'brier':>8s} {'climo':>8s} {'beats':>6s}")
    cells = []
    for sym in syms:
        try:
            bars = data_mod.fetch(args.exchange, sym, args.timeframe or "1h",
                                  args.candles, seed=cfg.seed)
        except Exception as e:  # noqa: BLE001 — a dead market is a row
            print(f"  {sym:10s}   — no data: {type(e).__name__}")
            continue
        for h in horizons:
            ev, _ = fc.walk_forward(bars, horizon=h, min_train=args.min_train,
                                    block=args.block, label=args.label)
            if ev is None:
                print(f"  {sym:10s} {h:4d}   — not enough history")
                continue
            base = max(ev.base_rate, 1 - ev.base_rate)
            k = round(ev.accuracy * ev.n)
            p = _binom_p(k, ev.n, base)
            edge = ev.accuracy - base
            beats = "YES" if (ev.brier < ev.base_brier and edge > 0) else "no"
            print(f"  {sym:10s} {h:4d} {ev.n:6d} {ev.accuracy:7.1%} "
                  f"{base:7.1%} {edge:+7.1%} {p:7.3f} {ev.brier:8.4f} "
                  f"{ev.base_brier:8.4f} {beats:>6s}")
            cells.append({"market": sym, "horizon": h, "n": ev.n,
                          "accuracy": ev.accuracy, "base": base, "edge": edge,
                          "p": p, "brier": ev.brier,
                          "base_brier": ev.base_brier})
    print()
    if not cells:
        return 1
    tested = len(cells)
    threshold = 0.05 / tested           # Bonferroni; cells are NOT independent
    winners = [c for c in cells if c["brier"] < c["base_brier"]]
    raw = [c for c in cells if c["p"] < 0.05]
    survived = [c for c in cells if c["p"] < threshold]
    print(f"  cells tested        {tested}")
    print(f"  beat climatology    {len(winners)}/{tested} on Brier "
          f"(a coin flip would give about {tested // 2})")
    print(f"  p < 0.05 raw        {len(raw)}  "
          f"(expected by chance alone: ~{0.05 * tested:.1f})")
    print(f"  survives Bonferroni {len(survived)}  (threshold p < {threshold:.4f})")
    if survived:
        print("  -> " + ", ".join(f"{c['market']}@{c['horizon']}" for c in survived))
    else:
        print("  -> nothing survives. Their windows overlap and the cells are "
              "correlated, so even this is generous.")
    print()
    return 0


def cmd_carry(cfg: Config, args) -> int:
    """Funding-rate carry: does the payment cover the cost of collecting it?"""
    from . import carry as carry_mod

    symbol = args.symbol or cfg.market.symbol
    print(f"  loading funding history for {symbol} …", flush=True)
    funding = carry_mod.funding_history(symbol, days=args.days)
    if len(funding) < 10:
        print("  not enough funding history (venue unreachable?)")
        return 2
    bars = data_mod.fetch("binance", symbol, "1h", max(args.candles, 400))
    cc = carry_mod.CarryConfig(
        lookback=args.lookback, entry_bps=args.entry_bps,
        exit_bps=args.exit_bps, spot_fee_bps=args.spot_fee_bps,
        perp_fee_bps=args.perp_fee_bps, slip_bps_leg=args.slip_bps_leg)
    res = carry_mod.run_carry(funding, bars, cc)
    if res is None:
        print("  could not simulate (missing prices)")
        return 2
    first = datetime.fromtimestamp(funding[0]["ts"] / 1000, tz=timezone.utc)
    last = datetime.fromtimestamp(funding[-1]["ts"] / 1000, tz=timezone.utc)
    span_days = max(1.0, (funding[-1]["ts"] - funding[0]["ts"]) / 86_400_000.0)
    print()
    print(f"  FUNDING CARRY — {symbol} @ 8h funding, {span_days:.0f} days "
          f"({first:%Y-%m-%d} → {last:%Y-%m-%d})")
    print("  " + "-" * 74)
    print(f"  rule             enter when {args.lookback}-period avg funding "
          f"≥ {args.entry_bps:g} bps, exit below {args.exit_bps:g} bps")
    print(f"  funding periods  {res.periods}  ·  held {res.periods_held} "
          f"({res.periods_held / max(1, res.periods):.0%} of the time)  ·  "
          f"{res.entries} round trips")
    print(f"  funding paid     {res.funding_bps:+9.2f} bps")
    print(f"  basis P&L        {res.basis_bps:+9.2f} bps   (spot vs perp marks)")
    print(f"  costs            {res.fees_bps:-9.2f} bps   "
          f"({res.round_trip_cost_bps:.0f} bps per round trip, 4 legs)")
    print(f"  net              {res.net_bps:+9.2f} bps   "
          f"= {res.net_pct:+.2f}% on notional  (annualised "
          f"{res.annualised_pct:+.1f}%)")
    print(f"  max drawdown     {res.max_drawdown_pct:.2f}%   worst period "
          f"{res.worst_period_bps:+.2f} bps")
    print(f"  buy & hold spot  {res.spot_return_pct:+.1f}% over the same window")
    print(f"  break-even       {res.breakeven_periods or float('nan'):.0f} "
          f"funding periods of average size to pay one round trip "
          f"(avg hold {res.avg_hold_periods:.1f})")
    print(f"  verdict          {res.verdict()}")
    print()
    print("  Not modelled: perp-leg liquidation, custody/exchange risk, margin, "
          "or two venues.")
    print("  Returns are per unit of NOTIONAL, not per unit of capital at risk.")
    print()
    return 0 if res.net_bps > 0 else 1


def cmd_audit(cfg: Config, args) -> int:
    """Adversarial self-audit: run the scenarios and show what breaks."""
    results = audit_mod.run_all(angle=args.angle)
    saved = audit_mod.save_report(".", results, angle=args.angle)
    order = {"GAP": 0, "INFO": 1, "PASS": 2}
    results.sort(key=lambda r: order.get(r.verdict, 3))
    gaps = [r for r in results if r.verdict == "GAP"]
    print()
    print("  MAGICWAND SELF-AUDIT")
    print("  " + "-" * 76)
    for r in results:
        mark = {"GAP": "GAP ", "INFO": "info", "PASS": "pass"}[r.verdict]
        print(f"  [{mark}] {r.name}")
        print(f"         {r.detail}")
        if r.fix and args.verbose:
            print(f"         fix: {r.fix}")
        print()
    print(f"  {len(gaps)} gap(s), "
          f"{sum(1 for r in results if r.verdict == 'INFO')} note(s), "
          f"{sum(1 for r in results if r.verdict == 'PASS')} pass(es)")
    print(f"  saved to {audit_mod.report_path('.')}")
    print()
    return 0 if not gaps else 1


def cmd_keys(cfg: Config, args) -> int:
    """What do I need to connect? Answered from the code, not from memory."""
    items = integrations_mod.collect(".")
    marks = {"connected": "CONNECTED", "absent": "NOT SET", "public": "no key",
             "unsupported": "not wired", "not-applicable": "n/a",
             "unknown": "?"}
    print()
    print("  MAGICWAND INTEGRATIONS")
    print("  " + "-" * 74)
    for it in items:
        req = integrations_mod.REQUIREMENT_LABEL.get(it.requirement, it.requirement)
        status = marks.get(it.status, it.status)
        flag = "  " if it.requirement in ("none", "missing") else (
            "!!" if it.status != "connected" else "ok")
        print(f"  {flag} {it.name}")
        print(f"      need: {req:14s} state: {status:11s} {it.detail}")
        print(f"      does: {it.what_it_powers}")
        if it.where_to_get:
            print(f"      get:  {it.where_to_get}")
        if it.cost:
            print(f"      cost: {it.cost}")
        if it.notes and args.verbose:
            print(f"      note: {it.notes}")
        print()
    need = integrations_mod.missing_required(".")
    if need:
        print(f"  {len(need)} thing(s) needed and not connected:")
        for it in need:
            print(f"    - {it.name}: {it.where_to_get}")
        print()
        print("  Connect it in the console: Settings -> paste key -> Save -> Test connection")
    else:
        print("  Everything required is connected. Real agents and vision are live.")
    print()
    if not args.verbose:
        print("  (run with --verbose for permission and privacy notes)")
        print()
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
    l.add_argument("--equity", type=float, default=0.0)
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

    cm = sub.add_parser("compose", help="model composes a strategy; the gate judges it")
    cm.add_argument("--symbol")
    cm.add_argument("--exchange", choices=["coinbase", "binance", "kraken"])
    cm.add_argument("--candles", type=int, default=1500)
    cm.add_argument("--model", default="deepseek-flash")
    cm.add_argument("--offline", action="store_true")
    cm.set_defaults(fn=cmd_compose)

    fc = sub.add_parser("forecast", help="algorithmic trend forecast, honestly scored")
    fc.add_argument("--symbol")
    fc.add_argument("--exchange", choices=["coinbase", "binance", "kraken"],
                    help="candle source (binance has the deepest history)")
    fc.add_argument("--timeframe", default="",
                    help="bar size: 1h (default from config), 4h, 6h, 1d")
    fc.add_argument("--candles", type=int, default=3000)
    fc.add_argument("--horizon", type=int, default=6)
    fc.add_argument("--min-train", type=int, default=200)
    fc.add_argument("--block", type=int, default=50)
    fc.add_argument("--label", choices=["direction", "triple"],
                    default="direction",
                    help="direction: will price be higher? triple: was a "
                         "tradable move hit before the stop?")
    fc.add_argument("--groups", default="",
                    help="extra feature groups, comma separated: "
                         "cross,session,funding")
    fc.add_argument("--barrier-atr", type=float, default=1.0)
    fc.add_argument("--sweep", action="store_true",
                    help="run the label x feature-group matrix and compare")
    fc.add_argument("--labels", default="direction,triple",
                    help="labels for --sweep (default both)")
    fc.add_argument("--ladder", choices=["cumulative", "ends", "full"],
                    default="cumulative",
                    help="cumulative: price, +cross, +session, +funding. "
                         "ends: only price-only and all groups. full: only all "
                         "groups. Cheap options matter on long histories.")
    fc.add_argument("--by-regime", action="store_true",
                    help="fit a separate model per volatility regime and compare")
    fc.add_argument("--min-regime-train", type=int, default=60)
    fc.set_defaults(fn=cmd_forecast)

    jo = sub.add_parser("journal",
                        help="paper-run scorecard: trades, refusals, outages")
    jo.add_argument("--days", type=int, default=30)
    jo.add_argument("--save", action="store_true")
    jo.set_defaults(fn=cmd_journal)

    eg = sub.add_parser("edge",
                        help="measure the strategy edge and the size it justifies")
    eg.add_argument("--symbol", default="")
    eg.add_argument("--exchange", default="")
    eg.add_argument("--timeframe", default="")
    eg.add_argument("--candles", type=int, default=0)
    eg.set_defaults(fn=cmd_edge)

    sc = sub.add_parser("scan",
                        help="many markets x horizons, chance-corrected verdict")
    sc.add_argument("--symbols", default="", help="default: the whole registry")
    sc.add_argument("--horizons", default="6,24,72")
    sc.add_argument("--label", choices=["direction", "triple"], default="triple")
    sc.add_argument("--exchange", default="binance")
    sc.add_argument("--timeframe", default="")
    sc.add_argument("--candles", type=int, default=12000)
    sc.add_argument("--min-train", type=int, default=600)
    sc.add_argument("--block", type=int, default=1000)
    sc.set_defaults(fn=cmd_scan)

    ca = sub.add_parser("carry",
                        help="funding-rate carry: does the payment cover the cost?")
    ca.add_argument("--symbol", default="")
    ca.add_argument("--days", type=int, default=800)
    ca.add_argument("--candles", type=int, default=25000)
    ca.add_argument("--lookback", type=int, default=3)
    ca.add_argument("--entry-bps", type=float, default=1.0)
    ca.add_argument("--exit-bps", type=float, default=0.0)
    ca.add_argument("--spot-fee-bps", type=float, default=10.0)
    ca.add_argument("--perp-fee-bps", type=float, default=5.0)
    ca.add_argument("--slip-bps-leg", type=float, default=2.0)
    ca.set_defaults(fn=cmd_carry)

    au = sub.add_parser("audit", help="adversarial scenario self-audit")
    au.add_argument("--verbose", action="store_true")
    au.add_argument("--angle", choices=["all", "behaviour", "durability"],
                    default="all")
    au.set_defaults(fn=cmd_audit)

    k = sub.add_parser("keys", help="what API keys are needed, and what is connected")
    k.add_argument("--verbose", action="store_true")
    k.set_defaults(fn=cmd_keys)

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

    src = sub.add_parser("sources",
                         help="probe every data source and say what came back")
    src.add_argument("--symbol", default="", help="market to check (default: config)")
    src.add_argument("--json", action="store_true")
    src.set_defaults(fn=cmd_sources)

    stg = sub.add_parser("storage", help="local data: size, prune, backup, restore")
    stg.add_argument("--prune", action="store_true", help="delete old snapshots")
    stg.add_argument("--backup", action="store_true", help="write a local archive")
    stg.add_argument("--restore", metavar="ARCHIVE")
    stg.add_argument("--force", action="store_true", help="overwrite data/ on restore")
    stg.add_argument("--allow-running", action="store_true",
                     help="restore even while a console is running (dangerous)")
    stg.set_defaults(fn=cmd_storage)

    sv = sub.add_parser("serve", help="local web dashboard")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--live", action="store_true",
                    help="(deprecated) live is automatic once a key exists")
    sv.add_argument("--offline", action="store_true",
                    help="force the offline stub model even with a key")
    sv.add_argument("--auto", action="store_true",
                    help="start the autonomous scheduler immediately")
    sv.add_argument("--interval", type=int, default=900,
                    help="autopilot seconds between ticks (default 900)")
    sv.add_argument("--equity", type=float, default=0.0,
                    help="paper equity to run at (default: config). Logged, "
                         "because a balance is an assumption until a broker "
                         "confirms it")
    sv.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    cfg = _load_config(args.config)
    return args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
