"""MAGICWAND — the autonomous runner.

One tick, unattended, no human in the loop:

  1. fetch fresh market data
  2. grade any agent claims whose horizon has passed
  3. manage the desk's open position (stop / target / time)
  4. run the research cycle (veto check, agent claims recorded)
  5. optionally propose a new strategy and run it through the gate
  6. let the desk open an "assured" trade on a VALIDATED strategy
  7. write everything to the ledger

Step 6 is the only place a position is opened, and it requires a strategy that
passed `propose.validate` plus a research veto that did not fire. If no
validated strategy exists, the desk does nothing. That is the intended
behaviour: no proof, no trade — not "some proof, tiny trade".

The runner never waits for a person. It also never asks a model for permission
to trade: the model may invent, the backtest decides, the code executes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from . import data as data_mod
from . import research as research_mod
from . import research_loop
from .backtest import BacktestOptions, run_backtest
from .broker import check_exit, PaperBroker, Portfolio
from .common import Bar, ExitReason, iso, parse_iso, utcnow
from .config import Config
from .desk import AgentDesk, DeskConfig, DeskRefused
from .indicators import atr as atr_ind
from .ledger import Ledger
from .llm import DeepSeekClient, fake_proposal
from .reflect import PolicyStore, reflect_and_persist
from .propose import (MAX_PROPOSALS_PER_DAY, StrategyRegistry, proposals_today,
                      propose, validate)


@dataclass
class TickReport:
    ts: str
    symbol: str
    price: float = 0.0
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    research_decision: str = ""
    desk_equity: float = 0.0
    strategy_trades: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _total_equity(cfg: Config, root: Path) -> float:
    """Real account equity, falling back to the configured starting value.

    The desk is funded from this. Reading it from the live state file matters:
    funding off `cfg.risk.equity` would size the desk for a $50 account even
    when the paper account had grown (or shrunk) to something else.
    """
    p = root / cfg.ledger.state_path
    if p.exists():
        try:
            return float(json.loads(p.read_text()).get("equity", cfg.risk.equity))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return float(cfg.risk.equity)


def _registry_path(root: Path) -> Path:
    return root / "data" / "strategies.json"


def _policy_store(root: Path) -> PolicyStore:
    return PolicyStore(root / "data" / "policy.json")


def _memory_path(root: Path) -> Path:
    return root / "data" / "memory.json"


def load_memory(root: Path) -> dict:
    """Durable lessons: demoted families, evolved policy, last reflection."""
    p = _memory_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(root: Path, mem: dict) -> None:
    p = _memory_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem, indent=2))


def _load_strategy(cfg: Config, spec_dict: dict):
    from .strategy import StrategyConfig, build_parameterised
    params = dict(spec_dict.get("params", {}))
    return build_parameterised(spec_dict["family"], params)


def _manage_desk(cfg: Config, desk: AgentDesk, bars: list[Bar],
                 rep: TickReport) -> None:
    """Stops, targets, and time exits. Never needs a human."""
    pos = desk.state.position
    if pos is None:
        return
    last = bars[-1]
    pos.bars_held += 1
    hit = check_exit(pos, last)
    if hit is not None:
        kind = "stop" if hit.reason is ExitReason.STOP else "target"
        t = desk.close_trade(hit.price, hit.note, kind=kind, now=last.ts)
        if t is not None:
            rep.actions.append(f"desk CLOSED {kind.upper()}: net {t.net:+.4f} "
                               f"({t.return_pct:+.2%})")
        return
    if desk.dcfg.max_hold_bars and pos.bars_held >= desk.dcfg.max_hold_bars:
        t = desk.close_trade(last.close, "time stop", kind="time", now=last.ts)
        if t is not None:
            rep.actions.append(f"desk CLOSED time stop: net {t.net:+.4f} "
                               f"({t.return_pct:+.2%})")


def _maybe_propose(cfg: Config, client: DeepSeekClient, ledger: Ledger,
                   symbol: str, market_context: str, root: Path,
                   rep: TickReport, offline: bool = False,
                   blacklist: Optional[list[str]] = None) -> Optional[dict]:
    """Ask for a new strategy, validate it, register it only if it passes."""
    if proposals_today(ledger) >= MAX_PROPOSALS_PER_DAY:
        rep.notes.append(f"proposal budget used ({MAX_PROPOSALS_PER_DAY}/day)")
        return None
    try:
        spec = propose(client, symbol=symbol, market_context=market_context,
                       proposer_fake=fake_proposal if offline else None,
                       blacklist=blacklist)
    except Exception as e:  # noqa: BLE001 - a bad proposal must not stop the runner
        ledger.append("proposal", {"ok": False, "error": str(e)[:400]})
        rep.notes.append(f"proposal rejected before validation: {e}")
        return None

    bars = data_mod.fetch(cfg.market.exchange, spec.symbol or symbol,
                          cfg.market.timeframe, max(600, cfg.market.candles),
                          seed=cfg.seed)
    result = validate(spec, bars, cfg)
    ledger.append("proposal", {
        "ok": result.passed,
        "spec": spec.to_dict(),
        "summary": result.summary,
        "metrics": result.metrics,
        "gates": {k: {"ok": v[0], "detail": v[1]} for k, v in result.gates.items()},
    })
    if not result.passed:
        rep.notes.append(f"strategy '{spec.name}' rejected: {result.summary}")
        return None
    reg = StrategyRegistry(_registry_path(root))
    item = reg.add(spec, result)
    rep.actions.append(f"strategy '{spec.name}' PASSED the gate and was registered "
                       f"(holdout {result.metrics.get('holdout_return', 0):+.2%})")
    return item


def tick(cfg: Config, client: DeepSeekClient, dcfg: DeskConfig,
         root: str | Path = ".", propose_enabled: bool = True,
         subreddit: str = "solana",
         snapshot: Optional[research_mod.MarketSnapshot] = None,
         offline: bool = False, allow_stale_strategy: bool = False,
         reflect_enabled: bool = True) -> TickReport:
    root = Path(root)
    ledger = Ledger(root / cfg.ledger.path)
    symbol = cfg.market.symbol
    bars = data_mod.fetch(cfg.market.exchange, symbol, cfg.market.timeframe,
                          cfg.market.candles, seed=cfg.seed)
    if not bars:
        raise RuntimeError("no bars returned; refusing to tick on empty data")
    price = bars[-1].close
    a = atr_ind(bars, cfg.strategy.atr_period)
    atr = a[-1] or price * 0.01
    rep = TickReport(ts=iso(bars[-1].ts), symbol=symbol, price=price)

    # 2. grade matured agent claims
    graded = research_loop.grade_agent_claims(ledger, symbol, price)
    if graded:
        rep.actions.append(f"graded {graded} agent claim(s)")

    # 3. manage the desk
    desk = AgentDesk(cfg, dcfg, root=root)
    if dcfg.enabled:
        desk.fund(_total_equity(cfg, root))
        _manage_desk(cfg, desk, bars, rep)

    # 4. research
    if snapshot is None:
        if cfg.market.exchange == "synthetic":
            # Offline/demo mode: never touch the network. A fixture snapshot
            # keeps the tick fast and deterministic, and stops tests from
            # hammering free endpoints.
            snapshot = research_mod.fixture_snapshot(symbol, price)
        else:
            base = symbol.split("-")[0]
            snapshot = research_mod.build_snapshot(
                base, subreddit=subreddit, price=price, cache_dir=str(root))
    cycle = research_loop.run_cycle(
        cfg, client, price=price, atr=atr, symbol=symbol, subreddit=subreddit,
        dcfg=None, root=root, snapshot=snapshot, ledger=ledger)
    rep.research_decision = cycle.report.decision
    rep.notes.append(f"research: {cycle.report.summary}")

    # 5. reflect on the record, then propose (rate-limited)
    mem = load_memory(root)
    if reflect_enabled:
        ref = reflect_and_persist(ledger, _policy_store(root))
        mem["blacklisted_families"] = ref.blacklisted_families
        mem["preferred_families"] = ref.preferred_families
        mem["agent_weights"] = ref.agent_weights
        mem["last_reflection"] = ref.to_dict()
        save_memory(root, mem)
        if ref.changes:
            rep.actions.append(ref.summary)

    market_context = snapshot.to_prompt(max_news=8, max_posts=4)
    if propose_enabled and client.available():
        _maybe_propose(cfg, client, ledger, symbol, market_context, root, rep,
                       offline=offline,
                       blacklist=mem.get("blacklisted_families", []))

    # 6. trade ONLY a validated strategy, and only if research did not veto
    rep.desk_equity = desk.equity(price) if dcfg.enabled else 0.0

    # Record a mark every tick. Without this the equity chart has no series at
    # all, because only the manual `step` path used to write marks.
    try:
        account = _total_equity(cfg, root)
        ledger.append("mark", {
            "equity": round(account, 6),
            "price": round(price, 6),
            "desk_equity": round(rep.desk_equity, 6),
            "by": "tick",
        })
    except Exception:  # noqa: BLE001 - a mark is bookkeeping, never fatal
        pass
    if not dcfg.enabled:
        return rep
    if desk.state.position is not None:
        rep.notes.append("desk already holds a position; not opening another")
        return rep
    if cycle.report.is_veto:
        rep.notes.append(f"desk stood down: research veto by "
                         f"{', '.join(cycle.report.vetoed_by)}")
        return rep

    reg = StrategyRegistry(_registry_path(root))
    best = reg.best()
    if best is None:
        rep.notes.append("desk stood down: no strategy has passed the gate yet")
        return rep
    spec_dict = best["spec"]
    try:
        strat = _load_strategy(cfg, spec_dict)
    except ValueError as e:
        rep.notes.append(f"registered strategy unreadable: {e}")
        return rep

    # Does the validated strategy want to be long right now, on fresh data?
    intent = strat.decide(bars, len(bars) - 1, in_position=False)
    if intent is None or intent.action != "enter":
        rep.notes.append(f"strategy '{spec_dict.get('name')}' says wait")
        return rep

    # And does it still work on the latest data? A strategy that passed once
    # and no longer does is stale, and a stale strategy is not traded.
    #
    # `allow_stale_strategy` exists ONLY so tests can prove the wiring of the
    # trade branch itself. The CLI never sets it, so unattended runs always
    # revalidate. If you find yourself reaching for it in production, you have
    # already disabled the thing that keeps this safe.
    if not allow_stale_strategy:
        fresh = validate(_spec_from_dict(spec_dict), bars, cfg)
        if not fresh.passed:
            rep.notes.append(f"strategy '{spec_dict.get('name')}' no longer passes "
                             f"validation: {fresh.summary}")
            ledger.append("strategy_stale", {"name": spec_dict.get("name"),
                                             "summary": fresh.summary})
            return rep

    out = desk.open_trade(
        symbol=symbol, price=price, atr=atr,
        agent=f"strategy:{spec_dict.get('name')}",
        reason=(f"[{spec_dict.get('name')}] {spec_dict.get('hypothesis', '')[:200]} | "
                f"signal: {intent.reason[:200]}"),
        expected_move=intent.expected_move,
        confidence=1.0,   # the backtest is the confidence; no model self-report
    )
    if out.get("ok"):
        reg.mark_deployed(spec_dict.get("name", ""))
        rep.actions.append("desk TRADE: " + out.get("detail", ""))
    else:
        rep.notes.append(f"desk refused: {out.get('reason')}")
    return rep


def _spec_from_dict(d: dict):
    from .propose import StrategySpec
    return StrategySpec(**d)


def run(cfg: Config, client: DeepSeekClient, dcfg: DeskConfig,
        root: str | Path = ".", interval: int = 300, max_ticks: int = 0,
        propose_enabled: bool = True, subreddit: str = "solana",
        offline: bool = False, reflect_enabled: bool = True) -> int:
    """The unattended loop. Ctrl-C to stop; state survives restarts."""
    print(f"MAGICWAND autonomous runner — {cfg.market.symbol} @ {cfg.market.timeframe}")
    print(f"  interval {interval}s   desk enabled: {dcfg.enabled}   "
          f"propose: {propose_enabled}")
    print("  paper only. No real money. Ctrl-C to stop.\n")
    n = 0
    while True:
        n += 1
        try:
            rep = tick(cfg, client, dcfg, root=root,
                       propose_enabled=propose_enabled, subreddit=subreddit,
                       offline=offline, reflect_enabled=reflect_enabled)
            stamp = utcnow().strftime("%H:%M:%S")
            print(f"[{stamp}] price {rep.price:.4f}  desk {rep.desk_equity:.2f}  "
                  f"research {rep.research_decision or '-'}")
            for a_ in rep.actions:
                print(f"    + {a_}")
            for n_ in rep.notes:
                print(f"    · {n_}")
        except KeyboardInterrupt:
            print("\nstopped by operator.")
            return 0
        except Exception as e:  # noqa: BLE001 - the loop must survive a bad tick
            print(f"[{utcnow():%H:%M:%S}] tick failed: {type(e).__name__}: {e}")
        if max_ticks and n >= max_ticks:
            print(f"\nfinished {n} tick(s).")
            return 0
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nstopped by operator.")
            return 0
