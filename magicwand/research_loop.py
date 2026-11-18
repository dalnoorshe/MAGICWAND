"""MAGICWAND — research loop.

Wires the pieces together:

    snapshot -> agents -> synthesis -> [veto the strategy | record support]
                                     -> [optionally: the desk may act]

Every agent verdict is written to the ledger as a claim with a horizon, so it
can be graded like everything else. That is what turns "four agents had
opinions" into a track record you can inspect and act on.

There is no code path here that opens a position in the main account. Only
the desk can open a trade, and only inside its walls.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional

from . import research as research_mod
from .agents import (AgentVerdict, ResearchReport, agent_names, run_all,
                     synthesize)
from .common import iso, parse_iso, utcnow
from .config import Config
from .desk import AgentDesk, DeskConfig
from .ledger import Ledger, grade_claim, make_claim, scorecard
from .llm import DeepSeekClient
from .vision_agent import read_chart


# --------------------------------------------------------------------------
# Agent track record
# --------------------------------------------------------------------------

# A reader needs this many INDEPENDENT price paths before its influence moves
# away from neutral. Claims graded together share a price path, so 575 claims
# over 31 distinct realized moves are 31 observations, not 575.
MIN_WINDOWS = 20
# Pseudo-count pulling a small sample toward neutral influence.
SHRINK = 10.0


def _influence_from(skill: float, windows: int) -> float:
    """Map measured skill to influence, shrunk toward neutral.

    Deliberately asymmetric in effect: a reader with negative skill gets less
    say, never an INVERTED say. Flipping a bad reader is an overfit dressed up
    as cleverness, and it fails the first time the market changes character.
    """
    if windows < MIN_WINDOWS:
        return 0.5
    shrink = windows / (windows + SHRINK)
    return max(0.05, min(1.0, 0.5 + skill * 1.2 * shrink))


def agent_track_record(ledger: Ledger) -> dict:
    """Per-reader SKILL, not per-reader accuracy.

    A raw hit rate is meaningless on its own, and this project had exactly that
    bug: over a two-week sample where 76% of 24h windows fell, every reader
    leaned bearish, so "always bearish" scored 76%. Two readers outscored that
    baseline by nothing at all while carrying influence weights of 0.62 and
    0.60, and one reader with 26 claims scored 88% and got 0.96.

    So each reader is scored against the base rate for the direction it called,
    over the same horizon, and the sample is counted in distinct price paths.
    Everything here is recomputable from the ledger by hand.
    """
    claims: dict[str, dict] = {}
    for e in ledger.entries():
        p = e.get("payload") or {}
        if e.get("kind") == "agent_claim":
            claims[p.get("claim_id")] = p
    graded: list[tuple[dict, dict]] = []
    for e in ledger.entries():
        if e.get("kind") != "grade":
            continue
        p = e.get("payload") or {}
        src = claims.get(p.get("claim_id"))
        if src is not None:
            graded.append((src, p))
    if not graded:
        return {}

    # How often the price actually rose, per horizon. This is the baseline a
    # reader has to beat: saying "down" in a down market is not a forecast.
    rose: dict[int, int] = {}
    total: dict[int, int] = {}
    for src, g in graded:
        h = int(src.get("horizon_bars") or 24)
        total[h] = total.get(h, 0) + 1
        if float(g.get("realized_move") or 0.0) > 0:
            rose[h] = rose.get(h, 0) + 1
    p_up = {h: (rose.get(h, 0) / n) for h, n in total.items() if n}

    stats: dict[str, dict] = {}
    for src, g in graded:
        agent = src.get("agent", "unknown")
        s = stats.setdefault(agent, {"n": 0, "right": 0, "wrong": 0,
                                     "overstated": 0, "flat": 0,
                                     "_base": [], "_paths": set()})
        s["n"] += 1
        verdict = g.get("verdict", "flat")
        if verdict in s:
            s[verdict] += 1
        h = int(src.get("horizon_bars") or 24)
        up = p_up.get(h, 0.5)
        expected = float(src.get("expected_move") or 0.0)
        s["_base"].append(up if expected > 0 else 1.0 - up)
        s["_paths"].add(round(float(g.get("realized_move") or 0.0), 6))

    out: dict[str, dict] = {}
    for agent, s in stats.items():
        n = s["n"]
        windows = len(s["_paths"])
        hit = s["right"] / n if n else 0.0
        base = (sum(s["_base"]) / n) if n else 0.0
        skill = hit - base
        # Standard error of the NULL model (the base rate), not of the
        # observed hit rate: at hit = 100% the observed variance is zero, which
        # would make every perfect-but-tiny sample "significant".
        p0 = max(0.0, min(1.0, base))
        se = ((p0 * (1.0 - p0) / windows) ** 0.5) if windows else 0.0
        out[agent] = {
            "n": n,
            "windows": windows,
            "right": s["right"],
            "wrong": s["wrong"],
            "overstated": s["overstated"],
            "flat": s["flat"],
            "hit_rate": round(hit, 4),
            "base_rate": round(base, 4),
            "skill": round(skill, 4),
            "se": round(se, 4),
            # Positive skill this large is unlikely to be luck; it is still not
            # proof, so it only moves influence, never a trade.
            "significant": bool(skill > 2.0 * se),
            "influence": round(_influence_from(skill, windows), 3),
        }
    return out


# --------------------------------------------------------------------------
# Verdict -> ledger claims
# --------------------------------------------------------------------------

def record_verdicts(ledger: Ledger, report: ResearchReport, price: float,
                    when=None) -> list[str]:
    """Persist each reader's forecast as a gradable claim. Returns claim ids.

    The claim is ALWAYS stamped with the moment the forecast was made, never
    with the snapshot timestamp. Otherwise a stale snapshot would backdate
    every claim and grade it immediately against a price it already knew about
    — a lookahead bug hiding inside the honesty machinery.
    """
    when = when or utcnow()
    ids: list[str] = []
    for v in report.agents:
        if v.stub:
            continue          # a placeholder must not enter the track record
        if not v.trusted or not v.claims:
            continue
        # Use the agent's first concrete forecast; that is what gets graded.
        c = v.claims[0]
        expected = c.expected_move()
        if expected == 0.0:
            continue
        cid = f"agent-{v.agent}-{ledger._seq}"
        ledger.append("agent_claim", {
            "claim_id": cid,
            "agent": v.agent,
            "symbol": report.symbol,
            "statement": c.statement,
            "direction": c.direction,
            "confidence": round(c.confidence, 3),
            "expected_move": round(expected, 6),
            "horizon_bars": c.horizon_hours,
            "entry_price": price,
            "stance": v.stance,
            "thesis": v.thesis,
            "ts": iso(when),
            "note": "agent forecast; graded automatically when the horizon passes",
        }, when=when)
        ids.append(cid)
    return ids


def grade_agent_claims(ledger: Ledger, symbol: str, price: float,
                       now=None) -> int:
    """Grade any agent claim whose horizon has elapsed. Returns how many."""
    now = now or utcnow()
    graded = {e["payload"].get("claim_id") for e in ledger.entries()
              if e.get("kind") == "grade"}
    pending = [e["payload"] for e in ledger.entries()
               if e.get("kind") == "agent_claim"
               and e["payload"].get("claim_id") not in graded]
    n = 0
    for c in pending:
        try:
            age_h = (now - parse_iso(c["ts"])).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        if age_h < float(c.get("horizon_bars", 24)):
            continue
        entry_price = float(c.get("entry_price", 0.0))
        if entry_price <= 0:
            continue
        realized = (price - entry_price) / entry_price
        ledger.append("grade", grade_claim(c, realized, int(age_h), "horizon"),
                      when=now)
        n += 1
    return n


# --------------------------------------------------------------------------
# One research cycle
# --------------------------------------------------------------------------

@dataclass
class ResearchCycle:
    report: ResearchReport
    recorded_claims: list[str] = field(default_factory=list)
    graded: int = 0
    desk_outcome: Optional[dict] = None
    track_record: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "report": asdict(self.report),
            "recorded_claims": self.recorded_claims,
            "graded": self.graded,
            "desk_outcome": self.desk_outcome,
            "track_record": self.track_record,
        }


INFLIGHT_NAME = "research-inflight.json"
# A marker older than this means the process died mid-cycle, not that a cycle
# is still running. Reporting a dead run as "researching now" would be a lie.
INFLIGHT_MAX_AGE = 900.0


def inflight_path(root: str | Path) -> Path:
    return Path(root) / "data" / INFLIGHT_NAME


def brief_verdict(v: AgentVerdict) -> dict:
    """The parts of a verdict worth painting while the cycle is still running."""
    def num(x: object) -> float:
        try:
            got = float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return got if math.isfinite(got) else 0.0

    return {
        "agent": str(v.agent),
        "stance": str(v.stance),
        "confidence": round(num(v.confidence), 3),
        "evidence_score": round(num(v.evidence_score), 3),
        "corroboration": int(num(v.corroboration)),
        "coverage": round(num(getattr(v, "coverage", 1.0)), 3),
        "thesis": (v.thesis or "")[:280],
        "error": (v.error or "")[:200],
    }


def mark_inflight(root: str | Path, symbol: str = "",
                  agents: Optional[list[str]] = None,
                  now: Optional[datetime] = None,
                  plan: Optional[list[str]] = None) -> None:
    """Say out loud that a research cycle is running, on disk.

    The console cannot otherwise tell "the last cycle finished 4 minutes ago"
    from "one is running right now", so the panel looked frozen either way.
    Written atomically and failure is ignored: a progress marker must never be
    able to break the research it is describing.
    """
    p = inflight_path(root)
    payload = {"started_at": iso(now or utcnow()), "symbol": symbol,
               "agents": list(agents or []), "plan": list(plan or []),
               "phase": "starting", "current": "", "done": [], "failing": []}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except OSError:
        pass


def update_inflight(root: str | Path, **fields: Any) -> None:
    """Merge progress into the marker, atomically. Never raises.

    A cycle takes a minute of model calls; the console should be able to show
    which reader is being read right now instead of a static "running" badge.
    """
    p = inflight_path(root)
    try:
        cur = json.loads(p.read_text())
    except (OSError, ValueError):
        cur = {}
    if not isinstance(cur, dict):
        cur = {}
    cur.setdefault("started_at", iso(utcnow()))
    cur.update(fields)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(cur))
        tmp.replace(p)
    except OSError:
        pass


def clear_inflight(root: str | Path) -> None:
    try:
        inflight_path(root).unlink()
    except OSError:
        pass


def read_inflight(root: str | Path, now: Optional[datetime] = None,
                  max_age: float = INFLIGHT_MAX_AGE) -> Optional[dict]:
    """The running cycle, or None.

    Fail soft in every direction — absent, corrupt, unreadable or stale all
    mean "not running". A corrupt file must not raise inside build_state and
    take the whole console down with it.
    """
    p = inflight_path(root)
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    started = raw.get("started_at")
    try:
        age = ((now or utcnow()) - parse_iso(str(started))).total_seconds()
    except (TypeError, ValueError):
        return None
    if not (age >= 0 and age <= max_age):
        return None
    plan = raw.get("plan")
    done = raw.get("done")
    failure = raw.get("failing")
    return {"started_at": started, "age_seconds": round(age, 1),
            "symbol": raw.get("symbol", ""),
            "agents": raw.get("agents", []),
            "phase": str(raw.get("phase", "")),
            "current": str(raw.get("current", "")),
            "plan": [str(x) for x in plan] if isinstance(plan, list) else [],
            "done": [d for d in done if isinstance(d, dict) and d.get("agent")]
                    if isinstance(done, list) else [],
            "failing": [str(x) for x in failure]
                       if isinstance(failure, list) else []}


def _run_cycle_body(cfg: Config, client: DeepSeekClient, price: float,
                    atr: float, symbol: Optional[str] = None,
                    subreddit: str = "solana", dcfg: Optional[DeskConfig] = None,
                    root: str | Path = ".",
                    snapshot: Optional[research_mod.MarketSnapshot] = None,
                    agents: Optional[list[str]] = None,
                    now: Optional[datetime] = None,
                    ledger: Optional[Ledger] = None,
                    vision: bool = False,
                    bars: Optional[list] = None,
                    chart_title: str = "",
                    position: Optional[dict] = None,
                    progress: Optional[Callable[..., None]] = None
                    ) -> ResearchCycle:
    root = Path(root)
    now = now or utcnow()
    symbol = symbol or cfg.market.symbol
    # Share one Ledger per call: several instances racing on one file is what
    # broke the hash chain, and the chain is the thing we cannot afford to lose.
    ledger = ledger or Ledger(root / cfg.ledger.path)
    base = symbol.split("-")[0] if "-" in symbol else symbol

    # Live progress, so the console can show a cycle happening instead of
    # replaying the previous one. Observation only: nothing here can alter a
    # verdict, and every callback is optional.
    def note(**fields: Any) -> None:
        if progress is not None:
            progress(**fields)

    finished: list[dict] = []

    if snapshot is None:
        # Fetching the sources takes real time (RSS, exchange APIs); say so
        # rather than looking stalled with an empty roster.
        note(phase="snapshot", current="")
        snapshot = research_mod.build_snapshot(
            base, subreddit=subreddit, price=price,
            cache_dir=str(root) if cfg.market.exchange != "synthetic" else None)

    track = agent_track_record(ledger)

    def on_plan(names: list[str]) -> None:
        note(phase="reading", current="", plan=list(names), done=[],
             failing=[])

    def on_start(name: str) -> None:
        note(phase="reading", current=name)

    def on_done(v: AgentVerdict) -> None:
        finished.append(brief_verdict(v))
        note(phase="reading", current="", done=list(finished))

    verdicts = run_all(snapshot, client, agents=agents, on_plan=on_plan,
                       on_start=on_start, on_done=on_done)

    # The chart reader sees a picture of the same series. It fails soft: if the
    # model cannot see (offline fake, no key), the verdict is degraded and the
    # text readers carry on.
    chart_verdict = None
    if vision and bars:
        note(phase="chart", current="charts")
        chart_verdict = read_chart(
            bars, client, title=chart_title or f"{symbol} {cfg.market.timeframe}",
            symbol=symbol, save_dir=root / "data" / "charts",
            position=position)
        if chart_verdict.error:
            ledger.append("vision_degraded",
                          {"agent": "charts", "error": chart_verdict.error[:300]})
            note(phase="chart", current="", failing=["charts"])
        else:
            verdicts.append(chart_verdict)
            finished.append(brief_verdict(chart_verdict))
            note(phase="chart", current="", done=list(finished))

    note(phase="synthesis", current="")
    report = synthesize(verdicts, symbol, snapshot.ts, track=track)
    report.chart = chart_verdict
    report.cost_usd = round(client.usage.cost_usd, 6)
    report.calls = client.usage.calls

    # Grade anything that matured before recording new claims.
    graded = grade_agent_claims(ledger, symbol, price, now=now)
    recorded = record_verdicts(ledger, report, price, when=now)

    # Persist the verdicts themselves, not just the summary: the dashboard has
    # to be able to show WHY a decision was made after a restart.
    ledger.append("research", {
        "ts": report.ts,
        "symbol": symbol,
        "decision": report.decision,
        "evidence_score": report.evidence_score,
        "coverage": report.coverage,
        "vetoed_by": report.vetoed_by,
        "single_source_declined": report.single_source,
        "summary": report.summary,
        "calls": report.calls,
        "cost_usd": report.cost_usd,
        "sources_missing": snapshot.missing(),
        "stub": bool(verdicts) and all(getattr(v, "stub", False) for v in verdicts),
        # What the image actually contained, so the console can say why an
        # hourly chart looks unchanged for an hour even though it is redrawn
        # on every tick.
        "chart_bars": ({"shown": min(len(bars), 120),
                        "last_bar_utc": iso(bars[-1].ts),
                        "timeframe": cfg.market.timeframe}
                       if bars else None),
        "chart": None if chart_verdict is None else {
            "stance": chart_verdict.stance,
            "confidence": chart_verdict.confidence,
            "thesis": chart_verdict.thesis,
            "observations": chart_verdict.observations,
            "image_path": chart_verdict.image_path,
            "injected_text_seen": chart_verdict.injected_text_seen,
            "error": chart_verdict.error,
        },
        "agents": [{
            "agent": v.agent,
            "stance": v.stance,
            "confidence": v.confidence,
            "evidence_score": v.evidence_score,
            "coverage": round(float(v.coverage), 3),
            "thesis": v.thesis[:400],
            "corroboration": v.corroboration,
            "error": v.error,
            "warnings": v.warnings[:3],
        } for v in report.agents],
    })

    # The desk may act, but only on an assured support verdict.
    desk_outcome = None
    if dcfg is not None and dcfg.enabled:
        desk = AgentDesk(cfg, dcfg, root=root)
        pf_path = root / cfg.ledger.state_path
        total_equity = cfg.risk.equity
        if pf_path.exists():
            try:
                total_equity = float(json.loads(pf_path.read_text()).get(
                    "equity", cfg.risk.equity))
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        desk.fund(total_equity)
        confident = max((v for v in verdicts
                         if v.trusted and v.stance == "support"),
                        key=lambda v: v.confidence, default=None)
        if (report.decision == "support" and confident is not None
                and confident.confidence >= dcfg.min_confidence
                and confident.claims):
            c = confident.claims[0]
            desk_outcome = desk.open_trade(
                symbol=symbol, price=price, atr=atr, agent=confident.agent,
                reason=f"{confident.agent}: {confident.thesis[:300]}",
                expected_move=c.expected_move(),
                confidence=confident.confidence)
        else:
            desk_outcome = {"ok": False, "reason": (
                f"no assured support verdict (decision={report.decision}, "
                f"best confidence="
                f"{max((v.confidence for v in verdicts), default=0.0):.2f})")}

    return ResearchCycle(
        report=report,
        recorded_claims=recorded,
        graded=graded,
        desk_outcome=desk_outcome,
        track_record=agent_track_record(ledger),
    )


def run_cycle(cfg: Config, client: DeepSeekClient, price: float,
              atr: float, symbol: Optional[str] = None,
              subreddit: str = "solana", dcfg: Optional[DeskConfig] = None,
              root: str | Path = ".", snapshot: Optional[research_mod.MarketSnapshot] = None,
              agents: Optional[list[str]] = None,
              now: Optional[datetime] = None,
              ledger: Optional[Ledger] = None,
              vision: bool = False,
              bars: Optional[list] = None,
              chart_title: str = "",
              position: Optional[dict] = None) -> ResearchCycle:
    """Run one research cycle, with an on-disk "I am running" marker.

    The marker exists so the console can show a cycle in progress instead of
    silently replaying the previous one, and it is cleared in a `finally` so a
    crash cannot leave the panel claiming work that stopped.
    """
    def progress(**fields: Any) -> None:
        update_inflight(root, **fields)

    plan = list(agents) if agents else agent_names()
    mark_inflight(root, symbol=symbol or cfg.market.symbol,
                  agents=agents, now=now, plan=plan)
    try:
        return _run_cycle_body(
            cfg, client, price, atr, symbol, subreddit, dcfg, root, snapshot,
            agents, now, ledger, vision, bars, chart_title, position, progress)
    finally:
        clear_inflight(root)
