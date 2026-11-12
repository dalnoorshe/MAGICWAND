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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional

from . import research as research_mod
from .agents import (AgentVerdict, ResearchReport, run_all, synthesize)
from .common import iso, parse_iso, utcnow
from .config import Config
from .desk import AgentDesk, DeskConfig
from .ledger import Ledger, grade_claim, make_claim, scorecard
from .llm import DeepSeekClient
from .vision_agent import read_chart


# --------------------------------------------------------------------------
# Agent track record
# --------------------------------------------------------------------------

def agent_track_record(ledger: Ledger) -> dict:
    """Per-agent hit rate from graded claims. The basis for influence."""
    by_id: dict[str, dict] = {}
    for e in ledger.entries():
        p = e.get("payload", {}) or {}
        if e.get("kind") == "agent_claim":
            by_id[p.get("claim_id")] = p
    stats: dict[str, dict] = {}
    for e in ledger.entries():
        if e.get("kind") != "grade":
            continue
        p = e.get("payload", {}) or {}
        src = by_id.get(p.get("claim_id"))
        if not src:
            continue
        agent = src.get("agent", "unknown")
        s = stats.setdefault(agent, {"n": 0, "right": 0, "wrong": 0,
                                     "overstated": 0, "flat": 0})
        s["n"] += 1
        verdict = p.get("verdict", "flat")
        if verdict in s:
            s[verdict] += 1
    for agent, s in stats.items():
        s["hit_rate"] = round(s["right"] / s["n"], 4) if s["n"] else 0.0
    return stats


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


def run_cycle(cfg: Config, client: DeepSeekClient, price: float,
              atr: float, symbol: Optional[str] = None,
              subreddit: str = "solana", dcfg: Optional[DeskConfig] = None,
              root: str | Path = ".", snapshot: Optional[research_mod.MarketSnapshot] = None,
              agents: Optional[list[str]] = None,
              now: Optional[datetime] = None,
              ledger: Optional[Ledger] = None,
              vision: bool = False,
              bars: Optional[list] = None,
              chart_title: str = "") -> ResearchCycle:
    root = Path(root)
    now = now or utcnow()
    symbol = symbol or cfg.market.symbol
    # Share one Ledger per call: several instances racing on one file is what
    # broke the hash chain, and the chain is the thing we cannot afford to lose.
    ledger = ledger or Ledger(root / cfg.ledger.path)
    base = symbol.split("-")[0] if "-" in symbol else symbol

    if snapshot is None:
        snapshot = research_mod.build_snapshot(
            base, subreddit=subreddit, price=price,
            cache_dir=str(root) if cfg.market.exchange != "synthetic" else None)

    track = agent_track_record(ledger)
    verdicts = run_all(snapshot, client, agents=agents)

    # The chart reader sees a picture of the same series. It fails soft: if the
    # model cannot see (offline fake, no key), the verdict is degraded and the
    # text readers carry on.
    chart_verdict = None
    if vision and bars:
        chart_verdict = read_chart(
            bars, client, title=chart_title or f"{symbol} {cfg.market.timeframe}",
            symbol=symbol, save_dir=root / "data" / "charts")
        if chart_verdict.error:
            ledger.append("vision_degraded",
                          {"agent": "charts", "error": chart_verdict.error[:300]})
        else:
            verdicts.append(chart_verdict)

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
        "vetoed_by": report.vetoed_by,
        "summary": report.summary,
        "calls": report.calls,
        "cost_usd": report.cost_usd,
        "sources_missing": snapshot.missing(),
        "stub": bool(verdicts) and all(getattr(v, "stub", False) for v in verdicts),
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
            "thesis": v.thesis[:400],
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
