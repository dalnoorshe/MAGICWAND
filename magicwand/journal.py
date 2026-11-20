"""MAGICWAND — the paper journal.

The point of a 30-day paper run is not to make money. It is to find out whether
the machine refuses to trade **for the right reasons**, and whether it records
those reasons well enough to check a month later.

So this module answers four questions, from the ledger alone:

  1. Did anything trade? If not, what stopped it, and how often?
  2. Did the equity mark drift, and by how much?
  3. What did the research cost, and did it change its mind?
  4. How many days had NO decision that could be traded — and was that a refusal
     or a silence?

That last one is the interesting distinction. A day with no trade because the
gate refused is a working system. A day with no trade because nothing ran is an
outage, and it must not look the same in the report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .common import iso, parse_iso, utcnow
from .config import Config
from .ledger import Ledger


def _day_key(when: str) -> str:
    try:
        return parse_iso(when).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return "unknown"


@dataclass
class DayRow:
    day: str
    ticks: int = 0
    research: int = 0
    proposals: int = 0
    trades: int = 0
    vetoes: int = 0
    refusals: dict = field(default_factory=dict)
    equity_end: Optional[float] = None
    cost_usd: float = 0.0
    # The current day is not finished, so silence in it is not an outage yet.
    in_progress: bool = False

    def note(self) -> str:
        if self.trades:
            return f"{self.trades} trade(s)"
        if self.ticks == 0 and self.research == 0:
            if self.in_progress:
                return "in progress — nothing recorded yet today"
            return "NOTHING RAN — outage, not a refusal"
        if self.vetoes:
            top = (max(self.refusals.items(), key=lambda kv: kv[1])[0]
                   if self.refusals else "unspecified")
            return f"no trade: {self.vetoes} refusal(s), mostly {top}"
        if self.research:
            return "no trade: research ran, no entry signal"
        return "no trade: no signal, nothing refused"

    def to_dict(self) -> dict:
        return {"day": self.day, "ticks": self.ticks, "research": self.research,
                "proposals": self.proposals, "trades": self.trades,
                "vetoes": self.vetoes, "refusals": self.refusals,
                "equity_end": self.equity_end, "cost_usd": round(self.cost_usd, 6),
                "in_progress": self.in_progress, "note": self.note()}


def build_journal(root: str | Path, cfg: Optional[Config] = None,
                  days: int = 30, now: Optional[datetime] = None) -> dict:
    """Summarise the last `days` of ledger activity. Never raises on odd data."""
    cfg = cfg or Config()
    root = Path(root)
    ledger = Ledger(root / cfg.ledger.path)
    now = now or utcnow()
    start = now - timedelta(days=days)

    rows: dict[str, DayRow] = {}
    marks: list[tuple[datetime, float]] = []
    trade_count = 0
    trade_net = 0.0
    wins = 0
    total_cost = 0.0
    tiers: list[tuple[datetime, str]] = []

    for e in ledger.entries():
        when_raw = e.get("ts") or (e.get("payload") or {}).get("ts") or ""
        try:
            when = parse_iso(when_raw)
        except (ValueError, TypeError):
            continue
        if when < start:
            continue
        day = _day_key(when_raw)
        row = rows.setdefault(day, DayRow(day=day))
        kind = e.get("kind")
        p = e.get("payload") or {}
        if kind == "mark":
            eq = p.get("equity")
            if isinstance(eq, (int, float)):
                row.equity_end = float(eq)
                marks.append((when, float(eq)))
            row.ticks += 1
        elif kind == "research":
            row.research += 1
            cost = p.get("cost_usd")
            if isinstance(cost, (int, float)):
                row.cost_usd += float(cost)
                total_cost += float(cost)
        elif kind == "proposal":
            row.proposals += 1
        elif kind == "veto":
            row.vetoes += 1
            reason = str(p.get("reason") or "unspecified")[:90]
            row.refusals[reason] = row.refusals.get(reason, 0) + 1
        elif kind == "tick_request":
            row.ticks += 1
        elif kind == "tier_change":
            tiers.append((when, str(p.get("to_name") or p.get("to") or "?")))
        elif kind in ("fill", "trade"):
            row.trades += 1
            trade_count += 1
            net = p.get("net")
            if isinstance(net, (int, float)):
                trade_net += float(net)
                if net > 0:
                    wins += 1

    # Every day in the effective window gets a row, including days with NO
    # ledger entries at all. Without this a run that was off for a week looks
    # like a clean six-day run, and the report quietly hides the outage — the
    # same failure mode as a chart that redraws itself into looking prescient.
    first_activity = None
    for e in ledger.entries():
        raw = e.get("ts") or ""
        try:
            first_activity = parse_iso(raw)
        except (ValueError, TypeError):
            continue
        break
    effective_start = start
    if first_activity and first_activity > start:
        effective_start = first_activity.replace(hour=0, minute=0, second=0,
                                                 microsecond=0)
    day = effective_start
    while day <= now:
        rows.setdefault(day.strftime("%Y-%m-%d"),
                        DayRow(day=day.strftime("%Y-%m-%d")))
        day += timedelta(days=1)

    today = now.strftime("%Y-%m-%d")
    if today in rows:
        rows[today].in_progress = True

    ordered = [rows[k] for k in sorted(rows)]
    equity_first = marks[0][1] if marks else None
    equity_last = marks[-1][1] if marks else None
    change = ((equity_last - equity_first)
              if equity_first not in (None, 0) and equity_last is not None else None)
    pct = (change / equity_first * 100.0
           if change is not None and equity_first else None)
    outages = [r.day for r in ordered
               if r.ticks == 0 and r.research == 0 and not r.in_progress]
    refused_days = [r.day for r in ordered if r.trades == 0 and r.vetoes > 0]
    silent_days = [r.day for r in ordered if r.trades == 0 and r.vetoes == 0
                   and (r.ticks or r.research)]

    top_refusals = Counter()
    for r in ordered:
        top_refusals.update(r.refusals)

    if trade_count == 0 and refused_days:
        verdict = (f"no trades in {len(ordered)} day(s), and every quiet day has a "
                   f"refusal behind it — the gate is doing its job")
    elif trade_count == 0 and outages:
        verdict = (f"no trades and {len(outages)} day(s) with nothing running — "
                   f"that is an outage, not a decision")
    elif trade_count == 0:
        verdict = "no trades and nothing refused: no signal ever fired"
    else:
        verdict = f"{trade_count} trade(s), net {trade_net:+.2f} on the paper desk"

    return {
        "generated_at": iso(now),
        "window_days": days,
        "first_day": ordered[0].day if ordered else None,
        "last_day": ordered[-1].day if ordered else None,
        "days": [r.to_dict() for r in ordered],
        "days_covered": len(ordered),
        "equity_first": equity_first,
        "equity_last": equity_last,
        "equity_change": round(change, 4) if change is not None else None,
        "equity_change_pct": round(pct, 3) if pct is not None else None,
        "trades": trade_count,
        "wins": wins,
        "trade_net": round(trade_net, 4),
        "research_cycles": sum(r.research for r in ordered),
        "cost_usd": round(total_cost, 4),
        "refusals": dict(top_refusals.most_common(10)),
        "refused_days": refused_days,
        "silent_days": silent_days,
        "outage_days": outages,
        "tier_changes": [{"when": iso(w), "to": t} for w, t in tiers],
        "verdict": verdict,
    }


def format_journal(j: dict) -> str:
    lines = [f"  PAPER JOURNAL — last {j['window_days']} days "
             f"({j.get('first_day')} → {j.get('last_day')})",
             "  " + "-" * 74]
    lines.append(f"  equity           {j['equity_first']} → {j['equity_last']} "
                 f"({j['equity_change']:+.2f})" if j["equity_change"] is not None
                 else "  equity           no marks recorded")
    lines.append(f"  trades           {j['trades']}  (wins {j['wins']}, "
                 f"net {j['trade_net']:+.2f})")
    lines.append(f"  research cycles  {j['research_cycles']}  ·  "
                 f"cost ${j['cost_usd']:.4f}")
    lines.append("")
    if j["days"]:
        lines.append(f"  {'day':12s} {'ticks':>6s} {'resrch':>7s} "
                     f"{'trades':>7s} {'vetoes':>7s}  what happened")
        for d in j["days"]:
            lines.append(f"  {d['day']:12s} {d['ticks']:6d} {d['research']:7d} "
                         f"{d['trades']:7d} {d['vetoes']:7d}  {d['note']}")
        lines.append("")
    if j["refusals"]:
        lines.append("  refusal reasons (top):")
        for reason, n in list(j["refusals"].items())[:6]:
            lines.append(f"    {n:3d}×  {reason}")
        lines.append("")
    if j["outage_days"]:
        lines.append(f"  OUTAGE DAYS (nothing ran): {', '.join(j['outage_days'])}")
    if j["silent_days"]:
        lines.append(f"  quiet but not refused: {', '.join(j['silent_days'])}")
    lines.append(f"  verdict          {j['verdict']}")
    return "\n".join(lines)
