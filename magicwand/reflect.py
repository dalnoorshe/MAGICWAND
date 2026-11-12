"""MAGICWAND — reflection.

The difference between a scripted bot and an autonomous system is whether the
system's behaviour changes because of what it learned. This module provides
that, and it does it *mechanically*:

  * it reads the ledger (its own history) and computes what has actually worked
  * it adjusts behaviour inside human-set bounds and nowhere else
  * it explains every adjustment in plain English, and records it

There is no language model in this loop. A model may propose *strategies*
(ideas), but what the system does about its own track record is arithmetic.
That is deliberate: if reflection were also an LLM opinion, the system would
be grading itself with the same instrument that produced the belief.

What it may change (and the hard bounds, from gatekeeper.TUNABLE_BOUNDS):
    risk_per_trade, stop_atr_mult, target_atr_mult, trade_frequency_cap
What it may decide about the world:
    which strategy families to stop proposing (demoted)
    which agents' opinions to weight less (already done in agents.py)
What it may NEVER change:
    its tier, the fees it pays, the gate, the ledger.

Everything is derived from closed trades and graded claims. A system that
"learns" from unrealized P&L is learning from noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from statistics import mean
from typing import Optional

from .common import safe_mean, utcnow, iso
from .gatekeeper import TUNABLE_BOUNDS, Policy, TIER_NAMES
from .ledger import Ledger


MIN_SAMPLE = 6            # fewer trades than this is noise, not a signal
DEMOTE_EXPECTANCY = 0.0   # at or below this per trade, stop proposing a family
PROMOTE_EXPECTANCY = 0.004  # above this per trade, favour it


@dataclass
class Finding:
    topic: str            # family | agent | policy | overall
    subject: str
    observation: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Reflection:
    ts: str
    findings: list[Finding]
    blacklisted_families: list[str]
    preferred_families: list[str]
    agent_weights: dict
    policy_before: dict
    policy_after: dict
    changes: list[str]
    sample: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        if not self.changes:
            return f"reflection: {len(self.findings)} findings, no changes warranted"
        return "reflection: " + "; ".join(self.changes)


# --------------------------------------------------------------------------
# Reading history
# --------------------------------------------------------------------------

def _closed_trades(ledger: Ledger) -> list[dict]:
    """Desk round-trips, with the strategy family that opened them."""
    out = []
    for e in ledger.entries():
        if e.get("kind") != "desk_close":
            continue
        p = e.get("payload", {}) or {}
        out.append({
            "agent": p.get("opened_by", ""),
            "net": float(p.get("net", 0.0)),
            "return_pct": float(p.get("return_pct", 0.0)),
            "reason": p.get("reason", ""),
        })
    return out


def _agent_grades(ledger: Ledger) -> dict[str, dict]:
    claims: dict[str, dict] = {}
    for e in ledger.entries():
        if e.get("kind") == "agent_claim":
            p = e.get("payload", {}) or {}
            claims[p.get("claim_id")] = p
    out: dict[str, dict] = {}
    for e in ledger.entries():
        if e.get("kind") != "grade":
            continue
        p = e.get("payload", {}) or {}
        src = claims.get(p.get("claim_id"))
        if not src:
            continue
        a = src.get("agent", "?")
        s = out.setdefault(a, {"n": 0, "right": 0, "wrong": 0})
        s["n"] += 1
        if p.get("verdict") == "right":
            s["right"] += 1
        elif p.get("verdict") == "wrong":
            s["wrong"] += 1
    for s in out.values():
        s["hit_rate"] = round(s["right"] / s["n"], 4) if s["n"] else 0.0
    return out


def _family_of(agent_label: str) -> str:
    """Agent labels look like 'strategy:sma_trend' or 'derivatives'."""
    if agent_label.startswith("strategy:"):
        return agent_label.split(":", 1)[1]
    return agent_label


# --------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------

def reflect(ledger: Ledger, policy: Policy) -> Reflection:
    trades = _closed_trades(ledger)
    grades = _agent_grades(ledger)
    findings: list[Finding] = []
    blacklist: list[str] = []
    prefer: list[str] = []
    changes: list[str] = []

    # ---- per-family performance ----
    by_family: dict[str, list[float]] = {}
    for t in trades:
        fam = _family_of(t["agent"])
        by_family.setdefault(fam, []).append(t["return_pct"])
    for fam, rs in by_family.items():
        n = len(rs)
        exp = safe_mean(rs)
        wins = sum(1 for r in rs if r > 0)
        findings.append(Finding(
            "family", fam,
            f"{n} trades, expectancy {exp:+.3%}, hit rate {wins / n:.0%}",
            {"n": n, "expectancy": round(exp, 6),
             "hit_rate": round(wins / n, 4)}))
        if n >= MIN_SAMPLE and exp <= DEMOTE_EXPECTANCY:
            blacklist.append(fam)
            findings.append(Finding(
                "family", fam,
                f"DEMOTED: {n} trades at {exp:+.3%} expectancy — this family "
                f"loses after costs; stop proposing it", {"n": n}))
        elif n >= MIN_SAMPLE and exp >= PROMOTE_EXPECTANCY:
            prefer.append(fam)

    # ---- per-agent calibration ----
    weights: dict[str, float] = {}
    for agent, s in grades.items():
        if s["n"] < MIN_SAMPLE:
            continue
        hit = s["hit_rate"]
        weights[agent] = round(max(0.05, min(1.0, 0.5 + (hit - 0.5) * 1.2)), 3)
        findings.append(Finding(
            "agent", agent,
            f"{s['n']} graded claims, hit rate {hit:.0%} -> influence {weights[agent]:.2f}",
            {"n": s["n"], "hit_rate": hit}))

    # ---- policy adaptation (bounded arithmetic, not opinion) ----
    before = policy.effective()
    new = replace(policy)

    # Size off the families the system will KEEP trading, not every trade in
    # history. Counting trades from a family we just demoted would let a dead
    # strategy's results drive future risk.
    kept = [t["return_pct"] for t in trades
            if _family_of(t["agent"]) not in blacklist]
    exp = safe_mean(kept) if kept else 0.0
    if len(kept) >= MIN_SAMPLE:
        if exp < 0:
            # Losing: shrink exposure. Never grow it to "recover".
            new = replace(new, risk_per_trade=new.risk_per_trade * 0.75)
            changes.append(f"shrank risk_per_trade to {new.risk_per_trade:.4f} "
                           f"(kept-family expectancy {exp:+.3%})")
        elif exp > PROMOTE_EXPECTANCY:
            new = replace(new, risk_per_trade=new.risk_per_trade * 1.10)
            changes.append(f"raised risk_per_trade to {new.risk_per_trade:.4f} "
                           f"(kept-family expectancy {exp:+.3%})")
    elif blacklist and not kept:
        # Everything that traded was demoted. Do not keep sizing for ghosts.
        new = replace(new, risk_per_trade=new.risk_per_trade * 0.75)
        changes.append(f"shrank risk_per_trade to {new.risk_per_trade:.4f} "
                       f"(every traded family was demoted)")

        # Stops that are being hit far more than targets suggest stops are too
        # tight for this volatility regime; widen within the bound.
        stops = sum(1 for t in trades if t["reason"] == "stop")
        targets = sum(1 for t in trades if t["reason"] == "target")
        if stops > targets * 2 and stops >= 4:
            new = replace(new, stop_atr_mult=new.stop_atr_mult * 1.15)
            changes.append(f"widened stop_atr_mult to {new.stop_atr_mult:.2f} "
                           f"({stops} stops vs {targets} targets)")

    # Clamp into the human bounds. This is what "never exceeds its ceiling" means.
    new = replace(
        new,
        risk_per_trade=_clamp(new.risk_per_trade, *TUNABLE_BOUNDS["risk_per_trade"]),
        stop_atr_mult=_clamp(new.stop_atr_mult, *TUNABLE_BOUNDS["stop_atr_mult"]),
        target_atr_mult=_clamp(new.target_atr_mult, *TUNABLE_BOUNDS["target_atr_mult"]),
        trade_frequency_cap=_clamp(new.trade_frequency_cap,
                                   *TUNABLE_BOUNDS["trade_frequency_cap"]),
        reasoning=_reasoning(changes, exp if len(kept) >= MIN_SAMPLE else None),
    )
    after = dict(new.effective())
    after["reasoning"] = new.reasoning

    if not trades and not grades:
        findings.append(Finding("overall", "system",
                                "no closed trades and no graded claims yet — "
                                "nothing to learn from, so nothing changes",
                                {"trades": 0, "graded": 0}))

    return Reflection(
        ts=iso(utcnow()),
        findings=findings,
        blacklisted_families=blacklist,
        preferred_families=prefer,
        agent_weights=weights,
        policy_before=before,
        policy_after=after,
        changes=changes,
        sample={"trades": len(trades), "graded_agents": len(grades),
                "families": {k: len(v) for k, v in by_family.items()}},
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _reasoning(changes: list[str], expectancy: Optional[float]) -> str:
    if not changes:
        return "no change: evidence did not warrant one"
    return (f"evolved after {len(changes)} adjustment(s)"
            + (f"; overall expectancy {expectancy:+.3%}" if expectancy is not None else ""))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

class PolicyStore:
    """The evolved policy lives on disk so it survives restarts."""

    def __init__(self, path) -> None:
        from pathlib import Path
        self.path = Path(path)

    def load(self, default: Optional[Policy] = None) -> Policy:
        if not self.path.exists():
            return default or Policy()
        try:
            d = json.loads(self.path.read_text())
            from .gatekeeper import RiskTier
            tier = RiskTier(int(d.get("tier", int(RiskTier.PAPER_DESK))))
            return Policy(
                tier=tier,
                risk_per_trade=float(d.get("risk_per_trade", 0.01)),
                stop_atr_mult=float(d.get("stop_atr_mult", 2.0)),
                target_atr_mult=float(d.get("target_atr_mult", 4.0)),
                trade_frequency_cap=float(d.get("trade_frequency_cap", 1.0)),
                reasoning=str(d.get("reasoning", "")),
            )
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            return default or Policy()

    def save(self, policy: Policy) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "tier": int(policy.tier),
            "tier_name": TIER_NAMES[policy.tier],
            "risk_per_trade": policy.risk_per_trade,
            "stop_atr_mult": policy.stop_atr_mult,
            "target_atr_mult": policy.target_atr_mult,
            "trade_frequency_cap": policy.trade_frequency_cap,
            "reasoning": policy.reasoning,
            "saved_at": iso(utcnow()),
        }, indent=2))


def reflect_and_persist(ledger: Ledger, store: PolicyStore) -> Reflection:
    """Reflect, save the evolved policy, and record the reflection."""
    policy = store.load()
    ref = reflect(ledger, policy)
    evolved = Policy(
        tier=policy.tier,   # tier is NEVER changed by reflection
        risk_per_trade=_clamp(ref.policy_after.get("risk_per_trade", policy.risk_per_trade),
                              *TUNABLE_BOUNDS["risk_per_trade"]),
        stop_atr_mult=_clamp(ref.policy_after.get("stop_atr_mult", policy.stop_atr_mult),
                             *TUNABLE_BOUNDS["stop_atr_mult"]),
        target_atr_mult=_clamp(ref.policy_after.get("target_atr_mult",
                                                    policy.target_atr_mult),
                               *TUNABLE_BOUNDS["target_atr_mult"]),
        trade_frequency_cap=_clamp(ref.policy_after.get("trade_frequency_cap",
                                                        policy.trade_frequency_cap),
                                   *TUNABLE_BOUNDS["trade_frequency_cap"]),
        reasoning=ref.policy_after.get("reasoning", "") or "no change: evidence did not warrant one",
    )
    store.save(evolved)
    ledger.append("reflection", {
        "summary": ref.summary,
        "changes": ref.changes,
        "blacklisted_families": ref.blacklisted_families,
        "preferred_families": ref.preferred_families,
        "agent_weights": ref.agent_weights,
        "policy_before": ref.policy_before,
        "policy_after": ref.policy_after,
        "sample": ref.sample,
        "findings": [asdict(f) for f in ref.findings],
    })
    return ref
