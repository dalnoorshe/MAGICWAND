"""MAGICWAND — risk tiers and the gatekeeper.

Ported from CHIMERA's safety invariant, which is the right idea and was missing
here:

    risk ceilings come from the TEMPLATE, not the AI.
    The brain can name an agent anything; it can never mint permissions.

For money, the template is the tier. A tier is chosen by a human, stamped into
every proposal, and enforced mechanically on every order. An agent cannot
propose its way into a higher tier any more than a CHIMERA agent could mint
itself a DESTRUCTIVE tool.

Tiers, in order:

    0  RESEARCH      read-only. Analyse, propose, validate. No orders at all.
    1  PAPER_DESK    trades only inside the sandbox pot, hard-capped.
    2  SCALED_DESK   larger sandbox ceiling, still sandbox money.
    3  LIVE          NOT IMPLEMENTED. There is no live broker in this codebase,
                     so this tier exists to be refused.

Every order goes through `Gatekeeper.check()`. It can only ever return a
verdict; it has no ability to place an order itself, and there is no path
around it because `AgentDesk.open_trade` calls it before the broker.

The autonomous system may evolve its *behaviour* — sizing, trade frequency,
which families it trusts — but only inside ranges a human set for its tier.
It cannot evolve its own permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional


class RiskTier(IntEnum):
    RESEARCH = 0
    PAPER_DESK = 1
    SCALED_DESK = 2
    LIVE = 3


class GateError(RuntimeError):
    pass


TIER_NAMES = {
    RiskTier.RESEARCH: "research",
    RiskTier.PAPER_DESK: "paper_desk",
    RiskTier.SCALED_DESK: "scaled_desk",
    RiskTier.LIVE: "live (unimplemented)",
}

# Per-tier hard ceilings. A human writes these; nothing at runtime writes them.
# The numbers are deliberately stingy — the point is that the ceiling is real,
# not that it is generous.
TIER_CEILINGS: dict[RiskTier, dict] = {
    RiskTier.RESEARCH: {
        "max_pot_fraction": 0.0,      # no pot at all
        "max_trade_pct_of_pot": 0.0,
        "max_trades_per_day": 0,
        "min_confidence": 1.01,       # unreachable
        "min_trades_before_autonomy": 0,
    },
    RiskTier.PAPER_DESK: {
        "max_pot_fraction": 0.10,
        "max_trade_pct_of_pot": 0.10,
        "max_trades_per_day": 1,
        "min_confidence": 0.80,
        "min_trades_before_autonomy": 0,
    },
    RiskTier.SCALED_DESK: {
        "max_pot_fraction": 0.25,
        "max_trade_pct_of_pot": 0.20,
        "max_trades_per_day": 3,
        "min_confidence": 0.80,
        "min_trades_before_autonomy": 20,
    },
    RiskTier.LIVE: {
        "max_pot_fraction": 0.0,
        "max_trade_pct_of_pot": 0.0,
        "max_trades_per_day": 0,
        "min_confidence": 1.01,
        "min_trades_before_autonomy": 10 ** 9,
    },
}

# Behaviour the autonomous system MAY tune, within human-set bounds.
# (min, max) is a hard clamp no evolution can exceed.
TUNABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "risk_per_trade": (0.005, 0.02),          # paper account, per trade
    "stop_atr_mult": (1.0, 4.0),
    "target_atr_mult": (1.0, 8.0),
    "trade_frequency_cap": (1.0, 4.0),        # trades/day the system may use
}


@dataclass
class Policy:
    """The operator's current, evolved behaviour. All within tier ceilings."""
    tier: RiskTier = RiskTier.PAPER_DESK
    risk_per_trade: float = 0.01
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 4.0
    trade_frequency_cap: float = 1.0
    reasoning: str = "default policy: conservative sizing, one trade per day"

    # ---- the ceiling applied to this policy ----

    def ceiling(self) -> dict:
        return dict(TIER_CEILINGS[self.tier])

    def effective(self) -> dict:
        """Policy values after clamping to the tier ceiling. Computed, never stored."""
        c = self.ceiling()
        return {
            "tier": self.tier,
            "tier_name": TIER_NAMES[self.tier],
            "max_pot_fraction": c["max_pot_fraction"],
            "max_trade_pct_of_pot": c["max_trade_pct_of_pot"],
            "max_trades_per_day": min(int(round(self.trade_frequency_cap)),
                                      int(c["max_trades_per_day"])),
            "min_confidence": c["min_confidence"],
            "risk_per_trade": _clamp(self.risk_per_trade, *TUNABLE_BOUNDS["risk_per_trade"]),
            "stop_atr_mult": _clamp(self.stop_atr_mult, *TUNABLE_BOUNDS["stop_atr_mult"]),
            "target_atr_mult": _clamp(self.target_atr_mult, *TUNABLE_BOUNDS["target_atr_mult"]),
        }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class Verdict:
    allowed: bool
    reason: str
    tier: RiskTier
    checks: dict = None

    def __post_init__(self):
        if self.checks is None:
            self.checks = {}


class Gatekeeper:
    """Every order passes through here. It cannot place orders; only judge them.

    There is no `bypass` flag and no environment variable that skips a check.
    If a call needs to be refused, it is refused.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    # ---- permission checks ----

    def check_tier_request(self, requested: RiskTier,
                           from_human: bool) -> Verdict:
        """Changing tiers requires a human. The AI may never raise its own."""
        if requested > self.policy.tier and not from_human:
            return Verdict(False, (
                f"refused: cannot raise tier from {TIER_NAMES[self.policy.tier]} "
                f"to {TIER_NAMES[requested]} without a human. Agents may propose, "
                f"never mint permissions."), self.policy.tier)
        return Verdict(True, "tier change permitted", requested)

    def check_order(self, *, pot_fraction: float, trade_pct_of_pot: float,
                    trades_today: int, confidence: float,
                    symbol_whitelisted: bool) -> Verdict:
        """The mechanical gate. Pure function of policy and the proposed order."""
        eff = self.policy.effective()
        checks: dict[str, tuple[bool, str]] = {}

        checks["tier"] = (self.policy.tier != RiskTier.LIVE,
                          f"tier {TIER_NAMES[self.policy.tier]}")
        checks["tier_active"] = (self.policy.tier >= RiskTier.PAPER_DESK,
                                 "tier permits trading" if self.policy.tier >= RiskTier.PAPER_DESK
                                 else "tier is read-only")
        checks["pot"] = (pot_fraction <= eff["max_pot_fraction"] + 1e-12,
                         f"pot {pot_fraction:.1%} <= ceiling {eff['max_pot_fraction']:.1%}")
        checks["size"] = (trade_pct_of_pot <= eff["max_trade_pct_of_pot"] + 1e-12,
                          f"size {trade_pct_of_pot:.1%} <= ceiling "
                          f"{eff['max_trade_pct_of_pot']:.1%}")
        checks["count"] = (trades_today < eff["max_trades_per_day"],
                           f"{trades_today}/{eff['max_trades_per_day']} trades today")
        checks["confidence"] = (confidence >= eff["min_confidence"],
                                f"confidence {confidence:.2f} >= {eff['min_confidence']:.2f}")
        checks["symbol"] = (symbol_whitelisted, "symbol is whitelisted")

        failed = [name for name, (ok, _) in checks.items() if not ok]
        reason = ("denied: " + "; ".join(checks[f][1] for f in failed)) if failed \
            else "all gate checks passed"
        return Verdict(not failed, reason, self.policy.tier, checks)

    # ---- the only way a tier changes ----

    def human_set_tier(self, tier: RiskTier) -> Policy:
        """A human action. The AI has no function that calls this."""
        return replace(self.policy, tier=tier,
                       reasoning=f"tier set by human to {TIER_NAMES[tier]}")


def default_policy() -> Policy:
    return Policy()
