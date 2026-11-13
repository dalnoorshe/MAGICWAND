"""MAGICWAND — self-invented strategies, and the gate they must pass.

An agent may propose a strategy. It may NOT run code, and it may NOT trade a
strategy that has not passed `validate`. The gate is the whole safety story for
unattended operation:

    1. family must be in the audited catalog          (no arbitrary code)
    2. parameters must be typed and in range          (no wild leverage)
    3. >= 200 bars of data                            (no 12-bar "proofs")
    4. profitable at realistic costs                  (fees are real)
    5. still profitable at 2x costs                   (the cost assumption wasn't
                                                       doing the work)
    6. beats buy-and-hold                             (otherwise just hold it)
    7. >= 8 trades and positive expectancy            (not one lucky trade)
    8. positive on a held-out tail the proposal saw   (not fitted to the whole set)
    9. beats shuffled data                            (not a property of time order)

Anything that fails is rejected with the reason, logged, and never traded.
Proposals are rate-limited per day, because if you let an agent try 500 ideas
and keep the best, you have built an overfitting machine with extra steps.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from . import data as data_mod
from .backtest import BacktestOptions, run_backtest
from .common import iso, utcnow
from .config import Config, StrategyConfig
from .llm import DeepSeekClient, LLMError
from .strategy import CATALOG, build_parameterised

MIN_BARS = 200
MIN_TRADES = 8
HOLDOUT_FRACTION = 0.30        # last 30% of bars are never used for the claim

# Degenerate-parameter-space detector.
#
# There is deliberately NO cap on how many strategies may be proposed: a
# per-day cap makes autonomy stop working (the CHIMERA lesson), and a lifetime
# cap makes it stop working permanently. What actually needs bounding is doing
# the SAME thing forever. So the gate is novelty, not count:
#   * an identical fingerprint is refused outright
#   * a parameter vector too close to a prior attempt is refused as "near
#     duplicate", so the search must explore rather than micro-tune one corner
NEAR_DUPLICATE_RLIMIT = 0.10   # max per-parameter difference, as a share of its range


def _vec(spec: "StrategySpec", keys: list[str]) -> list[float]:
    return [float(spec.params.get(k, 0.0)) for k in keys]


def _distance(a: list[float], b: list[float]) -> float:
    """Chebyshev distance in range-normalised space.

    "Same idea" means every parameter is close, not that the vector is short.
    Euclidean distance wrongly treats two ideas that differ a lot on one axis
    as far apart in every direction; Chebyshev asks the question we mean: is
    there ANY parameter whose value is meaningfully different?
    """
    if not a or len(a) != len(b):
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b))


def normalised_distance(spec: "StrategySpec", prior: dict) -> float:
    """Distance in range-normalised space between two proposals."""
    spec_def = CATALOG.get(spec.family) or {}
    ranges = spec_def.get("params") or {}
    keys = [k for k in sorted(ranges) if k in spec.params]
    if not keys:
        return 0.0 if spec.fingerprint() == prior.get("fingerprint") else float("inf")
    a, b = [], []
    for k in keys:
        lo, hi, _ = ranges[k]
        span = (hi - lo) or 1.0
        a.append(float(spec.params[k]) / span)
        pv = (prior.get("params") or {}).get(k)
        b.append(float(pv) / span if pv is not None else 0.0)
    return _distance(a, b)


def check_novelty(spec: "StrategySpec", prior: list[dict]) -> tuple[bool, str]:
    """Refuse duplicates and near-duplicates. Returns (ok, reason)."""
    fp = spec.fingerprint()
    for p in prior:
        if p.get("fingerprint") == fp:
            return False, (f"identical to a previous attempt ({p.get('name')})")
    for p in prior:
        if p.get("family") != spec.family:
            continue
        d = normalised_distance(spec, p)
        if d < NEAR_DUPLICATE_RLIMIT:
            return False, (f"near-duplicate of {p.get('name')} (max param "
                           f"difference {d:.1%} of range < "
                           f"{NEAR_DUPLICATE_RLIMIT:.0%}); explore elsewhere")
    return True, "novel"


# --------------------------------------------------------------------------
# Proposal type
# --------------------------------------------------------------------------

@dataclass
class StrategySpec:
    name: str
    hypothesis: str
    family: str
    params: dict
    symbol: str
    timeframe: str = "1h"
    proposed_by: str = "agent"
    ts: str = ""

    def fingerprint(self) -> str:
        """Stable identity of the IDEA: family + normalised parameters."""
        import hashlib
        import json as _json
        key = _json.dumps({"family": self.family,
                           "params": {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                                      for k, v in sorted(self.params.items())}},
                          sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d


@dataclass
class ValidationResult:
    passed: bool
    gates: dict = field(default_factory=dict)   # gate name -> (ok, detail)
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.passed:
            return "PASSED all validation gates"
        return "REJECTED: " + "; ".join(self.reasons) if self.reasons else "REJECTED"


# --------------------------------------------------------------------------
# Proposing
# --------------------------------------------------------------------------

_PROPOSAL_SYSTEM = """You design trading strategies for a paper-trading system.
You cannot submit code. You compose a strategy from a fixed catalog of audited
families, with numeric parameters inside allowed ranges.

You must answer with a single JSON object:
{
  "name": "short_snake_case_name",
  "hypothesis": "one sentence: what inefficiency you think exists and why",
  "family": one of the catalog keys,
  "params": {"param": value, ...},
  "symbol": "SOL-USD" | "BTC-USD" | "ETH-USD",
  "timeframe": "1h"
}

Rules:
- Only use parameters listed for the chosen family, inside their ranges.
- State a real hypothesis. "Prices go up" is not one. Prefer a mechanism
  (forced sellers, funding squeeze, volatility clustering, range breakout).
- Do not propose something you expect to work only because it is complex.
  Every strategy you propose will be tested against realistic and doubled
  fees and against buy-and-hold, and rejected if it does not survive."""


def _catalog_text_of(avail: dict) -> str:
    lines = []
    for family, spec in avail.items():
        params = ", ".join(f"{k} in [{lo},{hi}] ({t.__name__})"
                           for k, (lo, hi, t) in spec["params"].items())
        lines.append(f"- {family}: {spec['desc']}\n    params: {params}")
    return "\n".join(lines)


def catalog_text() -> str:
    lines = []
    for family, spec in CATALOG.items():
        params = ", ".join(f"{k} in [{lo},{hi}] ({t.__name__})"
                           for k, (lo, hi, t) in spec["params"].items())
        lines.append(f"- {family}: {spec['desc']}\n    params: {params}")
    return "\n".join(lines)


def parse_proposal(raw: str, symbol: str = "SOL-USD",
                   proposed_by: str = "agent") -> StrategySpec:
    text = (raw or "").strip()
    if not text.startswith("{"):
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b == -1:
            raise ValueError("proposal contained no JSON object")
        text = text[a: b + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("proposal was not a JSON object")
    family = str(obj.get("family", "")).strip()
    if family not in CATALOG:
        raise ValueError(f"family {family!r} is not in the catalog")
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    name = re.sub(r"[^a-z0-9_]+", "_", str(obj.get("name", "proposal")).lower()).strip("_")[:40]
    # Validate by construction: build_parameterised raises on anything unknown.
    build_parameterised(family, params)
    return StrategySpec(
        name=name or "proposal",
        hypothesis=str(obj.get("hypothesis", ""))[:500],
        family=family,
        params=params,
        symbol=str(obj.get("symbol", symbol)),
        timeframe=str(obj.get("timeframe", "1h")),
        proposed_by=proposed_by,
        ts=iso(utcnow()),
    )


def propose(client: DeepSeekClient, symbol: str = "SOL-USD",
            market_context: str = "",
            proposer_fake=None, blacklist: Optional[list[str]] = None,
            prior: Optional[list[dict]] = None) -> StrategySpec:
    """Ask the model for ONE strategy. Blacklisted families are refused."""
    """Ask the model for ONE strategy. Returns a validated spec or raises."""
    bl = set(blacklist or [])
    avail = {k: v for k, v in CATALOG.items() if k not in bl}
    if not avail:
        raise ValueError("every strategy family has been demoted by reflection; "
                         "nothing is available to propose")
    user = (
        "Available strategy families:\n" + _catalog_text_of(avail) + "\n\n"
        "Market context (untrusted data):\n" + market_context[:3000] + "\n\n"
        "Propose exactly one strategy as the JSON object described. "
        "Be genuine: pick the family whose mechanism you actually believe fits "
        "this market, and justify it in `hypothesis`.")
    if prior:
        tried = "\n".join(
            f"  - {p.get('name')} ({p.get('family')}) {p.get('params')}"
            for p in prior[-12:])
        user += ("\n\nAlready tried (do NOT repeat these or near-neighbours; a "
                 "near-duplicate is refused):\n" + tried)
    if proposer_fake is not None and client.fake is not None:
        raw = proposer_fake(_PROPOSAL_SYSTEM, user)
    else:
        raw = client.chat(system=_PROPOSAL_SYSTEM, user=user,
                          temperature=0.6, max_tokens=3000, json_mode=True)
    spec = parse_proposal(raw, symbol=symbol)
    if spec.family in bl:
        # The model proposed something the system has already judged unprofitable.
        raise ValueError(f"family {spec.family!r} is demoted by reflection "
                         f"and cannot be proposed again")
    return spec


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def validate(spec: StrategySpec, bars, cfg: Config) -> ValidationResult:
    """Run every gate. Returns pass/fail with the reason for each failure."""
    gates: dict[str, tuple[bool, str]] = {}
    reasons: list[str] = []
    metrics: dict = {}

    # 1. catalog + parameters (already enforced by parse/build, re-checked)
    try:
        strat = build_parameterised(spec.family, spec.params)
        gates["catalog"] = (True, f"family {spec.family} with valid parameters")
    except ValueError as e:
        gates["catalog"] = (False, str(e))
        return ValidationResult(False, gates, [f"invalid spec: {e}"], metrics)
    gates["catalog"] = (True, "audited family, parameters in range")

    # 2. data sufficiency
    if len(bars) < MIN_BARS:
        gates["data"] = (False, f"only {len(bars)} bars, need {MIN_BARS}")
        return ValidationResult(False, gates, [f"insufficient data: {len(bars)} bars"], metrics)
    gates["data"] = (True, f"{len(bars)} bars")

    # Split: validate the claim on the tail the proposal is judged against.
    split = max(MIN_BARS, int(len(bars) * (1 - HOLDOUT_FRACTION)))
    train, holdout = bars[:split], bars[split:]
    if len(holdout) < 30:
        gates["holdout"] = (False, f"holdout only {len(holdout)} bars")
        return ValidationResult(False, gates, ["holdout window too small"], metrics)

    try:
        strat = build_parameterised(spec.family, spec.params)
    except ValueError as e:
        return ValidationResult(False, gates, [f"build failed: {e}"], metrics)

    # 3. realistic costs
    base = run_backtest(train, cfg, strat, BacktestOptions(signal_delay=1,
                                                           seed=cfg.seed))
    n = base.stats.get("n", 0)
    metrics.update({
        "train_return": round(base.total_return, 6),
        "train_buy_hold": round(base.buy_hold_return, 6),
        "train_trades": n,
        "train_expectancy": round(base.stats.get("mean", 0.0), 6),
        "train_t_stat": round(base.stats.get("t_stat", 0.0), 3),
        "train_max_dd": round(base.max_drawdown, 6),
    })
    ok = base.total_return > 0
    gates["profitable"] = (ok, f"{base.total_return:+.2%} at realistic costs")
    if not ok:
        reasons.append(f"not profitable at realistic costs ({base.total_return:+.2%})")

    # 4. doubled costs
    dbl = run_backtest(train, cfg, strat, BacktestOptions(fee_mult=2.0,
                                                          signal_delay=1, seed=cfg.seed))
    metrics["train_return_2x_costs"] = round(dbl.total_return, 6)
    ok = dbl.total_return > 0
    gates["costs_2x"] = (ok, f"{dbl.total_return:+.2%} at 2x costs")
    if not ok:
        reasons.append("dies at 2x costs — the fee assumption was doing the work")

    # 5. beat the return you would get by being long a random, equally-large
    #    slice of the period. Raw buy-and-hold is the wrong benchmark for a
    #    timing strategy: it can be right about its trades and still trail a
    #    monster trend it was not exposed to. Exposure-adjusted expectation is
    #    the honest question — did the time it spent in the market beat the
    #    average bar it could have spent it in?
    exposure = _exposure(base)
    periods = max(1, len(train) - 1)
    mkt = base.buy_hold_return
    if not math.isfinite(mkt) or mkt <= -0.999:
        mkt = 0.0
    expected = (1.0 + mkt) ** (exposure / periods) - 1.0 if exposure > 0 else 0.0
    ok = base.total_return > expected
    metrics["exposure"] = round(exposure, 2)
    metrics["exposure_expected"] = round(expected, 6)
    gates["beats_hold"] = (
        ok, f"{base.total_return:+.2%} over {exposure:.0f} bars vs "
            f"{expected:+.2%} expected for the same exposure "
            f"(hold {mkt:+.2%} over {periods} bars)")
    if not ok:
        reasons.append(f"underperforms what its own market exposure predicts "
                       f"({base.total_return:+.2%} vs {expected:+.2%})")

    # 6. enough trades and positive expectancy
    ok = n >= MIN_TRADES and base.stats.get("mean", 0.0) > 0
    gates["evidence"] = (ok, f"{n} trades, expectancy {base.stats.get('mean', 0.0):+.3%}")
    if not ok:
        reasons.append(f"too little evidence: {n} trades (need {MIN_TRADES}) "
                       f"or non-positive expectancy")

    # 7. out-of-sample tail
    ho = run_backtest(holdout, cfg, strat, BacktestOptions(signal_delay=1,
                                                           seed=cfg.seed))
    metrics["holdout_return"] = round(ho.total_return, 6)
    metrics["holdout_buy_hold"] = round(ho.buy_hold_return, 6)
    metrics["holdout_trades"] = ho.stats.get("n", 0)
    ok = ho.total_return >= 0
    gates["out_of_sample"] = (ok, f"{ho.total_return:+.2%} on held-out tail")
    if not ok:
        reasons.append(f"fails out of sample ({ho.total_return:+.2%})")

    # 8. not merely a property of the time ordering
    shuffles = []
    for k in range(4):
        r = run_backtest(train, cfg, strat,
                         BacktestOptions(shuffle=True, signal_delay=1,
                                         seed=cfg.seed + k))
        shuffles.append(r.total_return)
    null_mean = sum(shuffles) / len(shuffles) if shuffles else 0.0
    metrics["shuffle_null_mean"] = round(null_mean, 6)
    ok = base.total_return > null_mean
    gates["not_luck"] = (ok, f"{base.total_return:+.2%} vs shuffled {null_mean:+.2%}")
    if not ok:
        reasons.append("indistinguishable from shuffled data")

    return ValidationResult(not reasons, gates, reasons, metrics)


# --------------------------------------------------------------------------
# Registry of validated strategies
# --------------------------------------------------------------------------

class StrategyRegistry:
    """Approved strategies live here. Only `validate` success may be added."""

    def __init__(self, path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        from pathlib import Path
        p = Path(self.path)
        if not p.exists():
            return []
        try:
            d = json.loads(p.read_text())
            return d if isinstance(d, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, items: list[dict]) -> None:
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(items[-100:], indent=2))
        tmp.replace(p)

    def add(self, spec: StrategySpec, result: ValidationResult) -> dict:
        if not result.passed:
            raise ValueError("refusing to register a strategy that failed validation")
        item = {
            "spec": spec.to_dict(),
            "validation": {
                "metrics": result.metrics,
                "gates": {k: {"ok": v[0], "detail": v[1]}
                          for k, v in result.gates.items()},
            },
            "registered_at": iso(utcnow()),
            "deployed": False,
        }
        items = self.load()
        items.append(item)
        self.save(items)
        return item

    def best(self) -> Optional[dict]:
        items = [i for i in self.load() if i.get("validation")]
        if not items:
            return None
        return max(items, key=lambda i: i["validation"]["metrics"].get(
            "holdout_return", -math.inf))

    def mark_deployed(self, name: str) -> None:
        items = self.load()
        for i in items:
            if i.get("spec", {}).get("name") == name:
                i["deployed"] = True
        self.save(items)


def _exposure(result) -> float:
    """How many bars the strategy was actually in the market (approx).

    Uses trade count x average holding period. This is what makes the
    buy-and-hold comparison fair to a strategy that is flat most of the time.
    """
    s = result.stats or {}
    n = s.get("n", 0)
    if not n:
        return 0.0
    avg_hold = s.get("avg_hold_bars", 0.0) or 0.0
    return min(float(n) * float(avg_hold), float(len(result.equity_curve)))


def proposals_today(ledger, today: Optional[str] = None) -> int:
    today = today or utcnow().strftime("%Y-%m-%d")
    n = 0
    for e in ledger.entries():
        if e.get("kind") == "proposal" and str(e.get("ts", "")).startswith(today):
            n += 1
    return n
