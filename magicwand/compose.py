"""MAGICWAND — strategy composition.

The three catalog strategies (SMA trend, mean reversion, breakout) are templates.
The model does not have to pick one: it can COMPOSE a strategy from audited
building blocks — a small number of ANDed conditions, an ATR stop/target, and a
time stop. That is genuine invention inside a space we can validate, rather than
"pick option 2 and change a number".

Why a grammar and not generated code
------------------------------------
Arbitrary code that trades money is un-auditable, un-checkable, and impossible to
backtest honestly without re-reading it every time. A composition is data:
clauses that a person can read in one line, each from a named feature, each with
a numeric bound. The gate then tests it exactly like any other strategy, so an
invented idea is judged by evidence and not by novelty.

What a composition looks like

    {"clauses": [{"feature": "rsi", "op": "<", "value": 32},
                 {"feature": "dist_slow", "op": ">", "value": -0.02}],
     "stop_atr": 2.0, "target_atr": 4.0, "max_hold_bars": 48,
     "hypothesis": "capitulation below the slow mean while oversold"}

Reading: buy when RSI < 32 AND price is more than 2% below the slow mean; exit on
the ATR stop/target, when either condition stops holding, or after 48 bars.

NOTE ON SECRECY: this file does not attempt to make a strategy unreadable. A
composition that trades is observable in its fills, and a strategy that lives on
a machine you control cannot be hidden from someone who has that machine. The
durable asset is the graded record, not the trick — see docs/PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from . import indicators as ind
from .common import Bar, Side
from .config import StrategyConfig
from .strategy import Intent, Strategy


# --------------------------------------------------------------------------
# The audited vocabulary
# --------------------------------------------------------------------------

# feature -> (min, max, human description, higher means)
FEATURES: dict[str, tuple[float, float, str]] = {
    "rsi": (5.0, 95.0, "RSI(14), 0-100"),
    "zscore": (-4.0, 4.0, "price vs its own 40-bar mean, in std devs"),
    "dist_fast": (-0.25, 0.25, "close / fast MA - 1"),
    "dist_slow": (-0.35, 0.35, "close / slow MA - 1"),
    "ma_slope": (-0.20, 0.20, "fast MA change over 5 bars, relative"),
    "atr_ratio": (0.3, 3.0, "ATR now / ATR 20 bars ago"),
    "range_pos": (0.0, 1.0, "where close sits in the 20-bar range, 0=low"),
    "vol_z": (-4.0, 4.0, "volume vs its 20-bar mean, in std devs"),
    "ret5": (-0.30, 0.30, "5-bar return"),
}

OPS = ("<", "<=", ">", ">=")
MAX_CLAUSES = 3


class CompositionError(ValueError):
    pass


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str
    value: float

    def ok(self, f: dict) -> bool:
        v = f.get(self.feature)
        if v is None:
            return False
        if self.op == "<":
            return v < self.value
        if self.op == "<=":
            return v <= self.value
        if self.op == ">":
            return v > self.value
        if self.op == ">=":
            return v >= self.value
        return False

    def label(self) -> str:
        return f"{self.feature} {self.op} {self.value:g}"


@dataclass
class Composition:
    clauses: list[Clause]
    stop_atr: float = 2.0
    target_atr: float = 4.0
    max_hold_bars: int = 48
    hypothesis: str = ""
    fast: int = 10
    slow: int = 40

    # ---- identity ----

    def fingerprint(self) -> str:
        """Stable identity of the IDEA, for novelty checks and the ledger."""
        key = json.dumps({
            "clauses": sorted([c.feature, c.op, round(c.value, 6)]
                              for c in self.clauses),
            "stop_atr": round(self.stop_atr, 3),
            "target_atr": round(self.target_atr, 3),
            "max_hold": self.max_hold_bars,
            "fast": self.fast, "slow": self.slow,
        }, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def describe(self) -> str:
        return " AND ".join(c.label() for c in self.clauses)

    def to_dict(self) -> dict:
        return {
            "clauses": [{"feature": c.feature, "op": c.op, "value": c.value}
                        for c in self.clauses],
            "stop_atr": self.stop_atr, "target_atr": self.target_atr,
            "max_hold_bars": self.max_hold_bars,
            "hypothesis": self.hypothesis,
            "fast": self.fast, "slow": self.slow,
            "fingerprint": self.fingerprint(),
            "expression": self.describe(),
        }


# --------------------------------------------------------------------------
# Parsing / validation
# --------------------------------------------------------------------------

def _num(value, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise CompositionError(f"{name} must be a number") from e
    if v != v or v in (float("inf"), float("-inf")):
        raise CompositionError(f"{name} must be finite")
    return v


def parse(payload: dict | str) -> Composition:
    """Build a validated Composition from model output or a dict."""
    from .jsonfix import parse_json_lenient
    if isinstance(payload, str):
        parsed = parse_json_lenient(payload)
        if not parsed.ok:
            raise CompositionError(parsed.note or "unparseable composition")
        data = parsed.obj
    else:
        data = payload
    if not isinstance(data, dict):
        raise CompositionError("composition must be an object")

    raw_clauses = data.get("clauses")
    if not isinstance(raw_clauses, list) or not raw_clauses:
        raise CompositionError("composition needs at least one clause")
    if len(raw_clauses) > MAX_CLAUSES:
        raise CompositionError(f"at most {MAX_CLAUSES} clauses")

    clauses: list[Clause] = []
    seen: set[str] = set()
    for c in raw_clauses:
        if not isinstance(c, dict):
            raise CompositionError("each clause must be an object")
        feature = str(c.get("feature", "")).strip()
        if feature not in FEATURES:
            raise CompositionError(
                f"unknown feature {feature!r}; available: {sorted(FEATURES)}")
        if feature in seen:
            raise CompositionError(f"duplicate clause for {feature!r}")
        seen.add(feature)
        op = str(c.get("op", "")).strip()
        if op not in OPS:
            raise CompositionError(f"unknown operator {op!r}; use {OPS}")
        value = _num(c.get("value"), f"value for {feature}")
        lo, hi, _ = FEATURES[feature]
        if not (lo <= value <= hi):
            raise CompositionError(
                f"{feature}={value:g} outside the audited range [{lo:g}, {hi:g}]")
        clauses.append(Clause(feature, op, value))

    stop_atr = _num(data.get("stop_atr", 2.0), "stop_atr")
    target_atr = _num(data.get("target_atr", 4.0), "target_atr")
    if not (0.5 <= stop_atr <= 6.0):
        raise CompositionError("stop_atr must be between 0.5 and 6")
    if not (0.0 <= target_atr <= 12.0):
        raise CompositionError("target_atr must be between 0 and 12")
    try:
        max_hold = int(data.get("max_hold_bars", 48))
    except (TypeError, ValueError) as e:
        raise CompositionError("max_hold_bars must be an integer") from e
    if not (1 <= max_hold <= 500):
        raise CompositionError("max_hold_bars must be between 1 and 500")

    return Composition(
        clauses=clauses, stop_atr=stop_atr, target_atr=target_atr,
        max_hold_bars=max_hold,
        hypothesis=str(data.get("hypothesis", ""))[:400],
        fast=int(data.get("fast", 10)), slow=int(data.get("slow", 40)))


# --------------------------------------------------------------------------
# The strategy
# --------------------------------------------------------------------------

class ComposedStrategy(Strategy):
    """Buy when every clause holds; exit when any stops holding."""

    name = "composed"

    def __init__(self, comp: Composition, cfg: Optional[StrategyConfig] = None) -> None:
        base = cfg or StrategyConfig()
        base.fast = comp.fast
        base.slow = comp.slow
        base.stop_atr = comp.stop_atr
        base.target_atr = comp.target_atr
        base.max_hold_bars = comp.max_hold_bars
        super().__init__(base)
        self.comp = comp

    def warmup(self) -> int:
        return max(self.cfg.slow, 60) + 1

    def _row(self, closes: list[float], bars: list[Bar], i: int) -> dict:
        """Feature values at bar i. Causal: uses bars 0..i only."""
        f: dict = {}
        r = ind.rsi(bars[: i + 1], 14)
        f["rsi"] = r[i] if i < len(r) and r[i] is not None else None
        z = ind.zscore(closes, 40)
        f["zscore"] = z[i] if i < len(z) else None
        fast = ind.sma(closes, max(2, self.cfg.fast))[i]
        slow = ind.sma(closes, max(2, self.cfg.slow))[i]
        px = closes[i]
        f["dist_fast"] = (px / fast - 1.0) if fast else None
        f["dist_slow"] = (px / slow - 1.0) if slow else None
        if i >= 5:
            fast_prev = ind.sma(closes, max(2, self.cfg.fast))[i - 5]
            f["ma_slope"] = ((fast - fast_prev) / abs(fast_prev)) if (fast and fast_prev) else None
        else:
            f["ma_slope"] = None
        a = self._series("atr", bars)
        atr_now = a[i] if i < len(a) else None
        atr_prev = a[i - 20] if i >= 20 and i - 20 < len(a) else None
        f["atr_ratio"] = (atr_now / atr_prev) if (atr_now and atr_prev) else None
        window = bars[max(0, i - 19): i + 1]
        hi = max(b.high for b in window)
        lo = min(b.low for b in window)
        f["range_pos"] = ((px - lo) / (hi - lo)) if hi > lo else 0.5
        vols = [b.volume for b in window]
        vmean = sum(vols) / len(vols)
        vsd = (sum((v - vmean) ** 2 for v in vols) / len(vols)) ** 0.5
        f["vol_z"] = ((bars[i].volume - vmean) / vsd) if vsd > 0 else 0.0
        f["ret5"] = (px / closes[i - 5] - 1.0) if i >= 5 and closes[i - 5] else None
        return f

    def decide(self, bars, i, in_position):
        if i < self.warmup():
            return None
        closes = [b.close for b in bars[: i + 1]]
        row = self._row(closes, bars, i)
        satisfied = [c.ok(row) for c in self.comp.clauses]
        a = self._series("atr", bars)[i]
        if not a:
            return None
        px = bars[i].close

        if in_position:
            if not all(satisfied):
                return Intent("exit", Side.SELL,
                              f"composition no longer holds "
                              f"({', '.join(c.label() for c, ok in zip(self.comp.clauses, satisfied) if not ok)})")
            return None

        if all(satisfied):
            stop_d = self.comp.stop_atr * a
            tgt_d = self.comp.target_atr * a
            return Intent(
                "enter", Side.BUY,
                f"composition holds: {self.comp.describe()}",
                stop_distance=stop_d, target_distance=tgt_d,
                expected_move=(tgt_d / px) if tgt_d else 0.0,
                max_hold_bars=self.comp.max_hold_bars)
        return None


def build(comp: Composition, cfg: Optional[StrategyConfig] = None) -> ComposedStrategy:
    return ComposedStrategy(comp, cfg)


# --------------------------------------------------------------------------
# Novelty against previously tried compositions
# --------------------------------------------------------------------------

_PROMPT = """You design trading strategies by COMPOSING audited building blocks.
You cannot write code. You combine at most three conditions with AND, plus stops.

Available features (name: range — meaning):
{features}

Operators: <  <=  >  >=

Output one JSON object and nothing else:
{{
  "clauses": [{{"feature": "...", "op": "...", "value": 0.0}}, ...],
  "stop_atr": 0.5-6.0,          // stop distance in ATR units
  "target_atr": 0.0-12.0,       // 0 disables the take profit
  "max_hold_bars": 1-500,       // time stop
  "hypothesis": "one sentence: WHY this should work"
}}

Rules:
- Every value must sit inside its range. Out-of-range values are refused.
- One clause per feature. Keep it to the conditions that carry the idea.
- Prefer a mechanism (forced sellers, funding squeeze, exhaustion, range break)
  over a shape you noticed in the last few candles.
- Emit one line; never put a raw newline inside a string.
"""


def prompt_text() -> str:
    lines = [f"- {name}: [{lo:g}, {hi:g}] — {desc}"
             for name, (lo, hi, desc) in sorted(FEATURES.items())]
    return _PROMPT.format(features="\n".join(lines))


def propose(client, symbol: str = "SOL-USD", market_context: str = "",
            proposer_fake=None) -> Composition:
    """Ask the model to compose a strategy from the vocabulary."""
    user = (prompt_text() + "\n\nMarket context (untrusted):\n"
            + (market_context or "(none)")[:3000]
            + "\n\nReturn only the JSON object.")
    if proposer_fake is not None and getattr(client, "fake", None) is not None:
        raw = proposer_fake(prompt_text(), user)
    else:
        raw = client.chat(system="You compose trading strategies as JSON data.",
                          user=user, temperature=0.95, max_tokens=1200,
                          json_mode=True)
    return parse(raw)


def validate_composition(comp: Composition, bars, cfg):
    """Run a composition through the SAME validation gate as any strategy."""
    from .propose import StrategySpec, validate
    spec = StrategySpec(
        name=f"composed_{comp.fingerprint()}", hypothesis=comp.hypothesis,
        family="composed", params={}, symbol=cfg.market.symbol, ts="")
    return validate(spec, bars, cfg, strategy=build(comp, cfg.strategy))


def is_novel(comp: Composition, prior: list[dict],
             overlap: float = 0.5) -> tuple[bool, str]:
    """Refuse a composition whose clause set overlaps an earlier one.

    Jaccard over clause labels: two ideas that share more than `overlap` of
    their conditions are the same bet wearing different numbers.
    """
    mine = {c.label() for c in comp.clauses}
    for p in prior:
        theirs = set(p.get("clause_set") or [])
        if not theirs:
            continue
        shared = len(mine & theirs)
        if shared and shared / len(mine | theirs) > overlap:
            return False, f"too similar to an earlier composition ({p.get('name')})"
        if p.get("fingerprint") == comp.fingerprint():
            return False, f"identical to an earlier composition ({p.get('name')})"
    return True, "novel"
