"""MAGICWAND — strategies.

A strategy may only look at bars that have already closed. It returns an
Intent (or None) plus a plain-English reason. It never sees a price it
could not have seen, and it has no access to the broker or the risk engine.

Strategies included:
  * sma_trend     — trend following on SMA fast/slow (the default)
  * mean_revert   — buy dips below a rolling z-score, sell rallies above it
  * coin_flip     — the honest baseline. If a strategy cannot beat this,
                    it has no edge at all. It exists to keep us humble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import indicators as ind
from .common import Bar, Side
from .config import StrategyConfig


@dataclass(frozen=True)
class Intent:
    action: str                  # "enter" | "exit"
    side: Side
    reason: str
    stop_distance: float = 0.0   # in quote currency per unit
    target_distance: float = 0.0
    expected_move: float = 0.0   # the wand's CLAIM, fractional, graded later
    max_hold_bars: int = 0


class Strategy:
    name = "base"

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self._last_explain = ""

    def warmup(self) -> int:
        """Number of bars needed before signals are meaningful."""
        return max(self.cfg.slow, self.cfg.atr_period) + 1

    # ---- indicator cache ----
    # Recomputing an indicator on every bar is O(n^2) and made strategy
    # validation take minutes. `prepare` computes each series once, causally,
    # so index i still only ever sees bars 0..i.
    _cache: Optional[dict] = None
    _cache_n: int = 0

    def prepare(self, bars: list[Bar]) -> None:
        self._cache = None
        self._cache_n = len(bars)
        closes = [b.close for b in bars]
        self._cache = {
            "closes": closes,
            "sma_fast": ind.sma(closes, max(2, self.cfg.fast)),
            "sma_slow": ind.sma(closes, max(2, self.cfg.slow)),
            "z": ind.zscore(closes, max(2, self.cfg.slow)),
            "atr": ind.atr(bars, self.cfg.atr_period),
        }

    def _series(self, name: str, bars: list[Bar]):
        if self._cache is None or self._cache_n != len(bars):
            self.prepare(bars)
        return self._cache[name]  # type: ignore[index]

    def decide(self, bars: list[Bar], i: int, in_position: bool) -> Optional[Intent]:
        raise NotImplementedError

    # ---- helpers shared by the concrete strategies ----

    def _atr_at(self, bars: list[Bar], i: int) -> Optional[float]:
        # Recomputation per bar is O(n^2) in the worst case but n is small
        # (<= a few thousand bars) and clarity beats cleverness here.
        a = ind.atr(bars[: i + 1], self.cfg.atr_period)
        return a[i] if i < len(a) else None


class SmaTrend(Strategy):
    """Long when fast SMA is above slow SMA and price is above fast SMA.

    Trend following: it loses in chop and wins in sustained moves. That
    trade-off is the whole strategy, so chop is not a bug to be tuned away.
    """
    name = "sma_trend"

    def decide(self, bars, i, in_position):
        cfg = self.cfg
        if i < self.warmup():
            return None
        fast = self._series("sma_fast", bars)[i]
        slow = self._series("sma_slow", bars)[i]
        a = self._series("atr", bars)[i]
        if fast is None or slow is None or not a:
            return None

        px = bars[i].close
        long_ok = fast > slow and px > fast
        flat = fast <= slow

        if in_position:
            if flat:
                return Intent("exit", Side.SELL, "trend flipped: fast SMA fell below slow SMA")
            return None

        if long_ok:
            stop_d = cfg.stop_atr * a
            tgt_d = cfg.target_atr * a if cfg.target_atr > 0 else 0.0
            expected = tgt_d / px if tgt_d else 0.0
            return Intent(
                "enter", Side.BUY,
                f"trend up: fast SMA {fast:.2f} > slow SMA {slow:.2f}, price {px:.2f} above both",
                stop_distance=stop_d, target_distance=tgt_d,
                expected_move=expected, max_hold_bars=cfg.max_hold_bars,
            )
        return None


class MeanRevert(Strategy):
    """Buy when price is stretched below its rolling mean, exit on reversion.

    The opposite bet to trend following. Which one works depends on the
    regime, which is exactly why we measure instead of believing.
    """
    name = "mean_revert"

    def warmup(self) -> int:
        return max(self.cfg.slow, self.cfg.atr_period) + 1

    def decide(self, bars, i, in_position):
        cfg = self.cfg
        if i < self.warmup():
            return None
        z = self._series("z", bars)[i]
        a = self._series("atr", bars)[i]
        if z is None or not a:
            return None
        px = bars[i].close

        if in_position:
            if z >= 0:
                return Intent("exit", Side.SELL, f"price reverted to the mean (z={z:+.2f})")
            return None

        if z <= -getattr(self.cfg, "z_entry", 1.5):
            stop_d = cfg.stop_atr * a
            tgt_d = cfg.target_atr * a if cfg.target_atr > 0 else 0.0
            return Intent(
                "enter", Side.BUY,
                f"oversold: price {abs(z):.2f} sd below its {cfg.slow}-bar mean",
                stop_distance=stop_d, target_distance=tgt_d,
                expected_move=(tgt_d / px) if tgt_d else 0.0,
                max_hold_bars=cfg.max_hold_bars,
            )
        return None


class Breakout(Strategy):
    """Buy a close above the highest high of the last `lookback` bars.

    Trend/breakout family: different failure mode from SMA crossovers (it
    whipsaws in ranges but catches the start of moves). Included so the
    hypothesis generator has more than one real idea to choose from.
    """
    name = "breakout"

    def warmup(self) -> int:
        return max(self.cfg.slow, self.cfg.atr_period) + 1

    def decide(self, bars, i, in_position):
        cfg = self.cfg
        look = max(5, int(cfg.slow))
        if i < self.warmup():
            return None
        window = bars[i - look: i]
        a = self._series("atr", bars)[i]
        if not window or not a:
            return None
        px = bars[i].close
        prior_high = max(b.high for b in window)
        if in_position:
            if px < min(b.low for b in window):
                return Intent("exit", Side.SELL,
                              f"breakout failed: close {px:.2f} below {look}-bar low")
            return None
        if px > prior_high:
            stop_d = cfg.stop_atr * a
            tgt_d = cfg.target_atr * a if cfg.target_atr > 0 else 0.0
            return Intent(
                "enter", Side.BUY,
                f"breakout: close {px:.2f} above {look}-bar high {prior_high:.2f}",
                stop_distance=stop_d, target_distance=tgt_d,
                expected_move=(tgt_d / px) if tgt_d else 0.0,
                max_hold_bars=cfg.max_hold_bars)
        return None


class CoinFlip(Strategy):
    """Random entries with the same exit rules. The control group.

    If a real strategy's numbers look like this one's, it isn't a strategy.
    """
    name = "coin_flip"

    def __init__(self, cfg: StrategyConfig, seed: int = 7) -> None:
        super().__init__(cfg)
        import random
        self._rng = random.Random(seed)
        self._p = 0.05  # enter on roughly 5% of eligible bars

    def decide(self, bars, i, in_position):
        cfg = self.cfg
        if i < self.warmup():
            return None
        a = self._series("atr", bars)[i]
        if not a:
            return None
        px = bars[i].close
        if in_position:
            if self._rng.random() < self._p:
                return Intent("exit", Side.SELL, "random exit")
            return None
        if self._rng.random() < self._p:
            stop_d = cfg.stop_atr * a
            tgt_d = cfg.target_atr * a if cfg.target_atr > 0 else 0.0
            return Intent(
                "enter", Side.BUY, "random entry (control group — no edge expected)",
                stop_distance=stop_d, target_distance=tgt_d,
                expected_move=(tgt_d / px) if tgt_d else 0.0,
                max_hold_bars=cfg.max_hold_bars,
            )
        return None


REGISTRY = {
    "sma_trend": SmaTrend,
    "mean_revert": MeanRevert,
    "breakout": Breakout,
    "coin_flip": CoinFlip,
}

# --------------------------------------------------------------------------
# The audited catalog.
#
# Agents may only combine strategy families from this table with typed,
# range-checked parameters. They cannot submit code. This is what makes an
# autonomous agent safe to run unattended: its ideas are data, and the data
# is validated before anything trades.
# --------------------------------------------------------------------------

# family -> (allowed params, human description)
CATALOG: dict[str, dict] = {
    "sma_trend": {
        "desc": "long while a fast SMA is above a slow SMA and price is above both",
        "params": {"fast": (2, 200, int), "slow": (5, 400, int),
                   "stop_atr": (0.5, 6.0, float), "target_atr": (0.0, 12.0, float),
                   "max_hold_bars": (1, 500, int)},
    },
    "mean_revert": {
        "desc": "buy when price is stretched below its rolling mean, exit on reversion",
        "params": {"slow": (5, 400, int), "z_entry": (0.5, 4.0, float),
                   "stop_atr": (0.5, 6.0, float), "target_atr": (0.0, 12.0, float),
                   "max_hold_bars": (1, 500, int)},
    },
    "breakout": {
        "desc": "buy a close above the highest high of the last N bars",
        "params": {"slow": (5, 400, int), "stop_atr": (0.5, 6.0, float),
                   "target_atr": (0.0, 12.0, float), "max_hold_bars": (1, 500, int)},
    },
}


def build_parameterised(family: str, params: dict) -> Strategy:
    """Build a strategy from a validated catalog spec. Raises on anything odd.

    This is the ONLY way an agent-invented strategy reaches the backtester.
    """
    if family not in CATALOG:
        raise ValueError(f"unknown strategy family {family!r}")
    spec = CATALOG[family]
    clean: dict = {}
    for key, value in params.items():
        if key not in spec["params"]:
            raise ValueError(f"parameter {key!r} is not allowed for {family!r}")
        lo, hi, typ = spec["params"][key]
        try:
            v = typ(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"parameter {key!r} must be {typ.__name__}") from e
        if not (lo <= v <= hi):
            raise ValueError(f"parameter {key!r}={v} outside [{lo}, {hi}]")
        clean[key] = v
    defaults = {"name": family, "fast": 12, "slow": 48, "atr_period": 14,
                "stop_atr": 2.0, "target_atr": 4.0, "max_hold_bars": 72}
    defaults.update(clean)
    return build(StrategyConfig(**defaults))


def build(cfg: StrategyConfig, seed: int = 7) -> Strategy:
    cls = REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(f"unknown strategy {cfg.name!r}; have {sorted(REGISTRY)}")
    if cls is CoinFlip:
        return cls(cfg, seed=seed)
    return cls(cfg)
