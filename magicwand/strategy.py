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
        closes = [b.close for b in bars[: i + 1]]
        fast = ind.sma(closes, cfg.fast)[i]
        slow = ind.sma(closes, cfg.slow)[i]
        a = ind.atr(bars[: i + 1], cfg.atr_period)[i]
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
        closes = [b.close for b in bars[: i + 1]]
        z = ind.zscore(closes, cfg.slow)[i]
        a = ind.atr(bars[: i + 1], cfg.atr_period)[i]
        if z is None or not a:
            return None
        px = bars[i].close

        if in_position:
            if z >= 0:
                return Intent("exit", Side.SELL, f"price reverted to the mean (z={z:+.2f})")
            return None

        if z <= -1.5:
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
        a = ind.atr(bars[: i + 1], cfg.atr_period)[i]
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
    "coin_flip": CoinFlip,
}


def build(cfg: StrategyConfig, seed: int = 7) -> Strategy:
    cls = REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(f"unknown strategy {cfg.name!r}; have {sorted(REGISTRY)}")
    if cls is CoinFlip:
        return cls(cfg, seed=seed)
    return cls(cfg)
