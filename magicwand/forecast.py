"""MAGICWAND — an algorithmic trend forecaster.

This is the real thing the operator asked for: a model that computes a
probability for the next move, from numbers, causally, and then gets graded.

What it does
------------
For every bar it builds a small feature vector from bars 0..i ONLY, then a
logistic regression estimates P(close rises over the next H bars). The model is
trained **walk-forward**: for each test block it is fitted only on bars strictly
before that block, so no prediction ever sees its own answer.

What it outputs
---------------
    p_up      probability the close is higher H bars later
    trend     -1..1  (2 * p_up - 1), signed confidence
    regime    "calm" | "normal" | "volatile" from ATR percentile
    claim     a falsifiable one-line forecast, gradeable by the ledger

How it is judged
----------------
Not by how it sounds. Accuracy, Brier score, log loss, a calibration table, and
a comparison against two baselines: the base rate (always guess the majority
direction) and the coin flip (0.5 everywhere, Brier 0.25). A model that cannot
beat the base rate has learned nothing, and the report says so.

What it is NOT
--------------
Not a guarantee, not a trading system on its own, and not allowed to open
positions. It produces a probability; the gate and the veto rules still decide
what happens to money. Trend prediction is a weak signal in efficient markets —
the honest expectation is a small edge at best, and often none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .common import Bar


# --------------------------------------------------------------------------
# Features — strictly causal: row i depends on bars 0..i only
# --------------------------------------------------------------------------

FEATURES = (
    "r1", "r5", "r20", "ma_slope", "dist_ma",
    "vol_ratio", "range_pos", "vol_z", "rsi_n",
)


def _sma(xs: list[float], n: int, i: int) -> Optional[float]:
    if i + 1 < n:
        return None
    return sum(xs[i + 1 - n: i + 1]) / n


def _atr_at(bars: list[Bar], n: int, i: int) -> Optional[float]:
    if i + 1 < n:
        return None
    total = 0.0
    for k in range(i + 1 - n, i + 1):
        if k == 0:
            total += bars[k].high - bars[k].low
        else:
            pc = bars[k - 1].close
            total += max(bars[k].high - bars[k].low,
                         abs(bars[k].high - pc), abs(bars[k].low - pc))
    return total / n


def _rsi_at(bars: list[Bar], n: int, i: int) -> Optional[float]:
    if i + 1 < n + 1:
        return None
    gains = losses = 0.0
    for k in range(i + 1 - n, i + 1):
        d = bars[k].close - bars[k - 1].close
        if d > 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100.0 - 100.0 / (1.0 + rs)


def features_at(bars: list[Bar], i: int) -> Optional[dict]:
    """Feature row for bar i, or None when there is not enough history."""
    if i < 60:
        return None
    c = bars[i].close
    closes = [b.close for b in bars]

    def ret(n: int) -> Optional[float]:
        if i - n < 0 or closes[i - n] == 0:
            return None
        return c / closes[i - n] - 1.0

    r1, r5, r20 = ret(1), ret(5), ret(20)
    fast = _sma(closes, 10, i)
    slow = _sma(closes, 40, i)
    fast_prev = _sma(closes, 10, i - 5) if i >= 5 else None
    atr = _atr_at(bars, 14, i)
    atr_prev = _atr_at(bars, 14, i - 20) if i >= 20 else None
    rsi = _rsi_at(bars, 14, i)
    if None in (r1, r5, r20, fast, slow, fast_prev, atr, atr_prev, rsi):
        return None
    if atr == 0 or slow == 0:
        return None

    window = bars[max(0, i - 20): i + 1]
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    range_pos = ((c - lo) / (hi - lo)) if hi > lo else 0.5
    vols = [b.volume for b in window]
    vmean = sum(vols) / len(vols)
    vsd = (sum((v - vmean) ** 2 for v in vols) / len(vols)) ** 0.5
    vol_z = ((bars[i].volume - vmean) / vsd) if vsd > 0 else 0.0

    return {
        "r1": r1, "r5": r5, "r20": r20,
        "ma_slope": (fast - fast_prev) / abs(fast_prev) if fast_prev else 0.0,
        "dist_ma": c / slow - 1.0,
        "vol_ratio": atr / atr_prev if atr_prev else 1.0,
        "range_pos": range_pos,
        "vol_z": vol_z,
        "rsi_n": rsi / 100.0,
    }


# --------------------------------------------------------------------------
# Logistic regression — written out, because there is no numpy here
# --------------------------------------------------------------------------

@dataclass
class LogisticModel:
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    l2: float = 1.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, rows: list[list[float]], ys: list[int],
            epochs: int = 300, lr: float = 0.15) -> "LogisticModel":
        n = len(rows)
        d = len(rows[0]) if rows else 0
        self.mean = [sum(r[j] for r in rows) / n for j in range(d)]
        self.scale = []
        for j in range(d):
            var = sum((r[j] - self.mean[j]) ** 2 for r in rows) / n
            self.scale.append(math.sqrt(var) or 1.0)
        self.weights = [0.0] * d
        self.bias = 0.0
        for _ in range(epochs):
            gw = [0.0] * d
            gb = 0.0
            for r, y in zip(rows, ys):
                x = [(r[j] - self.mean[j]) / self.scale[j] for j in range(d)]
                p = self._sigmoid(sum(w * v for w, v in zip(self.weights, x)) + self.bias)
                err = p - y
                for j in range(d):
                    gw[j] += err * x[j]
                gb += err
            for j in range(d):
                self.weights[j] -= lr * (gw[j] / n + self.l2 * self.weights[j] / n)
            self.bias -= lr * gb / n
        return self

    def predict(self, row: list[float]) -> float:
        x = [(row[j] - self.mean[j]) / self.scale[j] for j in range(len(row))]
        return self._sigmoid(sum(w * v for w, v in zip(self.weights, x)) + self.bias)


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

@dataclass
class Forecast:
    ts: str
    symbol: str
    p_up: float
    horizon: int
    regime: str
    features: dict = field(default_factory=dict)

    @property
    def trend(self) -> float:
        return 2.0 * self.p_up - 1.0

    def claim_text(self) -> str:
        direction = "higher" if self.p_up >= 0.5 else "lower"
        return (f"close {direction} in {self.horizon} bars "
                f"(p={self.p_up:.2f}, {self.regime} regime)")

    def expected_move(self, vol: float) -> float:
        """Signed expectation sized by the model's edge, not its confidence."""
        return (self.trend) * abs(vol)


@dataclass
class Evaluation:
    n: int
    accuracy: float
    brier: float
    log_loss: float
    base_rate: float
    base_brier: float
    base_accuracy: float
    calibration: list[dict]
    by_regime: dict

    @property
    def beats_base(self) -> bool:
        """Skill means BOTH: better probabilities than climatology AND better
        than a coin flip on direction. One without the other is noise."""
        return (self.brier < self.base_brier - 1e-6
                and self.accuracy > 0.5 + 1e-9)

    def summary(self) -> str:
        if self.beats_base:
            verdict = "skill: beats climatology AND coin flip"
        elif self.brier >= self.base_brier:
            verdict = ("NO skill: worse probabilities than predicting the base "
                       "rate every time")
        else:
            verdict = ("NO skill: probabilities beat climatology but direction "
                       "is no better than a coin flip")
        return (f"{self.n} out-of-sample forecasts · accuracy {self.accuracy:.1%} "
                f"(climatology {self.base_accuracy:.1%}) · "
                f"Brier {self.brier:.4f} (climatology {self.base_brier:.4f}) · "
                f"{verdict}")


def build_dataset(bars: list[Bar], horizon: int = 6) -> tuple[list[list[float]], list[int], list[int]]:
    rows: list[list[float]] = []
    ys: list[int] = []
    idxs: list[int] = []
    for i in range(len(bars) - horizon):
        f = features_at(bars, i)
        if f is None:
            continue
        future = bars[i + horizon].close
        rows.append([f[k] for k in FEATURES])
        ys.append(1 if future > bars[i].close else 0)
        idxs.append(i)
    return rows, ys, idxs


def _regime(bars: list[Bar], i: int) -> str:
    atr = _atr_at(bars, 14, i)
    hist = [_atr_at(bars, 14, k) for k in range(max(1, i - 100), i + 1)]
    hist = [a for a in hist if a]
    if not atr or len(hist) < 20:
        return "normal"
    below = sum(1 for a in hist if a <= atr) / len(hist)
    if below < 0.33:
        return "calm"
    if below > 0.67:
        return "volatile"
    return "normal"


def _brier(ps: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)


def _logloss(ps: list[float], ys: list[int]) -> float:
    eps = 1e-12
    total = 0.0
    for p, y in zip(ps, ys):
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(ys)


def _calibration(ps: list[float], ys: list[int], bins: int = 5) -> list[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(sel),
            "predicted": round(sum(p for p, _ in sel) / len(sel), 3),
            "actual": round(sum(y for _, y in sel) / len(sel), 3),
        })
    return out


REGIMES = ("calm", "normal", "volatile")


def walk_forward(bars: list[Bar], horizon: int = 6, min_train: int = 200,
                 block: int = 50, by_regime: bool = False,
                 min_regime_train: int = 60
                 ) -> tuple[Optional[Evaluation], list[Forecast]]:
    """Train on the past, predict a block, roll forward. No peeking.

    `by_regime` fits a separate model per volatility regime instead of one model
    that averages the regimes away. A trend rule that works in a calm market and
    fails in a volatile one gets blended into a single middling coefficient by
    the pooled model; fitting each separately lets the slopes differ. It also
    divides the training data by three, so a regime below `min_regime_train`
    falls back to the pooled model and says so rather than pretending.
    """
    rows, ys, idxs = build_dataset(bars, horizon)
    if len(rows) < min_train + block:
        return None, []

    # The regime test looks at ATR's own trailing 100 bars, so it is causal.
    # Compute it once per bar: recomputing it inside the loop made this the
    # dominant cost of the whole evaluation.
    regime_of = [_regime(bars, i) for i in range(len(bars))]

    ps: list[float] = []
    outs: list[int] = []
    regimes: dict[str, list[tuple[float, int]]] = {}
    fell_back: dict[str, bool] = {r: False for r in REGIMES}
    start = min_train
    while start < len(rows):
        end = min(start + block, len(rows))
        # PURGE: a training label must resolve BEFORE the test block begins.
        # Labels look `horizon` bars ahead, so the last few training samples
        # overlap the test period and leak its outcome. Without this the model
        # is scored on information it could not have had.
        cutoff = idxs[start] - horizon
        train = [k for k in range(start) if idxs[k] < cutoff]
        pooled = LogisticModel().fit([rows[k] for k in train],
                                     [ys[k] for k in train])
        per_regime: dict[str, LogisticModel] = {}
        if by_regime:
            for name in REGIMES:
                sel = [k for k in train if regime_of[idxs[k]] == name]
                if len(sel) >= min_regime_train:
                    per_regime[name] = LogisticModel().fit(
                        [rows[k] for k in sel], [ys[k] for k in sel])
        for k in range(start, end):
            name = regime_of[idxs[k]]
            model = per_regime.get(name)
            if by_regime and model is None:
                fell_back[name] = True
            p = (model or pooled).predict(rows[k])
            ps.append(p)
            outs.append(ys[k])
            regimes.setdefault(name, []).append((p, ys[k]))
        start = end

    base = sum(outs) / len(outs)
    majority = 1 if base >= 0.5 else 0
    # The honest baseline is CLIMATOLOGY: predict the base rate every time.
    # Its Brier score is exactly base*(1-base). Comparing against a hard 0/1
    # prediction (Brier 0.448) made a worse-than-coin-flip model look skilful.
    climo_brier = base * (1.0 - base)
    ev = Evaluation(
        n=len(ps),
        accuracy=sum(1 for p, y in zip(ps, outs) if (p >= 0.5) == (y == 1)) / len(outs),
        brier=_brier(ps, outs),
        log_loss=_logloss(ps, outs),
        base_rate=base,
        base_brier=climo_brier,
        base_accuracy=sum(1 for y in outs if y == majority) / len(outs),
        calibration=_calibration(ps, outs),
        # Per regime, compared against that regime's OWN climatology: a regime
        # that is 60% "up" makes 60% accuracy worthless, exactly as a whole
        # sample that falls 76% of the time makes a bearish reader worthless.
        by_regime={r: {"n": len(v),
                       "accuracy": sum(1 for p, y in v if (p >= 0.5) == (y == 1)) / len(v),
                       "avg_p": sum(p for p, _ in v) / len(v),
                       "brier": _brier([p for p, _ in v], [y for _, y in v]),
                       "base_rate": sum(y for _, y in v) / len(v),
                       "base_brier": (lambda b: b * (1.0 - b))(
                           sum(y for _, y in v) / len(v)),
                       "fallback": fell_back.get(r, False)}
                   for r, v in regimes.items()},
    )

    # final model on everything, to forecast the NEXT bar
    final = LogisticModel().fit(rows, ys)
    tail: list[Forecast] = []
    for back in (0,):
        i = len(bars) - horizon - back
        f = features_at(bars, i)
        if f is None:
            continue
        name = regime_of[i] if i < len(regime_of) else _regime(bars, i)
        if by_regime:
            sel = [k for k in range(len(rows)) if regime_of[idxs[k]] == name]
            final_regime = (LogisticModel().fit([rows[k] for k in sel],
                                                [ys[k] for k in sel])
                            if len(sel) >= min_regime_train else final)
        else:
            final_regime = final
        p = final_regime.predict([f[k] for k in FEATURES])
        tail.append(Forecast(ts=bars[i].ts.isoformat(), symbol="",
                             p_up=p, horizon=horizon,
                             regime=name, features=f))
    return ev, tail
