"""MAGICWAND — funding-rate carry, the first non-predictive experiment.

Every question this project has asked so far was predictive: which way will the
price go? Six branches said no. This one asks a different kind of question.

Perpetual swaps pay a FUNDING rate every 8 hours. When it is positive, longs pay
shorts. So you can hold one unit of spot and one unit of perp short — market
neutral, no view on price — and collect that payment for as long as it lasts.
The edge, if there is one, is not a forecast. It is a fee you are paid for taking
the crowded side of a persistent imbalance.

The honest accounting is what makes or breaks it:

  * funding is collected on the PERP leg only, at the published rate, on the
    live mark price;
  * the two legs are tracked separately, so the basis (mark vs spot) is a real
    P&L line and not hand-waved away;
  * fees and slippage are charged on all FOUR legs of a round trip (open spot,
    open perp, close spot, close perp);
  * the decision at each funding time uses only funding already published.

What this model does NOT claim
    * No liquidation of the perp leg. A violent up-move needs margin on the
      short leg; without it you are liquidated and left holding spot, which is
      no longer neutral. Real carry dies this way, not from funding flipping.
    * No exchange/custody risk, which is the actual reason carry yields exist.
    * No margin or collateral requirement, so returns are stated per unit of
      NOTIONAL, not per unit of capital-at-risk.
    * Both legs on one venue. Split across two venues, capital doubles.

Read the numbers with all four of those in mind.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import data as data_mod
from .common import utcnow
from .markets import binance_perp, binance_spot


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def funding_history(symbol: str, days: int = 800) -> list[dict]:
    """Funding records with the mark price, newest last.

    Paged forward from a start time: the venue's default page is only its most
    recent 500 records, which silently truncated an earlier experiment to 137
    days. Never trust a default page size.
    """
    from .research import SourceError, _get_json

    pair = binance_perp(symbol)
    if not pair:
        return []
    now_ms = int(utcnow().timestamp() * 1000)
    out: dict[int, dict] = {}
    start = now_ms - days * 86_400_000
    try:
        while start < now_ms:
            url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}"
                   f"&startTime={start}&endTime={now_ms}&limit=1000")
            rows = _get_json(url)
            if not rows:
                break
            for row in rows:
                ts = int(row["fundingTime"])
                out[ts] = {"ts": ts, "rate": float(row["fundingRate"]),
                           "mark": float(row.get("markPrice") or 0.0)}
            if len(rows) < 1000:
                break
            nxt = int(rows[-1]["fundingTime"]) + 1
            if nxt <= start:
                break
            start = nxt
    except (SourceError, ValueError, KeyError, TypeError):
        pass
    return [out[k] for k in sorted(out)]


def spot_at(bars: list, ts_ms: int) -> Optional[float]:
    """Spot close of the last hourly bar that had closed by `ts_ms`.

    Binary search, not a linear scan: this is called once per funding period and
    a scan would make the whole simulation quadratic.
    """
    epochs = _epochs_of(bars)
    k = bisect_right(epochs, ts_ms) - 1
    return bars[k].close if k >= 0 else None


_EPOCH_CACHE: dict[int, tuple[int, list[int]]] = {}


def _epochs_of(bars: list) -> list[int]:
    key = id(bars)
    hit = _EPOCH_CACHE.get(key)
    if hit is not None and hit[0] == len(bars):
        return hit[1]
    epochs = [int(b.ts.replace(tzinfo=timezone.utc).timestamp() * 1000) for b in bars]
    _EPOCH_CACHE[key] = (len(bars), epochs)
    return epochs


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class CarryConfig:
    lookback: int = 3              # funding periods averaged before deciding
    entry_bps: float = 1.0         # enter when that average is at least this
    exit_bps: float = 0.0          # leave when it drops below this
    # Binance VIP0 taker: 0.10% spot, 0.05% USD-M futures. Both legs, both ways.
    spot_fee_bps: float = 10.0
    perp_fee_bps: float = 5.0
    slip_bps_leg: float = 2.0      # assumed, per leg
    notional: float = 1000.0       # accounting unit


@dataclass
class CarryResult:
    periods: int
    periods_held: int
    entries: int
    funding_bps: float             # collected on the perp leg
    basis_bps: float               # spot leg + perp leg mark-to-market
    fees_bps: float                # fees + slippage actually paid
    net_bps: float
    net_pct: float
    annualised_pct: float
    max_drawdown_pct: float
    worst_period_bps: float
    spot_return_pct: float
    round_trip_cost_bps: float
    avg_funding_per_period_bps: float
    breakeven_periods: Optional[float]
    avg_hold_periods: float
    trades: list[dict] = field(default_factory=list)
    curve: list[float] = field(default_factory=list)

    def verdict(self) -> str:
        if self.entries == 0:
            return "never entered: funding never cleared the entry threshold"
        if self.net_bps > 0:
            return "positive net carry on this sample (see the caveats above)"
        if self.funding_bps > self.fees_bps:
            return "funding positive but the basis and fees ate it"
        return "fees exceeded the funding collected"


def run_carry(funding: list[dict], bars: list,
              cfg: Optional[CarryConfig] = None) -> Optional[CarryResult]:
    """Walk the funding series. The decision at each funding time uses only rates
    already published, and executes at that time's own spot and mark."""
    cfg = cfg or CarryConfig()
    if len(funding) < cfg.lookback + 2:
        return None
    # One SIDE = two legs (spot + perp). A round trip pays both sides, plus
    # slippage on all four fills.
    side_cost = (cfg.spot_fee_bps + cfg.perp_fee_bps) + 2 * cfg.slip_bps_leg
    round_trip = 2 * side_cost

    open_pos: Optional[dict] = None
    funding_bps = basis_bps = fees_bps = 0.0
    cum = 0.0
    entries = periods_held = 0
    holds: list[int] = []
    trades: list[dict] = []
    curve: list[float] = []
    worst = 0.0
    last_marks: Optional[tuple[float, float]] = None

    for i, rec in enumerate(funding):
        rate = rec["rate"]
        mark = rec["mark"] or 0.0
        spot = spot_at(bars, rec["ts"]) or 0.0
        if not spot or not mark:
            continue

        # A position open across this funding time collects (or pays) it.
        if open_pos is not None:
            got = rate * 10_000.0
            funding_bps += got
            cum += got
            periods_held += 1
            open_pos["periods"] += 1
            open_pos["funding"] += got

        window = [r["rate"] for r in funding[max(0, i - cfg.lookback + 1): i + 1]]
        avg_bps = (sum(window) / len(window) * 10_000.0) if window else 0.0

        if open_pos is not None and avg_bps < cfg.exit_bps:
            # Spot leg is LONG: return = S1/S0 - 1.
            # Perp leg is SHORT: return = 1 - M1/M0, NOT M0/M1 - 1. The
            # reciprocal form measures the move relative to the EXIT notional
            # and silently profits from volatility itself (by AM-GM the two
            # ratios sum positive), which on a 48% decline fabricated +2095 bps
            # of "basis" and made a losing carry look like a winner.
            basis = ((spot / open_pos["spot_in"] - 1.0)
                     + (1.0 - mark / open_pos["mark_in"])) * 10_000.0
            basis_bps += basis
            cum += basis
            cum -= side_cost
            fees_bps += side_cost
            holds.append(open_pos["periods"])
            trades.append({"in_ts": open_pos["ts"], "out_ts": rec["ts"],
                           "periods": open_pos["periods"],
                           "funding_bps": round(open_pos["funding"] + rate * 10_000.0, 2),
                           "basis_bps": round(basis, 2),
                           "net_bps": round(open_pos["funding"] + rate * 10_000.0
                                            + basis - round_trip, 2)})
            open_pos = None
        elif open_pos is None and i >= cfg.lookback - 1 and avg_bps >= cfg.entry_bps:
            cum -= side_cost
            fees_bps += side_cost
            entries += 1
            open_pos = {"ts": rec["ts"], "spot_in": spot, "mark_in": mark,
                        "periods": 0, "funding": 0.0}

        last_marks = (spot, mark)
        open_pos_funding = open_pos
        curve.append(cum)
        if len(curve) > 1:
            worst = min(worst, curve[-1] - curve[-2])

    if not curve:
        return None
    # Close anything still open at the last marks we saw. Leaving it open would
    # quietly hide the exit fee and the basis at the end of the window.
    if open_pos is not None and last_marks is not None:
        spot_end, mark_end = last_marks
        basis = ((spot_end / open_pos["spot_in"] - 1.0)
                 + (1.0 - mark_end / open_pos["mark_in"])) * 10_000.0
        basis_bps += basis
        cum += basis - side_cost
        fees_bps += side_cost
        trades.append({"in_ts": open_pos["ts"], "out_ts": funding[-1]["ts"],
                       "periods": open_pos["periods"],
                       "funding_bps": round(open_pos["funding"], 2),
                       "basis_bps": round(basis, 2),
                       "net_bps": round(open_pos["funding"] + basis - round_trip, 2)})
        curve[-1] = cum

    net = funding_bps + basis_bps - fees_bps
    spot_first = spot_at(bars, funding[0]["ts"]) or 0.0
    spot_last = spot_at(bars, funding[-1]["ts"]) or 0.0
    span_days = max(1.0, (funding[-1]["ts"] - funding[0]["ts"]) / 86_400_000.0)
    avg_per_period = funding_bps / max(1, periods_held)
    peak = 0.0
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    return CarryResult(
        periods=len(curve),
        periods_held=periods_held,
        entries=entries,
        funding_bps=round(funding_bps, 2),
        basis_bps=round(basis_bps, 2),
        fees_bps=round(fees_bps, 2),
        net_bps=round(net, 2),
        net_pct=round(net / 100.0, 3),
        annualised_pct=round(net / 100.0 * 365.0 / span_days, 2),
        max_drawdown_pct=round(max_dd / 100.0, 3),
        worst_period_bps=round(worst, 2),
        spot_return_pct=(round((spot_last / spot_first - 1.0) * 100.0, 2)
                         if spot_first else 0.0),
        round_trip_cost_bps=round(round_trip, 2),
        avg_funding_per_period_bps=round(avg_per_period, 3),
        breakeven_periods=(round(round_trip / avg_per_period, 1)
                           if avg_per_period > 0 else None),
        avg_hold_periods=(round(sum(holds) / len(holds), 1) if holds else 0.0),
        trades=trades,
        curve=[round(c, 2) for c in curve],
    )
