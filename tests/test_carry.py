"""Funding carry: the accounting has to be right or the answer is a lie.

Two bugs were caught here by hand-checking, and both made carry look better than
it is:

  * the short perp leg was booked as `M0/M1 - 1` instead of `1 - M1/M0`. The
    reciprocal measures the move against the EXIT notional, and by AM-GM the two
    ratios sum positive, so a delta-neutral position appeared to earn +2095 bps
    of "basis" while SOL fell 48%.
  * a position still open at the end of the window was never closed, hiding its
    exit fee and its final basis.

These tests are the hand arithmetic that catches both.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from magicwand.carry import CarryConfig, run_carry
from magicwand.common import Bar

H = 8 * 3600 * 1000


def _t0() -> int:
    return int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _bars(prices: dict[int, float], hours: int) -> list[Bar]:
    """Hourly flat bars from a {hour: price} map (piecewise constant)."""
    t0 = _t0()
    out = []
    price = 100.0
    for h in range(hours):
        price = prices.get(h, price)
        ts = datetime.fromtimestamp((t0 + h * 3_600_000) / 1000, tz=timezone.utc)
        out.append(Bar(ts=ts, open=price, high=price, low=price, close=price,
                       volume=1.0))
    return out


def _funding(rates: list[float], marks: list[float] | None = None) -> list[dict]:
    t0 = _t0()
    marks = marks or [100.0] * len(rates)
    return [{"ts": t0 + k * H, "rate": r, "mark": m}
            for k, (r, m) in enumerate(zip(rates, marks))]


class TestNeutrality(unittest.TestCase):
    def test_a_neutral_position_does_not_profit_from_a_price_crash(self):
        """SOL down 48%, mark tracking spot: basis P&L must be ~zero."""
        rates = [0.0001] * 20
        marks = [100.0 * (0.995 ** k) for k in range(20)]
        bars = _bars({h: 100.0 * (0.995 ** (h // 8)) for h in range(160)}, 160)
        res = run_carry(_funding(rates, marks), bars,
                        CarryConfig(lookback=3, entry_bps=0.5, exit_bps=-1))
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.basis_bps, 0.0, places=6,
                               msg="a delta-neutral position booked basis P&L")

    def test_a_neutral_position_does_not_profit_from_a_rally_either(self):
        rates = [0.0001] * 20
        marks = [100.0 * (1.01 ** k) for k in range(20)]
        bars = _bars({h: 100.0 * (1.01 ** (h // 8)) for h in range(160)}, 160)
        res = run_carry(_funding(rates, marks), bars,
                        CarryConfig(lookback=3, entry_bps=0.5, exit_bps=-1))
        assert res is not None
        self.assertAlmostEqual(res.basis_bps, 0.0, places=6)


class TestAccounting(unittest.TestCase):
    def test_one_side_costs_two_legs_and_a_round_trip_costs_two_sides(self):
        rates = [0.0001] * 20
        bars = _bars({}, 160)
        cfg = CarryConfig(lookback=3, entry_bps=0.5, exit_bps=-1,
                          spot_fee_bps=10, perp_fee_bps=5, slip_bps_leg=2)
        res = run_carry(_funding(rates), bars, cfg)
        assert res is not None
        # The rule never exits, but the window end closes the position, so both
        # sides are charged: 2 x (10 + 5 + 2 + 2) = 38 bps.
        self.assertAlmostEqual(res.fees_bps, 38.0, places=6)
        self.assertAlmostEqual(res.round_trip_cost_bps, 38.0, places=6)

    def test_a_short_perp_entered_at_a_premium_earns_the_convergence(self):
        # mark 1% above spot on entry, back to spot on exit: the short leg gains.
        rates = [0.0001] * 6
        marks = [101.0, 101.0, 101.0, 100.0, 100.0, 100.0]
        bars = _bars({}, 60)
        res = run_carry(_funding(rates, marks), bars,
                        CarryConfig(lookback=2, entry_bps=0.5, exit_bps=-1))
        assert res is not None
        self.assertGreater(res.basis_bps, 90.0,
                           "converging basis was not credited")
        self.assertLess(res.basis_bps, 110.0)

    def test_a_position_still_open_at_the_end_is_closed_and_charged(self):
        rates = [0.0001] * 20
        bars = _bars({}, 160)
        res = run_carry(_funding(rates), bars,
                        CarryConfig(lookback=3, entry_bps=-100, exit_bps=-1000))
        assert res is not None
        # entry + exit both charged even though the rule never exits
        self.assertAlmostEqual(res.fees_bps, 38.0, places=6)

    def test_holding_the_whole_window_earns_exactly_the_funding_less_one_round_trip(self):
        rates = [0.001] * 30           # 10 bps per period
        bars = _bars({}, 240)
        res = run_carry(_funding(rates), bars,
                        CarryConfig(lookback=3, entry_bps=-100, exit_bps=-1000))
        assert res is not None
        # The rule needs `lookback` periods before it may enter, so it is in the
        # market for 27 of the 30 funding times, not all 30.
        self.assertAlmostEqual(res.funding_bps, 270.0, places=6)
        self.assertAlmostEqual(res.net_bps, 270.0 - 38.0, places=6)


class TestNoLookahead(unittest.TestCase):
    def test_a_future_funding_spike_cannot_pull_in_an_earlier_entry(self):
        # quiet, then one enormous funding period, then quiet again.
        rates = [0.0] * 6 + [0.01] + [0.0] * 6
        bars = _bars({}, 120)
        res = run_carry(_funding(rates), bars,
                        CarryConfig(lookback=1, entry_bps=1.0, exit_bps=-1))
        assert res is not None
        self.assertGreaterEqual(res.entries, 1)
        first = res.trades[0]["in_ts"]
        spike = _t0() + 6 * H
        self.assertGreaterEqual(first, spike,
                                "the position was opened before the funding "
                                "that justified it existed")

    def test_zero_funding_never_enters(self):
        res = run_carry(_funding([0.0] * 20), _bars({}, 160),
                        CarryConfig(lookback=3, entry_bps=1.0, exit_bps=0.0))
        assert res is not None
        self.assertEqual(res.entries, 0)
        self.assertEqual(res.net_bps, 0.0)
        self.assertIn("never entered", res.verdict())


class TestBreakeven(unittest.TestCase):
    def test_breakeven_is_the_round_trip_cost_over_the_average_funding(self):
        rates = [0.0002] * 40            # 2 bps per period
        bars = _bars({}, 320)
        res = run_carry(_funding(rates), bars,
                        CarryConfig(lookback=3, entry_bps=-100, exit_bps=-1000))
        assert res is not None
        self.assertAlmostEqual(res.avg_funding_per_period_bps, 2.0, places=6)
        self.assertAlmostEqual(res.breakeven_periods, 38.0 / 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
