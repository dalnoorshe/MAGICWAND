"""Sizing from a measured edge: the part that decides whether money is lost.

The rule this file exists to defend: **an unproven edge sizes to zero**, not to a
cautious percentage. Every other module asks "are we right?"; this one decides
how much is at stake, and it must fail to zero when the answer is "we do not
know".
"""

from __future__ import annotations

import math
import unittest

from magicwand.sizing import (EdgeEstimate, edge_from_returns,
                              edge_from_trades, size_from_edge)


class TestEdgeEstimate(unittest.TestCase):
    def test_standard_error_shrinks_with_more_trades(self):
        few = EdgeEstimate(10.0, 80.0, 10)
        many = EdgeEstimate(10.0, 80.0, 1000)
        self.assertGreater(few.se_bps, many.se_bps)
        self.assertAlmostEqual(many.se_bps, 80.0 / math.sqrt(1000), places=6)

    def test_lower_bound_is_below_the_mean(self):
        e = EdgeEstimate(10.0, 80.0, 100)
        self.assertLess(e.lower_bound_bps(), e.mean_bps)
        self.assertAlmostEqual(e.lower_bound_bps(2.0), 10.0 - 16.0, places=6)

    def test_one_trade_has_no_usable_error(self):
        e = edge_from_returns([50.0])
        self.assertEqual(e.n, 1)
        self.assertEqual(e.se_bps, float("inf"))
        self.assertFalse(size_from_edge(e, 0.0).ok)

    def test_no_trades_is_not_an_edge(self):
        e = edge_from_returns([])
        self.assertEqual(e.n, 0)
        self.assertFalse(size_from_edge(e, 0.0).ok)

    def test_non_finite_inputs_are_ignored(self):
        e = edge_from_returns([10.0, float("nan"), 20.0, float("inf")])
        self.assertEqual(e.n, 2)
        self.assertTrue(math.isfinite(e.mean_bps))


class TestTheRule(unittest.TestCase):
    def test_an_edge_indistinguishable_from_zero_sizes_to_zero(self):
        # +2 bps mean with 8 bps standard error: not a 2 bps edge.
        e = EdgeEstimate(2.0, 80.0, 100)
        s = size_from_edge(e, per_trade_cost_bps=0.0)
        self.assertFalse(s.ok)
        self.assertEqual(s.fraction, 0.0)
        self.assertIn("does not cover costs", s.reason)

    def test_costs_come_off_before_any_sizing(self):
        e = EdgeEstimate(50.0, 40.0, 400)      # se = 2 bps, lower = 46 bps
        self.assertTrue(size_from_edge(e, per_trade_cost_bps=10.0).ok)
        self.assertFalse(size_from_edge(e, per_trade_cost_bps=50.0).ok,
                         "a 38 bps round trip was ignored")

    def test_a_large_negative_edge_never_goes_short(self):
        e = EdgeEstimate(-60.0, 40.0, 400)
        s = size_from_edge(e, 0.0)
        self.assertFalse(s.ok)
        self.assertEqual(s.fraction, 0.0)

    def test_a_strong_edge_is_capped_not_levered(self):
        e = EdgeEstimate(400.0, 20.0, 5000)
        s = size_from_edge(e, 0.0, max_fraction=0.10)
        self.assertTrue(s.ok)
        self.assertLessEqual(s.fraction, 0.10)

    def test_half_kelly_is_applied(self):
        # Chosen so the raw Kelly stays BELOW the cap, otherwise the cap hides
        # the haircut and the test proves nothing.
        e = EdgeEstimate(200.0, 2000.0, 10_000)   # se 20 bps -> lower 160 bps
        s = size_from_edge(e, 0.0, max_fraction=1.0)
        mu = s.lower_bps / 10_000.0
        sigma = 2000.0 / 10_000.0
        self.assertAlmostEqual(s.fraction, 0.5 * mu / sigma ** 2, places=6)
        self.assertLess(s.fraction, 1.0, "the cap masked the haircut")

    def test_no_estimate_is_refused_rather_than_defaulted(self):
        s = size_from_edge(None, 38.0)
        self.assertFalse(s.ok)
        self.assertEqual(s.fraction, 0.0)
        self.assertIn("confidence alone", s.reason)


class TestRiskEngineIntegration(unittest.TestCase):
    def test_without_an_edge_the_old_ceilings_still_apply(self):
        from magicwand.common import utcnow
        from magicwand.config import RiskConfig
        from magicwand.risk import RiskEngine
        r = RiskEngine(RiskConfig()).size_entry(500.0, 70.0, 2.0, 0, utcnow())
        self.assertTrue(r.allowed)

    def test_an_unproven_edge_blocks_the_order_entirely(self):
        from magicwand.common import utcnow
        from magicwand.config import RiskConfig
        from magicwand.risk import RiskEngine
        e = EdgeEstimate(2.0, 80.0, 50)
        r = RiskEngine(RiskConfig()).size_entry(500.0, 70.0, 2.0, 0, utcnow(),
                                               edge=e, per_trade_cost_bps=38.0)
        self.assertFalse(r.allowed)
        self.assertEqual(r.qty, 0.0)
        self.assertIn("no bet", r.reason)

    def test_a_proven_edge_caps_below_the_ceiling(self):
        from magicwand.common import utcnow
        from magicwand.config import RiskConfig
        from magicwand.risk import RiskEngine
        when, eng = utcnow(), RiskEngine(RiskConfig())
        ceiling = eng.size_entry(500.0, 70.0, 2.0, 0, when)
        e = EdgeEstimate(60.0, 40.0, 200)
        limited = eng.size_entry(500.0, 70.0, 2.0, 0, when, edge=e,
                                 per_trade_cost_bps=38.0)
        self.assertTrue(limited.allowed)
        self.assertLess(limited.qty, ceiling.qty)
        self.assertIn("kelly", limited.reason)

    def test_non_finite_equity_is_still_refused_before_sizing(self):
        from magicwand.common import utcnow
        from magicwand.config import RiskConfig
        from magicwand.risk import RiskEngine
        e = EdgeEstimate(60.0, 40.0, 200)
        r = RiskEngine(RiskConfig()).size_entry(float("nan"), 70.0, 2.0, 0,
                                               utcnow(), edge=e)
        self.assertFalse(r.allowed)


if __name__ == "__main__":
    unittest.main()


class TestTradeConversion(unittest.TestCase):
    """Trade.return_pct is fractional. A 100x unit error here argues for a bet."""

    def test_a_one_percent_trade_is_a_hundred_bps(self):
        class FakeTrade:
            return_pct = 0.01
        e = edge_from_trades([FakeTrade()])
        self.assertAlmostEqual(e.mean_bps, 100.0, places=6)

    def test_a_losing_strategy_reads_as_losing(self):
        class FakeTrade:
            def __init__(self, r):
                self.return_pct = r
        e = edge_from_trades([FakeTrade(-0.0114) for _ in range(50)])
        self.assertAlmostEqual(e.mean_bps, -114.0, places=6)
        self.assertFalse(size_from_edge(e, 140.0).ok)
