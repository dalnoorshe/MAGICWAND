"""A reader's record must be measured against what the market gave away free.

The bug this pins is the one that made the whole self-improvement loop fake, and
it hid in plain arithmetic: over the graded sample, 76% of 24h windows fell. So
"price will fall" was right 76% of the time without knowing anything. Raw hit
rates were 58-88%, two readers were *worse* than a constant bearish call, and
the one with 26 claims scored 88% and was handed 0.96 influence.

Two corrections, both tested here:

  * skill = hit rate - the base rate for the direction actually called;
  * the sample is counted in DISTINCT price paths. 575 claims over 31 distinct
    realized moves are 31 observations, not 575.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from magicwand.agents import AgentVerdict, _weight, synthesize
from magicwand.ledger import Ledger
from magicwand.reflect import reflect
from magicwand.research_loop import (MIN_WINDOWS, agent_track_record,
                                     _influence_from)


def _ledger(rows) -> Ledger:
    """rows: (agent, expected_move, realized_move, horizon_hours)"""
    d = tempfile.TemporaryDirectory()
    led = Ledger(Path(d.name) / "ledger.jsonl")
    led._tmpdir = d  # type: ignore[attr-defined]  keep it alive
    for i, (agent, expected, realized, horizon) in enumerate(rows):
        cid = f"c{i}"
        led.append("agent_claim", {
            "claim_id": cid, "agent": agent, "symbol": "SOL-USD",
            "expected_move": expected, "horizon_bars": horizon,
            "entry_price": 100.0, "ts": "2026-11-01T00:00:00Z"})
        led.append("grade", {
            "claim_id": cid, "expected_move": expected,
            "realized_move": realized,
            "verdict": "right" if (expected > 0) == (realized > 0) else "wrong",
            "magnitude_ratio": 1.0, "bars_taken": horizon,
            "exit_reason": "horizon"})
    return led


class TestBaseRate(unittest.TestCase):
    def test_always_bearish_in_a_falling_market_has_no_skill(self):
        """The core insight: repeating the drift scores skill exactly 0."""
        # 24 distinct windows, 18 fall (75%), 6 rise. Reader always says "down".
        rows = [("news", -0.02, -0.01 * (i + 1) if i < 18 else 0.01 * (i - 17), 24)
                for i in range(24)]
        tr = agent_track_record(_ledger(rows))["news"]
        self.assertAlmostEqual(tr["hit_rate"], 0.75, places=3)
        self.assertAlmostEqual(tr["base_rate"], 0.75, places=3)
        self.assertAlmostEqual(tr["skill"], 0.0, places=3)
        self.assertEqual(tr["influence"], 0.5,
                         "repeating the market's drift earned influence")

    def test_beating_the_base_rate_earns_influence(self):
        # Same market, but this reader calls the six up-windows as well.
        rows = ([("charts", -0.02, -0.01 * (i + 1), 24) for i in range(18)]
                + [("charts", 0.02, 0.01 * (i + 1), 24) for i in range(6)])
        tr = agent_track_record(_ledger(rows))["charts"]
        self.assertAlmostEqual(tr["hit_rate"], 1.0, places=3)
        # Null model: a random caller with the same 18-down/6-up mix scores
        # (18*0.75 + 6*0.25)/24 = 0.625, so skill is 0.375.
        self.assertAlmostEqual(tr["skill"], 0.375, places=3)
        self.assertEqual(tr["windows"], 24)
        self.assertGreater(tr["influence"], 0.5)
        self.assertTrue(tr["significant"])

    def test_worse_than_the_base_rate_loses_influence_but_is_never_inverted(self):
        """Anti-skill needs a reader that argues with the trend, not one that
        follows it — following it is exactly zero by construction."""
        rows = ([("derivatives", 0.02, -0.01 * (i + 1), 24) for i in range(18)]
                + [("derivatives", -0.02, 0.01 * (i + 1), 24) for i in range(6)])
        tr = agent_track_record(_ledger(rows))["derivatives"]
        self.assertEqual(tr["hit_rate"], 0.0)
        self.assertLess(tr["skill"], 0.0)
        self.assertLess(tr["influence"], 0.5)
        self.assertGreaterEqual(tr["influence"], 0.05,
                                "a bad reader was inverted instead of muted")


class TestEffectiveSample(unittest.TestCase):
    def test_one_price_path_is_one_observation(self):
        """Many claims graded off the same move must not look like many trials."""
        rows = [("social", -0.02, -0.06, 24) for _ in range(60)]
        tr = agent_track_record(_ledger(rows))["social"]
        self.assertEqual(tr["n"], 60)
        self.assertEqual(tr["windows"], 1)
        self.assertEqual(tr["influence"], 0.5,
                         "60 copies of one observation moved influence")

    def test_windows_are_counted_distinctly(self):
        rows = ([("social", -0.02, -0.01 * (i + 1), 24) for i in range(20)]
                + [("social", 0.02, 0.01 * (i + 1), 24) for i in range(5)])
        tr = agent_track_record(_ledger(rows))["social"]
        self.assertEqual(tr["windows"], 25)
        # (20*0.8 + 5*0.2)/25 = 0.68 null, hit is 1.0 -> skill 0.32.
        self.assertAlmostEqual(tr["skill"], 0.32, places=3)
        self.assertGreater(tr["influence"], 0.5)

    def test_below_the_window_floor_influence_is_neutral(self):
        rows = ([("charts", -0.02, -0.01 * (i + 1), 24)
                 for i in range(MIN_WINDOWS - 3)]
                + [("charts", 0.02, 0.01 * (i + 1), 24) for i in range(2)])
        tr = agent_track_record(_ledger(rows))["charts"]
        self.assertLess(tr["windows"], MIN_WINDOWS)
        self.assertEqual(tr["influence"], 0.5)

    def test_shrinkage_is_monotone_in_sample_size(self):
        small = _influence_from(0.2, 20)
        big = _influence_from(0.2, 500)
        self.assertGreater(big, small)
        self.assertLessEqual(big, 1.0)


class TestWeightsUseSkill(unittest.TestCase):
    def test_influence_drives_weight_not_hit_rate(self):
        track = {"news": {"n": 100, "hit_rate": 0.95, "influence": 0.4}}
        self.assertAlmostEqual(_weight("news", track), 0.4, places=3)

    def test_a_record_without_influence_is_neutral(self):
        self.assertEqual(_weight("news", {"news": {"n": 100, "hit_rate": 0.95}}), 0.5)
        self.assertEqual(_weight("news", {}), 0.5)
        self.assertEqual(_weight("news", None), 0.5)

    def test_synthesis_still_works_with_a_skill_based_record(self):
        track = {"news": {"influence": 0.9}, "social": {"influence": 0.2}}
        rep = synthesize([AgentVerdict(agent="news", stance="neutral",
                                       confidence=0.6, evidence_score=0.3),
                          AgentVerdict(agent="social", stance="neutral",
                                       confidence=0.6, evidence_score=-0.3)],
                         "SOL-USD", "t", track=track)
        self.assertGreater(rep.evidence_score, 0.0,
                           "the higher-influence reader should dominate")


class TestReflectionReportsSkill(unittest.TestCase):
    def test_findings_name_the_base_rate_and_the_window_count(self):
        rows = [("news", -0.02, -0.01 * (i + 1) if i < 18 else 0.01 * (i - 17), 24)
                for i in range(24)]
        from magicwand.gatekeeper import Policy
        ref = reflect(_ledger(rows), Policy())
        text = " ".join(f.observation for f in ref.findings if f.topic == "agent")
        self.assertIn("base", text)
        self.assertIn("independent windows", text)
        self.assertIn("skill", text)
        self.assertEqual(ref.agent_weights.get("news"), 0.5)


if __name__ == "__main__":
    unittest.main()
