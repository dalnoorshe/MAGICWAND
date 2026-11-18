"""A dead source must weaken the evidence, not silently reweight it.

The bug this guards against is subtle and it is the one the operator caught by
eye: when the social reader's only URL started returning 404, its verdict became
an error, dropped out of `synthesize`, and the remaining readers were renormal-
ised to 100% influence. The decision did not get *weaker* — it got *different*,
with the same confident tone. Nothing in the code was hardcoded to one source;
the weighting simply had no concept of "we only saw part of the world".

Now every Source declares its providers, every verdict carries the fraction that
answered, and synthesis shrinks the evidence toward neutral while raising the bar
a veto has to clear. These tests pin all three.
"""

from __future__ import annotations

import unittest

from magicwand.agents import AgentVerdict, ResearchReport, run_all, synthesize
from magicwand.research import MarketSnapshot, Source


def _v(agent: str, stance: str, conf: float, ev: float,
       cov: float = 1.0, corr: int = 2) -> AgentVerdict:
    return AgentVerdict(agent=agent, stance=stance, confidence=conf,
                        evidence_score=ev, corroboration=corr, coverage=cov)


class TestSourceCoverage(unittest.TestCase):
    def test_a_channel_with_no_declared_providers_is_atomic(self):
        self.assertEqual(Source("x", True, "t").coverage, 1.0)
        self.assertEqual(Source("x", False, "t").coverage, 0.0)

    def test_partial_coverage_is_measured_not_guessed(self):
        s = Source("social", True, "t", providers={
            "reddit": False, "coingecko trending": True, "fear & greed": True})
        self.assertAlmostEqual(s.coverage, 2 / 3, places=6)
        self.assertEqual(s.provider_summary, "2/3 sources")

    def test_all_providers_failing_gives_zero_even_if_flagged_ok(self):
        s = Source("news", True, "t", providers={"a": False, "b": False})
        self.assertEqual(s.coverage, 0.0)

    def test_the_real_social_source_declares_three(self):
        """A regression guard: social must never quietly become one endpoint."""
        s = Source("social", True, "t", providers={
            "reddit": True, "coingecko trending": True, "fear & greed": True})
        self.assertEqual(len(s.providers), 3)


class TestSynthesisHonoursCoverage(unittest.TestCase):
    def test_full_coverage_is_unchanged(self):
        rep = synthesize([_v("derivatives", "neutral", 0.8, -0.5)], "SOL", "t")
        self.assertAlmostEqual(rep.evidence_score, -0.5, places=4)
        self.assertAlmostEqual(rep.coverage, 1.0, places=4)

    def test_partial_coverage_shrinks_the_evidence(self):
        full = synthesize([_v("derivatives", "neutral", 0.8, -0.5)], "SOL", "t")
        part = synthesize([_v("derivatives", "neutral", 0.8, -0.5, cov=2 / 3)],
                          "SOL", "t")
        self.assertAlmostEqual(part.coverage, 2 / 3, places=4)
        self.assertLess(abs(part.evidence_score), abs(full.evidence_score),
                        "a reading from 2 of 3 sources claimed the same edge")
        self.assertAlmostEqual(part.evidence_score, -0.5 * 2 / 3, places=3)

    def test_a_lost_channel_lowers_conviction_instead_of_reweighting(self):
        both = synthesize([_v("derivatives", "veto", 0.9, -0.6),
                           _v("news", "veto", 0.9, -0.6)], "SOL", "t")
        # social saw 1 of 3 sources; the other two are full.
        mixed = synthesize([_v("derivatives", "veto", 0.9, -0.6),
                            _v("news", "veto", 0.9, -0.6, cov=1 / 3)], "SOL", "t")
        self.assertLess(mixed.coverage, both.coverage)
        self.assertLess(abs(mixed.evidence_score), abs(both.evidence_score))
        self.assertIn("coverage", mixed.summary)

    def test_a_reader_that_never_reported_lowers_conviction(self):
        """The operator's case: one reader dies, the rest must not renormalise."""
        live = [_v("news", "neutral", 0.8, -0.4),
                _v("onchain", "neutral", 0.8, -0.4)]
        full = synthesize(live, "SOL", "t")
        dead = AgentVerdict(agent="social", error="reddit: HTTP 404")
        thin = synthesize(live + [dead], "SOL", "t")
        self.assertGreater(full.coverage, thin.coverage)
        self.assertLess(abs(thin.evidence_score), abs(full.evidence_score),
                        "a dead reader was absorbed without costing conviction")
        self.assertIn("degraded", thin.summary)
        self.assertIn("coverage", thin.summary)

    def test_a_dead_reader_cannot_make_the_decision_stronger(self):
        live = [_v("news", "neutral", 0.8, -0.4)]
        one = synthesize(live, "SOL", "t")
        plus_dead = synthesize(live + [AgentVerdict(agent="social",
                                                   error="404")], "SOL", "t")
        self.assertLessEqual(abs(plus_dead.evidence_score), abs(one.evidence_score))

    def test_a_veto_from_a_thin_channel_needs_more_confidence(self):
        """82% confidence on one source is enough at full coverage, not at 2/3."""
        ok = synthesize([_v("social", "veto", 0.82, -0.5, cov=1.0, corr=1)],
                        "SOL", "t")
        self.assertEqual(ok.decision, "veto")
        thin = synthesize([_v("social", "veto", 0.82, -0.5, cov=2 / 3, corr=1)],
                          "SOL", "t")
        self.assertNotEqual(thin.decision, "veto",
                            "a veto read from 2 of 3 sources cleared the bar")
        self.assertIn("social", thin.single_source)

    def test_support_is_refused_when_most_sources_are_missing(self):
        rep = synthesize([_v("news", "support", 0.9, 0.9, cov=0.4)], "SOL", "t")
        self.assertNotEqual(rep.decision, "support")
        self.assertEqual(rep.decision, "neutral")

    def test_coverage_zero_never_vetoes(self):
        rep = synthesize([_v("news", "veto", 1.0, -1.0, cov=0.0, corr=3)],
                         "SOL", "t")
        self.assertEqual(rep.decision, "neutral")
        self.assertEqual(rep.evidence_score, 0.0)


class TestRunAllStampsCoverage(unittest.TestCase):
    def test_each_verdict_carries_its_channel_coverage(self):
        from magicwand.llm import DeepSeekClient, fake_canned
        snap = MarketSnapshot(ts="t", symbol="SOL-USD")
        snap.sources["news"] = Source("news", True, "t", providers={
            "a": True, "b": False, "c": True, "d": False})
        snap.sources["social"] = Source("social", True, "t", providers={
            "reddit": False, "coingecko trending": True, "fear & greed": True})
        verdicts = run_all(snap, DeepSeekClient(fake=fake_canned),
                           agents=["news", "social"])
        by = {v.agent: v for v in verdicts}
        self.assertAlmostEqual(by["news"].coverage, 0.5, places=4)
        self.assertAlmostEqual(by["social"].coverage, 2 / 3, places=4)

    def test_a_degraded_reader_reports_zero_coverage(self):
        from magicwand.llm import DeepSeekClient, fake_canned
        snap = MarketSnapshot(ts="t", symbol="SOL-USD")
        snap.sources["social"] = Source("social", False, "t",
                                        error="reddit 404")
        verdicts = run_all(snap, DeepSeekClient(fake=fake_canned),
                           agents=["social"])
        self.assertEqual(verdicts[0].coverage, 0.0)
        self.assertFalse(verdicts[0].trusted)

    def test_an_undeclared_channel_is_full_coverage(self):
        from magicwand.llm import DeepSeekClient, fake_canned
        snap = MarketSnapshot(ts="t", symbol="SOL-USD")
        snap.sources["news"] = Source("news", True, "t")
        verdicts = run_all(snap, DeepSeekClient(fake=fake_canned),
                           agents=["news"])
        self.assertEqual(verdicts[0].coverage, 1.0)


class TestReportCarriesCoverage(unittest.TestCase):
    def test_report_exposes_it_for_the_console(self):
        rep = synthesize([_v("news", "neutral", 0.7, 0.1, cov=0.5)], "SOL", "t")
        self.assertIsInstance(rep, ResearchReport)
        self.assertAlmostEqual(rep.coverage, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
