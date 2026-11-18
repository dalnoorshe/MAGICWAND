"""The doctor's job is to distinguish "our code is broken" from "the internet
said no", and to keep doing it even when every source is down.

Two rules worth testing:

  * a probe that raises is a row in the report, never a crash — the whole point
    is to run this when things are broken;
  * running it must not disturb the runtime's Reddit backoff, or a diagnostic
    would quietly cost an extra request on every tick afterwards.
"""

from __future__ import annotations

import unittest
from datetime import timedelta

from magicwand import doctor
from magicwand import research as research_mod
from magicwand.config import Config


class TestDoctor(unittest.TestCase):
    def test_all_ok(self):
        probes = [("a", lambda: (True, "fine")), ("b", lambda: (True, "also fine"))]
        report = doctor.run(Config(), probes=probes)
        self.assertTrue(report["all_ok"])
        self.assertEqual(report["answered"], 2)
        self.assertEqual([r["name"] for r in report["results"]], ["a", "b"])

    def test_a_failing_probe_is_a_row_not_an_exception(self):
        def boom():
            raise RuntimeError("HTTP 429 from somewhere")

        report = doctor.run(Config(), probes=[("bad", boom),
                                              ("good", lambda: (True, "ok"))])
        self.assertFalse(report["all_ok"])
        bad = report["results"][0]
        self.assertFalse(bad["ok"])
        self.assertIn("HTTP 429", bad["detail"])
        self.assertIn("RuntimeError", bad["detail"])

    def test_a_source_that_answers_no_is_reported_as_no(self):
        report = doctor.run(Config(), probes=[("reddit", lambda: (False, "HTTP 403"))])
        self.assertFalse(report["all_ok"])
        self.assertEqual(report["answered"], 0)
        self.assertIn("403", report["results"][0]["detail"])

    def test_report_is_readable_and_names_the_suspect(self):
        report = doctor.run(Config(), probes=[
            ("reddit", lambda: (False, "HTTP 403 blocked by network security")),
            ("candles", lambda: (True, "9 bars"))])
        text = doctor.format_report(report)
        self.assertIn("FAIL", text)
        self.assertIn("reddit", text)
        self.assertIn("1/2 answered", text)
        self.assertIn("not a bug in the parser", text)

    def test_an_all_ok_report_prints_no_excuses(self):
        report = doctor.run(Config(), probes=[("candles", lambda: (True, "9 bars"))])
        text = doctor.format_report(report)
        self.assertIn("ok", text)
        self.assertNotIn("not a bug in the parser", text)


class TestRedditProbeKeepsTheBackoff(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_entries = research_mod._reddit_entries
        research_mod.reset_reddit_cooldown()

    def tearDown(self) -> None:
        research_mod._reddit_entries = self._saved_entries
        research_mod.reset_reddit_cooldown()

    def test_a_failed_probe_leaves_the_runtime_cooldown_alone(self):
        research_mod._REDDIT_DOWN_UNTIL = (research_mod.utcnow()
                                           + timedelta(minutes=30))
        saved = research_mod.reddit_cooldown_until()
        research_mod._reddit_entries = lambda sub: ([], "HTTP 429 blocked")
        result = doctor._run("reddit", doctor.probe_reddit("SOL-USD"))
        self.assertFalse(result.ok)
        self.assertIn("429", result.detail)
        self.assertEqual(research_mod.reddit_cooldown_until(), saved,
                         "the diagnostic disturbed the runtime's backoff")

    def test_a_successful_probe_clears_the_cooldown(self):
        research_mod._REDDIT_DOWN_UNTIL = (research_mod.utcnow()
                                           + timedelta(minutes=30))
        research_mod._reddit_entries = lambda sub: ([{"title": "gm"}], "")
        result = doctor._run("reddit", doctor.probe_reddit("SOL-USD"))
        self.assertTrue(result.ok)
        self.assertIn("1 posts", result.detail)
        self.assertIsNone(research_mod.reddit_cooldown_until())


if __name__ == "__main__":
    unittest.main()
