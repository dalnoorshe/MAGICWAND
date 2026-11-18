"""The console must be able to tell "just finished" from "running now".

The research panel showed a full set of verdicts the instant the page loaded,
because it replays the last cycle out of the ledger. Nothing was faked, but the
freshness was never stated — so a four-hour-old reading and a one-second-old
reading looked identical, and the operator had no way to trust either.

These tests pin the two halves of the fix:

  * a running cycle leaves a marker on disk, cleared in a `finally`;
  * reading it fails soft in every direction, because a corrupt progress file
    must never be able to take the console down or claim work that stopped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from magicwand import research_loop
from magicwand.common import iso, utcnow
from magicwand.config import Config
from magicwand.dashboard import build_state


class TestInflightMarker(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_mark_then_read_then_clear(self):
        now = utcnow()
        research_loop.mark_inflight(self.root, symbol="SOL-USD",
                                    agents=["news", "charts"], now=now)
        got = research_loop.read_inflight(self.root, now=now)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["symbol"], "SOL-USD")
        self.assertEqual(got["agents"], ["news", "charts"])
        self.assertLess(got["age_seconds"], 1.0)
        research_loop.clear_inflight(self.root)
        self.assertIsNone(research_loop.read_inflight(self.root, now=now))

    def test_absent_marker_is_not_running(self):
        self.assertIsNone(research_loop.read_inflight(self.root))

    def test_corrupt_marker_does_not_raise(self):
        p = research_loop.inflight_path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json at all")
        self.assertIsNone(research_loop.read_inflight(self.root))

    def test_non_dict_payload_does_not_raise(self):
        p = research_loop.inflight_path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[1, 2, 3]")
        self.assertIsNone(research_loop.read_inflight(self.root))

    def test_missing_started_at_does_not_raise(self):
        p = research_loop.inflight_path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"symbol": "SOL-USD"}))
        self.assertIsNone(research_loop.read_inflight(self.root))

    def test_dead_marker_older_than_the_cap_is_not_running(self):
        """A crashed cycle must not be reported as work in progress."""
        now = utcnow()
        p = research_loop.inflight_path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        stale = now - timedelta(seconds=research_loop.INFLIGHT_MAX_AGE + 60)
        p.write_text(json.dumps({"started_at": iso(stale), "symbol": "SOL-USD"}))
        self.assertIsNone(research_loop.read_inflight(self.root, now=now))
        # ...but it is readable while it is still within the cap.
        fresh = now - timedelta(seconds=5)
        p.write_text(json.dumps({"started_at": iso(fresh), "symbol": "SOL-USD"}))
        self.assertIsNotNone(research_loop.read_inflight(self.root, now=now))

    def test_future_marker_is_treated_as_not_running(self):
        """A clock jump must not produce a negative age that reads as fresh."""
        now = utcnow()
        p = research_loop.inflight_path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"started_at": iso(now + timedelta(hours=2))}))
        self.assertIsNone(research_loop.read_inflight(self.root, now=now))

    def test_marker_write_never_raises_on_an_unwritable_root(self):
        """Progress reporting is never allowed to break the research."""
        blocker = self.root / "data"
        blocker.write_text("I am a file, not a directory")
        research_loop.mark_inflight(self.root, symbol="SOL-USD")
        self.assertIsNone(research_loop.read_inflight(self.root))
        research_loop.clear_inflight(self.root)


class TestStateExposesInflight(unittest.TestCase):
    def test_build_state_reports_a_running_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            research_loop.mark_inflight(root, symbol="SOL-USD")
            state = build_state(root, Config())
            self.assertIn("research_inflight", state)
            self.assertIsNotNone(state["research_inflight"])
            assert state["research_inflight"] is not None
            self.assertEqual(state["research_inflight"]["symbol"], "SOL-USD")

    def test_build_state_survives_a_corrupt_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = research_loop.inflight_path(root)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("}{")
            state = build_state(root, Config())  # must not raise
            self.assertIsNone(state["research_inflight"])


if __name__ == "__main__":
    unittest.main()


class TestCycleReportsProgress(unittest.TestCase):
    """A cycle must describe itself while it runs, not only when it lands."""

    def _run(self, agents_list):
        from magicwand import research as research_mod
        from magicwand.llm import DeepSeekClient, fake_canned

        seen: list[dict] = []
        real = research_loop.update_inflight

        def spy(root, **fields):
            seen.append(dict(fields))
            return real(root, **fields)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config()
            snap = research_mod.fixture_snapshot("SOL-USD", price=71.0)
            research_loop.update_inflight = spy
            try:
                cycle = research_loop.run_cycle(
                    cfg, DeepSeekClient(fake=fake_canned), price=71.0, atr=1.2,
                    root=root, symbol="SOL-USD", snapshot=snap, vision=False,
                    agents=agents_list)
            finally:
                research_loop.update_inflight = real
            leftover = research_loop.read_inflight(root)
        return seen, cycle, leftover

    def test_roster_is_published_before_any_reader_finishes(self):
        seen, _cycle, leftover = self._run(["news", "social"])
        self.assertTrue(seen, "no progress was reported at all")
        first = seen[0]
        self.assertEqual(first.get("plan"), ["news", "social"])
        self.assertIn(first.get("phase"), ("snapshot", "reading"))
        self.assertEqual(first.get("done"), [])
        self.assertIsNone(leftover, "the marker survived the cycle")

    def test_each_reader_is_announced_then_recorded(self):
        seen, cycle, _ = self._run(["news", "social"])
        currents = [e["current"] for e in seen if e.get("current")]
        self.assertIn("news", currents)
        self.assertIn("social", currents)
        # The last event that reports verdicts carries everyone who came back,
        # in the order they were read.
        reported = [e["done"] for e in seen if e.get("done")]
        self.assertTrue(reported, "no verdict was ever reported as finished")
        self.assertEqual([d["agent"] for d in reported[-1]],
                         [v.agent for v in cycle.report.agents])
        self.assertTrue(all(d["agent"] in {"news", "social"}
                            for d in reported[-1]))

    def test_a_finished_cycle_is_reported_as_synthesising(self):
        seen, _cycle, _ = self._run(["news", "social"])
        self.assertIn("synthesis", [e.get("phase") for e in seen])

    def test_progress_carries_no_promises_about_unfinished_readers(self):
        """A reader is listed only after it has a verdict."""
        seen, _cycle, _ = self._run(["news", "social"])
        for event in seen:
            done = event.get("done") or []
            self.assertTrue(all(d.get("agent") and "stance" in d for d in done))
