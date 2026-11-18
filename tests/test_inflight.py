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
