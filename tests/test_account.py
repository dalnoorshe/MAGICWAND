"""Account resolution: the balance must be real, sourced, and never guessed."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand.account import AccountError, resolve_account
from magicwand.config import Config


class GoodBroker:
    connected = True
    name = "test-broker"
    def __init__(self, eq=812.34):
        self.eq = eq
    def fetch_equity(self):
        return self.eq


class BrokenBroker(GoodBroker):
    def fetch_equity(self):
        raise TimeoutError("venue timed out")


class Disconnected(GoodBroker):
    connected = False


class TestResolution(unittest.TestCase):
    def test_broker_wins_over_paper_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data").mkdir()
            (Path(d) / "data" / "state.json").write_text('{"equity": 50}')
            snap = resolve_account(d, Config(), broker=GoodBroker(812.34))
            self.assertEqual(snap.source, "broker")
            self.assertAlmostEqual(snap.equity, 812.34)
            self.assertFalse(snap.stale)

    def test_paper_file_used_without_a_broker(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data").mkdir()
            (Path(d) / "data" / "state.json").write_text('{"equity": 123.45}')
            snap = resolve_account(d, Config())
            self.assertEqual(snap.source, "paper")
            self.assertAlmostEqual(snap.equity, 123.45)

    def test_config_is_the_last_resort(self):
        with tempfile.TemporaryDirectory() as d:
            snap = resolve_account(d, Config())
            self.assertEqual(snap.source, "config")
            self.assertAlmostEqual(snap.equity, Config().risk.equity)

    def test_corrupt_state_falls_back_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data").mkdir()
            (Path(d) / "data" / "state.json").write_text("{broken")
            snap = resolve_account(d, Config())
            self.assertEqual(snap.source, "config")
            self.assertTrue(snap.stale, "a fallback balance must be flagged")
            self.assertIn("unreadable", snap.detail)

    def test_nonsense_equity_rejected(self):
        for bad in ('{"equity": 0}', '{"equity": -5}', '{"equity": "abc"}',
                    '{"equity": null}'):
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "data").mkdir()
                (Path(d) / "data" / "state.json").write_text(bad)
                snap = resolve_account(d, Config())
                self.assertEqual(snap.source, "config", bad)
                self.assertGreater(snap.equity, 0)

    def test_connected_broker_without_balance_raises(self):
        """Guessing while connected to a real account is the one unacceptable case."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AccountError) as ctx:
                resolve_account(d, Config(), broker=BrokenBroker())
            self.assertIn("refusing to size", str(ctx.exception))

    def test_negative_broker_balance_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AccountError):
                resolve_account(d, Config(), broker=GoodBroker(-10.0))

    def test_disconnected_broker_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data").mkdir()
            (Path(d) / "data" / "state.json").write_text('{"equity": 77}')
            snap = resolve_account(d, Config(), broker=Disconnected(999))
            self.assertEqual(snap.source, "paper")
            self.assertAlmostEqual(snap.equity, 77)

    def test_snapshot_is_serialisable(self):
        with tempfile.TemporaryDirectory() as d:
            json.dumps(resolve_account(d, Config()).to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
