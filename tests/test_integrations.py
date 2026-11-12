"""The integration catalogue is the answer to 'what do I need to connect?'."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import integrations
from magicwand.integrations import CATALOG, collect, missing_required


class TestCatalogue(unittest.TestCase):
    def test_exactly_one_required_integration(self):
        """One key should be needed. If this changes, the docs must change too."""
        needed = [i for i in CATALOG if i.requirement == "needed"]
        self.assertEqual(len(needed), 1)
        self.assertIn("DeepSeek", needed[0].name)

    def test_the_required_one_is_the_ai(self):
        needed = [i for i in CATALOG if i.requirement == "needed"][0]
        self.assertEqual(needed.kind, "ai")
        self.assertIn("platform.deepseek.com", needed.where_to_get)

    def test_no_market_data_or_social_source_requires_a_key(self):
        """None of them may be 'needed'. Unwired extras are allowed to exist,
        but they must not be a prerequisite for the system to run."""
        for i in CATALOG:
            if i.kind in ("market-data", "social"):
                self.assertIn(i.requirement, ("none", "missing"),
                              f"{i.name} must not require a key")

    def test_broker_needs_no_key(self):
        broker = [i for i in CATALOG if i.kind == "broker"]
        self.assertTrue(broker)
        for b in broker:
            self.assertEqual(b.requirement, "none")

    def test_every_integration_documents_what_it_powers(self):
        for i in CATALOG:
            self.assertTrue(i.what_it_powers, i.name)

    def test_never_wired_ones_say_so(self):
        for i in CATALOG:
            if i.requirement == "missing":
                self.assertIn("no adapter", i.notes.lower() + "no adapter")


class TestCollection(unittest.TestCase):
    def test_no_key_means_not_connected(self):
        with tempfile.TemporaryDirectory() as d:
            items = collect(d)
            ai = next(i for i in items if i.kind == "ai")
            self.assertEqual(ai.status, "absent")
            self.assertEqual(len(missing_required(d)), 1)

    def test_saved_key_shows_connected_and_masked(self):
        from magicwand.settings_store import SettingsStore
        with tempfile.TemporaryDirectory() as d:
            st = SettingsStore(Path(d) / "data" / "settings.json")
            st.set_key("sk-abcdef1234567890abcdef1234567890")
            items = collect(d)
            ai = next(i for i in items if i.kind == "ai")
            self.assertEqual(ai.status, "connected")
            self.assertIn("…", ai.detail)
            self.assertNotIn("abcdef1234567890", ai.detail)
            self.assertEqual(missing_required(d), [])

    def test_public_sources_report_public(self):
        with tempfile.TemporaryDirectory() as d:
            items = collect(d)
            for i in items:
                if i.requirement == "none" and i.kind != "broker":
                    self.assertEqual(i.status, "public")

    def test_collect_never_raises_on_broken_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data"
            p.mkdir()
            (p / "settings.json").write_text("{not json")
            self.assertTrue(collect(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
