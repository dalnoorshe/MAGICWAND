"""The paper journal has one job: make a 30-day run falsifiable.

Three claims it must support, and the failure it must not allow:

  * "it refused to trade" — only sayable when refusals are actually recorded;
  * "it never signalled" — different from refusing, and must not be conflated;
  * "nothing ran" — an outage. It must NEVER look like a quiet, disciplined
    market. A run that was off for a week and reports six calm days is lying by
    omission, which is exactly the class of thing this project keeps fixing.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from magicwand import journal
from magicwand.common import iso, utcnow
from magicwand.config import Config
from magicwand.ledger import Ledger


def _ledger(events) -> Path:
    """events: (kind, payload, when) in a temp root. Returns the root."""
    root = Path(tempfile.mkdtemp())
    cfg = Config()
    led = Ledger(root / cfg.ledger.path)
    for kind, payload, when in events:
        led.append(kind, payload, when=when)
    return root


class TestJournal(unittest.TestCase):
    def setUp(self) -> None:
        self.now = utcnow()

    def _when(self, days_ago: int, hour: int = 12):
        return (self.now - timedelta(days=days_ago)).replace(hour=hour,
                                                            minute=0, second=0,
                                                            microsecond=0)

    def test_a_refusal_day_is_distinguishable_from_silence(self):
        root = _ledger([
            ("mark", {"equity": 500.0}, self._when(2)),
            ("veto", {"reason": "trade too small: 4.20 < exchange minimum 5.00"},
             self._when(2, 13)),
            ("mark", {"equity": 500.0}, self._when(1)),
            ("research", {"cost_usd": 0.004, "agents": []}, self._when(1)),
        ])
        j = journal.build_journal(root, Config(), days=3, now=self.now)
        by_day = {d["day"]: d for d in j["days"]}
        refused = [d for d in j["days"] if d["vetoes"]]
        silent = [d for d in j["days"] if d["trades"] == 0 and d["vetoes"] == 0
                  and (d["ticks"] or d["research"])]
        self.assertTrue(refused, "the refusal day vanished")
        self.assertTrue(silent, "the quiet day vanished")
        self.assertTrue(refused[0]["refusals"])
        self.assertIn("refusal", refused[0]["note"])
        self.assertIn("no entry signal", silent[0]["note"])
        self.assertTrue(by_day)

    def test_a_day_with_no_entries_is_reported_as_an_outage(self):
        """The whole point: a gap must not read as a disciplined quiet day."""
        root = _ledger([
            ("mark", {"equity": 500.0}, self._when(4)),
            ("mark", {"equity": 500.0}, self._now_minus_hours()),
        ])
        j = journal.build_journal(root, Config(), days=5, now=self.now)
        days = {d["day"]: d for d in j["days"]}
        gap_days = [d for d in j["days"] if d["ticks"] == 0 and d["research"] == 0]
        self.assertTrue(gap_days, "the empty days were not listed at all")
        for d in gap_days:
            self.assertIn("NOTHING RAN", d["note"])
        self.assertTrue(j["outage_days"])
        self.assertIn("outage", j["verdict"])

    def _now_minus_hours(self):
        return self.now - timedelta(hours=1)

    def test_a_trade_is_counted_with_its_net(self):
        root = _ledger([
            ("fill", {"net": 1.5, "symbol": "SOL-USD"}, self._when(1)),
            ("mark", {"equity": 501.5}, self._when(1, 13)),
        ])
        j = journal.build_journal(root, Config(), days=2, now=self.now)
        self.assertEqual(j["trades"], 1)
        self.assertAlmostEqual(j["trade_net"], 1.5, places=6)
        self.assertIn("trade(s)", j["verdict"])

    def test_equity_change_comes_from_marks_not_from_hope(self):
        root = _ledger([
            ("mark", {"equity": 500.0}, self._when(2)),
            ("mark", {"equity": 512.5}, self._when(1)),
        ])
        j = journal.build_journal(root, Config(), days=3, now=self.now)
        self.assertAlmostEqual(j["equity_change"], 12.5, places=6)
        self.assertAlmostEqual(j["equity_change_pct"], 2.5, places=6)

    def test_research_cost_is_summed(self):
        root = _ledger([
            ("research", {"cost_usd": 0.004}, self._when(2)),
            ("research", {"cost_usd": 0.006}, self._when(1)),
        ])
        j = journal.build_journal(root, Config(), days=3, now=self.now)
        self.assertAlmostEqual(j["cost_usd"], 0.01, places=6)
        self.assertEqual(j["research_cycles"], 2)

    def test_an_empty_ledger_produces_a_report_not_a_crash(self):
        root = Path(tempfile.mkdtemp())
        j = journal.build_journal(root, Config(), days=30, now=self.now)
        self.assertEqual(j["trades"], 0)
        self.assertIsNone(j["equity_first"])
        text = journal.format_journal(j)
        self.assertIn("no marks recorded", text)

    def test_nothing_refused_and_nothing_signalled_says_so(self):
        root = _ledger([("research", {"cost_usd": 0.0}, self._when(1))])
        j = journal.build_journal(root, Config(), days=2, now=self.now)
        self.assertIn("no signal", j["verdict"])
        self.assertFalse(j["refused_days"])
