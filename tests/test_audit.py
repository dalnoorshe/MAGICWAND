"""The self-audit must keep finding the things it found once.

Each scenario that was a real gap now has a regression test here, so a future
change that removes the guard fails the suite instead of quietly regressing.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import audit, data as dm
from magicwand.common import Bar, Side, Position, iso, utcnow
from magicwand.config import Config
from magicwand.desk import AgentDesk, DeskConfig
from magicwand.gatekeeper import RiskTier


class TestAuditRuns(unittest.TestCase):
    def test_no_gaps_remain(self):
        results = audit.run_all()
        gaps = [r for r in results if r.verdict == "GAP"]
        self.assertEqual(gaps, [], f"regressions: {[g.name for g in gaps]}")

    def test_every_scenario_returns_an_outcome(self):
        for r in audit.run_all():
            self.assertIn(r.verdict, ("PASS", "GAP", "INFO"))
            self.assertTrue(r.detail)


class TestStaleDataGuard(unittest.TestCase):
    def test_recent_bars_are_not_stale(self):
        bars = [Bar(ts=utcnow() - timedelta(minutes=30), open=1, high=2,
                    low=0.5, close=1.5, volume=1)]
        self.assertFalse(dm.is_stale(bars, "1h", 3.0))

    def test_old_bars_are_stale(self):
        bars = [Bar(ts=utcnow() - timedelta(hours=10), open=1, high=2,
                    low=0.5, close=1.5, volume=1)]
        self.assertTrue(dm.is_stale(bars, "1h", 3.0))

    def test_empty_bars_are_stale(self):
        self.assertTrue(dm.is_stale([], "1h", 3.0))

    def test_synthetic_is_exempt_in_the_runner(self):
        """Synthetic fixtures carry fixed 2024 timestamps; they must not trip
        the staleness guard and break every test."""
        import inspect
        from magicwand import runner
        src = inspect.getsource(runner)
        self.assertIn('!= "synthetic"', src)


class TestBarsNotTicks(unittest.TestCase):
    """max_hold_bars is in bars. A 5-minute tick on a 1h chart is not one bar."""

    def test_same_bar_counts_once(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            desk = AgentDesk(cfg, DeskConfig(enabled=True,
                            tier=RiskTier.PAPER_DESK), root=d)
            desk.fund(5000.0)
            out = desk.open_trade("SOL-USD", 80.0, 2.0, "charts", "x", 0.02, 0.9)
            self.assertTrue(out["ok"], out)
            ts = utcnow()
            counted = [desk.advance_bars(ts) for _ in range(12)]
            self.assertEqual(counted.count(True), 1,
                             "twelve ticks in one bar must count one bar")
            self.assertEqual(desk.state.position.bars_held, 1)

    def test_new_bar_counts_again(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            desk = AgentDesk(cfg, DeskConfig(enabled=True,
                            tier=RiskTier.PAPER_DESK), root=d)
            desk.fund(5000.0)
            desk.open_trade("SOL-USD", 80.0, 2.0, "charts", "x", 0.02, 0.9)
            t0 = utcnow()
            desk.advance_bars(t0)
            desk.advance_bars(t0 + timedelta(hours=1))
            self.assertEqual(desk.state.position.bars_held, 2)


class TestAffordability(unittest.TestCase):
    def test_small_account_is_warned(self):
        from magicwand.dashboard import _affordability_warning
        cfg = Config()
        msg = _affordability_warning(Path("."), cfg, "SOL-USD")
        self.assertIn("NO market", msg)

    def test_large_account_is_not_warned(self):
        from magicwand.dashboard import _affordability_warning
        cfg = Config()
        cfg.risk.equity = 5000.0
        self.assertEqual(_affordability_warning(Path("."), cfg, "SOL-USD"), "")


class TestTickCost(unittest.TestCase):
    def test_cost_scales_with_interval(self):
        from magicwand.autopilot import Autopilot
        from magicwand.config import Config
        with tempfile.TemporaryDirectory() as d:
            fast = Autopilot(d, Config(), interval=60)
            slow = Autopilot(d, Config(), interval=900)
            self.assertAlmostEqual(fast.estimated_daily_cost(),
                                   slow.estimated_daily_cost() * 15, places=6)

    def test_status_exposes_the_estimate(self):
        from magicwand.autopilot import Autopilot
        from magicwand.config import Config
        with tempfile.TemporaryDirectory() as d:
            st = Autopilot(d, Config(), interval=300).status()
            self.assertIn("estimated_daily_cost_usd", st)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRoutineIntegration(unittest.TestCase):
    """`wand audit` is part of the routine: it must run at startup, save its
    result, and be visible in the console."""

    def test_report_is_saved_and_readable(self):
        from magicwand.audit import load_report, run_all, save_report
        with tempfile.TemporaryDirectory() as d:
            results = run_all(angle="durability")
            body = save_report(d, results, angle="durability")
            self.assertIn("ts", body)
            self.assertEqual(body["gaps"], 0)
            self.assertGreater(body["passes"], 0)
            back = load_report(d)
            self.assertEqual(back["gaps"], 0)
            self.assertEqual(back["angle"], "durability")

    def test_missing_report_is_empty_not_fatal(self):
        from magicwand.audit import load_report
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_report(d), {})

    def test_corrupt_report_is_ignored(self):
        from magicwand.audit import load_report, report_path
        with tempfile.TemporaryDirectory() as d:
            p = report_path(d)
            p.parent.mkdir(parents=True)
            p.write_text("{not json")
            self.assertEqual(load_report(d), {})

    def test_launcher_runs_the_audit_and_can_skip_it(self):
        import inspect
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mw_launcher4",
            str(Path(__file__).resolve().parents[1] / "magicwand-launcher.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = inspect.getsource(mod)
        self.assertIn("_audit.run_all", src)
        self.assertIn("--skip-audit", src)

    def test_console_shows_the_audit(self):
        html = (Path(__file__).resolve().parents[1]
                / "magicwand" / "dashboard.html").read_text()
        self.assertIn('id="auditbadge"', html)
        self.assertIn("renderAudit", html)
        self.assertIn('id="auditLine"', html)

    def test_state_exposes_the_report(self):
        import inspect
        from magicwand import dashboard
        self.assertIn('"audit"', inspect.getsource(dashboard.build_state))


class TestMemoryFile(unittest.TestCase):
    """The lessons must survive a context reset: AGENTS.md is the memory."""

    def _memory(self) -> str:
        return (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text()

    def test_memory_exists_and_names_the_audit(self):
        text = self._memory()
        self.assertIn("wand.py audit", text)
        self.assertIn("routine", text.lower())

    def test_memory_records_the_hard_won_invariants(self):
        text = self._memory().lower()
        for phrase in ("fail closed", "never guess a balance", "void only",
                       "bars are not ticks", "non-finite", "no lookahead"):
            self.assertIn(phrase.replace("void only", "veto only"), text, phrase)

    def test_memory_forbids_destructive_data_commands(self):
        self.assertIn("Never run `rm` against `data/`", self._memory())

    def test_memory_states_the_honest_blockers(self):
        text = self._memory()
        self.assertIn("approved **0** strategies", text)
        self.assertIn("$50", text)
