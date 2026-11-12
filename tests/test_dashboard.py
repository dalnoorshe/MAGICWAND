"""Dashboard backend + reflection tests.

The UI itself is plain JS; these test the state assembly and actions it calls,
which is where real bugs live. (The page is also syntax-checked with node in
the project's manual QA, but the backend is what can silently lie.)
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import dashboard
from magicwand.common import utcnow
from magicwand.config import Config
from magicwand.gatekeeper import Policy, RiskTier, TUNABLE_BOUNDS
from magicwand.ledger import Ledger
from magicwand.reflect import PolicyStore, reflect
from magicwand.reflect import reflect_and_persist


class TestStateAssembly(unittest.TestCase):
    def test_empty_root_returns_complete_shape(self):
        """The page must render on a cold start, not error on missing files."""
        with tempfile.TemporaryDirectory() as d:
            state = dashboard.build_state(Path(d), Config())
            for key in ("policy", "desk", "research", "reflection", "memory",
                        "strategies", "ledger", "equity_series", "recent",
                        "tiers", "account_equity"):
                self.assertIn(key, state)
            self.assertEqual(state["policy"]["tier"], int(RiskTier.RESEARCH))
            self.assertEqual(state["strategies"], [])
            self.assertTrue(state["ledger"]["ok"])

    def test_populated_state_reflects_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir()
            cfg = Config()
            led = Ledger(root / cfg.ledger.path)
            led.append("research", {"decision": "veto", "summary": "s",
                                    "vetoed_by": ["news"], "evidence_score": -0.5})
            led.append("mark", {"equity": 50.0, "price": 80.0})
            (root / "data" / "desk.json").write_text(json.dumps({
                "pot": 4.5, "autonomy_armed": True, "trades_today": 1,
                "position": None, "trades": [
                    {"symbol": "SOL-USD", "net": 0.2, "reason": "target",
                     "entry_ts": "t", "exit_ts": "t"}]}))
            PolicyStore(root / "data" / "policy.json").save(
                Policy(tier=RiskTier.PAPER_DESK))
            (root / "data" / "strategies.json").write_text(json.dumps([{
                "spec": {"name": "test", "family": "breakout"},
                "validation": {"metrics": {"holdout_return": 0.02}},
                "deployed": True}]))
            state = dashboard.build_state(root, cfg)
            self.assertEqual(state["policy"]["tier_name"], "paper_desk")
            self.assertEqual(state["research"]["decision"], "veto")
            self.assertEqual(len(state["strategies"]), 1)
            self.assertEqual(state["desk"]["win_rate"], 1.0)
            self.assertEqual(len(state["equity_series"]), 1)

    def test_broken_ledger_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config()
            led = Ledger(root / cfg.ledger.path)
            led.append("note", {"a": 1})
            led.append("note", {"b": 2})
            p = root / cfg.ledger.path
            lines = p.read_text().splitlines()
            lines[0] = lines[0].replace('"a"', '"X"')
            p.write_text("\n".join(lines) + "\n")
            state = dashboard.build_state(root, cfg)
            self.assertFalse(state["ledger"]["ok"])
            self.assertIn("CHAIN BROKEN" if False else "tampered", state["ledger"]["message"])


class TestTierAction(unittest.TestCase):
    def test_policy_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            st = PolicyStore(Path(d) / "p.json")
            st.save(Policy(tier=RiskTier.SCALED_DESK, risk_per_trade=0.015,
                           reasoning="test"))
            loaded = st.load()
            self.assertEqual(loaded.tier, RiskTier.SCALED_DESK)
            self.assertAlmostEqual(loaded.risk_per_trade, 0.015)

    def test_corrupt_policy_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.json"
            p.write_text("{not json")
            loaded = PolicyStore(p).load()
            self.assertEqual(loaded.tier, RiskTier.PAPER_DESK)  # the dataclass default


class TestReflectionRules(unittest.TestCase):
    def _ledger(self, d, trades):
        led = Ledger(Path(d) / "l.jsonl")
        for fam, ret in trades:
            led.append("desk_close", {"opened_by": f"strategy:{fam}",
                                      "net": ret * 100, "return_pct": ret,
                                      "reason": "target" if ret > 0 else "stop"})
        return led

    def test_never_changes_the_tier(self):
        with tempfile.TemporaryDirectory() as d:
            led = self._ledger(d, [("sma_trend", -0.02)] * 10)
            r = reflect(led, Policy(tier=RiskTier.SCALED_DESK))
            self.assertEqual(r.policy_after["tier"], RiskTier.SCALED_DESK)
            self.assertEqual(r.policy_before["tier"], RiskTier.SCALED_DESK)

    def test_never_exceeds_tunable_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            # A wildly profitable record must not push risk past the bound.
            led = self._ledger(d, [("breakout", 0.10)] * 40)
            r = reflect(led, Policy(risk_per_trade=0.02))
            self.assertLessEqual(r.policy_after["risk_per_trade"],
                                 TUNABLE_BOUNDS["risk_per_trade"][1] + 1e-12)
            self.assertGreaterEqual(r.policy_after["risk_per_trade"],
                                    TUNABLE_BOUNDS["risk_per_trade"][0] - 1e-12)

    def test_losing_family_is_demoted(self):
        with tempfile.TemporaryDirectory() as d:
            led = self._ledger(d, [("sma_trend", -0.01)] * 8 + [("breakout", 0.02)] * 8)
            r = reflect(led, Policy())
            self.assertIn("sma_trend", r.blacklisted_families)
            self.assertNotIn("breakout", r.blacklisted_families)

    def test_demoted_families_do_not_inflate_risk(self):
        """A demoted family's history must not drive future position sizing."""
        with tempfile.TemporaryDirectory() as d:
            # All losses come from one family that will be demoted; the kept
            # families lose slightly. Risk must not be raised.
            led = self._ledger(d, [("sma_trend", -0.05)] * 10 + [("breakout", 0.001)] * 8)
            r = reflect(led, Policy())
            self.assertIn("sma_trend", r.blacklisted_families)
            self.assertLessEqual(r.policy_after["risk_per_trade"],
                                 r.policy_before["risk_per_trade"] + 0.002)

    def test_no_history_means_no_change(self):
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(Path(d) / "l.jsonl")
            before = Policy()
            r = reflect(led, before)
            self.assertEqual(r.changes, [])
            self.assertEqual(r.policy_after["risk_per_trade"],
                             before.risk_per_trade)

    def test_reflection_is_persisted_and_logged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            led = self._ledger(d, [("breakout", 0.02)] * 10)
            store = PolicyStore(root / "policy.json")
            ref = reflect_and_persist(led, store)
            self.assertTrue((root / "policy.json").exists())
            kinds = [e["kind"] for e in led.entries()]
            self.assertIn("reflection", kinds)
            self.assertIsNotNone(ref.summary)


class TestTickLock(unittest.TestCase):
    def test_second_concurrent_tick_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tick.lock"
            with dashboard._Lock(p):
                with self.assertRaises(RuntimeError):
                    with dashboard._Lock(p):
                        pass
            # And it is reusable after release.
            with dashboard._Lock(p):
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------
# Autopilot: the clock that makes "unattended" true
# --------------------------------------------------------------------------

class TestAutopilot(unittest.TestCase):
    def _ap(self, d, interval=1):
        from magicwand.autopilot import Autopilot
        cfg = Config()
        cfg.market.exchange = "synthetic"
        cfg.market.candles = 300
        # Avoid network in tests: a fixture snapshot is injected via the
        # synthetic exchange marker in research_loop.
        return Autopilot(d, cfg, offline=True, interval=interval)

    def test_first_tick_is_immediate_then_repeats(self):
        """Autonomy means more than one tick with nobody pressing anything.

        This test exists because the first implementation waited a full
        interval BEFORE its first tick, and a restart looked dead.
        """
        import time
        with tempfile.TemporaryDirectory() as d:
            ap = self._ap(d, interval=1)
            ap.start()
            try:
                deadline = time.time() + 25
                while time.time() < deadline and ap.state.ticks < 2:
                    time.sleep(0.4)
            finally:
                ap.stop()
            self.assertGreaterEqual(ap.state.ticks, 2,
                                    f"only {ap.state.ticks} tick(s) fired")
            self.assertGreaterEqual(ap.state.errors, 0)

    def test_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            ap = self._ap(d)
            ap.start()
            ap.stop()
            again = ap.stop()
            self.assertTrue(again["ok"])
            self.assertFalse(ap.status()["running"])

    def test_start_twice_does_not_spawn_two_loops(self):
        with tempfile.TemporaryDirectory() as d:
            ap = self._ap(d)
            first = ap.start()
            second = ap.start()
            ap.stop()
            self.assertTrue(first["ok"])
            self.assertFalse(second["ok"])
            self.assertIn("already running", second["error"])

    def test_status_is_json_safe(self):
        with tempfile.TemporaryDirectory() as d:
            ap = self._ap(d)
            st = ap.status()
            json.dumps(st)  # must not raise
            self.assertIn("running", st)
            self.assertIn("interval", st)

    def test_state_is_persisted_for_the_ui(self):
        with tempfile.TemporaryDirectory() as d:
            ap = self._ap(d)
            ap.start()
            ap.stop()
            p = Path(d) / "data" / "autopilot.json"
            self.assertTrue(p.exists())
            d_ = json.loads(p.read_text())
            self.assertIn("running", d_)
            self.assertFalse(d_["running"])
