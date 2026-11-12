"""Tests for self-invented strategies, the validation gate, and the runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import data as data_mod
from magicwand import runner
from magicwand.common import Bar, utcnow
from magicwand.config import Config
from magicwand.desk import DeskConfig
from magicwand.gatekeeper import RiskTier
from magicwand.ledger import Ledger
from magicwand.llm import DeepSeekClient
from magicwand.propose import (MIN_BARS, StrategyRegistry, StrategySpec,
                               catalog_text, parse_proposal, proposals_today,
                               validate)
from magicwand.strategy import CATALOG, build_parameterised


def spec(family="sma_trend", params=None, symbol="SOL-USD") -> StrategySpec:
    return StrategySpec(name="test_idea", hypothesis="test", family=family,
                        params=params or {"slow": 24, "fast": 8},
                        symbol=symbol, ts=utcnow().isoformat())


class TestCatalogSafety(unittest.TestCase):
    def test_only_catalog_families_exist(self):
        self.assertIn("sma_trend", CATALOG)
        self.assertIn("mean_revert", CATALOG)
        self.assertIn("breakout", CATALOG)

    def test_code_injection_is_impossible(self):
        """There is no way to submit code: only family + typed numbers."""
        with self.assertRaises(ValueError):
            build_parameterised("sma_trend", {"slow": "__import__('os').system('id')"})
        with self.assertRaises(ValueError):
            build_parameterised("os_system", {"slow": 20})

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            build_parameterised("sma_trend", {"stop_atr": 500.0})
        with self.assertRaises(ValueError):
            build_parameterised("sma_trend", {"max_hold_bars": -5})

    def test_unknown_param_rejected(self):
        with self.assertRaises(ValueError):
            build_parameterised("sma_trend", {"leverage": 100})

    def test_parse_proposal_rejects_bad_family(self):
        with self.assertRaises(ValueError):
            parse_proposal(json.dumps({"name": "x", "family": "magic",
                                       "params": {}, "hypothesis": "h"}))
        with self.assertRaises(ValueError):
            parse_proposal("no json here at all")


class TestValidationGate(unittest.TestCase):
    def test_insufficient_data_rejected(self):
        cfg = Config()
        bars = data_mod.synthetic_bars(120, seed=1)
        r = validate(spec(), bars, cfg)
        self.assertFalse(r.passed)
        self.assertFalse(r.gates["data"][0])
        self.assertIn("insufficient data", " ".join(r.reasons))

    def test_random_walk_strategy_usually_rejected(self):
        """On a fair coin, most invented strategies must fail the gate.

        This is the anti-overfitting property: the gate must not be a
        formality. If this ever fails, the gate has stopped working.
        """
        cfg = Config()
        passed = 0
        tried = 0
        for seed in range(2):
            bars = data_mod.synthetic_bars(420, seed=seed, drift=0.0)
            for fam, params in (("sma_trend", {"slow": 30, "fast": 10}),
                                ("breakout", {"slow": 25})):
                tried += 1
                if validate(spec(fam, params), bars, cfg).passed:
                    passed += 1
        self.assertLessEqual(passed, 1,
                             f"{passed}/{tried} strategies passed on pure noise")

    def test_all_gates_reported(self):
        cfg = Config()
        bars = data_mod.synthetic_bars(500, seed=3)
        r = validate(spec(), bars, cfg)
        for gate in ("catalog", "data", "profitable", "costs_2x",
                     "beats_hold", "evidence", "out_of_sample", "not_luck"):
            self.assertIn(gate, r.gates, f"gate {gate} was not evaluated")

    def test_rejection_carries_reasons(self):
        cfg = Config()
        bars = data_mod.synthetic_bars(500, seed=5, drift=-0.002)  # downtrend
        r = validate(spec("sma_trend", {"slow": 40, "fast": 10}), bars, cfg)
        self.assertFalse(r.passed)
        self.assertTrue(r.reasons)
        self.assertIn("REJECTED", r.summary)

    def test_registry_refuses_unvalidated(self):
        with tempfile.TemporaryDirectory() as d:
            reg = StrategyRegistry(Path(d) / "s.json")
            cfg = Config()
            bars = data_mod.synthetic_bars(500, seed=9, drift=-0.003)
            r = validate(spec(), bars, cfg)
            if not r.passed:
                with self.assertRaises(ValueError):
                    reg.add(spec(), r)

    def test_registry_roundtrip_of_validated_item(self):
        with tempfile.TemporaryDirectory() as d:
            reg = StrategyRegistry(Path(d) / "s.json")
            cfg = Config()
            bars = data_mod.synthetic_bars(600, seed=2, drift=0.0)
            # Manufacture a passing result to test storage, not the gate.
            from magicwand.propose import ValidationResult
            r = ValidationResult(True, {"catalog": (True, "ok")}, [],
                                 {"holdout_return": 0.01})
            item = reg.add(spec(), r)
            self.assertFalse(item["deployed"])
            self.assertEqual(len(reg.load()), 1)
            self.assertIsNotNone(reg.best())

    def test_proposal_budget_counts_today(self):
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(Path(d) / "l.jsonl")
            led.append("proposal", {"ok": False})
            led.append("proposal", {"ok": False})
            self.assertEqual(proposals_today(led), 2)


class TestProposalParsing(unittest.TestCase):
    def test_valid_proposal(self):
        s = parse_proposal(json.dumps({
            "name": "Funding Squeeze!",
            "hypothesis": "crowded shorts",
            "family": "mean_revert",
            "params": {"slow": 30, "z_entry": 2.0},
            "symbol": "SOL-USD", "timeframe": "1h"}))
        self.assertEqual(s.family, "mean_revert")
        self.assertEqual(s.name, "funding_squeeze")

    def test_json_wrapped_in_prose(self):
        s = parse_proposal('Here you go:\n{"name":"x","family":"breakout",'
                           '"params":{"slow":20},"hypothesis":"h"}')
        self.assertEqual(s.family, "breakout")

    def test_catalog_text_mentions_ranges(self):
        t = catalog_text()
        self.assertIn("sma_trend", t)
        self.assertIn("[5,400]", t)


class TestRunner(unittest.TestCase):
    def _cfg(self, d):
        cfg = Config()
        cfg.market.exchange = "synthetic"
        cfg.market.candles = 400
        (Path(d) / "data").mkdir(parents=True, exist_ok=True)
        return cfg

    def _snapshot(self, ts):
        from magicwand.research import MarketSnapshot, Source
        s = MarketSnapshot(ts=ts, symbol="SOL-USD", price=80.0)
        s.sources["derivatives"] = Source("derivatives", True, ts,
                                          {"funding_rate_pct": 0.01})
        s.sources["news"] = Source("news", True, ts, {"count": 0, "items": []})
        s.sources["social"] = Source("social", False, ts, error="429")
        s.sources["onchain"] = Source("onchain", True, ts, {"mempool_tx": 60000})
        return s

    def test_tick_does_nothing_without_validated_strategy(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            client = DeepSeekClient(fake=lambda s, u: json.dumps({
                "stance": "neutral", "confidence": 0.3, "evidence_score": 0.0,
                "thesis": "quiet", "claims": []}))
            dcfg = DeskConfig(enabled=True, tier=RiskTier.PAPER_DESK)
            rep = runner.tick(cfg, client, dcfg, root=d, propose_enabled=False,
                              snapshot=self._snapshot(utcnow().isoformat()))
            self.assertEqual(rep.strategy_trades, 0)
            self.assertTrue(any("no strategy has passed" in n
                                for n in rep.notes), rep.notes)

    def test_tick_stands_down_on_research_veto(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            # Seed a validated strategy into the registry.
            reg = StrategyRegistry(Path(d) / "data" / "strategies.json")
            from magicwand.propose import ValidationResult
            reg.add(spec(), ValidationResult(True, {"catalog": (True, "ok")}, [],
                                             {"holdout_return": 0.02}))
            veto_client = DeepSeekClient(fake=lambda s, u: json.dumps({
                "stance": "veto", "confidence": 0.95, "evidence_score": -0.9,
                "thesis": "hack", "claims": []}))
            dcfg = DeskConfig(enabled=True, tier=RiskTier.PAPER_DESK)
            rep = runner.tick(cfg, veto_client, dcfg, root=d,
                              propose_enabled=False,
                              snapshot=self._snapshot(utcnow().isoformat()))
            self.assertEqual(rep.research_decision, "veto")
            self.assertTrue(any("veto" in n for n in rep.notes), rep.notes)
            self.assertEqual(rep.strategy_trades, 0)

    def test_empty_data_raises_instead_of_trading_blind(self):
        """A tick with no data must fail loudly, never trade on nothing."""
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            cfg.market.candles = 0
            client = DeepSeekClient(fake=lambda s, u: "{}")
            with self.assertRaises(RuntimeError):
                runner.tick(cfg, client, DeskConfig(enabled=False), root=d,
                            propose_enabled=False,
                            snapshot=self._snapshot(utcnow().isoformat()))

    def test_run_loop_survives_a_failing_tick(self):
        """The unattended loop must not die because one tick failed."""
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            cfg.market.candles = 0          # every tick raises
            client = DeepSeekClient(fake=lambda s, u: "{}")
            # max_ticks=2 with interval 0: it should survive both and return.
            rc = runner.run(cfg, client, DeskConfig(enabled=False), root=d,
                            interval=0, max_ticks=2, propose_enabled=False)
            self.assertEqual(rc, 0)

    def test_no_key_no_proposal(self):
        client = DeepSeekClient(fake=None, api_key="")
        import os
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            self.assertFalse(client.available())
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved


class TestAutonomousTradeBranch(unittest.TestCase):
    """The runner must be ABLE to trade without a human — and must not
    trade anything the gate rejected."""

    def _snapshot(self, ts):
        from magicwand.research import MarketSnapshot, Source
        s = MarketSnapshot(ts=ts, symbol="SOL-USD", price=80.0)
        s.sources["derivatives"] = Source("derivatives", True, ts,
                                          {"funding_rate_pct": 0.01})
        s.sources["news"] = Source("news", True, ts, {"count": 0, "items": []})
        s.sources["social"] = Source("social", True, ts, {"posts_24h": 25})
        s.sources["onchain"] = Source("onchain", True, ts, {"mempool_tx": 60000})
        return s

    def test_trade_branch_opens_a_position_without_human_input(self):
        """Seed an approved strategy and an always-enter signal.

        This proves the wiring: research does not veto, a gate-approved
        strategy says 'enter', and the desk opens a trade with no human and
        no further permission check.
        """
        from magicwand.strategy import Strategy, Intent
        from magicwand.common import Side

        class AlwaysEnter(Strategy):
            name = "always_enter"

            def warmup(self):
                return 1

            def decide(self, bars, i, in_position):
                if in_position:
                    return None
                return Intent("enter", Side.BUY, "test signal", stop_distance=2.0,
                              target_distance=4.0, expected_move=0.02,
                              max_hold_bars=24)

        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.market.exchange = "synthetic"
            cfg.market.candles = 400
            (Path(d) / "data").mkdir(parents=True, exist_ok=True)
            # Parent equity 5000 => desk pot 500 => 10% position = 50, which is
            # comfortably above the exchange minimum. (A 500 parent gives a 5.00
            # position, exactly at the minimum, and slippage then refuses it —
            # which is itself a true and useful property of this system.)
            (Path(d) / cfg.ledger.state_path).write_text(
                json.dumps({"cash": 5000.0, "equity": 5000.0}))
            reg = StrategyRegistry(Path(d) / "data" / "strategies.json")
            from magicwand.propose import ValidationResult
            reg.add(spec("breakout", {"slow": 20}),
                    ValidationResult(True, {"catalog": (True, "ok")}, [],
                                     {"holdout_return": 0.02}))

            # Re-arm the desk in config: the previous test in this class may
            # have left a revoked state behind in the shared temp roots.
            DeskConfig(enabled=True, autonomy_armed=True)
            neutral = DeepSeekClient(fake=lambda s_, u: json.dumps({
                "stance": "neutral", "confidence": 0.2, "evidence_score": 0.0,
                "thesis": "quiet", "claims": []}))

            import magicwand.runner as R
            orig = R._load_strategy
            R._load_strategy = lambda cfg_, spec_dict: AlwaysEnter(cfg_.strategy)
            try:
                rep = R.tick(cfg, neutral, DeskConfig(enabled=True,
                                                     tier=RiskTier.PAPER_DESK), root=d,
                             propose_enabled=False,
                             snapshot=self._snapshot(utcnow().isoformat()),
                             allow_stale_strategy=True)
            finally:
                R._load_strategy = orig

            self.assertNotEqual(rep.research_decision, "veto")
            self.assertTrue(any("TRADE" in a for a in rep.actions),
                            f"expected a desk trade, got {rep.actions} / {rep.notes}")
            from magicwand.desk import AgentDesk
            desk = AgentDesk(cfg, DeskConfig(enabled=True,
                                             tier=RiskTier.PAPER_DESK), root=d)
            self.assertIsNotNone(desk.state.position)
            # Parent account untouched: the agent only ever spent desk money.
            self.assertEqual(json.loads((Path(d) / cfg.ledger.state_path).read_text())["cash"],
                             5000.0)

    def test_trade_branch_blocks_on_veto(self):
        """Same setup, but a research veto must stop the trade."""
        from magicwand.strategy import Strategy, Intent
        from magicwand.common import Side

        class AlwaysEnter(Strategy):
            name = "always_enter"
            def warmup(self): return 1
            def decide(self, bars, i, in_position):
                if in_position: return None
                return Intent("enter", Side.BUY, "test", stop_distance=2.0,
                              target_distance=4.0, expected_move=0.02,
                              max_hold_bars=24)

        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.market.exchange = "synthetic"
            cfg.market.candles = 400
            (Path(d) / "data").mkdir(parents=True, exist_ok=True)
            (Path(d) / cfg.ledger.state_path).write_text(
                json.dumps({"cash": 500.0, "equity": 500.0}))
            reg = StrategyRegistry(Path(d) / "data" / "strategies.json")
            from magicwand.propose import ValidationResult
            reg.add(spec("breakout", {"slow": 20}),
                    ValidationResult(True, {"catalog": (True, "ok")}, [],
                                     {"holdout_return": 0.02}))
            veto = DeepSeekClient(fake=lambda s_, u: json.dumps({
                "stance": "veto", "confidence": 0.95, "evidence_score": -0.9,
                "thesis": "hack", "claims": []}))
            import magicwand.runner as R
            orig = R._load_strategy
            R._load_strategy = lambda cfg_, spec_dict: AlwaysEnter(cfg_.strategy)
            try:
                rep = R.tick(cfg, veto, DeskConfig(enabled=True,
                                                   tier=RiskTier.PAPER_DESK), root=d,
                             propose_enabled=False,
                             snapshot=self._snapshot(utcnow().isoformat()),
                             allow_stale_strategy=True)
            finally:
                R._load_strategy = orig
            self.assertEqual(rep.research_decision, "veto")
            self.assertFalse(any("TRADE" in a for a in rep.actions))
            from magicwand.desk import AgentDesk
            desk = AgentDesk(cfg, DeskConfig(enabled=True,
                                             tier=RiskTier.PAPER_DESK), root=d)
            self.assertIsNone(desk.state.position)
