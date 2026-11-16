"""Strategy composition: invention inside an auditable grammar."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import data as dm
from magicwand.common import Bar
from magicwand.compose import (Composition, CompositionError, FEATURES,
                               MAX_CLAUSES, build, is_novel, parse,
                               validate_composition)
from magicwand.config import Config
from magicwand.llm import DeepSeekClient, fake_composition
from magicwand.compose import propose


def comp_payload(**over):
    base = {"clauses": [{"feature": "rsi", "op": "<", "value": 35}],
            "stop_atr": 2.0, "target_atr": 4.0, "max_hold_bars": 48,
            "hypothesis": "oversold"}
    base.update(over)
    return base


class TestParsing(unittest.TestCase):
    def test_valid(self):
        c = parse(comp_payload())
        self.assertEqual(len(c.clauses), 1)
        self.assertEqual(c.clauses[0].feature, "rsi")

    def test_unknown_feature_rejected(self):
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=[{"feature": "magic", "op": "<", "value": 1}]))

    def test_bad_operator_rejected(self):
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=[{"feature": "rsi", "op": "~", "value": 30}]))

    def test_out_of_range_value_rejected(self):
        for feature, bad in (("rsi", 999), ("range_pos", -1), ("zscore", 99)):
            with self.assertRaises(CompositionError):
                parse(comp_payload(clauses=[{"feature": feature, "op": "<",
                                             "value": bad}]))

    def test_too_many_clauses_rejected(self):
        clauses = [{"feature": f, "op": "<", "value": 0.0}
                   for f in list(FEATURES)[:MAX_CLAUSES + 1]]
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=clauses))

    def test_duplicate_feature_rejected(self):
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=[
                {"feature": "rsi", "op": "<", "value": 35},
                {"feature": "rsi", "op": ">", "value": 20}]))

    def test_empty_rejected(self):
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=[]))

    def test_nan_rejected(self):
        with self.assertRaises(CompositionError):
            parse(comp_payload(clauses=[{"feature": "rsi", "op": "<",
                                         "value": float("nan")}]))

    def test_json_string_with_prose(self):
        c = parse('Here: {"clauses":[{"feature":"ret5","op":"<","value":-0.01}],'
                  '"stop_atr":2,"target_atr":3,"max_hold_bars":24}')
        self.assertEqual(c.clauses[0].feature, "ret5")


class TestStrategy(unittest.TestCase):
    def test_features_are_causal(self):
        bars = dm.synthetic_bars(300, seed=3)
        c = parse(comp_payload())
        strat = build(c)
        i = 250
        before = strat._row([b.close for b in bars[:i + 1]], bars, i)
        tampered = list(bars)
        for k in range(i + 1, len(bars)):
            b = bars[k]
            tampered[k] = Bar(ts=b.ts, open=b.open * 3, high=b.high * 3,
                              low=b.low * 3, close=b.close * 3, volume=b.volume)
        after = strat._row([b.close for b in bars[:i + 1]], tampered, i)
        # features for i come from bars 0..i, so identical
        self.assertEqual(before, after)

    def test_exit_when_a_clause_stops_holding(self):
        c = parse(comp_payload())
        strat = build(c)
        bars = dm.synthetic_bars(200, seed=4)
        found = False
        for i in range(strat.warmup(), len(bars)):
            intent = strat.decide(bars, i, in_position=True)
            if intent is not None:
                self.assertEqual(intent.action, "exit")
                found = True
                break
        self.assertTrue(found, "a composition must be able to exit")

    def test_fingerprint_is_stable_and_distinguishes(self):
        a = parse(comp_payload())
        b = parse(comp_payload())
        c = parse(comp_payload(clauses=[{"feature": "rsi", "op": "<", "value": 40}]))
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())


class TestNovelty(unittest.TestCase):
    def test_identical_is_not_novel(self):
        a = parse(comp_payload())
        prior = [{"name": "x", "fingerprint": a.fingerprint(),
                  "clause_set": [c.label() for c in a.clauses]}]
        self.assertFalse(is_novel(a, prior)[0])

    def test_different_is_novel(self):
        a = parse(comp_payload())
        other = parse(comp_payload(clauses=[{"feature": "ret5", "op": "<",
                                             "value": -0.02}]))
        prior = [{"name": "x", "fingerprint": other.fingerprint(),
                  "clause_set": [c.label() for c in other.clauses]}]
        self.assertTrue(is_novel(a, prior)[0])


class TestGate(unittest.TestCase):
    def test_composition_runs_the_same_gate(self):
        bars = dm.synthetic_bars(600, seed=9, drift=0.001)
        cfg = Config()
        cfg.risk.equity = 1000.0
        result = validate_composition(parse(comp_payload()), bars, cfg)
        for gate in ("profitable", "costs_2x", "beats_hold", "evidence",
                     "out_of_sample", "not_luck"):
            self.assertIn(gate, result.gates)

    def test_insufficient_data_rejected(self):
        bars = dm.synthetic_bars(80, seed=1)
        result = validate_composition(parse(comp_payload()), bars, Config())
        self.assertFalse(result.passed)


class TestProposer(unittest.TestCase):
    def test_offline_composer_produces_valid_compositions(self):
        client = DeepSeekClient(fake=fake_composition)
        for _ in range(3):
            c = propose(client, "SOL-USD", proposer_fake=fake_composition)
            self.assertGreaterEqual(len(c.clauses), 1)
            self.assertTrue(c.hypothesis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
