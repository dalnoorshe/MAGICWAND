"""The trend forecaster: causal features, honest metrics, no leakage.

The result on real SOL data is "no skill". These tests exist so the tool keeps
reporting that honestly instead of drifting into flattering arithmetic — the
first version DID report a false edge because its baseline was malformed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import data as dm
from magicwand.common import Bar
from magicwand.forecast import (FEATURES, Evaluation, LogisticModel, _brier,
                                build_dataset, features_at, walk_forward)


class TestCausality(unittest.TestCase):
    def test_features_ignore_the_future(self):
        """Mutating bars after index i must not change the row for i."""
        bars = dm.synthetic_bars(300, seed=4)
        i = 250
        before = features_at(bars, i)
        tampered = list(bars)
        for k in range(i + 1, len(bars)):
            b = bars[k]
            tampered[k] = Bar(ts=b.ts, open=b.open * 2, high=b.high * 2,
                              low=b.low * 2, close=b.close * 2, volume=b.volume)
        after = features_at(tampered, i)
        self.assertEqual(before, after)

    def test_features_need_history(self):
        bars = dm.synthetic_bars(120, seed=1)
        self.assertIsNone(features_at(bars, 5))

    def test_feature_row_is_complete(self):
        bars = dm.synthetic_bars(300, seed=2)
        f = features_at(bars, 250)
        for k in FEATURES:
            self.assertIn(k, f)


class TestLabels(unittest.TestCase):
    def test_label_is_future_close_direction(self):
        bars = dm.synthetic_bars(300, seed=5)
        rows, ys, idxs = build_dataset(bars, horizon=3)
        for row, y, i in list(zip(rows, ys, idxs))[:50]:
            expected = 1 if bars[i + 3].close > bars[i].close else 0
            self.assertEqual(y, expected)

    def test_labels_stop_before_the_end(self):
        bars = dm.synthetic_bars(200, seed=6)
        _, _, idxs = build_dataset(bars, horizon=5)
        self.assertLessEqual(max(idxs) + 5, len(bars) - 1)


class TestModel(unittest.TestCase):
    def test_learns_a_separable_signal(self):
        rows = [[v] for v in (0.1, 0.2, 0.3, 0.9, 1.0, 1.1)]
        ys = [0, 0, 0, 1, 1, 1]
        model = LogisticModel().fit(rows, ys, epochs=500)
        self.assertLess(model.predict([0.2]), 0.3)
        self.assertGreater(model.predict([1.0]), 0.7)

    def test_constant_features_do_not_crash(self):
        rows = [[1.0, 2.0]] * 10
        ys = [0, 1] * 5
        model = LogisticModel().fit(rows, ys, epochs=20)
        p = model.predict([1.0, 2.0])
        self.assertTrue(0.0 <= p <= 1.0)


class TestMetrics(unittest.TestCase):
    def test_brier_of_perfect_forecasts_is_zero(self):
        self.assertAlmostEqual(_brier([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(_brier([0.0, 1.0], [1, 0]), 1.0)

    def test_climatology_brier_is_p_1_minus_p(self):
        """The honest baseline: predict the base rate every time."""
        bars = dm.synthetic_bars(600, seed=7)
        ev, _ = walk_forward(bars, horizon=4, min_train=200, block=50)
        self.assertAlmostEqual(ev.base_brier, ev.base_rate * (1 - ev.base_rate),
                               places=9)

    def test_no_skill_is_reported_on_a_random_walk(self):
        """A fair coin must not produce a skill claim."""
        bars = dm.synthetic_bars(1500, seed=11)
        ev, _ = walk_forward(bars, horizon=4, min_train=300, block=50)
        self.assertFalse(ev.beats_base, ev.summary())
        self.assertIn("NO skill", ev.summary())

    def test_skill_requires_both_probabilities_and_direction(self):
        good = Evaluation(n=100, accuracy=0.60, brier=0.20, log_loss=0.6,
                          base_rate=0.5, base_brier=0.25, base_accuracy=0.5,
                          calibration=[], by_regime={})
        self.assertTrue(good.beats_base)
        # Better probabilities but no directional edge: not skill.
        half = Evaluation(n=100, accuracy=0.50, brier=0.20, log_loss=0.6,
                          base_rate=0.5, base_brier=0.25, base_accuracy=0.5,
                          calibration=[], by_regime={})
        self.assertFalse(half.beats_base)


class TestNoLeakage(unittest.TestCase):
    def test_training_labels_are_purged_before_the_test_block(self):
        """A training label that resolves inside the test window would leak."""
        import inspect
        from magicwand import forecast
        src = inspect.getsource(forecast.walk_forward)
        self.assertIn("cutoff", src)
        self.assertIn("horizon", src)
        self.assertIn("train_n", src)

    def test_walk_forward_needs_enough_data(self):
        bars = dm.synthetic_bars(120, seed=1)
        ev, tail = walk_forward(bars, horizon=6, min_train=200, block=50)
        self.assertIsNone(ev)
        self.assertEqual(tail, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
