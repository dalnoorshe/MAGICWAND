"""Tests for the new inputs (cross-asset, session, funding) and the new label.

Two things can go wrong here and both are silent:

  * a feature that peeks. If a BTC bar, a funding rate or a correlation window
    can see one bar into the future, every score downstream is fiction and the
    number will look *better*, not worse.
  * a feature that never arrives. The funding index was wired wrong for ten
    minutes during development: every funding feature was silently absent, the
    model trained happily on price-only columns, and the experiment would have
    reported "funding adds nothing" while measuring nothing. Hence the
    presence tests.
"""

from __future__ import annotations

import unittest
from datetime import timedelta

from magicwand import data as data_mod
from magicwand import features as feat_mod
from magicwand import forecast as fc
from magicwand.common import iso


def _bars(n=300, seed=7):
    return data_mod.synthetic_bars(n, seed=seed, timeframe="1h")


def _ctx(bars, groups=("cross", "session", "funding")):
    """A context with no network: reuse the same series for every asset."""
    series = feat_mod.load_series("SOL-USD", "1h", bars)
    epochs = [feat_mod._epoch(bars, i) for i in range(len(bars))]
    ctx = feat_mod.Context(epochs=epochs, sol=series, btc=dict(series),
                           eth=dict(series))
    # Funding published every 8 hours, strictly in the past.
    ctx.set_funding([(epochs[0] * 1000 + k * 8 * 3600 * 1000,
                      0.0001 * (k % 5)) for k in range(len(bars) // 8 + 1)])
    return ctx


class TestFeatureCausality(unittest.TestCase):
    def test_no_extra_feature_changes_when_the_future_is_removed(self):
        bars = _bars()
        ctx = _ctx(bars)
        for i in (120, 200, 280):
            full = feat_mod.extras(bars, i, ctx, ("cross", "session", "funding"))
            cut = feat_mod.extras(bars[: i + 1], i, _ctx(bars[: i + 1]),
                                 ("cross", "session", "funding"))
            self.assertEqual(set(full), set(cut),
                             "a feature disappeared when the future was cut off")
            for k in full:
                self.assertAlmostEqual(full[k], cut[k], places=12,
                                       msg=f"{k} peeked at the future")

    def test_a_missing_counterpart_bar_omits_the_feature_instead_of_guessing(self):
        bars = _bars()
        ctx = _ctx(bars)
        ctx.btc = {}                       # venue down
        ex = feat_mod.extras(bars, len(bars) - 1, ctx, ("cross",))
        self.assertNotIn("btc_ret6", ex)
        self.assertNotIn("btc_ret24", ex)

    def test_funding_uses_only_rates_already_published(self):
        bars = _bars()
        ctx = _ctx(bars)
        i = len(bars) - 1
        ts = ctx.epochs[i] * 1000
        published = [r for t, r in ctx.funding if t <= ts]
        self.assertEqual(ctx.funding_at(ts), published[-1])
        # The next rate (if any) must be invisible from here.
        future = [(t, r) for t, r in ctx.funding if t > ts]
        if future:
            self.assertNotEqual(ctx.funding_at(ts), future[0][1])

    def test_funding_features_are_actually_present(self):
        bars = _bars()
        ctx = _ctx(bars)
        ex = feat_mod.extras(bars, len(bars) - 1, ctx, ("funding",))
        for key in feat_mod.FUNDING_FEATURES:
            self.assertIn(key, ex, f"{key} never arrived — the index is broken")

    def test_session_features_are_bounded(self):
        bars = _bars()
        ctx = _ctx(bars)
        ex = feat_mod.extras(bars, 50, ctx, ("session",))
        self.assertLessEqual(abs(ex["hour_sin"]), 1.0)
        self.assertLessEqual(abs(ex["hour_cos"]), 1.0)


class TestTripleBarrier(unittest.TestCase):
    def _bar(self, o, h, l, c, ts=None):
        from magicwand.common import Bar
        return Bar(ts=ts or _bars(1)[0].ts, open=o, high=h, low=l, close=c,
                   volume=1.0)

    def test_upper_hit_first_labels_one(self):
        bars = [self._bar(100, 101, 99, 100)]
        # ATR needs history, so build a flat series then one clean up-run.
        flat = [self._bar(100, 101, 99, 100) for _ in range(20)]
        up = [self._bar(100 + 5 * k, 106 + 5 * k, 99 + 5 * k, 105 + 5 * k)
              for k in range(1, 4)]
        label, note = fc.triple_barrier(flat + up, 19, 3, barrier_atr=0.5)
        self.assertEqual(label, 1)
        self.assertEqual(note, "upper")

    def test_lower_hit_first_labels_zero(self):
        flat = [self._bar(100, 101, 99, 100) for _ in range(20)]
        down = [self._bar(100 - 5 * k, 101 - 5 * k, 94 - 5 * k, 95 - 5 * k)
                for k in range(1, 4)]
        label, note = fc.triple_barrier(flat + down, 19, 3, barrier_atr=0.5)
        self.assertEqual(label, 0)
        self.assertEqual(note, "lower")

    def test_it_never_looks_past_the_horizon(self):
        flat = [self._bar(100, 101, 99, 100) for _ in range(20)]
        # One quiet bar inside the horizon, then an explosion outside it. With
        # horizon=1 at index 19 only bar 20 is visible.
        # Tight bar: does not touch either barrier (ATR of the flat run is 2.0,
        # so the barriers sit at 101 / 99 and this bar spans 99.6-100.4).
        quiet = [self._bar(100, 100.4, 99.6, 100)]
        later = [self._bar(200, 260, 199, 255) for _ in range(3)]
        label, note = fc.triple_barrier(flat + quiet + later, 19, 1,
                                        barrier_atr=0.5)
        self.assertEqual(note, "no_barrier")
        self.assertEqual(label, 0, "a move beyond the horizon changed the label")

    def test_a_missing_atr_is_refused_not_guessed(self):
        bars = self._bar, self._bar(100, 101, 99, 100)
        label, note = fc.triple_barrier([bars[1]], 0, 3)
        self.assertIsNone(label)
        self.assertEqual(note, "no_atr")


class TestDatasetAssembly(unittest.TestCase):
    def test_triple_labels_have_their_own_base_rate(self):
        bars = _bars(400)
        stats: dict = {}
        rows, ys, idxs = fc.build_dataset(bars, 6, label="triple", stats=stats)
        self.assertEqual(len(rows), len(ys))
        self.assertEqual(len(ys), len(idxs))
        self.assertIn("n", stats)
        self.assertGreater(stats.get("upper", 0) + stats.get("lower", 0), 0)
        # Every row got exactly one label, and the tallies agree with n.
        tallied = sum(stats.get(k, 0) for k in
                      ("upper", "lower", "both_same_bar", "no_barrier"))
        self.assertEqual(tallied, stats["n"])

    def test_feature_groups_extend_the_row_and_never_impute(self):
        bars = _bars(400)
        ctx = _ctx(bars)
        plain, _, _ = fc.build_dataset(bars, 6)
        wide, _, _ = fc.build_dataset(bars, 6, groups=("cross", "session"),
                                      ctx=ctx)
        names = fc.feature_names(("cross", "session"))
        self.assertEqual(len(wide[0]), len(names))
        self.assertGreater(len(wide[0]), len(plain[0]))
        ctx.btc = {}
        thin, _, _ = fc.build_dataset(bars, 6, groups=("cross",), ctx=ctx)
        self.assertLess(len(thin), len(plain) + 1,
                        "rows survived with a missing feature")

    def test_build_dataset_refuses_groups_without_context(self):
        with self.assertRaises(ValueError):
            fc.build_dataset(_bars(100), 6, groups=("cross",))

    def test_feature_names_starts_with_the_price_features(self):
        names = fc.feature_names(("funding",))
        self.assertEqual(names[: len(fc.FEATURES)], list(fc.FEATURES))


if __name__ == "__main__":
    unittest.main()


class TestFundingPaging(unittest.TestCase):
    """The first version asked for the venue's default page and got 500 records
    — 137 days — while claiming to evaluate two years. It silently turned the
    feature comparison into a comparison on a fifth of the data.
    """

    def test_paging_walks_forward_from_a_window_not_from_the_default_page(self):
        import inspect
        src = inspect.getsource(feat_mod.load_funding)
        self.assertIn("startTime", src)
        self.assertIn("endTime", src)
        self.assertIn("while start < now_ms", src)

    def test_a_stuck_page_cannot_loop_forever(self):
        """If the venue keeps returning a full page at the same timestamp, stop."""
        calls = []
        import magicwand.research as research_mod
        saved = research_mod._get_json

        def fake(url):
            calls.append(url)
            # always a full page, with a timestamp that never advances
            return [{"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"}] * 1000

        research_mod._get_json = fake
        try:
            out = feat_mod.load_funding("SOL-USD", days=30, page=1000)
        finally:
            research_mod._get_json = saved
        self.assertLessEqual(len(calls), 2, "the pager did not stop on a stuck page")
        self.assertEqual(len(out), 1)


class TestBarLengths(unittest.TestCase):
    """Coinbase's granularity whitelist is not the list of bar lengths.

    Binance serves 4h bars; Coinbase does not. Using one map for both jobs made
    `wand.py forecast --timeframe 4h` die with a KeyError — a crash, not a wrong
    number, but it only surfaced because the experiment was actually run.
    """

    def test_four_hour_bars_have_a_length(self):
        from magicwand import data as data_mod
        self.assertEqual(data_mod._bar_seconds("4h"), 14400)
        self.assertEqual(data_mod.timeframe_seconds("4h"), 14400)

    def test_unknown_timeframes_raise_instead_of_assuming_an_hour(self):
        from magicwand import data as data_mod
        with self.assertRaises(data_mod.FeedError):
            data_mod._bar_seconds("7h")

    def test_every_supported_binance_interval_has_a_length(self):
        from magicwand import data as data_mod
        for tf in data_mod._BINANCE_INTERVAL:
            self.assertGreater(data_mod._bar_seconds(tf), 0, tf)

    def test_coinbase_granularity_still_rejects_4h(self):
        """The Coinbase whitelist must not grow just because Binance is happy."""
        from magicwand import data as data_mod
        self.assertNotIn("4h", data_mod._GRANULARITY)


class TestScanStatistics(unittest.TestCase):
    """The scan exists to stop us fooling ourselves with many tests.

    If you try enough cells, one will look good. The only defence is to say how
    many were tried and require the winner to clear that bar.
    """

    def test_a_coin_flip_is_not_significant(self):
        from magicwand.cli import _binom_p
        self.assertGreater(_binom_p(500, 1000, 0.5), 0.4)

    def test_a_clear_win_is_significant(self):
        from magicwand.cli import _binom_p
        self.assertLess(_binom_p(550, 1000, 0.5), 0.001)

    def test_a_loss_is_never_significant(self):
        from magicwand.cli import _binom_p
        self.assertGreater(_binom_p(450, 1000, 0.5), 0.99)

    def test_the_threshold_tightens_with_more_cells(self):
        from magicwand.cli import _binom_p
        # 540/1000 clears an uncorrected 0.05 bar but not a 12-cell one.
        p = _binom_p(540, 1000, 0.5)
        self.assertLess(p, 0.05)
        self.assertGreater(p, 0.05 / 12)
