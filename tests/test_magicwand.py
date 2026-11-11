"""MAGICWAND test suite. Plain unittest — no pytest needed.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import backtest as bt
from magicwand import data as data_mod
from magicwand import indicators as ind
from magicwand import paper
from magicwand.broker import PaperBroker, Portfolio, check_exit
from magicwand.common import Bar, ExitReason, Position, Side
from magicwand.config import Config
from magicwand.ledger import GENESIS, Ledger, grade_claim, make_claim, scorecard
from magicwand.risk import RiskEngine
from magicwand.strategy import build as build_strategy


def mkbar(i: int, o: float, h: float, l: float, c: float, vol: float = 10.0) -> Bar:
    return Bar(ts=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
               open=o, high=h, low=l, close=c, volume=vol)


class TestIndicators(unittest.TestCase):
    def test_sma_is_causal(self):
        xs = [1, 2, 3, 4, 5, 6]
        s = ind.sma(xs, 3)
        self.assertIsNone(s[0])
        self.assertIsNone(s[1])
        self.assertEqual(s[2], 2.0)
        self.assertEqual(s[-1], 5.0)
        # A value must never depend on later elements.
        self.assertEqual(ind.sma(xs[:4], 3)[3], s[3])

    def test_atr_positive_and_warm(self):
        bars = [mkbar(i, 100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)]
        a = ind.atr(bars, 14)
        self.assertIsNone(a[5])
        self.assertIsNotNone(a[20])
        self.assertGreater(a[20], 0)

    def test_rsi_bounds(self):
        bars = [mkbar(i, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(40)]
        r = ind.rsi(bars, 14)
        self.assertIsNotNone(r[30])
        self.assertGreaterEqual(r[30], 0)
        self.assertLessEqual(r[30], 100)


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_respects_risk_per_trade(self):
        re = RiskEngine(self.cfg.risk)
        d = re.size_entry(equity=1000.0, price=100.0, stop_distance=5.0,
                          open_positions=0, when=datetime.now(timezone.utc))
        self.assertTrue(d.allowed, d.reason)
        # 1% of 1000 = 10 risked; stop 5 => qty 2
        self.assertAlmostEqual(d.qty, 2.0, places=6)

    def test_position_cap(self):
        re = RiskEngine(self.cfg.risk)
        d = re.size_entry(equity=1000.0, price=100.0, stop_distance=0.01,
                          open_positions=0, when=datetime.now(timezone.utc))
        self.assertTrue(d.allowed)
        self.assertLessEqual(d.qty * 100.0, 1000.0 * self.cfg.risk.max_position_pct + 1e-6)

    def test_small_account_is_refused_not_rounded_up(self):
        """At $50 equity, a 25% position cap only allows $12.50 of notional.
        A price/stop combination wanting less than the $5 exchange minimum
        must be refused outright, never rounded up to fit."""
        cfg = Config()
        cfg.risk.equity = 50.0
        re = RiskEngine(cfg.risk)
        # qty from risk = 0.5/30 = 0.0167 units; at $100 that is $1.67 < $5.
        d = re.size_entry(equity=50.0, price=100.0, stop_distance=30.0,
                          open_positions=0, when=datetime.now(timezone.utc))
        self.assertFalse(d.allowed, d.reason)
        self.assertIn("too small", d.reason)

    def test_small_account_can_still_trade(self):
        """The same $50 account CAN trade a normal setup — the refusal above
        is about size, not panic. It sizes to the position cap."""
        cfg = Config()
        cfg.risk.equity = 50.0
        re = RiskEngine(cfg.risk)
        d = re.size_entry(equity=50.0, price=100.0, stop_distance=2.0,
                          open_positions=0, when=datetime.now(timezone.utc))
        self.assertTrue(d.allowed, d.reason)
        self.assertAlmostEqual(d.qty * 100.0, 12.5, places=6)  # 25% of 50
        self.assertIn("max-position-size", d.reason)

    def test_kill_switch_blocks(self):
        cfg = Config()
        cfg.risk.kill_switch = True
        re = RiskEngine(cfg.risk)
        d = re.size_entry(1000.0, 100.0, 5.0, 0, datetime.now(timezone.utc))
        self.assertFalse(d.allowed)
        self.assertIn("kill switch", d.reason)

    def test_daily_loss_halts(self):
        re = RiskEngine(self.cfg.risk)
        now = datetime.now(timezone.utc)
        re.update_equity(1000.0, now)          # morning: baseline 1000
        re.update_equity(970.0, now)           # marked down 3% intraday
        d = re.size_entry(970.0, 100.0, 5.0, 0, now)
        self.assertFalse(d.allowed)
        self.assertIn("daily loss", d.reason)

    def test_daily_baseline_is_not_rewritten_within_a_day(self):
        """Regression: a later realized result must not move the day's baseline."""
        re = RiskEngine(self.cfg.risk)
        now = datetime.now(timezone.utc)
        re.update_equity(1000.0, now)
        re.record_realized(-10.0, 990.0, now)
        self.assertAlmostEqual(re.state.day_start_equity, 1000.0, places=6)
        re.record_realized(-10.0, 980.0, now)
        self.assertAlmostEqual(re.state.day_start_equity, 1000.0, places=6)

    def test_max_open_positions(self):
        re = RiskEngine(self.cfg.risk)
        d = re.size_entry(1000.0, 100.0, 5.0, 1, datetime.now(timezone.utc))
        self.assertFalse(d.allowed)


class TestBroker(unittest.TestCase):
    def test_fills_are_pessimistic(self):
        cfg = Config()
        b = PaperBroker(cfg.costs, cfg.risk)
        pf = Portfolio(cash=1000.0)
        f = b.buy(pf, quote=100.0, qty=1.0, ts=datetime.now(timezone.utc),
                  reason="test", stop=95.0)
        self.assertGreater(f.price, 100.0, "buy fill must be worse than quote")
        self.assertGreater(f.fee, 0.0)
        t = b.sell(pf, quote=100.0, ts=datetime.now(timezone.utc), reason="exit")
        self.assertLess(t.exit_price, 100.0, "sell fill must be worse than quote")
        self.assertLess(t.net, 0.0, "round trip flat price must lose money to costs")

    def test_stop_wins_ties(self):
        pos = Position(symbol="X", side=Side.BUY, qty=1.0, entry_price=100.0,
                       entry_ts=datetime.now(timezone.utc), stop=95.0, target=105.0)
        bar = mkbar(0, 100, 110, 90, 100)  # hits both in one bar
        hit = check_exit(pos, bar)
        self.assertEqual(hit.reason, ExitReason.STOP)

    def test_equity_is_continuous_while_holding(self):
        """Regression: opening a position must not make equity collapse.

        Cash drops by the notional at entry, so mark() must add the position
        back at market. Before the fix, a $10 position made $50 of equity
        look like $40.
        """
        cfg = Config()
        b = PaperBroker(cfg.costs, cfg.risk)
        pf = Portfolio(cash=50.0)
        now = datetime.now(timezone.utc)
        fill = b.buy(pf, quote=100.0, qty=0.1, ts=now, reason="t", stop=98.0)
        eq_flat = pf.mark(100.0)
        # Equity = cash + position at market. Right after entry, marked at the
        # quote, that is starting cash minus the entry fee and the slippage
        # cost of buying at 100.01 instead of 100.00 — and nothing else.
        expected = 50.0 - fill.fee - (fill.price - 100.0) * 0.1
        self.assertAlmostEqual(eq_flat, expected, places=9)
        self.assertGreater(eq_flat, 49.9)
        self.assertLess(eq_flat, 50.0)
        eq_up = pf.mark(101.0)
        self.assertAlmostEqual(eq_up - eq_flat, 0.1 * 1.0, places=9)  # exact P&L
        eq_down = pf.mark(98.0)
        self.assertAlmostEqual(eq_down - eq_flat, 0.1 * -2.0, places=9)
        self.assertLess(eq_down, eq_flat)     # real loss, not a fake collapse
        self.assertGreater(eq_down, 49.0)

    def test_closed_trade_matches_cash(self):
        cfg = Config()
        b = PaperBroker(cfg.costs, cfg.risk)
        pf = Portfolio(cash=50.0)
        now = datetime.now(timezone.utc)
        b.buy(pf, quote=100.0, qty=0.1, ts=now, reason="t", stop=98.0)
        t = b.sell(pf, quote=105.0, ts=now, reason="exit")
        self.assertAlmostEqual(pf.cash, 50.0 + t.net, places=9)
        self.assertAlmostEqual(pf.mark(105.0), pf.cash, places=9)


class TestBacktest(unittest.TestCase):
    def test_no_lookahead_fill(self):
        """A signal on bar i must fill at bar i+1's open, not bar i's close."""
        bars = data_mod.synthetic_bars(120, seed=3)
        cfg = Config()
        cfg.strategy.name = "coin_flip"
        strat = build_strategy(cfg.strategy, seed=3)
        r = bt.run_backtest(bars, cfg, strat)
        for t in r.trades:
            idx = next(i for i, b in enumerate(bars) if b.ts >= t.entry_ts)
            self.assertGreaterEqual(t.entry_price, bars[idx].low * 0.5)
            self.assertLessEqual(t.entry_price, bars[idx].high * 1.5)

    def test_deterministic(self):
        bars = data_mod.synthetic_bars(200, seed=11)
        cfg = Config()
        a = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=11))
        b = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=11))
        self.assertAlmostEqual(a.final_equity, b.final_equity, places=9)
        self.assertEqual(len(a.trades), len(b.trades))

    def test_zero_drift_market_has_no_edge(self):
        """The correct null: on a fair coin, sma_trend should not reliably profit."""
        cfg = Config()
        cfg.risk.equity = 1000.0
        results = []
        for seed in range(6):
            bars = data_mod.synthetic_bars(600, seed=seed, drift=0.0)
            r = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=seed))
            results.append(r.total_return)
        mean_ret = sum(results) / len(results)
        self.assertLess(mean_ret, 0.10, f"should not find a big edge on noise: {results}")

    def test_signal_delay_actually_changes_results(self):
        """The lag stress test is only honest if the delay does something.

        This existed as a no-op once: `signal_delay` was accepted and ignored,
        so the reality check reported 'no lag cost' no matter what. A fake
        test is worse than no test.
        """
        bars = data_mod.synthetic_bars(400, seed=9, drift=0.001)
        cfg = Config()
        cfg.risk.equity = 1000.0
        fast = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=9),
                               bt.BacktestOptions(signal_delay=1, seed=9))
        slow = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=9),
                               bt.BacktestOptions(signal_delay=3, seed=9))
        self.assertNotAlmostEqual(fast.final_equity, slow.final_equity, places=6)

    def test_uptrend_does_not_lose_money_by_accident(self):
        cfg = Config()
        cfg.risk.equity = 1000.0
        bars = data_mod.synthetic_bars(600, seed=2, drift=0.002)
        r = bt.run_backtest(bars, cfg, build_strategy(cfg.strategy, seed=2))
        self.assertGreater(r.final_equity, 0)


class TestLedger(unittest.TestCase):
    def test_chain_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.jsonl"
            led = Ledger(p)
            led.append("note", {"msg": "hello"})
            led.append("claim", make_claim("c1", "SOL-USD", 0.01, 100.0, 24, "test"))
            ok, msg = led.verify()
            self.assertTrue(ok, msg)
            lines = p.read_text().splitlines()
            lines[0] = lines[0].replace("hello", "goodbye")
            p.write_text("\n".join(lines) + "\n")
            ok, msg = Ledger(p).verify()
            self.assertFalse(ok)
            self.assertIn("tampered", msg)

    def test_grade_logic(self):
        claim = make_claim("c", "X", 0.02, 100.0, 24, "up")
        g = grade_claim(claim, 0.025, 10, "target")
        self.assertEqual(g["verdict"], "right")
        g2 = grade_claim(claim, -0.001, 10, "stop")
        self.assertEqual(g2["verdict"], "wrong")
        g3 = grade_claim(claim, 0.001, 10, "time")
        self.assertEqual(g3["verdict"], "overstated")

    def test_scorecard(self):
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(Path(d) / "l.jsonl")
            c = make_claim("c1", "X", 0.02, 100.0, 24, "up")
            led.append("claim", c)
            led.append("grade", grade_claim(c, 0.03, 5, "target"))
            sc = scorecard(led.entries())
            self.assertEqual(sc["grades"], 1)
            self.assertEqual(sc["hit_rate"], 1.0)


class TestPaperLoop(unittest.TestCase):
    def test_step_writes_ledger_and_state(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.market.exchange = "synthetic"
            cfg.market.candles = 200
            bars = data_mod.synthetic_bars(200, seed=5, drift=0.003)
            rep = paper.step(cfg, bars=bars, root=d)
            self.assertIn("equity", rep)
            p = Path(d) / cfg.ledger.path
            self.assertTrue(p.exists())
            led = Ledger(p)
            ok, msg = led.verify()
            self.assertTrue(ok, msg)
            st = paper.status(cfg, root=d)
            self.assertTrue(st["ledger_ok"])
            self.assertGreater(st["entries"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
