"""Symbol handling: one market per instance, chosen safely.

The trap this guards against: switching the market to BTC but still reading
r/solana, which would quietly corrupt the social reader's input.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import dashboard, runner
from magicwand.config import Config
from magicwand import markets
from magicwand.research import subreddit_for
from magicwand.settings_store import SettingsStore


class TestSubredditMapping(unittest.TestCase):
    def test_each_supported_symbol_has_a_community(self):
        for sym in markets.symbols():
            self.assertNotEqual(subreddit_for(sym), "")
            self.assertIsNotNone(markets.get(sym))

    def test_unknown_symbol_falls_back_to_the_broad_community(self):
        self.assertEqual(subreddit_for("DOGE-USD"), "CryptoCurrency")

    def test_btc_does_not_read_the_solana_community(self):
        self.assertEqual(subreddit_for("BTC-USD"), "Bitcoin")
        self.assertNotEqual(subreddit_for("BTC-USD"), "solana")


class TestSymbolResolution(unittest.TestCase):
    def test_settings_symbol_wins_over_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = SettingsStore(root / "data" / "settings.json")
            s = st.load()
            s.symbol = "BTC-USD"
            st.save(s)
            cfg = Config()
            self.assertEqual(cfg.market.symbol, "SOL-USD")
            self.assertEqual(runner._resolve_symbol(cfg, root), "BTC-USD")

    def test_config_is_the_fallback_without_settings(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(runner._resolve_symbol(Config(), Path(d)), "SOL-USD")


class TestSupportedSymbols(unittest.TestCase):
    def test_supported_set_is_deliberate(self):
        """The dashboard accepts exactly the markets the registry lists."""
        self.assertEqual(set(dashboard._SUPPORTED_SYMBOLS), set(markets.symbols()))

    def test_every_supported_symbol_is_on_the_desk_whitelist(self):
        from magicwand.desk import DeskConfig
        self.assertTrue(set(dashboard._SUPPORTED_SYMBOLS)
                        <= set(DeskConfig().whitelist))


class TestMarketsRegistry(unittest.TestCase):
    """Adding a market must be a data change, and every listed market must have
    a real source for candles, derivatives and social."""

    def test_listed_markets(self):
        from magicwand import markets
        self.assertEqual(markets.symbols(), ["BTC-USD", "ETH-USD", "PAXG-USD", "SOL-USD"])

    def test_every_market_has_all_three_sources(self):
        from magicwand import markets
        for sym, m in markets.MARKETS.items():
            self.assertTrue(m.coinbase, f"{sym} needs a Coinbase product")
            self.assertTrue(m.kraken, f"{sym} needs a Kraken pair")
            self.assertTrue(m.binance_perp, f"{sym} needs a Binance perp")
            self.assertTrue(m.subreddit, f"{sym} needs a subreddit")

    def test_gold_is_labelled_as_a_proxy(self):
        from magicwand import markets
        gold = markets.get("PAXG-USD")
        self.assertIsNotNone(gold)
        self.assertIn("XAU", gold.note)
        self.assertIn("token risk", gold.note)

    def test_venue_names_are_translated(self):
        from magicwand import markets
        self.assertEqual(markets.coinbase_product("PAXG-USD"), "PAXG-USD")
        self.assertEqual(markets.kraken_pair("PAXG-USD"), "PAXGUSD")
        self.assertEqual(markets.kraken_pair("BTC-USD"), "XBTUSD")
        self.assertEqual(markets.binance_perp("PAXG-USD"), "PAXGUSDT")

    def test_desk_whitelist_matches_the_registry(self):
        from magicwand import markets
        from magicwand.desk import DeskConfig
        self.assertEqual(set(DeskConfig().whitelist), set(markets.symbols()))

    def test_dashboard_supported_set_matches_the_registry(self):
        from magicwand import markets
        self.assertEqual(set(dashboard._SUPPORTED_SYMBOLS), set(markets.symbols()))

    def test_exchange_support_filter(self):
        from magicwand import markets
        self.assertIn("PAXG-USD", markets.for_exchange("coinbase"))
        self.assertIn("PAXG-USD", markets.for_exchange("kraken"))


class TestExchangeSelection(unittest.TestCase):
    def test_exchange_resolves_from_settings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = SettingsStore(root / "data" / "settings.json")
            s = st.load()
            s.exchange = "kraken"
            st.save(s)
            self.assertEqual(runner._resolve_exchange(Config(), root), "kraken")

    def test_invalid_exchange_falls_back_to_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = SettingsStore(root / "data" / "settings.json")
            s = st.load()
            s.exchange = "mt5"
            st.save(s)
            cfg = Config()
            self.assertEqual(runner._resolve_exchange(cfg, root), cfg.market.exchange)

    def test_market_must_exist_on_the_chosen_venue(self):
        self.assertTrue(dashboard._market_on_exchange("PAXG-USD", "kraken"))
        self.assertTrue(dashboard._market_on_exchange("BTC-USD", "coinbase"))
        self.assertFalse(dashboard._market_on_exchange("NOTREAL-USD", "coinbase"))
