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
from magicwand.research import SUBREDDIT_FOR_SYMBOL, subreddit_for
from magicwand.settings_store import SettingsStore


class TestSubredditMapping(unittest.TestCase):
    def test_each_supported_symbol_has_a_community(self):
        for sym in ("SOL-USD", "BTC-USD", "ETH-USD"):
            self.assertIn(sym, SUBREDDIT_FOR_SYMBOL)
            self.assertNotEqual(subreddit_for(sym), "")

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
        self.assertEqual(set(dashboard._SUPPORTED_SYMBOLS),
                         {"BTC-USD", "ETH-USD", "SOL-USD"})

    def test_every_supported_symbol_is_on_the_desk_whitelist(self):
        from magicwand.desk import DeskConfig
        self.assertTrue(set(dashboard._SUPPORTED_SYMBOLS)
                        <= set(DeskConfig().whitelist))
