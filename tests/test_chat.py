"""The operator chat: read-only by construction."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import chat
from magicwand.common import iso, utcnow
from magicwand.dashboard import build_state
from magicwand.config import Config


def fake_state() -> dict:
    ts = iso(utcnow())
    return {
        "server_time": ts, "model_mode": "live", "symbol": "PAXG-USD",
        "exchange": "coinbase", "timeframe": "1h", "account_equity": 50.0,
        "settings": {"symbol": "PAXG-USD", "exchange": "coinbase",
                     "has_key": True, "key_source": "saved in console",
                     "model": "deepseek-flash",
                     "markets": [{"symbol": "PAXG-USD"}, {"symbol": "SOL-USD"}]},
        "policy": {"tier_name": "research",
                   "effective": {"max_pot_fraction": 0.0}},
        "desk": {"cash": 5.0, "trades": 0, "win_rate": None, "trades_today": 0,
                 "autonomy_armed": True, "position": None},
        "research": {"decision": "veto", "evidence_score": -0.42,
                     "stub": False, "summary": "5/5 readers reported",
                     "agents": [{"agent": "news", "stance": "veto",
                                 "confidence": 0.68, "evidence_score": -0.55,
                                 "thesis": "macro risk-off", "error": ""}]},
        "reflection": {"changes": []}, "memory": {"blacklisted_families": []},
        "strategies": [], "ledger": {"message": "chain intact", "entries": 800},
        "autopilot": {"running": True, "ticks": 12, "interval": 900},
        "backups": {"archives": 2, "age_hours": 1.5},
    }


class TestContext(unittest.TestCase):
    def test_context_carries_the_facts_the_model_needs(self):
        ctx = chat.build_context(fake_state())
        for phrase in ("PAXG-USD", "tier: research", "decision=veto",
                       "chain intact", "available_markets", "macro risk-off"):
            self.assertIn(phrase, ctx, phrase)

    def test_context_states_the_controls_belong_to_the_human(self):
        ctx = chat.build_context(fake_state())
        self.assertIn("the operator must do these", ctx)
        self.assertIn("Settings", ctx)

    def test_context_is_stable_with_an_empty_state(self):
        ctx = chat.build_context({})
        self.assertIn("latest_research: none", ctx)


class TestReadOnly(unittest.TestCase):
    """The chat must not be able to mutate anything.

    Enforced by absence of capability, not by asking the model nicely: the
    module does not import the broker, the desk, the ledger or the settings
    writer, and its system prompt says it cannot act.
    """

    def test_no_mutating_capability_is_imported(self):
        """Check the import surface, not docstring prose.

        If this module never imports the desk, the broker, the ledger writer or
        the policy store, it cannot mutate anything no matter what the model
        says — which is a property of the code, not of the prompt.
        """
        src = inspect.getsource(chat)
        imports = [ln for ln in src.splitlines()
                   if ln.strip().startswith(("import ", "from "))]
        joined = " ".join(imports)
        for forbidden in ("desk", "broker", "ledger", "gatekeeper", "reflect",
                          "propose", "runner"):
            self.assertNotIn(forbidden, joined,
                             f"chat must not import {forbidden!r}")

    def test_no_tool_calling_wire(self):
        """No `tools=` / `tool_choice` request field: there is nothing to call."""
        src = inspect.getsource(chat)
        self.assertNotIn("tools=", src)
        self.assertNotIn("tool_choice", src)

    def test_prompt_forbids_action(self):
        self.assertIn("CANNOT act", chat.SYSTEM)
        self.assertIn("no tools", chat.SYSTEM)


class TestHistory(unittest.TestCase):
    def test_roundtrip_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            chat.save_history(d, [])
            msgs = [{"role": "user", "text": f"m{i}", "ts": "t"} for i in range(100)]
            chat.save_history(d, msgs)
            back = chat.load_history(d)
            self.assertEqual(len(back), chat.MAX_HISTORY)
            self.assertEqual(back[-1]["text"], "m99")

    def test_corrupt_history_is_empty_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data" / "chat.json"
            p.parent.mkdir(parents=True)
            p.write_text("{not json")
            self.assertEqual(chat.load_history(d), [])

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            chat.save_history(d, [{"role": "user", "text": "x"}])
            chat.clear_history(d)
            self.assertEqual(chat.load_history(d), [])


class TestReplyGuards(unittest.TestCase):
    def test_empty_message_rejected(self):
        out = chat.reply(".", "   ", fake_state())
        self.assertFalse(out.ok)
        self.assertIn("empty", out.error)

    def test_overlong_message_rejected(self):
        out = chat.reply(".", "x" * (chat.MAX_MESSAGE_CHARS + 1), fake_state())
        self.assertFalse(out.ok)
        self.assertIn("too long", out.error)

    def test_no_key_means_a_clear_message_not_a_crash(self):
        import os
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as d:
                out = chat.reply(d, "hello", fake_state())
            self.assertFalse(out.ok)
            self.assertIn("Settings", out.error)
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
