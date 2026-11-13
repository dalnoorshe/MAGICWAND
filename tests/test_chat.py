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


class TestConsoleStructure(unittest.TestCase):
    """The chat lives in the header as a drawer, not as a card in the grid.

    Structural tests on the page are worth it here: the two previous layout
    bugs (a modal taller than the screen, a chat card in the middle of the
    dashboard) were invisible to the Python tests.
    """

    @staticmethod
    def _html() -> str:
        return (Path(__file__).resolve().parents[1]
                / "magicwand" / "dashboard.html").read_text()

    def test_header_has_a_chat_button(self):
        self.assertIn('id="btnChat"', self._html())

    def test_chat_is_a_drawer(self):
        html = self._html()
        self.assertIn('id="chatDrawer"', html)
        self.assertIn("class=\"drawer\"", html)

    def test_chat_is_not_inside_the_dashboard_grid(self):
        html = self._html()
        main = html.split("<main>")[1].split("</main>")[0]
        self.assertNotIn("chatLog", main)
        self.assertNotIn("chatInput", main)

    def test_no_duplicate_element_ids(self):
        import collections
        import re
        ids = re.findall(r'id="([^"]+)"', self._html())
        dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
        self.assertEqual(dupes, [], f"duplicate ids: {dupes}")

    def test_bubbles_align_by_speaker(self):
        html = self._html()
        self.assertIn('class="bubble-row ${mine ? "mine" : "theirs"}"', html)
        self.assertIn(".bubble.mine", html)
        self.assertIn(".bubble.theirs", html)
        self.assertIn("justify-content: flex-end", html)

    def test_new_chat_button_exists(self):
        self.assertIn('id="btnChatNew"', self._html())

    def test_markdown_is_escaped_before_tagging(self):
        """Rendering must escape first, then add only our own tags."""
        html = self._html()
        idx_esc = html.index("let t = esc(raw);")
        idx_bold = html.index(r"\*\*([^*]+)\*\*")
        self.assertLess(idx_esc, idx_bold)

    def test_escape_and_shortcut_are_wired(self):
        html = self._html()
        self.assertIn('"Escape"', html)
        self.assertIn("toggleChat", html)

    def test_modal_is_height_bounded(self):
        """The settings modal must never be taller than the viewport again."""
        html = self._html()
        self.assertIn("max-height: min(92vh", html)
        self.assertIn("overflow-y: auto", html)


class TestNewConversation(unittest.TestCase):
    """A new chat must ARCHIVE, not destroy: the conversation is evidence."""

    def test_new_archives_and_empties(self):
        with tempfile.TemporaryDirectory() as d:
            chat.save_history(d, [
                {"role": "user", "text": "why veto?", "ts": "t1"},
                {"role": "assistant", "text": "because news", "ts": "t2"}])
            out = chat.start_new(d)
            self.assertTrue(out["ok"])
            self.assertEqual(out["archived"], 2)
            self.assertEqual(chat.load_history(d), [])
            self.assertEqual(chat.archived_count(d), 2)

    def test_archive_is_append_only_across_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                chat.save_history(d, [{"role": "user", "text": f"q{i}"}])
                chat.start_new(d)
            self.assertEqual(chat.archived_count(d), 3)

    def test_new_on_empty_history_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            out = chat.start_new(d)
            self.assertTrue(out["ok"])
            self.assertEqual(out["archived"], 0)

    def test_archive_survives_a_clear(self):
        with tempfile.TemporaryDirectory() as d:
            chat.save_history(d, [{"role": "user", "text": "keep me"}])
            chat.start_new(d)
            chat.clear_history(d)
            self.assertEqual(chat.archived_count(d), 1)


class TestProposalDiagnostics(unittest.TestCase):
    """'Nothing passed the gate' is unhelpful alone; the funnel explains why.

    First real data: 141 pitched, 7 reached the gate, 87 identical reruns. That
    is a proposer problem, not a strategy problem — and it was invisible.
    """

    def _ledger_with(self, d, outcomes):
        from magicwand.ledger import Ledger
        led = Ledger(Path(d) / "l.jsonl")
        for kind in outcomes:
            if kind == "passed":
                led.append("proposal", {"ok": True, "spec": {"name": "s", "family": "breakout"}})
            elif kind == "identical":
                led.append("proposal", {"ok": False, "summary": "refused: identical to a previous attempt (x)"})
            elif kind == "near":
                led.append("proposal", {"ok": False, "summary": "refused: near-duplicate of x"})
            elif kind == "gate":
                led.append("proposal", {"ok": False,
                                        "summary": "REJECTED: not profitable",
                                        "gates": {"profitable": {"ok": False},
                                                  "costs_2x": {"ok": False}}})
            else:
                led.append("proposal", {"ok": False, "summary": "proposal contained no JSON object"})
        return led

    def test_counts_each_outcome(self):
        from magicwand.dashboard import _proposal_stats
        with tempfile.TemporaryDirectory() as d:
            led = self._ledger_with(d, ["passed", "identical", "identical", "near",
                                        "gate", "malformed"])
            st = _proposal_stats(led)
            self.assertEqual(st["total"], 6)
            self.assertEqual(st["counts"]["passed"], 1)
            self.assertEqual(st["counts"]["identical"], 2)
            self.assertEqual(st["counts"]["near-duplicate"], 1)
            self.assertEqual(st["counts"]["failed gate"], 1)
            self.assertEqual(st["validated"], 1)

    def test_failing_gates_are_ranked(self):
        from magicwand.dashboard import _proposal_stats
        with tempfile.TemporaryDirectory() as d:
            led = self._ledger_with(d, ["gate", "gate"])
            st = _proposal_stats(led)
            self.assertEqual(st["failing_gates"]["profitable"], 2)

    def test_empty_ledger_is_zero_not_error(self):
        from magicwand.dashboard import _proposal_stats
        with tempfile.TemporaryDirectory() as d:
            from magicwand.ledger import Ledger
            st = _proposal_stats(Ledger(Path(d) / "l.jsonl"))
            self.assertEqual(st["total"], 0)
            self.assertEqual(st["validated"], 0)


class TestExplorationPrompt(unittest.TestCase):
    """The proposer must be told to explore, and shown the ranges to explore in."""

    def test_prompt_demands_variation(self):
        from magicwand import propose
        self.assertIn("EXPLORE", propose._PROPOSAL_SYSTEM)
        self.assertIn("15%", propose._PROPOSAL_SYSTEM)

    def test_catalog_ranges_are_listed_for_the_model(self):
        from magicwand.propose import _catalog_ranges
        text = _catalog_ranges()
        self.assertIn("sma_trend", text)
        self.assertIn("slow 5..400", text)

    def test_sampling_temperature_encourages_novelty(self):
        import inspect
        from magicwand import propose
        self.assertIn("temperature=0.95", inspect.getsource(propose))
