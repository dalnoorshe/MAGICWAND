"""Streaming chat: tokens arrive incrementally, and failure is still legible."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import chat
from magicwand.llm import DeepSeekClient, LLMError, TruncatedResponse, fake_canned

STATE = {"symbol": "SOL-USD", "model_mode": "live",
         "settings": {"has_key": True}, "policy": {"tier_name": "research"},
         "desk": {}, "ledger": {"message": "chain intact"},
         "research": {"decision": "veto"}}


class TestClientStream(unittest.TestCase):
    def test_fake_client_streams_in_chunks(self):
        c = DeepSeekClient(fake=fake_canned)
        chunks = list(c.chat_stream("s", "u"))
        self.assertGreater(len(chunks), 1, "a stream of one chunk is not a stream")
        self.assertTrue("".join(chunks))

    def test_stream_joins_to_the_same_text(self):
        c = DeepSeekClient(fake=fake_canned)
        joined = "".join(c.chat_stream("s", "u"))
        self.assertIn("{", joined)
        self.assertIn("}", joined)


class TestReplyStream(unittest.TestCase):
    def _client(self):
        chat._CLIENT = DeepSeekClient(fake=fake_canned)
        saved = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "sk-test-000000000000000000"
        return saved

    def tearDown(self):
        chat._CLIENT = None

    def test_yields_deltas_then_done(self):
        saved = self._client()
        try:
            with tempfile.TemporaryDirectory() as d:
                events = list(chat.reply_stream(d, "hello", STATE))
            deltas = [e["delta"] for e in events if "delta" in e]
            finals = [e for e in events if e.get("done")]
            self.assertGreater(len(deltas), 1)
            self.assertEqual(len(finals), 1)
            self.assertIn("cost_usd", finals[0])
        finally:
            if saved is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = saved

    def test_history_is_written_once_at_the_end(self):
        saved = self._client()
        try:
            with tempfile.TemporaryDirectory() as d:
                list(chat.reply_stream(d, "hello", STATE))
                msgs = chat.load_history(d)
            self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
            self.assertTrue(msgs[1].get("streamed"))
        finally:
            if saved is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = saved

    def test_no_key_is_an_error_event_not_an_exception(self):
        chat._CLIENT = None
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as d:
                events = list(chat.reply_stream(d, "hello", STATE))
            self.assertEqual(len(events), 1)
            self.assertIn("error", events[0])
            self.assertIn("Settings", events[0]["error"])
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved

    def test_empty_message_is_an_error_event(self):
        events = list(chat.reply_stream(tempfile.gettempdir(), "  ", STATE))
        self.assertIn("error", events[0])


class TestConsoleUsesTheStream(unittest.TestCase):
    def _html(self) -> str:
        return (Path(__file__).resolve().parents[1]
                / "magicwand" / "dashboard.html").read_text()

    def test_ui_posts_to_the_stream_endpoint(self):
        self.assertIn("/api/chat/stream", self._html())

    def test_ui_reads_the_body_incrementally(self):
        html = self._html()
        self.assertIn("getReader()", html)
        self.assertIn("ev.delta", html)

    def test_ui_has_a_live_bubble(self):
        html = self._html()
        self.assertIn("liveBubble", html)
        self.assertIn("liveText", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
