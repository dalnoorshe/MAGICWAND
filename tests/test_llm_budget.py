"""Token-budget handling: the failure that produced 'empty response' and
'Unterminated string' in the research panel.

Thinking mode is on by default at effort `high`; its chain of thought is
returned in `reasoning_content` but consumes the same max_tokens budget. A small
budget therefore yields empty content (all spent thinking) or JSON cut mid-string.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import magicwand.llm as L
from magicwand.llm import DeepSeekClient, LLMError, TruncatedResponse


class Recorder:
    """Stands in for _post_json: records bodies, returns scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, url, body, key, timeout):
        self.bodies.append(json.loads(json.dumps(body)))
        if not self.replies:
            raise AssertionError("more calls than scripted replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(content="", finish="stop", reasoning=""):
    return {"choices": [{"message": {"content": content,
                                     "reasoning_content": reasoning},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


class TestThinkingControl(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient(api_key="sk-test", max_tokens_ceiling=8192)

    def test_thinking_disabled_by_default(self):
        rec = Recorder([reply('{"ok":true}')])
        orig = L._post_json
        L._post_json = rec
        try:
            self.client.chat("s", "u", json_mode=True)
        finally:
            L._post_json = orig
        body = rec.bodies[0]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertIn("temperature", body)

    def test_thinking_enabled_is_explicit_and_drops_temperature(self):
        c = DeepSeekClient(api_key="sk-test", thinking="enabled")
        rec = Recorder([reply('{"ok":true}')])
        orig = L._post_json
        L._post_json = rec
        try:
            c.chat("s", "u")
        finally:
            L._post_json = orig
        body = rec.bodies[0]
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertNotIn("temperature", body,
                         "temperature has no effect in thinking mode")


class TestTruncationRecovery(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient(api_key="sk-test", max_tokens_ceiling=8192)

    def test_truncated_answer_retries_with_bigger_budget(self):
        rec = Recorder([
            reply('{"stance":"veto","thesis":"the news flow is dom', finish="length",
                  reasoning="x" * 500),
            reply('{"stance":"veto","thesis":"done","confidence":0.8}'),
        ])
        orig = L._post_json
        L._post_json = rec
        try:
            out = self.client.chat("s", "u", max_tokens=1100, json_mode=True)
        finally:
            L._post_json = orig
        self.assertIn("done", out)
        self.assertEqual(len(rec.bodies), 2)
        self.assertGreater(rec.bodies[1]["max_tokens"], rec.bodies[0]["max_tokens"])
        # and it stops thinking, which is what ate the budget
        self.assertEqual(rec.bodies[1]["thinking"], {"type": "disabled"})
        self.assertEqual(self.client.usage.truncations, 1)

    def test_empty_content_with_reasoning_retries(self):
        rec = Recorder([
            reply("", finish="stop", reasoning="y" * 400),
            reply('{"ok":1}'),
        ])
        orig = L._post_json
        L._post_json = rec
        try:
            out = self.client.chat("s", "u", max_tokens=1100)
        finally:
            L._post_json = orig
        self.assertIn("ok", out)
        self.assertEqual(self.client.usage.empty_content, 1)

    def test_reasoning_size_is_tracked_for_diagnosis(self):
        rec = Recorder([reply('{"ok":1}', reasoning="z" * 321)])
        orig = L._post_json
        L._post_json = rec
        try:
            self.client.chat("s", "u")
        finally:
            L._post_json = orig
        self.assertEqual(self.client.usage.reasoning_tokens, 321)

    def test_budget_escalation_is_capped(self):
        c = DeepSeekClient(api_key="sk-test", max_tokens_ceiling=2500)
        rec = Recorder([reply("", finish="length")] * 4)
        orig = L._post_json
        L._post_json = rec
        try:
            with self.assertRaises(LLMError):
                c.chat("s", "u", max_tokens=1000, retries=3)
        finally:
            L._post_json = orig
        self.assertTrue(all(b["max_tokens"] <= 2500 for b in rec.bodies))

    def test_persistent_truncation_fails_loudly(self):
        rec = Recorder([reply("", finish="length")] * 4)
        orig = L._post_json
        L._post_json = rec
        try:
            with self.assertRaises(LLMError) as ctx:
                self.client.chat("s", "u", retries=2)
        finally:
            L._post_json = orig
        self.assertIn("failed after", str(ctx.exception))


class TestReadersUseEnoughBudget(unittest.TestCase):
    def test_agents_request_a_large_budget(self):
        import inspect
        from magicwand import agents, propose, vision_agent
        self.assertIn("max_tokens=4096", inspect.getsource(agents))
        self.assertIn("max_tokens=3000", inspect.getsource(propose))
        self.assertIn("max_tokens=3000", inspect.getsource(vision_agent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
