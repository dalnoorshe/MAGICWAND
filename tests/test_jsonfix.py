"""Lenient JSON parsing: repair what is repairable, refuse what is not."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand.agents import parse_verdict
from magicwand.jsonfix import extract_object_text, parse_json_lenient
from magicwand.propose import parse_proposal


class TestRepairs(unittest.TestCase):
    def test_raw_newline_inside_a_string(self):
        """The failure seen in the console: a model writing prose with breaks.

        JSON forbids literal control characters in strings, so json.loads fails
        with 'Unterminated string' / 'Invalid control character'.
        """
        raw = '{"stance":"veto","thesis":"line one\nline two","confidence":0.8}'
        r = parse_json_lenient(raw)
        self.assertTrue(r.ok)
        self.assertTrue(r.repaired)
        self.assertEqual(r.obj["thesis"], "line one\nline two")
        self.assertEqual(r.obj["confidence"], 0.8)

    def test_tab_inside_a_string(self):
        r = parse_json_lenient('{"thesis":"a\tb"}')
        self.assertTrue(r.ok)
        self.assertEqual(r.obj["thesis"], "a\tb")

    def test_trailing_commas(self):
        r = parse_json_lenient('{"a":1,"b":[1,2,],}')
        self.assertTrue(r.ok)
        self.assertEqual(r.obj, {"a": 1, "b": [1, 2]})

    def test_truncated_object_is_closed_and_flagged(self):
        r = parse_json_lenient('{"thesis":"cut off mid sent')
        self.assertTrue(r.ok)
        self.assertTrue(r.repaired)
        self.assertIn("incomplete", r.note)

    def test_fenced_and_prose_wrapped(self):
        self.assertTrue(parse_json_lenient('```json\n{"a":1}\n```').ok)
        self.assertTrue(parse_json_lenient('Here you go: {"a":1} — done').ok)

    def test_valid_json_is_not_marked_repaired(self):
        r = parse_json_lenient('{"a":1}')
        self.assertTrue(r.ok)
        self.assertFalse(r.repaired)

    def test_unrepairable_input_fails_with_context(self):
        for bad in ("", "no json at all", "totally not a response"):
            r = parse_json_lenient(bad)
            self.assertFalse(r.ok, bad)
            self.assertTrue(r.note)

    def test_degenerate_braces_repair_to_an_empty_object(self):
        """`{,}` is syntactically recoverable, so parsing succeeds — but the
        result is an empty object, which schema validation then rejects. Repair
        fixes syntax; it never manufactures a valid verdict."""
        r = parse_json_lenient("{,}")
        self.assertTrue(r.ok)
        self.assertEqual(r.obj, {})
        v = parse_verdict("news", "{,}")
        self.assertFalse(v.trusted)


class TestExtraction(unittest.TestCase):
    def test_picks_the_object_out_of_prose(self):
        self.assertEqual(extract_object_text('x {"a":1} y'), '{"a":1}')

    def test_prefers_fenced_block(self):
        text = 'note {"wrong":1}\n```json\n{"right":2}\n```'
        self.assertIn("right", extract_object_text(text))


class TestVerdictIntegration(unittest.TestCase):
    def test_malformed_verdict_is_repaired_and_reported(self):
        raw = ('{"stance":"veto","confidence":0.77,"evidence_score":-0.5,'
               '"thesis":"two lines\nhere","claims":[],"warnings":[]}')
        v = parse_verdict("news", raw)
        self.assertTrue(v.trusted, v.error)
        self.assertEqual(v.stance, "veto")
        self.assertTrue(any("auto-repaired" in w for w in v.warnings),
                        "a repaired answer must be reported, not hidden")

    def test_unrepairable_verdict_stays_untrusted(self):
        v = parse_verdict("news", "I could not analyse this.")
        self.assertFalse(v.trusted)
        self.assertTrue(v.error)

    def test_proposal_repair(self):
        raw = ('{"name":"x","hypothesis":"a\nb","family":"breakout",'
               '"params":{"slow":20},"symbol":"SOL-USD"}')
        spec = parse_proposal(raw)
        self.assertEqual(spec.family, "breakout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
