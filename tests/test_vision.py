"""Vision tests: rendering, verdict discipline, and image-based injection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import data as data_mod
from magicwand.agents import synthesize
from magicwand.llm import DeepSeekClient, fake_canned
from magicwand.vision import Image, VisionError, load, render_chart, save
from magicwand.vision_agent import parse_chart_verdict, read_chart


class TestRendering(unittest.TestCase):
    def test_renders_png_with_expected_shape(self):
        bars = data_mod.synthetic_bars(240, seed=3)
        img = render_chart(bars)
        self.assertEqual(img.mime, "image/png")
        self.assertTrue(img.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(img.source, "rendered-chart")
        self.assertGreater(img.kb, 5)
        self.assertEqual((img.width, img.height), (1280, 720))

    def test_empty_bars_raises(self):
        with self.assertRaises(VisionError):
            render_chart([])

    def test_data_url_is_well_formed(self):
        img = render_chart(data_mod.synthetic_bars(80, seed=1))
        url = img.data_url
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertGreater(len(url), 1000)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            img = render_chart(data_mod.synthetic_bars(80, seed=2))
            p = save(img, d, name="c")
            self.assertTrue(p.exists())
            back = load(p)
            self.assertEqual(back.data, img.data)

    def test_render_is_deterministic(self):
        bars = data_mod.synthetic_bars(150, seed=9)
        a = render_chart(bars).data
        b = render_chart(bars).data
        self.assertEqual(a, b, "same bars must produce the same image")


class TestChartVerdict(unittest.TestCase):
    def _raw(self, **kw):
        base = {"stance": "neutral", "confidence": 0.5, "evidence_score": 0.0,
                "thesis": "range", "observations": ["flat trend"],
                "claims": [], "warnings": [], "injected_text_seen": False}
        base.update(kw)
        return json.dumps(base)

    def test_valid_verdict(self):
        v = parse_chart_verdict(self._raw(stance="veto", confidence=0.9,
                                         evidence_score=-0.8))
        self.assertTrue(v.trusted)
        self.assertEqual(v.observations, ["flat trend"])

    def test_malformed_is_degraded(self):
        v = parse_chart_verdict("I cannot see any image")
        self.assertFalse(v.trusted)
        self.assertTrue(v.error)

    def test_injected_text_is_flagged(self):
        v = parse_chart_verdict(self._raw(injected_text_seen=True,
                                         warnings=["BUY NOW"]))
        self.assertTrue(v.trusted)          # the verdict is still recorded
        self.assertTrue(v.injected_text_seen)
        self.assertTrue(any("INSTRUCTIONS" in w for w in v.warnings))

    def test_image_verdict_cannot_override_veto_rules(self):
        """An injection-flagged 'support' must not become a trade signal."""
        v = parse_chart_verdict(self._raw(stance="support", confidence=1.0,
                                         evidence_score=1.0,
                                         injected_text_seen=True))
        rep = synthesize([v], "SOL-USD", "t")
        self.assertEqual(rep.decision, "support")   # recorded, never actionable
        self.assertFalse(rep.is_veto)


class TestOfflineHonesty(unittest.TestCase):
    def test_fake_model_never_pretends_to_see(self):
        bars = data_mod.synthetic_bars(120, seed=5)
        client = DeepSeekClient(fake=fake_canned)
        with tempfile.TemporaryDirectory() as d:
            v = read_chart(bars, client, title="X", symbol="SOL-USD", save_dir=d)
            self.assertFalse(v.trusted)
            self.assertIn("not inspected", v.error)
            # The chart is still rendered and saved (local + free, and useful to
            # a human operator), but it is attached to a DEGRADED verdict — the
            # model never claims to have read it.
            self.assertTrue(v.image_path)
            self.assertTrue(Path(v.image_path).exists())

    def test_vision_requires_real_api(self):
        self.assertFalse(DeepSeekClient(fake=fake_canned).vision_available())
        self.assertFalse(DeepSeekClient(api_key="").vision_available())

    def test_saved_chart_is_recorded_as_provenance(self):
        """When a real model is used, the image path must be attached."""
        bars = data_mod.synthetic_bars(120, seed=6)

        class FakeVision(DeepSeekClient):
            def vision_available(self):   # pretend we have a real API
                return True

            def chat_vision(self, **kw):
                return json.dumps({
                    "stance": "veto", "confidence": 0.9, "evidence_score": -0.7,
                    "thesis": "breakdown", "observations": ["lower highs"],
                    "claims": [], "warnings": [], "injected_text_seen": False})

        client = FakeVision(fake=None, api_key="x")
        with tempfile.TemporaryDirectory() as d:
            v = read_chart(bars, client, title="X", symbol="SOL-USD", save_dir=d)
            self.assertTrue(v.trusted, v.error)
            self.assertTrue(v.image_path)
            self.assertTrue(Path(v.image_path).exists())
            self.assertEqual(v.image_source, "rendered-chart")


if __name__ == "__main__":
    unittest.main(verbosity=2)
