"""Settings store: the API key must be storable, maskable, and un-leakable."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand.llm import DeepSeekClient
from magicwand.settings_store import Settings, SettingsError, SettingsStore, resolve_key

KEY = "sk-abcdef1234567890abcdef1234567890"


class TestSettings(unittest.TestCase):
    def _store(self, d):
        return SettingsStore(Path(d) / "settings.json")

    def test_save_and_masked_hint(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            pub = st.public_dict()
            self.assertTrue(pub["has_key"])
            self.assertIn("…", pub["key_hint"])
            self.assertNotIn(KEY, json.dumps(pub))

    def test_the_secret_never_appears_in_public_dict(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            blob = json.dumps(st.public_dict())
            self.assertNotIn(KEY, blob)
            self.assertNotIn(KEY[10:30], blob)

    def test_file_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            mode = stat.S_IMODE((Path(d) / "settings.json").stat().st_mode)
            self.assertEqual(mode, 0o600, oct(mode))

    def test_rejects_obvious_non_keys(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            # An empty key is a no-op (use clear_key to remove); it is not an error.
            for bad in ("hello", "sk-", "sk-short", "AKIA123456789"):
                with self.assertRaises(SettingsError, msg=bad):
                    st.set_key(bad)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            st.clear_key()
            self.assertFalse(st.public_dict()["has_key"])

    def test_env_wins_over_saved(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            saved = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = "sk-envkey123456789012345678"
            try:
                s = st.load()
                self.assertEqual(s.key_source(), "environment")
                self.assertEqual(resolve_key(d), "sk-envkey123456789012345678")
            finally:
                if saved is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = saved

    def test_corrupt_file_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text("{not json")
            st = SettingsStore(p)
            self.assertFalse(st.public_dict()["has_key"])

    def test_client_uses_saved_key(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._store(d)
            st.set_key(KEY)
            saved = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                c = DeepSeekClient(settings_path=str(Path(d) / "settings.json"))
                self.assertTrue(c.available())
                self.assertTrue(c.vision_available())
            finally:
                if saved is not None:
                    os.environ["DEEPSEEK_API_KEY"] = saved

    def test_no_key_means_unavailable_not_fake(self):
        with tempfile.TemporaryDirectory() as d:
            saved = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                c = DeepSeekClient(settings_path=str(Path(d) / "settings.json"))
                self.assertFalse(c.available())
                self.assertFalse(c.vision_available())
            finally:
                if saved is not None:
                    os.environ["DEEPSEEK_API_KEY"] = saved

    def test_repr_does_not_leak(self):
        s = Settings(_deepseek_key=KEY)
        self.assertNotIn(KEY, repr(s))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestModelCatalogue(unittest.TestCase):
    """The model list must come from the API, never from a guess."""

    def test_normalises_api_payload(self):
        from magicwand.llm import list_models

        class FakeClient:
            timeout = 5.0
            def _key(self):
                return "sk-fake"

        import magicwand.llm as L
        payload = {"object": "list", "data": [
            {"id": "deepseek-flash", "name": "Flash", "owned_by": "deepseek",
             "context_window": 1000000, "max_output_tokens": 384000,
             "input_modalities": ["text", "image"],
             "effort": {"supported_levels": ["low", "medium", "high"]}},
            {"id": "deepseek-v4-pro", "name": "V4 Pro", "owned_by": "deepseek",
             "input_modalities": ["text"]},
            {"id": "deepseek-v4-flash-vision-exp", "input_modalities": ["text"]},
            {"no_id": True},
        ]}
        orig = L._get_json_auth
        L._get_json_auth = lambda url, key, timeout=20.0: payload
        try:
            models = list_models(FakeClient())
        finally:
            L._get_json_auth = orig
        ids = [m["id"] for m in models]
        self.assertIn("deepseek-flash", ids)
        self.assertNotIn(None, ids)
        self.assertEqual(len(models), 3)      # the no_id entry dropped
        flash = next(m for m in models if m["id"] == "deepseek-flash")
        self.assertTrue(flash["vision"])
        self.assertEqual(flash["context_window"], 1000000)
        self.assertTrue(flash["reasoner"])
        pro = next(m for m in models if m["id"] == "deepseek-v4-pro")
        self.assertFalse(pro["vision"])
        # Vision-capable models sort first, because that is what we want to pick.
        self.assertTrue(models[0]["vision"])

    def test_cache_roundtrip(self):
        from magicwand.settings_store import load_models_cache, save_models_cache
        with tempfile.TemporaryDirectory() as d:
            save_models_cache(d, [{"id": "x", "vision": True}], source="api")
            c = load_models_cache(d)
            self.assertEqual(c["source"], "api")
            self.assertEqual(c["models"][0]["id"], "x")

    def test_unknown_model_rejected_by_dashboard(self):
        from magicwand import dashboard
        h = dashboard.Handler.__new__(dashboard.Handler)   # no socket needed
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(h._model_is_known(Path(d), "made-up-model"))
            self.assertTrue(h._model_is_known(Path(d), "deepseek-flash"))
