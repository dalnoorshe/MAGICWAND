"""Storage: retention must protect state, and backup must round-trip locally."""

from __future__ import annotations

import json
import os
import stat
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import storage
from magicwand.storage import (PROTECTED_NAMES, backup, housekeep, prune_snapshots,
                               restore, storage_report)

LEGACY = ("ledger.jsonl", "desk.json", "policy.json", "settings.json",
          "memory.json", "strategies.json")


def _mk(root: Path, name: str, content: str = "{}", age_days: float = 0) -> Path:
    p = root / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
    return p


class TestRetention(unittest.TestCase):
    def test_old_snapshots_pruned_beyond_the_keep_window(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(60):
                _mk(root, f"snapshot_X_{i:03d}.json", "x" * 500, age_days=40)
            rep = prune_snapshots(root, keep=40, max_age_days=14)
            self.assertEqual(len(rep.removed), 20)
            self.assertEqual(rep.kept, 40)
            self.assertGreater(rep.bytes_freed, 0)

    def test_recent_snapshots_survive_regardless_of_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(60):
                _mk(root, f"snapshot_R_{i:03d}.json", "x" * 10, age_days=0.01)
            rep = prune_snapshots(root, keep=5, max_age_days=14)
            self.assertEqual(len(rep.removed), 0, "recent evidence must not be pruned")

    def test_protected_state_is_never_touched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in LEGACY:
                _mk(root, name, '{"critical": true}', age_days=400)
            for i in range(80):
                _mk(root, f"snapshot_P_{i:03d}.json", "x", age_days=400)
            prune_snapshots(root, keep=5, max_age_days=1)
            for name in LEGACY:
                p = root / "data" / name
                self.assertTrue(p.exists(), f"{name} was deleted!")
                self.assertIn("critical", p.read_text())

    def test_ledger_is_protected_by_name(self):
        for name in ("ledger.jsonl", "settings.json", "desk.json"):
            self.assertIn(name, PROTECTED_NAMES)

    def test_charts_pruned_by_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data" / "charts").mkdir(parents=True)
            for i in range(10):
                p = root / "data" / "charts" / f"c{i}.png"
                p.write_bytes(b"x" * 100)
                t = time.time() + i
                os.utime(p, (t, t))
            rep = storage.prune_charts(root, keep=4)
            self.assertEqual(len(rep.removed), 6)
            self.assertEqual(len(list((root / "data" / "charts").glob("*.png"))), 4)

    def test_housekeep_reports_totals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(10):
                _mk(root, f"snapshot_H_{i:03d}.json", "x" * 100, age_days=30)
            out = housekeep(root, keep_snapshots=3, max_age_days=1)
            self.assertIn("total_kb_freed", out)
            self.assertGreater(out["total_kb_freed"], 0)


class TestBackup(unittest.TestCase):
    def test_backup_contains_state_and_is_private(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _mk(root, "ledger.jsonl", '{"a":1}\n')
            _mk(root, "policy.json", '{"tier":0}')
            arch = backup(root)
            self.assertTrue(arch.exists())
            mode = stat.S_IMODE(arch.stat().st_mode)
            self.assertEqual(mode, 0o600, "a backup may contain the API key")
            with tarfile.open(arch, "r:gz") as tar:
                names = tar.getnames()
            self.assertTrue(any("ledger.jsonl" in n for n in names))
            self.assertTrue(any("policy.json" in n for n in names))

    def test_restore_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            _mk(src, "ledger.jsonl", '{"x":42}\n')
            arch = backup(src, out_dir=Path(d) / "out")

            dst = Path(d) / "dst"
            dst.mkdir()
            out = restore(dst, arch, allow_running=True)
            self.assertTrue(out["ok"], out)
            self.assertEqual((dst / "data" / "ledger.jsonl").read_text(), '{"x":42}\n')

    def test_restore_refuses_to_clobber_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            _mk(src, "ledger.jsonl", '{"x":1}\n')
            arch = backup(src, out_dir=Path(d) / "out")
            dst = Path(d) / "dst"
            _mk(dst, "ledger.jsonl", '{"live":true}\n')
            out = restore(dst, arch, overwrite=False, allow_running=True)
            self.assertFalse(out["ok"])
            self.assertIn("not empty", out["error"])
            self.assertIn("live", (dst / "data" / "ledger.jsonl").read_text())

    def test_restore_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            evil = Path(d) / "evil.tar.gz"
            with tarfile.open(evil, "w:gz") as tar:
                info = tarfile.TarInfo("../../etc/evil.conf")
                payload = b"pwned"
                info.size = len(payload)
                import io
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                restore(Path(d) / "dst", evil, overwrite=True, allow_running=True)

    def test_restore_moves_old_data_aside_instead_of_deleting(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            _mk(src, "ledger.jsonl", '{"new":1}\n')
            arch = backup(src, out_dir=Path(d) / "out")
            dst = Path(d) / "dst"
            _mk(dst, "ledger.jsonl", '{"old":1}\n')
            out = restore(dst, arch, overwrite=True, allow_running=True)
            self.assertTrue(out["ok"], out)
            self.assertTrue(out["previous_data_moved_to"])
            aside = Path(out["previous_data_moved_to"])
            self.assertTrue(aside.exists())
            self.assertIn("old", (aside / "ledger.jsonl").read_text())
            self.assertIn("new", (dst / "data" / "ledger.jsonl").read_text())

    def test_storage_report_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _mk(root, "ledger.jsonl", "x" * 100)
            _mk(root, "snapshot_A.json", "x" * 50)
            rep = storage_report(root)
            self.assertTrue(rep["exists"])
            self.assertIn("snapshots (prunable)", rep["groups_kb"])
            # The ledger is PROTECTED state, not a separate prunable group.
            self.assertIn("state (protected)", rep["groups_kb"])
            self.assertIn("ledger.jsonl", rep["protected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
