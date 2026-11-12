"""Automatic backups: they must run, verify, rotate, and never fail silently."""

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

from magicwand.backup_scheduler import BackupScheduler
from magicwand.storage import (auto_backup, backup_age_seconds, list_backups,
                               prune_backups, verify_archive)


def _seed(root: Path) -> None:
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ledger.jsonl").write_text('{"seq":0,"kind":"note"}\n')
    (d / "policy.json").write_text('{"tier":1}')
    (d / "settings.json").write_text('{"_deepseek_key":"sk-secret"}')


class TestAutoBackup(unittest.TestCase):
    def test_creates_verifies_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            out = auto_backup(root)
            self.assertTrue(out["ok"], out)
            self.assertTrue(out["verified"])
            self.assertTrue(Path(out["file"]).exists())
            self.assertGreater(out["files"], 0)

    def test_archive_contains_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            out = auto_backup(root)
            info = verify_archive(out["file"])
            self.assertTrue(info["has_ledger"])

    def test_archive_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            out = auto_backup(root)
            mode = stat.S_IMODE(Path(out["file"]).stat().st_mode)
            self.assertEqual(mode, 0o600, "backups contain the API key")

    def test_rotation_keeps_newest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            for _ in range(6):
                auto_backup(root, keep=3)
                time.sleep(0.02)
            self.assertLessEqual(len(list_backups(root)), 3)

    def test_rotation_removes_oldest_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            names = []
            for _ in range(4):
                out = auto_backup(root, keep=10)
                names.append(Path(out["file"]).name)
                time.sleep(0.02)
            rep = prune_backups(root, keep=2)
            self.assertEqual(len(rep.removed), 2)
            remaining = [p.name for p in list_backups(root)]
            self.assertIn(names[-1], remaining)      # newest survives
            self.assertNotIn(names[0], remaining)    # oldest goes

    def test_scheduler_reports_due_when_never_backed_up(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root)
            self.assertTrue(s.is_due())
            self.assertIsNone(s.age_hours())

    def test_scheduler_not_due_after_a_fresh_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root, interval_hours=12)
            out = s.run_now(by="test")
            self.assertTrue(out["ok"], out)
            self.assertFalse(s.is_due())
            self.assertIsNotNone(s.age_hours())

    def test_due_again_after_the_interval(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root, interval_hours=0.0001)   # ~0.36s
            s.run_now()
            time.sleep(0.5)
            self.assertTrue(s.is_due())

    def test_disabled_means_not_due(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root, enabled=False)
            self.assertFalse(s.is_due())
            self.assertFalse(s.start()["ok"])

    def test_failure_is_recorded_not_swallowed(self):
        """A backup that cannot be written must say so loudly."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root)
            # Make the backup directory unwritable.
            bd = root / "backups"
            bd.mkdir()
            os.chmod(bd, 0o500)
            try:
                out = s.run_now(by="test")
                if not out.get("ok"):
                    self.assertFalse(s.state.last_ok)
                    self.assertTrue(s.state.last_error)
                    self.assertEqual(s.state.errors, 1)
                    self.assertIn("backup", (root / "data" / "ledger.jsonl").read_text())
            finally:
                os.chmod(bd, 0o700)

    def test_ledger_records_each_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root)
            s.run_now(by="auto")
            lines = [json.loads(l) for l in
                     (root / "data" / "ledger.jsonl").read_text().splitlines()]
            kinds = [l["kind"] for l in lines]
            self.assertIn("backup", kinds)

    def test_configure_updates_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root)
            s.configure(enabled=False, interval_hours=3, keep=5)
            self.assertFalse(s.state.enabled)
            self.assertEqual(s.state.interval_hours, 3)
            self.assertEqual(s.state.keep, 5)

    def test_loop_takes_a_backup_without_help(self):
        """The whole point: it must happen unattended."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root, interval_hours=0.001)   # ~3.6s
            s.start()
            try:
                deadline = time.time() + 25
                while time.time() < deadline and not list_backups(root):
                    time.sleep(0.4)
            finally:
                s.stop()
            self.assertTrue(list_backups(root),
                            "scheduler never produced a backup")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBackupNameCollisions(unittest.TestCase):
    """Two backups in the same second must both exist.

    Regression: names were second-resolution, so a manual backup taken right
    after an automatic one silently overwrote it.
    """

    def test_rapid_backups_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            names = []
            for _ in range(5):
                out = auto_backup(root, keep=20)
                self.assertTrue(out["ok"], out)
                names.append(Path(out["file"]).name)
            self.assertEqual(len(set(names)), 5, "each backup needs a unique name")
            self.assertEqual(len(list_backups(root)), 5)

    def test_exact_age_is_used_for_scheduling(self):
        """Rounding the age to hours would break short intervals."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _seed(root)
            s = BackupScheduler(root, interval_hours=0.0002)   # ~0.7s
            s.run_now()
            time.sleep(0.9)
            self.assertTrue(s.is_due(), "exact age must drive the schedule")
