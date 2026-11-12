"""Service manager: pidfile handling, stale detection, and stop semantics.

Cross-call persistence cannot be tested inside the agent sandbox (each call
gets its own PID namespace with --die-with-parent), so these tests exercise
the *logic* with processes spawned and inspected within a single test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magicwand import service
from magicwand.service import ServiceState, is_running, pid_alive, read_state, stop


class TestPidHandling(unittest.TestCase):
    def test_pid_alive_true_for_self(self):
        self.assertTrue(pid_alive(os.getpid()))

    def test_pid_alive_false_for_impossible_pid(self):
        self.assertFalse(pid_alive(2_000_000_000))

    def test_not_running_when_no_state(self):
        with tempfile.TemporaryDirectory() as d:
            running, st = is_running(Path(d))
            self.assertFalse(running)
            self.assertEqual(st.pid, 0)

    def test_stale_pidfile_reports_not_running(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = subprocess.Popen([sys.executable, "-c", "pass"])
            dead = p.pid
            p.wait()
            time.sleep(0.1)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "data" / "server.pid").write_text(str(dead))
            (root / "data" / "server.json").write_text(json.dumps(
                {"pid": dead, "port": 1, "url": "http://x", "started_at": "",
                 "log": "", "mode": "offline"}))
            running, st = is_running(root)
            self.assertFalse(running, "a recycled/dead pid must not look running")
            self.assertEqual(st.pid, dead)

    def test_recycled_pid_for_foreign_process_is_not_ours(self):
        """If the recorded pid now belongs to something else, say not running."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(parents=True)
            # Our own test process is alive but is NOT a wand server.
            (root / "data" / "server.pid").write_text(str(os.getpid()))
            (root / "data" / "server.json").write_text(json.dumps(
                {"pid": os.getpid(), "port": 1, "url": "", "started_at": "",
                 "log": "", "mode": "offline"}))
            running, _ = is_running(root)
            self.assertFalse(running)

    def test_read_state_tolerates_garbage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(parents=True)
            (root / "data" / "server.json").write_text("{not json")
            st = read_state(root)
            self.assertEqual(st.pid, 0)

    def test_stop_when_not_running_is_ok(self):
        with tempfile.TemporaryDirectory() as d:
            out = stop(Path(d))
            self.assertTrue(out["ok"])
            self.assertFalse(out["stopped"])


class TestStopSignals(unittest.TestCase):
    def test_stop_terminates_a_live_process(self):
        """Stop must actually stop a process, not just clear a file."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(parents=True)
            # A stand-in "server": long-lived, and its cmdline contains 'wand'
            # so the ownership check accepts it.
            script = root / "fake_wand_server.py"
            script.write_text("import time\ntime.sleep(120)\n")
            p = subprocess.Popen([sys.executable, str(script), "wand"],
                                 stdout=subprocess.DEVNULL)
            try:
                self.assertTrue(pid_alive(p.pid))
                (root / "data" / "server.pid").write_text(str(p.pid))
                (root / "data" / "server.json").write_text(json.dumps(
                    {"pid": p.pid, "port": 1, "url": "", "started_at": "",
                     "log": "", "mode": "offline"}))
                running, _ = is_running(root)
                self.assertTrue(running, "live owned process should look running")
                out = stop(root, timeout=10)
                self.assertTrue(out["ok"], out)
                p.wait(timeout=10)
                self.assertIsNotNone(p.poll())
            finally:
                if p.poll() is None:
                    p.kill()


class TestStateFile(unittest.TestCase):
    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(parents=True)
            st = ServiceState(pid=4242, port=8787, url="http://127.0.0.1:8787",
                              started_at="now", log="/tmp/x", mode="live")
            service._write_state(root, st)
            back = read_state(root)
            self.assertEqual(back.pid, 4242)
            self.assertEqual(back.mode, "live")
            self.assertEqual((root / "data" / "server.pid").read_text().strip(), "4242")


if __name__ == "__main__":
    unittest.main(verbosity=2)
