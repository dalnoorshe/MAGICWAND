"""MAGICWAND — automatic backups.

A safety net you have to remember to pull is not a safety net. This runs inside
the server and takes a verified, rotated backup on a schedule, so the two things
that cannot be regenerated — the hash-chained ledger of every decision, and your
API key — are never more than a few hours old.

Rules:

  * Verify after writing. A backup nobody has opened is a hope, not a backup.
    Every archive is read back and checked for the ledger before it counts.
  * Rotate. Keep the newest N and delete older ones, so disk use is bounded.
  * Never block trading. The backup takes its own lock; if a tick is running
    the archive is a moment stale, which is fine — the ledger is append-only
    and a partially-written line is discarded on read.
  * Fail loudly. Errors are counted, logged to data/backup_state.json and the
    ledger, and surfaced in the console. Silent backup failure is the worst
    possible outcome, because you stop worrying.
  * Local only. Archives land in backups/ with mode 0600. Nothing is uploaded.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .common import iso, utcnow
from .ledger import Ledger
from .storage import auto_backup, backup_age_seconds, list_backups, latest_backup

CHECK_EVERY_SECONDS = 60


@dataclass
class BackupState:
    enabled: bool = True
    interval_hours: float = 12.0
    keep: int = 14
    runs: int = 0
    errors: int = 0
    last_run: str = ""
    last_ok: bool = False
    last_file: str = ""
    last_error: str = ""
    last_kb: float = 0.0
    next_due: str = ""


class BackupScheduler:
    def __init__(self, root: str | Path, cfg_ledger_path: str = "data/ledger.jsonl",
                 interval_hours: float = 12.0, keep: int = 14,
                 enabled: bool = True) -> None:
        self.root = Path(root)
        self.ledger_path = cfg_ledger_path
        self.state = BackupState(enabled=enabled, interval_hours=interval_hours,
                                 keep=keep)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state_path = self.root / "data" / "backup_state.json"
        self._file_lock = self.root / "data" / "backup.lock"
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            d = json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for key in ("runs", "errors", "last_run", "last_ok", "last_file",
                    "last_error", "last_kb"):
            if key in d:
                setattr(self.state, key, d[key])

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({
                **asdict(self.state),
                "age_hours": self.age_hours(),
                "archives": len(list_backups(self.root)),
                "latest": (latest_backup(self.root).name
                           if latest_backup(self.root) else ""),
                "saved_at": iso(utcnow()),
            }, indent=2))
        except OSError:
            pass

    # ---- status ----

    def _exact_age_hours(self) -> Optional[float]:
        age = backup_age_seconds(self.root)
        return age / 3600.0 if age is not None else None

    def age_hours(self) -> Optional[float]:
        """Display value. The schedule uses the exact age, not this rounding."""
        a = self._exact_age_hours()
        return round(a, 2) if a is not None else None

    def is_due(self) -> bool:
        if not self.state.enabled:
            return False
        age = self._exact_age_hours()
        if age is None:
            return True                     # never backed up
        return age >= self.state.interval_hours

    def status(self) -> dict:
        age = self.age_hours()
        due_in = None
        if age is not None:
            due_in = round(max(0.0, (self.state.interval_hours - age) * 3600))
        return {
            **asdict(self.state),
            "age_hours": age,
            "archives": len(list_backups(self.root)),
            "latest": (latest_backup(self.root).name
                       if latest_backup(self.root) else ""),
            "due": self.is_due(),
            "due_in_seconds": due_in,
            "running": bool(self._thread and self._thread.is_alive()),
        }

    # ---- control ----

    def start(self) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "error": "already running",
                        "status": self.status()}
            if not self.state.enabled:
                return {"ok": False, "error": "automatic backups are disabled",
                        "status": self.status()}
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop,
                                            name="magicwand-backups", daemon=True)
            self._thread.start()
            return {"ok": True, "status": self.status()}

    def stop(self, join_timeout: float = 5.0) -> dict:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)
        return {"ok": True, "status": self.status()}

    def configure(self, enabled: Optional[bool] = None,
                  interval_hours: Optional[float] = None,
                  keep: Optional[int] = None) -> dict:
        if enabled is not None:
            self.state.enabled = bool(enabled)
        if interval_hours is not None:
            self.state.interval_hours = max(0.05, float(interval_hours))
        if keep is not None:
            self.state.keep = max(1, int(keep))
        self._persist()
        return {"ok": True, "status": self.status()}

    # ---- the work ----

    def run_now(self, by: str = "auto") -> dict:
        """Take a backup immediately, regardless of the schedule."""
        try:
            self._file_lock.parent.mkdir(parents=True, exist_ok=True)
            import fcntl
            with self._file_lock.open("w") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return {"ok": False, "error": "another backup is running"}
                out = auto_backup(self.root, keep=self.state.keep)
        except OSError as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        self.state.runs += 1
        self.state.last_run = iso(utcnow())
        if out.get("ok"):
            self.state.last_ok = True
            self.state.last_file = str(out.get("file", ""))
            self.state.last_kb = float(out.get("kb") or 0.0)
            self.state.last_error = ""
        else:
            self.state.last_ok = False
            self.state.errors += 1
            self.state.last_error = str(out.get("error", "unknown"))[:400]
        self._persist()

        try:
            Ledger(self.root / self.ledger_path).append("backup", {
                "ok": out.get("ok"),
                "file": Path(str(out.get("file", ""))).name if out.get("file") else "",
                "kb": out.get("kb"),
                "verified": out.get("verified"),
                "rotated_out": out.get("rotated_out"),
                "by": by,
                "error": self.state.last_error,
            })
        except OSError:
            pass
        return out

    def _loop(self) -> None:
        # Back up soon after boot if the schedule says it is due, then keep
        # checking. The first check is short so a fresh install has a backup.
        first_delay = 5.0
        while not self._stop.is_set():
            if self._stop.wait(first_delay):
                break
            first_delay = CHECK_EVERY_SECONDS
            try:
                if self.is_due():
                    out = self.run_now(by="auto")
                    flag = "ok" if out.get("ok") else f"FAILED: {out.get('error')}"
                    print(f"[backup {utcnow():%H:%M:%S}] {flag}", flush=True)
            except Exception as e:  # noqa: BLE001 - the scheduler must not die
                self.state.errors += 1
                self.state.last_error = f"LOOP: {type(e).__name__}: {e}"
                self.state.last_run = iso(utcnow())
                self._persist()
                print(f"[backup {utcnow():%H:%M:%S}] LOOP ERROR: {e}", flush=True)


# Module-level singleton, shared by the server and the CLI.
_INSTANCE: Optional[BackupScheduler] = None


def get(root: str | Path, interval_hours: float = 12.0,
        keep: int = 14, enabled: bool = True) -> BackupScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = BackupScheduler(root, interval_hours=interval_hours,
                                    keep=keep, enabled=enabled)
    return _INSTANCE
