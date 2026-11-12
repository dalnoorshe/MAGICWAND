"""MAGICWAND — autopilot.

The runner's `tick()` is autonomous but *externally* driven: something has to
call it. This module supplies the missing clock, so the system keeps operating
when nobody is watching.

Design notes that matter:

  * A single daemon thread, started inside the dashboard server. There is one
    tick in flight at a time; the same lock file the manual button uses is
    held, so a human clicking "Run one tick" cannot double-trade with the
    scheduler.
  * Failures are caught per tick and recorded. An autopilot that dies silently
    is worse than one that never started.
  * Every scheduled tick is logged with `by: autopilot`, so the ledger always
    distinguishes machine-driven actions from human ones.
  * The interval is deliberately slow by default. Research data moves on hours,
    not seconds, and the strategy's timeframe is 1h. Ticking every 60 seconds
    would burn tokens to re-read the same candles.
  * Stopping is idempotent and flushes state.

The autopilot cannot raise the risk tier. It can only run whatever tick does,
which is already ceilinged by the gatekeeper.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .common import iso, utcnow
from .config import Config


@dataclass
class AutoState:
    running: bool = False
    interval: int = 900
    offline: bool = True
    started_at: str = ""
    last_tick_at: str = ""
    last_result: str = ""
    ticks: int = 0
    errors: int = 0
    next_tick_at: str = ""


class Autopilot:
    """A clock that drives `runner.tick` on a schedule."""

    def __init__(self, root: str | Path, cfg: Config,
                 offline: bool = True, interval: int = 900) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.state = AutoState(interval=int(interval), offline=bool(offline))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._state_path = self.root / "data" / "autopilot.json"
        self._lock = threading.Lock()

    # ---- status ----

    def status(self) -> dict:
        d = asdict(self.state)
        d["next_in_seconds"] = self._seconds_to_next()
        return d

    def _seconds_to_next(self) -> Optional[int]:
        if not self.state.running or not self.state.next_tick_at:
            return None
        try:
            from .common import parse_iso
            delta = (parse_iso(self.state.next_tick_at) - utcnow()).total_seconds()
            return max(0, int(delta))
        except (ValueError, TypeError):
            return None

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self.status(), indent=2))
        except OSError:
            pass

    # ---- control ----

    def start(self, interval: Optional[int] = None,
              offline: Optional[bool] = None) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "error": "autopilot is already running",
                        "status": self.status()}
            if interval is not None:
                self.state.interval = max(30, int(interval))
            if offline is not None:
                self.state.offline = bool(offline)
            self._stop = threading.Event()
            self.state.running = True
            self.state.started_at = iso(utcnow())
            self._thread = threading.Thread(target=self._loop, name="magicwand-autopilot",
                                            daemon=True)
            self._thread.start()
            self._persist()
            return {"ok": True, "status": self.status()}

    def stop(self, join_timeout: float = 5.0) -> dict:
        """Stop the clock and wait for the thread to finish its current tick.

        Joining matters: a caller that tears down the working directory right
        after stop() would otherwise race the thread still writing state files.
        """
        with self._lock:
            self._stop.set()
            self.state.running = False
            self.state.next_tick_at = ""
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        self._persist()
        return {"ok": True, "status": self.status()}

    # ---- the loop ----

    def _loop(self) -> None:
        """Tick now, then every `interval` seconds until stopped.

        The first tick is immediate — a freshly started autopilot must do
        something, not sit idle for a full interval. `Event.wait(timeout)` is
        only used to *pace* subsequent ticks, and it returns early on stop.
        """
        from datetime import timedelta

        self._log("autopilot started")
        try:
            while not self._stop.is_set():
                self._run_one()
                if self._stop.is_set():
                    break
                self.state.next_tick_at = iso(
                    utcnow() + timedelta(seconds=self.state.interval))
                self._persist()
                if self._stop.wait(self.state.interval):
                    break
        except Exception as e:  # noqa: BLE001 - never die silently
            self.state.errors += 1
            self.state.last_result = f"LOOP DIED: {type(e).__name__}: {e}"
            self._log(f"LOOP DIED: {type(e).__name__}: {e}")
            self._persist()
        finally:
            self.state.running = False
            self.state.next_tick_at = ""
            self._log("autopilot stopped")
            self._persist()

    def _log(self, msg: str) -> None:
        line = f"[autopilot {utcnow():%H:%M:%S}] {msg}"
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            p = self.root / "data" / "autopilot.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _run_one(self) -> None:
        from .dashboard import _Lock, run_tick
        from .gatekeeper import RiskTier
        from .reflect import PolicyStore

        try:
            policy = PolicyStore(self.root / "data" / "policy.json").load()
            # The tier is whatever the human set. Autopilot never invents one.
            tier = policy.tier if policy.tier >= RiskTier.PAPER_DESK else RiskTier.RESEARCH
            with _Lock(self.root / "data" / "tick.lock"):
                from .dashboard import _tick_with_client
                rep = _tick_with_client(self.root, self.cfg,
                                        offline=self.state.offline, tier=tier)
            self.state.ticks += 1
            self.state.last_tick_at = iso(utcnow())
            acts = rep.get("actions") or []
            notes = rep.get("notes") or []
            self.state.last_result = " | ".join((acts + notes)[:3]) or "no action"
            self._log(f"tick {self.state.ticks}: "
                      f"{self.state.last_result[:160]}")
        except RuntimeError as e:
            # The manual button holds the lock; that is not an error.
            self.state.last_result = f"skipped: {e}"
        except Exception as e:  # noqa: BLE001 - the autopilot must survive anything
            self.state.errors += 1
            self.state.last_tick_at = iso(utcnow())
            self.state.last_result = f"error: {type(e).__name__}: {e}"
            try:
                from .ledger import Ledger
                Ledger(self.root / self.cfg.ledger.path).append(
                    "autopilot_error",
                    {"error": f"{type(e).__name__}: {e}"[:400]})
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._persist()


# --------------------------------------------------------------------------
# Module-level singleton: the dashboard and the CLI share one autopilot.
# --------------------------------------------------------------------------

_INSTANCE: Optional[Autopilot] = None


def get(root: str | Path, cfg: Config) -> Autopilot:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Autopilot(root, cfg)
    return _INSTANCE
