"""MAGICWAND — the ledger.

Every decision the wand makes is appended here, hash-chained so that a
later edit to history is detectable. This is the part of the project that
makes it different from a screenshot: the record cannot be quietly rewritten.

Event types:
  claim     — the wand stated an expectation (e.g. "this trade should make +1.2%")
  fill      — an order executed (paper)
  veto      — the risk engine refused or shrank a trade, and why
  grade     — a claim was scored against what actually happened
  mark      — periodic equity snapshot
  note      — operator or system message
  kill      — operator stopped the bot

The scorecard reads the ledger and answers one question: when the wand
said something would happen, how often was it right, and by how much did
it flatter itself?
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .common import iso, utcnow


GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = GENESIS
        self._seq = 0
        if self.path.exists():
            self._load_tail()

    # ---- chain state ----

    def _load_tail(self) -> None:
        last: Optional[dict] = None
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        if last:
            self._last_hash = str(last.get("hash", GENESIS))
            self._seq = int(last.get("seq", 0)) + 1

    @staticmethod
    def _hash(seq: int, prev: str, ts: str, kind: str, payload: dict) -> str:
        blob = json.dumps(
            {"seq": seq, "prev": prev, "ts": ts, "kind": kind, "payload": payload},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- writing ----

    def append(self, kind: str, payload: dict, when: Optional[datetime] = None) -> dict:
        """Append one entry, keeping the chain valid under concurrency.

        Sequence state lives in memory, so two Ledger objects on the same file
        will otherwise write duplicate sequence numbers and break the chain.
        That happened here (several components open the ledger within one
        tick). We take an exclusive file lock and resync the tail from disk
        before choosing the next sequence number, so the chain is authoritative
        on disk rather than in any process's memory.
        """
        ts = iso(when or utcnow())
        with self.path.open("a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                seq, prev = self._resync_from(f)
                h = self._hash(seq, prev, ts, kind, payload)
                entry = {"seq": seq, "prev": prev, "ts": ts, "kind": kind,
                         "payload": payload, "hash": h}
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())  # survive a crash mid-run
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._last_hash = h
        self._seq = seq + 1
        return entry

    def _resync_from(self, f) -> tuple[int, str]:
        """Read the true tail of the file and return (next_seq, last_hash)."""
        last: Optional[dict] = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
        if not last:
            return 0, GENESIS
        return int(last.get("seq", 0)) + 1, str(last.get("hash", GENESIS))

    # ---- reading ----

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, message)."""
        prev = GENESIS
        for expected_seq, e in enumerate(self.entries()):
            if e.get("seq") != expected_seq:
                return False, f"sequence gap at entry {expected_seq}"
            if e.get("prev") != prev:
                return False, f"broken chain link at entry {expected_seq}"
            want = self._hash(e["seq"], e["prev"], e["ts"], e["kind"], e["payload"])
            if want != e.get("hash"):
                return False, f"hash mismatch at entry {expected_seq} (tampered?)"
            prev = e["hash"]
        return True, f"chain intact: {expected_seq + 1 if self.entries() else 0} entries"


# --------------------------------------------------------------------------
# Claims and their grades — the wand keeping itself honest
# --------------------------------------------------------------------------

def make_claim(trade_id: str, symbol: str, expected_move: float,
               entry_price: float, horizon_bars: int, reason: str,
               when: Optional[datetime] = None) -> dict:
    return {
        "claim_id": trade_id,
        "symbol": symbol,
        "claim": "price will rise" if expected_move > 0 else "price will fall",
        "expected_move": round(float(expected_move), 6),
        "entry_price": float(entry_price),
        "horizon_bars": int(horizon_bars),
        "stated_reason": reason,
        "ts": iso(when or utcnow()),
    }


def grade_claim(claim: dict, realized_move: float, bars_taken: int,
                exit_reason: str) -> dict:
    """Score a claim. Honest, and unflattering by construction."""
    expected = float(claim.get("expected_move", 0.0))
    direction_ok = (expected > 0) == (realized_move > 0)
    magnitude_ratio = (abs(realized_move) / abs(expected)) if expected else 0.0
    # Flattery detection: claims that are much bigger than reality are noise.
    if not direction_ok:
        verdict = "wrong"
    elif magnitude_ratio >= 0.5:
        verdict = "right"
    elif realized_move == 0:
        verdict = "flat"
    else:
        verdict = "overstated"
    return {
        "claim_id": claim.get("claim_id"),
        "expected_move": round(expected, 6),
        "realized_move": round(float(realized_move), 6),
        "direction_ok": direction_ok,
        "magnitude_ratio": round(magnitude_ratio, 3),
        "verdict": verdict,
        "bars_taken": int(bars_taken),
        "exit_reason": exit_reason,
        "ts": iso(utcnow()),
    }


def scorecard(entries: list[dict]) -> dict:
    grades = [e["payload"] for e in entries if e.get("kind") == "grade"]
    claims = [e["payload"] for e in entries if e.get("kind") == "claim"]
    vetoes = [e["payload"] for e in entries if e.get("kind") == "veto"]
    if not grades:
        return {
            "claims": len(claims), "grades": 0,
            "verdict": "no claims have been graded yet — nothing to brag about, "
                       "and nothing to condemn",
        }
    right = sum(1 for g in grades if g["verdict"] == "right")
    wrong = sum(1 for g in grades if g["verdict"] == "wrong")
    overstated = sum(1 for g in grades if g["verdict"] == "overstated")
    flat = sum(1 for g in grades if g["verdict"] == "flat")
    ratios = [g["magnitude_ratio"] for g in grades if g["magnitude_ratio"]]
    return {
        "claims": len(claims),
        "grades": len(grades),
        "right": right,
        "wrong": wrong,
        "overstated": overstated,
        "flat": flat,
        "hit_rate": right / len(grades),
        "avg_magnitude_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
        "vetoes": len(vetoes),
        "verdict": (
            f"{right}/{len(grades)} directional claims held up "
            f"({right / len(grades):.0%}). Average realized move was "
            f"{(sum(ratios) / len(ratios)):.0%} of what was claimed."
        ) if ratios else f"{right}/{len(grades)} directional claims held up.",
    }
