"""MAGICWAND — local storage housekeeping.

Two jobs:

  1. RETENTION. Research snapshots are written on every tick (each one is a
     full copy of news/social/derivatives data). Left alone they grow without
     bound. They are *evidence*, not state: the decision they supported is
     already in the hash-chained ledger, so old snapshots can be pruned safely.
     What is NEVER pruned: the ledger, the desk, the policy, memory,
     strategies, settings, and the most recent charts.

  2. BACKUP / RESTORE. One archive of the whole data directory, made locally,
     restorable locally. No cloud, no database. If you never run it, nothing is
     uploaded anywhere; if you do, the file is on your disk and yours to move.

Both are deterministic and report exactly what they did — a housekeeping step
that silently deletes things is worse than no housekeeping.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Files that must never be pruned or lost. Anything matching these is kept.
PROTECTED_NAMES = {
    "ledger.jsonl",
    "desk.json",
    "policy.json",
    "memory.json",
    "settings.json",
    "strategies.json",
    "autopilot.json",
    "models.json",
}

PRUNABLE_GLOBS = ("snapshot_*.json",)


@dataclass
class PruneReport:
    removed: list[str] = field(default_factory=list)
    kept: int = 0
    bytes_freed: int = 0
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "removed": len(self.removed),
            "removed_files": self.removed[:50],
            "kept": self.kept,
            "bytes_freed": self.bytes_freed,
            "kb_freed": round(self.bytes_freed / 1024, 1),
            "skipped": self.skipped,
        }


def prune_snapshots(root: str | Path, keep: int = 40,
                    max_age_days: int = 14) -> PruneReport:
    """Delete OLD research snapshots, keeping the newest `keep` and anything
    younger than `max_age_days`. Never touches protected state."""
    data = Path(root) / "data"
    rep = PruneReport()
    if not data.exists():
        return rep

    candidates: list[Path] = []
    for pattern in PRUNABLE_GLOBS:
        candidates.extend(data.glob(pattern))
    # Newest first by mtime.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    cutoff = time.time() - max_age_days * 86400
    for idx, p in enumerate(candidates):
        if p.name in PROTECTED_NAMES:
            rep.skipped.append(p.name)
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        # Keep the newest `keep` regardless of age, and anything recent.
        if idx < keep or st.st_mtime >= cutoff:
            rep.kept += 1
            continue
        try:
            size = st.st_size
            p.unlink()
            rep.removed.append(p.name)
            rep.bytes_freed += size
        except OSError:
            rep.skipped.append(p.name)
    return rep


def prune_charts(root: str | Path, keep: int = 60) -> PruneReport:
    """Keep the most recent chart images. Older ones stay in the ledger as
    claims; the PNG is convenience, not evidence."""
    rep = PruneReport()
    d = Path(root) / "data" / "charts"
    if not d.exists():
        return rep
    files = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        try:
            size = p.stat().st_size
            p.unlink()
            rep.removed.append(p.name)
            rep.bytes_freed += size
        except OSError:
            rep.skipped.append(p.name)
    rep.kept = min(len(files), keep)
    return rep


def housekeep(root: str | Path, keep_snapshots: int = 40,
              keep_charts: int = 60, max_age_days: int = 14) -> dict:
    """Run all retention. Safe to call on every start."""
    root = Path(root)
    snaps = prune_snapshots(root, keep=keep_snapshots, max_age_days=max_age_days)
    charts = prune_charts(root, keep=keep_charts)
    return {
        "snapshots": snaps.to_dict(),
        "charts": charts.to_dict(),
        "total_kb_freed": round((snaps.bytes_freed + charts.bytes_freed) / 1024, 1),
    }


# --------------------------------------------------------------------------
# Backup / restore — local files only
# --------------------------------------------------------------------------

def backup(root: str | Path, out_dir: str | Path | None = None) -> Path:
    """Write one .tar.gz of the data directory. Returns the archive path.

    The archive contains the API key if one is saved. It is created with mode
    0600 and named plainly so it is obvious what it holds — encrypt it yourself
    if you intend to move it anywhere.
    """
    import os
    import stat

    root = Path(root)
    data = root / "data"
    out = Path(out_dir) if out_dir else root / "backups"
    out.mkdir(parents=True, exist_ok=True)
    # Microseconds plus a collision counter: two backups in the same second
    # must never overwrite each other. (They did, before this was fixed —
    # a manual backup right after an automatic one silently replaced it.)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive = out / f"magicwand-backup-{stamp}.tar.gz"
    n = 1
    while archive.exists():
        archive = out / f"magicwand-backup-{stamp}-{n}.tar.gz"
        n += 1

    with tarfile.open(archive, "w:gz") as tar:
        if data.exists():
            for p in sorted(data.rglob("*")):
                if p.is_file() and p.name not in ("server.pid", "tick.lock"):
                    tar.add(p, arcname=str(Path("data") / p.relative_to(data)))
        for name in ("README.md", "pyproject.toml"):
            f = root / name
            if f.exists():
                tar.add(f, arcname=name)

    try:
        os.chmod(archive, stat.S_IRUSR | stat.S_IWUSR)   # 0600: it has a secret
    except OSError:
        pass
    return archive


def restore(root: str | Path, archive: str | Path,
            overwrite: bool = False, allow_running: bool = False) -> dict:
    """Extract a backup into the project.

    Two guards, both learned the hard way:

      * refuse while a console is RUNNING — restoring under a live server means
        it keeps writing while you overwrite, which is how a ledger gets
        corrupted;
      * never destroy what is there — existing data is moved aside to
        data.superseded-<timestamp>/ rather than deleted, so a wrong archive is
        recoverable.
    """
    root = Path(root)
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(archive)

    if not allow_running:
        try:
            from .service import detect_instances
            live = detect_instances()
        except Exception:  # noqa: BLE001
            live = []
        if live:
            return {"ok": False,
                    "error": (f"MAGICWAND is running on port(s) {live}. "
                              f"Stop it first: python3 wand.py service stop "
                              f"(or pass --allow-running if you are certain)"),
                    "running_on": live}

    data = root / "data"
    existing = [p.name for p in data.iterdir()] if data.exists() else []
    if existing and not overwrite:
        return {"ok": False, "error": "data/ is not empty; pass overwrite=True "
                                      "to replace it", "existing": existing[:20]}

    moved_to = ""
    if existing and overwrite:
        # Move aside instead of deleting: a mistaken restore must be undoable.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        aside = root / f"data.superseded-{stamp}"
        try:
            data.rename(aside)
            moved_to = str(aside)
        except OSError as e:
            return {"ok": False, "error": f"could not move data/ aside: {e}"}

    extracted: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse absolute paths or traversal — an archive is untrusted input.
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"refusing unsafe path in archive: {name}")
        tar.extractall(root)
        extracted = [m.name for m in tar.getmembers() if m.isfile()]
    return {"ok": True, "extracted": len(extracted), "root": str(root),
            "previous_data_moved_to": moved_to}


# --------------------------------------------------------------------------
# Automatic backups
# --------------------------------------------------------------------------

BACKUP_DIR = "backups"


def backup_dir(root: str | Path) -> Path:
    return Path(root) / BACKUP_DIR


def list_backups(root: str | Path) -> list[Path]:
    d = backup_dir(root)
    if not d.exists():
        return []
    return sorted(d.glob("magicwand-backup-*.tar.gz"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def latest_backup(root: str | Path) -> Path | None:
    found = list_backups(root)
    return found[0] if found else None


def backup_age_seconds(root: str | Path) -> float | None:
    p = latest_backup(root)
    if p is None:
        return None
    try:
        return max(0.0, time.time() - p.stat().st_mtime)
    except OSError:
        return None


def verify_archive(path: str | Path) -> dict:
    """Read the archive back. A backup nobody has opened is a hope, not a backup."""
    p = Path(path)
    try:
        with tarfile.open(p, "r:gz") as tar:
            names = tar.getnames()
        has_ledger = any(n.endswith("ledger.jsonl") for n in names)
        return {"ok": True, "files": len(names), "has_ledger": has_ledger,
                "kb": round(p.stat().st_size / 1024, 1)}
    except (tarfile.TarError, OSError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def prune_backups(root: str | Path, keep: int = 14) -> PruneReport:
    """Keep the newest `keep` archives; delete older ones."""
    rep = PruneReport()
    files = list_backups(root)
    for p in files[keep:]:
        try:
            size = p.stat().st_size
            p.unlink()
            rep.removed.append(p.name)
            rep.bytes_freed += size
        except OSError:
            rep.skipped.append(p.name)
    rep.kept = min(len(files), keep)
    return rep


def auto_backup(root: str | Path, keep: int = 14,
                verify: bool = True) -> dict:
    """Create, verify, and rotate a backup in one step.

    Safe to call on a schedule. Returns what happened, including why it failed
    — an automatic backup that silently stops working is worse than none,
    because you stop worrying about it.
    """
    root = Path(root)
    try:
        archive = backup(root)
    except (OSError, tarfile.TarError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    info = verify_archive(archive) if verify else {"ok": True}
    if not info.get("ok"):
        return {"ok": False, "error": f"archive failed verification: {info.get('error')}",
                "file": archive.name}
    if verify and not info.get("has_ledger") and (root / "data" / "ledger.jsonl").exists():
        return {"ok": False, "file": archive.name,
                "error": "archive is missing the ledger"}
    pruned = prune_backups(root, keep=keep)
    return {"ok": True, "file": str(archive), "kb": info.get("kb"),
            "files": info.get("files"), "verified": bool(verify),
            "rotated_out": len(pruned.removed), "kept": pruned.kept}


def storage_report(root: str | Path) -> dict:
    """What is on disk and how big it is. For the console and `wand storage`."""
    root = Path(root)
    data = root / "data"
    if not data.exists():
        return {"exists": False, "total_kb": 0, "files": {}}
    groups: dict[str, int] = {}
    total = 0
    for p in data.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size
        total += size
        if p.name in PROTECTED_NAMES:
            key = "state (protected)"
        elif p.suffix == ".jsonl":
            key = "ledger"
        elif p.suffix == ".png":
            key = "charts"
        elif p.name.startswith("snapshot_"):
            key = "snapshots (prunable)"
        elif p.suffix == ".log":
            key = "logs"
        else:
            key = "other"
        groups[key] = groups.get(key, 0) + size
    return {
        "exists": True,
        "total_kb": round(total / 1024, 1),
        "groups_kb": {k: round(v / 1024, 1) for k, v in sorted(groups.items())},
        "protected": sorted(p.name for p in data.iterdir()
                            if p.is_file() and p.name in PROTECTED_NAMES),
        "backups": [p.name for p in list_backups(root)],
        "backup_age_hours": (round(backup_age_seconds(root) / 3600, 2)
                             if backup_age_seconds(root) is not None else None),
    }
