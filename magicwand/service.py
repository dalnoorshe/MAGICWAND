"""MAGICWAND — background service management.

The panic this solves: closing the terminal killed the server, so the UI
vanished and it *looked* like data was lost. It never was — all state is on
disk — but a UI that dies with a window is a bad experience and a worse
habit.

`start` detaches the server from the terminal (setsid + nohup), records its pid
and log under data/, and returns immediately. `stop` terminates it politely.
`status` reports whether it is up and, if not, why.

There is no systemd, no container, no daemon library: a pidfile and a detached
child process. It survives closing the terminal, and it does not survive a
reboot — `autostart` installs a user-level systemd unit for that, only if the
user asks for it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ServiceState:
    pid: int = 0
    port: int = 0
    url: str = ""
    started_at: str = ""
    log: str = ""
    mode: str = "offline"


def _data(root: Path) -> Path:
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pidfile(root: Path) -> Path:
    return _data(root) / "server.pid"


def _statefile(root: Path) -> Path:
    return _data(root) / "server.json"


def _logfile(root: Path) -> Path:
    return _data(root) / "server.log"


def read_state(root: Path = ROOT) -> ServiceState:
    p = _statefile(root)
    if not p.exists():
        return ServiceState()
    try:
        d = json.loads(p.read_text())
        return ServiceState(
            pid=int(d.get("pid", 0)), port=int(d.get("port", 0)),
            url=str(d.get("url", "")), started_at=str(d.get("started_at", "")),
            log=str(d.get("log", "")), mode=str(d.get("mode", "offline")))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return ServiceState()


def _write_state(root: Path, st: ServiceState) -> None:
    _statefile(root).write_text(json.dumps(asdict(st), indent=2))
    _pidfile(root).write_text(str(st.pid))


def pid_alive(pid: int) -> bool:
    """True only if the pid exists AND is not a zombie.

    `os.kill(pid, 0)` succeeds for a zombie, so a naive check reports "alive"
    forever if the parent has not reaped the child — and `stop()` then waits
    for a process that is already dead. /proc stat field 3 is the state; 'Z'
    means defunct.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                 # exists, owned by someone else — still check state
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass                 # no /proc (non-Linux) — trust the signal result
    return True


def is_running(root: Path = ROOT) -> tuple[bool, ServiceState]:
    """True only if the recorded pid is alive AND looks like our server."""
    st = read_state(root)
    if not pid_alive(st.pid):
        return False, st
    try:
        cmd = Path(f"/proc/{st.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "ignore")
    except OSError:
        return True, st      # non-Linux: trust the pid
    if "wand" not in cmd and "magicwand" not in cmd:
        return False, st     # pid was recycled by something else
    return True, st


def detect_instances(host: str = "127.0.0.1",
                     ports: range = range(8787, 8800)) -> list[int]:
    """Ports in range where a MAGICWAND console is answering.

    Works for instances started any way — the pidfile cannot, because a
    foreground `./start.sh` never writes one.
    """
    import json as _json
    import urllib.error
    import urllib.request
    found: list[int] = []
    for port in ports:
        try:
            with urllib.request.urlopen(
                    f"http://{host}:{port}/api/health", timeout=0.35) as r:
                data = _json.loads(r.read().decode())
                if isinstance(data, dict) and data.get("app") == "magicwand":
                    found.append(port)
        except (urllib.error.URLError, OSError, ValueError, TypeError):
            continue
    return found


def _wait_for_http(url: str, timeout: float = 20.0) -> bool:
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    return False


def start(root: Path = ROOT, port: int = 8787, interval: int = 900,
          mode: str = "offline", extra: Optional[list[str]] = None,
          autostart: bool = True) -> dict:
    running, st = is_running(root)
    if running:
        return {"ok": False, "error": "already running",
                "status": asdict(st)}

    log = _logfile(root)
    cmd = [sys.executable, str(root / "wand.py"), "serve", "--port", str(port)]
    if autostart:
        # interval <= 0 explicitly means "start with the autopilot idle".
        cmd += ["--auto", "--interval", str(max(60, interval))]
    if mode == "live":
        cmd.append("--live")
    if extra:
        cmd.extend(extra)

    with open(log, "ab") as logfh:
        logfh.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{' '.join(cmd)} ===\n".encode())
        logfh.flush()
        # setsid: detach from this terminal's session so closing the window
        # does not SIGHUP the server. stdin closed, output to the log.
        proc = subprocess.Popen(
            cmd, cwd=str(root), stdout=logfh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)

    url = f"http://127.0.0.1:{port}"
    state = ServiceState(pid=proc.pid, port=port, url=url,
                         started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
                         log=str(log), mode=mode)
    _write_state(root, state)

    if not _wait_for_http(url, timeout=25):
        # It may have chosen a different port, or died. Say which.
        tail = ""
        try:
            tail = log.read_text()[-800:]
        except OSError:
            pass
        return {"ok": False, "error": "server did not come up", "log_tail": tail,
                "status": asdict(state)}
    return {"ok": True, "status": asdict(state), "url": url}


def stop(root: Path = ROOT, timeout: float = 10.0) -> dict:
    st = read_state(root)
    if not pid_alive(st.pid):
        _pidfile(root).unlink(missing_ok=True)
        return {"ok": True, "stopped": False, "reason": "was not running"}
    try:
        os.kill(st.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"ok": True, "stopped": False, "reason": "already gone"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(st.pid):
            _pidfile(root).unlink(missing_ok=True)
            return {"ok": True, "stopped": True, "pid": st.pid}
        time.sleep(0.2)
    # Refuse to SIGKILL silently; report so the operator decides.
    return {"ok": False, "stopped": False,
            "error": f"pid {st.pid} did not exit after {timeout:.0f}s"}


def status(root: Path = ROOT) -> dict:
    running, st = is_running(root)
    out = {"running": running, "pid": st.pid, "url": st.url, "mode": st.mode,
           "started_at": st.started_at, "log": st.log}
    if running and st.url:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{st.url}/api/health", timeout=3) as r:
                out["health"] = json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            out["health_error"] = f"{type(e).__name__}: {e}"
    return out


def tail_log(root: Path = ROOT, lines: int = 30) -> str:
    p = _logfile(root)
    if not p.exists():
        return "(no server log yet)"
    try:
        content = p.read_text(errors="ignore").splitlines()
    except OSError:
        return "(log unreadable)"
    return "\n".join(content[-lines:])


# --------------------------------------------------------------------------
# Optional: survive a reboot (user systemd unit)
# --------------------------------------------------------------------------

UNIT_NAME = "magicwand.service"


def autostart_install(root: Path = ROOT, port: int = 8787,
                      interval: int = 900) -> dict:
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / UNIT_NAME
    unit.write_text(f"""[Unit]
Description=MAGICWAND paper-trading console
After=network-online.target

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={sys.executable} {root / 'wand.py'} serve --port {port} --auto --interval {interval}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
""")
    cmds = [["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", UNIT_NAME]]
    results = []
    for c in cmds:
        try:
            r = subprocess.run(c, capture_output=True, text=True, timeout=30)
            results.append({"cmd": " ".join(c), "rc": r.returncode,
                            "err": (r.stderr or "").strip()[:200]})
        except (OSError, subprocess.TimeoutExpired) as e:
            results.append({"cmd": " ".join(c), "rc": -1, "err": str(e)[:200]})
    ok = all(x["rc"] == 0 for x in results)
    return {"ok": ok, "unit": str(unit), "results": results,
            "note": ("survives reboot and terminal close; stop with "
                     "`systemctl --user stop magicwand`") if ok else
                    "unit written, but enabling it failed — see results"}
