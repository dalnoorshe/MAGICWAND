#!/usr/bin/env python3
"""MAGICWAND — one command.

    ./magicwand                 start the console and open your browser
    ./magicwand --no-browser    start it, don't open a browser
    ./magicwand --port 9000     use a different port

What it does, in order:
  1. checks this is the right directory and the environment is usable
  2. tells you what is connected (market data always; DeepSeek if a key is set)
  3. starts the autonomous server
  4. opens the console in your browser
  5. keeps running until Ctrl-C

If no DeepSeek key is saved yet, it still starts — the console has a "Connect
API" button, and everything works offline with the stand-in model meanwhile.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Line-buffer output. Python block-buffers stdout when it is a pipe or a file,
# so a launcher that then runs forever shows nothing until it exits — which
# looks exactly like a hang. This bit me once; it is cheap to prevent.
try:
    sys.stdout.reconfigure(line_buffering=True)   # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)   # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def _ok(msg: str) -> str:
    return f"  {GREEN}✓{OFF} {msg}"


def _warn(msg: str) -> str:
    return f"  {YELLOW}!{OFF} {msg}"


def _bad(msg: str) -> str:
    return f"  {RED}✗{OFF} {msg}"


def preflight() -> bool:
    """Checks that matter. Returns False only if we cannot start at all."""
    print(f"\n{BOLD}MAGICWAND{OFF} {DIM}— paper-trading system{OFF}\n")
    ok = True

    if sys.version_info < (3, 10):
        print(_bad(f"Python 3.10+ required (this is {sys.version.split()[0]})"))
        return False
    print(_ok(f"python {sys.version.split()[0]}"))

    if not (ROOT / "magicwand" / "dashboard.py").exists():
        print(_bad(f"this does not look like MAGICWAND: {ROOT}"))
        return False
    print(_ok(f"workspace {ROOT}"))

    try:
        from PIL import Image  # noqa: F401
        print(_ok("Pillow present — the agent can draw and read charts"))
    except ImportError:
        print(_warn("Pillow missing — vision disabled. Fix: pip install pillow"))

    from magicwand.settings_store import SettingsStore, resolve_key
    st = SettingsStore(ROOT / "data" / "settings.json")
    pub = st.public_dict()
    if pub["has_key"]:
        print(_ok(f"DeepSeek key connected ({pub['key_hint']}, "
                  f"from {pub['key_source']}) — live agents + vision"))
    else:
        print(_warn("no DeepSeek key yet — the console will offer to connect one"))
        print(f"    {DIM}you can also: export DEEPSEEK_API_KEY=sk-...{OFF}")

    # Backup health belongs in the preflight: it is the only thing here that
    # cannot be regenerated.
    try:
        from magicwand.storage import backup_age_seconds, list_backups
        age = backup_age_seconds(ROOT)
        n = len(list_backups(ROOT))
        if age is None:
            print(_warn("no backup yet — one will be taken automatically"))
        else:
            hours = age / 3600.0
            when = (f"{hours:.1f}h ago" if hours >= 1 else f"{int(age / 60)}m ago")
            print(_ok(f"backups: {n} kept, last {when} (automatic)"))
    except Exception:  # noqa: BLE001
        print(_warn("backup status unavailable"))

    print(_ok("market data: public feeds, no key needed"))
    print(_warn("broker: none — this build cannot place real orders (paper only)"))
    return ok


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _open_browser_later(url: str, delay: float = 1.5) -> None:
    def go():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a browser that won't open is not fatal
            pass
    threading.Thread(target=go, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="magicwand",
                                 description="Start the MAGICWAND console.")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--interval", type=int, default=0,
                    help="autopilot seconds between ticks (0 = don't autostart)")
    ap.add_argument("--live", action="store_true",
                    help="require the real model; refuse to fall back to offline")
    ap.add_argument("--bg", action="store_true",
                    help="run detached in the background (survives closing this terminal)")
    args = ap.parse_args(argv)

    if not preflight():
        return 1

    from magicwand.service import detect_instances
    # Scan the requested port range AND the standard one. Scanning only from
    # the requested port meant starting on 8788 missed the instance on 8787.
    scan = sorted(set(range(args.port, args.port + 12)) | set(range(8787, 8800)))
    existing = detect_instances(args.host, scan)
    if existing:
        print(_warn(f"MAGICWAND is already running on port(s) {existing}"))
        print(f"    {DIM}Two instances share the same files — don't run both.{OFF}")
        print(f"    {DIM}Stop it:  python3 wand.py service stop{OFF}")
        print(f"    {DIM}Or open:  http://{args.host}:{existing[0]}{OFF}\n")
        return 0

    port = args.port
    for candidate in range(port, port + 12):
        if port_free(args.host, candidate):
            port = candidate
            break
    else:
        print(_bad(f"no free port in {args.port}-{args.port + 11}"))
        return 1
    if port != args.port:
        print(_warn(f"port {args.port} busy, using {port}"))

    from magicwand.config import Config
    from magicwand.dashboard import serve
    from magicwand.settings_store import SettingsStore, resolve_key

    cfg = Config()
    pub = SettingsStore(ROOT / "data" / "settings.json").public_dict()
    if pub["has_key"]:
        cfg.market.symbol = pub.get("symbol") or cfg.market.symbol
    if args.live and not resolve_key(ROOT):
        print(_bad("--live was requested but no DeepSeek key is connected"))
        print(f"    {DIM}start without --live, then use Connect API in the console{OFF}")
        return 2

    url = f"http://{args.host}:{port}"
    interval = args.interval or 900
    autostart = args.interval != 0

    print()
    print(f"  {BOLD}{url}{OFF}")
    if autostart:
        print(f"  {DIM}autopilot: every {interval}s, unattended{OFF}")
    else:
        print(f"  {DIM}autopilot: idle (press Start in the console){OFF}")
    print(f"  {DIM}Ctrl-C to stop{OFF}\n")

    # Detached mode: hand off to the service manager, which setsid()s the
    # server so closing this terminal cannot kill it.
    if args.bg:
        from magicwand import service as svc
        out = svc.start(root=ROOT, port=port, interval=interval,
                        mode="live" if pub["has_key"] else "offline",
                        autostart=(args.interval != 0))
        if not out.get("ok"):
            print(_bad(f"failed to start in background: {out.get('error')}"))
            if out.get("log_tail"):
                print(out["log_tail"])
            return 1
        real_url = out.get("url", url)
        print(_ok(f"running in the background at {real_url}  (pid {out['status']['pid']})"))
        print(f"    {DIM}log:    {out['status']['log']}{OFF}")
        print(f"    {DIM}stop:   python3 wand.py service stop{OFF}")
        print(f"    {DIM}status: python3 wand.py service status{OFF}\n")
        if not args.no_browser:
            webbrowser.open(real_url)
        return 0

    if not args.no_browser:
        _open_browser_later(url)

    try:
        return serve(root=ROOT, host=args.host, port=port, cfg=cfg,
                     offline=not pub["has_key"], autostart=autostart,
                     interval=interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
