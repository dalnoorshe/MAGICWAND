"""MAGICWAND — local web dashboard.

A single-file, dependency-free HTTP server (standard library only) that serves
one HTML page and a JSON state endpoint. It reads the same files the bot
writes — ledger, desk state, policy, memory, strategies — so there is exactly
one source of truth.

Binding is 127.0.0.1 by default: this is an operator console, not a public
service. It has no authentication, so do not expose it to a network.

Endpoints:
    GET  /                 the dashboard
    GET  /api/state        everything the page renders
    POST /api/tick         run one autonomous tick (offline or live model)
    POST /api/reflect      re-run reflection and persist it
    POST /api/tier         set the risk tier  {tier: 0..3, confirm: true}
    GET  /api/health       liveness + whether a model key is configured

Concurrency: one tick at a time, guarded by a lock file, so hitting the button
twice cannot double-trade.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .common import iso, parse_iso, utcnow
from .config import Config
from .gatekeeper import TIER_NAMES, Policy, RiskTier
from .ledger import Ledger, scorecard
from .reflect import PolicyStore, reflect_and_persist
from .settings_store import (SettingsError, SettingsStore, load_models_cache,
                             resolve_key, save_models_cache)


# Documented default only. Never invented beyond this: the live catalogue from
# /models is the source of truth, and the picker is built from it.
_FALLBACK_MODEL_IDS = {"deepseek-flash", "deepseek-v4-pro"}


def get_autopilot(root: Path, cfg: Config, offline: bool = True):
    from .autopilot import get as _get
    return _get(root, cfg)


# --------------------------------------------------------------------------
# State assembly
# --------------------------------------------------------------------------

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _equity_series(ledger: Ledger, limit: int = 400) -> list[dict]:
    pts = []
    for e in ledger.entries():
        if e.get("kind") != "mark":
            continue
        p = e.get("payload", {}) or {}
        pts.append({"ts": e.get("ts"), "equity": p.get("equity"),
                    "price": p.get("price")})
    return pts[-limit:]


def _recent(ledger: Ledger, limit: int = 60) -> list[dict]:
    out = []
    for e in ledger.entries()[-limit:]:
        p = e.get("payload", {}) or {}
        out.append({"kind": e.get("kind"), "ts": e.get("ts"),
                    "payload": p})
    return list(reversed(out))


def build_state(root: Path, cfg: Config) -> dict:
    data = root / "data"
    ledger = Ledger(root / cfg.ledger.path)
    ok, msg = ledger.verify()
    desk = _read_json(data / "desk.json", {})
    policy_raw = _read_json(data / "policy.json", {})
    mem = _read_json(data / "memory.json", {})
    strategies = _read_json(data / "strategies.json", [])
    service = _read_json(data / "service.json", {})
    account = _read_json(root / cfg.ledger.state_path, {})

    tier_int = int(policy_raw.get("tier", int(RiskTier.RESEARCH)))
    try:
        tier = RiskTier(tier_int)
    except ValueError:
        tier = RiskTier.RESEARCH
    policy = Policy(
        tier=tier,
        risk_per_trade=float(policy_raw.get("risk_per_trade", 0.01)),
        stop_atr_mult=float(policy_raw.get("stop_atr_mult", 2.0)),
        target_atr_mult=float(policy_raw.get("target_atr_mult", 4.0)),
        trade_frequency_cap=float(policy_raw.get("trade_frequency_cap", 1.0)),
        reasoning=str(policy_raw.get("reasoning", "")),
    )

    autopilot_raw = _read_json(data / "autopilot.json", {})
    last_research = None
    last_proposal = None
    for e in ledger.entries():
        if e.get("kind") == "research":
            last_research = e.get("payload")
        if e.get("kind") == "proposal":
            last_proposal = e.get("payload")

    desk_trades = _recent_trades(desk)
    wins = sum(1 for t in desk_trades if t.get("net", 0) > 0)
    return {
        "server_time": iso(utcnow()),
        "symbol": cfg.market.symbol,
        "timeframe": cfg.market.timeframe,
        "exchange": cfg.market.exchange,
        "account_equity": account.get("equity"),
        "policy": {
            "tier": int(tier),
            "tier_name": TIER_NAMES[tier],
            "effective": policy.effective(),
            "reasoning": policy.reasoning,
        },
        "tiers": {str(int(t)): n for t, n in TIER_NAMES.items()},
        "desk": {
            "cash": desk.get("pot"),
            "position": desk.get("position"),
            "trades": len(desk_trades),
            "win_rate": round(wins / len(desk_trades), 3) if desk_trades else None,
            "autonomy_armed": desk.get("autonomy_armed"),
            "revoked_reason": desk.get("revoked_reason"),
            "trades_today": desk.get("trades_today"),
            "recent_trades": desk_trades[-12:],
        },
        "research": last_research,
        "chart": _latest_chart(data),
        "proposal": last_proposal,
        "reflection": mem.get("last_reflection"),
        "memory": {
            "blacklisted_families": mem.get("blacklisted_families", []),
            "preferred_families": mem.get("preferred_families", []),
            "agent_weights": mem.get("agent_weights", {}),
        },
        "strategies": [{
            "name": (s.get("spec") or {}).get("name"),
            "family": (s.get("spec") or {}).get("family"),
            "hypothesis": (s.get("spec") or {}).get("hypothesis"),
            "metrics": (s.get("validation") or {}).get("metrics", {}),
            "deployed": s.get("deployed"),
            "registered_at": s.get("registered_at"),
        } for s in strategies],
        "ledger": {
            "ok": ok, "message": msg, "entries": len(ledger.entries()),
            "scorecard": scorecard(ledger.entries()),
        },
        "equity_series": _equity_series(ledger),
        "recent": _recent(ledger),
        "service": service,
        "autopilot": autopilot_raw,
        "settings": SettingsStore(data / "settings.json").public_dict(),
        "models": load_models_cache(root),
        "model_mode": ("live" if resolve_key(root) and not service.get("offline")
                       else "offline"),
    }


def _latest_chart(data: Path) -> Optional[dict]:
    d = data / "charts"
    if not d.exists():
        return None
    files = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    f = files[0]
    return {"file": f.name, "kb": round(f.stat().st_size / 1024, 1),
            "at": iso(dt_from_ts(f.stat().st_mtime))}


def dt_from_ts(ts: float):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _recent_trades(desk: dict) -> list[dict]:
    out = []
    for t in desk.get("trades", []) or []:
        try:
            out.append({
                "symbol": t.get("symbol"),
                "net": float(t.get("net", 0.0)),
                "reason": t.get("reason"),
                "entry_ts": t.get("entry_ts"),
                "exit_ts": t.get("exit_ts"),
            })
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

class _Lock:
    """One autonomous tick at a time, across threads and processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        import fcntl
        self._thread_lock.acquire()
        self._fh = self.path.open("w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._thread_lock.release()
            raise RuntimeError("a tick is already running")
        return self

    def __exit__(self, *exc):
        import fcntl
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._thread_lock.release()
        return False


def _tick_with_client(root: Path, cfg: Config, offline: bool = True,
                      tier: Optional[RiskTier] = None) -> dict:
    """Run one tick and return its report dict. Caller must hold the tick lock."""
    from .desk import DeskConfig
    from .llm import DeepSeekClient, fake_canned
    from .runner import tick

    st = SettingsStore(root / "data" / "settings.json")
    settings = st.load()
    use_fake = offline or not resolve_key(root)
    client = DeepSeekClient(
        model=settings.model,
        fake=(fake_canned if use_fake else None),
        max_calls_per_day=60,
        max_cost_per_day_usd=1.0,
        settings_path=str(root / "data" / "settings.json"),
    )
    if not client.available():
        raise RuntimeError("no API key: connect one in Settings, or run offline")

    policy = PolicyStore(root / "data" / "policy.json").load()
    eff_tier = tier if tier is not None else policy.tier
    dcfg = DeskConfig(enabled=True, tier=eff_tier,
                      min_confidence=policy.effective()["min_confidence"])
    rep = tick(cfg, client, dcfg, root=root, offline=offline,
               propose_enabled=True, reflect_enabled=True,
               # Always render the chart; the agent degrades honestly when the
               # model cannot actually see it.
               vision=True)
    return rep.to_dict()


def run_tick(root: Path, cfg: Config, offline: bool = True,
             tier: Optional[RiskTier] = None,
             by: str = "human") -> dict:
    """One tick, triggered by a human (the dashboard button)."""
    try:
        with _Lock(root / "data" / "tick.lock"):
            Ledger(root / cfg.ledger.path).append(
                "tick_request", {"by": by})
            rep = _tick_with_client(root, cfg, offline=offline, tier=tier)
        return {"ok": True, "report": rep}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

PAGE = None  # set below by _load_page


def _load_page() -> str:
    return (_HTML_PAGE)


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")
    cfg: Config = Config()
    offline: bool = True

    def log_message(self, fmt, *args):  # quieter console
        return

    # ---- helpers ----

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str) -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # ---- routes ----

    def _settings_get(self) -> None:
        st = SettingsStore(self.root / "data" / "settings.json")
        self._json(st.public_dict())

    def _settings_post(self) -> None:
        """Save or clear the API key. This is the only way a key enters the app
        without an environment variable, and it is a human action."""
        st = SettingsStore(self.root / "data" / "settings.json")
        b = self._body()
        action = str(b.get("action", "save"))
        if action == "clear":
            st.clear_key()
            Ledger(self.root / self.cfg.ledger.path).append(
                "settings_change", {"key": "deepseek", "action": "cleared",
                                    "by": "human", "via": "dashboard"})
            self._json({"ok": True, "settings": st.public_dict()})
            return
        if action != "save":
            self._json({"ok": False, "error": f"unknown action {action!r}"}, 400)
            return
        # Save fields INDEPENDENTLY. The model picker posts without a key, and
        # an earlier version treated that as "clear the key" — silently wiping
        # the operator's credential when they changed model.
        key = b.get("key")
        if key is not None and str(key).strip():
            try:
                st.set_key(str(key))
            except SettingsError as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
        cur = st.load()
        if b.get("model"):
            model = str(b["model"])
            if not self._model_is_known(self.root, model):
                self._json({"ok": False,
                            "error": f"model {model!r} is not in the fetched "
                                     f"catalogue; press 'Fetch models' first"}, 400)
                return
            cur.model = model
        if b.get("autotick_seconds"):
            cur.autotick_seconds = max(60, int(b["autotick_seconds"]))
        st.save(cur)
        Ledger(self.root / self.cfg.ledger.path).append(
            "settings_change", {"key": "deepseek", "action": "saved",
                                "hint": cur.masked_key(), "model": cur.model,
                                "by": "human", "via": "dashboard"})
        self._json({"ok": True, "settings": st.public_dict()})

    def _model_is_known(self, root: Path, model: str) -> bool:
        """Only allow models the API actually reported (or the built-in default)."""
        cache = load_models_cache(root)
        ids = {m.get("id") for m in (cache.get("models") or []) if isinstance(m, dict)}
        return model in ids or model in _FALLBACK_MODEL_IDS

    def _models_fetch(self) -> None:
        """Fetch the real model list for the connected key.

        Never invents a catalogue: on failure it reports the error and returns
        the last cached list (or a documented fallback) labelled as such, so a
        picker can never silently offer a model that does not exist.
        """
        from .llm import DeepSeekClient, LLMError, list_models
        key = resolve_key(self.root)
        st = SettingsStore(self.root / "data" / "settings.json")
        cached = load_models_cache(self.root)
        if not key:
            self._json({"ok": False, "error": "no key connected yet",
                        "models": cached.get("models", []),
                        "source": "cache" if cached else "none"}, 400)
            return
        client = DeepSeekClient(model=st.load().model,
                                settings_path=str(self.root / "data" / "settings.json"))
        try:
            models = list_models(client)
            save_models_cache(self.root, models, source="api")
            self._json({"ok": True, "models": models, "source": "api",
                        "count": len(models)})
        except LLMError as e:
            self._json({"ok": False, "error": str(e)[:300],
                        "models": cached.get("models", []),
                        "source": "cache" if cached else "none"}, 400)

    def _test_key(self) -> None:
        """Make one tiny real call so the operator knows the key works."""
        key = resolve_key(self.root)
        if not key:
            self._json({"ok": False, "error": "no key saved yet"}, 400)
            return
        from .llm import DeepSeekClient, LLMError
        st = SettingsStore(self.root / "data" / "settings.json")
        client = DeepSeekClient(model=st.load().model,
                                settings_path=str(self.root / "data" / "settings.json"))
        try:
            out = client.chat(system="Reply with the single word: ok",
                              user="ping", max_tokens=8, json_mode=False,
                              retries=0)
            payload = {"ok": True, "reply": (out or "").strip()[:60],
                       "model": client.model,
                       "cost_usd": round(client.usage.cost_usd, 6)}
            # Cheap opportunity to refresh the model catalogue and balance.
            try:
                from .llm import get_balance, list_models
                models = list_models(client)
                save_models_cache(self.root, models, source="api")
                payload["models"] = models
                payload["balance"] = get_balance(client)
            except LLMError:
                pass
            self._json(payload)
        except LLMError as e:
            self._json({"ok": False, "error": str(e)[:300]}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}, 500)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._html(_load_page())
        elif path == "/api/state":
            self._json(build_state(self.root, self.cfg))
        elif path == "/api/settings":
            self._settings_get()
        elif path == "/api/chart":
            charts = sorted((self.root / "data" / "charts").glob("*.png"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            if not charts:
                self._json({"error": "no chart rendered yet"}, 404)
                return
            body = charts[0].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/storage":
            from .storage import storage_report
            self._json(storage_report(self.root))
        elif path == "/api/health":
            import os
            self._json({"ok": True, "app": "magicwand", "time": iso(utcnow()),
                        "has_key": bool(os.environ.get("DEEPSEEK_API_KEY")),
                        "offline": self.offline})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/tick":
                b = self._body()
                out = run_tick(self.root, self.cfg,
                               offline=bool(b.get("offline", self.offline)))
                self._json(out)
            elif path == "/api/reflect":
                ledger = Ledger(self.root / self.cfg.ledger.path)
                ref = reflect_and_persist(
                    ledger, PolicyStore(self.root / "data" / "policy.json"))
                # Mirror into memory.json: that is what the page reads, and a
                # reflection the UI cannot see is a reflection nobody acts on.
                mem = _read_json(self.root / "data" / "memory.json", {})
                mem["blacklisted_families"] = ref.blacklisted_families
                mem["preferred_families"] = ref.preferred_families
                mem["agent_weights"] = ref.agent_weights
                mem["last_reflection"] = ref.to_dict()
                (self.root / "data" / "memory.json").write_text(
                    json.dumps(mem, indent=2))
                self._json({"ok": True, "reflection": ref.to_dict()})
            elif path == "/api/settings":
                self._settings_post()
            elif path == "/api/storage":
                from .storage import backup, housekeep
                b = self._body()
                action = str(b.get("action", "prune"))
                if action == "backup":
                    p = backup(self.root)
                    Ledger(self.root / self.cfg.ledger.path).append(
                        "backup", {"file": p.name, "kb": round(p.stat().st_size / 1024, 1),
                                   "by": "human", "via": "dashboard"})
                    self._json({"ok": True, "file": str(p),
                                "kb": round(p.stat().st_size / 1024, 1)})
                else:
                    self._json({"ok": True, "result": housekeep(self.root)})
            elif path == "/api/models":
                self._models_fetch()
            elif path == "/api/test-key":
                self._test_key()
            elif path == "/api/autopilot":
                b = self._body()
                ap = get_autopilot(self.root, self.cfg, self.offline)
                action = str(b.get("action", "status"))
                if action == "start":
                    out = ap.start(interval=b.get("interval"),
                                   offline=b.get("offline", self.offline))
                elif action == "stop":
                    out = ap.stop()
                else:
                    out = {"ok": True, "status": ap.status()}
                self._json(out)
            elif path == "/api/tier":
                b = self._body()
                if not b.get("confirm"):
                    self._json({"ok": False, "error": "confirmation required"}, 400)
                    return
                try:
                    tier = RiskTier(int(b.get("tier", 0)))
                except (TypeError, ValueError):
                    self._json({"ok": False, "error": "bad tier"}, 400)
                    return
                store = PolicyStore(self.root / "data" / "policy.json")
                pol = store.load()
                store.save(Policy(tier=tier, risk_per_trade=pol.risk_per_trade,
                                  stop_atr_mult=pol.stop_atr_mult,
                                  target_atr_mult=pol.target_atr_mult,
                                  trade_frequency_cap=pol.trade_frequency_cap,
                                  reasoning=f"tier set by HUMAN via dashboard to "
                                            f"{TIER_NAMES[tier]}"))
                Ledger(self.root / self.cfg.ledger.path).append(
                    "tier_change", {"tier": int(tier), "name": TIER_NAMES[tier],
                                    "by": "human", "via": "dashboard"})
                self._json({"ok": True, "tier": int(tier), "name": TIER_NAMES[tier]})
            else:
                self._json({"error": "not found"}, 404)
        except RuntimeError as e:
            self._json({"ok": False, "error": str(e)}, 409)
        except Exception as e:  # noqa: BLE001 - a dashboard must not 500 silently
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def serve(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8787,
          cfg: Optional[Config] = None, offline: bool = True,
          autostart: bool = False, interval: int = 900) -> int:
    Handler.root = Path(root).resolve()
    Handler.cfg = cfg or Config()
    Handler.offline = offline
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"MAGICWAND dashboard  {url}")
    print(f"  root    {Handler.root}")
    print(f"  symbol  {Handler.cfg.market.symbol} @ {Handler.cfg.market.timeframe}")
    (Handler.root / "data").mkdir(parents=True, exist_ok=True)
    (Handler.root / "data" / "service.json").write_text(json.dumps({
        "mode": "offline" if offline else "live",
        "pid": __import__("os").getpid(),
        "started_at": iso(utcnow()),
        "url": url,
    }, indent=2))
    # Retention on every start: old research snapshots are evidence, not state,
    # and they accumulate one per tick. The ledger (which holds the decisions)
    # is protected and never pruned.
    try:
        from .storage import housekeep
        hk = housekeep(Handler.root)
        freed = hk.get("total_kb_freed", 0)
        if freed:
            print(f"  housekeeping: pruned old snapshots, freed {freed} KB")
    except Exception as e:  # noqa: BLE001 - housekeeping must never block startup
        print(f"  housekeeping skipped: {type(e).__name__}: {e}")
    print(f"  mode    {'offline (fake model)' if offline else 'live DeepSeek'}")
    if autostart:
        ap = get_autopilot(Handler.root, Handler.cfg, offline)
        ap.state.offline = offline
        out = ap.start(interval=interval, offline=offline)
        print(f"  AUTOPILOT {'started' if out.get('ok') else 'failed'}: "
              f"every {interval}s, no human needed")
    else:
        print("  autopilot idle — start it from the console or with --auto")
    print("  Ctrl-C to stop. Local only — no authentication.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0


try:
    with open(Path(__file__).with_name("dashboard.html"), encoding="utf-8") as _f:
        _HTML_PAGE = _f.read()
except OSError:  # packaged install without the template
    _HTML_PAGE = "<h1>MAGICWAND</h1><p>dashboard.html is missing from the package.</p>"
