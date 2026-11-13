"""MAGICWAND — local settings.

Where the operator's API key lives, and how everything else finds it.

Design rules, because this file holds a secret:

  * The key is stored in `data/settings.json` with mode 0600 (owner-only) and
    the file is never committed (data/ is gitignored).
  * The key is NEVER sent to the browser. The UI can set it, ask whether one
    exists, and test it — but the GET endpoint returns only a masked hint
    (`sk-...abcd`) so you can recognise which key is loaded.
  * Environment still works (`DEEPSEEK_API_KEY`) and wins if set, so CI and
    shell users are unaffected. The UI is for the person who does not want to
    edit .bashrc.
  * Set via the UI is an explicit human action and is recorded in the ledger as
    `settings_change` — no secret in the payload, only the fact that it changed.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


class SettingsError(RuntimeError):
    pass


@dataclass
class Settings:
    # Non-secret, shown in the UI
    model: str = "deepseek-flash"
    autotick_seconds: int = 900
    subreddit: str = "solana"
    symbol: str = "SOL-USD"
    # Candle feed. EMPTY means "not chosen" — a default here would silently
    # override an explicit config choice (it did: synthetic became coinbase).
    exchange: str = ""
    # Backups: on by default, every 12h, keep the newest 14.
    auto_backup: bool = True
    backup_interval_hours: float = 12.0
    backup_keep: int = 14
    # Secret material — never serialised to the browser
    _deepseek_key: str = field(default="", repr=False)

    def masked_key(self) -> str:
        """A hint that lets you tell WHICH key is loaded without revealing it."""
        k = self.effective_key()
        if not k:
            return ""
        if len(k) <= 12:
            return "•" * len(k)
        return f"{k[:5]}…{k[-4:]}"

    def effective_key(self) -> str:
        """Env wins, so a shell export still beats the stored value."""
        return (os.environ.get("DEEPSEEK_API_KEY") or self._deepseek_key or "").strip()

    def key_source(self) -> str:
        if os.environ.get("DEEPSEEK_API_KEY"):
            return "environment"
        if self._deepseek_key:
            return "saved in console"
        return "none"


class SettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ---- load / save ----

    def load(self) -> Settings:
        if not self.path.exists():
            return Settings()
        try:
            d = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return Settings()
        if not isinstance(d, dict):
            return Settings()
        return Settings(
            model=str(d.get("model", "deepseek-flash")),
            autotick_seconds=int(d.get("autotick_seconds", 900)),
            subreddit=str(d.get("subreddit", "solana")),
            symbol=str(d.get("symbol", "SOL-USD")),
            exchange=str(d.get("exchange", "")),
            auto_backup=bool(d.get("auto_backup", True)),
            backup_interval_hours=float(d.get("backup_interval_hours", 12.0)),
            backup_keep=int(d.get("backup_keep", 14)),
            _deepseek_key=str(d.get("_deepseek_key", "")),
        )

    def save(self, s: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "model": s.model,
            "autotick_seconds": int(s.autotick_seconds),
            "subreddit": s.subreddit,
            "symbol": s.symbol,
            "exchange": s.exchange or "coinbase",   # display default only
            "auto_backup": bool(s.auto_backup),
            "backup_interval_hours": float(s.backup_interval_hours),
            "backup_keep": int(s.backup_keep),
            "_deepseek_key": s._deepseek_key,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=2))
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)   # 0600
        except OSError:
            pass
        tmp.replace(self.path)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    # ---- secret handling ----

    def set_key(self, key: str) -> Settings:
        key = (key or "").strip()
        if key and not key.startswith("sk-"):
            raise SettingsError(
                "that does not look like a DeepSeek key (expected it to start "
                "with 'sk-'). Get one at https://platform.deepseek.com/api_keys")
        if key and len(key) < 20:
            raise SettingsError("key looks too short to be valid")
        s = self.load()
        s._deepseek_key = key
        self.save(s)
        return s

    def clear_key(self) -> Settings:
        s = self.load()
        s._deepseek_key = ""
        self.save(s)
        return s

    def public_dict(self) -> dict:
        """What the browser is allowed to see. No secrets, ever."""
        from .markets import MARKETS
        s = self.load()
        return {
            "model": s.model,
            "autotick_seconds": s.autotick_seconds,
            "subreddit": s.subreddit,
            "symbol": s.symbol,
            "exchange": s.exchange or "coinbase",   # display default only
            "auto_backup": bool(s.auto_backup),
            "backup_interval_hours": float(s.backup_interval_hours),
            "backup_keep": int(s.backup_keep),
            "markets": [{"symbol": m.symbol, "name": m.name, "note": m.note,
                         "exchange_ok": m.supports(s.exchange)}
                        for m in MARKETS.values()],
            "has_key": bool(s.effective_key()),
            "key_hint": s.masked_key(),
            "key_source": s.key_source(),
        }


def models_cache_path(root: str | Path) -> Path:
    return Path(root) / "data" / "models.json"


def load_models_cache(root: str | Path) -> dict:
    """Last successfully fetched model catalogue, so the picker works offline."""
    p = models_cache_path(root)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_models_cache(root: str | Path, models: list[dict], source: str = "api") -> None:
    p = models_cache_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "fetched_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "source": source,
        "models": models,
    }, indent=2))


def resolve_key(root: str | Path) -> str:
    """The key every component should use: environment first, then the console."""
    return SettingsStore(Path(root) / "data" / "settings.json").load().effective_key()
