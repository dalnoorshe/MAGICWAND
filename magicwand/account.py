"""MAGICWAND — one source of truth for the account balance.

The question this answers: "when we connect a live account, will it fetch and
count the real balance?"

Today there is no live adapter, so the answer is "paper numbers". But the
resolution order is now explicit and single-sourced, so when an adapter arrives
it plugs in at exactly one place:

    1. broker      a connected live/paper broker's own reported equity
    2. paper       data/state.json (the paper parent account)
    3. config      cfg.risk.equity (the configured starting balance)

The caller is told WHICH source was used and whether a corrupted file forced a
fallback, because a silent fallback to a default balance is how a system
believes it has $50 when it has $812.

A broker adapter is any object with:
    connected: bool
    fetch_equity() -> float          # account value in quote currency
    name: str                        # "alpaca-paper", "mt5-demo", ...
If `fetch_equity` raises, the balance is NOT silently replaced — the failure is
reported and the desk refuses to size trades on a number it cannot trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config


class AccountError(RuntimeError):
    pass


@dataclass
class AccountSnapshot:
    equity: float
    source: str          # broker | paper | config
    detail: str = ""
    stale: bool = False  # the number came from a fallback, not a live read

    def to_dict(self) -> dict:
        return {"equity": self.equity, "source": self.source,
                "detail": self.detail, "stale": self.stale}


def _paper_equity(root: Path) -> tuple[Optional[float], str]:
    p = root / "data" / "state.json"
    if not p.exists():
        return None, ""
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return None, f"state.json unreadable ({type(e).__name__}) — using config"
    try:
        eq = float(d.get("equity"))
    except (TypeError, ValueError):
        return None, "state.json has no usable equity — using config"
    if not (eq > 0) or eq != eq or eq == float("inf"):
        return None, "state.json equity is nonsensical — using config"
    return eq, ""


def resolve_account(root: str | Path, cfg: Config,
                    broker: object | None = None) -> AccountSnapshot:
    """Best available account value, with provenance.

    Raises AccountError only when a CONNECTED broker fails to report a balance:
    trading on an assumed number while connected to a real account is the one
    case where guessing is unacceptable.
    """
    root = Path(root)

    if broker is not None and getattr(broker, "connected", False):
        name = getattr(broker, "name", "broker")
        try:
            eq = float(broker.fetch_equity())          # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            raise AccountError(
                f"{name} is connected but did not report a balance "
                f"({type(e).__name__}: {e}); refusing to size trades on an "
                f"assumed number") from e
        if not (eq > 0):
            raise AccountError(f"{name} reported a non-positive balance ({eq})")
        return AccountSnapshot(eq, "broker", f"live balance from {name}")

    paper, problem = _paper_equity(root)
    if paper is not None:
        return AccountSnapshot(paper, "paper", problem or "paper account state")

    return AccountSnapshot(float(cfg.risk.equity), "config",
                           problem or "configured starting balance",
                           stale=bool(problem))
