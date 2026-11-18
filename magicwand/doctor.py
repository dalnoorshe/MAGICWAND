"""MAGICWAND — the sources doctor.

Answers one question, honestly: **is the data actually arriving?**

Every reader depends on public HTTP endpoints, and the failure that hurts most
is the quiet one — a source that stopped answering weeks ago and now renders as
a permanent red card nobody investigates. This module probes each one and
reports what came back, in the source's own words.

    python3 wand.py sources
    python3 wand.py sources --json

It is also the first thing to run when a panel looks wrong, because it
distinguishes "our code is broken" from "the internet said no". Reddit, for
example, answers `429` or `403 blocked by network security` to whole networks;
no amount of parsing fixes that, and the doctor says so instead of letting it
look like a bug.

Nothing here trades, writes to the ledger, or spends model tokens. The only
cost is a handful of HTTP requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import research as research_mod
from .common import iso, utcnow
from .config import Config
from .markets import base_symbol, subreddit_for

# A probe returns (ok, what happened). Raising is also fine: a raise is a
# failure with a stack trace, and the doctor reports it rather than crashing.
ProbeFn = Callable[[], "tuple[bool, str]"]


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    ms: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail,
                "ms": self.ms}


def _run(name: str, fn: ProbeFn) -> Result:
    started = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 — a probe failing is the point
        detail = f"{type(e).__name__}: {e}"
        ok = False
    return Result(name, bool(ok), str(detail)[:300],
                  int((time.perf_counter() - started) * 1000))


# --------------------------------------------------------------------------
# The probes
# --------------------------------------------------------------------------

def probe_reddit(symbol: str) -> ProbeFn:
    def go() -> tuple[bool, str]:
        sub = subreddit_for(symbol)
        # A diagnostic must actually ask, even if the runtime is in cooldown...
        saved = research_mod.reddit_cooldown_until()
        research_mod.reset_reddit_cooldown()
        entries: list = []
        error = ""
        try:
            entries, error = research_mod._reddit_entries(sub)  # noqa: SLF001
        finally:
            # ...but if it failed again, leave the runtime's backoff as it was:
            # running a diagnostic should not cost us another wasted request
            # on every tick.
            if not entries and saved is not None:
                research_mod._REDDIT_DOWN_UNTIL = saved  # noqa: SLF001
        if entries:
            return True, f"r/{sub}: {len(entries)} posts"
        return False, f"r/{sub}: {error or 'no entries'}"
    return go


def probe_attention(symbol: str) -> ProbeFn:
    def go() -> tuple[bool, str]:
        base = base_symbol(symbol)
        got = research_mod._crowd_attention(base)  # noqa: SLF001
        if not got:
            return False, "CoinGecko trending returned nothing"
        if got.get("trending"):
            return True, (f"{base} trending at rank {got.get('trending_rank')} "
                          f"of {got.get('trending_out_of')}")
        leaders = ", ".join(got.get("trending_top", [])[:5])
        return True, (f"{base} not in trending top {got.get('trending_out_of')} "
                      f"(led by {leaders})")
    return go


def probe_mood() -> ProbeFn:
    def go() -> tuple[bool, str]:
        got = research_mod._crowd_mood()  # noqa: SLF001
        if not got:
            return False, "Fear & Greed returned nothing usable"
        return True, (f"{got.get('fear_greed')}/100 "
                      f"({got.get('fear_greed_label') or 'unlabelled'})")
    return go


def probe_news() -> ProbeFn:
    def go() -> tuple[bool, str]:
        src = research_mod.fetch_news(hours=24)
        if not src.ok:
            return False, src.error or "news fetch failed"
        items = len(src.data.get("items", []) or [])
        errors = src.data.get("errors") or []
        detail = f"{items} headlines"
        if errors:
            detail += f"; some feeds failed: {'; '.join(errors)[:120]}"
        return True, detail
    return go


def probe_derivatives(symbol: str) -> ProbeFn:
    def go() -> tuple[bool, str]:
        src = research_mod.fetch_derivatives(symbol)
        if not src.ok:
            return False, src.error or "derivatives fetch failed"
        d = src.data
        return True, (f"funding {d.get('funding_rate_pct')}%, "
                      f"L/S {d.get('long_short_ratio')}, "
                      f"OI chg {d.get('open_interest_change_pct')}%")
    return go


def probe_onchain() -> ProbeFn:
    def go() -> tuple[bool, str]:
        src = research_mod.fetch_onchain()
        if not src.ok:
            return False, src.error or "on-chain fetch failed"
        d = src.data
        return True, (f"mempool {d.get('mempool_tx')}, "
                      f"stablecoins {d.get('stablecoin_supply_change_pct')}%, "
                      f"mkt cap {d.get('total_mcap_change_pct')}%")
    return go


def probe_candles(cfg: Config) -> ProbeFn:
    def go() -> tuple[bool, str]:
        from . import data as data_mod
        sym = cfg.market.symbol
        ex = (cfg.market.exchange or "coinbase").lower()
        tf = cfg.market.timeframe
        if ex == "synthetic":
            return False, "configured exchange is synthetic — no network probe"
        if ex == "kraken":
            from .markets import get
            m = get(sym)
            pair = m.kraken if m else ""
            if not pair:
                return False, f"{sym} has no Kraken pair in the registry"
            bars = data_mod.kraken_candles(pair, tf, 3)
        elif ex == "binance":
            from .markets import binance_spot
            pair = binance_spot(sym)
            if not pair:
                return False, f"{sym} has no Binance spot pair in the registry"
            bars = data_mod.binance_candles(pair, tf, 3)
        else:
            bars = data_mod.coinbase_candles(sym, tf, 3)
        if not bars:
            return False, f"no candles came back from {ex}"
        return True, f"{len(bars)} bars, last {bars[-1].ts:%Y-%m-%d %H:%M}Z"
    return go


def _baseline(cfg: Config, symbol: str) -> list[tuple[str, ProbeFn]]:
    return [
        ("candles", probe_candles(cfg)),
        ("derivatives", probe_derivatives(symbol)),
        ("news", probe_news()),
        ("reddit", probe_reddit(symbol)),
        ("coingecko trending", probe_attention(symbol)),
        ("fear & greed", probe_mood()),
        ("onchain", probe_onchain()),
    ]


def run(cfg: Optional[Config] = None, symbol: Optional[str] = None,
        probes: Optional[list[tuple[str, ProbeFn]]] = None) -> dict:
    """Probe every source. Never raises: a failure is a row, not an exception."""
    cfg = cfg or Config()
    symbol = symbol or cfg.market.symbol
    todo = probes if probes is not None else _baseline(cfg, symbol)
    results = [_run(name, fn) for name, fn in todo]
    answered = sum(1 for r in results if r.ok)
    return {
        "symbol": symbol,
        "checked_at": iso(utcnow()),
        "answered": answered,
        "total": len(results),
        "results": [r.to_dict() for r in results],
        "all_ok": answered == len(results) and bool(results),
    }


def format_report(report: dict) -> str:
    """A table a human can read at 2am without decoding JSON."""
    lines = [f"sources — {report.get('symbol')} "
             f"({report.get('answered')}/{report.get('total')} answered)"]
    width = max([len(r["name"]) for r in report.get("results", [])] + [6])
    for r in report.get("results", []):
        mark = "ok  " if r["ok"] else "FAIL"
        lines.append(f"  {mark}  {r['name'].ljust(width)}  {r['detail']}")
    if not report.get("all_ok"):
        lines.append("")
        lines.append("  A source saying no is not a bug in the parser. Reddit, for")
        lines.append("  one, blocks whole networks (429 / 403 \"blocked by network")
        lines.append("  security\"); readers report what they got instead of")
        lines.append("  inventing the rest.")
    return "\n".join(lines)
