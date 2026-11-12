"""MAGICWAND — the integration catalogue.

One place that answers "what do I need to connect?", generated from the code's
own host list so it cannot drift from reality.

Three states matter:

    NEEDED      — without it, something important does not work
    OPTIONAL    — better with it; the system degrades honestly without it
    NONE        — no key exists or is wanted; it just works (public data)
    MISSING     — implemented as an adapter slot, but no provider is wired yet

Nothing here prints a secret. The only thing shown about a key is a masked
hint that comes from settings_store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Integration:
    name: str
    kind: str                     # ai | market-data | social | onchain | broker
    requirement: str              # needed | optional | none | missing
    what_it_powers: str
    where_to_get: str = ""
    cost: str = ""
    notes: str = ""
    hosts: list[str] = field(default_factory=list)
    # runtime state, filled by `collect`
    status: str = "unknown"       # connected | absent | public | unsupported
    detail: str = ""


CATALOG: list[Integration] = [
    # ---------------- the only key you actually need ----------------
    Integration(
        name="DeepSeek (the AI)",
        kind="ai",
        requirement="needed",
        what_it_powers=("all four research readers (derivatives, news, social, "
                        "on-chain), the chart reader's vision, and the strategy "
                        "proposer"),
        where_to_get="https://platform.deepseek.com/api_keys",
        cost=("deepseek-flash ≈ $0.15–0.30 per 1M input tokens; ~$0.002 per "
              "research cycle, ~$0.0003 per chart image"),
        notes=("BYOK, no special permissions or scopes — it is a plain inference "
               "API. Paste it in Settings; it is stored locally at mode 0600 and "
               "never sent back to the browser. An env var DEEPSEEK_API_KEY also "
               "works and wins if set."),
        hosts=["api.deepseek.com", "platform.deepseek.com"],
    ),
    # ---------------- market data: no keys at all ----------------
    Integration(
        name="Coinbase Exchange (candles)",
        kind="market-data",
        requirement="none",
        what_it_powers="the price bars your strategy trades and the chart the agent sees",
        cost="free, public, no key",
        notes="Default feed. An API key is only needed to place orders — which "
              "this build does not do.",
        hosts=["api.exchange.coinbase.com"],
    ),
    Integration(
        name="Binance Futures (public stats)",
        kind="market-data",
        requirement="none",
        what_it_powers="funding rate, long/short ratios, open interest, taker pressure",
        cost="free, public, no key",
        notes="Read-only public endpoints. No account, no signature, no key.",
        hosts=["fapi.binance.com"],
    ),
    Integration(
        name="Kraken (fallback candles)",
        kind="market-data",
        requirement="none",
        what_it_powers="alternative OHLC source if Coinbase fails",
        cost="free, public, no key",
        hosts=["api.kraken.com"],
    ),
    Integration(
        name="Crypto news RSS",
        kind="market-data",
        requirement="none",
        what_it_powers="headlines for the news reader",
        cost="free, public, no key",
        hosts=["cointelegraph.com", "decrypt.co", "coindesk.com",
               "bitcoinmagazine.com"],
    ),
    Integration(
        name="Reddit (social volume)",
        kind="social",
        requirement="none",
        what_it_powers="post volume and titles for the social reader",
        cost="free, public, no key",
        notes=("The JSON API is blocked for scripts; the Atom feed works with a "
               "bot user-agent. It rate-limits aggressively — expect the social "
               "reader to sit at DEGRADED much of the time."),
        hosts=["www.reddit.com"],
    ),
    Integration(
        name="On-chain / flows proxies",
        kind="onchain",
        requirement="none",
        what_it_powers="mempool depth, stablecoin supply, market-cap flows",
        cost="free, public, no key",
        notes=("These are PROXIES, not exchange netflow, and the prompt says so. "
               "See the optional items below for the real thing."),
        hosts=["blockchain.info", "stablecoins.llama.fi", "api.coingecko.com"],
    ),
    # ---------------- slots with no provider wired ----------------
    Integration(
        name="Whale Alert / Glassnode (true exchange netflow)",
        kind="onchain",
        requirement="missing",
        what_it_powers="real exchange inflows/outflows instead of proxies",
        where_to_get="https://whale-alert.io  ·  https://glassnode.com",
        cost="paid",
        notes=("The on-chain reader's data structure has a slot for this, but no "
               "adapter is written. Adding one is a small, contained change."),
    ),
    Integration(
        name="Longer historical candles",
        kind="market-data",
        requirement="missing",
        what_it_powers="backtests over years rather than ~12 days",
        cost="free tiers exist",
        notes=("The strategy gate would be far more trustworthy with years of "
               "bars. Not wired yet; Coinbase's free window is what it is."),
    ),
    # ---------------- the broker question ----------------
    Integration(
        name="Broker / exchange trading key",
        kind="broker",
        requirement="none",
        what_it_powers=("nothing — there is no live order path in this build"),
        cost="—",
        notes=("Do not connect one. `PaperBroker` is the only execution path. "
               "If live trading is ever added, the key must be trade-only "
               "(never withdrawal), IP allow-listed, and gated behind a human "
               "tier change."),
    ),
]


REQUIREMENT_LABEL = {
    "needed": "NEEDED",
    "optional": "optional",
    "none": "no key needed",
    "missing": "not wired",
}


def collect(root: str | Path) -> list[Integration]:
    """Fill in each integration's live status. Never raises, never leaks a key."""
    from .settings_store import SettingsStore

    root = Path(root)
    out: list[Integration] = []
    for base in CATALOG:
        it = Integration(**{**base.__dict__, "hosts": list(base.hosts)})
        if it.kind == "ai":
            try:
                st = SettingsStore(root / "data" / "settings.json")
                s = st.load()
                if s.effective_key():
                    it.status = "connected"
                    it.detail = f"{s.masked_key()} · from {s.key_source()}"
                else:
                    it.status = "absent"
                    it.detail = "no key yet — paste one in Settings"
            except Exception:  # noqa: BLE001
                it.status = "absent"
                it.detail = "settings unreadable"
        elif it.requirement == "none" and it.kind == "broker":
            it.status = "not-applicable"
            it.detail = "paper trading only"
        elif it.requirement == "none":
            it.status = "public"
            it.detail = "no key required"
        elif it.requirement == "missing":
            it.status = "unsupported"
            it.detail = "no adapter written"
        else:
            it.status = "unknown"
        out.append(it)
    return out


def missing_required(root: str | Path) -> list[Integration]:
    return [i for i in collect(root)
            if i.requirement == "needed" and i.status != "connected"]
