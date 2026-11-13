"""MAGICWAND — the operator chat.

A place to talk to the model about what the system is doing.

The rule that makes this safe, and it is not negotiable:

    the chat is READ-ONLY.

It receives a compact description of the current state — mode, tier, market,
desk, latest verdicts, reflection, ledger health — and it can explain, diagnose
and recommend. It has **no tools**, cannot place an order, cannot change a
setting, and cannot touch the ledger. If it wants something changed it says
which UI control or command the human should use.

That asymmetry is deliberate. An assistant that can act on its own advice turns
every conversational mistake into a live change, and it makes the audit trail
useless: you could no longer tell whether a setting changed because you decided
it or because a model suggested it in passing.

History is stored locally in data/chat.json. Nothing is sent anywhere except the
one completion request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .common import iso, utcnow
from .llm import BudgetExceeded, DeepSeekClient, LLMError
from .settings_store import SettingsStore, resolve_key

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY = 60          # messages kept on disk
MAX_CONTEXT_MESSAGES = 12  # messages sent to the model per turn


SYSTEM = """You are the MAGICWAND operator assistant, embedded in the trading
console. You help the operator understand what the system is doing.

Hard rules:
- You CANNOT act. You have no tools. You cannot place or close trades, change
  the risk tier, change the market, edit settings, or write to the ledger.
- If the operator asks for a change, tell them exactly where to do it (a control
  in the console, or a CLI command) and note that they must do it themselves.
- Ground every answer in the STATE block. Do not invent numbers, positions,
  readings or history. If something is not in the state, say it is not available.
- Be brief. Two or three sentences unless asked to explain at length. Plain
  English, no marketing, no predictions about prices.
- Be honest about limitations: paper trading only, no live broker, no guarantee
  of profit, and the gate may have approved nothing at all.

What the system is:
- A paper-trading research system. Four text readers (derivatives, news, social,
  on-chain) plus a chart reader with vision produce verdicts: support / neutral /
  veto. Only a VETO is actionable: it blocks a trade. Readers can never open one.
- Strategies invented by the model must pass a validation gate (realistic costs,
  2x costs, minimum trades, exposure-adjusted benchmark, out-of-sample, and a
  shuffled-data null test) before the desk may trade them.
- A separate sandboxed "desk" holds the money that may trade. Tier ceilings are
  set by a human: research (read-only) < paper_desk (10% pot, 1 trade/day) <
  scaled_desk (25%, 3/day) < live (refused; no live broker exists).
- Everything is written to a hash-chained ledger. Reflection adjusts sizing
  within fixed bounds and demotes losing strategy families.
"""


# --------------------------------------------------------------------------
# State -> compact context
# --------------------------------------------------------------------------

def build_context(state: dict) -> str:
    """Turn /api/state into a compact, factual block for the model."""
    lines: list[str] = []
    add = lines.append

    settings = state.get("settings") or {}
    policy = state.get("policy") or {}
    desk = state.get("desk") or {}
    research = state.get("research") or {}
    ledger = state.get("ledger") or {}
    memo = state.get("memory") or {}
    autopilot = state.get("autopilot") or {}
    backup = state.get("backups") or {}

    add(f"time: {state.get('server_time')}")
    add(f"model_mode: {state.get('model_mode')} (live = real DeepSeek, offline = stub)")
    add(f"market: {state.get('symbol')} on {state.get('exchange')} @ {state.get('timeframe')}")
    add(f"configured_symbol: {settings.get('symbol')} exchange: {settings.get('exchange')}")
    add(f"api_key_connected: {settings.get('has_key')} (source: {settings.get('key_source')})")
    add(f"model: {settings.get('model')}")
    add(f"tier: {policy.get('tier_name')} — {json.dumps(policy.get('effective'))}")
    add(f"account_equity: {state.get('account_equity')}")
    add(f"desk: cash={desk.get('cash')} trades={desk.get('trades')} "
        f"win_rate={desk.get('win_rate')} today={desk.get('trades_today')} "
        f"autonomy_armed={desk.get('autonomy_armed')} "
        f"position={json.dumps(desk.get('position'))}")

    if research:
        add(f"latest_research: decision={research.get('decision')} "
            f"evidence={research.get('evidence_score')} stub={research.get('stub')}")
        add(f"  summary: {research.get('summary')}")
        for a in (research.get("agents") or [])[:5]:
            verdict = a.get("error") or a.get("stance")
            thesis = (a.get("thesis") or a.get("error") or "")[:180]
            add(f"  reader {a.get('agent')}: {verdict} conf={a.get('confidence')} "
                f"ev={a.get('evidence_score')} :: {thesis}")
    else:
        add("latest_research: none recorded yet")

    refl = state.get("reflection") or {}
    if refl:
        add(f"reflection_changes: {refl.get('changes')}")
    add(f"demoted_families: {memo.get('blacklisted_families')}")
    add(f"gate_approved_strategies: {len(state.get('strategies') or [])}")
    props = state.get("proposals") or {}
    if props:
        add(f"proposals_total: {props.get('total')} outcomes={props.get('counts')}")
        add(f"  proposals_actually_validated: {props.get('validated')} "
            f"failing_gates={props.get('failing_gates')}")
        for pr in (props.get("recent") or [])[-3:]:
            add(f"  last_proposal: {pr.get('name')} ({pr.get('family')}) "
                f"ok={pr.get('ok')} :: {pr.get('summary')}")
    add(f"ledger: {ledger.get('message')} entries={ledger.get('entries')}")
    add(f"autopilot: running={autopilot.get('running')} ticks={autopilot.get('ticks')} "
        f"interval={autopilot.get('interval')}s last={autopilot.get('last_result')}")
    add(f"backups: {backup.get('archives')} kept, last_age_hours={backup.get('age_hours')}")

    markets = [m.get("symbol") for m in (settings.get("markets") or [])]
    add(f"available_markets: {markets}")
    add("how to change things (the operator must do these):")
    add("  market: Settings -> Market, or ./start.sh --symbol <SYM>")
    add("  venue: Settings -> Market -> Coinbase/Kraken")
    add("  api key: Settings -> paste -> Save -> Test connection")
    add("  tier: click a tier card in the console (human only)")
    add("  backups: 'Back up now' in Local storage")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

def history_path(root: str | Path) -> Path:
    return Path(root) / "data" / "chat.json"


def load_history(root: str | Path) -> list[dict]:
    p = history_path(root)
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
        return d.get("messages", []) if isinstance(d, dict) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(root: str | Path, messages: list[dict]) -> None:
    p = history_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps({"messages": messages[-MAX_HISTORY:]}, indent=2))
    except OSError:
        pass


def clear_history(root: str | Path) -> None:
    save_history(root, [])


def archive_path(root: str | Path) -> Path:
    return Path(root) / "data" / "chat-archive.jsonl"


def start_new(root: str | Path) -> dict:
    """Start a fresh conversation, keeping the old one.

    "Clear" destroying history was the wrong default for a project whose whole
    point is an audit trail: a conversation where you asked "why did it veto?"
    and got an answer is evidence about what you were told. So a new chat
    appends the old one to data/chat-archive.jsonl and empties the live log.
    """
    root = Path(root)
    old = load_history(root)
    archived = 0
    if old:
        try:
            p = archive_path(root)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                for m in old:
                    f.write(json.dumps({"conversation_end": iso(utcnow()),
                                        **m}, sort_keys=True) + "\n")
            archived = len(old)
        except OSError:
            return {"ok": False, "error": "could not archive the conversation"}
    save_history(root, [])
    return {"ok": True, "archived": archived}


def archived_count(root: str | Path) -> int:
    p = archive_path(root)
    if not p.exists():
        return 0
    try:
        with p.open() as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# --------------------------------------------------------------------------
# The client (one per process, with its own budget)
# --------------------------------------------------------------------------

_CLIENT: Optional[DeepSeekClient] = None


def get_client(root: str | Path) -> DeepSeekClient:
    global _CLIENT
    if _CLIENT is None:
        st = SettingsStore(Path(root) / "data" / "settings.json").load()
        _CLIENT = DeepSeekClient(
            model=st.model or "deepseek-flash",
            settings_path=str(Path(root) / "data" / "settings.json"),
            max_calls_per_day=200,
            max_cost_per_day_usd=1.00,
            thinking="disabled",
        )
    return _CLIENT


# --------------------------------------------------------------------------
# Reply
# --------------------------------------------------------------------------

@dataclass
class ChatReply:
    ok: bool
    reply: str = ""
    error: str = ""
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reply": self.reply, "error": self.error,
                "cost_usd": round(self.cost_usd, 6)}


def reply(root: str | Path, message: str, state: dict) -> ChatReply:
    root = Path(root)
    message = (message or "").strip()
    if not message:
        return ChatReply(False, error="empty message")
    if len(message) > MAX_MESSAGE_CHARS:
        return ChatReply(False, error=f"message too long (max {MAX_MESSAGE_CHARS})")
    if not resolve_key(root):
        return ChatReply(False, error=(
            "no API key connected — the chat needs one. Settings -> paste key."))

    client = get_client(root)
    client.settings_path = str(root / "data" / "settings.json")

    history = load_history(root)
    convo = history[-MAX_CONTEXT_MESSAGES:]
    turns = []
    for m in convo:
        role = "assistant" if m.get("role") == "assistant" else "user"
        turns.append(f"{role}: {m.get('text', '')[:600]}")
    prompt = (
        "STATE\n" + build_context(state) + "\n\n"
        + ("CONVERSATION SO FAR\n" + "\n".join(turns) + "\n\n" if turns else "")
        + f"USER: {message}\n\nAnswer as the operator assistant. Remember: you "
          f"cannot act, and you must ground claims in STATE."
    )
    try:
        text = client.chat(system=SYSTEM, user=prompt, max_tokens=1200,
                           temperature=0.3, json_mode=False)
    except BudgetExceeded as e:
        return ChatReply(False, error=f"budget: {e}")
    except LLMError as e:
        return ChatReply(False, error=str(e)[:400])

    now = iso(utcnow())
    history = history + [
        {"role": "user", "text": message, "ts": now},
        {"role": "assistant", "text": text, "ts": now,
         "cost_usd": round(client.usage.cost_usd, 6)},
    ]
    save_history(root, history)
    return ChatReply(True, reply=text, cost_usd=client.usage.cost_usd)
