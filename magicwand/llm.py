"""MAGICWAND — DeepSeek client (BYOK).

Bring your own key: read from `DEEPSEEK_API_KEY`, never stored, never logged.
The model does research and explanation. It never touches order placement —
that is enforced in magicwand/agents.py, not by asking the model politely.

Costs are tracked per call and capped per day, because an agent that re-reads
the news every five minutes is a subscription, not a strategy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .common import utcnow

API_URL = "https://api.deepseek.com/chat/completions"
MODELS_URL = "https://api.deepseek.com/models"
BALANCE_URL = "https://api.deepseek.com/user/balance"

# USD per 1M tokens. Peak/off-peak differ (peak = 01:00-04:00 and 06:00-10:00
# UTC, weekdays). We assume the *higher* (peak) price so budgeting is safe.
PRICES = {
    "deepseek-flash": {"in": 0.30, "out": 1.20},
    "deepseek-v4-pro": {"in": 1.32, "out": 3.96},
}


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


class TruncatedResponse(LLMError):
    """The model hit max_tokens before finishing its answer.

    Thinking mode is ON BY DEFAULT at effort `high`, and its chain of thought is
    returned separately in `reasoning_content` but still consumes the same token
    budget. With a small budget that means `content` arrives empty (all budget
    spent on thinking) or cut off mid-string (a truncated JSON object) — which
    looks like a model failure but is a budget failure.
    """


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    truncations: int = 0
    empty_content: int = 0
    reasoning_tokens: int = 0
    day: str = ""

    def add(self, pt: int, ct: int, model: str) -> float:
        p = PRICES.get(model, PRICES["deepseek-flash"])
        c = (pt / 1e6) * p["in"] + (ct / 1e6) * p["out"]
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.cost_usd += c
        return c


@dataclass
class DeepSeekClient:  # noqa: D101
    model: str = "deepseek-flash"
    api_key: Optional[str] = None
    max_calls_per_day: int = 400
    max_cost_per_day_usd: float = 2.00
    timeout: float = 90.0
    # 'disabled' | 'enabled'. Structured extraction (the schema-bound readers and
    # the proposer) runs with thinking OFF: it is faster, cheaper, deterministic,
    # and cannot consume the JSON budget with a chain of thought.
    thinking: str = "disabled"
    # Ceiling for automatic budget escalation on a truncated/empty answer.
    max_tokens_ceiling: int = 8192
    # A deterministic offline stand-in, used by tests and by `--offline`.
    fake: Optional[Any] = None
    # Whether verdicts from `fake` are PLACEHOLDERS (the offline stand-in) or a
    # test double standing in for a real model. Placeholders can never veto and
    # are never graded; test doubles behave like the real thing.
    stub_mode: bool = True
    usage: Usage = field(default_factory=Usage)
    # Optional path to the console settings file. When set, a key saved through
    # the UI is used without any environment variable.
    settings_path: Optional[str] = None

    # ---- key handling ----

    def _stored_key(self) -> str:
        if not self.settings_path:
            return ""
        try:
            from .settings_store import SettingsStore
            return SettingsStore(self.settings_path).load().effective_key()
        except Exception:  # noqa: BLE001 - a broken settings file is not fatal
            return ""

    def _key(self) -> str:
        k = (self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
             or self._stored_key())
        if not k and self.fake is None:
            raise LLMError(
                "no DeepSeek API key. Set DEEPSEEK_API_KEY (BYOK), or run with "
                "--offline to use the deterministic fake model.")
        return k

    def vision_available(self) -> bool:
        """Vision needs the real API; the offline fake cannot see images."""
        return self.fake is None and bool(self.api_key
                                          or os.environ.get("DEEPSEEK_API_KEY")
                                          or self._stored_key())

    def available(self) -> bool:
        return bool(self.fake is not None or self.api_key
                    or os.environ.get("DEEPSEEK_API_KEY") or self._stored_key())

    # ---- budgeting ----

    def _roll_day(self) -> None:
        today = utcnow().strftime("%Y-%m-%d")
        if self.usage.day != today:
            self.usage = Usage(day=today)

    def _check_budget(self, estimated_calls: int = 1) -> None:
        self._roll_day()
        if self.usage.calls + estimated_calls > self.max_calls_per_day:
            raise BudgetExceeded(
                f"daily call budget reached ({self.usage.calls}/{self.max_calls_per_day}). "
                f"Research pauses; trading strategy is unaffected.")
        if self.usage.cost_usd >= self.max_cost_per_day_usd:
            raise BudgetExceeded(
                f"daily cost budget reached (${self.usage.cost_usd:.4f}"
                f"/${self.max_cost_per_day_usd:.2f}). Research pauses.")

    # ---- chat ----

    def chat(self, system: str, user: str, temperature: float = 0.3,
             max_tokens: int = 1200, json_mode: bool = True,
             retries: int = 2, images: Optional[list] = None) -> str:
        """One completion. `images` is a list of objects with a `data_url`."""
        if images:
            return self.chat_vision(system=system, user=user, images=images,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    json_mode=json_mode, retries=retries)
        self._check_budget()
        if self.fake is not None:
            text = self.fake(system, user)
            self.usage.calls += 1
            return text

        key = self._key()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
            # Explicit, because the default is ON at effort 'high'.
            "thinking": {"type": self.thinking},
        }
        if self.thinking != "enabled":
            # temperature is ignored in thinking mode, so only send it when it
            # can actually have an effect.
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        budget = max_tokens
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                body["max_tokens"] = budget
                raw = _post_json(API_URL, body, key, self.timeout)
                text, truncated, reason_chars = _read_choice(raw)
                u = raw.get("usage", {}) or {}
                self.usage.add(int(u.get("prompt_tokens", 0)),
                               int(u.get("completion_tokens", 0)), self.model)
                self.usage.reasoning_tokens += reason_chars

                if truncated:
                    self.usage.truncations += 1
                    raise TruncatedResponse(
                        f"answer truncated at max_tokens={budget} "
                        f"(reasoning used ~{reason_chars} chars)")
                if not text.strip():
                    # All budget went to reasoning, or the model said nothing.
                    self.usage.empty_content += 1
                    raise TruncatedResponse(
                        f"empty content (reasoning ~{reason_chars} chars, "
                        f"finish={_finish(raw)})")
                return text
            except TruncatedResponse as e:
                last = e
                self.usage.errors += 1
                # Escalate the budget and turn thinking off: this is the failure
                # mode where thinking ate the JSON.
                body["thinking"] = {"type": "disabled"}
                budget = min(self.max_tokens_ceiling, max(budget * 2, budget + 1024))
                if attempt < retries:
                    continue
            except (LLMError, OSError, TimeoutError) as e:
                last = e
                self.usage.errors += 1
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
        raise LLMError(f"DeepSeek request failed after {retries + 1} attempts: {last}")


# DeepSeek resizes every image so that it costs at most 1024 tokens. We bill
# the worst case so budgeting can never surprise the operator.
IMAGE_TOKENS_MAX = 1024


def chat_vision(self: DeepSeekClient, system: str, user: str,
                images: list, temperature: float = 0.2,
                max_tokens: int = 1100, json_mode: bool = True,
                retries: int = 1) -> str:
    """Send text + images. Images are supported in USER messages only."""
    self._check_budget()
    if not images:
        return self.chat(system, user, temperature=temperature,
                         max_tokens=max_tokens, json_mode=json_mode)

    if self.fake is not None:
        # The offline model cannot see. It says so rather than pretending.
        text = self.fake(system, user)
        try:
            obj = json.loads(text)
            obj.setdefault("warnings", []).append(
                "offline fake model: the image was NOT actually inspected")
            text = json.dumps(obj)
        except (json.JSONDecodeError, AttributeError):
            pass
        self.usage.calls += 1
        self.usage.add(0, 0, self.model)
        self.usage.prompt_tokens += IMAGE_TOKENS_MAX * len(images)
        return text

    key = self._key()
    content: list[dict] = [{"type": "text", "text": user}]
    for img in images:
        content.append({"type": "image_url",
                        "image_url": {"url": img.data_url, "detail": "high"}})
    body: dict[str, Any] = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": system},   # text only: images in
            {"role": "user", "content": content},    # system msgs are a 400
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": self.thinking},
    }
    if self.thinking != "enabled":
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            raw = _post_json(API_URL, body, key, self.timeout)
            text, truncated, reason_chars = _read_choice(raw)
            u = raw.get("usage", {}) or {}
            pt = int(u.get("prompt_tokens", 0))
            self.usage.add(pt, int(u.get("completion_tokens", 0)), self.model)
            self.usage.reasoning_tokens += reason_chars
            if truncated or not text.strip():
                self.usage.truncations += 1
                raise TruncatedResponse(
                    f"vision answer truncated/empty (finish={_finish(raw)}, "
                    f"reasoning ~{reason_chars} chars, max_tokens={max_tokens})")
            return text
        except (LLMError, OSError, TimeoutError) as e:
            last = e
            self.usage.errors += 1
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise LLMError(f"DeepSeek vision request failed after {retries + 1} attempts: {last}")


def _finish(raw: dict) -> str:
    try:
        return str((raw.get("choices") or [{}])[0].get("finish_reason") or "")
    except (IndexError, AttributeError):
        return ""


def _read_choice(raw: dict) -> tuple[str, bool, int]:
    """Return (content, truncated, reasoning_chars).

    `reasoning_content` is billed against the same budget, so its size is the
    diagnostic that explains an empty or cut-off answer.
    """
    choices = raw.get("choices") or []
    if not choices:
        raise LLMError(f"empty choices: {str(raw)[:200]}")
    msg = choices[0].get("message") or {}
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    finish = str(choices[0].get("finish_reason") or "")
    return text, finish == "length", len(reasoning)


def _get_json_auth(url: str, key: str, timeout: float = 20.0) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, method="GET",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"network error: {e.reason}") from e


def list_models(client: "DeepSeekClient") -> list[dict]:
    """Fetch the models the key can actually use, with their capabilities.

    Returns a normalised list:
        {id, name, vision, context_window, max_output_tokens, reasoner}

    This is what the model picker is built from — never a hardcoded guess.
    Vision support in particular is read from `input_modalities`, because
    offering an image to a model that cannot see it is a silent failure.
    """
    key = client._key()
    raw = _get_json_auth(MODELS_URL, key, client.timeout)
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        raise LLMError(f"unexpected /models payload: {str(raw)[:200]}")
    out: list[dict] = []
    for m in data:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        raw_id = str(m["id"])
        mods = m.get("input_modalities") or ["text"]
        vision = "image" in mods if isinstance(mods, list) else False
        # Some builds report vision only via the retired -vision- name.
        if "vision" in raw_id:
            vision = True
        effort = (m.get("effort") or {}).get("supported_levels") or []
        out.append({
            "id": raw_id,
            "name": m.get("name") or raw_id,
            "vision": bool(vision),
            "context_window": m.get("context_window"),
            "max_output_tokens": m.get("max_output_tokens"),
            "effort_levels": effort,
            "reasoner": bool(effort),
            "owned_by": m.get("owned_by"),
        })
    out.sort(key=lambda x: (not x["vision"], x["id"]))
    return out


def get_balance(client: "DeepSeekClient") -> dict:
    """Remaining balance for the key. Nice to see before a long run."""
    key = client._key()
    raw = _get_json_auth(BALANCE_URL, key, client.timeout)
    infos = raw.get("balance_infos") if isinstance(raw, dict) else None
    if not isinstance(infos, list) or not infos:
        return {"available": raw.get("is_available"), "amount": None,
                "currency": None}
    first = infos[0] if isinstance(infos[0], dict) else {}
    return {
        "available": raw.get("is_available"),
        "amount": first.get("total_balance"),
        "granted": first.get("granted_balance"),
        "topped_up": first.get("topped_up_balance"),
        "currency": first.get("currency"),
    }


def _post_json(url: str, body: dict, key: str, timeout: float) -> dict:
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        raise LLMError(f"HTTP {e.code} from DeepSeek: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"network error: {e.reason}") from e


# --------------------------------------------------------------------------
# A fake model for tests and offline demos. It is deliberately dull: it reads
# the numbers out of the prompt and returns a schema-shaped answer. Its value
# is that the pipeline can be exercised without a key or a network.
# --------------------------------------------------------------------------

def fake_proposal(system: str, user: str) -> str:
    """Offline stand-in for the strategy proposer.

    Reads the catalog ranges out of the prompt and emits a deterministic,
    in-range proposal. It is not intelligent; its job is to let the whole
    gate be exercised without a key. The gate, not this function, decides.
    """
    import re
    family = "sma_trend"
    if "breakout" in user and "mean_revert" in user:
        # Pick deterministically from what the context emphasises.
        if "oversold" in user.lower() or "mean" in user.lower():
            family = "mean_revert"
        elif "breakout" in user.lower() or "high" in user.lower():
            family = "breakout"
    # Vary with how many ideas have already been tried, so the offline model
    # keeps producing GENUINELY NEW proposals instead of the same one forever.
    # The novelty gate would refuse a repeat, which is correct but makes an
    # offline demo look stalled.
    tried = user.count("  - ")
    bump = (tried % 8) * 24
    if family == "mean_revert":
        params = {"slow": 20 + bump, "z_entry": 1.5 + 0.2 * (tried % 5),
                  "stop_atr": 2.0, "target_atr": 3.0, "max_hold_bars": 48}
    elif family == "breakout":
        params = {"slow": 16 + bump, "stop_atr": 2.0, "target_atr": 4.0,
                  "max_hold_bars": 48}
    else:
        params = {"fast": 8 + (tried % 6) * 2, "slow": 24 + bump,
                  "stop_atr": 2.0, "target_atr": 4.0, "max_hold_bars": 72}
    family = family  # kept explicit for readability
    return json.dumps({
        "name": f"offline_{family}_{tried}",
        "hypothesis": ("offline placeholder: testing whether this family's "
                       "mechanism survives realistic and doubled costs"),
        "family": family,
        "params": params,
        "symbol": "SOL-USD",
        "timeframe": "1h",
    })


def fake_canned(system: str, user: str) -> str:
    low = user.lower()
    bearish, bullish = 0, 0
    for word, sign in (("crash", -1), ("hack", -1), ("lawsuit", -1), ("ban", -1),
                       ("surge", 1), ("rally", 1), ("approve", 1), ("inflow", 1)):
        if word in low:
            if sign > 0:
                bullish += 1
            else:
                bearish += 1
    score = 0.0
    if bullish or bearish:
        score = (bullish - bearish) / (bullish + bearish)
    stance = "veto" if score < -0.3 else "support" if score > 0.3 else "neutral"
    return json.dumps({
        "stance": stance,
        "confidence": 0.6,
        "evidence_score": round(score, 2),
        "thesis": f"offline fake model: counted {bullish} bullish and {bearish} bearish cues",
        "claims": [{
            "statement": "price will rise over the horizon" if score >= 0 else
                         "price will fall over the horizon",
            "direction": "up" if score >= 0 else "down",
            "confidence": 0.6,
            "horizon_hours": 24,
            "reasoning": "offline fake model; not a real forecast",
        }],
        "warnings": ["this is the deterministic offline model, not DeepSeek"],
    })


DeepSeekClient.chat_vision = chat_vision  # type: ignore[attr-defined]
