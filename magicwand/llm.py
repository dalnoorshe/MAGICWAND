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


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
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
    # A deterministic offline stand-in, used by tests and by `--offline`.
    fake: Optional[Any] = None
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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                raw = _post_json(API_URL, body, key, self.timeout)
                choices = raw.get("choices") or []
                if not choices:
                    raise LLMError(f"empty choices: {str(raw)[:200]}")
                text = choices[0].get("message", {}).get("content", "") or ""
                u = raw.get("usage", {}) or {}
                self.usage.add(int(u.get("prompt_tokens", 0)),
                               int(u.get("completion_tokens", 0)), self.model)
                return text
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
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            raw = _post_json(API_URL, body, key, self.timeout)
            choices = raw.get("choices") or []
            if not choices:
                raise LLMError(f"empty choices: {str(raw)[:200]}")
            text = choices[0].get("message", {}).get("content", "") or ""
            u = raw.get("usage", {}) or {}
            pt = int(u.get("prompt_tokens", 0))
            self.usage.add(pt, int(u.get("completion_tokens", 0)), self.model)
            return text
        except (LLMError, OSError, TimeoutError) as e:
            last = e
            self.usage.errors += 1
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise LLMError(f"DeepSeek vision request failed after {retries + 1} attempts: {last}")


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
