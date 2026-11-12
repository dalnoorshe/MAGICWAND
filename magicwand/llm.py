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
class DeepSeekClient:
    model: str = "deepseek-flash"
    api_key: Optional[str] = None
    max_calls_per_day: int = 400
    max_cost_per_day_usd: float = 2.00
    timeout: float = 90.0
    # A deterministic offline stand-in, used by tests and by `--offline`.
    fake: Optional[Any] = None
    usage: Usage = field(default_factory=Usage)

    # ---- key handling ----

    def _key(self) -> str:
        k = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not k and self.fake is None:
            raise LLMError(
                "no DeepSeek API key. Set DEEPSEEK_API_KEY (BYOK), or run with "
                "--offline to use the deterministic fake model.")
        return k

    def available(self) -> bool:
        return bool(self.fake is not None or self.api_key or os.environ.get("DEEPSEEK_API_KEY"))

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
             retries: int = 2) -> str:
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
