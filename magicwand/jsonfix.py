"""MAGICWAND — tolerant JSON parsing for model output.

Models are asked for a JSON object and usually give one. Two failure modes
remain, and they are different problems:

  1. TRUNCATION — the answer stopped mid-object because the token budget ran
     out. `finish_reason == "length"`. Handled in llm.py by retrying with a
     bigger budget; by the time text reaches here it should be complete.

  2. MALFORMATION — the answer is complete but not valid JSON. The commonest
     cause is a literal newline inside a string value: JSON forbids raw control
     characters in strings, models emit them anyway, and `json.loads` reports
     `Invalid control character` or `Unterminated string` depending on where it
     gives up.

This module repairs case 2 before declaring failure, and says when it did — a
repair that changes meaning must never be silent.

Nothing here is a security boundary. The parsed result is still schema-validated
by the caller (agents.parse_verdict) and the agent still cannot open a trade.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class ParseResult:
    __slots__ = ("obj", "repaired", "note")

    def __init__(self, obj: Any = None, repaired: bool = False, note: str = "") -> None:
        self.obj = obj
        self.repaired = repaired
        self.note = note

    @property
    def ok(self) -> bool:
        return self.obj is not None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_object_text(text: str) -> str:
    """Pull the most likely JSON object out of a chatty response."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start: end + 1]
    if start != -1:
        return text[start:]
    return text


# --------------------------------------------------------------------------
# Repairs, in order of least invasive first
# --------------------------------------------------------------------------

def _escape_control_chars_in_strings(s: str) -> tuple[str, int]:
    """Escape literal newlines/tabs/CR that occur INSIDE string literals.

    Walks the text tracking whether we are inside a string, honouring backslash
    escapes. Outside strings the characters stay as they are, because JSON
    allows (and needs) whitespace there.
    """
    out: list[str] = []
    in_str = False
    escaped = False
    fixed = 0
    for ch in s:
        if in_str:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                continue
            if ch in "\n\r\t":
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
                fixed += 1
                continue
            if ord(ch) < 0x20:
                out.append("\\u%04x" % ord(ch))
                fixed += 1
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
                escaped = False
                continue
            if ch == "\\":     # stray backslash outside a string: drop it
                out.pop()
                fixed += 1
    return "".join(out), fixed


def _strip_trailing_commas(s: str) -> tuple[str, int]:
    """Remove commas immediately before a closing brace/bracket."""
    out = re.sub(r",(\s*[}\]])", r"\1", s)
    return out, (len(s) - len(out))


def _close_open_structures(s: str) -> tuple[str, int]:
    """Close strings/braces left open by a cut-off answer.

    Only used as a last resort, and the caller is told it happened: a repaired
    object may be missing fields, which the schema check then rejects.
    """
    in_str = False
    escaped = False
    stack: list[str] = []
    for ch in s:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    added = 0
    out = s
    if in_str:
        out += '"'
        added += 1
    for closer in reversed(stack):
        out += closer
        added += 1
    return out, added


def parse_json_lenient(text: str) -> ParseResult:
    """Best-effort parse of model output. Reports whether repair was needed."""
    body = extract_object_text(text)
    if not body:
        return ParseResult(None, False, "no JSON object in response")

    # 1. straight parse
    try:
        return ParseResult(json.loads(body), False, "")
    except json.JSONDecodeError as first:
        err = first

    # 2. escape control characters inside strings
    fixed, n_ctrl = _escape_control_chars_in_strings(body)
    if n_ctrl:
        try:
            return ParseResult(json.loads(fixed), True,
                               f"escaped {n_ctrl} control char(s) inside strings")
        except json.JSONDecodeError as e:
            err = e

    # 3. trailing commas
    nocomma, n_comma = _strip_trailing_commas(fixed)
    if n_comma:
        try:
            return ParseResult(json.loads(nocomma), True,
                               f"removed {n_comma} trailing comma(s)")
        except json.JSONDecodeError as e:
            err = e
    else:
        nocomma = fixed

    # 4. close unterminated strings/structures (last resort)
    closed, n_added = _close_open_structures(nocomma)
    if n_added:
        try:
            obj = json.loads(closed)
            return ParseResult(obj, True,
                               f"closed {n_added} unterminated structure(s) — "
                               f"object may be incomplete")
        except json.JSONDecodeError as e:
            err = e

    return ParseResult(None, False,
                       f"invalid JSON: {err.msg} at line {err.lineno} "
                       f"column {err.colno} (char {err.pos}); "
                       f"{len(body)} chars, tail={body[-60:]!r}")
