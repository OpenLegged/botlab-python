"""Normalize Python declaration values to their JSON/JS counterparts."""

from __future__ import annotations

import math
from typing import Any

from ._errors import BotLabError


def json_value(value: Any, label: str, _stack: frozenset[int] = frozenset()) -> Any:
    """Normalize a Python declaration value to its JS counterpart.

    Pyodide converts Python ``None`` to JavaScript ``undefined``; inside a
    declaration mapping it represents JSON ``null``, so it is normalized here
    before the value is validated. Numbers must be finite; integers are kept
    as ``int`` so the JSON serialization matches JS integer rendering.
    """
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BotLabError(f"[botlab] {label} must be finite")
        if value.is_integer() and abs(value) < 2**31:
            return int(value)
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if id(value) in _stack:
            raise BotLabError(f"[botlab] {label} must not contain a cycle")
        stack = _stack | {id(value)}
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise BotLabError(f"[botlab] {label} keys must be non-empty strings")
            result[key] = json_value(item, f"{label}.{key}", stack)
        return result
    raise BotLabError(f"[botlab] {label} must be JSON-serializable")
