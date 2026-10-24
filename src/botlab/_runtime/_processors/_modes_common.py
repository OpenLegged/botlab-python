"""Coercion and validation helpers shared by the mode processors."""

import math
import re

import numpy as np

_LIST_SPLIT = re.compile(r"[\s,]+")


def _fail(label, message):
    raise ValueError(f"[{label}] {message}")


def _is_empty(raw):
    """JS null / undefined / '' — the values TS treats as "not provided"."""
    return raw is None or (isinstance(raw, str) and raw == "")


def _number(raw):
    """Mirror JS Number() for plan config scalars; junk coerces to NaN."""
    if isinstance(raw, (bool, np.bool_)):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def _is_integer_number(value):
    return math.isfinite(value) and value == math.floor(value)


def _trunc(value):
    """Mirror JS Math.trunc: NaN and infinities pass through unchanged."""
    if math.isnan(value) or math.isinf(value):
        return value
    return float(math.trunc(value))


def _integer_value(raw, fallback, field, label):
    if _is_empty(raw):
        return fallback
    value = _number(raw)
    if not _is_integer_number(value):
        _fail(label, f"{field} must be an integer")
    return int(value)


def _required_integer(raw, field, label):
    if _is_empty(raw):
        _fail(label, f"{field} is required")
    return _integer_value(raw, 0, field, label)


def _mode_from_input(buffer, label):
    """Read the mode code from a scalar input (fractions are truncated)."""
    mode = _trunc(float(buffer[0]))
    if not _is_integer_number(mode):
        _fail(label, "mode must be an integer")
    return int(mode)


def _dim_at(input_dims, slot):
    return input_dims[slot] if 0 <= slot < len(input_dims) else 0


def _input_at(inputs, slot):
    return inputs[slot] if 0 <= slot < len(inputs) else None
