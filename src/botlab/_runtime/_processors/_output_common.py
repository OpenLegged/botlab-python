"""Validation and coercion helpers shared by the output processors."""

import math

import numpy as np


def _processor_error(label, message):
    return ValueError(f"[{label}] {message}")


def _require_input_dim(input_dims, label):
    dim = input_dims[0] if input_dims else 0
    if not dim:
        raise _processor_error(label, "inputDims[0] is required")
    return dim


def _is_blank(raw):
    return raw is None or (isinstance(raw, str) and raw.strip() == "")


def _number(raw):
    """JS ``Number()`` coercion: null -> 0, bools -> 0/1, numeric strings parse."""
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return float("nan")
    return float("nan")


def _is_integer(value):
    """JS ``Number.isInteger`` on an already-coerced float."""
    value = float(value)
    return math.isfinite(value) and value.is_integer()


def _first_configured(*values):
    for value in values:
        if not _is_blank(value):
            return value
    return None


def _first_bad_index(mask):
    return int(np.argmax(mask))
