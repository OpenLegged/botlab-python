"""Tensor metadata and model-node config parsing helpers."""

import json
import math
import re

import numpy as np


def _coerce_number(value):
    """JS ``Number()`` semantics for plan config scalars (NaN for garbage)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def _is_integer_number(value):
    return math.isfinite(value) and float(value).is_integer()


def _shape_text(shape):
    return "x".join(str(dim) for dim in shape)


def _move_first(values, index):
    return [values[index], *values[:index], *values[index + 1:]]


def tensor_element_count(dims):
    count = 1
    for dim in dims:
        count *= dim
    return count


def tensor_dims_from_metadata(metadata, label):
    """Static dims for a state/constant tensor; symbolic dims collapse to 1."""
    if not metadata or not metadata.get("is_tensor"):
        raise ValueError(f"[ModelProcessor] {label} shape metadata is required")
    shape = metadata.get("shape")
    if not shape:
        raise ValueError(f"[ModelProcessor] {label} shape metadata is required")
    dims = []
    for dim in shape:
        if (
            not isinstance(dim, bool)
            and isinstance(dim, (int, float))
            and math.isfinite(dim)
            and dim > 0
        ):
            dims.append(math.floor(dim))
        elif isinstance(dim, str) and dim:
            dims.append(1)
        else:
            raise ValueError(f'[ModelProcessor] {label} has invalid dimension "{dim}"')
    return dims


def input_tensor_dims_from_metadata(metadata, label, input_length):
    """Resolve the primary input shape for ``input_length`` flat values.

    Mirrors the TS rule: no tensor shape feeds ``[1, input_length]``; a fully
    static shape must match exactly; one dynamic dim (or a leading dynamic
    pair) absorbs the remainder; anything more is ambiguous and fails.
    """
    shape = metadata.get("shape") if metadata and metadata.get("is_tensor") else None
    if not shape:
        return [1, input_length]

    dims = []
    unknown_indices = []
    known_product = 1
    for index, dim in enumerate(shape):
        if (
            not isinstance(dim, bool)
            and isinstance(dim, (int, float))
            and math.isfinite(dim)
            and dim > 0
        ):
            value = math.floor(dim)
            dims.append(value)
            known_product *= value
            continue
        dims.append(1)
        unknown_indices.append(index)

    if not unknown_indices:
        if known_product != input_length:
            raise ValueError(
                f"[ModelProcessor] {label} expects {known_product} values, got {input_length}"
            )
        return dims

    if input_length % known_product != 0:
        raise ValueError(
            f"[ModelProcessor] {label} shape {_shape_text(shape)} cannot fit "
            f"{input_length} input values"
        )

    remaining = input_length // known_product
    if len(unknown_indices) == 1:
        dims[unknown_indices[0]] = remaining
    elif len(unknown_indices) == 2 and unknown_indices[0] == 0:
        dims[unknown_indices[1]] = remaining
    else:
        raise ValueError(
            f"[ModelProcessor] {label} shape {_shape_text(shape)} has ambiguous "
            "dynamic dimensions"
        )

    if tensor_element_count(dims) != input_length:
        raise ValueError(
            f"[ModelProcessor] {label} resolved shape does not match {input_length} "
            "input values"
        )
    return dims


def numeric_tensor_data(value, label):
    """Numeric ndarray or an explicit failure (bool reads numeric, as in ORT-Web)."""
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc":
        raise ValueError(f"[ModelProcessor] {label} is not a numeric tensor")
    return value


def _static_element_count(metadata):
    """Element count when every dim is a positive integer, else None."""
    shape = metadata.get("shape") if metadata else None
    if not metadata or not metadata.get("is_tensor") or not shape:
        return None
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            return None
        count *= dim
    return count


def _numeric_dim_product(metadata):
    """Browser registry inference: product of the positive numeric dims."""
    shape = metadata.get("shape") if metadata else None
    if not metadata or not metadata.get("is_tensor") or not shape:
        return 0
    dim = 1
    for part in shape:
        if isinstance(part, bool) or not isinstance(part, (int, float)):
            continue
        if not math.isfinite(part) or part <= 0:
            continue
        dim *= part
    return dim if dim > 1 else 0


def _configured_io_name(raw, field):
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"[ModelProcessor] {field} must be a non-empty string when provided")
    return raw.strip()


def _configured_output_dim(raw):
    if raw is None or raw == "":
        return None
    value = _coerce_number(raw)
    if not _is_integer_number(value) or value <= 0:
        raise ValueError("[ModelProcessor] outputDim must be a positive integer when provided")
    return int(value)


def _number_list(raw, label):
    if isinstance(raw, bool):
        raise ValueError(f"[ModelProcessor] {label} must be a number list")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            raise ValueError(f"[ModelProcessor] {label} must contain finite numbers")
        return [float(raw)]
    if isinstance(raw, str):
        values = [
            _coerce_number(part) for part in re.split(r"[\s,]+", raw) if part
        ]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        values = [_coerce_number(item) for item in raw]
    else:
        raise ValueError(f"[ModelProcessor] {label} must be a number list")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"[ModelProcessor] {label} must contain finite numbers")
    return values


def _parse_constant_specs(raw, field):
    if raw is None or raw == "":
        return []
    parsed = raw
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError as error:
            raise ValueError(f"[ModelProcessor] {field} must be valid JSON: {error}") from error
    if field == "constants" and isinstance(parsed, dict):
        return [{"name": name, "values": values} for name, values in parsed.items()]
    if not isinstance(parsed, list):
        expected = "a mapping or an array" if field == "constants" else "an array"
        raise ValueError(f"[ModelProcessor] {field} must be {expected}")
    for index, spec in enumerate(parsed):
        if not isinstance(spec, dict):
            raise ValueError(f"[ModelProcessor] {field}[{index}] must be an object")
    return parsed
