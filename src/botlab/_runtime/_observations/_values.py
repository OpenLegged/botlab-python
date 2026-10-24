"""Number coercion, vector parsing and the slice/transform chain shared by observation readers."""

import math

import numpy as np


def _error(prefix, message):
    return ValueError(f"[{prefix}] {message}")


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
    value = float(value)
    return math.isfinite(value) and value.is_integer()


def _flat(source, label):
    """``requireArray`` + flatten: MjData gives 2-D views, TS reads flat ones."""
    if source is None:
        raise _error("mujocoObservations", f"{label} is missing or not array-like")
    try:
        array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError):
        raise _error("mujocoObservations", f"{label} is missing or not array-like") from None
    if array.ndim == 0:
        raise _error("mujocoObservations", f"{label} is missing or not array-like")
    return array.reshape(-1)


def _float32_slice(source, label, start=0, end=None):
    """``float32Slice``: bounds-checked finite-checked float32 copy."""
    values = _flat(source, label)
    length = len(values)
    if start < 0:
        raise _error("mujocoObservations", f"{label} slice start {start} is negative")
    if end is not None and end < start:
        raise _error("mujocoObservations", f"{label} slice end {end} is before start {start}")
    if start > length or (end is not None and end > length):
        shown = length if end is None else end
        raise _error(
            "mujocoObservations",
            f"{label} slice {start}:{shown} exceeds length {length}",
        )
    safe_end = length if end is None else end
    output = values[start:safe_end]
    bad = np.flatnonzero(~np.isfinite(output))
    if len(bad):
        raise _error("mujocoObservations", f"{label}[{int(bad[0])}] must be finite")
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# mujocoObservationTransforms.ts mirror
# ---------------------------------------------------------------------------


def parse_number_list(raw, label):
    if raw is None:
        return None

    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        value = _number(raw)
        if not math.isfinite(value):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return [value]

    if isinstance(raw, str):
        parts = [part for part in raw.replace(",", " ").split() if part]
        if not parts:
            return None
        values = [_number(part) for part in parts]
        if not all(math.isfinite(value) for value in values):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return values

    if isinstance(raw, (list, tuple, np.ndarray)):
        values = [_number(value) for value in raw]
        if not all(math.isfinite(value) for value in values):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return values

    raise _error(
        "mujocoObservations",
        f"{label} must be a number, list string, or numeric array",
    )


def _first_present(config, keys):
    """JS ``??`` chain: only ``None`` (and missing keys) fall through."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def configured_dim(config, fallback):
    raw = _first_present(config or {}, ("dim",))
    if raw is None or raw == "":
        if fallback > 0:
            return int(math.floor(fallback))
        raise _error("mujocoObservations", "observation dimension could not be inferred")
    dim = _number(raw)
    if math.isfinite(dim) and dim > 0:
        return int(math.floor(dim))
    raise _error(
        "mujocoObservations",
        f"observation dimension must be positive, got {raw}",
    )


def vector_from(raw, dim, label, fallback=0.0):
    if not _is_integer(dim) or dim <= 0:
        raise _error("mujocoObservations", f"{label} dimension must be a positive integer")
    dim = int(dim)
    output = np.full(dim, fallback, dtype=np.float32)
    values = parse_number_list(raw, label)
    if not values:
        return output
    count = min(len(values), dim)
    output[:count] = values[:count]
    return output


def constant_vector(config):
    config = config or {}
    values = parse_number_list(_first_present(config, ("values",)), "obs-constant values")
    if not values:
        raise _error("mujocoObservations", "obs-constant requires a finite values array")
    dim = configured_dim(config, len(values))
    if len(values) != dim:
        raise _error(
            "mujocoObservations",
            f"obs-constant values length {len(values)} does not match dim {dim}",
        )
    return np.asarray(values, dtype=np.float32)


def _index_map(config):
    config = config or {}
    values = parse_number_list(
        _first_present(config, ("indices",)), "indices"
    )
    if not values:
        return None
    indices = [int(math.trunc(value)) for value in values]
    if any(value < 0 or not _is_integer(value) for value in values):
        raise _error("mujocoObservations", "indices must be non-negative integers")
    return indices


def _default_value_source(config):
    return _first_present(
        config,
        ("defaultAngles",),
    )


def _mapped_value(values, output_index, source_index, has_indices, fallback):
    if values is None:
        return fallback
    if len(values) == 1:
        return values[0]
    index = source_index if has_indices else output_index
    if index < 0 or index >= len(values):
        raise _error(
            "mujocoObservations",
            f"transform value index {index} exceeds length {len(values)}",
        )
    return values[index]


def transform_vector(values, config):
    """``transformVector``: index gather, default subtraction, scalar/vector scale."""
    config = config or {}
    values = np.asarray(values, dtype=np.float32)
    defaults = parse_number_list(_default_value_source(config), "default position")
    indices = _index_map(config)
    parsed_scale = parse_number_list(_first_present(config, ("scale",)), "scale")
    scalar_scale = parsed_scale[0] if parsed_scale and len(parsed_scale) == 1 else math.nan
    scale_values = parsed_scale if parsed_scale and len(parsed_scale) > 1 else None

    if indices is not None:
        for index in indices:
            if index < 0 or index >= len(values):
                raise _error(
                    "mujocoObservations",
                    f"source index {index} exceeds observation length {len(values)}",
                )
        source_indices = np.asarray(indices, dtype=np.int64)
    else:
        source_indices = np.arange(len(values), dtype=np.int64)

    output = values[source_indices].astype(np.float32)
    output_length = len(output)
    if defaults is not None:
        output -= np.asarray(
            [
                _mapped_value(defaults, i, int(source_indices[i]), indices is not None, 0.0)
                for i in range(output_length)
            ],
            dtype=np.float32,
        )
    if scale_values is not None:
        output *= np.asarray(
            [
                _mapped_value(scale_values, i, int(source_indices[i]), indices is not None, 1.0)
                for i in range(output_length)
            ],
            dtype=np.float32,
        )
    elif math.isfinite(scalar_scale):
        output *= np.float32(scalar_scale)
    return output


def _finite_copy(values, label):
    values = _flat(values, label)
    bad = np.flatnonzero(~np.isfinite(values))
    if len(bad):
        raise _error("botlab", f"{label}[{int(bad[0])}] must be finite")
    return values.astype(np.float32)


def _positive_integer(value, label):
    number = _number(value)
    if not _is_integer(number) or number <= 0:
        raise _error("botlab", f"{label} must be a positive integer")
    return int(number)


def _string_list(raw, label):
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        raise _error("botlab", f"{label} must be a non-empty string list")
    return list(raw)
