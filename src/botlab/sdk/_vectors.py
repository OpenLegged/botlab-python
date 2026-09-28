"""Numeric validation and plain-vector math, the non-Signal half of the SDK."""
import math


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_values(values, label):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite numeric sequence") from error
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def require_vector(values, size, label="vector"):
    size = _positive_integer(size, f"{label} size")
    vector = _finite_values(values, label)
    if len(vector) != size:
        raise ValueError(f"{label} must contain {size} values, got {len(vector)}")
    return vector


def _expand_vector(values, size, label):
    if isinstance(values, bool):
        raise ValueError(f"{label} must be a finite number or numeric sequence")
    if isinstance(values, (int, float)):
        value = float(values)
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return [value] * size
    return require_vector(values, size, label)


def _negated(value, label):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return -_finite_number(value, label)
    return [-item for item in _finite_values(value, label)]


def _reciprocal(value, label):
    scalar = isinstance(value, (int, float)) and not isinstance(value, bool)
    divisors = [_finite_number(value, label)] if scalar else _finite_values(value, label)
    if any(divisor == 0.0 for divisor in divisors):
        raise ValueError(f"{label} must not contain zero")
    inverted = [1.0 / divisor for divisor in divisors]
    return inverted[0] if scalar else inverted


def clip(values, lower, upper):
    vector = _finite_values(values, "clip values")
    if not vector:
        raise ValueError("clip values must not be empty")
    lower_values = _expand_vector(lower, len(vector), "clip lower")
    upper_values = _expand_vector(upper, len(vector), "clip upper")
    if any(low > high for low, high in zip(lower_values, upper_values)):
        raise ValueError("clip lower values must not exceed upper values")
    return [
        min(max(value, low), high)
        for value, low, high in zip(vector, lower_values, upper_values)
    ]


def reorder(values, indices):
    vector = _finite_values(values, "reorder values")
    output = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("reorder indices must be integers")
        if index < 0 or index >= len(vector):
            raise ValueError(
                f"reorder index {index} is outside a vector of length {len(vector)}"
            )
        output.append(vector[index])
    if not output:
        raise ValueError("reorder indices must not be empty")
    return output


def affine(values, scale, offset):
    vector = _finite_values(values, "affine values")
    if not vector:
        raise ValueError("affine values must not be empty")
    scales = _expand_vector(scale, len(vector), "affine scale")
    offsets = _expand_vector(offset, len(vector), "affine offset")
    return [
        value * scale_value + offset_value
        for value, scale_value, offset_value in zip(vector, scales, offsets)
    ]
