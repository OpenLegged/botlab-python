"""Math-family plan processors — numpy ports of the TS engine nodes.

Source of truth: src/domain/runtime/engine/processors/{Math,VectorMath,
SignalMath,SpatialMath,Clamp,SignalFilter,SignalNoise,Clock,PhaseClock}
Processor.ts. Semantics are mirrored operation for operation: config keys
keep their TS camelCase spelling, error messages keep the TS shapes, output
buffers are float32 (the Float32Array equivalents), and compute() never
writes into its input aliases. All per-element math is vectorized; the only
Python loops left are over slots/parameters or O(log n) doubling steps.

Documented deviations (the local executor is synchronous and context-free):

- ClockProcessor: the TS node reads frame.time / frame.index from the strict
  simulation step context through computeAsync. There is no step context
  here, so the processor owns the control-step counter: every compute()
  call publishes the current step and then advances it, and
  ``unit="seconds"`` scales the step with the configured ``dt`` (required —
  no frame to source it from). reset() rewinds the counter to zero.
- SignalNoiseProcessor consumes a compute's uniforms only after the input
  has been validated, so a rejected compute leaves the random stream
  untouched (TS draws per element and aborts midway).
- MathProcessor maps JSON ``null`` operands to "missing" (TS ``Number(null)``
  would coerce to 0); numeric strings are not coerced.
"""

import math

import numpy as np

from .._quat import quat_multiply

_LCG_MULTIPLIER = 1664525
_LCG_INCREMENT = 1013904223
_LCG_MODULUS = 2 ** 32


def _error(label, message):
    """processorError(label, message) — ``[Label] message`` ValueError."""
    return ValueError(f"[{label}] {message}")


def _to_float(value):
    """JS ``Number(value)`` for the scalar config paths (NaN when absent)."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _first_nonfinite_index(values):
    return int(np.flatnonzero(~np.isfinite(values))[0])


def _require_input_dim(input_dims, label):
    """requireInputDim — the first slot must have a positive dimension."""
    dim = input_dims[0] if input_dims else 0
    if not dim:
        raise _error(label, "inputDims[0] is required")
    return dim


def _coerce_limit_vector(raw, dim, label, field):
    """coerceLimitVector — scalar broadcast or exact-length float64 vector."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not math.isfinite(raw):
            raise _error(label, f"{field} must be finite")
        return np.full(dim, float(raw), dtype=np.float64)
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise _error(label, f"{field} must be a finite number or a {dim}-value array")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise _error(label, f"{field} must contain exactly {dim} finite numbers") from None
    if values.ndim != 1 or values.size != dim or not np.isfinite(values).all():
        raise _error(label, f"{field} must contain exactly {dim} finite numbers")
    return values


def _broadcast_operand(raw, dim, name):
    """MathProcessor operand broadcast — stored float32 like the TS vectors."""
    label = "MathProcessor"
    if isinstance(raw, (list, tuple, np.ndarray)):
        try:
            values = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            raise _error(label, f"{name} contains non-finite numbers") from None
        if values.ndim != 1:
            raise _error(label, f"{name} contains non-finite numbers")
        if values.size != dim:
            raise _error(label, f"{name} must contain exactly {dim} numbers, got {values.size}")
        if not np.isfinite(values).all():
            raise _error(label, f"{name} contains non-finite numbers")
        return values.astype(np.float32)
    number = _to_float(raw)
    if not math.isfinite(number):
        raise _error(label, f"{name} must be a finite number or array of size {dim}")
    return np.full(dim, number, dtype=np.float32)


def _lcg_uniforms(state, count):
    """First ``count`` uniforms of the TS LCG stream, vectorized.

    state_{n+1} = (a * state_n + c) mod 2^32 has the closed form
    state_n = (a^n * s0 + c * (1 + a + ... + a^{n-1})) mod 2^32, so the whole
    block is evaluated with doubling instead of a per-element Python loop.
    Returns (uniforms, next_state).
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64), state
    modulus = np.uint64(_LCG_MODULUS)
    powers = np.ones(1, dtype=np.uint64)
    while powers.size < count + 1:
        step = int(powers[-1]) * _LCG_MULTIPLIER % _LCG_MODULUS
        powers = np.concatenate([powers, powers * np.uint64(step) % modulus])
    geometric = np.cumsum(powers)  # exact in uint64 for any realistic count
    terms = powers[1 : count + 1] * np.uint64(state) % modulus
    terms = terms + (np.uint64(_LCG_INCREMENT) * geometric[:count] % modulus)
    states = terms % modulus
    uniforms = states.astype(np.float64) / float(_LCG_MODULUS)
    return uniforms, int(states[-1])


def _quat_rotate(quaternion, norm_squared, vectors, inverse):
    """Rotate (batch, 3) vectors by wxyz quaternions; tolerates non-unit input."""
    w = quaternion[:, 0]
    x, y, z = quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    if inverse:
        x, y, z = -x, -y, -z
    axis = np.stack([x, y, z], axis=1)
    tangent = 2.0 * np.cross(axis, vectors)
    return vectors + (w[:, None] * tangent + np.cross(axis, tangent)) / norm_squared[:, None]


def _quat_matrix(quaternion, norm_squared):
    """Row-major (batch, 3, 3) rotation matrices; scales by 2 / normSquared."""
    scale = (2.0 / norm_squared)[:, None]
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    matrix = np.empty((quaternion.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1.0 - yy - zz
    matrix[:, 0, 1] = xy - wz
    matrix[:, 0, 2] = xz + wy
    matrix[:, 1, 0] = xy + wz
    matrix[:, 1, 1] = 1.0 - xx - zz
    matrix[:, 1, 2] = yz - wx
    matrix[:, 2, 0] = xz - wy
    matrix[:, 2, 1] = yz + wx
    matrix[:, 2, 2] = 1.0 - xx - yy
    return matrix


def _quat_slerp(a, norm_squared_a, b, norm_squared_b, t):
    """Shortest-arc slerp with the TS linear fallback above dot 0.9995."""
    a_unit = a / np.sqrt(norm_squared_a)[:, None]
    b_unit = b / np.sqrt(norm_squared_b)[:, None]
    dot = (a_unit * b_unit).sum(axis=1)
    flip = dot < 0
    b_unit = np.where(flip[:, None], -b_unit, b_unit)
    dot = np.abs(dot)
    with np.errstate(invalid="ignore"):
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        weight_a = np.sin((1.0 - t) * theta) / sin_theta
        weight_b = np.sin(t * theta) / sin_theta
        exact = weight_a[:, None] * a_unit + weight_b[:, None] * b_unit
        linear = a_unit + t[:, None] * (b_unit - a_unit)
        linear = linear / np.sqrt((linear * linear).sum(axis=1))[:, None]
    return np.where((dot > 0.9995)[:, None], linear, exact)


class MathProcessor:
    """``output = input * scale + offset`` (proc-math: add / multiply / affine)."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._scale = np.zeros(0, dtype=np.float32)
        self._offset = np.zeros(0, dtype=np.float32)

    def init(self, config, input_dims):
        dim = _require_input_dim(input_dims, "MathProcessor")
        operation = config.get("operation")
        if operation == "add":
            self._scale = np.ones(dim, dtype=np.float32)
            self._offset = _broadcast_operand(config.get("value"), dim, "value")
        elif operation == "multiply":
            self._scale = _broadcast_operand(config.get("value"), dim, "value")
            self._offset = np.zeros(dim, dtype=np.float32)
        elif operation == "affine":
            self._scale = _broadcast_operand(config.get("scale"), dim, "scale")
            self._offset = _broadcast_operand(config.get("offset"), dim, "offset")
        else:
            raise _error("MathProcessor", f'unsupported operation "{operation}"')
        self.output_buffers = [np.zeros(dim, dtype=np.float32)]

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _error("MathProcessor", "Missing input")
        output = self.output_buffers[0]
        if len(source) != len(output):
            raise _error(
                "MathProcessor",
                f"input length changed from {len(output)} to {len(source)}",
            )
        result = (
            source.astype(np.float64)
            * self._scale.astype(np.float64)
            + self._offset.astype(np.float64)
        )
        output[:] = result


class VectorMathProcessor:
    """Elementwise arithmetic between two per-frame signals (proc-vector-math)."""

    _OPERATIONS = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
    }

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._apply = None
        self._left_dim = 0
        self._right_dim = 0

    def init(self, config, input_dims):
        label = "VectorMathProcessor"
        if len(input_dims) != 2:
            raise _error(label, "requires exactly two inputs")
        left, right = input_dims[0], input_dims[1]
        if not left or not right:
            raise _error(label, "both inputs must have a positive dimension")
        if left != right and left != 1 and right != 1:
            raise _error(
                label,
                f"inputs must have equal dimensions or one scalar, got {left} and {right}",
            )
        operation = config.get("operation")
        apply = self._OPERATIONS.get(operation) if isinstance(operation, str) else None
        if apply is None:
            raise _error(label, f"operation must be one of {', '.join(self._OPERATIONS)}")
        self._apply = apply
        self._left_dim = left
        self._right_dim = right
        self.output_buffers = [np.zeros(max(left, right), dtype=np.float32)]

    def compute(self, inputs):
        label = "VectorMathProcessor"
        left = inputs[0] if len(inputs) > 0 else None
        right = inputs[1] if len(inputs) > 1 else None
        if (
            left is None
            or right is None
            or len(left) != self._left_dim
            or len(right) != self._right_dim
        ):
            raise _error(label, "input dimensions changed after init")
        with np.errstate(divide="ignore", invalid="ignore"):
            result = self._apply(left.astype(np.float64), right.astype(np.float64))
        self.output_buffers[0][:] = result


class SignalMathProcessor:
    """Unary / grouped-reduction / binary vector math (proc-signal-math)."""

    _ELEMENTWISE = ("abs", "sqrt", "sin", "cos", "exp", "log")
    _REDUCTIONS = ("sum", "min", "max", "norm", "normalize")
    _BINARY = ("dot", "cross")

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._operation = "abs"
        self._input_dims = []
        self._width = 0
        self._batch_count = 0

    @classmethod
    def _read_operation(cls, raw):
        label = "SignalMathProcessor"
        if isinstance(raw, str) and raw in cls._ELEMENTWISE + cls._REDUCTIONS + cls._BINARY:
            return raw
        raise _error(
            label,
            "operation must be one of "
            + ", ".join(cls._ELEMENTWISE + cls._REDUCTIONS + cls._BINARY),
        )

    @staticmethod
    def _read_width(raw, input_dim):
        value = input_dim if raw is None else _to_float(raw)
        if (
            not math.isfinite(value)
            or value <= 0
            or value != int(value)
            or input_dim % value != 0
        ):
            raise _error(
                "SignalMathProcessor",
                f"width must be a positive divisor of input dimension {input_dim}",
            )
        return int(value)

    def init(self, config, input_dims):
        label = "SignalMathProcessor"
        self._operation = self._read_operation(config.get("operation"))
        self._input_dims = list(input_dims)
        first = input_dims[0] if input_dims else 0
        if first <= 0:
            raise _error(label, "first input must have a positive dimension")

        if self._operation in self._BINARY:
            if len(input_dims) != 2:
                raise _error(label, f"{self._operation} requires exactly two inputs")
            second = input_dims[1]
            self._width = self._read_width(config.get("width"), first)
            if second <= 0 or second % self._width != 0:
                raise _error(
                    label,
                    f"second input dimension must be a positive multiple of width {self._width}",
                )
            first_batches = first // self._width
            second_batches = second // self._width
            self._batch_count = max(first_batches, second_batches)
            if any(
                batches != 1 and batches != self._batch_count
                for batches in (first_batches, second_batches)
            ):
                raise _error(label, "input batches must match or one input must contain one batch")
            if self._operation == "cross" and self._width != 3:
                raise _error(label, "cross requires width 3")
            output_dim = self._batch_count * (1 if self._operation == "dot" else 3)
        else:
            if len(input_dims) != 1:
                raise _error(label, f"{self._operation} requires exactly one input")
            self._width = self._read_width(config.get("width"), first)
            self._batch_count = first // self._width
            output_dim = (
                self._batch_count
                if self._operation in ("sum", "min", "max", "norm")
                else first
            )
        self.output_buffers = [np.zeros(output_dim, dtype=np.float32)]

    def compute(self, inputs):
        label = "SignalMathProcessor"
        if len(inputs) != len(self._input_dims) or any(
            source is None or len(source) != dim
            for source, dim in zip(inputs, self._input_dims)
        ):
            raise _error(label, "input dimensions changed after init")
        for source in inputs:
            if not np.isfinite(source).all():
                raise _error(label, f"input[{_first_nonfinite_index(source)}] must be finite")

        output = self.output_buffers[0]
        operation = self._operation
        if operation in self._BINARY:
            self._compute_binary(inputs[0], inputs[1], output)
            return

        values = inputs[0].reshape(-1, self._width).astype(np.float64)
        if operation == "abs":
            output[:] = np.abs(values).reshape(-1)
        elif operation == "sqrt":
            negative = np.flatnonzero(values < 0)
            if negative.size:
                raise _error(label, f"sqrt input[{int(negative[0])}] must be non-negative")
            output[:] = np.sqrt(values).reshape(-1)
        elif operation == "sin":
            output[:] = np.sin(values).reshape(-1)
        elif operation == "cos":
            output[:] = np.cos(values).reshape(-1)
        elif operation == "exp":
            output[:] = np.exp(values).reshape(-1)
        elif operation == "log":
            nonpositive = np.flatnonzero(values <= 0)
            if nonpositive.size:
                raise _error(label, f"log input[{int(nonpositive[0])}] must be positive")
            output[:] = np.log(values).reshape(-1)
        elif operation == "sum":
            output[:] = values.sum(axis=1)
        elif operation == "min":
            output[:] = values.min(axis=1)
        elif operation == "max":
            output[:] = values.max(axis=1)
        elif operation == "norm":
            output[:] = np.sqrt((values * values).sum(axis=1))
        else:  # normalize
            norm_squared = (values * values).sum(axis=1)
            inverse = np.zeros_like(norm_squared)
            nonzero = norm_squared > 0
            inverse[nonzero] = 1.0 / np.sqrt(norm_squared[nonzero])
            output[:] = (values * inverse[:, None]).reshape(-1)
            return
        if not np.isfinite(output).all():
            raise _error(label, f"{operation} result must be finite")

    def _compute_binary(self, first, second, output):
        label = "SignalMathProcessor"
        left = first.reshape(-1, self._width).astype(np.float64)
        right = second.reshape(-1, self._width).astype(np.float64)
        if self._operation == "dot":
            output[:] = (left * right).sum(axis=1)
            if not np.isfinite(output).all():
                raise _error(label, "dot result must be finite")
            return
        ax, ay, az = left[:, 0], left[:, 1], left[:, 2]
        bx, by, bz = right[:, 0], right[:, 1], right[:, 2]
        output[:] = np.stack(
            [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx],
            axis=1,
        ).reshape(-1)
        if not np.isfinite(output).all():
            raise _error(label, "cross result must be finite")


# (required inputs, per-slot (slot, width, label), output width, broadcast label)
_SPATIAL_SPECS = {
    "quat_inverse": (1, ((0, 4, "quaternion"),), 4, None),
    "quat_heading": (1, ((0, 4, "quaternion"),), 4, None),
    "quat_to_matrix": (1, ((0, 4, "quaternion"),), 9, None),
    "quat_to_rotation6d": (1, ((0, 4, "quaternion"),), 6, None),
    "quat_to_tangent_normal": (1, ((0, 4, "quaternion"),), 6, None),
    "quat_multiply": (
        2,
        ((0, 4, "first quaternion"), (1, 4, "second quaternion")),
        4,
        "quaternion",
    ),
    "relative_quaternion": (
        2,
        ((0, 4, "orientation"), (1, 4, "reference")),
        4,
        "quaternion",
    ),
    "quat_rotate": (2, ((0, 4, "quaternion"), (1, 3, "vector")), 3, "rotation"),
    "quat_rotate_inverse": (2, ((0, 4, "quaternion"), (1, 3, "vector")), 3, "rotation"),
    "quat_slerp": (
        3,
        ((0, 4, "first quaternion"), (1, 4, "second quaternion"), (2, 1, "interpolation")),
        4,
        "slerp",
    ),
    "relative_position": (
        3,
        ((0, 3, "position"), (1, 3, "origin"), (2, 4, "orientation")),
        3,
        "relative position",
    ),
}

_SPATIAL_OPERATIONS = (
    "quat_multiply",
    "quat_inverse",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_to_matrix",
    "quat_to_rotation6d",
    "quat_to_tangent_normal",
    "quat_slerp",
    "quat_heading",
    "relative_position",
    "relative_quaternion",
)


class SpatialMathProcessor:
    """Quaternion / spatial operations on wxyz data (proc-spatial-math)."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._operation = "quat_multiply"
        self._input_dims = []
        self._batch_count = 0
        self._output_width = 0

    def init(self, config, input_dims):
        label = "SpatialMathProcessor"
        operation = config.get("operation")
        if operation not in _SPATIAL_SPECS:
            raise _error(
                label, f"operation must be one of {', '.join(_SPATIAL_OPERATIONS)}"
            )
        required, slots, output_width, broadcast_label = _SPATIAL_SPECS[operation]
        if len(input_dims) != required:
            raise _error(label, f"{operation} requires exactly {required} inputs")

        batches = []
        for slot, width, name in slots:
            dim = input_dims[slot]
            if dim <= 0 or dim % width != 0:
                raise _error(
                    label,
                    f"{name} dimension must be a positive multiple of {width}, got {dim}",
                )
            batches.append(dim // width)

        if broadcast_label is None:
            self._batch_count = batches[0]
        else:
            self._batch_count = max(batches)
            if any(count != 1 and count != self._batch_count for count in batches):
                raise _error(
                    label,
                    f"{broadcast_label} batches must match or one input must contain one batch",
                )
        self._operation = operation
        self._input_dims = list(input_dims)
        self._output_width = output_width
        self.output_buffers = [np.zeros(self._batch_count * output_width, dtype=np.float32)]

    def compute(self, inputs):
        label = "SpatialMathProcessor"
        if len(inputs) != len(self._input_dims) or any(
            source is None or len(source) != dim
            for source, dim in zip(inputs, self._input_dims)
        ):
            raise _error(label, "input dimensions changed after init")

        output = self.output_buffers[0]
        operation = self._operation

        def quaternion(slot, name):
            values = inputs[slot].reshape(-1, 4).astype(np.float64)
            if not np.isfinite(values).all():
                raise _error(label, f"{name} must contain finite numbers")
            norm_squared = (values * values).sum(axis=1)
            if not np.isfinite(norm_squared).all() or (norm_squared == 0).any():
                raise _error(label, f"{name} must not contain a zero quaternion")
            return values, norm_squared

        def vector(slot, name):
            values = inputs[slot].reshape(-1, 3).astype(np.float64)
            if not np.isfinite(values).all():
                raise _error(label, f"{name} must contain finite numbers")
            return values

        if operation == "quat_multiply":
            a, _ = quaternion(0, "first quaternion")
            b, _ = quaternion(1, "second quaternion")
            result = quat_multiply(a, b)
        elif operation == "quat_inverse":
            a, norm_squared = quaternion(0, "quaternion")
            conjugate = a * np.array([1.0, -1.0, -1.0, -1.0])
            result = conjugate / norm_squared[:, None]
        elif operation in ("quat_rotate", "quat_rotate_inverse"):
            q, norm_squared = quaternion(0, "quaternion")
            v = vector(1, "vector")
            result = _quat_rotate(q, norm_squared, v, operation == "quat_rotate_inverse")
        elif operation == "quat_to_matrix":
            q, norm_squared = quaternion(0, "quaternion")
            result = _quat_matrix(q, norm_squared).reshape(-1, 9)
        elif operation == "quat_to_rotation6d":
            q, norm_squared = quaternion(0, "quaternion")
            flat = _quat_matrix(q, norm_squared).reshape(-1, 9)
            result = flat[:, [0, 1, 3, 4, 6, 7]]
        elif operation == "quat_to_tangent_normal":
            q, norm_squared = quaternion(0, "quaternion")
            flat = _quat_matrix(q, norm_squared).reshape(-1, 9)
            result = flat[:, [0, 3, 6, 2, 5, 8]]
        elif operation == "quat_slerp":
            a, norm_squared_a = quaternion(0, "first quaternion")
            b, norm_squared_b = quaternion(1, "second quaternion")
            interpolation = inputs[2].reshape(-1, 1).astype(np.float64)
            if not np.isfinite(interpolation).all():
                raise _error(label, "interpolation must contain finite numbers")
            t = interpolation[:, 0]
            if ((t < 0) | (t > 1)).any():
                raise _error(label, "interpolation must be between 0 and 1")
            result = _quat_slerp(a, norm_squared_a, b, norm_squared_b, t)
        elif operation == "quat_heading":
            q, norm_squared = quaternion(0, "quaternion")
            inverse_norm = 1.0 / np.sqrt(norm_squared)
            w = q[:, 0] * inverse_norm
            x = q[:, 1] * inverse_norm
            y = q[:, 2] * inverse_norm
            z = q[:, 3] * inverse_norm
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            result = np.stack(
                [np.cos(yaw / 2.0), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0)],
                axis=1,
            )
        elif operation == "relative_position":
            position = vector(0, "position")
            origin = vector(1, "origin")
            q, norm_squared = quaternion(2, "orientation")
            result = _quat_rotate(q, norm_squared, position - origin, True)
        else:  # relative_quaternion
            orientation, _ = quaternion(0, "orientation")
            reference, norm_squared = quaternion(1, "reference")
            inverse_reference = reference * np.array([1.0, -1.0, -1.0, -1.0])
            inverse_reference = inverse_reference / norm_squared[:, None]
            result = quat_multiply(inverse_reference, orientation)

        output[:] = result.reshape(-1)
        if not np.isfinite(output).all():
            raise _error(label, f"{operation} result must be finite")


class ClampProcessor:
    """Per-element clamp against configured min/max (proc-clamp)."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._min_values = np.zeros(0, dtype=np.float64)
        self._max_values = np.zeros(0, dtype=np.float64)

    def init(self, config, input_dims):
        label = "ClampProcessor"
        dim = _require_input_dim(input_dims, label)
        self.output_buffers = [np.zeros(dim, dtype=np.float32)]
        self._min_values = _coerce_limit_vector(config.get("min"), dim, label, "min")
        self._max_values = _coerce_limit_vector(config.get("max"), dim, label, "max")
        if (self._min_values > self._max_values).any():
            index = int(np.flatnonzero(self._min_values > self._max_values)[0])
            raise _error(label, f"min[{index}] must be <= max[{index}]")

    def compute(self, inputs):
        label = "ClampProcessor"
        source = inputs[0] if inputs else None
        if source is None:
            raise _error(label, "Missing input")
        output = self.output_buffers[0]
        if len(source) != len(output):
            raise _error(
                label,
                f"Input length {len(source)} does not match output length {len(output)}",
            )
        if not np.isfinite(source).all():
            raise _error(label, f"input[{_first_nonfinite_index(source)}] is not finite")
        np.clip(source, self._min_values, self._max_values, out=output)


class SignalFilterProcessor:
    """Temporal signal filters: ema / moving_average / difference (proc-signal-filter)."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._input_dim = 0
        self._kind = "ema"
        self._alpha = 0.0
        self._window = 0
        self._dt = 0.0
        self._previous = np.zeros(0, dtype=np.float32)
        self._samples = np.zeros((0, 0), dtype=np.float32)
        self._sums = np.zeros(0, dtype=np.float64)
        self._cursor = 0
        self._sample_count = 0
        self._has_previous = False

    def init(self, config, input_dims):
        label = "SignalFilterProcessor"
        self._input_dim = _require_input_dim(input_dims, label)
        if len(input_dims) != 1:
            raise _error(label, "requires exactly one input")
        kind = config.get("kind")
        if kind not in ("ema", "moving_average", "difference"):
            raise _error(label, 'kind must be "ema", "moving_average", or "difference"')
        self._kind = kind
        self.output_buffers = [np.zeros(self._input_dim, dtype=np.float32)]
        self._previous = np.zeros(self._input_dim, dtype=np.float32)
        self._samples = np.zeros((0, self._input_dim), dtype=np.float32)
        self._sums = np.zeros(self._input_dim, dtype=np.float64)
        self._cursor = 0
        self._sample_count = 0
        self._has_previous = False

        if kind == "ema":
            alpha = _to_float(config.get("alpha"))
            if not math.isfinite(alpha) or alpha <= 0 or alpha > 1:
                raise _error(label, "alpha must be a finite number greater than 0 and at most 1")
            self._alpha = alpha
        elif kind == "moving_average":
            window = _to_float(config.get("window"))
            if not math.isfinite(window) or window != int(window) or window <= 0:
                raise _error(label, "window must be a positive integer")
            self._window = int(window)
            self._samples = np.zeros((self._window, self._input_dim), dtype=np.float32)
        else:
            dt = _to_float(config.get("dt"))
            if not math.isfinite(dt) or dt <= 0:
                raise _error(label, "dt must be a finite number greater than 0")
            self._dt = dt

    def compute(self, inputs):
        label = "SignalFilterProcessor"
        source = inputs[0] if inputs else None
        if len(inputs) != 1 or source is None or len(source) != self._input_dim:
            raise _error(label, "input dimensions changed after init")
        if not np.isfinite(source).all():
            raise _error(label, f"input[{_first_nonfinite_index(source)}] must be finite")
        if self._kind == "ema":
            self._compute_ema(source)
        elif self._kind == "moving_average":
            self._compute_moving_average(source)
        else:
            self._compute_difference(source)

    def reset(self):
        self.output_buffers[0].fill(0)
        self._previous.fill(0)
        self._samples.fill(0)
        self._sums.fill(0)
        self._cursor = 0
        self._sample_count = 0
        self._has_previous = False

    def _check_output_finite(self):
        output = self.output_buffers[0]
        if not np.isfinite(output).all():
            raise _error(
                "SignalFilterProcessor",
                f"output[{_first_nonfinite_index(output)}] must be finite",
            )

    def _compute_ema(self, source):
        output = self.output_buffers[0]
        if not self._has_previous:
            output[:] = source
            self._has_previous = True
            return
        result = self._alpha * source.astype(np.float64) + (1.0 - self._alpha) * output.astype(
            np.float64
        )
        output[:] = result
        self._check_output_finite()

    def _compute_moving_average(self, source):
        output = self.output_buffers[0]
        if self._sample_count == self._window:
            self._sums -= self._samples[self._cursor].astype(np.float64)
        else:
            self._sample_count += 1
        self._samples[self._cursor] = source
        self._sums += source.astype(np.float64)
        output[:] = self._sums / self._sample_count
        self._check_output_finite()
        self._cursor = (self._cursor + 1) % self._window

    def _compute_difference(self, source):
        output = self.output_buffers[0]
        if not self._has_previous:
            output.fill(0)
            self._previous[:] = source
            self._has_previous = True
            return
        output[:] = (source.astype(np.float64) - self._previous.astype(np.float64)) / self._dt
        self._check_output_finite()
        self._previous[:] = source


class SignalNoiseProcessor:
    """Seeded LCG noise injection: uniform / gaussian (proc-signal-noise).

    Mirrors the TS random source exactly: the linear congruential generator
    state = (1664525 * state + 1013904223) mod 2^32 with uniform() =
    state / 2^32, and the Box-Muller pair that keeps the sine component as
    the next gaussian's spare — the stream continues across compute() calls
    and restarts from the seed on reset().
    """

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._input_dim = 0
        self._distribution = "uniform"
        self._scale = np.zeros(0, dtype=np.float64)
        self._bias = np.zeros(0, dtype=np.float64)
        self._initial_state = 0
        self._state = 0
        self._spare_gaussian = None

    def init(self, config, input_dims):
        label = "SignalNoiseProcessor"
        self._input_dim = _require_input_dim(input_dims, label)
        if len(input_dims) != 1:
            raise _error(label, "requires exactly one input")
        distribution = config.get("distribution")
        if distribution not in ("uniform", "gaussian"):
            raise _error(label, 'distribution must be "uniform" or "gaussian"')
        self._distribution = distribution
        seed = _to_float(config.get("seed"))
        if not math.isfinite(seed) or seed != int(seed) or seed < 0 or seed > _LCG_MODULUS - 1:
            raise _error(label, "seed must be a uint32 integer")
        self._initial_state = int(seed)
        self._state = self._initial_state
        self._spare_gaussian = None
        self._scale = _coerce_limit_vector(config.get("scale"), self._input_dim, label, "scale")
        self._bias = _coerce_limit_vector(config.get("bias"), self._input_dim, label, "bias")
        self.output_buffers = [np.zeros(self._input_dim, dtype=np.float32)]

    def compute(self, inputs):
        label = "SignalNoiseProcessor"
        source = inputs[0] if inputs else None
        if len(inputs) != 1 or source is None or len(source) != self._input_dim:
            raise _error(label, "input dimensions changed after init")
        if not np.isfinite(source).all():
            raise _error(label, f"input[{_first_nonfinite_index(source)}] must be finite")

        if self._distribution == "uniform":
            sample = self._draw_uniforms(self._input_dim) * 2.0 - 1.0
        else:
            sample = self._draw_gaussians(self._input_dim)
        result = source.astype(np.float64) + self._bias + self._scale * sample
        output = self.output_buffers[0]
        output[:] = result
        if not np.isfinite(output).all():
            raise _error(label, f"output[{_first_nonfinite_index(output)}] must be finite")

    def reset(self):
        self._state = self._initial_state
        self._spare_gaussian = None
        self.output_buffers[0].fill(0)

    def _draw_uniforms(self, count):
        uniforms, self._state = _lcg_uniforms(self._state, count)
        return uniforms

    def _draw_gaussians(self, count):
        """Box-Muller with the TS spare parity kept across compute() calls."""
        spare = self._spare_gaussian
        pending = count - (1 if spare is not None else 0)
        pairs = (pending + 1) // 2
        uniforms, self._state = _lcg_uniforms(self._state, 2 * pairs)
        pairs_view = uniforms.reshape(-1, 2)
        radius = np.sqrt(-2.0 * np.log(1.0 - pairs_view[:, 0]))
        angle = 2.0 * np.pi * pairs_view[:, 1]
        cosine = radius * np.cos(angle)
        sine = radius * np.sin(angle)

        sample = np.empty(count, dtype=np.float64)
        rest = sample
        if spare is not None:
            sample[0] = spare
            rest = sample[1:]
        rest[0::2] = cosine[: (len(rest) + 1) // 2]
        rest[1::2] = sine[: len(rest) // 2]
        self._spare_gaussian = float(sine[-1]) if len(rest) % 2 else None
        return sample


class ClockProcessor:
    """Simulation clock (proc-clock), frame-driven like the TS twin.

    The TS processor only works through computeAsync(context.frame) — seconds
    reads frame.time, steps reads frame.index. The local executor mirrors
    that with the explicit frame channel: ``uses_frame = True`` processors
    receive the control-step frame in compute(inputs, frame). No internal
    counter exists here, so a missing frame is an explicit error.
    """

    uses_frame = True

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._unit = "seconds"

    def init(self, config, input_dims):
        label = "ClockProcessor"
        unit = config.get("unit")
        if input_dims or (unit != "seconds" and unit != "steps"):
            raise _error(label, "expected no inputs and unit seconds or steps")
        self._unit = unit

    def compute(self, inputs, frame=None):
        if frame is None:
            raise _error(
                "ClockProcessor",
                "needs the simulation frame; the executor must pass frame=",
            )
        value = float(frame.time) if self._unit == "seconds" else float(frame.index)
        if not math.isfinite(value) or value < 0:
            raise _error("ClockProcessor", "invalid simulation clock")
        self.output_buffers[0][0] = value

    def reset(self):
        self.output_buffers[0].fill(0)


class PhaseClockProcessor:
    """Unit phase clock: (cos, sin) of 2*pi*step/periodSteps (proc-phase-clock)."""

    def __init__(self):
        self.output_buffers = [np.zeros(2, dtype=np.float32)]
        self._period_steps = 1

    def init(self, config, input_dims):
        label = "PhaseClockProcessor"
        first = input_dims[0] if input_dims else 0
        if first != 1:
            raise _error(label, f"step input must have dimension 1, got {first}")
        period_steps = _to_float(config.get("periodSteps"))
        if not math.isfinite(period_steps) or period_steps != int(period_steps) or period_steps <= 0:
            raise _error(label, "periodSteps must be a positive integer")
        self._period_steps = int(period_steps)

    def compute(self, inputs):
        label = "PhaseClockProcessor"
        source = inputs[0] if inputs else None
        if source is None or len(source) != 1:
            raise _error(label, "Missing scalar step input")
        step = float(source[0])
        if not math.isfinite(step) or step < 0:
            raise _error(label, "step input must be non-negative and finite")
        angle = 2.0 * math.pi * step / self._period_steps
        output = self.output_buffers[0]
        output[0] = math.cos(angle)
        output[1] = math.sin(angle)

    def reset(self):
        """Stateless by design: the phase derives from the step input each compute."""


REGISTRY = {
    "proc-math": MathProcessor,
    "proc-vector-math": VectorMathProcessor,
    "proc-clamp": ClampProcessor,
    "proc-phase-clock": PhaseClockProcessor,
    "proc-clock": ClockProcessor,
    "proc-spatial-math": SpatialMathProcessor,
    "proc-signal-math": SignalMathProcessor,
    "proc-signal-noise": SignalNoiseProcessor,
    "proc-signal-filter": SignalFilterProcessor,
}
