"""Part-type node processors — numpy mirror of the TypeScript part processors.

Each class ports its counterpart from
src/domain/runtime/engine/processors/ one-to-one: config keys keep their
camelCase spelling, validation raises the same error shapes
(``[<ProcessorName>] message``), dimension inference and defaults match the
TS sources, and compute writes into owned ``output_buffers`` without mutating
the aliased inputs. Pass-through nodes (reroute, signal label) replace their
output buffer with the input reference, exactly like the TS implementations,
so downstream nodes keep zero-copy aliasing.

None of these processors owns temporal state, so none implements ``reset``
(the TS classes do not either). ``REGISTRY`` maps the same node type keys as
SimulationEngine.ts's ProcessorRegistry for this group.
"""

import math

import numpy as np


def _processor_error(label, message):
    return ValueError(f"[{label}] {message}")


def _require_input_dim(input_dims, label):
    dim = input_dims[0] if input_dims else 0
    if not dim:
        raise _processor_error(label, "inputDims[0] is required")
    return dim


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


def _raw_is_integer(value):
    """JS ``Number.isInteger`` on a raw config value (no string coercion)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return _is_integer(value)


def _finite_number(raw, fallback, field, label):
    if raw is None or raw == "":
        return fallback
    value = _number(raw)
    if not math.isfinite(value):
        raise _processor_error(label, f"{field} must be finite")
    return value


def _integer_value(raw, fallback, field, label):
    if raw is None or raw == "":
        return fallback
    value = _number(raw)
    if not _is_integer(value):
        raise _processor_error(label, f"{field} must be an integer")
    return int(value)


class RerouteProcessor:
    """Identity waypoint node; the output buffer aliases the input buffer."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._expected_length = 0

    def init(self, config, input_dims):
        dim = _require_input_dim(input_dims, "RerouteProcessor")
        self._expected_length = dim
        self.output_buffers[0] = np.zeros(dim, dtype=np.float32)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error("RerouteProcessor", "Missing input")
        if len(source) != self._expected_length:
            raise _processor_error(
                "RerouteProcessor",
                f"input length changed from {self._expected_length} to {len(source)}",
            )
        self.output_buffers[0] = source


class ReorderProcessor:
    """Gathers the input vector through a configured index list."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._input_dim = 0
        self._indices = np.zeros(0, dtype=np.int64)

    @staticmethod
    def _parse_indices(raw, input_dim):
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raise ValueError("[ReorderProcessor] indices must be a number list")
        values = [_number(value) for value in raw]
        if not values:
            raise ValueError("[ReorderProcessor] indices must not be empty")
        indices = np.empty(len(values), dtype=np.int64)
        for position, value in enumerate(values):
            if not _is_integer(value) or value < 0 or value >= input_dim:
                raise ValueError(
                    "[ReorderProcessor] indices must be integers between "
                    f"0 and {input_dim - 1}"
                )
            indices[position] = int(value)
        return indices

    def init(self, config, input_dims):
        self._input_dim = _require_input_dim(input_dims, "ReorderProcessor")
        self._indices = self._parse_indices(config.get("indices"), self._input_dim)
        self.output_buffers[0] = np.zeros(len(self._indices), dtype=np.float32)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise ValueError("[ReorderProcessor] Missing input")
        if len(source) != self._input_dim:
            raise ValueError(
                "[ReorderProcessor] input length changed from "
                f"{self._input_dim} to {len(source)}"
            )
        self.output_buffers[0][:] = source[self._indices]


class ResizeProcessor:
    """Bilinear image resize with optional axis flips (hwc or chw layout)."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._input_width = 0
        self._input_height = 0
        self._output_width = 0
        self._output_height = 0
        self._channels = 1
        self._layout = "hwc"
        self._flip_x = False
        self._flip_y = False
        self._x0 = np.zeros(0, dtype=np.int64)
        self._x1 = np.zeros(0, dtype=np.int64)
        self._y0 = np.zeros(0, dtype=np.int64)
        self._y1 = np.zeros(0, dtype=np.int64)
        self._wx = np.zeros(0, dtype=np.float32)
        self._wy = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _positive_integer(raw, field):
        value = _number(raw)
        if not _is_integer(value) or value <= 0:
            raise _processor_error("ResizeProcessor", f"{field} must be a positive integer")
        return int(value)

    @staticmethod
    def _resize_layout(raw):
        if raw is None or raw == "":
            return "hwc"
        if raw in ("hwc", "chw"):
            return raw
        raise _processor_error("ResizeProcessor", 'layout must be "hwc" or "chw"')

    @staticmethod
    def _flip_flag(raw):
        return raw is True or raw == "true" or raw == 1 or raw == "1"

    def init(self, config, input_dims):
        input_dim = _require_input_dim(input_dims, "ResizeProcessor")
        raw_width = config.get("inputWidth")
        if raw_width is None:
            raw_width = config.get("width")
        self._input_width = self._positive_integer(raw_width, "inputWidth")
        raw_height = config.get("inputHeight")
        if raw_height is None:
            raw_height = config.get("height")
        self._input_height = self._positive_integer(raw_height, "inputHeight")
        self._output_width = self._positive_integer(config.get("outputWidth"), "outputWidth")
        self._output_height = self._positive_integer(
            config.get("outputHeight"), "outputHeight"
        )
        self._layout = self._resize_layout(config.get("layout"))
        self._flip_x = self._flip_flag(config.get("flipX"))
        self._flip_y = self._flip_flag(config.get("flipY"))

        frame_pixels = self._input_width * self._input_height
        raw_channels = config.get("channels")
        if raw_channels is None or raw_channels == "":
            channels = input_dim / frame_pixels
        else:
            channels = _number(raw_channels)
        if not _is_integer(channels) or channels <= 0:
            raise _processor_error("ResizeProcessor", "channels must be a positive integer")
        self._channels = int(channels)

        expected_input_dim = frame_pixels * self._channels
        if input_dim != expected_input_dim:
            raise _processor_error(
                "ResizeProcessor",
                f"input length {input_dim} does not match "
                f"inputWidth*inputHeight*channels {expected_input_dim}",
            )

        self.output_buffers[0] = np.zeros(
            self._output_width * self._output_height * self._channels, dtype=np.float32
        )
        self._precompute_sampling()

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error("ResizeProcessor", "Missing input")
        expected_input_dim = self._input_width * self._input_height * self._channels
        if len(source) != expected_input_dim:
            raise _processor_error(
                "ResizeProcessor",
                f"input length changed from {expected_input_dim} to {len(source)}",
            )

        output = self.output_buffers[0]
        if (
            self._input_width == self._output_width
            and self._input_height == self._output_height
            and not self._flip_x
            and not self._flip_y
        ):
            output[:] = source
            return
        self._resize_bilinear(source, output)

    def _precompute_sampling(self):
        scale_x = self._input_width / self._output_width
        scale_y = self._input_height / self._output_height
        out_x = np.arange(self._output_width, dtype=np.float64)
        out_y = np.arange(self._output_height, dtype=np.float64)
        sample_x = self._output_width - 1 - out_x if self._flip_x else out_x
        sample_y = self._output_height - 1 - out_y if self._flip_y else out_y
        src_x = (sample_x + 0.5) * scale_x - 0.5
        src_y = (sample_y + 0.5) * scale_y - 0.5
        self._x0 = np.clip(np.floor(src_x), 0, self._input_width - 1).astype(np.int64)
        self._x1 = np.minimum(self._x0 + 1, self._input_width - 1)
        # Weights are stored as float32 and widened back to float64 at compute
        # time, matching the TS Float32Array storage read back as JS doubles.
        self._wx = (src_x - self._x0).astype(np.float32)
        self._y0 = np.clip(np.floor(src_y), 0, self._input_height - 1).astype(np.int64)
        self._y1 = np.minimum(self._y0 + 1, self._input_height - 1)
        self._wy = (src_y - self._y0).astype(np.float32)

    def _resize_bilinear(self, source, output):
        wx = self._wx.astype(np.float64)
        wy = self._wy.astype(np.float64)
        inv_wx = 1.0 - wx
        inv_wy = 1.0 - wy
        if self._layout == "chw":
            frames = source.reshape(
                self._channels, self._input_height, self._input_width
            ).astype(np.float64)
            top = (
                frames[:, self._y0, :] * inv_wy[None, :, None]
                + frames[:, self._y1, :] * wy[None, :, None]
            )
            result = (
                top[:, :, self._x0] * inv_wx[None, None, :]
                + top[:, :, self._x1] * wx[None, None, :]
            )
        else:
            frames = source.reshape(
                self._input_height, self._input_width, self._channels
            ).astype(np.float64)
            top = (
                frames[self._y0, :, :] * inv_wy[:, None, None]
                + frames[self._y1, :, :] * wy[:, None, None]
            )
            result = (
                top[:, self._x0, :] * inv_wx[:, None]
                + top[:, self._x1, :] * wx[:, None]
            )
        output[:] = result.reshape(-1)


class MuxProcessor:
    """Concatenates multiple input vectors into one, tracking segments."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self.output_segments = [[]]
        self._offsets = []

    def init(self, config, input_dims):
        raw_channels = config.get("channels")
        channels = _number(4 if raw_channels is None else raw_channels)
        if not _is_integer(channels) or channels <= 0:
            raise ValueError("[MuxProcessor] channels must be a positive integer")
        channels = int(channels)
        if len(input_dims) != channels:
            raise ValueError(
                f"[MuxProcessor] expected {channels} inputs, got {len(input_dims)}"
            )

        offsets = []
        total_dim = 0
        for slot, dim in enumerate(input_dims):
            if not dim:
                raise ValueError(f"[MuxProcessor] Missing input at slot {slot}")
            offsets.append(total_dim)
            total_dim += dim
        self._offsets = offsets

        self.output_buffers[0] = np.zeros(total_dim, dtype=np.float32)
        self.output_segments[0] = [dim for dim in input_dims if dim > 0]

    def compute(self, inputs):
        output = self.output_buffers[0]
        for slot in range(len(inputs)):
            source = inputs[slot]
            if source is None:
                raise ValueError(f"[MuxProcessor] Missing input at slot {slot}")
            start = self._offsets[slot]
            output[start : start + len(source)] = source


class DemuxProcessor:
    """Splits one vector into several fixed-size output buffers."""

    def __init__(self):
        self.output_buffers = []
        self._input_dim = 0

    def init(self, config, input_dims):
        self._input_dim = _require_input_dim(input_dims, "DemuxProcessor")

        sizes = config.get("sizes")
        if isinstance(sizes, (list, tuple, np.ndarray)) and len(sizes) > 0:
            raw_sizes = list(sizes)
        else:
            raw_channels = config.get("channels")
            if raw_channels is None:
                channel_count = 4
            else:
                channel_count = _number(raw_channels)
            if not _is_integer(channel_count) or channel_count < 0:
                raise _processor_error(
                    "DemuxProcessor", "channels must be a non-negative integer"
                )
            raw_sizes = [1] * int(channel_count)

        for size in raw_sizes:
            if not _raw_is_integer(size) or size <= 0:
                raise _processor_error(
                    "DemuxProcessor", "output sizes must be positive integers"
                )
        total_size = sum(int(size) for size in raw_sizes)
        if total_size != self._input_dim:
            raise _processor_error(
                "DemuxProcessor",
                f"output sizes total {total_size} does not match "
                f"input dimension {self._input_dim}",
            )

        self.output_buffers = [
            np.zeros(int(size), dtype=np.float32) for size in raw_sizes
        ]

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise ValueError("[DemuxProcessor] Missing input")
        if len(source) != self._input_dim:
            raise ValueError(
                f"[DemuxProcessor] input length {len(source)} does not match "
                f"compiled dimension {self._input_dim}"
            )
        offset = 0
        for output in self.output_buffers:
            output[:] = source[offset : offset + len(output)]
            offset += len(output)


class OneHotProcessor:
    """Encodes a scalar index input into a one-hot float vector."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]

    def init(self, config, input_dims):
        input_dim = input_dims[0] if input_dims else 0
        if input_dim != 1:
            raise _processor_error(
                "OneHotProcessor", f"input must have dimension 1, got {input_dim}"
            )
        size = _number(config.get("size"))
        if not _is_integer(size) or size <= 0:
            raise _processor_error("OneHotProcessor", "size must be a positive integer")
        self.output_buffers[0] = np.zeros(int(size), dtype=np.float32)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error("OneHotProcessor", "Missing input")
        size = len(self.output_buffers[0])
        index = float(source[0])
        if not _is_integer(index) or index < 0 or index >= size:
            raise _processor_error(
                "OneHotProcessor",
                f"index must be an integer between 0 and {size - 1}",
            )
        self.output_buffers[0].fill(0.0)
        self.output_buffers[0][int(index)] = 1.0


class OneHotIndexProcessor:
    """Emits the first input index above a threshold (argmax with cutoff)."""

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._input_dim = 0
        self._threshold = 0.0
        self._default_index = 0

    def init(self, config, input_dims):
        self._input_dim = _require_input_dim(input_dims, "OneHotIndexProcessor")
        self._threshold = _finite_number(
            config.get("threshold"), 0.0, "threshold", "OneHotIndexProcessor"
        )
        self._default_index = _integer_value(
            config.get("defaultIndex"), 0, "defaultIndex", "OneHotIndexProcessor"
        )
        self.output_buffers[0] = np.zeros(1, dtype=np.float32)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error("OneHotIndexProcessor", "Missing input")
        if len(source) != self._input_dim:
            raise _processor_error(
                "OneHotIndexProcessor",
                f"input length changed from {self._input_dim} to {len(source)}",
            )

        values = np.asarray(source, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.all():
            first_bad = int(np.argmin(finite))
            raise _processor_error(
                "OneHotIndexProcessor", f"input[{first_bad}] must be finite"
            )
        hits = np.nonzero(values > self._threshold)[0]
        best_index = int(hits[0]) if hits.size else self._default_index
        self.output_buffers[0][0] = best_index


class WhereProcessor:
    """Selects between two equal-dimension branches on a scalar condition."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._dim = 0

    def init(self, config, input_dims):
        if len(input_dims) != 3:
            raise _processor_error(
                "WhereProcessor",
                "requires condition, when_true, and when_false inputs",
            )
        if input_dims[0] != 1:
            raise _processor_error(
                "WhereProcessor", f"condition input must be scalar, got {input_dims[0]}"
            )
        true_dim = input_dims[1]
        false_dim = input_dims[2]
        if not true_dim or true_dim != false_dim:
            raise _processor_error(
                "WhereProcessor",
                "branch inputs must have the same positive dimension, "
                f"got {true_dim} and {false_dim}",
            )
        self._dim = true_dim
        self.output_buffers[0] = np.zeros(true_dim, dtype=np.float32)

    def compute(self, inputs):
        condition = inputs[0] if len(inputs) > 0 else None
        when_true = inputs[1] if len(inputs) > 1 else None
        when_false = inputs[2] if len(inputs) > 2 else None
        if condition is None or len(condition) != 1 or not np.isfinite(condition[0]):
            raise _processor_error(
                "WhereProcessor", "condition input must be a finite scalar"
            )
        if (
            when_true is None
            or len(when_true) != self._dim
            or when_false is None
            or len(when_false) != self._dim
        ):
            raise _processor_error(
                "WhereProcessor", "branch input dimensions changed after init"
            )
        self.output_buffers[0][:] = when_true if condition[0] != 0 else when_false


class LogicProcessor:
    """Combines connected scalar booleans with all / any / not."""

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._operator = "all"

    def init(self, config, input_dims):
        operator = config.get("operator")
        if not isinstance(operator, str) or operator not in ("all", "any", "not"):
            raise _processor_error(
                "LogicProcessor", "operator must be one of all, any, not"
            )
        if not input_dims or any(dim != 1 for dim in input_dims):
            raise _processor_error(
                "LogicProcessor", "all inputs must be connected scalar signals"
            )
        if operator == "not" and len(input_dims) != 1:
            raise _processor_error("LogicProcessor", "not requires exactly one input")
        self._operator = operator
        self.output_buffers[0] = np.zeros(1, dtype=np.float32)

    def compute(self, inputs):
        truthy = []
        for index in range(len(inputs)):
            signal = inputs[index]
            if signal is None or len(signal) != 1 or not np.isfinite(signal[0]):
                raise _processor_error(
                    "LogicProcessor", f"input[{index}] must be a finite scalar"
                )
            truthy.append(bool(signal[0] != 0))
        if self._operator == "not":
            result = not truthy[0] if truthy else False
        elif self._operator == "all":
            result = all(truthy)
        else:
            result = any(truthy)
        self.output_buffers[0][0] = 1.0 if result else 0.0


class CompareProcessor:
    """Compares one scalar against a config constant or a second scalar."""

    _OPERATORS = ("gt", "gte", "lt", "lte", "eq", "ne")

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._operator = "eq"
        self._constant = None

    def init(self, config, input_dims):
        operator = config.get("operator")
        if not isinstance(operator, str) or operator not in self._OPERATORS:
            raise _processor_error(
                "CompareProcessor", "operator must be one of gt, gte, lt, lte, eq, ne"
            )
        if len(input_dims) not in (1, 2):
            raise _processor_error(
                "CompareProcessor", "requires one or two scalar inputs"
            )
        for index, dim in enumerate(input_dims):
            if dim != 1:
                raise _processor_error(
                    "CompareProcessor", f"inputDims[{index}] must be 1, got {dim}"
                )

        has_constant = "value" in config
        if has_constant == (len(input_dims) == 2):
            raise _processor_error(
                "CompareProcessor",
                "constant comparisons require exactly one input"
                if has_constant
                else "signal comparisons require exactly two inputs",
            )
        if has_constant:
            value = config["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
            ):
                raise _processor_error("CompareProcessor", "value must be finite")
            self._constant = float(value)
        else:
            self._constant = None
        self._operator = operator
        self.output_buffers[0] = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _compare(left, right, operator):
        if operator == "gt":
            return left > right
        if operator == "gte":
            return left >= right
        if operator == "lt":
            return left < right
        if operator == "lte":
            return left <= right
        if operator == "eq":
            return left == right
        if operator == "ne":
            return left != right
        raise _processor_error(
            "CompareProcessor", f"Unsupported operator: {str(operator)}"
        )

    def compute(self, inputs):
        left = inputs[0] if inputs else None
        right = inputs[1] if len(inputs) > 1 else None
        if left is None or len(left) != 1:
            raise _processor_error("CompareProcessor", "left input must remain scalar")
        if not np.isfinite(left[0]):
            raise _processor_error("CompareProcessor", "left input must be finite")
        if self._constant is not None:
            right_value = self._constant
        elif right is not None:
            right_value = float(right[0])
        else:
            right_value = None
        if right_value is None or not math.isfinite(right_value):
            raise _processor_error(
                "CompareProcessor", "right input must be a finite scalar"
            )
        self.output_buffers[0][0] = (
            1.0 if self._compare(float(left[0]), right_value, self._operator) else 0.0
        )


class SignalLabelProcessor:
    """Labelled pass-through for signal send/receive virtual-edge nodes."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self.output_segments = None
        self._label = ""
        self._expected_length = 0

    def _label_for_error(self):
        return f'"{self._label}"' if self._label else "unlabeled signal label node"

    def init(self, config, input_dims):
        input_dim = input_dims[0] if input_dims else 0
        label = config.get("signalLabel")
        self._label = label.strip() if isinstance(label, str) else ""
        if not input_dim:
            raise ValueError(
                f"[SignalLabelProcessor] {self._label_for_error()} "
                "requires an input signal"
            )

        self._expected_length = input_dim
        self.output_buffers[0] = np.zeros(input_dim, dtype=np.float32)
        upstream = config.get("upstreamInputSegments")
        if not isinstance(upstream, list):
            upstream = config.get("inputSegments")
        input_segments = upstream[0] if isinstance(upstream, list) and upstream else None
        if isinstance(input_segments, list):
            self.output_segments = [list(input_segments)]
        else:
            self.output_segments = None

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise ValueError(
                f"[SignalLabelProcessor] {self._label_for_error()} missing input"
            )
        if len(source) != self._expected_length:
            raise ValueError(
                f"[SignalLabelProcessor] {self._label_for_error()} input length "
                f"changed from {self._expected_length} to {len(source)}"
            )
        self.output_buffers[0] = source


REGISTRY = {
    "proc-reroute": RerouteProcessor,
    "proc-reorder": ReorderProcessor,
    "proc-resize": ResizeProcessor,
    "proc-mux": MuxProcessor,
    "proc-demux": DemuxProcessor,
    "proc-one-hot": OneHotProcessor,
    "proc-onehot-index": OneHotIndexProcessor,
    "proc-where": WhereProcessor,
    "proc-logic": LogicProcessor,
    "proc-compare": CompareProcessor,
    "proc-signal-send": SignalLabelProcessor,
    "proc-signal-receive": SignalLabelProcessor,
}
