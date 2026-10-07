"""Temporal processors: history stacking and signal delay.

Python/numpy mirror of the TypeScript engine processors
(src/domain/runtime/engine/processors/HistoryProcessor.ts and
DelayProcessor.ts) with identical config keys (including the snake_case
aliases the TS parser accepts), defaults, error messages, ring-buffer timing
and IsaacGym / IsaacLab stacking layouts. Gather indices are precomputed at
init so each compute step is a single vectorized numpy pass; input buffers
are read-only aliases and are never written back.
"""

import math
import re

import numpy as np

_HISTORY = "HistoryProcessor"
_DELAY = "DelayProcessor"

_MISSING = object()
_NAN = float("nan")


def _processor_error(label, message):
    return ValueError(f"[{label}] {message}")


def _require_input_dim(input_dims, label):
    dim = input_dims[0] if input_dims else 0
    if not dim:
        raise _processor_error(label, "inputDims[0] is required")
    return dim


def _to_number(raw):
    """Coerce a config value like JS ``Number()`` (null -> 0, garbage -> NaN)."""
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
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
            return _NAN
    return _NAN


def _is_integer(value):
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def _coalesce(config, keys, default=_MISSING):
    """First non-nullish value among ``keys`` (JS ``a ?? b`` chain)."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _is_number_like(raw):
    return isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool)


class HistoryProcessor:
    """Stacks recent samples of one signal into a policy history window.

    IsaacGym layout keeps whole frames together (frame-major); IsaacLab layout
    runs each observation term through time before the next term (term-major,
    using the configured or upstream input segments).
    """

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._ring = np.zeros(0, dtype=np.float32)
        self._stack_size = 0
        self._source_buffer_size = 0
        self._item_size = 0
        self._input_segments = []
        self._sample_offsets = None
        self._slot_deltas = np.zeros(0, dtype=np.intp)
        self._frame_of = np.zeros(0, dtype=np.intp)
        self._within_of = np.zeros(0, dtype=np.intp)
        self._cursor = 0
        self._sample_decimation = 1
        self._step_count = 0
        self._order = "oldest-first"
        self._stack_mode = "isaacgym"
        self._has_sample = False
        self._initial_fill_value = None

    def init(self, config, input_dims):
        stack_size = _to_number(config.get("stackSize", _MISSING))
        if not _is_integer(stack_size) or stack_size <= 0:
            raise _processor_error(_HISTORY, "stackSize must be a positive integer")

        raw_source = config.get("sourceBufferSize", _MISSING)
        if raw_source is _MISSING or raw_source is None:
            source_buffer_size = float(stack_size)
        else:
            source_buffer_size = _to_number(raw_source)
        if not _is_integer(source_buffer_size) or source_buffer_size < stack_size:
            raise _processor_error(
                _HISTORY,
                "sourceBufferSize must be an integer greater than or equal to stackSize",
            )

        sample_decimation = _to_number(
            _coalesce(config, ("sampleDecimation", "sample_decimation"), default=1)
        )
        if not _is_integer(sample_decimation) or sample_decimation <= 0:
            raise _processor_error(_HISTORY, "sampleDecimation must be a positive integer")

        order = _coalesce(config, ("order", "historyOrder"), default="oldest-first")
        if order != "oldest-first" and order != "newest-first":
            raise _processor_error(
                _HISTORY, "order must be \"oldest-first\" or \"newest-first\""
            )

        stack_mode = _coalesce(config, ("stackMode", "historyStackMode"), default="isaacgym")
        if stack_mode != "isaacgym" and stack_mode != "isaaclab":
            raise _processor_error(
                _HISTORY, "stackMode must be \"isaacgym\" or \"isaaclab\""
            )

        self._stack_size = int(stack_size)
        self._source_buffer_size = int(source_buffer_size)
        self._sample_offsets = self._normalize_sample_offsets(
            config.get("sampleOffsets", _MISSING)
        )
        self._sample_decimation = int(sample_decimation)
        self._order = order
        self._stack_mode = stack_mode
        self._step_count = 0
        self._has_sample = False
        self._item_size = _require_input_dim(input_dims, _HISTORY)
        self._initial_fill_value = self._parse_initial_fill_value(
            _coalesce(config, ("initialFillValue", "initial_fill_value")), self._item_size
        )
        configured = self._read_configured_input_segments(config.get("inputSegments", _MISSING))
        upstream = config.get("upstreamInputSegments", _MISSING)
        upstream_first = upstream[0] if isinstance(upstream, list) and upstream else None
        self._input_segments = self._normalize_input_segments(
            configured if configured is not None else upstream_first, self._item_size
        )

        self.output_buffers = [np.zeros(self._stack_size * self._item_size, dtype=np.float32)]
        self._ring = np.zeros(self._source_buffer_size * self._item_size, dtype=np.float32)
        self._cursor = 0
        self._build_gather()

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error(_HISTORY, "Missing input")
        if len(source) != self._item_size:
            raise _processor_error(
                _HISTORY,
                f"input length {len(source)} does not match itemSize {self._item_size}",
            )

        if self._step_count % self._sample_decimation == 0:
            if self._has_sample:
                start = self._cursor * self._item_size
                self._ring[start : start + self._item_size] = source
            else:
                self._fill_initial_history(source)
                self._has_sample = True
            self._cursor = (self._cursor + 1) % self._source_buffer_size
        self._step_count += 1

        ring_indices = (self._cursor + self._slot_deltas) % self._source_buffer_size
        frame_base = ring_indices * self._item_size
        self.output_buffers[0][:] = self._ring[frame_base[self._frame_of] + self._within_of]

    def reset(self):
        self._ring.fill(0.0)
        self.output_buffers[0].fill(0.0)
        self._cursor = 0
        self._step_count = 0
        self._has_sample = False

    def _build_gather(self):
        """Precompute the gather template so compute is one numpy take."""
        slots = np.arange(self._stack_size, dtype=np.intp)
        if self._sample_offsets is not None:
            self._slot_deltas = np.asarray(self._sample_offsets, dtype=np.intp)
        elif self._order == "newest-first":
            self._slot_deltas = -(slots + 1)
        else:
            self._slot_deltas = -(self._stack_size - slots)

        if self._stack_mode == "isaaclab":
            frame_of = []
            within_of = []
            input_offset = 0
            for segment_size in self._input_segments:
                frame_of.append(np.repeat(slots, segment_size))
                within_of.append(
                    input_offset
                    + np.tile(np.arange(segment_size, dtype=np.intp), self._stack_size)
                )
                input_offset += segment_size
            self._frame_of = np.concatenate(frame_of)
            self._within_of = np.concatenate(within_of)
        else:
            self._frame_of = np.repeat(slots, self._item_size)
            self._within_of = np.tile(np.arange(self._item_size, dtype=np.intp), self._stack_size)

    def _fill_initial_history(self, source):
        fill = (
            self._initial_fill_value
            if self._initial_fill_value is not None
            else np.asarray(source, dtype=np.float32)
        )
        self._ring[:] = np.tile(fill, self._source_buffer_size)

    def _read_configured_input_segments(self, raw):
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            return None
        first = raw[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return list(first)
        return list(raw)

    def _normalize_input_segments(self, segments, item_size):
        if segments is None or len(segments) == 0:
            return [item_size]

        normalized = [_to_number(segment) for segment in segments]
        if any(not _is_integer(segment) or segment <= 0 for segment in normalized):
            raise _processor_error(_HISTORY, "inputSegments must contain positive integers")

        total = sum(normalized)
        if total != item_size:
            raise _processor_error(
                _HISTORY,
                f"inputSegments total {int(total)} does not match itemSize {item_size}",
            )
        return [int(segment) for segment in normalized]

    def _normalize_sample_offsets(self, raw):
        if raw is None or raw is _MISSING or (isinstance(raw, str) and raw == ""):
            return None
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raise _processor_error(_HISTORY, "sampleOffsets must be a number array")

        values = [_to_number(value) for value in raw]
        if len(values) != self._stack_size:
            raise _processor_error(
                _HISTORY,
                f"sampleOffsets length {len(values)} must match stackSize {self._stack_size}",
            )
        if any(not _is_integer(value) or value == 0 for value in values):
            raise _processor_error(_HISTORY, "sampleOffsets must contain non-zero integers")
        if any(abs(value) > self._source_buffer_size for value in values):
            raise _processor_error(
                _HISTORY,
                f"sampleOffsets must be within sourceBufferSize {self._source_buffer_size}",
            )
        return [int(value) for value in values]

    def _parse_initial_fill_value(self, raw, item_size):
        if raw is _MISSING or raw is None or (isinstance(raw, str) and raw == ""):
            return None

        if _is_number_like(raw):
            values = [float(raw)]
        elif isinstance(raw, str):
            values = [_to_number(part) for part in re.split(r"[\s,]+", raw) if part]
        elif isinstance(raw, (list, tuple, np.ndarray)):
            values = [float(value) for value in raw]
        else:
            values = None

        if not values or any(not math.isfinite(value) for value in values):
            raise _processor_error(
                _HISTORY, "initialFillValue must be a finite number or number list"
            )
        if len(values) != 1 and len(values) != item_size:
            raise _processor_error(
                _HISTORY, f"initialFillValue must contain 1 or {item_size} numbers"
            )

        fill = np.zeros(item_size, dtype=np.float32)
        fill[:] = values[0] if len(values) == 1 else values
        return fill


class DelayProcessor:
    """Delays a signal by ``totalSteps``, subtracting a connected sensor age.

    ``primeWithFirstSample`` fills the not-yet-formed delay history with the
    first real sample instead of the configured fill value.
    """

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self.output_segments = None
        self._dim = 0
        self._total_steps = 0
        self._ring = np.zeros(0, dtype=np.float32)
        self._fill = np.zeros(0, dtype=np.float32)
        self._cursor = 0
        self._prime_with_first_sample = False
        self._primed = False

    def init(self, config, input_dims):
        self._dim = _require_input_dim(input_dims, _DELAY)
        age_dim = input_dims[1] if len(input_dims) > 1 else None
        if age_dim is not None and age_dim != 0 and age_dim != 1:
            raise _processor_error(_DELAY, "age input must have dimension 1 when connected")

        total_steps = _to_number(config.get("totalSteps", _MISSING))
        if not _is_integer(total_steps) or total_steps < 0:
            raise _processor_error(_DELAY, "totalSteps must be a non-negative integer")

        if "fill" not in config:
            raise _processor_error(_DELAY, "fill is required")

        raw_prime = config.get("primeWithFirstSample", _MISSING)
        if raw_prime is not _MISSING and not isinstance(raw_prime, bool):
            raise _processor_error(_DELAY, "primeWithFirstSample must be a boolean")

        self._total_steps = int(total_steps)
        self._prime_with_first_sample = raw_prime is True
        self._fill = self._finite_fill(config["fill"], self._dim)
        self._ring = np.zeros((self._total_steps + 1) * self._dim, dtype=np.float32)
        self.output_buffers = [self._fill.copy()]
        upstream = config.get("upstreamInputSegments", _MISSING)
        segments = (
            upstream[0]
            if isinstance(upstream, list) and upstream and isinstance(upstream[0], list)
            else None
        )
        self.output_segments = [list(segments) if segments is not None else [self._dim]]
        self.reset()

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error(_DELAY, "Missing source input")
        if len(source) != self._dim:
            raise _processor_error(
                _DELAY,
                f"source input length changed from {self._dim} to {len(source)}",
            )

        raw_age = 0.0
        if len(inputs) > 1 and inputs[1] is not None:
            raw_age = float(inputs[1][0]) if len(inputs[1]) else _NAN
        if not _is_integer(raw_age) or raw_age < 0:
            raise _processor_error(_DELAY, "age input must be a non-negative integer")
        age = min(self._total_steps, int(raw_age))
        delay_steps = max(0, self._total_steps - age)
        slots = self._total_steps + 1

        if self._prime_with_first_sample and not self._primed:
            self._ring[:] = np.tile(np.asarray(source, dtype=np.float32), slots)
            self._primed = True
        start = self._cursor * self._dim
        self._ring[start : start + self._dim] = source
        self._cursor = (self._cursor + 1) % slots
        read_slot = (self._cursor - delay_steps - 1 + slots) % slots
        self.output_buffers[0][:] = self._ring[read_slot * self._dim : (read_slot + 1) * self._dim]

    def reset(self):
        self._ring[:] = np.tile(self._fill, self._total_steps + 1)
        self.output_buffers[0][:] = self._fill
        self._cursor = 0
        self._primed = False

    def _finite_fill(self, raw, dim):
        if _is_number_like(raw):
            value = float(raw)
            if not math.isfinite(value):
                raise _processor_error(_DELAY, "fill must be finite")
            return np.full(dim, value, dtype=np.float32)
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raise _processor_error(
                _DELAY, f"fill must be a finite number or a {dim}-value array"
            )
        values = [float(value) for value in raw]
        if len(values) != dim or any(not math.isfinite(value) for value in values):
            raise _processor_error(_DELAY, f"fill must contain exactly {dim} finite numbers")
        return np.asarray(values, dtype=np.float32)


REGISTRY = {
    "proc-history": HistoryProcessor,
    "proc-delay": DelayProcessor,
}
