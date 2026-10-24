"""Output/action node processors — numpy mirror of the TypeScript processors.

Each class ports its counterpart from
src/domain/runtime/engine/processors/ one-to-one: config keys keep their
camelCase spelling, validation raises the same error shapes
(``[<ProcessorName>] message``), dimension inference and defaults match the
TS sources, and compute writes into owned ``output_buffers`` without mutating
the aliased inputs. Deliberate local-runtime differences, all documented:

- ``ActionHookProcessor`` invokes the injected Python callable directly from
  ``compute`` (the browser raises "requires stepAsync" because JS hooks may
  be async; local hooks are synchronous). The hook is injected as the
  ``run`` config key or reassigned as the public ``run`` attribute and is
  called as ``run(action)`` — hosts that need the step frame inject a
  closure. Hook exceptions terminate the control step; there is no fallback.
- Motion processors do not load data: on the TypeScript side the host loads
  motion bytes (fetch / asset registry / NPZ). Locally the host loads the
  motion and injects it as the ``motionData`` config key — a mapping with
  the snake_case shape returned by ``botlab_host.loadMotion`` (frame_count,
  body_count, dof_count, fps, joint_position, joint_velocity, body_position,
  body_quaternion, body_linear_velocity, body_angular_velocity, optional
  joint_names / body_names, flat float buffers).
- ``MotionReferenceProcessor`` keeps the legacy 29 DOF / 67-dim
  frame-increment contract; ``MotionSampleProcessor`` keeps the arbitrary
  DOF absolute-channel sampling contract of ``Motion.sample()``.
- The scope trace store is a plain in-process mirror of
  src/domain/runtime/scopeTraceStore.ts (rolling window + finite check);
  UI listeners/notify throttling are browser-only and not ported.

``REGISTRY`` maps the same node type keys as SimulationEngine.ts's
ProcessorRegistry for this group (including the scene-robot / scene-terrain
scene nodes that share ControlMergeProcessor).
"""

import re
import time
from collections import deque

import numpy as np

from ._motion_reference import MotionReferenceProcessor, MotionSampleProcessor
from ._output_common import (
    _first_bad_index,
    _is_blank,
    _number,
    _processor_error,
    _require_input_dim,
)
from ._pd import PDProcessor

MAX_SCOPE_SAMPLES = 1200


# ---------------------------------------------------------------------------
# Scope trace store (mirror of src/domain/runtime/scopeTraceStore.ts)
# ---------------------------------------------------------------------------

_scope_traces = {}


def _make_labels(count, labels=None):
    result = []
    for index in range(count):
        label = ""
        if labels is not None and index < len(labels) and isinstance(labels[index], str):
            label = labels[index].strip()
        result.append(label if label else f"CH{index + 1}")
    return result


def append_scope_inputs(node_id, inputs, labels=None):
    """Append one sample per channel to the rolling trace of ``node_id``."""
    if not node_id:
        raise ValueError("[scopeTraceStore] nodeId is required")

    channel_count = sum(len(source) for source in inputs)
    if channel_count == 0:
        return

    now = time.perf_counter()
    trace = _scope_traces.get(node_id)
    if trace is None or len(trace["series"]) != channel_count:
        trace = {
            "started": now,
            "times": deque(maxlen=MAX_SCOPE_SAMPLES),
            "series": [deque(maxlen=MAX_SCOPE_SAMPLES) for _ in range(channel_count)],
            "labels": _make_labels(channel_count, labels),
            "sample_count": 0,
        }
        _scope_traces[node_id] = trace
    elif labels is not None and len(labels) == channel_count:
        trace["labels"] = _make_labels(channel_count, labels)

    trace["times"].append(now - trace["started"])
    channel_index = 0
    for source in inputs:
        values = np.asarray(source, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.all():
            raise ValueError(
                "[scopeTraceStore] Scope node "
                f'"{node_id}" received a non-finite value on channel '
                f"{channel_index + _first_bad_index(~finite) + 1}"
            )
        values_list = values.tolist()
        channels = trace["series"][channel_index : channel_index + len(values_list)]
        for series, value in zip(channels, values_list):
            series.append(value)
        channel_index += len(values_list)

    trace["sample_count"] += 1


def clear_scope_trace(node_id):
    _scope_traces.pop(node_id, None)


def scope_trace_snapshot(node_id):
    trace = _scope_traces.get(node_id)
    if trace is None:
        return {"times": [], "series": [], "labels": [], "sample_count": 0}
    return {
        "times": list(trace["times"]),
        "series": [list(series) for series in trace["series"]],
        "labels": list(trace["labels"]),
        "sample_count": trace["sample_count"],
    }


# ---------------------------------------------------------------------------
# ActionProcessor (out-action)
# ---------------------------------------------------------------------------


class ActionProcessor:
    """Sink node for control signals; the physical engine reads its input."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]

    def init(self, config, input_dims):
        dim = input_dims[0] if input_dims else 0
        self.output_buffers[0] = np.zeros(dim, dtype=np.float32)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            return
        if len(self.output_buffers[0]) != len(source):
            self.output_buffers[0] = np.array(source, dtype=np.float32)
        else:
            self.output_buffers[0][:] = source


# ---------------------------------------------------------------------------
# ActionHookProcessor (proc-action-hook)
# ---------------------------------------------------------------------------


class ActionHookProcessor:
    """The single per-plan action hook boundary between policy and output.

    Without a hook it copies its input; with a hook (``run`` config key or
    public ``run`` attribute, injected by the host) every control step calls
    ``run(action)`` once and requires a same-length, all-finite array-like
    return value. Any hook exception or invalid return terminates the step —
    the processor never falls back to the unmodified input.
    """

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self.run = None
        self._expected_length = 0

    def init(self, config, input_dims):
        self._expected_length = _require_input_dim(input_dims, "ActionHookProcessor")
        run = config.get("run")
        if run is not None and not callable(run):
            raise _processor_error(
                "ActionHookProcessor", "run must be a function when provided"
            )
        self.run = run
        self.output_buffers[0] = np.zeros(self._expected_length, dtype=np.float32)

    def compute(self, inputs):
        source = self._input(inputs)
        run = self.run
        if run is None:
            self.output_buffers[0][:] = source
            return

        result = run(source)
        if result is None or not hasattr(result, "__len__"):
            raise _processor_error(
                "ActionHookProcessor", "run must return an array-like value"
            )
        if len(result) != self._expected_length:
            raise _processor_error(
                "ActionHookProcessor",
                f"run result length {len(result)} does not match "
                f"input length {self._expected_length}",
            )
        try:
            values = np.asarray(result, dtype=np.float64)
        except (TypeError, ValueError):
            raise _processor_error(
                "ActionHookProcessor", "run must return an array-like value"
            ) from None
        if values.ndim != 1:
            raise _processor_error(
                "ActionHookProcessor", "run must return an array-like value"
            )
        finite = np.isfinite(values)
        if not finite.all():
            raise _processor_error(
                "ActionHookProcessor",
                f"run result[{_first_bad_index(~finite)}] must be finite",
            )
        self.output_buffers[0][:] = values

    def _input(self, inputs):
        source = inputs[0] if inputs else None
        if source is None:
            raise _processor_error("ActionHookProcessor", "Missing input")
        if len(source) != self._expected_length:
            raise _processor_error(
                "ActionHookProcessor",
                f"input length changed from {self._expected_length} "
                f"to {len(source)}",
            )
        return source


# ---------------------------------------------------------------------------
# ControlMergeProcessor (out-mujoco, scene-robot, scene-terrain)
# ---------------------------------------------------------------------------


def parse_joint2_motor_idx(raw, length, label):
    """Parse a joint -> motor index permutation (robotControlOrder.ts)."""
    if _is_blank(raw):
        return None
    if isinstance(raw, str):
        parts = [part for part in re.split(r"[\s,]+", raw) if part]
        values = np.array([_number(part) for part in parts], dtype=np.float64)
    elif isinstance(raw, (list, tuple, np.ndarray)):
        values = np.array([_number(value) for value in raw], dtype=np.float64)
    else:
        raise ValueError(
            f"[robotControlOrder] {label} must be a zero-based permutation list"
        )
    if len(values) != length:
        raise ValueError(
            f"[robotControlOrder] {label} must contain exactly {length} values"
        )
    integer = np.isfinite(values) & (values == np.floor(values))
    in_range = (values >= 0) & (values < length)
    if not (integer & in_range).all() or np.unique(values).size != length:
        raise ValueError(
            f"[robotControlOrder] {label} must be a zero-based permutation"
        )
    return values.astype(np.int32)


class ControlMergeProcessor:
    """Concatenates control inputs in slot order, or scatters them through
    the scene-robot joint -> motor permutation."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._input_dims = []
        self._joint2motor_idx = None
        self._label = "ControlMergeProcessor"

    def init(self, config, input_dims):
        self._input_dims = list(input_dims)
        dim = sum(input_dims)
        self.output_buffers[0] = np.zeros(dim, dtype=np.float32)
        node_id = config.get("nodeId")
        node_type = config.get("nodeType")
        self._label = (
            f"{node_id if node_id is not None else 'node'} "
            f"({node_type if node_type is not None else 'control-merge'})"
        )
        self._joint2motor_idx = (
            parse_joint2_motor_idx(
                config.get("joint2motorIdx"), dim, f"{self._label}.joint2motorIdx"
            )
            if node_type == "scene-robot" and dim > 0
            else None
        )

    def compute(self, inputs):
        if self._joint2motor_idx is not None:
            self._compute_joint_ordered(inputs)
            return
        output = self.output_buffers[0]
        offset = 0
        for slot in range(len(self._input_dims)):
            expected_dim = self._input_dims[slot]
            source = inputs[slot] if slot < len(inputs) else None
            if source is None:
                if expected_dim == 0:
                    continue
                raise ValueError(
                    f"[ControlMergeProcessor] Missing input at slot {slot}"
                )
            if len(source) != expected_dim:
                raise ValueError(
                    f"[ControlMergeProcessor] Slot {slot} produced "
                    f"{len(source)} values, expected {expected_dim}"
                )
            output[offset : offset + len(source)] = source
            offset += len(source)

    def _gather_inputs(self, inputs):
        parts = []
        for slot in range(len(self._input_dims)):
            expected_dim = self._input_dims[slot]
            source = inputs[slot] if slot < len(inputs) else None
            if source is None:
                if expected_dim == 0:
                    continue
                raise ValueError(
                    f"[ControlMergeProcessor] Missing input at slot {slot}"
                )
            if len(source) != expected_dim:
                raise ValueError(
                    f"[ControlMergeProcessor] Slot {slot} produced "
                    f"{len(source)} values, expected {expected_dim}"
                )
            parts.append(np.asarray(source, dtype=np.float64))
        if parts:
            return np.concatenate(parts)
        return np.zeros(0, dtype=np.float64)

    def _compute_joint_ordered(self, inputs):
        values = self._gather_inputs(inputs)
        finite = np.isfinite(values)
        if not finite.all():
            raise ValueError(
                f"[ControlMergeProcessor] {self._label} "
                f"input[{_first_bad_index(~finite)}] is not finite"
            )
        self.output_buffers[0][self._joint2motor_idx] = values


# ---------------------------------------------------------------------------
# ScopeProcessor (out-scope, out-logger)
# ---------------------------------------------------------------------------


class ScopeProcessor:
    """Tap into the data flow for visualization; pushes to the trace store."""

    def __init__(self):
        self.output_buffers = []
        self._node_id = ""
        self._labels = []

    def init(self, config, input_dims):
        node_id = config.get("nodeId")
        if not node_id:
            raise ValueError("[ScopeProcessor] nodeId is required")
        self._node_id = node_id
        labels = []
        for slot, dim in enumerate(input_dims):
            if dim <= 0:
                continue
            if dim == 1:
                labels.append(f"IN{slot + 1}")
            else:
                labels.extend(f"IN{slot + 1}[{index}]" for index in range(dim))
        self._labels = labels

    def compute(self, inputs):
        connected = [
            source for source in inputs if isinstance(source, np.ndarray)
        ]
        if not connected:
            return
        append_scope_inputs(self._node_id, connected, self._labels)


REGISTRY = {
    "out-action": ActionProcessor,
    "proc-action-hook": ActionHookProcessor,
    "out-mujoco": ControlMergeProcessor,
    "scene-robot": ControlMergeProcessor,
    "scene-terrain": ControlMergeProcessor,
    "out-scope": ScopeProcessor,
    "out-logger": ScopeProcessor,
    "proc-pd": PDProcessor,
    "proc-motion-reference": MotionReferenceProcessor,
    "proc-motion-sample": MotionSampleProcessor,
}
