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

import math
import re
import time
from collections import deque
from collections.abc import Mapping

import numpy as np

MOTION_REFERENCE_DIM = 67
MOTION_REFERENCE_DOF_COUNT = 29
DEFAULT_MOTION_REFERENCE_ANCHOR_LINK_INDEX = 0
DEFAULT_MOTION_REFERENCE_FRAME_INTERVAL = 5

MAX_SCOPE_SAMPLES = 1200

MOTION_CHANNELS = (
    "joint_position",
    "joint_velocity",
    "body_position",
    "body_quaternion",
    "body_linear_velocity",
    "body_angular_velocity",
)


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


# ---------------------------------------------------------------------------
# PDProcessor (proc-pd) and its controller helpers
# ---------------------------------------------------------------------------


def _fixed_number_list(raw, name, length, context):
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(f"[{context}] {name} must be an array with {length} numbers")
    values = [_number(value) for value in raw]
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError(
            f"[{context}] {name} must contain exactly {length} finite numbers"
        )
    return np.array(values, dtype=np.float32)


def _scalar_or_fixed_number_list(raw, name, length, context):
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(
        raw, bool
    ):
        if not math.isfinite(float(raw)):
            raise ValueError(f"[{context}] {name} must be finite")
        return np.full(length, float(raw), dtype=np.float32)
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(
            f"[{context}] {name} must be a finite number or "
            f"an array with {length} numbers"
        )
    values = [_number(value) for value in raw]
    if len(values) == 1 and length != 1:
        if not math.isfinite(values[0]):
            raise ValueError(f"[{context}] {name}[0] must be finite")
        return np.full(length, values[0], dtype=np.float32)
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError(
            f"[{context}] {name} must contain exactly {length} finite numbers"
        )
    return np.array(values, dtype=np.float32)


def _first_negative(values):
    negative = values < 0
    return _first_bad_index(negative) if negative.any() else None


def _fixed_non_negative_number_list(raw, name, length, context):
    values = _fixed_number_list(raw, name, length, context)
    index = _first_negative(values)
    if index is not None:
        raise ValueError(f"[{context}] {name}[{index}] must be >= 0")
    return values


def _fixed_non_negative_limit_list(raw, name, length, context):
    values = _scalar_or_fixed_number_list(raw, name, length, context)
    index = _first_negative(values)
    if index is not None:
        raise ValueError(f"[{context}] {name}[{index}] must be >= 0")
    return values


def _symmetric_clip_bounds(raw, name, length, context):
    limit = _fixed_non_negative_limit_list(raw, name, length, context)
    return -limit, limit


def _explicit_clip_bounds(raw_min, raw_max, min_name, max_name, length, context):
    if raw_min is None or raw_max is None:
        raise ValueError(
            f"[{context}] {min_name} and {max_name} must be set together"
        )
    mins = _scalar_or_fixed_number_list(raw_min, min_name, length, context)
    maxs = _scalar_or_fixed_number_list(raw_max, max_name, length, context)
    exceeded = mins > maxs
    if exceeded.any():
        index = _first_bad_index(exceeded)
        raise ValueError(
            f"[{context}] {min_name}[{index}] must be <= {max_name}[{index}]"
        )
    return mins, maxs


class _PDController:
    """Position PD gains and torque computation (engine/controllers/PDController.ts)."""

    def __init__(self):
        self._context = "PDController"
        self._dof_count = 0
        self._kps = np.zeros(0, dtype=np.float32)
        self._kds = np.zeros(0, dtype=np.float32)
        self._effort_min = np.zeros(0, dtype=np.float32)
        self._effort_max = np.zeros(0, dtype=np.float32)
        self._kps64 = np.zeros(0, dtype=np.float64)
        self._kds64 = np.zeros(0, dtype=np.float64)
        self._effort_min64 = np.zeros(0, dtype=np.float64)
        self._effort_max64 = np.zeros(0, dtype=np.float64)

    @property
    def kps(self):
        return self._kps

    @property
    def kds(self):
        return self._kds

    @property
    def effort_min(self):
        return self._effort_min

    @property
    def effort_max(self):
        return self._effort_max

    def init(self, config, dof_count, context):
        if not isinstance(dof_count, int) or dof_count <= 0:
            raise ValueError(f"[{context}] dofCount must be a positive integer")
        self._context = context
        self._dof_count = dof_count
        self._kps = _fixed_non_negative_number_list(config.get("kps"), "kps", dof_count, context)
        self._kds = _fixed_non_negative_number_list(config.get("kds"), "kds", dof_count, context)

        effort_min_raw = config.get("effortLimitMin")
        if effort_min_raw is None:
            effort_min_raw = config.get("effort_limit_min")
        effort_max_raw = config.get("effortLimitMax")
        if effort_max_raw is None:
            effort_max_raw = config.get("effort_limit_max")
        if effort_min_raw is not None or effort_max_raw is not None:
            self._effort_min, self._effort_max = _explicit_clip_bounds(
                effort_min_raw,
                effort_max_raw,
                "effortLimitMin",
                "effortLimitMax",
                dof_count,
                context,
            )
        else:
            effort_limit_raw = config.get("effortLimit")
            if effort_limit_raw is None:
                effort_limit_raw = config.get("effort_limit")
            self._effort_min, self._effort_max = _symmetric_clip_bounds(
                effort_limit_raw, "effortLimit", dof_count, context
            )
        self._kps64 = self._kps.astype(np.float64)
        self._kds64 = self._kds.astype(np.float64)
        self._effort_min64 = self._effort_min.astype(np.float64)
        self._effort_max64 = self._effort_max.astype(np.float64)

    def compute(self, target, qpos, qvel, output):
        if (
            len(target) != self._dof_count
            or len(qpos) != self._dof_count
            or len(qvel) != self._dof_count
            or len(output) != self._dof_count
        ):
            raise ValueError(
                f"[{self._context}] target, qpos, qvel, and output must all "
                f"have {self._dof_count} values"
            )
        tau = (
            (np.asarray(target, dtype=np.float64) - np.asarray(qpos, dtype=np.float64))
            * self._kps64
            - np.asarray(qvel, dtype=np.float64) * self._kds64
        )
        finite = np.isfinite(tau)
        if not finite.all():
            raise ValueError(
                f"[{self._context}] computed tau[{_first_bad_index(~finite)}] "
                "is not finite"
            )
        np.clip(tau, self._effort_min64, self._effort_max64, out=tau)
        output[:] = tau


def _resolve_pd_type(config):
    raw = config.get("pdType")
    if raw == "position":
        return "position"
    if raw == "velocity":
        return "velocity"
    if raw == "effort":
        return "effort"
    if raw is None or raw == "":
        return "position"
    raise ValueError(f'[PDProcessor] unsupported pdType "{str(raw)}"')


def _optional_non_negative_integer(raw, name):
    if _is_blank(raw):
        return 0
    value = _number(raw)
    if not _is_integer(value) or value < 0:
        raise ValueError(f"[PDProcessor] {name} must be a non-negative integer")
    return int(value)


def _optional_index_list(raw, name, length):
    if _is_blank(raw):
        return None
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(
            f"[PDProcessor] {name} must be an array with "
            f"{length} non-negative integers"
        )
    values = np.array([_number(value) for value in raw], dtype=np.float64)
    if len(values) != length:
        raise ValueError(f"[PDProcessor] {name} must contain exactly {length} values")
    bad = ~(np.isfinite(values) & (values == np.floor(values)) & (values >= 0))
    if bad.any():
        raise ValueError(
            f"[PDProcessor] {name}[{_first_bad_index(bad)}] "
            "must be a non-negative integer"
        )
    return values.astype(np.int32)


class PDProcessor:
    """Position / velocity / effort PD output with an optional substep view."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._pd_type = "position"
        self._dof_count = 0
        self._controller = _PDController()
        self._target_dof_pos = np.zeros(0, dtype=np.float32)
        self._target_blend_start = np.zeros(0, dtype=np.float32)
        self._target_blend_steps = 0
        self._target_blend_step = 0
        self._qpos_indices = None
        self._qvel_indices = None
        self._ctrl_indices = None
        self._vel_kps64 = np.zeros(0, dtype=np.float64)
        self._effort_min64 = np.zeros(0, dtype=np.float64)
        self._effort_max64 = np.zeros(0, dtype=np.float64)

    def init(self, config, input_dims):
        self._pd_type = _resolve_pd_type(config)
        if self._pd_type == "velocity":
            self._init_velocity(config, input_dims)
            return
        if self._pd_type == "effort":
            self._init_effort(config, input_dims)
            return
        self._init_position(config, input_dims)

    def _init_position(self, config, input_dims):
        target_dim = input_dims[0] if len(input_dims) > 0 else 0
        qpos_dim = input_dims[1] if len(input_dims) > 1 else 0
        qvel_dim = input_dims[2] if len(input_dims) > 2 else 0
        if not target_dim or not qpos_dim or not qvel_dim:
            raise ValueError(
                "[PDProcessor] position type requires target, qpos, and qvel inputs"
            )
        if target_dim != qpos_dim or qpos_dim != qvel_dim:
            raise ValueError(
                "[PDProcessor] input dimensions must match; got "
                f"target={target_dim}, qpos={qpos_dim}, qvel={qvel_dim}"
            )
        self._dof_count = target_dim
        self._controller.init(config, target_dim, "PDProcessor")
        self._target_dof_pos = np.zeros(target_dim, dtype=np.float32)
        self._target_blend_start = np.zeros(target_dim, dtype=np.float32)
        self._target_blend_steps = _optional_non_negative_integer(
            config.get("targetBlendSteps"), "targetBlendSteps"
        )
        self._target_blend_step = 0
        self._qpos_indices = _optional_index_list(
            _first_configured(config.get("qposIndices"), config.get("qpos_indices")),
            "qposIndices",
            target_dim,
        )
        self._qvel_indices = _optional_index_list(
            _first_configured(
                config.get("qvelIndices"), config.get("qvel_indices"), self._qpos_indices
            ),
            "qvelIndices",
            target_dim,
        )
        self._ctrl_indices = _optional_index_list(
            _first_configured(config.get("ctrlIndices"), config.get("ctrl_indices")),
            "ctrlIndices",
            target_dim,
        )
        self.output_buffers = [np.zeros(target_dim, dtype=np.float32)]

    def _init_velocity(self, config, input_dims):
        vel_dim = input_dims[0] if len(input_dims) > 0 else 0
        qvel_dim = input_dims[1] if len(input_dims) > 1 else 0
        if not vel_dim or not qvel_dim:
            raise ValueError(
                "[PDProcessor] velocity type requires vel_target and qvel inputs"
            )
        if vel_dim != qvel_dim:
            raise ValueError(
                "[PDProcessor] vel_target and qvel dimensions must match; "
                f"got {vel_dim} vs {qvel_dim}"
            )
        self._dof_count = vel_dim
        vel_kps = _fixed_non_negative_number_list(
            config.get("kps"), "kps", vel_dim, "PDProcessor"
        )
        self._vel_kps64 = vel_kps.astype(np.float64)
        effort_limit_raw = config.get("effortLimit")
        if effort_limit_raw is None:
            effort_limit_raw = config.get("effort_limit")
        effort_min, effort_max = _symmetric_clip_bounds(
            effort_limit_raw, "effortLimit", vel_dim, "PDProcessor"
        )
        self._effort_min64 = effort_min.astype(np.float64)
        self._effort_max64 = effort_max.astype(np.float64)
        self.output_buffers = [np.zeros(vel_dim, dtype=np.float32)]

    def _init_effort(self, config, input_dims):
        tau_dim = input_dims[0] if len(input_dims) > 0 else 0
        if not tau_dim:
            raise ValueError("[PDProcessor] effort type requires target_tau input")
        self._dof_count = tau_dim
        effort_limit_raw = config.get("effortLimit")
        if effort_limit_raw is None:
            effort_limit_raw = config.get("effort_limit")
        effort_min, effort_max = _symmetric_clip_bounds(
            effort_limit_raw, "effortLimit", tau_dim, "PDProcessor"
        )
        self._effort_min64 = effort_min.astype(np.float64)
        self._effort_max64 = effort_max.astype(np.float64)
        self.output_buffers = [np.zeros(tau_dim, dtype=np.float32)]

    def compute(self, inputs):
        if self._pd_type == "velocity":
            self._compute_velocity(inputs)
            return
        if self._pd_type == "effort":
            self._compute_effort(inputs)
            return
        self._compute_position(inputs)

    def reset(self):
        self.output_buffers[0].fill(0)
        self._target_dof_pos.fill(0)
        self._target_blend_start.fill(0)
        self._target_blend_step = 0

    def _compute_position(self, inputs):
        target = inputs[0] if len(inputs) > 0 else None
        qpos = inputs[1] if len(inputs) > 1 else None
        qvel = inputs[2] if len(inputs) > 2 else None
        if target is None or qpos is None or qvel is None:
            raise ValueError(
                "[PDProcessor] Missing required inputs for position PD"
            )
        effective_target = self._position_target(target, qpos)
        self._controller.compute(
            effective_target, qpos, qvel, self.output_buffers[0]
        )

    def _position_target(self, target, qpos):
        if self._target_blend_steps <= 0:
            self._target_dof_pos[:] = target
            return self._target_dof_pos

        if self._target_blend_step == 0:
            self._target_blend_start[:] = qpos

        alpha = (
            1.0
            if self._target_blend_steps <= 1
            else min(1.0, self._target_blend_step / (self._target_blend_steps - 1))
        )
        if alpha >= 1:
            self._target_dof_pos[:] = target
        else:
            start = self._target_blend_start.astype(np.float64)
            goal = np.asarray(target, dtype=np.float64)
            self._target_dof_pos[:] = start + alpha * (goal - start)
        self._target_blend_step += 1
        return self._target_dof_pos

    def _compute_velocity(self, inputs):
        vel_target = inputs[0] if len(inputs) > 0 else None
        qvel = inputs[1] if len(inputs) > 1 else None
        if vel_target is None or qvel is None:
            raise ValueError(
                "[PDProcessor] Missing vel_target or qvel input for velocity PD"
            )
        tau = self._vel_kps64 * (
            np.asarray(vel_target, dtype=np.float64)
            - np.asarray(qvel, dtype=np.float64)
        )
        finite = np.isfinite(tau)
        if not finite.all():
            raise ValueError(
                f"[PDProcessor] computed tau[{_first_bad_index(~finite)}] is not finite"
            )
        np.clip(tau, self._effort_min64, self._effort_max64, out=tau)
        self.output_buffers[0][:] = tau

    def _compute_effort(self, inputs):
        target_tau = inputs[0] if len(inputs) > 0 else None
        if target_tau is None:
            raise ValueError(
                "[PDProcessor] Missing target_tau input for effort control"
            )
        values = np.asarray(target_tau, dtype=np.float64)
        self.output_buffers[0][:] = np.clip(
            values, self._effort_min64, self._effort_max64
        )

    def get_substep_pd(self):
        """Live substep PD view for the physics layer (TS ``getSubstepPd``).

        Returns ``None`` unless this is a position PD; the arrays are views
        of the live state, so the caller re-reads them per physics substep.
        Optional index keys are present only when configured.
        """
        if self._pd_type != "position":
            return None
        view = {
            "target": self._target_dof_pos,
            "kps": self._controller.kps,
            "kds": self._controller.kds,
            "effort_min": self._controller.effort_min,
            "effort_max": self._controller.effort_max,
        }
        if self._qpos_indices is not None:
            view["qpos_indices"] = self._qpos_indices
        if self._qvel_indices is not None:
            view["qvel_indices"] = self._qvel_indices
        if self._ctrl_indices is not None:
            view["ctrl_indices"] = self._ctrl_indices
        return view


# ---------------------------------------------------------------------------
# Motion data injection (host-loaded buffers)
# ---------------------------------------------------------------------------


class _MotionData:
    """Validated motion buffers in the flat per-frame layout of the engine."""

    __slots__ = (
        "frame_count",
        "body_count",
        "dof_count",
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "dof_names",
        "body_names",
    )


def _resolve_motion_source(config, label):
    raw = config.get("motionSource")
    if isinstance(raw, Mapping):
        kind = raw.get("kind")
        path = raw.get("path")
        if kind == "url" and isinstance(path, str) and path.strip():
            return {"kind": "url", "path": path.strip()}
        prefix = raw.get("registryPrefix")
        if kind == "registry" and isinstance(prefix, str) and prefix.strip():
            cache_key = raw.get("cacheKey")
            return {
                "kind": "registry",
                "registryPrefix": prefix.strip(),
                "cacheKey": (
                    cache_key.strip()
                    if isinstance(cache_key, str) and cache_key.strip()
                    else None
                ),
            }
        buffers = raw.get("buffers")
        cache_key = raw.get("cacheKey")
        if (
            kind == "inline"
            and isinstance(buffers, Mapping)
            and isinstance(cache_key, str)
            and cache_key.strip()
        ):
            return {
                "kind": "inline",
                "cacheKey": cache_key.strip(),
                "buffers": buffers,
            }
        raise ValueError(f"[{label}] motionSource is malformed")
    path = config.get("motionPath")
    if isinstance(path, str) and path.strip():
        return {"kind": "url", "path": path.strip()}
    raise ValueError(f"[{label}] motionPath or motionSource is required")


def _motion_count(raw, what, label):
    value = _number(raw)
    if not _is_integer(value) or value <= 0:
        raise ValueError(f"[{label}] motionData {what} must be a positive integer")
    return int(value)


def _motion_channel_buffer(raw, name, expected_length, label):
    if raw is None:
        raise ValueError(f"[{label}] motionData is missing the {name} buffer")
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            f"[{label}] motionData {name} length {values.size} does not match "
            f"the motion shape {expected_length}"
        )
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"[{label}] motionData {name} contains a non-finite value "
            f"at {_first_bad_index(~finite)}"
        )
    return values


def _motion_names(raw, expected, name, label):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != expected:
        raise ValueError(f"[{label}] motionData {name} must contain {expected} names")
    names = [str(value) for value in raw]
    if len(set(names)) != len(names):
        raise ValueError(f"[{label}] motionData {name} must not contain duplicate names")
    return names


def _motion_data_from_config(config, label):
    """Read and validate the host-injected ``motionData`` mapping."""
    raw = config.get("motionData")
    if raw is None:
        raise ValueError(
            f"[{label}] motionData is required for the local runtime; "
            "the host must load the motion and inject its buffers before compile"
        )
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{label}] motionData must be a mapping of motion buffers")

    data = _MotionData()
    data.frame_count = _motion_count(raw.get("frame_count"), "frame_count", label)
    data.body_count = _motion_count(raw.get("body_count"), "body_count", label)
    data.dof_count = _motion_count(raw.get("dof_count"), "dof_count", label)
    fps = _number(raw.get("fps"))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"[{label}] motionData fps must be positive, got {fps}")
    data.fps = fps

    joints = data.frame_count * data.dof_count
    bodies3 = data.frame_count * data.body_count * 3
    bodies4 = data.frame_count * data.body_count * 4
    data.joint_pos = _motion_channel_buffer(
        raw.get("joint_position"), "joint_position", joints, label
    )
    data.joint_vel = _motion_channel_buffer(
        raw.get("joint_velocity"), "joint_velocity", joints, label
    )
    data.body_pos_w = _motion_channel_buffer(
        raw.get("body_position"), "body_position", bodies3, label
    )
    data.body_quat_w = _motion_channel_buffer(
        raw.get("body_quaternion"), "body_quaternion", bodies4, label
    )
    data.body_lin_vel_w = _motion_channel_buffer(
        raw.get("body_linear_velocity"), "body_linear_velocity", bodies3, label
    )
    data.body_ang_vel_w = _motion_channel_buffer(
        raw.get("body_angular_velocity"), "body_angular_velocity", bodies3, label
    )
    data.dof_names = _motion_names(
        raw.get("joint_names"), data.dof_count, "joint_names", label
    )
    data.body_names = _motion_names(
        raw.get("body_names"), data.body_count, "body_names", label
    )
    return data


# ---------------------------------------------------------------------------
# MotionReferenceProcessor (proc-motion-reference): legacy 29 DOF / 67-dim
# frame-increment contract (controls/motionReference.ts writeMotionReference)
# ---------------------------------------------------------------------------


def _clamp_frame(index, start, end):
    return min(max(index, start), end)


def _quat_inverse(q):
    norm = max(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1e-9)
    return (q[0] / norm, -q[1] / norm, -q[2] / norm, -q[3] / norm)


def _quat_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_apply(q, v):
    w, qx, qy, qz = q
    vx, vy, vz = v
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (
        vx + w * tx + qy * tz - qz * ty,
        vy + w * ty + qz * tx - qx * tz,
        vz + w * tz + qx * ty - qy * tx,
    )


def _matrix_from_quat(q):
    w, x, y, z = q
    two_s = 2.0 / max(w * w + x * x + y * y + z * z, 1e-9)
    return (
        1 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1 - two_s * (x * x + y * y),
    )


def write_motion_reference(
    data,
    frame_index,
    output,
    anchor_link_index=None,
    frame_interval=None,
    end_index=None,
    paused=False,
):
    """Write one legacy 67-dim reference frame increment (wxyz quaternions)."""
    if data.dof_count != MOTION_REFERENCE_DOF_COUNT:
        raise ValueError(
            "[motionReference] legacy motion reference requires "
            f"{MOTION_REFERENCE_DOF_COUNT} DOFs, got {data.dof_count}"
        )
    if len(output) != MOTION_REFERENCE_DIM:
        raise ValueError(
            "[motionReference] reference output length "
            f"{len(output)} must be {MOTION_REFERENCE_DIM}"
        )
    anchor = (
        DEFAULT_MOTION_REFERENCE_ANCHOR_LINK_INDEX
        if anchor_link_index is None
        else anchor_link_index
    )
    interval = (
        DEFAULT_MOTION_REFERENCE_FRAME_INTERVAL
        if frame_interval is None
        else frame_interval
    )
    if not _is_integer(anchor) or anchor < 0 or anchor >= data.body_count:
        raise ValueError(
            f"[motionReference] anchorLinkIndex {anchor} is outside body "
            f"count {data.body_count}"
        )
    if not _is_integer(interval) or interval <= 0:
        raise ValueError("[motionReference] frameInterval must be a positive integer")

    end = min(
        end_index if end_index is not None else data.frame_count - 1,
        data.frame_count - 1,
    )
    idx = _clamp_frame(frame_index, 1, end)
    last = idx if paused else _clamp_frame(idx - int(interval), 1, end)
    dof = data.dof_count

    output[0:dof] = data.joint_pos[idx * dof : (idx + 1) * dof]
    if paused:
        output[dof : dof * 2] = 0.0
    else:
        output[dof : dof * 2] = data.joint_vel[idx * dof : (idx + 1) * dof]

    current_pos = data.body_pos_w[
        (idx * data.body_count + anchor) * 3 : (idx * data.body_count + anchor) * 3 + 3
    ].astype(np.float64)
    last_pos = data.body_pos_w[
        (last * data.body_count + anchor) * 3 : (last * data.body_count + anchor) * 3 + 3
    ].astype(np.float64)
    current_quat = data.body_quat_w[
        (idx * data.body_count + anchor) * 4 : (idx * data.body_count + anchor) * 4 + 4
    ].astype(np.float64)
    last_quat = data.body_quat_w[
        (last * data.body_count + anchor) * 4 : (last * data.body_count + anchor) * 4 + 4
    ].astype(np.float64)

    last_inv = _quat_inverse(tuple(last_quat))
    target_pos = _quat_apply(
        last_inv,
        (
            current_pos[0] - last_pos[0],
            current_pos[1] - last_pos[1],
            current_pos[2] - last_pos[2],
        ),
    )
    target_quat = _quat_multiply(last_inv, tuple(current_quat))
    mat = _matrix_from_quat(target_quat)

    output[58] = target_pos[0]
    output[59] = target_pos[1]
    output[60] = target_pos[2]
    output[61] = mat[0]
    output[62] = mat[1]
    output[63] = mat[3]
    output[64] = mat[4]
    output[65] = mat[6]
    output[66] = mat[7]


def _optional_integer(raw, fallback, name, label):
    if raw is None or raw == "":
        return fallback
    value = _number(raw)
    if not _is_integer(value):
        raise ValueError(f"[{label}] {name} must be an integer")
    return int(value)


def _optional_positive_integer(raw, fallback, name, label):
    value = _optional_integer(raw, fallback, name, label)
    if value <= 0:
        raise ValueError(f"[{label}] {name} must be positive")
    return value


def _command_codes_from_config(raw, required, label):
    if raw is None:
        if required:
            raise ValueError(
                f"[{label}] commandCodes is required when the command "
                "input is connected"
            )
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"[{label}] commandCodes must be an object")
    names = ("pauseConfigured", "pauseCurrent", "resume")
    if len(raw) != len(names) or any(name not in raw for name in names):
        raise ValueError(
            f"[{label}] commandCodes must contain exactly pauseConfigured, "
            "pauseCurrent, and resume"
        )
    values = []
    for name in names:
        value = _number(raw[name])
        if not _is_integer(value):
            raise ValueError(f"[{label}] commandCodes.{name} must be an integer")
        values.append(int(value))
    if len(set(values)) != len(values):
        raise ValueError(f"[{label}] commandCodes values must be distinct")
    return {
        "pauseConfigured": values[0],
        "pauseCurrent": values[1],
        "resume": values[2],
    }


class MotionReferenceProcessor:
    """Legacy 29 DOF / 67-dim motion reference reader (per-frame increments)."""

    def __init__(self):
        self.output_buffers = [np.zeros(MOTION_REFERENCE_DIM, dtype=np.float32)]
        self._motion_source = None
        self._start_index = 0
        self._end_index = None
        self._pause_index = 0
        self._frame_interval = 1
        self._anchor_link_index = 0
        self._loop = False
        self._pause_mode = "none"
        self._command_codes = None
        self._last_command_code = None
        self._active_mode = None
        self._refer_index = 0
        self._motion = None

    def init(self, config, input_dims=None):
        input_dims = list(input_dims or [])
        self._motion = None
        self._motion_source = _resolve_motion_source(config, "MotionReferenceProcessor")
        self._start_index = _optional_integer(
            config.get("startIndex"), 0, "startIndex", "MotionReferenceProcessor"
        )
        raw_end = config.get("endIndex")
        self._end_index = (
            None
            if raw_end is None or raw_end == ""
            else _optional_integer(raw_end, -1, "endIndex", "MotionReferenceProcessor")
        )
        self._pause_index = _optional_integer(
            config.get("pauseIndex"), 0, "pauseIndex", "MotionReferenceProcessor"
        )
        if self._pause_index < 0:
            self._pause_index = 0
        self._frame_interval = _optional_positive_integer(
            config.get("frameInterval"),
            1,
            "frameInterval",
            "MotionReferenceProcessor",
        )
        self._anchor_link_index = _optional_integer(
            config.get("anchorLinkIndex"),
            0,
            "anchorLinkIndex",
            "MotionReferenceProcessor",
        )
        if self._anchor_link_index < 0:
            raise ValueError(
                "[MotionReferenceProcessor] anchorLinkIndex must be non-negative"
            )
        self._loop = config.get("loop") is True
        self._pause_mode = "none"
        has_command_input = bool(input_dims[1]) if len(input_dims) > 1 else False
        self._command_codes = _command_codes_from_config(
            config.get("commandCodes"), has_command_input, "MotionReferenceProcessor"
        )
        self._last_command_code = None
        raw_active = config.get("activeMode")
        self._active_mode = (
            None
            if raw_active is None or raw_active == ""
            else _optional_integer(
                raw_active, 0, "activeMode", "MotionReferenceProcessor"
            )
        )
        self._refer_index = self._start_index
        self._motion = _motion_data_from_config(config, "MotionReferenceProcessor")

    def compute(self, inputs=None):
        if self._motion is None:
            raise ValueError(
                "[MotionReferenceProcessor] motion data is not loaded"
            )
        self._advance_and_write(inputs or [])

    def dispose(self):
        self._motion = None

    def _advance_and_write(self, inputs):
        mode_input = inputs[0] if len(inputs) > 0 else None
        command_input = inputs[1] if len(inputs) > 1 else None
        if mode_input is not None and len(mode_input) != 1:
            raise ValueError(
                "[MotionReferenceProcessor] mode input must have dimension 1, "
                f"got {len(mode_input)}"
            )
        active = (
            mode_input is None
            or self._active_mode is None
            or math.trunc(float(mode_input[0])) == self._active_mode
        )
        self._sample_command_input(command_input)

        end_index = self._normalized_end_index(self._motion)
        paused = False

        if not active:
            self._refer_index = min(max(self._start_index, 1), end_index)
            paused = True
        elif self._pause_mode == "configured":
            self._refer_index = min(max(self._pause_index, 1), end_index)
            paused = True
        elif self._pause_mode == "current":
            paused = True
        else:
            self._refer_index += 1
            if self._refer_index >= end_index:
                self._refer_index = (
                    max(self._start_index, 0) if self._loop else end_index
                )

        write_motion_reference(
            self._motion,
            self._refer_index,
            self.output_buffers[0],
            anchor_link_index=self._anchor_link_index,
            frame_interval=self._frame_interval,
            end_index=end_index,
            paused=paused,
        )

    def _sample_command_input(self, command_input):
        if command_input is None:
            return
        if len(command_input) != 1:
            raise ValueError(
                "[MotionReferenceProcessor] command input must have dimension 1, "
                f"got {len(command_input)}"
            )
        command_code = _number(command_input[0])
        if not _is_integer(command_code):
            raise ValueError(
                "[MotionReferenceProcessor] command input must contain an integer code"
            )
        if command_code == self._last_command_code:
            return
        self._last_command_code = command_code
        command_codes = self._command_codes
        if command_codes is None:
            raise ValueError(
                "[MotionReferenceProcessor] commandCodes is required for command input"
            )
        if command_code == command_codes["pauseConfigured"]:
            self._pause_mode = "configured"
        elif command_code == command_codes["pauseCurrent"]:
            self._pause_mode = "current"
        elif command_code == command_codes["resume"]:
            self._pause_mode = "none"

    def _normalized_end_index(self, motion):
        if self._end_index is None or self._end_index < self._start_index:
            return motion.frame_count - 1
        return min(self._end_index, motion.frame_count - 1)


# ---------------------------------------------------------------------------
# MotionSampleProcessor (proc-motion-sample): Motion.sample() semantics
# ---------------------------------------------------------------------------

_SAMPLE_LABEL = "MotionSampleProcessor"


def _channel_width(channel):
    if channel == "body_quaternion":
        return 4
    if channel.startswith("body_"):
        return 3
    return 1


def _channel_from_config(raw):
    if raw in MOTION_CHANNELS:
        return raw
    raise ValueError(
        f"[{_SAMPLE_LABEL}] channel is required and must be a supported motion channel"
    )


def _parse_indices(raw):
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) == 0:
        raise ValueError(
            f"[{_SAMPLE_LABEL}] indices must be a non-empty array of "
            "integers or names"
        )
    numeric = all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        for value in raw
    )
    if numeric:
        values = [_number(value) for value in raw]
        if any(not _is_integer(value) or value < 0 for value in values):
            raise ValueError(
                f"[{_SAMPLE_LABEL}] numeric indices must be non-negative integers"
            )
        return [int(value) for value in values]
    if all(isinstance(value, str) and value.strip() for value in raw):
        return [value.strip() for value in raw]
    raise ValueError(
        f"[{_SAMPLE_LABEL}] indices must contain only integers or only names"
    )


def _interpolation_from_config(raw):
    if raw is None or raw == "linear":
        return "linear"
    if raw == "nearest":
        return "nearest"
    raise ValueError(f"[{_SAMPLE_LABEL}] interpolation must be linear or nearest")


def _channel_source(motion, channel):
    return {
        "joint_position": motion.joint_pos,
        "joint_velocity": motion.joint_vel,
        "body_position": motion.body_pos_w,
        "body_quaternion": motion.body_quat_w,
        "body_linear_velocity": motion.body_lin_vel_w,
        "body_angular_velocity": motion.body_ang_vel_w,
    }[channel]


def _resolved_indices(motion, channel, requested):
    count = motion.dof_count if channel.startswith("joint_") else motion.body_count
    if len(requested) == 0:
        raise ValueError(f"[{_SAMPLE_LABEL}] indices must not be empty")
    if isinstance(requested[0], int):
        if any(value >= count for value in requested):
            raise ValueError(
                f"[{_SAMPLE_LABEL}] indices are outside the {channel} "
                f"channel count {count}"
            )
        return np.array(requested, dtype=np.int32)
    names = motion.dof_names if channel.startswith("joint_") else motion.body_names
    if names is None:
        raise ValueError(
            f"[{_SAMPLE_LABEL}] named {channel} selection requires explicit "
            "motion name metadata"
        )
    resolved = []
    for name in requested:
        try:
            index = names.index(name)
        except ValueError:
            raise ValueError(
                f'[{_SAMPLE_LABEL}] {channel} name "{name}" is not present '
                "in motion metadata"
            ) from None
        resolved.append(index)
    return np.array(resolved, dtype=np.int32)


def _assert_quaternions(motion):
    norms = np.linalg.norm(
        motion.body_quat_w.reshape(-1, 4).astype(np.float64), axis=1
    )
    bad = ~np.isfinite(norms) | (norms <= 1e-8)
    if bad.any():
        raise ValueError(
            f"[{_SAMPLE_LABEL}] body quaternion {_first_bad_index(bad)} must "
            "have a non-zero finite norm"
        )


def _slerp_quaternions(output, source, first_offsets, second_offsets, blend):
    """Vectorized shortest-path SLERP mirroring the TS ``writeSlerp``."""
    columns = np.arange(4)
    a = source[first_offsets[:, None] + columns].astype(np.float64)
    b = source[second_offsets[:, None] + columns].astype(np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.sum(a * b, axis=1)
    flip = dot < 0
    b[flip] = -b[flip]
    dot = np.minimum(np.abs(dot), 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sqrt(np.maximum(1.0 - dot * dot, 0.0))
    nearly_parallel = dot > 0.9995
    safe_sin = np.where(nearly_parallel, 1.0, sin_theta)
    weight_a = np.where(
        nearly_parallel, 1.0 - blend, np.sin((1.0 - blend) * theta) / safe_sin
    )
    weight_b = np.where(nearly_parallel, blend, np.sin(blend * theta) / safe_sin)
    result = weight_a[:, None] * a + weight_b[:, None] * b
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    output[:] = result.reshape(-1)


class MotionSampleProcessor:
    """Per-frame sampling of absolute motion channels at a time signal."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._motion_source = None
        self._motion = None
        self._channel = "joint_position"
        self._requested_indices = []
        self._indices = np.zeros(0, dtype=np.int32)
        self._interpolation = "linear"
        self._loop = False

    def init(self, config, input_dims=None):
        input_dims = list(input_dims or [])
        self._motion = None
        self._indices = np.zeros(0, dtype=np.int32)
        if len(input_dims) != 1 or input_dims[0] != 1:
            raise ValueError(f"[{_SAMPLE_LABEL}] time input must have dimension 1")
        self._motion_source = _resolve_motion_source(config, _SAMPLE_LABEL)
        self._channel = _channel_from_config(config.get("channel"))
        self._requested_indices = _parse_indices(config.get("indices"))
        self._interpolation = _interpolation_from_config(config.get("interpolation"))
        raw_loop = config.get("loop")
        if raw_loop is not None and not isinstance(raw_loop, bool):
            raise ValueError(f"[{_SAMPLE_LABEL}] loop must be a boolean")
        self._loop = raw_loop is True
        self.output_buffers[0] = np.zeros(
            len(self._requested_indices) * _channel_width(self._channel),
            dtype=np.float32,
        )
        self._motion = _motion_data_from_config(config, _SAMPLE_LABEL)
        self._indices = _resolved_indices(
            self._motion, self._channel, self._requested_indices
        )
        if self._channel == "body_quaternion":
            _assert_quaternions(self._motion)

    def compute(self, inputs):
        source = inputs[0] if inputs else None
        if source is None or len(source) != 1:
            raise ValueError(f"[{_SAMPLE_LABEL}] time input must have dimension 1")
        time_value = float(source[0])
        if not math.isfinite(time_value):
            raise ValueError(f"[{_SAMPLE_LABEL}] time input must be finite")
        motion = self._motion
        if motion is None:
            raise ValueError(f"[{_SAMPLE_LABEL}] motion data is not loaded")

        frame_coordinate = self._frame_coordinate(time_value, motion)
        first_frame = math.floor(frame_coordinate)
        blend = frame_coordinate - first_frame
        if self._loop:
            second_frame = (first_frame + 1) % motion.frame_count
            nearest_frame = math.floor(frame_coordinate + 0.5) % motion.frame_count
        else:
            second_frame = min(first_frame + 1, motion.frame_count - 1)
            nearest_frame = min(
                math.floor(frame_coordinate + 0.5), motion.frame_count - 1
            )

        data = _channel_source(motion, self._channel)
        width = _channel_width(self._channel)
        channel_count = (
            motion.dof_count if self._channel.startswith("joint_") else motion.body_count
        )
        stride = channel_count * width
        base = self._indices.astype(np.int64) * width
        output = self.output_buffers[0]

        if self._interpolation == "nearest":
            offsets = int(nearest_frame) * stride + base
            if self._channel == "body_quaternion":
                _slerp_quaternions(output, data, offsets, offsets, 0.0)
            else:
                columns = np.arange(width)
                output[:] = data[offsets[:, None] + columns].reshape(-1)
        elif self._channel == "body_quaternion":
            first_offsets = int(first_frame) * stride + base
            second_offsets = int(second_frame) * stride + base
            _slerp_quaternions(
                output, data, first_offsets, second_offsets, blend
            )
        else:
            columns = np.arange(width)
            first_offsets = int(first_frame) * stride + base
            second_offsets = int(second_frame) * stride + base
            first_values = data[first_offsets[:, None] + columns].astype(np.float64)
            second_values = data[second_offsets[:, None] + columns].astype(np.float64)
            output[:] = (
                first_values + blend * (second_values - first_values)
            ).reshape(-1)

    def dispose(self):
        self._motion = None
        self._indices = np.zeros(0, dtype=np.int32)

    def _frame_coordinate(self, time_value, motion):
        if self._loop:
            period = motion.frame_count / motion.fps
            wrapped = ((time_value % period) + period) % period
            return wrapped * motion.fps
        return min(max(time_value * motion.fps, 0.0), motion.frame_count - 1)


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
