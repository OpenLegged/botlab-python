"""Motion reference/sample processors and host-injected motion data."""

import math
from collections.abc import Mapping

import numpy as np

from .._quat import quat_multiply
from ._output_common import _first_bad_index, _is_integer, _number

MOTION_REFERENCE_DIM = 67
MOTION_REFERENCE_DOF_COUNT = 29
DEFAULT_MOTION_REFERENCE_ANCHOR_LINK_INDEX = 0
DEFAULT_MOTION_REFERENCE_FRAME_INTERVAL = 5


MOTION_CHANNELS = (
    "joint_position",
    "joint_velocity",
    "body_position",
    "body_quaternion",
    "body_linear_velocity",
    "body_angular_velocity",
)


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
    target_quat = quat_multiply(np.asarray(last_inv), current_quat)
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
