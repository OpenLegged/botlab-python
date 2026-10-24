"""obs-commands vector assembly (mujocoObservationCommands.ts local subset mirror)."""

import math
import re

import numpy as np

from ._kinematics import _base_body_index, _has_root_free_joint
from ._values import (
    _error,
    _first_present,
    _flat,
    _is_integer,
    _number,
    _string_list,
    configured_dim,
    parse_number_list,
    vector_from,
)

DEFAULT_COMMAND_DIM = 5


# ---------------------------------------------------------------------------
# mujocoObservationCommands.ts mirror — local (non-interactive) subset
# ---------------------------------------------------------------------------

_INTERACTIVE_COMMAND_MODES = frozenset(
    {
        "keyboard",
        "gamepad",
        "keyboard-gamepad",
        "keyboard-buttons",
        "gamepad-buttons",
        "keyboard-gamepad-buttons",
    }
)

# mujocoObservationCommands.ts pose-velocity constants (config keys keep
# camelCase).
DEFAULT_POSE_VELOCITY_LOOKAHEAD = 2.0
DEFAULT_POSE_VELOCITY_STIFFNESS = 2.0
DEFAULT_HEADING_STIFFNESS = 2.0
DEFAULT_TARGET_DISTANCE_THRESHOLD = 0.4
FORWARD_ACTIVE_EPSILON = 1e-4
YAW_ACTIVE_EPSILON = 1e-4

# keyboardCommands.ts commandAxisConfig defaults.
_DEFAULT_COMMAND_AXIS_MIN = (-1.0, -1.0, -1.0)
_DEFAULT_COMMAND_AXIS_MAX = (1.0, 1.0, 1.0)
_DEFAULT_COMMAND_AXIS_INDICES = (0, 1, 2)


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def _wrap_pi(value):
    wrapped = value
    while wrapped > math.pi:
        wrapped -= 2 * math.pi
    while wrapped < -math.pi:
        wrapped += 2 * math.pi
    return wrapped


def _finite_number_field(raw, field, fallback):
    """``finiteNumber``: missing/empty keeps the fallback, non-finite errors."""
    if raw is None or raw == "":
        return fallback
    value = _number(raw)
    if math.isfinite(value):
        return value
    raise _error("mujocoObservations", f"{field} must be finite")


def _positive_finite_number(raw, field, fallback):
    value = _finite_number_field(raw, field, fallback)
    if value > 0:
        return value
    raise _error("mujocoObservations", f"{field} must be > 0")


def _non_negative_finite_number(raw, field, fallback):
    value = _finite_number_field(raw, field, fallback)
    if value >= 0:
        return value
    raise _error("mujocoObservations", f"{field} must be >= 0")


# ---------------------------------------------------------------------------
# keyboardCommands.ts commandAxisConfig mirror (axis indices/bounds)
# ---------------------------------------------------------------------------


def _axis_number_list(raw, field):
    """``finiteNumberList`` input decoding: string / list / numeric array."""
    if isinstance(raw, str):
        return [_number(part) for part in raw.replace(",", " ").split() if part]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [_number(item) for item in raw]
    raise _error("keyboardCommands", f"{field} must be a number list")


def _command_axis_indices(config, dim):
    raw = config.get("commandIndices")
    if raw is None or raw == "":
        values = list(_DEFAULT_COMMAND_AXIS_INDICES)
    else:
        values = _axis_number_list(raw, "commandIndices")
        if len(values) != 3:
            raise _error(
                "keyboardCommands", "commandIndices must contain exactly 3 numbers"
            )
    if not all(math.isfinite(value) for value in values):
        raise _error("keyboardCommands", "commandIndices contains a non-finite number")
    if any(not _is_integer(value) for value in values):
        raise _error("keyboardCommands", "commandIndices must contain integers")

    seen = set()
    for index in values:
        if index < 0 or index >= dim:
            raise _error(
                "keyboardCommands",
                f"command index {index:g} is outside dim {dim}",
            )
        if index in seen:
            raise _error("keyboardCommands", f"duplicate command index {index:g}")
        seen.add(index)
    # Validated integers, cast for use as array/list indices.
    return [int(value) for value in values]


def _command_axis_bounds(raw, field, dim, indices, fallback):
    if raw is None or raw == "":
        values = list(fallback)
    else:
        values = _axis_number_list(raw, field)
    if len(values) != 3 and len(values) != dim:
        raise _error("keyboardCommands", f"{field} must contain 3 or {dim} numbers")
    if not all(math.isfinite(value) for value in values):
        raise _error("keyboardCommands", f"{field} contains a non-finite number")
    return values if len(values) == 3 else [values[index] for index in indices]


def command_axis_config(config, dim):
    """``commandAxisConfig``: the 3 command axes' indices and bounds."""
    config = config or {}
    indices = _command_axis_indices(config, dim)
    minimum = _command_axis_bounds(
        config.get("cmdMin"), "cmdMin", dim, indices, _DEFAULT_COMMAND_AXIS_MIN
    )
    maximum = _command_axis_bounds(
        config.get("cmdMax"), "cmdMax", dim, indices, _DEFAULT_COMMAND_AXIS_MAX
    )
    for i in range(3):
        if minimum[i] > maximum[i]:
            raise _error("keyboardCommands", f"cmdMin[{i}] must be <= cmdMax[{i}]")
    return {"indices": indices, "min": minimum, "max": maximum}


def _parse_root_x_range(raw, label):
    values = parse_number_list(raw, label)
    if not values or len(values) != 2:
        raise _error("mujocoObservations", f"{label} must contain exactly 2 numbers")
    if values[0] > values[1]:
        raise _error("mujocoObservations", f"{label}[0] must be <= {label}[1]")
    return (values[0], values[1])


def _parse_root_x_ranges(raw, label):
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple, np.ndarray)) and (
        not raw or not isinstance(raw[0], (list, tuple, np.ndarray))
    ):
        flat = parse_number_list(raw, label) or []
        if len(flat) == 2:
            return [(flat[0], flat[1])]
        if len(flat) % 2 != 0:
            raise _error(
                "mujocoObservations", f"{label} flat list must contain range pairs"
            )
        return [
            _parse_root_x_range([flat[i * 2], flat[i * 2 + 1]], f"{label}[{i}]")
            for i in range(len(flat) // 2)
        ]
    if isinstance(raw, (str, bool, int, float, np.integer, np.floating)):
        flat = parse_number_list(raw, label) or []
        if len(flat) == 2:
            return [(flat[0], flat[1])]
        if len(flat) % 2 != 0:
            raise _error(
                "mujocoObservations", f"{label} flat list must contain range pairs"
            )
        return [
            _parse_root_x_range([flat[i * 2], flat[i * 2 + 1]], f"{label}[{i}]")
            for i in range(len(flat) // 2)
        ]
    if not isinstance(raw, (list, tuple)):
        raise _error("mujocoObservations", f"{label} must be a range or range list")
    return [_parse_root_x_range(item, f"{label}[{i}]") for i, item in enumerate(raw)]


def _yaw_from_quaternion(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _root_pose_2d(context):
    """``rootPose2d``: the root's world (x, y, yaw) pose."""
    model, data = context["model"], context["data"]
    if _has_root_free_joint(model, data):
        qpos = _flat(data.qpos, "data.qpos")
        return (
            float(qpos[0]),
            float(qpos[1]),
            _yaw_from_quaternion(qpos[3], qpos[4], qpos[5], qpos[6]),
        )
    index = _base_body_index(model)
    xpos = _flat(data.xpos, "data.xpos")
    xquat = _flat(data.xquat, "data.xquat")
    pos_start = index * 3
    quat_start = index * 4
    return (
        float(xpos[pos_start]),
        float(xpos[pos_start + 1]),
        _yaw_from_quaternion(
            xquat[quat_start],
            xquat[quat_start + 1],
            xquat[quat_start + 2],
            xquat[quat_start + 3],
        ),
    )


def _root_pose_x(context):
    """x coordinate of rootPose2d (mujocoObservationCommands.ts)."""
    return _root_pose_2d(context)[0]


def _write_command_mapping(output, mapping, label):
    values = mapping.get("values")
    if values is not None:
        output[:] = vector_from(values, len(output), f"{label}.values")
        return
    index = _number(mapping.get("index"))
    if not _is_integer(index) or index < 0 or index >= len(output):
        raise _error(
            "mujocoObservations",
            f"{label}.index must be an integer in [0, {len(output) - 1}]",
        )
    raw_value = mapping.get("value")
    value = 1.0 if raw_value is None or raw_value == "" else _number(raw_value)
    if not math.isfinite(value):
        raise _error("mujocoObservations", f"{label}.value must be finite")
    output.fill(0.0)
    output[int(index)] = value


def _command_key_list(raw, label):
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [part for part in re.split(r"[\s,]+", raw) if part]
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    raise _error(
        "mujocoObservations", f"{label} must be a string or string array"
    )


def _local_command_mapping_active(mapping, label):
    all_keys = _command_key_list(
        _first_present(mapping, ("keys", "allKeys")), f"{label}.keys"
    )
    any_keys = _command_key_list(
        _first_present(mapping, ("key", "anyKey", "anyKeys")),
        f"{label}.anyKeys",
    )
    pressed = _LOCAL_COMMAND_INPUT_STATE["pressed_keys"]
    if all_keys and not all(key in pressed for key in all_keys):
        return False
    if any_keys and not any(key in pressed for key in any_keys):
        return False
    return True


def _mapped_command_vector(config, output, context):
    raw_mappings = _first_present(config, ("commandMappings", "commandMap", "mappings"))
    if raw_mappings is None or raw_mappings == "":
        return output
    if not isinstance(raw_mappings, (list, tuple)):
        raise _error("mujocoObservations", "commandMappings must be an array")

    for i, mapping in enumerate(raw_mappings):
        label = f"commandMappings[{i}]"
        if not isinstance(mapping, dict):
            raise _error("mujocoObservations", f"{label} must be an object")
        if not _local_command_mapping_active(mapping, label):
            continue
        ranges = [
            *_parse_root_x_ranges(
                _first_present(mapping, ("rootXRange", "rootRange")), f"{label}.rootXRange"
            ),
            *_parse_root_x_ranges(
                _first_present(mapping, ("rootXRanges", "rootRanges")), f"{label}.rootXRanges"
            ),
        ]
        if ranges:
            x = _root_pose_x(context)
            if not any(minimum <= x <= maximum for minimum, maximum in ranges):
                continue
        _write_command_mapping(output, mapping, label)
        return output
    return output


def _object_command_vector(raw, dim):
    def part(*keys):
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key]
        return 0
    return vector_from(
        [
            part("grasp"),
            part("linX", "vx", "x"),
            part("linY", "vy", "y"),
            part("yaw", "wz"),
            part("pitch"),
        ],
        dim,
        "obs-commands",
    )


_pose_velocity_state_by_node = {}

# Native headless runs have an input device state just like the browser host,
# but no keyboard/gamepad is attached to it. Keeping this as explicit state is
# important: an inactive key gate means "try the next mapping / keep the
# declared default", not "ignore a feature the runtime failed to implement".
_LOCAL_COMMAND_INPUT_STATE = {
    "pressed_keys": frozenset(),
    "pressed_buttons": frozenset(),
    "axis_active": False,
}


def clear_pose_velocity_states(node_id=None):
    """``resetPoseVelocityCommandState``: drop one node's state or all of it.

    The controller reset/dispose loop calls this per obs-commands node so a
    replayed run re-anchors centerY/targetX from the reset pose instead of
    the pre-reset trajectory.
    """
    if node_id is None:
        _pose_velocity_state_by_node.clear()
        return
    _pose_velocity_state_by_node.pop(node_id, None)


def _pose_velocity_command_vector(config, node_id, output, context):
    """``poseVelocityCommandVector`` (mujocoObservationCommands.ts).

    Waypoint-tracking rewrite of the command vector against the root pose:
    while a forward command is active the robot steers toward a lookahead
    target on the line it was walking (centerY anchors the lateral target,
    targetX only ever advances). The browser version receives the vector
    after the interactive overrides; the local baseline is the
    parameter-injected vector, which is exactly the browser's no-input
    behavior. This is per-control-step scalar math, so it follows the TS
    line by line instead of vectorizing.
    """
    config = config or {}
    if config.get("poseVelocityCommand") not in (True, "true", 1, "1"):
        return output
    if context is None:
        raise _error(
            "mujocoObservations",
            "poseVelocityCommand requires MuJoCo model/data context",
        )
    if len(output) < 3:
        raise _error(
            "mujocoObservations", "poseVelocityCommand requires command dim >= 3"
        )
    if not node_id:
        raise _error(
            "mujocoObservations", "poseVelocityCommand obs-commands requires nodeId"
        )

    axes = command_axis_config(config, len(output))
    indices, minimum, maximum = axes["indices"], axes["min"], axes["max"]
    raw_vx = float(output[indices[0]])
    raw_yaw = float(output[indices[2]])
    yaw_min_forward_velocity = _non_negative_finite_number(
        config.get("yawMinForwardVelocity"), "yawMinForwardVelocity", 0
    )
    # Yaw-only commands still get a minimum forward velocity so the robot
    # keeps walking through turns.
    yaw_forward_assist_active = (
        yaw_min_forward_velocity > 0
        and raw_vx <= FORWARD_ACTIVE_EPSILON
        and abs(raw_yaw) > YAW_ACTIVE_EPSILON
    )
    if yaw_forward_assist_active:
        output[indices[0]] = _clamp(yaw_min_forward_velocity, minimum[0], maximum[0])
        return output

    commanded_vx = float(output[indices[0]])
    forward_active = commanded_vx > FORWARD_ACTIVE_EPSILON
    pose_x, pose_y, pose_yaw = _root_pose_2d(context)

    state = _pose_velocity_state_by_node.get(node_id)
    if state is None or not forward_active or not state["wasForwardActive"]:
        state = {
            "centerY": pose_y,
            "targetX": pose_x
            + _positive_finite_number(
                config.get("targetLookahead"),
                "targetLookahead",
                DEFAULT_POSE_VELOCITY_LOOKAHEAD,
            ),
            "wasForwardActive": forward_active,
        }
        _pose_velocity_state_by_node[node_id] = state

    if not forward_active:
        state["wasForwardActive"] = False
        return output

    lookahead = _positive_finite_number(
        config.get("targetLookahead"),
        "targetLookahead",
        DEFAULT_POSE_VELOCITY_LOOKAHEAD,
    )
    velocity_stiffness = _positive_finite_number(
        config.get("velocityControlStiffness"),
        "velocityControlStiffness",
        DEFAULT_POSE_VELOCITY_STIFFNESS,
    )
    heading_stiffness = _positive_finite_number(
        config.get("headingControlStiffness"),
        "headingControlStiffness",
        DEFAULT_HEADING_STIFFNESS,
    )
    target_distance_threshold = _non_negative_finite_number(
        config.get("targetDisThreshold"),
        "targetDisThreshold",
        DEFAULT_TARGET_DISTANCE_THRESHOLD,
    )

    state["wasForwardActive"] = True
    state["targetX"] = max(state["targetX"], pose_x + lookahead)

    dx = state["targetX"] - pose_x
    dy = state["centerY"] - pose_y
    target_distance = math.hypot(dx, dy)
    if target_distance <= target_distance_threshold:
        output[indices[0]] = 0
        output[indices[1]] = 0
        output[indices[2]] = raw_yaw
        return output

    cos_yaw = math.cos(pose_yaw)
    sin_yaw = math.sin(pose_yaw)
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    target_heading = math.atan2(dy, dx)
    heading_command = _wrap_pi(target_heading - pose_yaw) * heading_stiffness

    output[indices[0]] = _clamp(
        body_x * velocity_stiffness, minimum[0], min(maximum[0], commanded_vx)
    )
    output[indices[1]] = _clamp(body_y * velocity_stiffness, minimum[1], maximum[1])
    output[indices[2]] = _clamp(heading_command + raw_yaw, minimum[2], maximum[2])
    return output


def _command_vector(config, node_id, context):
    config = config or {}
    command_mode = config.get("commandMode")
    if command_mode in (
        "keyboard-buttons",
        "gamepad-buttons",
        "keyboard-gamepad-buttons",
    ):
        return np.zeros(1, dtype=np.float32)
    if command_mode in _INTERACTIVE_COMMAND_MODES:
        # Explicit native-host input state is neutral. The declared project
        # command therefore stays unchanged, matching browser no-input state.
        assert not _LOCAL_COMMAND_INPUT_STATE["pressed_keys"]
        assert not _LOCAL_COMMAND_INPUT_STATE["pressed_buttons"]
        assert not _LOCAL_COMMAND_INPUT_STATE["axis_active"]

    dim = configured_dim(config, DEFAULT_COMMAND_DIM)
    raw = _first_present(config, ("values", "commands", "command", "cmd", "cmd_init"))
    if isinstance(raw, dict):
        output = _object_command_vector(raw, dim)
    else:
        output = vector_from(raw if raw is not None else [], dim, "obs-commands")

    if config.get("commandEncoding") in ("one-hot", "mapped") or config.get("encoding") in (
        "one-hot",
        "mapped",
    ):
        return _mapped_command_vector(config, output, context)
    return _pose_velocity_command_vector(config, node_id, output, context)


def _write_command_parameters(node_data, context):
    names = _string_list(node_data.get("parameterNames"), "obs-commands parameterNames")
    values = node_data.get("values")
    if not (
        isinstance(values, np.ndarray)
        and values.dtype == np.float32
        and len(values) == len(names)
    ):
        values = np.zeros(len(names), dtype=np.float32)
    parameters = context.get("parameters")
    if not callable(parameters):
        raise _error(
            "botlab",
            "obs-commands requires a parameters callable (name -> float) in "
            "the observation context",
        )
    for index, name in enumerate(names):
        values[index] = parameters(name)
    node_data["values"] = values
