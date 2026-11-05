"""Declarative observation order: expand config/observation.yaml's `order` section.

`observe()` is sugar over the same Signal primitives a handwritten controller
calls -- `robot.<primitive>()`, `sensor.<read>()`, `concat`, `history` -- so an
order spec and the equivalent expressions lower to the identical plan. The
channel vocabulary below is closed on purpose: anything outside it (escape
hatches like `mujoco()`, `contact()` builders, policy-specific processing)
stays in controller.py.
"""

from ._history import history
from ._resources import Sensor
from ._robot import waypoint_tracking
from ._signal import concat
from ._spatial import rotate

# Robot methods that read observation state and return a Signal.
_ROBOT_CHANNELS = frozenset({
    "base_angular_velocity",
    "base_linear_velocity",
    "projected_gravity",
    "base_height",
    "imu",
    "joint_position",
    "joint_velocity",
    "last_action",
    "commands",
    "body_position",
    "body_quaternion",
    "body_linear_velocity",
    "body_angular_velocity",
    "sensor_data",
    "height_scan",
})

# Calibration vectors an order entry may name instead of spelling out.
_CALIBRATION_VECTORS = ("default_angles",)

# Calibration quaternions a `rotate:` entry may name (IMU mounting correction).
_ROTATION_VECTORS = ("imu_extrinsic",)

# Camera channels that are plain Signals. `depth_frame` returns a frame object
# for custom pipelines, so those stay in controller.py.
_SENSOR_READS = ("rgb", "depth")

_HISTORY_KEYS = ("steps", "stride", "order", "layout", "fill")
_GROUP_KEYS = frozenset({"channels", "sensor", "read", "params", "history"})


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve_channel_params(params, label, robot, io):
    params = dict(_require_mapping(params, f"{label} params"))
    io_section = params.pop("io", None)
    if io_section is not None:
        if io is None:
            raise ValueError(
                f'{label} references io "{io_section}", but no io mapping was given'
            )
        if io_section not in io:
            raise ValueError(f'io has no "{io_section}" section (available: {sorted(io)})')
        params.update(io[io_section])
    if "waypoint" in params:
        waypoint = _require_mapping(params["waypoint"], f"{label} waypoint")
        params["waypoint"] = waypoint_tracking(**waypoint)
    default = params.get("default")
    if isinstance(default, str):
        if default not in _CALIBRATION_VECTORS:
            raise ValueError(
                f'{label} default "{default}" is not a named calibration vector '
                f"(available: {list(_CALIBRATION_VECTORS)})"
            )
        params["default"] = robot.default_angles
    return params


def _resolve_rotation(value, label, robot):
    """A `rotate:` entry names a calibration quaternion or spells one out (wxyz)."""
    if value is None:
        return None
    if isinstance(value, str):
        if value not in _ROTATION_VECTORS:
            raise ValueError(
                f'{label} rotate "{value}" is not a named calibration quaternion '
                f"(available: {list(_ROTATION_VECTORS)})"
            )
        return getattr(robot, value)
    if not isinstance(value, list):
        raise TypeError(f"{label} rotate must be a name or a wxyz list")
    return value


def _build_channel(entry, index, group_label, robot, io):
    if not isinstance(entry, dict) or len(entry) != 1:
        raise TypeError(
            f'"{group_label}" channel {index} must have exactly one channel kind, '
            f"got {sorted(entry) if isinstance(entry, dict) else entry}"
        )
    (name, params), = entry.items()
    if name not in _ROBOT_CHANNELS:
        raise ValueError(
            f'"{group_label}" channel "{name}" is not an observation primitive '
            f"(available: {sorted(_ROBOT_CHANNELS)})"
        )
    label = f'"{group_label}".{name}'
    params = dict(params or {})
    rotation = _resolve_rotation(params.pop("rotate", None), label, robot)
    resolved = _resolve_channel_params(params, label, robot, io)
    signal = getattr(robot, name)(**resolved)
    return rotate(signal, rotation) if rotation is not None else signal


def _build_history(source, spec, group_label):
    if not isinstance(spec, dict) or not spec:
        raise TypeError(f'"{group_label}".history must be a non-empty mapping')
    unknown = sorted(set(spec) - set(_HISTORY_KEYS))
    if unknown:
        raise ValueError(f'"{group_label}".history has unknown keys {unknown}')
    steps = spec.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError(f'"{group_label}".history needs a positive integer steps')
    kwargs = {key: spec[key] for key in _HISTORY_KEYS if key != "steps" and key in spec}
    return history(source, steps, **kwargs)


def observe(order, *, robot, sensors=None, io=None):
    """Build the observation signals named by an `order` spec.

    `order` is the `order:` section of ``config/observation.yaml``: a mapping of
    group name to group spec, and the group names are the returned keys.

    ```yaml
    order:
      state:
        channels:
          - base_angular_velocity: {scale: 0.25}
          - projected_gravity: {}
          - commands: {io: velocity_command, waypoint: {min_forward_velocity: 0.1}}
          - joint_position: {default: default_angles}
        history: {steps: 8, layout: by-term}
      depth:
        sensor: depth_camera
        read: depth
        params: {max_age: 5}
        history: {steps: 8, stride: 5}
    ```

    A `channels` group calls the Robot observation primitives in list order and
    concatenates them; a `sensor` group reads one camera channel of the
    `sensors` entry it names. `history:` wraps the group in :func:`history`
    with those exact kwargs. Four params are symbolic rather than literal:
    `io: <section>` merges that section of the `io` mapping into the kwargs,
    `waypoint: {..}` expands to ``waypoint_tracking(**{..})``,
    `default: default_angles` names the robot's calibrated default vector, and
    `rotate: imu_extrinsic` wraps the channel in
    ``rotate(signal, robot.imu_extrinsic)`` -- the IMU mounting correction, or
    a literal wxyz list for any other fixed rotation. Everything else passes
    through to the primitive unchanged.
    """
    order = _require_mapping(order, "order")
    if not order:
        raise ValueError("order must be a non-empty mapping")
    if sensors is None:
        sensors = {}

    outputs = {}
    for group_label, spec in order.items():
        spec = _require_mapping(spec, f'order group "{group_label}"')
        unknown = sorted(set(spec) - _GROUP_KEYS)
        if unknown:
            raise ValueError(f'"{group_label}" has unknown keys {unknown}')

        if "channels" in spec:
            if "sensor" in spec or "read" in spec:
                raise ValueError(f'"{group_label}" mixes channels with a sensor read')
            channels = spec["channels"]
            if not isinstance(channels, list) or not channels:
                raise TypeError(f'"{group_label}".channels must be a non-empty list')
            signal = concat([
                _build_channel(entry, index, group_label, robot, io)
                for index, entry in enumerate(channels)
            ])
        elif "sensor" in spec:
            sensor_label = spec["sensor"]
            if sensor_label not in sensors:
                raise ValueError(
                    f'"{group_label}" reads sensor "{sensor_label}", '
                    f"but the sensors given are {sorted(sensors)}"
                )
            sensor = sensors[sensor_label]
            if not isinstance(sensor, Sensor):
                raise TypeError(f'"{group_label}" sensor "{sensor_label}" is not a Sensor')
            read = spec.get("read")
            if read not in _SENSOR_READS:
                raise ValueError(
                    f'"{group_label}" read "{read}" is not one of {list(_SENSOR_READS)}'
                )
            params = spec.get("params") or {}
            params = _require_mapping(params, f'"{group_label}".params')
            signal = getattr(sensor, read)(**params)
        else:
            raise ValueError(f'"{group_label}" needs either channels or a sensor read')

        if "history" in spec:
            signal = _build_history(signal, spec["history"], group_label)
        outputs[group_label] = signal
    return outputs
