"""The Robot resource: observations, calibration, commands and output stages."""
import botlab_host

from ._declaration import (
    _ResourceValues,
    _declare_resource,
    _host_list,
    _host_snapshot,
    _host_value,
    _non_empty_string,
    _policy_pd_vector,
    default_joint_positions,
    resource_config,
)
from ._runtime_types import _ContactSignals, _entity_targets
from ._signal import Signal, _as_signal, concat
from ._spatial import relative_position, relative_quaternion
from ._vectors import (
    _expand_vector,
    _finite_number,
    _finite_values,
    _non_negative_integer,
    _positive_integer,
    require_vector,
)


def waypoint_tracking(
    *,
    lookahead=None,
    velocity_stiffness=None,
    heading_stiffness=None,
    distance_threshold=None,
    min_forward_velocity=None,
):
    """Turn a velocity command into waypoint tracking against the robot pose.

    Every argument is optional; anything left out keeps the engine's own default.
    """
    fields = {}
    for field, value, label in (
        ("targetLookahead", lookahead, "lookahead"),
        (
            "velocityControlStiffness",
            velocity_stiffness,
            "velocity_stiffness",
        ),
        ("headingControlStiffness", heading_stiffness, "heading_stiffness"),
    ):
        if value is None:
            continue
        value = _finite_number(value, f"waypoint_tracking {label}")
        if value <= 0:
            raise ValueError(f"waypoint_tracking {label} must be > 0")
        fields[field] = value
    for field, value, label in (
        ("targetDisThreshold", distance_threshold, "distance_threshold"),
        ("yawMinForwardVelocity", min_forward_velocity, "min_forward_velocity"),
    ):
        if value is None:
            continue
        value = _finite_number(value, f"waypoint_tracking {label}")
        if value < 0:
            raise ValueError(f"waypoint_tracking {label} must be >= 0")
        fields[field] = value
    data = {"poseVelocityCommand": True}
    data.update(fields)
    return data


class Robot:
    def __init__(self, source, *, preset=None, config=None, name=None):
        self.id = _declare_resource(
            "robot",
            source,
            preset=preset,
            config=config,
            name=name,
        )

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    # -- raw state, for controllers that run their own Python step ----------

    def observe(self):
        return _host_snapshot(botlab_host.observe(self.id))

    def action_dimension(self):
        return int(botlab_host.actionDimension(self.id))

    def default_joint_positions(self):
        return _host_list(botlab_host.defaultJointPositions(self.id))

    def control_parameter(self, name):
        return _host_list(botlab_host.controlParameter(self.id, str(name)))

    def position_pd(self, target, kp, kd, effort_limit):
        dimension = self.action_dimension()
        target_values = require_vector(target, dimension, "position_pd target")
        kp_values = _expand_vector(kp, dimension, "position_pd kp")
        kd_values = _expand_vector(kd, dimension, "position_pd kd")
        effort_values = _expand_vector(
            effort_limit,
            dimension,
            "position_pd effort_limit",
        )
        if any(value <= 0 for value in effort_values):
            raise ValueError("position_pd effort_limit values must be positive")
        return {
            "kind": "position-pd",
            "target": target_values,
            "kp": kp_values,
            "kd": kd_values,
            "effort_limit": effort_values,
        }

    def apply_action(self, action):
        botlab_host.applyAction(self.id, action)

    # -- preset-backed calibration -----------------------------------------
    #
    # Joint layout and actuator gains are asset calibration: they live in the
    # robot preset, and projects reference them instead of copying long vectors.

    def _calibration(self, field):
        return _ResourceValues(
            resource_config(self.id, field),
            lambda: botlab_host.controlParameter(self.id, field),
        )

    @property
    def policy_order(self):
        """Native joint index for each policy slot."""
        return self._calibration("policyJointOrder")

    @property
    def motor_order(self):
        """Policy slot for each native motor; index an action with it."""
        return self._calibration("motorJointOrder")

    @property
    def action_scale(self):
        return self._calibration("actionScale")

    @property
    def kp(self):
        return self._calibration("kp")

    @property
    def kd(self):
        return self._calibration("kd")

    @property
    def effort_limit(self):
        return self._calibration("effortLimit")

    @property
    def default_angles(self):
        return _ResourceValues(
            default_joint_positions(self.id),
            lambda: botlab_host.defaultJointPositions(self.id),
        )

    # -- observation signals -----------------------------------------------

    def base_angular_velocity(self, *, scale=None, source="root"):
        return Signal("obs-base-ang-vel", "base-angular-velocity", _observation_data(
            {
                "dim": 3,
                "velocitySource": _non_empty_string(source, "velocity source"),
                "rootAngularVelocityFrame": "body",
            },
            scale=scale,
        ))

    def base_linear_velocity(self, *, scale=None, source="root"):
        return Signal("obs-base-lin-vel", "base-linear-velocity", _observation_data(
            {"dim": 3, "velocitySource": _non_empty_string(source, "velocity source")},
            scale=scale,
        ))

    def projected_gravity(self, *, target="root"):
        target = _non_empty_string(target, "projected gravity target")
        data = {"dim": 3, "targetKind": "root" if target == "root" else "body"}
        if target != "root":
            data["target"] = target
        return Signal("obs-projected-gravity", "projected-gravity", data)

    def base_height(self):
        return Signal("obs-base-height", "base-height", {"dim": 1})

    def _entity_signal(self, query, targets, width):
        signals = [self.mujoco(query, width, target=str(target)) for target in _entity_targets(targets)]
        return signals[0] if len(signals) == 1 else concat(signals)

    def body_position(self, bodies, *, relative_to=None):
        """Body origins in world coordinates, or expressed in another body's frame."""
        positions = self._entity_signal("body.xpos", bodies, 3)
        if relative_to is None:
            return positions
        return relative_position(positions, self.body_position(relative_to), self.body_quaternion(relative_to))

    def body_quaternion(self, bodies, *, relative_to=None):
        """Body orientations in wxyz order, optionally relative to another body."""
        orientations = self._entity_signal("body.xquat", bodies, 4)
        if relative_to is None:
            return orientations
        return relative_quaternion(orientations, self.body_quaternion(relative_to))

    def body_linear_velocity(self, bodies, *, frame="world"):
        """Velocity at each body origin, explicitly in world or its local frame."""
        if frame not in ("world", "local"):
            raise ValueError('body velocity frame must be "world" or "local"')
        return self._entity_signal(f"body.linvel.{frame}", bodies, 3)

    def body_angular_velocity(self, bodies, *, frame="world"):
        if frame not in ("world", "local"):
            raise ValueError('body velocity frame must be "world" or "local"')
        return self._entity_signal(f"body.angvel.{frame}", bodies, 3)

    def sensor_data(self, name, *, size):
        """Read a named MJCF sensor (gyro, accelerometer, force, torque, etc.)."""
        return self.mujoco("sensor.data", _positive_integer(size, "sensor_data size"),
                           target=_non_empty_string(name, "sensor_data name"))

    def contact(self, targets, *, kind="body"):
        """World net forces, touching flags and physics-step contact timing signals."""
        if kind not in ("body", "geom"):
            raise ValueError('contact kind must be "body" or "geom"')
        return _ContactSignals(self, _entity_targets(targets), kind)

    def joint_position(self, *, order="policy", default=None, scale=None, _label="joint-position"):
        """Joint positions as `(position - default) * scale`, read in `order`.

        `order` is `"policy"` (this robot's calibrated policy joint order),
        `"native"` (the model's own order), or an explicit index sequence.
        """
        return Signal("obs-dof-pos", _label, _observation_data(
            {"dim": self.action_dimension()},
            order=self._joint_order(order),
            default=default,
            scale=scale,
        ))

    def joint_velocity(self, *, order="policy", default=None, scale=None, _label="joint-velocity"):
        return Signal("obs-dof-vel", _label, _observation_data(
            {"dim": self.action_dimension()},
            order=self._joint_order(order),
            default=default,
            scale=scale,
        ))

    def _joint_order(self, order):
        if order == "policy":
            return self.policy_order
        if order == "native":
            return None
        return order

    def imu(self, *, order=None, scale=None):
        """Root orientation quaternion followed by base angular velocity."""
        indices = None if order is None else list(order)
        return Signal("obs-imu", "imu", _observation_data(
            {"dim": 7 if indices is None else len(indices)},
            order=indices,
            scale=scale,
        ))

    def qpos(self, indices, *, default=None, scale=None):
        """Raw MuJoCo qpos entries, root included, gathered in the order given.

        Unlike `joint_position` this is the model's own layout, so the entries to read
        have to be named explicitly.
        """
        selected = list(indices)
        if not selected:
            raise ValueError("qpos indices must not be empty")
        return Signal("obs-qpos", "qpos", _observation_data(
            {"dim": len(selected)},
            order=selected,
            default=default,
            scale=scale,
        ))

    def qvel(self, indices, *, default=None, scale=None):
        """Raw MuJoCo qvel entries, root included, gathered in the order given."""
        selected = list(indices)
        if not selected:
            raise ValueError("qvel indices must not be empty")
        return Signal("obs-qvel", "qvel", _observation_data(
            {"dim": len(selected)},
            order=selected,
            default=default,
            scale=scale,
        ))

    def mujoco(self, query, size, *, target=None, start=None, scale=None):
        """Read `size` values from a MuJoCo model/data query, such as "sensor.data".

        This is the escape hatch for state the named observations above do not cover.
        `size` is required because the engine sizes the signal before the model is loaded.
        """
        data = {
            "query": _non_empty_string(query, "mujoco query"),
            "dim": _positive_integer(size, "mujoco size"),
            "length": _positive_integer(size, "mujoco size"),
        }
        if target is not None:
            data["target"] = _non_empty_string(str(target), "mujoco target")
        if start is not None:
            data["start"] = _non_negative_integer(start, "mujoco start")
        return Signal("obs-mujoco-query", f"mujoco-{query}", _observation_data(data, scale=scale))

    def height_scan(
        self,
        points,
        *,
        body,
        frame="yaw",
        lift=1.0,
        max_distance=10.0,
        geom_groups=None,
        scale=None,
    ):
        """Terrain height under each sample point, in world coordinates.

        Each point casts a ray straight down from `lift` above `body` and reports the world
        z of the first geom it hits, so stairs, boxes and meshes read the same way a
        heightfield does. `frame="yaw"` rotates the offsets by the body's heading;
        `frame="world"` keeps them axis-aligned. Nothing within `max_distance` reads as the
        bottom of that range, which keeps the signal a fixed-width vector.

        Heights are absolute: subtract what the policy was trained against, for example
        `robot.height_scan(...) - robot.base_height()`.

        Rays hit the robot's own geoms too -- `geom_groups` is the 6-flag MuJoCo group
        filter that keeps a scan on terrain geoms only.
        """
        samples = []
        for index, point in enumerate(points):
            pair = list(point)
            if len(pair) != 2:
                raise ValueError(f"height_scan point {index} must be an (x, y) pair")
            samples.extend(_finite_values(pair, f"height_scan point {index}"))
        if not samples:
            raise ValueError("height_scan points must not be empty")
        if frame not in ("yaw", "world"):
            raise ValueError('height_scan frame must be "yaw" or "world"')
        data = {
            "dim": len(samples) // 2,
            "points": samples,
            "body": _non_empty_string(str(body), "height_scan body"),
            "frame": frame,
            "lift": _finite_number(lift, "height_scan lift"),
            "maxDistance": _finite_number(max_distance, "height_scan max_distance"),
        }
        if data["lift"] <= 0 or data["maxDistance"] <= 0:
            raise ValueError("height_scan lift and max_distance must be positive")
        if geom_groups is not None:
            groups = [int(value) for value in geom_groups]
            if len(groups) != 6 or any(value not in (0, 1) for value in groups):
                raise ValueError("height_scan geom_groups must be six 0/1 flags")
            data["geomGroups"] = groups
        return Signal(
            "obs-height-scan",
            "height-scan",
            _observation_data(data, scale=scale),
        )

    def read(self, query, *, target=None, start=None, length=None):
        """Read a MuJoCo model/data query inside `Controller.step`.

        Same query names as the `mujoco()` observation signal -- "sensordata",
        "site.xpos", "joint.qfrc_actuator", "actuator.force" and the rest -- but
        returned as plain numbers for Python control code instead of a Signal.
        Named entities take `target`; `start` and `length` slice the result.
        """
        values = botlab_host.readMujoco(
            self.id,
            _non_empty_string(query, "read query"),
            None if target is None else _non_empty_string(str(target), "read target"),
            None if start is None else _non_negative_integer(start, "read start"),
            None if length is None else _positive_integer(length, "read length"),
        )
        return _host_list(values)

    def last_action(self):
        """The previous frame's Signal.action() value."""
        return Signal("obs-actions", "last-action", {"dim": self.action_dimension()})

    def velocity_pd(self, target, *, kp=None, effort_limit=None):
        """Convert a native-order joint velocity target into clipped actuator effort."""
        target = _as_signal(target, "velocity_pd target")
        dimension = self.action_dimension()
        return Signal(
            "proc-pd",
            "velocity-pd",
            {
                "pdType": "velocity",
                "kps": (
                    self.kp
                    if kp is None
                    else _policy_pd_vector(kp, dimension, "velocity_pd kp")
                ),
                "effortLimit": (
                    self.effort_limit
                    if effort_limit is None
                    else _policy_pd_vector(
                        effort_limit,
                        dimension,
                        "velocity_pd effort_limit",
                    )
                ),
            },
            (target, self.joint_velocity(order="native", _label="velocity-pd-joint-velocity")),
        )

    def effort_control(self, effort, *, effort_limit=None):
        """Clip a native-order effort signal to the robot's actuator limits."""
        effort = _as_signal(effort, "effort_control effort")
        dimension = self.action_dimension()
        return Signal(
            "proc-pd",
            "effort-control",
            {
                "pdType": "effort",
                "effortLimit": (
                    self.effort_limit
                    if effort_limit is None
                    else _policy_pd_vector(
                        effort_limit,
                        dimension,
                        "effort_control effort_limit",
                    )
                ),
            },
            (effort,),
        )

    def commands(
        self,
        *parameter_names,
        waypoint=None,
        input_mode="static",
        ramp_rate=None,
        keys=None,
        bindings=None,
        mappings=None,
    ):
        """Command vector read from project parameters, one axis per name.

        Axis limits come from each parameter's `range` in project(), so they
        are declared once rather than restated here. A ramp_rate of zero
        disables smoothing and applies interactive input immediately.
        """
        names = [
            _non_empty_string(name, f"command parameter {index}")
            for index, name in enumerate(parameter_names)
        ]
        if not names:
            raise ValueError("commands requires at least one parameter name")
        allowed_modes = ("static", "keyboard", "gamepad", "keyboard-gamepad")
        if input_mode not in allowed_modes:
            raise ValueError(f"commands input_mode must be one of {allowed_modes}")
        data = {
            "dim": len(names),
            "parameterNames": names,
            "commandMode": input_mode,
        }
        if ramp_rate is not None:
            ramp_rate = _finite_number(ramp_rate, "commands ramp_rate")
            if ramp_rate < 0:
                raise ValueError("commands ramp_rate must be non-negative")
            if input_mode not in ("keyboard", "gamepad", "keyboard-gamepad"):
                raise ValueError(
                    "commands ramp_rate requires keyboard, gamepad, or "
                    "keyboard-gamepad input_mode"
                )
            data["rampRate"] = ramp_rate
        ranges = [botlab_host.parameterRange(name) for name in names]
        bounds = [_host_value(item) for item in ranges]
        if all(bound is not None for bound in bounds):
            data["cmdMin"] = [float(low) for low, _high in bounds]
            data["cmdMax"] = [float(high) for _low, high in bounds]
        configured_inputs = sum(
            value is not None for value in (keys, bindings, mappings)
        )
        if configured_inputs > 1:
            raise ValueError("commands accepts only one of keys, bindings, or mappings")
        if keys is not None:
            if not isinstance(keys, dict):
                raise TypeError("commands keys must be a mapping")
            key_fields = {
                "forward": "keyForward",
                "backward": "keyBackward",
                "left": "keyLeft",
                "right": "keyRight",
                "yaw_left": "keyYawLeft",
                "yaw_right": "keyYawRight",
            }
            unknown = set(keys) - set(key_fields)
            if unknown:
                raise ValueError(f"commands keys has unknown fields {sorted(unknown)}")
            for name, value in keys.items():
                if value is None or value is False:
                    data[key_fields[name]] = False
                elif isinstance(value, str) and value:
                    data[key_fields[name]] = value
                else:
                    raise ValueError(
                        f"commands keys.{name} must be a non-empty string, None, or False"
                    )
        if bindings is not None:
            if not isinstance(bindings, dict) or not bindings:
                raise ValueError("commands bindings must be a non-empty mapping")
            if input_mode not in ("keyboard", "keyboard-gamepad"):
                raise ValueError(
                    "commands bindings require keyboard or keyboard-gamepad input_mode"
                )
            if not all(bound is not None for bound in bounds):
                missing = [name for name, bound in zip(names, bounds) if bound is None]
                raise ValueError(
                    "commands interactive bindings require parameter ranges for "
                    f"{missing}"
                )
            normalized_bindings = {}
            used_keys = set()
            for raw_axis, binding in bindings.items():
                if isinstance(raw_axis, int) and not isinstance(raw_axis, bool):
                    if raw_axis < 0 or raw_axis >= len(names):
                        raise ValueError(
                            f"commands binding axis {raw_axis} is outside {len(names)} parameters"
                        )
                    axis = str(raw_axis)
                elif isinstance(raw_axis, str) and raw_axis in names:
                    axis = raw_axis
                else:
                    raise ValueError(
                        "commands binding keys must be declared parameter names or axis indices"
                    )
                if not isinstance(binding, dict) or not binding:
                    raise ValueError(f"commands binding {raw_axis!r} must be a non-empty mapping")
                unknown = set(binding) - {"increase", "decrease"}
                if unknown:
                    raise ValueError(
                        f"commands binding {raw_axis!r} has unknown fields {sorted(unknown)}"
                    )
                normalized = {}
                for direction in ("increase", "decrease"):
                    if direction not in binding:
                        continue
                    key = binding[direction]
                    if key is None or key is False:
                        normalized[direction] = False
                    elif isinstance(key, str) and key:
                        if key in used_keys:
                            raise ValueError(f'commands binding key "{key}" is duplicated')
                        used_keys.add(key)
                        normalized[direction] = key
                    else:
                        raise ValueError(
                            f"commands binding {raw_axis!r}.{direction} must be a "
                            "non-empty string, None, or False"
                        )
                if not normalized:
                    raise ValueError(
                        f"commands binding {raw_axis!r} requires increase or decrease"
                    )
                if not any(key is not False for key in normalized.values()):
                    raise ValueError(
                        f"commands binding {raw_axis!r} requires an enabled key"
                    )
                normalized_bindings[axis] = normalized
            data["commandBindings"] = normalized_bindings
        if mappings is not None:
            if input_mode != "static":
                raise ValueError("commands mappings require input_mode='static'")
            if waypoint is not None:
                raise ValueError("commands mappings cannot be combined with waypoint tracking")
            if not isinstance(mappings, (list, tuple)) or not mappings:
                raise ValueError("commands mappings must be a non-empty sequence")
            normalized_mappings = []
            used_chords = set()
            for index, mapping in enumerate(mappings):
                label = f"commands mappings[{index}]"
                if not isinstance(mapping, dict):
                    raise TypeError(f"{label} must be a mapping")
                unknown = set(mapping) - {"key", "keys", "value", "values", "label"}
                if unknown:
                    raise ValueError(f"{label} has unknown fields {sorted(unknown)}")
                if ("key" in mapping) == ("keys" in mapping):
                    raise ValueError(f"{label} requires exactly one of key or keys")
                if "key" in mapping:
                    key = _non_empty_string(mapping["key"], f"{label} key")
                    chord = (key,)
                    key_data = {"key": key}
                else:
                    raw_keys = mapping["keys"]
                    if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
                        raise ValueError(f"{label} keys must be a non-empty sequence")
                    chord = tuple(
                        _non_empty_string(key, f"{label} keys[{key_index}]")
                        for key_index, key in enumerate(raw_keys)
                    )
                    if len(set(chord)) != len(chord):
                        raise ValueError(f"{label} keys must be unique")
                    key_data = {"keys": list(chord)}
                chord_key = tuple(sorted(chord))
                if chord_key in used_chords:
                    raise ValueError(f"{label} duplicates an earlier key chord")
                used_chords.add(chord_key)

                if ("value" in mapping) == ("values" in mapping):
                    raise ValueError(f"{label} requires exactly one of value or values")
                if "value" in mapping:
                    if len(names) != 1:
                        raise ValueError(f"{label} value requires a one-dimensional command")
                    values = _finite_values([mapping["value"]], f"{label} value")
                else:
                    values = require_vector(mapping["values"], len(names), f"{label} values")
                for axis, value in enumerate(values):
                    bound = bounds[axis]
                    if bound is not None and not (float(bound[0]) <= value <= float(bound[1])):
                        raise ValueError(
                            f"{label} value {value} is outside parameter {names[axis]!r} range"
                        )
                normalized = {**key_data, "values": values}
                if "label" in mapping:
                    normalized["label"] = _non_empty_string(
                        mapping["label"],
                        f"{label} label",
                    )
                normalized_mappings.append(normalized)
            data["commandEncoding"] = "mapped"
            data["commandMappings"] = normalized_mappings
        if waypoint is not None:
            if not isinstance(waypoint, dict):
                raise TypeError("commands waypoint must come from waypoint_tracking()")
            data.update(waypoint)
        return Signal("obs-commands", "commands", data)


def _observation_data(data, *, order=None, default=None, scale=None):
    if order is not None:
        data["indices"] = order
    if default is not None:
        data["defaultAngles"] = default
    if scale is not None:
        data["scale"] = scale
    return data
