"""The Robot resource: observations, calibration, commands and output stages."""
import botlab_host

from ._declaration import (
    _ResourceValues,
    _declare_resource,
    _host_list,
    _host_snapshot,
    _host_value,
    _localized_text,
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
    def imu_extrinsic(self):
        """IMU-to-body mounting rotation, wxyz; preset ``imuExtrinsic``.

        IMU readings arrive in the mounted frame; ``rotate(reading, imu_extrinsic)``
        expresses them in the body frame the policy was trained on.
        """
        return self._calibration("imuExtrinsic")

    @property
    def default_angles(self):
        return _ResourceValues(
            default_joint_positions(self.id),
            lambda: botlab_host.defaultJointPositions(self.id),
        )

    # -- observation signals -----------------------------------------------

    def base_angular_velocity(self, *, scale=None, source="root", imu_sensors=None):
        return Signal("obs-base-ang-vel", "base-angular-velocity", _observation_data(
            _with_imu_sensors({
                "dim": 3,
                "velocitySource": _non_empty_string(source, "velocity source"),
                "rootAngularVelocityFrame": "body",
            }, imu_sensors),
            scale=scale,
        ))

    def base_linear_velocity(self, *, scale=None, source="root", imu_sensors=None):
        return Signal("obs-base-lin-vel", "base-linear-velocity", _observation_data(
            _with_imu_sensors(
                {"dim": 3, "velocitySource": _non_empty_string(source, "velocity source")},
                imu_sensors,
            ),
            scale=scale,
        ))

    def projected_gravity(self, *, target="root", imu_sensors=None):
        target = _non_empty_string(target, "projected gravity target")
        data = {"dim": 3, "targetKind": "root" if target == "root" else "body"}
        if target != "root":
            data["target"] = target
        return Signal("obs-projected-gravity", "projected-gravity", _with_imu_sensors(data, imu_sensors))

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

    def imu(self, *, order=None, scale=None, imu_sensors=None):
        """Root orientation quaternion followed by base angular velocity."""
        indices = None if order is None else list(order)
        return Signal("obs-imu", "imu", _observation_data(
            _with_imu_sensors({"dim": 7 if indices is None else len(indices)}, imu_sensors),
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
        parameters=None,
        keyboard=None,
        gamepad=None,
        magnitude=None,
        ramp_rate=None,
        waypoint=None,
        rest=None,
    ):
        """Command vector read from project parameters, one axis per name.

        The vector is declared once, here in code -- it is the policy-side
        contract. Axis limits come from each parameter's `range` in project(),
        so they are declared once rather than restated here. ``keyboard=`` and
        ``gamepad=`` carry the devices' pure binding tables, the sections a
        ``config/keyboard.yaml`` / ``config/gamepad.yaml`` file provides:

        ``keyboard={"bindings": {axis: {increase, decrease}}}`` or
        ``{"mappings": [{key, value, label}]}``; ``gamepad={"bindings": {axis:
        {stick, axis, invert}}}``, ``{"mappings": [{button, value, label}]}``
        or ``deadzone``. Binding axes are parameter names (or axis indices)
        of this vector; neither device section declares the vector, so the
        two files never mirror each other. ``commandMode`` derives from which
        devices carry axis bindings (keyboard / gamepad / keyboard-gamepad).
        A ``ramp_rate`` of zero disables smoothing and applies interactive
        input immediately.

        When a device section carries ``mappings`` the call takes no parameter
        axes: the mappings' ``value`` / ``values`` ARE the command's value set
        -- a discrete mode command is a control quantity, not a tunable
        parameter, so it never appears in project() or the parameter panel.
        All mappings (both devices) must share one width; ``rest=`` gives the
        resting code the vector returns while no trigger is held (a number,
        a sequence matching the width, or the default zero).

        An axis may also be a finite number instead of a parameter name: a
        constant axis pinned to that value. It keeps the policy's trained
        command layout (e.g. a ``(forward, lateral, yaw)`` policy driven with
        lateral fixed at zero takes ``("forward_velocity", 0, "yaw_rate")``)
        without declaring a dead parameter for it; the plan carries it as an
        empty-string axis name. Constant axes clamp keys, sticks, and touch
        to the constant, and need the named axes' parameter ranges so the
        plan emits one bounds vector.

        ``magnitude=[names]`` marks axes as magnitude axes: they rest at zero
        and interactive inputs (keys, gamepad stick, touch) command plus or
        minus the parameter's current value instead of ramping between the
        range endpoints. The slider then sets the speed a key press commands,
        not a cruise speed.
        """
        if parameters is not None:
            if parameter_names:
                raise ValueError(
                    "commands takes either positional parameter names or parameters=[...], not both"
                )
            if not isinstance(parameters, (list, tuple)) or not parameters:
                raise ValueError(
                    "commands parameters must be a non-empty sequence of parameter names "
                    "or axis constants"
                )
            parameter_names = parameters
        keyboard_section = _command_device_section(keyboard, "keyboard", {"bindings", "mappings"})
        gamepad_section = _command_device_section(
            gamepad, "gamepad", {"bindings", "mappings", "deadzone"}
        )
        has_keyboard_mappings = keyboard_section is not None and "mappings" in keyboard_section
        has_gamepad_mappings = gamepad_section is not None and "mappings" in gamepad_section
        has_mappings = has_keyboard_mappings or has_gamepad_mappings
        # 事件映射自带值域：映射值就是合法值集，轴不占参数，rest 是静息码。
        if parameter_names and has_mappings:
            raise ValueError(
                "commands mappings own their value domain; declare no parameter axes "
                "alongside them"
            )
        if rest is not None and (parameter_names or not has_mappings):
            raise ValueError(
                "commands rest declares the mappings form's resting code; it takes no "
                "parameter axes and requires mappings"
            )
        mapped = []
        if has_keyboard_mappings:
            mapped.extend(_mapped_command_data(keyboard_section["mappings"], "keyboard"))
        if has_gamepad_mappings:
            mapped.extend(_mapped_command_data(gamepad_section["mappings"], "gamepad"))
        mapped_dim = None
        if mapped:
            widths = {len(entry["values"]) for entry in mapped}
            if len(widths) != 1:
                raise ValueError("commands mappings must share one width across their values")
            mapped_dim = widths.pop()
        if parameter_names:
            axes = [
                _command_axis(item, f"command axis {index}")
                for index, item in enumerate(parameter_names)
            ]
        elif mapped_dim is not None:
            axes = [(None, value) for value in _mapped_rest(rest, mapped_dim)]
        else:
            raise ValueError("commands requires parameter axes or device mappings")
        # 常量轴在 plan 里占一个空串名：参数名禁止空串，跨 Python/JS/JSON 形态唯一。
        names = [name if name is not None else "" for name, _constant in axes]
        constant_axes = [index for index, (_name, constant) in enumerate(axes) if constant is not None]
        has_constants = bool(constant_axes)
        if gamepad_section is not None and "deadzone" in gamepad_section and "bindings" not in gamepad_section:
            raise ValueError("commands gamepad deadzone requires bindings (mappings read no sticks)")
        # commandMode 由设备绑定推出：只有带轴绑定的设备算连续输入源（keyboard /
        # gamepad / keyboard-gamepad）；纯事件映射不改模式，向量保持 static。
        has_keyboard_axes = keyboard_section is not None and "bindings" in keyboard_section
        has_gamepad_axes = gamepad_section is not None and "bindings" in gamepad_section
        if has_keyboard_axes and has_gamepad_axes:
            input_mode = "keyboard-gamepad"
        elif has_keyboard_axes:
            input_mode = "keyboard"
        elif has_gamepad_axes:
            input_mode = "gamepad"
        else:
            input_mode = "static"
        bounds = []
        for name, constant in axes:
            if constant is not None:
                bounds.append((constant, constant))
                continue
            host_range = botlab_host.parameterRange(name)
            bounds.append(_host_value(host_range) if host_range is not None else None)
        data = {
            "dim": len(names),
            "parameterNames": names,
            "commandMode": input_mode,
        }
        if keyboard_section is not None and "bindings" in keyboard_section:
            data["commandBindings"] = _keyboard_bindings_data(
                keyboard_section["bindings"], names, constant_axes, bounds,
                "commands keyboard.bindings",
            )
        if gamepad_section is not None and "bindings" in gamepad_section:
            data["gamepadBindings"] = _gamepad_bindings_data(
                gamepad_section["bindings"], names, constant_axes, bounds,
                "commands gamepad.bindings",
            )
        if gamepad_section is not None and "deadzone" in gamepad_section:
            deadzone = gamepad_section["deadzone"]
            if isinstance(deadzone, bool) or not isinstance(deadzone, (int, float)) or not 0 <= deadzone < 1:
                raise ValueError("commands gamepad deadzone must be a number in [0, 1)")
            data["gamepadDeadzone"] = float(deadzone)
        if has_mappings:
            if waypoint is not None:
                raise ValueError("commands mappings cannot be combined with waypoint tracking")
            if magnitude is not None:
                raise ValueError("commands mappings cannot be combined with magnitude axes")
            data["commandEncoding"] = "mapped"
            data["commandMappings"] = mapped
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
        if has_constants:
            missing = [name for (name, _constant), bound in zip(axes, bounds) if bound is None]
            if missing:
                raise ValueError(
                    "commands constant axes need the named axes' parameter ranges to emit "
                    f"one bounds vector; missing ranges: {missing}"
                )
        if all(bound is not None for bound in bounds):
            data["cmdMin"] = [float(low) for low, _high in bounds]
            data["cmdMax"] = [float(high) for _low, high in bounds]
        if magnitude is not None:
            if isinstance(magnitude, str) or not isinstance(magnitude, (list, tuple)) or not magnitude:
                raise ValueError("commands magnitude must be a non-empty sequence of parameter names")
            magnitude_names = [
                _non_empty_string(name, f"commands magnitude[{index}]")
                for index, name in enumerate(magnitude)
            ]
            unknown = [name for name in magnitude_names if name not in names]
            if unknown:
                raise ValueError(
                    f"commands magnitude has unknown parameters {unknown}; the command axes are {names}"
                )
            if len(set(magnitude_names)) != len(magnitude_names):
                raise ValueError("commands magnitude has duplicate names")
            if input_mode == "static":
                raise ValueError(
                    "commands magnitude requires interactive input_mode "
                    "(keyboard, gamepad, or keyboard-gamepad)"
                )
            magnitude_set = set(magnitude_names)
            data["magnitude"] = [name in magnitude_set for name in names]
        if waypoint is not None:
            if not isinstance(waypoint, dict):
                raise TypeError("commands waypoint must come from waypoint_tracking()")
            if has_constants:
                raise ValueError("commands waypoint cannot combine with constant axes")
            data.update(waypoint)
        return Signal("obs-commands", "commands", data)


def _command_axis(item, label):
    """One command axis: a parameter name string, or a finite number pinning a
    constant axis (``commands("vx", 0, "yaw")`` -- the policy's command layout
    slot stays without a declared parameter)."""
    if isinstance(item, str):
        return _non_empty_string(item, label), None
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        value = _finite_number(item, label)
        return None, value
    raise ValueError(f"{label} must be a parameter name or a finite constant")


def _axis_list_for_error(axes):
    """Axis list for error messages: names as-is, constants as their value."""
    items = {name if name is not None else constant for name, constant in axes}
    return sorted(items, key=repr)


# 手柄按钮：Gamepad API standard mapping 的规范名（plan 契约层，与具体策略无关）。
_GAMEPAD_BUTTONS = frozenset({
    "A", "B", "X", "Y", "LB", "RB", "LT", "RT",
    "Back", "Start", "LStick", "RStick",
    "DPadUp", "DPadDown", "DPadLeft", "DPadRight",
})


def _command_device_section(section, device, allowed):
    """Validate one device binding table (``keyboard=`` / ``gamepad=``).

    A section is a pure binding table: exactly one of ``bindings`` (continuous
    axes) or ``mappings`` (discrete events); the command vector itself never
    appears here.
    """
    if section is None:
        return None
    if not isinstance(section, dict):
        raise TypeError(f"commands {device} must be a mapping (a config/{device}.yaml section)")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"commands {device} has unknown fields {sorted(unknown)}")
    declared = [field for field in ("bindings", "mappings") if field in section]
    if len(declared) != 1:
        raise ValueError(f"commands {device} requires exactly one of bindings or mappings")
    return section


def _resolve_command_axis(raw_axis, names, constant_axes, label):
    """Binding-table axis key: a declared parameter name or an axis index."""
    if isinstance(raw_axis, int) and not isinstance(raw_axis, bool):
        if raw_axis < 0 or raw_axis >= len(names):
            raise ValueError(
                f"{label} axis {raw_axis} is outside {len(names)} parameters"
            )
        if raw_axis in constant_axes:
            raise ValueError(f"{label} axis {raw_axis} is a constant axis and takes no bindings")
        return str(raw_axis)
    if isinstance(raw_axis, str) and raw_axis in names:
        return raw_axis
    raise ValueError(f"{label} keys must be declared parameter names or axis indices")


def _keyboard_bindings_data(bindings, names, constant_axes, bounds, label):
    """Keyboard binding table: ``{axis: {increase: KeyW, decrease: KeyS}}``."""
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError(f"{label} must be a non-empty mapping")
    if not all(bound is not None for bound in bounds):
        missing = [name for name, bound in zip(names, bounds) if bound is None]
        raise ValueError(f"{label} require parameter ranges for {missing}")
    normalized_bindings = {}
    used_keys = set()
    for raw_axis, binding in bindings.items():
        axis = _resolve_command_axis(raw_axis, names, constant_axes, label)
        if not isinstance(binding, dict) or not binding:
            raise ValueError(f"{label} {raw_axis!r} must be a non-empty mapping")
        unknown = set(binding) - {"increase", "decrease"}
        if unknown:
            raise ValueError(f"{label} {raw_axis!r} has unknown fields {sorted(unknown)}")
        normalized = {}
        for direction in ("increase", "decrease"):
            if direction not in binding:
                continue
            key = binding[direction]
            if key is None or key is False:
                normalized[direction] = False
            elif isinstance(key, str) and key:
                if key in used_keys:
                    raise ValueError(f'{label} key "{key}" is duplicated')
                used_keys.add(key)
                normalized[direction] = key
            else:
                raise ValueError(
                    f"{label} {raw_axis!r}.{direction} must be a non-empty string, None, or False"
                )
        if not normalized:
            raise ValueError(f"{label} {raw_axis!r} requires increase or decrease")
        if not any(key is not False for key in normalized.values()):
            raise ValueError(f"{label} {raw_axis!r} requires an enabled key")
        normalized_bindings[axis] = normalized
    return normalized_bindings


def _gamepad_bindings_data(bindings, names, constant_axes, bounds, label):
    """Gamepad binding table: ``{axis: {stick: L|R, axis: 0|1, invert: bool}}``.

    Each stick slot (stick, axis) drives at most one command axis -- two axes
    reading the same slot would fight over one physical input. Stick axes
    interpolate between the axis bounds, so the vector needs every parameter
    range (same requirement as keyboard bindings).
    """
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError(f"{label} must be a non-empty mapping")
    if not all(bound is not None for bound in bounds):
        missing = [name for name, bound in zip(names, bounds) if bound is None]
        raise ValueError(f"{label} require parameter ranges for {missing}")
    normalized_bindings = {}
    used_slots = {}
    for raw_axis, binding in bindings.items():
        axis = _resolve_command_axis(raw_axis, names, constant_axes, label)
        if not isinstance(binding, dict) or not binding:
            raise ValueError(f"{label} {raw_axis!r} must be a non-empty mapping")
        unknown = set(binding) - {"stick", "axis", "invert"}
        if unknown:
            raise ValueError(f"{label} {raw_axis!r} has unknown fields {sorted(unknown)}")
        stick = binding.get("stick")
        stick_axis = binding.get("axis")
        if stick not in ("L", "R"):
            raise ValueError(f"{label} {raw_axis!r}.stick must be L or R")
        if isinstance(stick_axis, bool) or not isinstance(stick_axis, int) or stick_axis not in (0, 1):
            raise ValueError(f"{label} {raw_axis!r}.axis must be 0 or 1")
        slot = (stick, stick_axis)
        if slot in used_slots:
            raise ValueError(
                f"{label} stick {stick}{stick_axis} is already bound to "
                f"{used_slots[slot]!r}"
            )
        used_slots[slot] = raw_axis
        invert = binding.get("invert", True)
        if not isinstance(invert, bool):
            raise ValueError(f"{label} {raw_axis!r}.invert must be a bool")
        normalized_bindings[axis] = {"stick": stick, "axis": stick_axis, "invert": invert}
    return normalized_bindings


def _mapped_command_data(mappings, device):
    """Event mappings: keyboard chords or gamepad buttons carrying the command's value set."""
    if not isinstance(mappings, (list, tuple)) or not mappings:
        raise ValueError(f"commands {device}.mappings must be a non-empty sequence")
    allowed = {"key", "keys"} if device == "keyboard" else {"button"}
    normalized_mappings = []
    used_triggers = set()
    for index, mapping in enumerate(mappings):
        label = f"commands {device}.mappings[{index}]"
        if not isinstance(mapping, dict):
            raise TypeError(f"{label} must be a mapping")
        unknown = set(mapping) - (allowed | {"value", "values", "label"})
        if unknown:
            raise ValueError(f"{label} has unknown fields {sorted(unknown)}")
        if device == "keyboard":
            if ("key" in mapping) == ("keys" in mapping):
                raise ValueError(f"{label} requires exactly one of key or keys")
            if "key" in mapping:
                key = _non_empty_string(mapping["key"], f"{label} key")
                trigger_data = {"key": key}
                trigger = (key,)
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
                trigger_data = {"keys": list(chord)}
                trigger = chord
            trigger_key = tuple(sorted(trigger))
        else:
            button = _non_empty_string(mapping.get("button"), f"{label} button")
            if button not in _GAMEPAD_BUTTONS:
                raise ValueError(
                    f"{label} button must be one of {sorted(_GAMEPAD_BUTTONS)}"
                )
            trigger_data = {"button": button}
            trigger_key = (button,)
        if trigger_key in used_triggers:
            raise ValueError(f"{label} duplicates an earlier trigger")
        used_triggers.add(trigger_key)

        if ("value" in mapping) == ("values" in mapping):
            raise ValueError(f"{label} requires exactly one of value or values")
        if "value" in mapping:
            values = _finite_values([mapping["value"]], f"{label} value")
        else:
            values = _finite_values(mapping["values"], f"{label} values")
        if not values:
            raise ValueError(f"{label} values must not be empty")
        normalized = {**trigger_data, "values": values}
        if "label" in mapping:
            normalized["label"] = _localized_text(mapping["label"], f"{label} label")
        normalized_mappings.append(normalized)
    return normalized_mappings


def _mapped_rest(rest, dim):
    """静息码：None 时全 0；标量只许一维；序列必须与命令同宽。"""
    if rest is None:
        return [0.0] * dim
    if isinstance(rest, (int, float)) and not isinstance(rest, bool):
        if dim != 1:
            raise ValueError("commands rest scalar requires a one-dimensional command")
        return [_finite_number(rest, "commands rest")]
    if isinstance(rest, (list, tuple)):
        return require_vector(rest, dim, "commands rest")
    raise ValueError("commands rest must be a number or a sequence of numbers")


def _with_imu_sensors(data, imu_sensors):
    """Write `imuSensors` only when given; an omitted field keeps the runtime's legacy rule."""
    if imu_sensors is not None:
        if not isinstance(imu_sensors, bool):
            raise TypeError("imu_sensors must be a bool")
        data["imuSensors"] = imu_sensors
    return data


def _observation_data(data, *, order=None, default=None, scale=None):
    if order is not None:
        data["indices"] = order
    if default is not None:
        data["defaultAngles"] = default
    if scale is not None:
        data["scale"] = scale
    return data
