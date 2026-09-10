import inspect
import math

from . import _host as botlab_host

try:
    from pyodide.ffi import create_proxy
except ImportError:
    # The static CPython host (botlab CLI) never wires JS callbacks; it only
    # collects declarations and the compiled plan, so proxies stay unused there.
    create_proxy = None


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_values(values, label):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite numeric sequence") from error
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def require_vector(values, size, label="vector"):
    size = _positive_integer(size, f"{label} size")
    vector = _finite_values(values, label)
    if len(vector) != size:
        raise ValueError(f"{label} must contain {size} values, got {len(vector)}")
    return vector


def _expand_vector(values, size, label):
    if isinstance(values, bool):
        raise ValueError(f"{label} must be a finite number or numeric sequence")
    if isinstance(values, (int, float)):
        value = float(values)
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return [value] * size
    return require_vector(values, size, label)


def clip(values, lower, upper):
    vector = _finite_values(values, "clip values")
    if not vector:
        raise ValueError("clip values must not be empty")
    lower_values = _expand_vector(lower, len(vector), "clip lower")
    upper_values = _expand_vector(upper, len(vector), "clip upper")
    if any(low > high for low, high in zip(lower_values, upper_values)):
        raise ValueError("clip lower values must not exceed upper values")
    return [
        min(max(value, low), high)
        for value, low, high in zip(vector, lower_values, upper_values)
    ]


def reorder(values, indices):
    vector = _finite_values(values, "reorder values")
    output = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("reorder indices must be integers")
        if index < 0 or index >= len(vector):
            raise ValueError(
                f"reorder index {index} is outside a vector of length {len(vector)}"
            )
        output.append(vector[index])
    if not output:
        raise ValueError("reorder indices must not be empty")
    return output


def affine(values, scale, offset):
    vector = _finite_values(values, "affine values")
    if not vector:
        raise ValueError("affine values must not be empty")
    scales = _expand_vector(scale, len(vector), "affine scale")
    offsets = _expand_vector(offset, len(vector), "affine offset")
    return [
        value * scale_value + offset_value
        for value, scale_value, offset_value in zip(vector, scales, offsets)
    ]


def quaternion_yaw(quaternion):
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    return math.atan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy * qy + qz * qz),
    )


def inverse_rotate(vector, quaternion):
    vx, vy, vz = require_vector(vector, 3, "vector")
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        (1 - 2 * (yy + zz)) * vx + 2 * (xy + wz) * vy + 2 * (xz - wy) * vz,
        2 * (xy - wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz + wx) * vz,
        2 * (xz + wy) * vx + 2 * (yz - wx) * vy + (1 - 2 * (xx + yy)) * vz,
    ]


def projected_gravity(quaternion):
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    return [
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ]


def _non_empty_string(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _plan_value(value, label):
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _plan_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            _non_empty_string(key, f"{label} key")
            output[key] = _plan_value(item, f"{label}.{key}")
        return output
    raise TypeError(f"{label} must be JSON-serializable")


# ---------------------------------------------------------------------------
# Resource declarations
#
# A project declares a resource by constructing it, and the handle it gets back
# *is* the reference: there is no string id to keep in sync across files. The
# source path doubles as the resource id -- the convention `models:` already
# used -- so a project only names a resource when two of them share a source.
# Projects that still declare resources in `botlab.yaml` reach the same handles
# through `sim.resources`, which binds to manifest entries instead of declaring.
# ---------------------------------------------------------------------------


def _resource_source(value, label):
    """A source is a path, or another resource the new one attaches to."""
    inner = getattr(value, "id", value)
    return _non_empty_string(inner, label)


def _declare_resource(
    kind,
    source,
    *,
    preset=None,
    config=None,
    name=None,
    provider=None,
):
    source_path = _resource_source(source, f"{kind} source")
    resource_id = source_path if name is None else _non_empty_string(name, f"{kind} name")
    resource_config = {} if config is None else config
    if not isinstance(resource_config, dict):
        raise TypeError(f"{kind} config must be a mapping")
    botlab_host.declareResource(
        kind,
        resource_id,
        source_path,
        None if preset is None else _non_empty_string(preset, f"{kind} preset"),
        None if provider is None else _non_empty_string(provider, f"{kind} provider"),
        _plan_value(resource_config, f"{kind} config"),
    )
    return resource_id


def project(name, *, control_hz=None, physics_hz=None, parameters=None):
    """Name the project and override the runtime defaults.

    Optional: a project that keeps the documented defaults (50 Hz control,
    200 Hz physics) and declares no tunable parameters never has to call this.
    """
    botlab_host.declareProject(
        _non_empty_string(name, "project name"),
        None if control_hz is None else _positive_integer(control_hz, "project control_hz"),
        None if physics_hz is None else _positive_integer(physics_hz, "project physics_hz"),
        _plan_value({} if parameters is None else parameters, "project parameters"),
    )


def resource_config(resource, field):
    return {
        "$botlab": "resource-config",
        "resource": _non_empty_string(resource, "resource_config resource"),
        "field": _non_empty_string(field, "resource_config field"),
    }


def default_joint_positions(resource):
    return {
        "$botlab": "default-joint-positions",
        "resource": _non_empty_string(resource, "default_joint_positions resource"),
    }


def _policy_pd_vector(value, size, label):
    if isinstance(value, dict) and value.get("$botlab") == "resource-config":
        return value
    return _expand_vector(value, size, label)


# ---------------------------------------------------------------------------
# Signals
#
# A controller is an expression: observations flow into a policy, the policy
# action flows into an output stage. Every call below returns a Signal, and
# passing a Signal into another call *is* the connection -- there are no node
# ids, ports, or edges to write down. PolicyController lowers the expression
# into the plan BotLab's TypeScript engine compiles once and runs per frame.
# ---------------------------------------------------------------------------

_HISTORY_LAYOUTS = {"by-frame": "isaacgym", "by-term": "isaaclab"}
_HISTORY_ORDERS = ("oldest-first", "newest-first")
_RESIZE_LAYOUTS = ("hwc", "chw")


def _require_signal(value, label):
    if not isinstance(value, Signal):
        raise TypeError(f"{label} must be a BotLab Signal")
    return value


class Signal:
    """One value in a controller expression, produced from zero or more inputs.

    Signals behave like the vectors they carry: ``scale * signal + offset`` is an
    affine map, ``signal[indices]`` reorders, and ``signal.clip(low, high)``
    limits. Passing a Signal into another call is what connects them.
    """

    __slots__ = (
        "_type",
        "_label",
        "_data",
        "_inputs",
        "_node_id",
        "_modify",
        "_node",
        "_output_slot",
    )

    def __init__(self, node_type, label, data, inputs=(), *, node_id=None, modify=None):
        self._type = _non_empty_string(node_type, "Signal type")
        self._label = _non_empty_string(label, "Signal label")
        self._data = data
        self._inputs = tuple(inputs)
        self._node_id = node_id
        self._modify = modify
        self._node = self
        self._output_slot = 0

    def __repr__(self):
        return f"<Signal {self._label} ({self._type})>"

    def __bool__(self):
        raise TypeError(
            "BotLab Signal values change at runtime and cannot be used in Python "
            "if/and/or/not. Use where(condition, when_true, when_false), & / | / ~, "
            "or a custom Controller.step() for arbitrary Python control flow."
        )

    def clip(self, low, high):
        return Signal(
            "proc-clamp",
            f"{self._label}-clip",
            {"min": low, "max": high},
            (self,),
        )

    def __getitem__(self, indices):
        """Gather values by index, the way ``actions[motor_order]`` reads in NumPy."""
        return take(self, indices)

    def at(self, index):
        """Select one vector element as a scalar Signal for comparisons."""
        return take(self, [_non_negative_integer(index, "Signal.at index")])

    def compare(self, operator, other):
        """Build a scalar runtime comparison against a number or scalar Signal."""
        if operator not in ("gt", "gte", "lt", "lte", "eq", "ne"):
            raise ValueError(
                "Signal.compare operator must be one of gt, gte, lt, lte, eq, ne"
            )
        if isinstance(other, Signal):
            return Signal(
                "proc-compare",
                f"{self._label}-{operator}",
                {"operator": operator},
                (self, other),
            )
        return Signal(
            "proc-compare",
            f"{self._label}-{operator}",
            {
                "operator": operator,
                "value": _finite_number(other, "Signal comparison value"),
            },
            (self,),
        )

    def gt(self, other):
        return self.compare("gt", other)

    def gte(self, other):
        return self.compare("gte", other)

    def lt(self, other):
        return self.compare("lt", other)

    def lte(self, other):
        return self.compare("lte", other)

    def eq(self, other):
        return self.compare("eq", other)

    def ne(self, other):
        return self.compare("ne", other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __and__(self, other):
        return all_of(self, other)

    def __or__(self, other):
        return any_of(self, other)

    def __invert__(self):
        return logical_not(self)

    def action(self, *, modify=None):
        """Mark this signal as the controller action.

        The marked value is fed back to ``robot.last_action()`` on the next frame
        and is the one per-frame Python extension point: ``modify(action, frame)``
        runs between the clipped policy-order action and the output stage.
        """
        if modify is not None and not callable(modify):
            raise TypeError("Signal.action modify must be callable or None")
        return Signal("proc-action-hook", "action", {}, (self,), modify=modify)

    def _affine(self, scale, offset):
        return Signal(
            "proc-math",
            f"{self._label}-affine",
            {"operation": "affine", "scale": scale, "offset": offset},
            (self,),
        )

    def __mul__(self, scale):
        return self._affine(scale, 0.0)

    __rmul__ = __mul__

    def __add__(self, offset):
        # `scale * signal + offset` is one affine node, not two: fold the offset
        # into the scale-only node this signal already is.
        if self._type == "proc-math" and self._data["offset"] == 0.0:
            return Signal(
                "proc-math",
                self._label,
                {"operation": "affine", "scale": self._data["scale"], "offset": offset},
                self._inputs,
            )
        return self._affine(1.0, offset)

    __radd__ = __add__

    def _output(self, slot, label):
        """Return another output of this node without duplicating the node."""
        slot = _non_negative_integer(slot, "Signal output slot")
        if slot == 0:
            return self
        output = Signal.__new__(Signal)
        output._type = self._type
        output._label = _non_empty_string(label, "Signal output label")
        output._data = self._data
        output._inputs = self._inputs
        output._node_id = self._node_id
        output._modify = self._modify
        output._node = self._node
        output._output_slot = slot
        return output


def concat(signals):
    """Concatenate signals into one vector, in the order given."""
    items = list(signals)
    if not items:
        raise ValueError("concat requires at least one signal")
    for index, signal in enumerate(items):
        _require_signal(signal, f"concat input {index}")
    if len(items) == 1:
        return items[0]
    return Signal("proc-mux", "concat", {"channels": len(items)}, items)


def take(source, indices):
    """Gather a vector signal by explicit non-negative indices."""
    source = _require_signal(source, "take source")
    if isinstance(indices, dict) and indices.get("$botlab") == "resource-config":
        return Signal(
            "proc-reorder",
            f"{source._label}-reordered",
            {"indices": indices},
            (source,),
        )
    if isinstance(indices, (int, slice, str)) or isinstance(indices, bool):
        raise TypeError(
            "take indices must be a sequence, such as signal[robot.motor_order]; "
            "single elements and slices are not signals"
        )
    selected = list(indices)
    if not selected:
        raise ValueError("take indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected):
        raise ValueError("take indices must contain non-negative integers")
    return Signal(
        "proc-reorder",
        f"{source._label}-reordered",
        {"indices": selected},
        (source,),
    )


def split(source, sizes):
    """Split one vector signal into ordered or named fixed-size outputs."""
    source = _require_signal(source, "split source")
    names = None
    if isinstance(sizes, dict):
        if not sizes:
            raise ValueError("split sizes must not be empty")
        names = [_non_empty_string(name, "split output name") for name in sizes]
        values = list(sizes.values())
    else:
        values = list(sizes)
    if not values:
        raise ValueError("split sizes must not be empty")
    normalized = [
        _positive_integer(size, f"split size {index}")
        for index, size in enumerate(values)
    ]
    node = Signal("proc-demux", f"{source._label}-split", {"sizes": normalized}, (source,))
    outputs = [node._output(index, f"{source._label}-part-{index}") for index in range(len(normalized))]
    if names is not None:
        return dict(zip(names, outputs))
    return tuple(outputs)


def _as_signal(source, label):
    """Accept a Signal, or a list of Signals to concatenate, anywhere one is used."""
    if isinstance(source, Signal):
        return source
    if isinstance(source, (list, tuple)):
        return concat(source)
    raise TypeError(f"{label} must be a Signal or a list of Signals")


def _logical_signal(operator, conditions):
    items = list(conditions)
    if not items:
        raise ValueError(f"{operator} requires at least one condition")
    for index, condition in enumerate(items):
        _require_signal(condition, f"{operator} condition {index}")
    if len(items) == 1 and operator != "not":
        return items[0]
    return Signal(
        "proc-logic",
        operator,
        {"operator": operator},
        items,
    )


def all_of(*conditions):
    """Return a scalar Signal that is true when every condition is non-zero."""
    return _logical_signal("all", conditions)


def any_of(*conditions):
    """Return a scalar Signal that is true when any condition is non-zero."""
    return _logical_signal("any", conditions)


def logical_not(condition):
    """Invert one scalar runtime condition."""
    return _logical_signal("not", (condition,))


def where(condition, when_true, when_false):
    """Choose between same-sized Signals from a scalar runtime condition.

    Both branches are part of the compiled graph. Use ``policy_bank`` instead when
    selecting between expensive ONNX policies so only the active policy is inferred.
    """
    condition = _require_signal(condition, "where condition")
    when_true = _as_signal(when_true, "where when_true")
    when_false = _as_signal(when_false, "where when_false")
    return Signal(
        "proc-where",
        "where",
        {},
        (condition, when_true, when_false),
    )


def history(
    source,
    size,
    *,
    stride=1,
    order="oldest-first",
    layout="by-frame",
    fill=None,
):
    """Stack the last `size` samples of `source`, taken every `stride` control steps.

    `source` is a Signal, or a list of Signals to concatenate first.

    `layout` says how the stacked values are laid out, and must match how the
    policy was trained: `"by-frame"` keeps each frame's values together, while
    `"by-term"` runs each concatenated term through time before the next term.
    """
    signal = _as_signal(source, "history source")
    size = _positive_integer(size, "history size")
    stride = _positive_integer(stride, "history stride")
    if order not in _HISTORY_ORDERS:
        raise ValueError(f"history order must be one of {_HISTORY_ORDERS}")
    if layout not in _HISTORY_LAYOUTS:
        raise ValueError(f"history layout must be one of {tuple(_HISTORY_LAYOUTS)}")

    data = {
        "stackSize": size,
        "order": order,
        "stackMode": _HISTORY_LAYOUTS[layout],
    }
    if fill is not None:
        if isinstance(fill, (int, float)) and not isinstance(fill, bool):
            fill = float(fill)
            if not math.isfinite(fill):
                raise ValueError("history fill must be finite")
        else:
            fill = _finite_values(fill, "history fill")
            if not fill:
                raise ValueError("history fill must not be empty")
        data["initialFillValue"] = fill
    if stride > 1:
        offsets = [-1 - stride * slot for slot in range(size - 1, -1, -1)]
        if order == "newest-first":
            offsets.reverse()
        data["sourceBufferSize"] = stride * (size - 1) + 2
        data["sampleOffsets"] = offsets
    return Signal("proc-history", f"{signal._label}-history", data, (signal,))


def _size_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a (width, height) pair")
    return (
        _positive_integer(value[0], f"{label} width"),
        _positive_integer(value[1], f"{label} height"),
    )


def constant(values, *, size=None):
    """A fixed vector, for policy inputs the simulation does not produce."""
    if (
        isinstance(values, dict)
        and values.get("$botlab") in ("resource-config", "default-joint-positions")
    ):
        if size is None:
            raise ValueError("constant resource values require an explicit size")
        size = _positive_integer(size, "constant size")
        return Signal("obs-constant", "constant", {"dim": size, "values": values})
    if size is None:
        vector = _finite_values(values, "constant values")
    else:
        size = _positive_integer(size, "constant size")
        if isinstance(values, bool) or not isinstance(values, (int, float)):
            raise ValueError("constant with size requires one finite numeric value")
        value = float(values)
        if not math.isfinite(value):
            raise ValueError("constant value must be finite")
        vector = [value] * size
    if not vector:
        raise ValueError("constant values must not be empty")
    return Signal("obs-constant", "constant", {"dim": len(vector), "values": vector})


def resize(source, *, input_size, output_size, channels=None, layout="hwc", flip_x=False, flip_y=False):
    """Resample an image signal, with `input_size` and `output_size` given as `(width, height)`.

    `channels` is inferred from the input width when omitted.
    """
    signal = _require_signal(source, "resize source")
    if layout not in _RESIZE_LAYOUTS:
        raise ValueError(f"resize layout must be one of {_RESIZE_LAYOUTS}")
    input_width, input_height = _size_pair(input_size, "resize input_size")
    output_width, output_height = _size_pair(output_size, "resize output_size")
    data = {
        "inputWidth": input_width,
        "inputHeight": input_height,
        "outputWidth": output_width,
        "outputHeight": output_height,
        "layout": layout,
        "flipX": bool(flip_x),
        "flipY": bool(flip_y),
    }
    if channels is not None:
        data["channels"] = _positive_integer(channels, "resize channels")
    return Signal("proc-resize", f"{signal._label}-resize", data, (signal,))


def mode_switch(command, *, initial, transitions):
    """Turn discrete command pulses into a persistent controller mode."""
    command = _require_signal(command, "mode_switch command")
    if isinstance(initial, bool) or not isinstance(initial, int):
        raise ValueError("mode_switch initial must be an integer")
    if isinstance(transitions, dict):
        items = [
            {"from": "any", "command": command_code, "to": target_mode}
            for command_code, target_mode in transitions.items()
        ]
    else:
        items = list(transitions)
    if not items:
        raise ValueError("mode_switch transitions must not be empty")
    normalized = []
    for index, transition in enumerate(items):
        if not isinstance(transition, dict):
            raise TypeError(f"mode_switch transition {index} must be a mapping")
        unknown = set(transition) - {"from", "command", "to"}
        if unknown:
            raise ValueError(
                f"mode_switch transition {index} has unknown fields {sorted(unknown)}"
            )
        if "command" not in transition or "to" not in transition:
            raise ValueError(
                f"mode_switch transition {index} requires command and to"
            )
        command_code = transition["command"]
        to_mode = transition["to"]
        if (
            isinstance(command_code, bool)
            or not isinstance(command_code, int)
            or isinstance(to_mode, bool)
            or not isinstance(to_mode, int)
        ):
            raise ValueError(
                f"mode_switch transition {index} command and to must be integers"
            )
        from_modes = transition.get("from", "any")
        if from_modes != "any":
            if isinstance(from_modes, int) and not isinstance(from_modes, bool):
                from_modes = [from_modes]
            else:
                try:
                    from_modes = list(from_modes)
                except TypeError as error:
                    raise ValueError(
                        f"mode_switch transition {index} from must be an integer, "
                        'integer sequence, or "any"'
                    ) from error
                if not from_modes or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in from_modes
                ):
                    raise ValueError(
                        f"mode_switch transition {index} from must contain integers"
                    )
        normalized.append({"from": from_modes, "command": command_code, "to": to_mode})
    return Signal(
        "proc-mode-switch",
        "mode-switch",
        {"initialMode": initial, "transitions": normalized},
        (command,),
    )


class _StateMachine:
    __slots__ = ("state", "steps")

    def __init__(self, signal):
        self.state = signal
        self.steps = signal._output(1, "state-steps")


def state_machine(command, *, initial, transitions):
    """Build a persistent finite-state machine in the native control path.

    Each transition has from and to plus at least one trigger: command for an
    edge-triggered command pulse, after_steps for time in the current state, or
    when for a scalar runtime condition. Triggers on one transition are
    conjunctive. reset_steps=False preserves elapsed steps across a sub-state
    transition.
    """
    command = _require_signal(command, "state_machine command")
    if isinstance(initial, bool) or not isinstance(initial, int):
        raise ValueError("state_machine initial must be an integer")
    items = list(transitions)
    if not items:
        raise ValueError("state_machine transitions must not be empty")
    normalized = []
    conditions = []
    for index, transition in enumerate(items):
        if not isinstance(transition, dict):
            raise TypeError(f"state_machine transition {index} must be a mapping")
        unknown = set(transition) - {
            "from", "to", "command", "after_steps", "when", "reset_steps"
        }
        if unknown:
            raise ValueError(
                f"state_machine transition {index} has unknown fields {sorted(unknown)}"
            )
        if "to" not in transition:
            raise ValueError(f"state_machine transition {index} requires to")
        if not any(name in transition for name in ("command", "after_steps", "when")):
            raise ValueError(
                f"state_machine transition {index} requires command, after_steps, or when"
            )
        to_state = transition["to"]
        if isinstance(to_state, bool) or not isinstance(to_state, int):
            raise ValueError(f"state_machine transition {index} to must be an integer")
        from_states = transition.get("from", "any")
        if from_states != "any":
            if isinstance(from_states, int) and not isinstance(from_states, bool):
                from_states = [from_states]
            else:
                try:
                    from_states = list(from_states)
                except TypeError as error:
                    raise ValueError(
                        f"state_machine transition {index} from must be an integer, "
                        'integer sequence, or "any"'
                    ) from error
                if not from_states or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in from_states
                ):
                    raise ValueError(
                        f"state_machine transition {index} from must contain integers"
                    )
        item = {"from": from_states, "to": to_state}
        if "command" in transition:
            command_code = transition["command"]
            if (
                isinstance(command_code, bool)
                or not isinstance(command_code, int)
                or command_code == 0
            ):
                raise ValueError(
                    f"state_machine transition {index} command must be a non-zero integer"
                )
            item["command"] = command_code
        if "after_steps" in transition:
            item["afterSteps"] = _non_negative_integer(
                transition["after_steps"],
                f"state_machine transition {index} after_steps",
            )
        if "when" in transition:
            condition = _require_signal(
                transition["when"], f"state_machine transition {index} when"
            )
            conditions.append(condition)
            item["conditionSlot"] = len(conditions)
        reset_steps = transition.get("reset_steps", True)
        if not isinstance(reset_steps, bool):
            raise TypeError(
                f"state_machine transition {index} reset_steps must be boolean"
            )
        if not reset_steps:
            item["resetSteps"] = False
        normalized.append(item)
    signal = Signal(
        "proc-state-machine",
        "state-machine",
        {"initialState": initial, "transitions": normalized},
        (command, *conditions),
    )
    return _StateMachine(signal)


def phase_clock(steps, *, period_steps):
    """Return cos/sin phase encoding for a scalar step signal."""
    steps = _require_signal(steps, "phase_clock steps")
    return Signal(
        "proc-phase-clock",
        "phase-clock",
        {"periodSteps": _positive_integer(period_steps, "phase_clock period_steps")},
        (steps,),
    )


def mode_select(mode, hold, choices, *, passive=0):
    """Choose a same-sized signal by mode, retaining ``hold`` for unknown modes."""
    mode = _require_signal(mode, "mode_select mode")
    hold = _require_signal(hold, "mode_select hold")
    if isinstance(passive, bool) or not isinstance(passive, int):
        raise ValueError("mode_select passive must be an integer")
    if not isinstance(choices, dict) or not choices:
        raise ValueError("mode_select choices must be a non-empty mapping")
    slot_modes = []
    inputs = [mode, hold]
    for selected, source in choices.items():
        if isinstance(selected, int) and not isinstance(selected, bool):
            selected_modes = [selected]
        elif isinstance(selected, (tuple, frozenset)):
            selected_modes = list(selected)
        else:
            raise ValueError(
                "mode_select choice keys must be integers or tuples/frozensets of integers"
            )
        if not selected_modes or any(
            isinstance(selected_mode, bool) or not isinstance(selected_mode, int)
            for selected_mode in selected_modes
        ):
            raise ValueError("mode_select choice modes must be integers")
        signal = _require_signal(source, f"mode_select choice {selected}")
        for selected_mode in selected_modes:
            if selected_mode in slot_modes:
                raise ValueError(f"mode_select mode {selected_mode} is duplicated")
            slot_modes.append(selected_mode)
            inputs.append(signal)
    return Signal(
        "proc-mode-select",
        "mode-select",
        {"passiveMode": passive, "slotModes": slot_modes},
        inputs,
    )


def one_hot_index(source, *, threshold=0.0, default=0):
    """Return the index of the largest value above ``threshold`` as a scalar signal."""
    source = _require_signal(source, "one_hot_index source")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("one_hot_index threshold must be finite")
    if isinstance(default, bool) or not isinstance(default, int):
        raise ValueError("one_hot_index default must be an integer")
    return Signal(
        "proc-onehot-index",
        "one-hot-index",
        {"threshold": threshold, "defaultIndex": default},
        (source,),
    )


def delay(source, total_steps, *, age=None, fill=0.0):
    """Delay a signal by a fixed total latency, optionally compensating sensor age."""
    source = _require_signal(source, "delay source")
    total_steps = _non_negative_integer(total_steps, "delay total_steps")
    if isinstance(fill, (int, float)) and not isinstance(fill, bool):
        fill = float(fill)
        if not math.isfinite(fill):
            raise ValueError("delay fill must be finite")
    else:
        fill = _finite_values(fill, "delay fill")
        if not fill:
            raise ValueError("delay fill must not be empty")
    inputs = [source]
    if age is not None:
        inputs.append(_require_signal(age, "delay age"))
    return Signal(
        "proc-delay",
        f"{source._label}-delay",
        {"totalSteps": total_steps, "fill": fill},
        inputs,
    )


_RULE_OPERATORS = ("gt", "gte", "lt", "lte", "eq", "ne")


def _rule_condition(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    branch_keys = [key for key in ("all", "any") if key in value]
    if branch_keys:
        if len(branch_keys) != 1 or set(value) != {branch_keys[0]}:
            raise ValueError(f'{label} must contain exactly one of "all" or "any"')
        key = branch_keys[0]
        conditions = value[key]
        if not isinstance(conditions, (list, tuple)) or not conditions:
            raise ValueError(f'{label}.{key} must be a non-empty list')
        return {
            key: [
                _rule_condition(condition, f"{label}.{key}[{index}]")
                for index, condition in enumerate(conditions)
            ]
        }
    if set(value) != {"index", "op", "value"}:
        raise ValueError(
            f'{label} leaf must contain exactly "index", "op", and "value"'
        )
    index = _non_negative_integer(value["index"], f"{label}.index")
    operator = value["op"]
    if operator not in _RULE_OPERATORS:
        raise ValueError(f"{label}.op must be one of {_RULE_OPERATORS}")
    comparison = value["value"]
    if (
        isinstance(comparison, bool)
        or not isinstance(comparison, (int, float))
        or not math.isfinite(comparison)
    ):
        raise ValueError(f"{label}.value must be a finite number")
    return {"index": index, "op": operator, "value": float(comparison)}


def rule_index(source, rules, *, default=0):
    """Return the value of the first recursively matched numeric rule."""
    source = _require_signal(source, "rule_index source")
    if isinstance(default, bool) or not isinstance(default, int):
        raise ValueError("rule_index default must be an integer")
    items = list(rules)
    if not items:
        raise ValueError("rule_index rules must not be empty")
    normalized = []
    for index, rule in enumerate(items):
        label = f"rule_index rule {index}"
        if not isinstance(rule, dict):
            raise TypeError(f"{label} must be a mapping")
        if set(rule) != {"value", "when"}:
            raise ValueError(f'{label} must contain exactly "value" and "when"')
        output = rule["value"]
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError(f"{label}.value must be an integer")
        normalized.append({
            "value": output,
            "when": _rule_condition(rule["when"], f"{label}.when"),
        })
    return Signal(
        "proc-rule-index",
        "rule-index",
        {"rules": normalized, "defaultValue": default},
        (source,),
    )


def one_hot(source, size):
    """Convert a scalar integer-index signal into a fixed-size one-hot vector."""
    source = _require_signal(source, "one_hot source")
    return Signal(
        "proc-one-hot",
        "one-hot",
        {"size": _positive_integer(size, "one_hot size")},
        (source,),
    )


def _motion_source_data(motion):
    if not isinstance(motion, Motion):
        raise TypeError("motion_reference motion must be a BotLab Motion")
    source = botlab_host.resourceSource(motion.id)
    source = _non_empty_string(str(source), "motion resource source")
    if source.startswith("registry://"):
        prefix = source[len("registry://"):]
        return {"motionSource": {"kind": "registry", "registryPrefix": prefix}}
    return {"motionPath": source}


def motion_reference(
    motion,
    *,
    mode=None,
    command=None,
    command_codes=None,
    active_mode=None,
    start=0,
    end=None,
    pause=0,
    frame_interval=1,
    anchor_link=0,
    loop=False,
):
    """Read the current frame from a motion resource in the engine hot path."""
    if command is not None and mode is None:
        raise ValueError("motion_reference command requires mode")
    if command is None and command_codes is not None:
        raise ValueError("motion_reference command_codes requires command")
    inputs = []
    if mode is not None:
        inputs.append(_require_signal(mode, "motion_reference mode"))
    if command is not None:
        inputs.append(_require_signal(command, "motion_reference command"))
        required_codes = {"pauseConfigured", "pauseCurrent", "resume"}
        if not isinstance(command_codes, dict) or set(command_codes) != required_codes:
            raise ValueError(
                "motion_reference command_codes must contain exactly "
                '"pauseConfigured", "pauseCurrent", and "resume"'
            )
        normalized_codes = {}
        for name in ("pauseConfigured", "pauseCurrent", "resume"):
            value = command_codes[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"motion_reference command_codes.{name} must be an integer")
            normalized_codes[name] = value
        if len(set(normalized_codes.values())) != len(normalized_codes):
            raise ValueError("motion_reference command_codes values must be distinct")
    data = _motion_source_data(motion)
    data.update({
        "startIndex": _non_negative_integer(start, "motion_reference start"),
        "pauseIndex": _non_negative_integer(pause, "motion_reference pause"),
        "frameInterval": _positive_integer(
            frame_interval,
            "motion_reference frame_interval",
        ),
        "anchorLinkIndex": _non_negative_integer(
            anchor_link,
            "motion_reference anchor_link",
        ),
        "loop": bool(loop),
    })
    if end is not None:
        data["endIndex"] = _non_negative_integer(end, "motion_reference end")
    if active_mode is not None:
        if isinstance(active_mode, bool) or not isinstance(active_mode, int):
            raise ValueError("motion_reference active_mode must be an integer")
        data["activeMode"] = active_mode
    if command is not None:
        data["commandCodes"] = normalized_codes
    return Signal("proc-motion-reference", "motion-reference", data, inputs)


_POLICY_FIELD_KINDS = ("copy", "mapped", "previous-action")


def _policy_key(value, label):
    return _non_empty_string(value, label)


def _policy_mode(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def policy_inputs(mode, inputs, policies):
    """Build the distinct input tensor for every policy in a policy bank.

    ``inputs`` names shared source signals. Policy fields refer to those names
    with ``source`` instead of exposing engine input-slot numbers.
    """
    mode = _require_signal(mode, "policy_inputs mode")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("policy_inputs inputs must be a non-empty mapping")
    source_names = []
    source_signals = []
    for name, source in inputs.items():
        name = _policy_key(name, "policy_inputs source name")
        if name in source_names:
            raise ValueError(f'policy_inputs source "{name}" is duplicated')
        source_names.append(name)
        source_signals.append(_require_signal(source, f'policy_inputs source "{name}"'))
    source_slots = {name: index + 1 for index, name in enumerate(source_names)}

    items = list(policies)
    if not items:
        raise ValueError("policy_inputs policies must not be empty")
    normalized = []
    keys = []
    for policy_index, policy in enumerate(items):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_inputs policy {policy_index} must be a mapping")
        unknown = set(policy) - {
            "key",
            "mode",
            "reference",
            "history_frames",
            "state_dim",
            "fields",
        }
        if unknown:
            raise ValueError(
                f"policy_inputs policy {policy_index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_inputs policy {policy_index} key")
        if key in keys:
            raise ValueError(f'policy_inputs policy key "{key}" is duplicated')
        keys.append(key)
        fields = policy.get("fields")
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError(f'policy_inputs policy "{key}" fields must not be empty')
        normalized_fields = []
        for field_index, field in enumerate(fields):
            if isinstance(field, str):
                field = {"source": field}
            if not isinstance(field, dict):
                raise TypeError(
                    f'policy_inputs policy "{key}" field {field_index} '
                    "must be a source name or mapping"
                )
            unknown_fields = set(field) - {
                "source",
                "kind",
                "length",
                "indices",
                "default",
                "scale",
            }
            if unknown_fields:
                raise ValueError(
                    f'policy_inputs policy "{key}" field {field_index} has unknown '
                    f"fields {sorted(unknown_fields)}"
                )
            source_name = _policy_key(
                field.get("source"),
                f'policy_inputs policy "{key}" field {field_index} source',
            )
            if source_name not in source_slots:
                raise ValueError(
                    f'policy_inputs policy "{key}" references unknown source '
                    f'"{source_name}"'
                )
            kind = field.get("kind", "copy")
            if kind not in _POLICY_FIELD_KINDS:
                raise ValueError(
                    f'policy_inputs policy "{key}" field {field_index} kind must be '
                    f"one of {_POLICY_FIELD_KINDS}"
                )
            normalized_field = {"kind": kind, "inputSlot": source_slots[source_name]}
            if "length" in field:
                normalized_field["length"] = _positive_integer(
                    field["length"],
                    f'policy_inputs policy "{key}" field {field_index} length',
                )
            if "indices" in field:
                raw_indices = field["indices"]
                if (
                    isinstance(raw_indices, dict)
                    and raw_indices.get("$botlab") == "resource-config"
                ):
                    indices = raw_indices
                else:
                    indices = list(raw_indices)
                    if not indices or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in indices
                    ):
                        raise ValueError(
                            f'policy_inputs policy "{key}" field {field_index} indices '
                            "must contain non-negative integers"
                        )
                    if len(indices) != len(set(indices)):
                        raise ValueError(
                            f'policy_inputs policy "{key}" field {field_index} indices '
                            "must be unique"
                        )
                normalized_field["indices"] = indices
            if "default" in field:
                normalized_field["defaultValues"] = field["default"]
            if "scale" in field:
                normalized_field["scale"] = field["scale"]
            normalized_fields.append(normalized_field)

        normalized_policy = {
            "key": key,
            "mode": _policy_mode(
                policy.get("mode"),
                f'policy_inputs policy "{key}" mode',
            ),
            "outputSlot": policy_index,
            "historyFrames": _positive_integer(
                policy.get("history_frames", 1),
                f'policy_inputs policy "{key}" history_frames',
            ),
            "fields": normalized_fields,
        }
        reference = policy.get("reference")
        if reference is not None:
            reference = _policy_key(
                reference,
                f'policy_inputs policy "{key}" reference',
            )
            if reference not in source_slots:
                raise ValueError(
                    f'policy_inputs policy "{key}" references unknown source '
                    f'"{reference}"'
                )
            normalized_policy["referenceSlot"] = source_slots[reference]
        if "state_dim" in policy:
            normalized_policy["stateDim"] = _positive_integer(
                policy["state_dim"],
                f'policy_inputs policy "{key}" state_dim',
            )
        normalized.append(normalized_policy)

    node = Signal(
        "proc-policy-input",
        "policy-inputs",
        {"modeSlot": 0, "inputLabels": ["MODE", *source_names], "policies": normalized},
        (mode, *source_signals),
    )
    return {
        key: node._output(index, f"policy-input-{key}")
        for index, key in enumerate(keys)
    }


def _model_constant_inputs(constants, input_name, label):
    if not isinstance(constants, dict) or not constants:
        raise ValueError(f"{label} constants must be a non-empty mapping")
    extra_inputs = []
    for name, values in constants.items():
        name = _non_empty_string(name, f"{label} constant name")
        if input_name is not None and name == input_name:
            raise ValueError(f'{label} constant "{name}" conflicts with input_name')
        vector = _finite_values(values, f'{label} constant "{name}"')
        if not vector:
            raise ValueError(f'{label} constant "{name}" must not be empty')
        extra_inputs.append({"name": name, "values": vector})
    return extra_inputs


def _model_call_data(raw, label):
    data = {}
    input_name = raw.get("input_name")
    if input_name is not None:
        data["inputName"] = _non_empty_string(input_name, f"{label} input_name")
    output_name = raw.get("output_name")
    if output_name is not None:
        data["outputName"] = _non_empty_string(output_name, f"{label} output_name")
    has_constants = raw.get("constants") is not None
    has_extra_inputs = raw.get("extra_inputs") is not None
    if has_constants and has_extra_inputs:
        raise ValueError(f"{label} accepts only one of constants or extra_inputs")
    if has_constants:
        data["extraInputs"] = _model_constant_inputs(
            raw["constants"],
            data.get("inputName"),
            label,
        )
    elif has_extra_inputs:
        data["extraInputs"] = raw["extra_inputs"]
    if "output_dim" in raw:
        data["outputDim"] = _positive_integer(raw["output_dim"], f"{label} output_dim")
    return data


def _policy_model_stage(raw, policy_key, stage_index):
    if not isinstance(raw, dict):
        raise TypeError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            "must be a mapping"
        )
    unknown = set(raw) - {
        "model",
        "key",
        "label",
        "output_dim",
        "input_name",
        "output_name",
        "constants",
        "extra_inputs",
    }
    if unknown:
        raise ValueError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            f"has unknown fields {sorted(unknown)}"
        )
    model = raw.get("model")
    if not isinstance(model, Model):
        raise TypeError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            "model must be a BotLab Model"
        )
    result = {
        "key": _policy_key(
            raw.get("key", f"stage-{stage_index}"),
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} key',
        ),
        "label": _policy_key(
            raw.get("label", raw.get("key", f"stage-{stage_index}")),
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} label',
        ),
        "resourceId": model.id,
    }
    result.update(_model_call_data(
        raw,
        f'policy_bank policy "{policy_key}" pipeline stage {stage_index}',
    ))
    return result


_COMBINED_POLICY_INPUT_FIELDS = {
    "reference",
    "history_frames",
    "state_dim",
    "fields",
}
_COMBINED_POLICY_MODEL_FIELDS = {
    "model",
    "pipeline",
    "label",
    "input_name",
    "output_name",
    "constants",
    "extra_inputs",
    "output_dim",
}


def _split_combined_policy_specs(policies):
    input_specs = []
    model_specs = []
    keys = set()
    modes = set()
    allowed = {"key", "mode", *_COMBINED_POLICY_INPUT_FIELDS, *_COMBINED_POLICY_MODEL_FIELDS}
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_bank combined policy {index} must be a mapping")
        unknown = set(policy) - allowed
        if unknown:
            raise ValueError(
                f"policy_bank combined policy {index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_bank combined policy {index} key")
        mode = _policy_mode(policy.get("mode"), f'policy_bank combined policy "{key}" mode')
        if key in keys:
            raise ValueError(f'policy_bank combined policy key "{key}" is duplicated')
        if mode in modes:
            raise ValueError(f"policy_bank combined policy mode {mode} is duplicated")
        keys.add(key)
        modes.add(mode)
        if "fields" not in policy:
            raise ValueError(f'policy_bank combined policy "{key}" requires fields')
        input_specs.append({
            name: policy[name]
            for name in ("key", "mode", *_COMBINED_POLICY_INPUT_FIELDS)
            if name in policy
        })
        model_specs.append({
            name: policy[name]
            for name in ("key", "mode", *_COMBINED_POLICY_MODEL_FIELDS)
            if name in policy
        })
    if not input_specs:
        raise ValueError("policy_bank combined policies must not be empty")
    return input_specs, model_specs


def policy_bank(
    mode,
    inputs=None,
    policies=None,
    *,
    sources=None,
    control_decimation=1,
    call_id=None,
):
    """Run exactly one ONNX policy (or explicit model pipeline) for ``mode``.

    Pass prebuilt ``inputs`` with the low-level two-table API, or pass named
    ``sources`` and combined policy specs containing both ``fields`` and the
    policy ``model``/``pipeline``.
    """
    mode = _require_signal(mode, "policy_bank mode")
    if sources is not None:
        if inputs is not None:
            raise ValueError("policy_bank accepts only one of inputs or sources")
        input_specs, policies = _split_combined_policy_specs(list(policies or ()))
        inputs = policy_inputs(mode, sources, input_specs)
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("policy_bank inputs must be a non-empty mapping")
    state_keys = []
    state_signals = []
    for key, source in inputs.items():
        key = _policy_key(key, "policy_bank input key")
        if key in state_keys:
            raise ValueError(f'policy_bank input key "{key}" is duplicated')
        state_keys.append(key)
        state_signals.append(_require_signal(source, f'policy_bank input "{key}"'))
    state_slots = {key: index + 1 for index, key in enumerate(state_keys)}

    items = list(policies)
    if not items:
        raise ValueError("policy_bank policies must not be empty")
    normalized = []
    resource_ids = []
    policy_keys = []
    for policy_index, policy in enumerate(items):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_bank policy {policy_index} must be a mapping")
        unknown = set(policy) - {
            "key",
            "mode",
            "model",
            "pipeline",
            "label",
            "input_name",
            "output_name",
            "constants",
            "output_dim",
            "extra_inputs",
        }
        if unknown:
            raise ValueError(
                f"policy_bank policy {policy_index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_bank policy {policy_index} key")
        if key in policy_keys:
            raise ValueError(f'policy_bank policy key "{key}" is duplicated')
        policy_keys.append(key)
        if key not in state_slots:
            raise ValueError(f'policy_bank policy "{key}" has no matching input')
        has_model = policy.get("model") is not None
        has_pipeline = policy.get("pipeline") is not None
        if has_model == has_pipeline:
            raise ValueError(
                f'policy_bank policy "{key}" must define exactly one of model or pipeline'
            )
        result = {
            "key": key,
            "mode": _policy_mode(policy.get("mode"), f'policy_bank policy "{key}" mode'),
            "inputSlot": state_slots[key],
            "label": _policy_key(
                policy.get("label", key),
                f'policy_bank policy "{key}" label',
            ),
        }
        if has_model:
            model = policy["model"]
            if not isinstance(model, Model):
                raise TypeError(f'policy_bank policy "{key}" model must be a BotLab Model')
            result["resourceId"] = model.id
            resource_ids.append(model.id)
        else:
            pipeline = list(policy["pipeline"])
            if not pipeline:
                raise ValueError(f'policy_bank policy "{key}" pipeline must not be empty')
            result["pipeline"] = [
                _policy_model_stage(stage, key, stage_index)
                for stage_index, stage in enumerate(pipeline)
            ]
            resource_ids.extend(stage["resourceId"] for stage in result["pipeline"])
        result.update(_model_call_data(policy, f'policy_bank policy "{key}"'))
        normalized.append(result)

    extra_inputs = set(state_keys) - set(policy_keys)
    if extra_inputs:
        raise ValueError(f"policy_bank inputs have no policy: {sorted(extra_inputs)}")
    node_id = resource_ids[0]
    if call_id is not None:
        call_id = _non_empty_string(call_id, "policy_bank call_id")
    return Signal(
        "model-onnx",
        node_id or "policy-bank",
        {
            "resourceId": node_id,
            "controlDecimation": _positive_integer(
                control_decimation,
                "policy_bank control_decimation",
            ),
            "policies": normalized,
        },
        (mode, *state_signals),
        node_id=call_id,
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
    fields = {
        "targetLookahead": lookahead,
        "velocityControlStiffness": velocity_stiffness,
        "headingControlStiffness": heading_stiffness,
        "targetDisThreshold": distance_threshold,
        "yawMinForwardVelocity": min_forward_velocity,
    }
    data = {"poseVelocityCommand": True}
    data.update({key: value for key, value in fields.items() if value is not None})
    return data


def _visit_signals(signal, seen, ordered):
    node = signal._node
    key = id(node)
    if key in seen:
        return
    seen.add(key)
    for source in node._inputs:
        _visit_signals(source, seen, ordered)
    ordered.append(node)


def _unique_node_id(label, taken):
    candidate = label
    suffix = 2
    while candidate in taken:
        candidate = f"{label}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _control_plan(robot_id, sink, pd_signal=None, control_signal=None):
    if (pd_signal is None) == (control_signal is None):
        raise ValueError("control plan requires exactly one output signal")
    ordered = []
    _visit_signals(sink, set(), ordered)

    hooks = [signal for signal in ordered if signal._type == "proc-action-hook"]
    if len(hooks) != 1:
        raise ValueError(
            "a control plan needs exactly one Signal.action(), got "
            f"{len(hooks)}"
        )

    taken = set()
    for signal in ordered:
        if signal._node_id is None:
            continue
        if signal._node_id in taken:
            raise ValueError(f'control plan node id "{signal._node_id}" is duplicated')
        taken.add(signal._node_id)

    node_ids = {
        id(signal): signal._node_id or _unique_node_id(signal._label, taken)
        for signal in ordered
    }

    nodes = []
    edges = []
    for signal in ordered:
        node_id = node_ids[id(signal)]
        nodes.append({
            "id": node_id,
            "type": signal._type,
            "data": _plan_value(signal._data, f'control plan node "{node_id}" data'),
        })
        for index, source in enumerate(signal._inputs):
            edges.append({
                "id": f"edge-{len(edges) + 1}",
                "source": node_ids[id(source._node)],
                "target": node_id,
                "sourcePort": f"out-{source._output_slot}",
                "targetPort": f"in-{index}",
            })

    output = {
        "policyActionNodeId": node_ids[id(hooks[0])],
    }
    if pd_signal is not None:
        output["pdNodeId"] = node_ids[id(pd_signal._node)]
    else:
        output.update({
            "kind": "raw",
            "controlNodeId": node_ids[id(control_signal._node)],
            "controlOutputSlot": control_signal._output_slot,
        })
    plan = {
        "version": 1,
        "robot": robot_id,
        "nodes": nodes,
        "edges": edges,
        "output": output,
    }
    return plan, hooks[0]._modify

class HistoryBuffer:
    def __init__(self, capacity, frame_size):
        self.capacity = _positive_integer(capacity, "HistoryBuffer capacity")
        self.frame_size = _positive_integer(frame_size, "HistoryBuffer frame_size")
        self._frames = []

    def clear(self):
        self._frames = []

    def append(self, values):
        frame = require_vector(values, self.frame_size, "HistoryBuffer frame")
        if not self._frames:
            self._frames = [list(frame) for _ in range(self.capacity)]
        else:
            self._frames.append(frame)
            del self._frames[:-self.capacity]

    def _require_ready(self):
        if not self._frames:
            raise RuntimeError("HistoryBuffer has no frames")

    def stack_segments(self, segments):
        self._require_ready()
        output = []
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValueError("HistoryBuffer segments must be (start, size) pairs")
            start, size = segment
            if (
                isinstance(start, bool)
                or isinstance(size, bool)
                or not isinstance(start, int)
                or not isinstance(size, int)
                or start < 0
                or size <= 0
                or start + size > self.frame_size
            ):
                raise ValueError(
                    f"HistoryBuffer segment {(start, size)} is outside frame size {self.frame_size}"
                )
            for frame in self._frames:
                output.extend(frame[start:start + size])
        return output

    def sample_ages(self, ages):
        self._require_ready()
        output = []
        newest = len(self._frames) - 1
        for age in ages:
            if (
                isinstance(age, bool)
                or not isinstance(age, int)
                or age < 0
                or age >= self.capacity
            ):
                raise ValueError(
                    f"HistoryBuffer age {age} must be between 0 and {self.capacity - 1}"
                )
            output.extend(self._frames[newest - age])
        return output


class Frame:
    def __init__(self, dt, time, index):
        self.dt = float(dt)
        self.time = float(time)
        self.index = int(index)


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
        return botlab_host.observe(self.id).to_py()

    def action_dimension(self):
        return int(botlab_host.actionDimension(self.id))

    def default_joint_positions(self):
        return botlab_host.defaultJointPositions(self.id).to_py()

    def control_parameter(self, name):
        return botlab_host.controlParameter(self.id, str(name)).to_py()

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

    @property
    def policy_order(self):
        """Native joint index for each policy slot."""
        return resource_config(self.id, "policyJointOrder")

    @property
    def motor_order(self):
        """Policy slot for each native motor; index an action with it."""
        return resource_config(self.id, "motorJointOrder")

    @property
    def action_scale(self):
        return resource_config(self.id, "actionScale")

    @property
    def kp(self):
        return resource_config(self.id, "kp")

    @property
    def kd(self):
        return resource_config(self.id, "kd")

    @property
    def effort_limit(self):
        return resource_config(self.id, "effortLimit")

    @property
    def default_angles(self):
        return default_joint_positions(self.id)

    # -- observation signals -----------------------------------------------

    def base_angular_velocity(self, *, scale=None, source="root"):
        return Signal("obs-base-ang-vel", "base-angular-velocity", _observation_data(
            {"dim": 3, "velocitySource": _non_empty_string(source, "velocity source")},
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
        keys=None,
        bindings=None,
        mappings=None,
    ):
        """Command vector read from project parameters, one axis per name.

        Axis limits come from each parameter's `range` in project(), so they
        are declared once rather than restated here.
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
        ranges = [botlab_host.parameterRange(name) for name in names]
        bounds = [None if item is None else item.to_py() for item in ranges]
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


class Model:
    def __init__(self, source, *, provider=None, config=None, name=None):
        self.id = _declare_resource(
            "model",
            source,
            provider=provider,
            config=config,
            name=name,
        )

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    async def predict(self, values):
        result = await botlab_host.predictModel(self.id, values)
        return result.to_py()

    def __call__(
        self,
        *inputs,
        input_name=None,
        output_name=None,
        constants=None,
        control_decimation=1,
        call_id=None,
    ):
        """Run this ONNX model on the given signals, concatenated in order."""
        if not inputs:
            raise ValueError(f'model "{self.id}" requires at least one input signal')
        source = _as_signal(
            inputs[0] if len(inputs) == 1 else list(inputs),
            f'model "{self.id}" input',
        )
        # The resource owns the loaded ONNX session; each call is a separate
        # plan node so caches, decimation, statistics, and recurrent state stay isolated.
        data = {
            "resourceId": self.id,
            "controlDecimation": _positive_integer(
                control_decimation,
                "model control_decimation",
            ),
        }
        data.update(_model_call_data({
            "input_name": input_name,
            "output_name": output_name,
            "constants": constants,
        }, "model"))
        if data["controlDecimation"] == 1:
            del data["controlDecimation"]
        if call_id is not None:
            call_id = _non_empty_string(call_id, "model call_id")
        return Signal(
            "model-onnx",
            self.id,
            data,
            (source,),
            node_id=call_id,
        )


class Motion:
    def __init__(self, source, *, config=None, name=None):
        self.id = _declare_resource("motion", source, config=config, name=name)

    @classmethod
    def _from_id(cls, resource_id):
        handle = cls.__new__(cls)
        handle.id = resource_id
        return handle

    @property
    def source(self):
        return str(botlab_host.resourceSource(self.id))


class _DepthFrame:
    __slots__ = ("data", "age_steps")

    def __init__(self, signal):
        self.data = signal
        self.age_steps = signal._output(1, "depth-age-steps")


class Sensor:
    def __init__(self, source, *, preset=None, config=None, name=None):
        self.id = _declare_resource(
            "sensor",
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

    def observe(self):
        return botlab_host.observeSensor(self.id).to_py()

    def depth(self, *, max_age=0):
        """Latest depth frame, rejected once it is older than `max_age` steps."""
        return Signal("obs-camera", "depth", {
            "resourceId": self.id,
            "maxAgeSteps": _non_negative_integer(max_age, "depth max_age"),
        })

    def depth_frame(self, *, max_age=0):
        """Depth data and its frame age, exposed as outputs of one camera node."""
        signal = Signal("obs-camera", "depth-frame", {
            "resourceId": self.id,
            "maxAgeSteps": _non_negative_integer(max_age, "depth_frame max_age"),
            "includeAgeSteps": True,
        })
        return _DepthFrame(signal)


class Scene:
    def __init__(self, source, *, preset=None, config=None, name=None):
        self.id = _declare_resource(
            "scene",
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

    async def update_heightfield(self, values):
        """Replace a procedural heightfield's normalized elevations at runtime.

        ``values`` may be a flat row-major sequence or a rectangular list of rows.
        The scene's configured grid fixes its dimensions; values must be finite and
        normalized to the MuJoCo ``[0, 1]`` heightfield range.
        """
        await botlab_host.updateHeightfield(self.id, values)


class Resources:
    """Handles for resources `botlab.yaml` already declared, looked up by id."""

    def robot(self, resource_id):
        botlab_host.requireResource(str(resource_id), "robot")
        return Robot._from_id(str(resource_id))

    def model(self, resource_id):
        botlab_host.requireResource(str(resource_id), "model")
        return Model._from_id(str(resource_id))

    def motion(self, resource_id):
        botlab_host.requireResource(str(resource_id), "motion")
        return Motion._from_id(str(resource_id))

    def sensor(self, resource_id):
        botlab_host.requireResource(str(resource_id), "sensor")
        return Sensor._from_id(str(resource_id))

    def scene(self, resource_id):
        botlab_host.requireResource(str(resource_id), "scene")
        return Scene._from_id(str(resource_id))


class Parameters:
    def get(self, name, default=None):
        value = botlab_host.parameter(str(name))
        return default if value is None else value


class OutputChannel:
    def __init__(self, name):
        self._name = name

    def update(self, value):
        getattr(botlab_host, self._name)(value)


class Controller:
    def __init__(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError("Controller robot must be a BotLab Robot")
        self.robot = robot

    def reset(self):
        pass

    def step(self, observation, frame):
        raise NotImplementedError("Controller.step(observation, frame) must return an action")


class PolicyController(Controller):
    """Run a complete signal expression in the engine hot path.

    Exactly one output is explicit: ``target`` enables BotLab position PD, while
    ``control`` writes native MuJoCo actuator controls directly. Both paths keep
    exactly one upstream ``Signal.action()`` as the previous-action/hook boundary.
    """

    def __init__(
        self,
        robot,
        *,
        target=None,
        control=None,
        kp=None,
        kd=None,
        effort_limit=None,
    ):
        if not isinstance(robot, Robot):
            raise TypeError("PolicyController robot must be a BotLab Robot")
        if (target is None) == (control is None):
            raise ValueError("PolicyController requires exactly one of target or control")
        super().__init__(robot)

        if control is not None:
            if any(value is not None for value in (kp, kd, effort_limit)):
                raise ValueError(
                    "PolicyController raw control does not accept kp, kd, or effort_limit"
                )
            control = _as_signal(control, "PolicyController control")
            sink = Signal("out-mujoco", "controller", {}, (control,))
            self._plan_data, self.modify_action = _control_plan(
                robot.id,
                sink,
                control_signal=control,
            )
        else:
            target = _as_signal(target, "PolicyController target")
            explicit_pd_values = (kp, kd, effort_limit)
            dimension = (
                robot.action_dimension()
                if any(value is not None for value in explicit_pd_values)
                else None
            )
            pd = Signal(
                "proc-pd",
                "position-pd",
                {
                    "pdType": "position",
                    "kps": (
                        robot.kp
                        if kp is None
                        else _policy_pd_vector(kp, dimension, "PolicyController kp")
                    ),
                    "kds": (
                        robot.kd
                        if kd is None
                        else _policy_pd_vector(kd, dimension, "PolicyController kd")
                    ),
                    "effortLimit": (
                        robot.effort_limit
                        if effort_limit is None
                        else _policy_pd_vector(
                            effort_limit,
                            dimension,
                            "PolicyController effort_limit",
                        )
                    ),
                },
                (
                    target,
                    robot.joint_position(order="native", _label="pd-joint-position"),
                    robot.joint_velocity(order="native", _label="pd-joint-velocity"),
                ),
            )
            sink = Signal("out-mujoco", "controller", {}, (pd,))
            self._plan_data, self.modify_action = _control_plan(robot.id, sink, pd)

    @property
    def plan(self):
        return _plan_value(self._plan_data, "PolicyController plan")


class PositionController(Controller):
    """Holds a fixed joint-position target through a Python-side PD command."""

    def __init__(self, robot, *, kp, kd, effort_limit, target=None):
        super().__init__(robot)
        dimension = robot.action_dimension()
        self.target = require_vector(
            robot.default_joint_positions() if target is None else target,
            dimension,
            "PositionController target",
        )
        self.kp = _expand_vector(kp, dimension, "PositionController kp")
        self.kd = _expand_vector(kd, dimension, "PositionController kd")
        self.effort_limit = _expand_vector(
            effort_limit,
            dimension,
            "PositionController effort_limit",
        )
        if any(value <= 0 for value in self.effort_limit):
            raise ValueError("PositionController effort_limit values must be positive")
        native = PolicyController(
            robot,
            target=constant(self.target).action(),
            kp=self.kp,
            kd=self.kd,
            effort_limit=self.effort_limit,
        )
        self._plan_data = native._plan_data
        self.modify_action = None

    def step(self, observation, frame):
        return self.robot.position_pd(
            self.target,
            self.kp,
            self.kd,
            self.effort_limit,
        )


class Simulation:
    @classmethod
    def from_project(cls):
        return cls()

    def __init__(self):
        self.resources = Resources()
        self.parameters = Parameters()
        self.metrics = OutputChannel("updateMetrics")
        self.diagnostics = OutputChannel("updateDiagnostics")

    def model(self, path):
        """The ONNX model declared at `path` in botlab.yaml's `models:` list.

        The path is the resource ID, so the reference here reads as the file it loads.
        """
        return self.resources.model(path)

    def request_reset(self, reason="Python controller requested reset"):
        botlab_host.requestReset(str(reason))

    def run(self, controller):
        is_group = isinstance(controller, (list, tuple))
        controllers = list(controller) if is_group else [controller]
        if not controllers or any(
            not isinstance(item, Controller) for item in controllers
        ):
            raise TypeError(
                "Simulation.run(controller) requires a BotLab Controller or "
                "a non-empty list/tuple of Controllers"
            )

        is_native_position_controller = type(controller) is PositionController
        if not is_group and (
            isinstance(controller, PolicyController) or is_native_position_controller
        ):
            modifier_proxy = None
            if controller.modify_action is not None:
                async def modify_action(action, dt, time, index):
                    result = controller.modify_action(
                        list(action),
                        Frame(dt, time, index),
                    )
                    if inspect.isawaitable(result):
                        result = await result
                    return result

                if create_proxy is None:
                    # Static host: the callable itself marks the plan as
                    # needing the Python runtime instead of a JS callback.
                    modifier_proxy = controller.modify_action
                else:
                    modifier_proxy = create_proxy(modify_action)
            try:
                if modifier_proxy is None:
                    botlab_host.registerPolicyController(controller._plan_data)
                else:
                    botlab_host.registerPolicyController(
                        controller._plan_data,
                        modifier_proxy,
                    )
            except Exception:
                if modifier_proxy is not None:
                    modifier_proxy.destroy()
                raise
            botlab_host.log(
                "info",
                "Python",
                f"sim.run({type(controller).__name__}) started",
            )
            return

        if any(isinstance(item, PolicyController) for item in controllers):
            raise TypeError(
                "Simulation.run controller groups support custom Controllers only"
            )

        robot_ids = [item.robot.id for item in controllers]
        if len(robot_ids) != len(set(robot_ids)):
            raise ValueError(
                "Simulation.run controllers must target distinct robots"
            )

        async def reset_controller():
            for item in controllers:
                result = item.reset()
                if inspect.isawaitable(result):
                    await result

        async def step_controller(dt, time, index):
            frame = Frame(dt, time, index)
            for item in controllers:
                observation = item.robot.observe()
                action = item.step(observation, frame)
                if inspect.isawaitable(action):
                    action = await action
                item.robot.apply_action(action)

        if create_proxy is not None:
            reset_proxy = create_proxy(reset_controller)
            step_proxy = create_proxy(step_controller)
        else:
            reset_proxy = reset_controller
            step_proxy = step_controller
        try:
            botlab_host.registerReset(reset_proxy)
            botlab_host.registerStep(step_proxy, robot_ids)
        except Exception:
            reset_proxy.destroy()
            step_proxy.destroy()
            raise
        label = "controllers" if is_group else "controller"
        botlab_host.log("info", "Python", f"sim.run({label}) started")
