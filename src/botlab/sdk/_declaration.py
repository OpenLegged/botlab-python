"""Host bridge, project and resource declarations, preset calibration handles."""
import inspect
import math
import os
import re

import botlab_host

from ._vectors import _expand_vector


def _host_value(value):
    """Host values cross as JsProxy under Pyodide and as plain data under CPython."""
    return value.to_py() if hasattr(value, "to_py") else value


def _host_list(value):
    converted = _host_value(value)
    if converted is None:
        return None
    if isinstance(converted, memoryview):
        # Host arrays arrive as zero-copy buffer views under Pyodide.
        return list(converted)
    return [float(item) for item in converted]


def _host_snapshot(value):
    """Host observation with every array materialized as a plain list.

    The browser bridge sends observation arrays as TypedArrays, which ``.to_py()``
    maps to zero-copy buffer views instead of walking a plain JS Array element by
    element; the native host already sends lists. Lists are the SDK contract
    (``robot.observe()`` mirrors the native ``.tolist()`` layout), so buffers are
    materialized here once in C rather than copied across the boundary.
    """
    converted = _host_value(value)
    if isinstance(converted, dict):
        return {key: _host_snapshot(item) for key, item in converted.items()}
    if isinstance(converted, memoryview):
        return list(converted)
    return converted


def _non_empty_string(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


# 展示文案支持的界面语言；与前端 src/domain/low-code/localizedText.ts 保持一致。
_TEXT_LOCALES = ("en", "zh")


def _localized_text(value, label):
    """Display text: a non-empty string, or {"en": ..., "zh": ...} with "en" as fallback."""
    if isinstance(value, str):
        # 纯字符串沿用原有规则，已有工程的计划输出不变。
        return _non_empty_string(value, label)
    if not isinstance(value, dict):
        raise TypeError(f'{label} must be a non-empty string or a {{"en": ..., "zh": ...}} mapping')
    unknown = sorted(set(value) - set(_TEXT_LOCALES))
    if unknown:
        raise ValueError(f"{label} has unsupported locales {unknown} (expected {list(_TEXT_LOCALES)})")
    if "en" not in value:
        raise ValueError(f'{label} must include an "en" entry')
    table = {}
    for locale in _TEXT_LOCALES:
        if locale not in value:
            continue
        text = value[locale]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label}.{locale} must be a non-empty string")
        table[locale] = text.strip()
    return table


def _plan_value(value, label):
    if isinstance(value, _ResourceValues):
        value = value._plan
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


_PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def _declaration_location():
    """Return the first project-owned Python frame for declaration diagnostics."""
    frame = inspect.currentframe()
    try:
        frame = None if frame is None else frame.f_back
        while frame is not None:
            filename = frame.f_code.co_filename
            sdk_frame = os.path.dirname(os.path.abspath(filename)) == _PACKAGE_DIRECTORY
            if not sdk_frame and filename.endswith(".py"):
                return {
                    "file": filename.replace("\\", "/").rsplit("/", 1)[-1],
                    "line": frame.f_lineno,
                    "column": 1,
                }
            frame = frame.f_back
    finally:
        del frame
    return None


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
        _declaration_location(),
    )
    return resource_id


def _respawn_binding(value):
    """Validate a ``project(respawn=...)`` device binding mapping.

    ``{"keyboard": {"key": "KeyR"}, "gamepad": {"button": "X"}}`` -- either
    device may be omitted, unknown fields are rejected, and an empty mapping
    is a declaration error.
    """
    if not isinstance(value, dict):
        raise TypeError("project respawn must be a mapping")
    unknown = [key for key in value if key not in ("keyboard", "gamepad")]
    if unknown:
        raise ValueError(f'project respawn has unknown key "{unknown[0]}" (only keyboard/gamepad)')
    binding = {}
    for device, field in (("keyboard", "key"), ("gamepad", "button")):
        section = value.get(device)
        if section is None:
            continue
        if not isinstance(section, dict) or list(section) != [field]:
            raise ValueError(
                f"project respawn.{device} must be a one-entry mapping "
                f"{{{field!r}: '<code>'}}"
            )
        code = section[field]
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"project respawn.{device}.{field} must be a non-empty string")
        binding[device] = {field: code}
    if not binding:
        raise ValueError("project respawn needs a keyboard key or a gamepad button")
    return binding


def project(name, *, control_hz=None, physics_hz=None, parameters=None, respawn=None):
    """Name the project and override the runtime defaults.

    Optional: a project that keeps the documented defaults (50 Hz control,
    200 Hz physics) and declares no tunable parameters never has to call this.

    ``respawn`` declares the project's respawn input -- the key or gamepad
    button that picks the robot back up where it fell, e.g.
    ``respawn={"keyboard": {"key": "KeyR"}, "gamepad": {"button": "X"}}``,
    typically read from the ``respawn`` sections of ``config/keyboard.yaml``
    and ``config/gamepad.yaml``. Respawn is a project input, not a platform
    default: a project that declares none has no respawn key.
    """
    botlab_host.declareProject(
        name,
        control_hz,
        physics_hz,
        _plan_value({} if parameters is None else parameters, "project parameters"),
        None if respawn is None else _plan_value(_respawn_binding(respawn), "project respawn"),
        _declaration_location(),
    )


# ---------------------------------------------------------------------------
# Control profile
#
# The external CLI-control face is declared, not derived: which parameter is
# the vx axis, what an action is called and which policy states it may start
# from are project decisions the compiled plan cannot guess. The declaration
# travels in the manifest; the browser page cross-checks it against the
# compiled plan before uploading a control descriptor to a middleware.
# ---------------------------------------------------------------------------

_CONTROL_SCHEMA = "botlab.control.profile.v3"
_CONTROL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CONTROL_CAPABILITY = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_CONTROL_RESERVED_CAPABILITIES = ("motion.twist", "control.stop")
_CONTROL_EXPERIENCE_LIMIT = 32768
# Per-kind command time budgets are project-declarable, but the platform hard
# caps are a protocol invariant mirrored here so an over-budget declaration
# fails at declaration time, not on first command. Keep in sync with
# CONTROL_BUDGET_CAPS_MS in src/domain/controls/controlDescriptor.ts.
_CONTROL_BUDGET_CAPS_MS = {"move_ms": 60_000, "event_ms": 300_000, "task_ms": 600_000}
_CANONICAL_DECIMAL_STATE_KEY = re.compile(r"^(0|[1-9][0-9]*)$")


def _state_code_key(key):
    """State codes arrive as ints from Python and as canonical decimal strings
    from restricted YAML, whose mapping keys can only be strings."""
    if isinstance(key, bool):
        raise ValueError("control_profile states keys must be integer state codes")
    if isinstance(key, int):
        return key
    if isinstance(key, str) and _CANONICAL_DECIMAL_STATE_KEY.match(key):
        return int(key)
    raise ValueError("control_profile states keys must be integer state codes")


def _control_state_code(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be non-negative integers")
    return value


def _control_state_codes(value, label):
    codes = list(value) if isinstance(value, (list, tuple)) else None
    if not codes:
        raise ValueError(f"{label} must be a non-empty sequence of state codes")
    codes = [_control_state_code(code, label) for code in codes]
    if len(set(codes)) != len(codes):
        raise ValueError(f"{label} must not repeat states")
    return codes


def action(name, event_value, *, allowed_states, target_states, capability=None, idempotent=False, task=False):
    """Declare one named action for a :func:`control_profile`.

    ``name`` is the CLI alias an AI assistant calls. ``event_value`` must be one
    of the mapped command values the project feeds its ``state_machine(...)``.
    ``allowed_states`` are the policy states the action may start from and
    ``target_states`` the steady states it reaches -- declared explicitly, never
    derived from the state machine's transitions. ``capability`` defaults to
    ``action.<name>``. ``task=True`` marks the action as carrying a
    natural-language task payload (the CLI ``task`` command), which the runtime
    enforces against these declared values.
    """
    return _normalize_action(
        {
            "name": name,
            "event_value": event_value,
            "allowed_states": allowed_states,
            "target_states": target_states,
            "capability": capability,
            "idempotent": idempotent,
            "task": task,
        },
        "control action",
    )


_ACTION_FIELDS = ("name", "event_value", "allowed_states", "target_states", "capability", "idempotent", "task")


def _normalize_action(item, label):
    """Validate and normalize one action mapping.

    ``action(...)`` declarations and plain mappings (the ones a project reads
    from ``config/control.yaml`` via :func:`read_config`) share this path:
    ``capability`` and ``idempotent`` default the same way, unknown fields are
    rejected rather than silently dropped.
    """
    if not isinstance(item, dict):
        raise TypeError(f"{label} must come from action(...)")
    unknown = set(item) - set(_ACTION_FIELDS)
    if unknown:
        raise ValueError(f"{label} has unknown fields {sorted(unknown)}")
    name = item.get("name")
    if not isinstance(name, str) or not _CONTROL_NAME.match(name):
        raise ValueError(f'{label} name "{name}" must match ^[a-z][a-z0-9_]{{0,63}}$')
    event_value = item.get("event_value")
    if isinstance(event_value, bool) or not isinstance(event_value, int) or event_value == 0:
        raise ValueError(f'{label} "{name}" event_value must be a non-zero integer')
    resolved_capability = f"action.{name}" if item.get("capability") is None else item["capability"]
    if (
        not isinstance(resolved_capability, str)
        or not _CONTROL_CAPABILITY.match(resolved_capability)
        or resolved_capability in _CONTROL_RESERVED_CAPABILITIES
    ):
        raise ValueError(
            f'{label} "{name}" capability "{resolved_capability}" must be '
            "dotted lowercase and must not use motion.twist or control.stop"
        )
    idempotent = item.get("idempotent", False)
    if not isinstance(idempotent, bool):
        raise TypeError(f'{label} "{name}" idempotent must be a boolean')
    task = item.get("task", False)
    if not isinstance(task, bool):
        raise TypeError(f'{label} "{name}" task must be a boolean')
    return {
        "name": name,
        "event_value": event_value,
        "allowed_states": _control_state_codes(
            item.get("allowed_states"), f'{label} "{name}" allowed_states'
        ),
        "target_states": _control_state_codes(
            item.get("target_states"), f'{label} "{name}" target_states'
        ),
        "capability": resolved_capability,
        "idempotent": idempotent,
        "task": task,
    }


def _control_state_node(value):
    """The plan node id string naming the state machine the descriptor drives.

    A state_machine(...) signal compiles to exactly one ``proc-state-machine``
    node whose id is its label ("state-machine"); the paired page cross-checks
    the declared name against the compiled plan before uploading anything.
    """
    return _non_empty_string(value, "control_profile state_node")


_CONTROL_FEEDBACK_CHANNELS = (
    "policy_observation",
    "position",
    "orientation",
    "base_velocity",
    "joint_positions",
)


def _control_feedback(value):
    """Normalize the AI observation face: the state-frame channels a
    non-privileged external controller may read.

    ``None`` (no feedback block in config/control.yaml) closes every channel:
    privileged ground truth is opt-in per project. The full five-channel mapping
    is always returned so re-declaration equality stays stable.
    """
    if value is None:
        return {channel: False for channel in _CONTROL_FEEDBACK_CHANNELS}
    if not isinstance(value, dict):
        raise TypeError("control_profile feedback must be a mapping of channel to boolean")
    unknown = set(value) - set(_CONTROL_FEEDBACK_CHANNELS)
    if unknown:
        raise ValueError(f"control_profile feedback has unknown fields {sorted(unknown)}")
    feedback = {}
    for channel in _CONTROL_FEEDBACK_CHANNELS:
        enabled = value.get(channel, False)
        if not isinstance(enabled, bool):
            raise TypeError(f"control_profile feedback {channel} must be a boolean")
        feedback[channel] = enabled
    return feedback


def _control_budgets(value):
    """Normalize the v2 command time budgets: per-kind ceilings the runtime
    enforces on move duration and event/task waits.

    Only explicitly declared slots survive (undeclared kinds fall back to the
    platform defaults at descriptor build), and every value is checked against
    the platform hard caps -- the caps themselves are a protocol invariant, not
    a project knob.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("control_profile budgets must be a mapping of budget to milliseconds")
    unknown = set(value) - set(_CONTROL_BUDGET_CAPS_MS)
    if unknown:
        raise ValueError(f"control_profile budgets has unknown fields {sorted(unknown)}")
    budgets = {}
    for slot, cap in _CONTROL_BUDGET_CAPS_MS.items():
        declared = value.get(slot)
        if declared is None:
            continue
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0 or declared > cap:
            raise ValueError(f"control_profile budgets {slot} must be an integer in (0, {cap}]")
        budgets[slot] = declared
    return budgets or None


def control_profile(
    robot,
    *,
    state_node,
    vx=None,
    yaw_rate=None,
    actions=(),
    motion_states=None,
    states=None,
    experience=None,
    feedback=None,
    budgets=None,
):
    """Declare the external CLI-control face of the project.

    ``robot`` is a ``Robot(...)`` handle, ``state_node`` the plan node id of the
    ``state_machine(...)`` driving the policy modes. The event source is
    structural: the state machine's command input is the mapped commands node,
    and each action's ``event_value`` is one of its mapping codes. ``vx`` /
    ``yaw_rate`` name the motion axes; omitting both declares a motionless
    control face (no ``move`` primitive). ``feedback`` declares the AI
    observation face: the state-frame channels a non-privileged controller may
    read (``policy_observation`` taps the policy input vector; ``position`` /
    ``orientation`` / ``base_velocity`` are world-frame ground truth;
    ``joint_positions`` is the robot's joint position span), all closed when
    omitted. ``budgets`` optionally tightens the per-kind command time budgets
    within the platform caps. Pass the result to :func:`declare_control`; the
    browser page cross-checks it against the compiled plan before pairing with
    a ``botlab control serve`` middleware.
    """
    items = list(actions) if isinstance(actions, (list, tuple)) else None
    if not items:
        raise ValueError("control_profile actions must be a non-empty sequence of action(...) declarations")
    declared_actions = []
    names = set()
    capabilities = set()
    event_values = []
    for index, item in enumerate(items):
        declared = _normalize_action(item, f"control_profile actions[{index}]")
        if declared["name"] in names:
            raise ValueError(f'duplicate control action name "{declared["name"]}"')
        names.add(declared["name"])
        if declared["capability"] in capabilities:
            raise ValueError(f'duplicate control capability id "{declared["capability"]}"')
        capabilities.add(declared["capability"])
        if declared["event_value"] not in event_values:
            event_values.append(declared["event_value"])
        declared_actions.append(declared)

    motion_states_codes = (
        None if motion_states is None
        else _control_state_codes(motion_states, "control_profile motion_states")
    )

    state_names = None
    if states is not None:
        if not isinstance(states, dict) or not states:
            raise ValueError("control_profile states must be a non-empty mapping of state code to name")
        state_names = {}
        for key, value in states.items():
            code = _state_code_key(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'control_profile states[{code}] must be a non-empty string')
            state_names[str(code)] = value

    if experience is not None:
        if not isinstance(experience, str) or not experience:
            raise ValueError("control_profile experience must be a non-empty string")
        if len(experience.encode("utf-8")) > _CONTROL_EXPERIENCE_LIMIT:
            raise ValueError(
                f"control_profile experience exceeds {_CONTROL_EXPERIENCE_LIMIT} bytes"
            )

    # v3: vx / yaw_rate 声明运动轴，二者必须成对出现；都省略 = 无运动轴的
    # 控制面（move 原语不存在），此时 motion_states 无从谈起。
    if (vx is None) != (yaw_rate is None):
        raise ValueError("control_profile vx and yaw_rate must be declared together")
    if vx is None:
        if motion_states_codes is not None:
            raise ValueError("control_profile motion_states requires the motion axes (vx, yaw_rate)")
        motion_parameters = None
    else:
        axis_vx = _non_empty_string(vx, "control_profile vx")
        axis_yaw = _non_empty_string(yaw_rate, "control_profile yaw_rate")
        if axis_vx == axis_yaw:
            raise ValueError("control_profile vx and yaw_rate must be distinct parameters")
        motion_parameters = {"vx": axis_vx, "yaw_rate": axis_yaw}

    return {
        "schema": _CONTROL_SCHEMA,
        "robot": _resource_source(robot, "control_profile robot"),
        "state_node": _control_state_node(state_node),
        "motion_parameters": motion_parameters,
        "motion_states": motion_states_codes,
        "states": state_names,
        "experience": experience,
        "feedback": _control_feedback(feedback),
        "budgets": _control_budgets(budgets),
        "actions": declared_actions,
    }


def _control_profile_value(control):
    if not isinstance(control, dict) or control.get("schema") != _CONTROL_SCHEMA:
        raise TypeError("declare_control expects a control_profile(...) declaration")
    return control


def declare_control(control):
    """Attach a :func:`control_profile` to the project declaration.

    Call it after ``project()``: the profile's motion axes must be declared
    parameters, and the browser page only pairs a project whose declaration
    matches its compiled plan.
    """
    botlab_host.declareProjectControl(
        _plan_value(_control_profile_value(control), "project control"),
        _declaration_location(),
    )


def resource_config(resource, field):
    return {
        "$botlab": "resource-config",
        "resource": _non_empty_string(resource, "resource_config resource"),
        "field": _non_empty_string(field, "resource_config field"),
    }


def _optional_resource_config(resource, field):
    return {
        "$botlab": "optional-resource-config",
        "resource": _non_empty_string(resource, "optional resource config resource"),
        "field": _non_empty_string(field, "optional resource config field"),
    }


def default_joint_positions(resource):
    return {
        "$botlab": "default-joint-positions",
        "resource": _non_empty_string(resource, "default_joint_positions resource"),
    }


class _ResourceValues:
    """Preset calibration: a plan reference that also reads as a plain Python sequence.

    Signal expressions keep the reference, so the compiled plan still resolves joint
    order and gains from the resource preset instead of baking them in. Project Python
    -- an arm helper, a custom controller -- can `len()`, index and iterate the same
    handle. Values are read on first use, so declaring a robot never needs the preset
    to be loaded yet.
    """

    __slots__ = ("_plan", "_read", "_values")

    def __init__(self, plan, read):
        self._plan = plan
        self._read = read
        self._values = None

    def _resolve(self):
        if self._values is None:
            values = _host_value(self._read())
            if values is None:
                raise ValueError(f"{self._label()} is not defined by the resource preset")
            # Host numbers arrive as floats; integral ones are joint indices far more
            # often than they are gains, and `motor_order[0]` reads better as 3 than 3.0.
            self._values = tuple(
                int(item) if float(item).is_integer() else float(item)
                for item in values
            )
        return self._values

    def _label(self):
        field = self._plan.get("field")
        return f'"{self._plan["resource"]}" {field}' if field else (
            f'"{self._plan["resource"]}" default joint positions'
        )

    def __len__(self):
        return len(self._resolve())

    def __iter__(self):
        return iter(self._resolve())

    def __getitem__(self, index):
        return self._resolve()[index]

    def __repr__(self):
        return f"<botlab calibration {self._label()}>"


def _plan_reference(value):
    """The plan-side form of a preset reference, or None for an ordinary value."""
    if isinstance(value, _ResourceValues):
        return value._plan
    if isinstance(value, dict) and "$botlab" in value:
        return value
    return None


def _policy_pd_vector(value, size, label):
    reference = _plan_reference(value)
    if reference is not None:
        return reference
    return _expand_vector(value, size, label)
