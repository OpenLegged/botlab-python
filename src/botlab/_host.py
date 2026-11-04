"""Static BotLab host for the offline CLI and the native runtime.

The browser runs ``botlab_host`` as a JS module wired into the simulation
bridge. This module is its CPython counterpart: it collects the declarations
a project's ``controller.py`` makes at import time, answers the few queries
the SDK needs while building signal expressions (``actionDimension``,
``parameterRange``), and captures the compiled controller plan that
``Simulation.run()`` registers. The native runtime (``botlab run``, the
locally maintained ``botlab._runtime`` package) installs its session via
``set_runtime_session`` before a project's ``main.py`` runs; runtime calls
then delegate to that session instead of raising.

It mirrors the TypeScript collector in BotLab
(``src/domain/python/pythonDeclaredProject.ts``) field by field so the
manifest produced here is the same shape the browser produces.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

DEFAULT_CONTROL_HZ = 50
DEFAULT_PHYSICS_HZ = 200

RESOURCE_KINDS = frozenset({"robot", "model", "scene", "sensor", "motion", "mesh"})
MODEL_PROVIDERS = frozenset({"auto", "webgpu", "webnn", "wasm"})

PRESET_FILE_SUFFIX = ".preset.yaml"

# Node types the release validator accepts (SimulationEngine ProcessorRegistry).
SUPPORTED_NODE_TYPES = frozenset({
    "obs-qpos", "obs-qvel", "obs-imu", "obs-constant", "obs-base-height",
    "obs-base-lin-vel", "obs-base-ang-vel", "obs-projected-gravity", "obs-commands",
    "obs-dof-pos", "obs-dof-vel", "obs-actions", "obs-camera", "obs-mujoco-query",
    "obs-height-scan",
    "proc-math", "proc-history", "proc-mux", "proc-demux", "proc-clamp",
    "proc-reorder", "proc-resize", "proc-reroute", "proc-pd",
    "proc-motion-reference", "proc-motion-sample", "proc-policy-input",
    "proc-onehot-index",
    "proc-action-hook", "proc-delay", "proc-rule-index", "proc-one-hot",
    "proc-compare", "proc-logic", "proc-where", "proc-state-machine",
    "proc-phase-clock", "proc-clock", "proc-spatial-math", "proc-signal-math",
    "proc-signal-noise", "proc-signal-filter", "proc-vector-math",
    "proc-fsm", "proc-mode-switch", "proc-select",
    "proc-mode-select", "proc-signal-send", "proc-signal-receive",
    "model-onnx", "out-action", "out-mujoco", "out-scope", "out-logger",
    "scene-robot", "scene-terrain",
})

MOTION_REQUIRED_FILES = (
    "body_pos_w.bin",
    "body_quat_w.bin",
    "body_lin_vel_w.bin",
    "body_ang_vel_w.bin",
    "joint_pos.bin",
    "joint_vel.bin",
    "fps.bin",
)


class BotLabError(RuntimeError):
    """A project failed BotLab's declaration or packaging contract."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BotLabError(f"[botlab] {label} must be a non-empty string")
    return value


def _json_value(value: Any, label: str, _stack: frozenset[int] = frozenset()) -> Any:
    """Normalize a Python declaration value to its JS counterpart.

    Pyodide converts Python ``None`` to JavaScript ``undefined``; inside a
    declaration mapping it represents JSON ``null``, so it is normalized here
    before the value is validated. Numbers must be finite; integers are kept
    as ``int`` so the JSON serialization matches JS integer rendering.
    """
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BotLabError(f"[botlab] {label} must be finite")
        if value.is_integer() and abs(value) < 2**31:
            return int(value)
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if id(value) in _stack:
            raise BotLabError(f"[botlab] {label} must not contain a cycle")
        stack = _stack | {id(value)}
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise BotLabError(f"[botlab] {label} keys must be non-empty strings")
            result[key] = _json_value(item, f"{label}.{key}", stack)
        return result
    raise BotLabError(f"[botlab] {label} must be JSON-serializable")


def _resource_source(value: Any, label: str) -> str:
    source = _text(value, label)
    if source.startswith("/") or "\\" in source or ".." in source.split("/"):
        raise BotLabError(f"[botlab] {label} must be a URL or a safe project-relative/public asset path")
    return source


def _positive_runtime_frequency(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BotLabError(f"[botlab] {label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BotLabError(f"[botlab] {label} must be a finite number")
    if not math.isfinite(value):
        raise BotLabError(f"[botlab] {label} must be a finite number")
    return value


def _declared_parameter(pid: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise BotLabError(f"[botlab] Parameter {pid!r} must be a mapping")
    known = {"label", "default", "range", "step", "unit", "options"}
    for key in value:
        if key not in known:
            raise BotLabError(f"[botlab] Parameter {pid!r} has unknown field {key!r}")
    default_value = _finite_number(value.get("default"), f"Parameter {pid!r} default")
    options = None
    if "options" in value:
        if any(key in value for key in ("range", "step", "unit")):
            raise BotLabError(f"[botlab] Parameter {pid!r} options cannot be combined with range, step or unit")
        if not isinstance(value["options"], list) or not value["options"]:
            raise BotLabError(f"[botlab] Parameter {pid!r} options must be a non-empty list")
        options, seen = [], set()
        for option in value["options"]:
            if not isinstance(option, dict) or set(option) != {"label", "value"}:
                raise BotLabError(f"[botlab] Parameter {pid!r} option requires only label and value")
            label = _text(option["label"], f"Parameter {pid!r} option label")
            number = _finite_number(option["value"], f"Parameter {pid!r} option value")
            if number != int(number) or abs(number) > 9007199254740991:
                raise BotLabError(f"[botlab] Parameter {pid!r} option value must be a safe integer")
            if number in seen:
                raise BotLabError(f"[botlab] Parameter {pid!r} option values must be unique")
            seen.add(number)
            options.append({"label": label, "value": number})
        if default_value not in seen:
            raise BotLabError(f"[botlab] Parameter {pid!r} default must be one of the option values")
    low = high = None
    if "range" in value:
        if not isinstance(value["range"], (list, tuple)) or len(value["range"]) != 2:
            raise BotLabError(f"[botlab] Parameter {pid!r} range must be [min, max]")
        low = _finite_number(value["range"][0], f"Parameter {pid!r} range minimum")
        high = _finite_number(value["range"][1], f"Parameter {pid!r} range maximum")
    if low is not None and high is not None and low > high:
        raise BotLabError(f"[botlab] Parameter {pid!r} min must not exceed max")
    if (low is not None and default_value < low) or (high is not None and default_value > high):
        raise BotLabError(f"[botlab] Parameter {pid!r} default is outside its range")
    step = value.get("step")
    if step is not None:
        step = _finite_number(step, f"Parameter {pid!r} step")
        if step <= 0:
            raise BotLabError(f"[botlab] Parameter {pid!r} step must be positive")
    result = {"type": "number", "default": default_value}
    if options is not None:
        result["options"] = options
    if low is not None:
        result["min"] = low
    if high is not None:
        result["max"] = high
    if step is not None:
        result["step"] = step
    if "label" in value:
        result["label"] = _text(value["label"], f"Parameter {pid!r} label")
    if "unit" in value:
        result["unit"] = _text(value["unit"], f"Parameter {pid!r} unit")
    return result


def _assert_robot_config(resource: dict) -> None:
    for field, minimum in (
        ("actionDimension", 1),
        ("jointQposOffset", 0),
        ("jointQvelOffset", 0),
    ):
        value = resource["config"].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparator = "positive" if minimum == 1 else "non-negative"
            raise BotLabError(
                f"[botlab] Robot {resource['id']!r} needs a {comparator} integer {field}"
            )


class HostSession:
    """One Apply: import controller.py, collect declarations, capture the plan."""

    def __init__(self, preset_lookup, project_dir=None):
        self._preset_lookup = preset_lookup
        # ``read_config`` reads project files by relative path; the declaration
        # session is the one place that knows the directory they live in.
        self._project_dir = None if project_dir is None else Path(project_dir)
        self._resources: list[dict] = []
        self._resource_ids: set[str] = set()
        self._parameters: dict[str, dict] = {}
        self._parameter_overrides: dict[str, float] = {}
        self._project_name: str | None = None
        self._control_hz: int | None = None
        self._physics_hz: int | None = None
        self._policy_controller_plan: dict | None = None
        self._plan_has_modifier = False
        self._declaration_closed = False

    def finish_declaration(self) -> None:
        """Close the declaration channel; keep queries alive for main.py.

        In the browser, Apply writes the manifest to the project store and
        Run executes main.py against it — resource constructors then fail
        while lookups keep working. This flag reproduces that split.
        """
        self._declaration_closed = True

    # -- collector entry points --------------------------------------------

    def declare_resource(self, kind, rid, source, preset, provider, config):
        if self._declaration_closed:
            raise BotLabError("[botlab] Resources can only be declared while a project is being applied")
        kind = _text(kind, "resource kind")
        if kind not in RESOURCE_KINDS:
            raise BotLabError(f"[botlab] Unknown resource kind {kind!r}")
        rid = _text(rid, "resource id")
        source = _resource_source(source, f"{kind} source")
        if rid in self._resource_ids:
            raise BotLabError(
                f"[botlab] Resource {rid!r} is already declared. Two resources "
                "sharing a source need distinct names: pass name=\"...\"."
            )
        if provider is not None:
            provider = _text(provider, f"{kind} provider")
            if kind != "model":
                raise BotLabError("[botlab] Only model resources accept a provider")
            if provider not in MODEL_PROVIDERS:
                raise BotLabError(
                    f"[botlab] Unknown model provider {provider!r}. Available providers: "
                    + ", ".join(sorted(MODEL_PROVIDERS)) + "."
                )
        if preset is not None:
            preset = _text(preset, f"{kind} preset")
        config = _json_value({} if config is None else config, f"{kind} config")
        if not isinstance(config, dict):
            raise BotLabError(f"[botlab] {kind} config must be a mapping")

        declared = {
            "id": rid,
            "kind": kind,
            "source": source,
        }
        if preset is not None:
            declared["preset"] = preset
        if kind == "model":
            declared["provider"] = provider if provider is not None else "auto"
        declared["config"] = dict(config)

        resolved = self._resolve_preset(declared)
        if resolved["kind"] == "robot":
            _assert_robot_config(resolved)
        self._resources.append(resolved)
        self._resource_ids.add(rid)

    def declare_project(self, name, control_hz, physics_hz, parameters):
        if self._declaration_closed:
            raise BotLabError("[botlab] project() can only be called while a project is being applied")
        if self._project_name is not None:
            raise BotLabError("[botlab] project() can only be called once")
        self._project_name = _text(name, "project name")
        if control_hz is not None:
            self._control_hz = _positive_runtime_frequency(control_hz, "project control_hz")
        if physics_hz is not None:
            self._physics_hz = _positive_runtime_frequency(physics_hz, "project physics_hz")
        if physics_hz is None or control_hz is None:
            control = self._control_hz if control_hz is not None else DEFAULT_CONTROL_HZ
            physics = self._physics_hz if physics_hz is not None else DEFAULT_PHYSICS_HZ
        else:
            control, physics = control_hz, physics_hz
        if physics % control != 0:
            raise BotLabError("[botlab] project physics_hz must be an integer multiple of control_hz")
        if not isinstance(parameters, dict):
            raise BotLabError("[botlab] project parameters must be a mapping")
        for pid, value in parameters.items():
            self._parameters[pid] = _declared_parameter(pid, value)

    def register_policy_controller(self, plan, modifier=None):
        self._policy_controller_plan = _json_value(plan, "policy controller plan")
        self._plan_has_modifier = modifier is not None

    # -- SDK runtime queries --------------------------------------------------

    def require_resource(self, rid, kind):
        for resource in self._resources:
            if resource["id"] == rid:
                if resource["kind"] != kind:
                    raise BotLabError(
                        f"[botlab] Resource {rid!r} is {resource['kind']}, expected {kind!r}"
                    )
                return resource
        raise BotLabError(f"[botlab] Unknown resource {rid!r}")

    def resource_source(self, rid):
        return self.require_resource(rid, self.require_resource(rid, self._kind_of(rid))["kind"])["source"]

    def _kind_of(self, rid):
        for resource in self._resources:
            if resource["id"] == rid:
                return resource["kind"]
        raise BotLabError(f"[botlab] Unknown resource {rid!r}")

    def action_dimension(self, rid):
        resource = self.require_resource(rid, "robot")
        value = resource["config"].get("actionDimension")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BotLabError(f"[botlab] {rid}.actionDimension must be a positive integer")
        return value

    def default_joint_positions(self, rid):
        resource = self.require_resource(rid, "robot")
        dimension = self.action_dimension(rid)
        config = resource["config"]
        if "defaultJointPositions" in config:
            values = config["defaultJointPositions"]
            if not isinstance(values, list) or len(values) != dimension:
                raise BotLabError(
                    f"[botlab] {rid}.defaultJointPositions must contain {dimension} finite numbers"
                )
            return values
        qpos_offset = config.get("jointQposOffset", 0)
        initial_qpos = config.get("initialQpos")
        if isinstance(initial_qpos, list):
            values = initial_qpos[qpos_offset:qpos_offset + dimension]
            if len(values) != dimension:
                raise BotLabError(
                    f"[botlab] {rid}.initialQpos joint state must contain {dimension} finite numbers"
                )
            return values
        raise BotLabError(f"[botlab] {rid} initial joint state is not available")

    def control_parameter(self, rid, field):
        resource = self.require_resource(rid, "robot")
        dimension = self.action_dimension(rid)
        raw = resource["config"].get(field)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return [raw] * dimension
        if isinstance(raw, list) and len(raw) == dimension:
            return raw
        raise BotLabError(f"[botlab] {rid}.{field} must contain {dimension} finite numbers")

    def parameter(self, name):
        # Overrides win over declared defaults: the runtime queries this every
        # control step, so an override applied between steps takes effect on
        # the next one — the headless counterpart of dragging a slider live.
        if name in self._parameter_overrides:
            return self._parameter_overrides[name]
        parameter = self._parameters.get(name)
        if parameter is None:
            return None
        return parameter.get("default")

    def set_parameter(self, pid, value):
        """Override a declared parameter's value, range-checked like the
        declaration. Parameters only; undeclared ids and out-of-range values
        are rejected the same way ``project()`` rejects them."""
        parameter = self._parameters.get(pid)
        if parameter is None:
            known = ", ".join(sorted(self._parameters)) or "(none declared)"
            raise BotLabError(
                f"[botlab] Parameter {pid!r} is not declared by this project: {known}"
            )
        value = _finite_number(value, f"Parameter {pid!r} value")
        if "options" in parameter and value not in {option["value"] for option in parameter["options"]}:
            raise BotLabError(f"[botlab] Parameter {pid!r} value must be one of the option values")
        low = parameter.get("min")
        high = parameter.get("max")
        if (low is not None and value < low) or (high is not None and value > high):
            bounds = []
            if low is not None:
                bounds.append(f"min {low}")
            if high is not None:
                bounds.append(f"max {high}")
            raise BotLabError(
                f"[botlab] Parameter {pid!r} value {value} is outside its range "
                f"({', '.join(bounds)})"
            )
        self._parameter_overrides[pid] = value

    def parameter_range(self, name):
        parameter = self._parameters.get(name)
        if parameter is None:
            return None
        if "min" not in parameter or "max" not in parameter:
            return None
        return [parameter["min"], parameter["max"]]

    def read_project_file(self, path):
        """The raw text of one project file — what ``read_config`` parses.

        Restricted to paths inside the project directory: no absolute paths,
        no ``..``. The SDK additionally only ever asks for ``config/*.yaml``.
        """
        name = str(path)
        if not name:
            raise BotLabError("[botlab] read_project_file needs a project-relative path")
        if Path(name).is_absolute() or ".." in name.replace("\\", "/").split("/"):
            raise BotLabError(f"[botlab] project file {name!r} must stay inside the project directory")
        if self._project_dir is None:
            raise BotLabError(
                f"[botlab] project file {name!r} needs a project directory; no declaration is active"
            )
        target = self._project_dir / name
        if not target.is_file():
            raise BotLabError(f"[botlab] project file {name!r} was not found in {self._project_dir}")
        return target.read_text(encoding="utf-8")

    def log(self, level, source, message):
        # Info-level SDK chatter (sim.run(...) started) is compile detail the
        # debug loop keeps behind --verbose; warnings and errors always print.
        import os

        if level == "info" and not os.environ.get("BOTLAB_VERBOSE"):
            return
        print(f"[{level}] [{source}] {message}")

    # -- preset resolution ----------------------------------------------------

    def _resolve_preset(self, declared):
        preset = declared.get("preset")
        if preset is None:
            return declared
        resolved = self._preset_lookup(preset)
        if resolved is None:
            raise BotLabError(f"Unknown BotLab resource preset {preset!r}.")
        preset_kind, preset_config, allow_overrides = resolved
        if preset_kind != declared["kind"]:
            raise BotLabError(
                f"BotLab resource preset {preset!r} is {preset_kind}, not {declared['kind']}."
            )
        if not allow_overrides and declared["config"]:
            raise BotLabError(
                f"BotLab resource preset {preset!r} no longer accepts config overrides; "
                "select and configure the controller in controller.py."
            )
        merged = dict(declared)
        merged["config"] = {**preset_config, **declared["config"]}
        return merged

    # -- results ----------------------------------------------------------------

    @property
    def policy_controller_plan(self):
        return self._policy_controller_plan

    @property
    def plan_has_modifier(self):
        return self._plan_has_modifier

    def project_manifest(self, fallback_name: str) -> dict:
        if not any(item["kind"] == "robot" for item in self._resources):
            raise BotLabError(
                '[botlab] A project must declare a robot: Robot("<model path>", preset="<preset>")'
            )
        return {
            "botlab": 1,
            "project": {
                "name": self._project_name or fallback_name,
                "entry": "main.py",
            },
            "runtime": {
                "language": "python",
                "simulator": "mujoco",
                "version": "3",
                "controlHz": self._control_hz if self._control_hz is not None else DEFAULT_CONTROL_HZ,
                "physicsHz": self._physics_hz if self._physics_hz is not None else DEFAULT_PHYSICS_HZ,
            },
            "resources": [dict(item, config=dict(item["config"])) for item in self._resources],
            "parameters": {pid: dict(value) for pid, value in self._parameters.items()},
            "outputs": {"variables": [], "curves": []},
        }


# The SDK module calls ``botlab_host.<fn>`` with positional args that carry JS
# shapes (None for null, etc.). Wrapping each method keeps the call sites plain.
ACTIVE: HostSession | None = None
RUN_SESSION: HostSession | None = None


def _active() -> HostSession:
    # Declaration calls must see the live collector; run-stage queries
    # (parameterRange, actionDimension, ...) resolve the detached session the
    # same way the browser bridge reads the applied manifest store.
    session = ACTIVE or RUN_SESSION
    if session is None:
        raise BotLabError("[botlab] Resources can only be declared while a project is being applied")
    return session


def begin_project_declaration(preset_lookup, project_dir=None) -> HostSession:
    global ACTIVE
    ACTIVE = HostSession(preset_lookup, project_dir)
    return ACTIVE


def finish_declaration() -> HostSession:
    """Close the declaration channel and detach the session for Run."""
    global ACTIVE, RUN_SESSION
    if ACTIVE is None:
        raise BotLabError("[botlab] No project declaration is active")
    ACTIVE.finish_declaration()
    RUN_SESSION = ACTIVE
    ACTIVE = None
    return RUN_SESSION


def end_project_declaration() -> None:
    global ACTIVE, RUN_SESSION
    ACTIVE = None
    RUN_SESSION = None


class _ProjectStateView:
    """The flat project state the native runtime drives (``botlab run``):
    ``reset(directory)`` opens a fresh declaration session, importing
    ``controller.py`` fills it, and ``manifest()``/``resource()`` read the
    collected declarations in the shape ``botlab._runtime.run`` consumes.
    """

    def __init__(self):
        self.directory = None

    def reset(self, directory=None):
        from ._project import _preset_lookup

        end_project_declaration()
        self.directory = directory
        if directory is not None:
            begin_project_declaration(_preset_lookup(Path(directory)), directory)

    def _session(self) -> HostSession:
        session = ACTIVE or RUN_SESSION
        if session is None:
            raise BotLabError("[botlab] project_state() has no declarations; reset(directory) first")
        return session

    def manifest(self):
        session = self._session()
        return {
            "name": session._project_name
            or Path(self.directory or ".").resolve().name,
            "control_hz": session._control_hz if session._control_hz is not None else DEFAULT_CONTROL_HZ,
            "physics_hz": session._physics_hz if session._physics_hz is not None else DEFAULT_PHYSICS_HZ,
            "parameters": {pid: dict(value) for pid, value in session._parameters.items()},
            "resources": [dict(item, config=dict(item["config"])) for item in session._resources],
        }

    def resource(self, resource_id, kind=None):
        session = self._session()
        rid = str(resource_id)
        item = session.require_resource(rid, session._kind_of(rid))
        if kind is not None and item["kind"] != str(kind):
            raise BotLabError(f'[botlab] resource "{rid}" is {item["kind"]}, expected {kind}')
        return item

    def set_parameter(self, name, value):
        """Override a declared parameter (headless parameter "slider").

        The runtime re-queries parameters every control step, so calling this
        between steps of a rollout changes the simulation from the next step
        on — the same live effect a key press or slider drag has in the
        browser. The declared default is untouched. Overrides live on the
        declaration session: ``HeadlessRun.reset()`` keeps them (a browser
        refresh keeps slider positions), while re-declaring the project
        (``project_state().reset(directory)``, or the next ``open_headless``)
        starts over from the declared defaults.
        """
        self._session().set_parameter(str(name), value)


_project_state = _ProjectStateView()


def project_state() -> _ProjectStateView:
    """The declarations of the project the native runtime most recently reset."""
    return _project_state


# The native runtime (`botlab run`) installs its session before the project's
# main.py runs; None keeps every runtime call on its offline error.
_runtime_session = None


def set_runtime_session(session) -> None:
    global _runtime_session
    _runtime_session = session


def _runtime_only(name, message):
    def call(*args, **kwargs):
        if _runtime_session is not None:
            return getattr(_runtime_session, name)(*args, **kwargs)
        raise BotLabError(message)

    call.__name__ = name
    return call


# -- botlab_host module surface (called by the vendored sdk/ package) ---------


def declareResource(kind, rid, source, preset, provider, config, location=None):  # noqa: N802
    # ``location`` (the declaring controller.py line) only feeds the browser
    # host's error attribution; the offline collector has no use for it.
    del location
    _active().declare_resource(kind, rid, source, preset, provider, config)


def declareProject(name, control_hz, physics_hz, parameters, location=None):  # noqa: N802
    del location
    _active().declare_project(name, control_hz, physics_hz, parameters)


def declareProjectControl(control, location=None):  # noqa: N802
    del location
    # The control declaration is a browser-pairing concern: the paired page
    # uploads it to the middleware at connect time. The native collector has
    # no page to pair with and drops it.
    del control


def requireResource(rid, kind):  # noqa: N802
    return _active().require_resource(rid, kind)


def resourceSource(rid):  # noqa: N802
    active = _active()
    return active.require_resource(rid, active._kind_of(rid))["source"]


def resourceConfig(rid, field):  # noqa: N802
    active = _active()
    return active.require_resource(rid, active._kind_of(rid))["config"].get(str(field))


def read_project_file(path):
    # The one snake_case member of the botlab_host surface: every host (browser
    # JS bridge, offline host, native collector) exposes it under this name so
    # the shared SDK's ``read_config`` call site never branches per host.
    return _active().read_project_file(path)


# List-returning queries stay plain (no JsProxy-style wrapper): the SDK's
# ``_host_value`` unwraps ``to_py`` when present and passes plain data through,
# while the native runtime reads these directly and expects Python lists.
def parameterRange(name):  # noqa: N802
    return _active().parameter_range(name)


def actionDimension(rid):  # noqa: N802
    return _active().action_dimension(rid)


def defaultJointPositions(rid):  # noqa: N802
    return _active().default_joint_positions(rid)


def controlParameter(rid, field):  # noqa: N802
    return _active().control_parameter(rid, field)


def parameter(name):  # noqa: N802
    return _active().parameter(name)


def registerPolicyController(plan, modifier=None):  # noqa: N802
    if _runtime_session is not None:
        return _runtime_session.registerPolicyController(plan, modifier)
    _active().register_policy_controller(plan, modifier)


def log(level, source, message):  # noqa: N802
    _active().log(level, source, message)


def registerReset(proxy):  # noqa: N802
    if _runtime_session is not None:
        return _runtime_session.registerReset(proxy)
    raise BotLabError("Custom Controllers require the live Python runtime; declare a PolicyController instead.")


def registerStep(proxy, robot_ids=None):  # noqa: N802
    if _runtime_session is not None:
        return _runtime_session.registerStep(proxy, robot_ids)
    raise BotLabError("Custom Controllers require the live Python runtime; declare a PolicyController instead.")


def registerSubstepObserver(callback):  # noqa: N802
    if _runtime_session is not None:
        return _runtime_session.registerSubstepObserver(callback)
    raise BotLabError(
        "on_substep is only supported by the local botlab-python runtime "
        "(botlab run --native or --headless); the browser editor and published "
        "packages do not run Python per physics substep"
    )


# Runtime calls the SDK makes from a running simulation. With a session
# installed they delegate 1:1 (the session owns the async shapes — e.g.
# predictModel returns the coroutine the SDK awaits); without one they keep
# the offline packager's hard errors.
observe = _runtime_only(
    "observe",
    "Robot.observe() requires the live simulation runtime; the offline packager never steps.",
)
observeSensor = _runtime_only(
    "observeSensor",
    "Sensor.observe() requires the live simulation runtime; the offline packager never steps.",
)
applyAction = _runtime_only(
    "applyAction",
    "Robot.apply_action() requires the live simulation runtime; the offline packager never steps.",
)
predictModel = _runtime_only(
    "predictModel",
    "Model.predict() requires the live ONNX runtime; the offline packager never infers.",
)
readMujoco = _runtime_only(
    "readMujoco",
    "Robot mujoco queries require the live simulation runtime; the offline packager never steps.",
)
loadMotion = _runtime_only(
    "loadMotion",
    "Motion.load() requires the live simulation runtime; the offline packager never loads trajectories.",
)
updateHeightfield = _runtime_only(
    "updateHeightfield",
    "Scene.update_heightfield() requires the live simulation runtime.",
)
updateMetrics = _runtime_only(
    "updateMetrics",
    "Metrics updates require the live simulation runtime.",
)
updateDiagnostics = _runtime_only(
    "updateDiagnostics",
    "Diagnostics updates require the live simulation runtime.",
)
requestReset = _runtime_only(
    "requestReset",
    "Simulation.request_reset() requires the live simulation runtime; the offline packager never resets.",
)
