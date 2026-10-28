"""Host bridge, project and resource declarations, preset calibration handles."""
import inspect
import math
import os

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


def project(name, *, control_hz=None, physics_hz=None, parameters=None):
    """Name the project and override the runtime defaults.

    Optional: a project that keeps the documented defaults (50 Hz control,
    200 Hz physics) and declares no tunable parameters never has to call this.
    """
    botlab_host.declareProject(
        name,
        control_hz,
        physics_hz,
        _plan_value({} if parameters is None else parameters, "project parameters"),
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
