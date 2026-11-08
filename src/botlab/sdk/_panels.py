"""Validate project display metadata once, independently of control execution."""
import re

_TELEMETRY_KEYS = ("baseSpeed", "targetSpeed", "roll", "pitch", "yaw", "torquePeak", "freqHz", "realtimeFactor", "distance", "trackingError")
_RUNTIME_KEYS = ("controlFrequency", "physicsFrequency", "realtimeFactor", "controlP95", "deadlineMiss", "fps", "gpuFrameLoadPct")
_TILE_IDS = ("view:attitude", "view:actuatorForces", "status:inferenceProvider")
_ARRAY_SOURCES = ("qpos", "qvel", "ctrl", "sensordata", "refq", "errq")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INDEX = re.compile(r"(0|[1-9][0-9]*)\Z")
_MAX_SAFE_INTEGER = 2 ** 53 - 1


def _fail(path, message):
    raise ValueError(f"{path}: {message}")


def _mapping(value, path):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(path, "must be a mapping with string keys")
    return value


def _fields(value, allowed, path):
    record = _mapping(value, path)
    for key in record:
        if key not in allowed:
            _fail(path, f'unknown key "{key}" (expected {", ".join(allowed)})')
    return record


def _label(value, path):
    if isinstance(value, str):
        if not value.strip():
            _fail(path, "must be a non-empty string")
        return value.strip()
    labels = _fields(value, ("en", "zh"), path)
    if "en" not in labels:
        _fail(path, 'must include an "en" entry')
    result = {}
    for key, text in labels.items():
        if not isinstance(text, str) or not text.strip():
            _fail(path, f'"{key}" must be a non-empty string')
        result[key] = text.strip()
    return result


def _ids(value, path, *, tiles=False):
    if not isinstance(value, list):
        _fail(path, "must be a list of variable ids")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            _fail(path, "entries must be non-empty strings")
        item = item.strip()
        if item in result:
            _fail(path, f'lists "{item}" twice')
        source, _, key = item.partition(":")
        valid = (
            tiles and item in _TILE_IDS
            or source == "telemetry" and key in _TELEMETRY_KEYS
            or source == "runtime" and key in _RUNTIME_KEYS
            or source == "metric" and _IDENTIFIER.fullmatch(key)
            or source in _ARRAY_SOURCES and _INDEX.fullmatch(key) and int(key) <= _MAX_SAFE_INTEGER
        )
        if not valid:
            _fail(path, f'invalid channel id "{item}"')
        result.append(item)
    return result


def _metric(value, path):
    record = _fields(value, ("label", "unit", "digits", "values"), path)
    if "label" not in record:
        _fail(path, "label is required")
    result = {"label": _label(record["label"], f"{path}.label")}
    if "unit" in record:
        unit = record["unit"]
        if not isinstance(unit, str) or not unit.strip():
            _fail(path, "unit must be a non-empty string")
        result["unit"] = unit.strip()
    if "digits" in record:
        digits = record["digits"]
        if isinstance(digits, bool) or not isinstance(digits, (int, float)) or not 0 <= digits <= 9 or int(digits) != digits:
            _fail(path, "digits must be an integer between 0 and 9")
        result["digits"] = int(digits)
    if "values" in record:
        values = _mapping(record["values"], f"{path}.values")
        if not values:
            _fail(path, "values must be a non-empty mapping")
        result["values"] = {}
        for code, label in values.items():
            try:
                number = int(code)
            except ValueError:
                _fail(path, f'values key "{code}" must be a canonical safe integer state code')
            if str(number) != code or abs(number) > _MAX_SAFE_INTEGER:
                _fail(path, f'values key "{code}" must be a canonical safe integer state code')
            result["values"][code] = _label(label, f'{path}.values["{code}"]')
    return result


def _panel(value, path, *, metrics=False):
    if value is None:
        value = {}
    path = getattr(value, "_source_path", path)
    record = _fields(value, ("candidates", "tiles", "metrics") if metrics else ("candidates", "watch"), path)
    selection_key = "tiles" if metrics else "watch"
    candidates = None
    if "candidates" in record and record["candidates"] != "auto":
        candidates = _ids(record["candidates"], f"{path}.candidates", tiles=metrics)
    selection = _ids(record[selection_key], f"{path}.{selection_key}", tiles=metrics) if selection_key in record else None
    if candidates is not None and selection is not None:
        for item in selection:
            if item not in candidates:
                _fail(path, f'{selection_key} id "{item}" is not listed in candidates')
    result = {selection_key: selection, "tileCandidates" if metrics else "watchCandidates": candidates}
    if metrics:
        result["metrics"] = {}
        if "metrics" in record:
            for key, entry in _mapping(record["metrics"], f"{path}.metrics").items():
                if not _IDENTIFIER.fullmatch(key):
                    _fail(path, f'metrics key "{key}" must be a Python identifier')
                result["metrics"][key] = _metric(entry, f'{path}.metrics["{key}"]')
    return result


def normalize_panels(value):
    """Resolve explicit ``project(panels=...)``; never discover files by name."""
    path = getattr(value, "_source_path", "project.panels")
    record = _fields(value, ("watch", "metrics"), path)
    return {
        **_panel(record.get("watch"), f"{path}.watch"),
        **_panel(record.get("metrics"), f"{path}.metrics", metrics=True),
    }


def validate_resolved_panels(value):
    """Return a validated copy of serialized panels; absent metadata stays absent."""
    keys = ("watch", "watchCandidates", "tiles", "tileCandidates", "metrics")
    record = _fields(value, keys, "project.panels")
    for key in keys:
        if key not in record:
            _fail("project.panels", f'missing field "{key}"')
    watch = {}
    metrics = {"metrics": record["metrics"]}
    for source, target, section in (
        ("watch", "watch", watch), ("watchCandidates", "candidates", watch),
        ("tiles", "tiles", metrics), ("tileCandidates", "candidates", metrics),
    ):
        if record[source] is not None:
            section[target] = record[source]
    return normalize_panels({"watch": watch, "metrics": metrics})
