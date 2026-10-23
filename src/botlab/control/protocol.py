"""Public wire constants and small, dependency-free input checks."""

from __future__ import annotations

import json
import math
from typing import Any

API_VERSION = 1
PROTOCOL = "botlab-control"
API_PREFIX = "/control/v1"
TERMINAL = frozenset({"completed", "cancelled", "failed", "expired"})
FAILED = frozenset({"cancelled", "failed", "expired"})


class ControlError(Exception):
    def __init__(self, code: str, message: str, **context: Any):
        super().__init__(message)
        self.code = code
        self.context = context

    def payload(self) -> dict:
        return {"ok": False, "error": {"code": self.code, "message": str(self)}, **self.context}


def finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ControlError("invalid_args", "JSON numbers must be finite.")
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)


def parse_json(text: str) -> Any:
    try:
        value = json.loads(text)
        finite_json(value)
        return value
    except (ValueError, RecursionError) as error:
        raise ControlError("invalid_json", "Expected valid JSON.") from error


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ControlError("invalid_identity", f"{name} must be a nonempty string (up to 256 characters).")
    return value


def validate_capabilities(payload: dict) -> list[dict]:
    if payload.get("api_version") != API_VERSION:
        raise ControlError("unsupported_version", "Unsupported Control API version.")
    identifier(payload.get("capabilities_revision"), "capabilities_revision")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ControlError("invalid_protocol", "capabilities must be an array.")
    seen = set()
    for capability in capabilities:
        if not isinstance(capability, dict):
            raise ControlError("invalid_protocol", "Each capability must be an object.")
        name = identifier(capability.get("id"), "capability.id")
        if name in seen:
            raise ControlError("invalid_protocol", "Duplicate capability ID.")
        seen.add(name)
        if capability.get("version") != 1:
            raise ControlError("unsupported_capability_version", f"Unsupported version of {name}.")
        required = ("input_schema", "completion_semantics", "units", "frame",
                    "max_duration_s", "stop_semantics")
        if any(key not in capability for key in required):
            raise ControlError("invalid_protocol", f"Capability {name} is missing required metadata.")
        if not isinstance(capability["input_schema"], dict):
            raise ControlError("invalid_protocol", f"Capability {name} has no input schema.")
        if not isinstance(capability["units"], dict):
            raise ControlError("invalid_protocol", f"Capability {name} must declare parameter units.")
        if capability["frame"] is not None and not isinstance(capability["frame"], str):
            raise ControlError("invalid_protocol", f"Capability {name} has an invalid frame.")
        for field in ("completion_semantics", "stop_semantics"):
            if not isinstance(capability[field], str) or not capability[field]:
                raise ControlError("invalid_protocol", f"Capability {name} has no {field}.")
        duration = capability["max_duration_s"]
        if duration is not None and (isinstance(duration, bool)
                                     or not isinstance(duration, (float, int)) or duration <= 0):
            raise ControlError("invalid_protocol", f"Capability {name} has invalid duration metadata.")
        aliases = capability.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ControlError("invalid_protocol", f"Capability {name} has invalid aliases.")
    return capabilities


def validate_args(value: Any, schema: dict, path: str = "args") -> None:
    """Check common schema constraints; middleware remains the authoritative validator."""
    try:
        _validate_args(value, schema, path)
    except (TypeError, AttributeError, RecursionError) as error:
        raise ControlError("invalid_protocol", "Middleware supplied a malformed input schema.") from error


def _validate_args(value: Any, schema: dict, path: str) -> None:
    finite_json(value)
    types = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool), "null": value is None,
    }
    kind = schema.get("type")
    if isinstance(kind, str) and kind in types and not types[kind]:
        raise ControlError("invalid_args", f"{path} must have type {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ControlError("invalid_args", f"{path} is not an allowed value.")
    if "const" in schema and value != schema["const"]:
        raise ControlError("invalid_args", f"{path} does not match the required value.")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ControlError("invalid_args", f"{path}.{key} is required.")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ControlError("invalid_args", f"{path} contains unknown fields.")
        for key, child in value.items():
            if key in properties:
                _validate_args(child, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ControlError("invalid_args", f"{path} has the wrong number of items.")
        if isinstance(schema.get("items"), dict):
            for index, child in enumerate(value):
                _validate_args(child, schema["items"], f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ControlError("invalid_args", f"{path} is outside the declared range.")
        if value <= schema.get("exclusiveMinimum", -math.inf) or value >= schema.get("exclusiveMaximum", math.inf):
            raise ControlError("invalid_args", f"{path} is outside the declared range.")
