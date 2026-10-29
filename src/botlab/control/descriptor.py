"""Project control descriptors: the page declares semantics at pairing time.

A descriptor turns the middleware project-agnostic: the browser page owns the
project (manifest + compiled plan) and uploads its control surface — capability
tables, FSM actions, signal mapping and the experience document — right after
pairing. The middleware remains the wire-semantics authority: it only checks
structure here and compiles declared semantics into the same adapter surface
microduck's hand-written adapter implements (capabilities / from_capability /
normalize / prepare / decode_state / observations).

Structural validation only. Cross-checks against the actual project (the robot
exists, the state node is a proc-state-machine, parameters are consumed) stay
on the page, which owns the manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .protocol import API_VERSION, ControlError, finite_json, identifier

DESCRIPTOR_VERSION = 1
MAX_EXPERIENCE_BYTES = 32768
MAX_STATE_CODE = 2**31 - 1
MIN_DURATION_S = 0.05
MAX_DURATION_S = 10.0
MOVE_TTL_MS = 500
EVENT_TIMEOUT_MS = 10000

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
RESERVED_CAPABILITIES = frozenset({"motion.twist", "control.stop"})

DESCRIPTOR_KEYS = frozenset({"v", "type", "profile", "motion", "actions", "states", "experience"})
AXIS_KEYS = frozenset({"vx", "yaw_rate"})
ACTION_KEYS = frozenset({"name", "event_value", "allowed_states", "target_states",
                         "capability", "idempotent"})

MOTION_COMPLETION = ("Bounded wall-clock motion ended and zero input was physically "
                     "committed; not distance arrival.")
ACTION_COMPLETION = "Target FSM state observed after physics commit; not proof of settled posture."
STOP_COMPLETION = "Zero motion confirmed by a committed physics snapshot."
STOP_SEMANTICS = "Zero motion and clear pending events; applied FSM actions may continue."


def _text(value, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ControlError("invalid_descriptor", f"{name} must be a nonempty string (up to {limit} characters)")
    return value


def _state_list(value, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ControlError("invalid_descriptor", f"{name} must be a nonempty integer array")
    for code in value:
        if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= MAX_STATE_CODE:
            raise ControlError("invalid_descriptor", f"{name} codes must be integers in 0..{MAX_STATE_CODE}")
    if len(set(value)) != len(value):
        raise ControlError("invalid_descriptor", f"{name} must not repeat codes")
    return list(value)


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ControlError("invalid_descriptor", f"{name} must be a finite number")
    return float(value)


def _axis(value, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"min", "max", "unit"}:
        raise ControlError("invalid_descriptor", f"motion.{name} must declare min, max and unit")
    low, high = _finite(value["min"], f"motion.{name}.min"), _finite(value["max"], f"motion.{name}.max")
    if not low < high:
        raise ControlError("invalid_descriptor", f"motion.{name} requires min < max")
    _text(value["unit"], f"motion.{name}.unit", 32)
    return {"min": low, "max": high, "unit": value["unit"]}


def validate_descriptor(payload) -> dict:
    """Structurally validate a project control descriptor; returns it unchanged."""
    if not isinstance(payload, dict) or set(payload) - DESCRIPTOR_KEYS:
        raise ControlError("invalid_descriptor", "Descriptor must be an object with known fields only")
    if payload.get("v") != DESCRIPTOR_VERSION or payload.get("type") != "descriptor":
        raise ControlError("invalid_descriptor", "Expected a v1 descriptor frame")
    try:
        finite_json(payload)
    except ControlError as error:
        raise ControlError("invalid_descriptor", "Descriptor must contain finite JSON values") from error

    profile = payload.get("profile")
    if not isinstance(profile, dict) or set(profile) != {
            "robot_id", "state_node_id", "motion_parameters", "event_parameter", "event_values"}:
        raise ControlError("invalid_descriptor", "profile must declare the full browser control profile")
    identifier(profile["robot_id"], "profile.robot_id")
    identifier(profile["state_node_id"], "profile.state_node_id")
    parameters = profile["motion_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != AXIS_KEYS:
        raise ControlError("invalid_descriptor", "motion_parameters must declare exactly vx and yaw_rate")
    names = [identifier(parameters[axis], f"motion_parameters.{axis}") for axis in sorted(AXIS_KEYS)]
    names.append(identifier(profile["event_parameter"], "profile.event_parameter"))
    if len(set(names)) != len(names):
        raise ControlError("invalid_descriptor", "motion and event parameter names must be distinct")
    values = profile["event_values"]
    if not isinstance(values, list) or not values:
        raise ControlError("invalid_descriptor", "event_values must be a nonempty integer array")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_STATE_CODE:
            raise ControlError("invalid_descriptor", "event_values must be nonzero integers")
    if len(set(values)) != len(values):
        raise ControlError("invalid_descriptor", "event_values must not repeat")

    motion = payload.get("motion")
    if not isinstance(motion, dict) or set(motion) != AXIS_KEYS | {"allowed_states"}:
        raise ControlError("invalid_descriptor", "motion must declare vx, yaw_rate and allowed_states")
    motion["vx"] = _axis(motion["vx"], "vx")
    motion["yaw_rate"] = _axis(motion["yaw_rate"], "yaw_rate")
    _state_list(motion["allowed_states"], "motion.allowed_states")

    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ControlError("invalid_descriptor", "actions must be an array")
    seen_names, seen_capabilities = set(), set()
    for action in actions:
        if not isinstance(action, dict) or set(action) - ACTION_KEYS \
                or not {"name", "event_value", "allowed_states", "target_states"} <= set(action):
            raise ControlError("invalid_descriptor", "Each action must declare name, event_value, allowed_states, target_states")
        name = action["name"]
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise ControlError("invalid_descriptor", f"Invalid action name: {name!r}")
        if name in seen_names:
            raise ControlError("invalid_descriptor", f"Duplicate action name: {name}")
        seen_names.add(name)
        value = action["event_value"]
        if isinstance(value, bool) or not isinstance(value, int) or value not in profile["event_values"]:
            raise ControlError("invalid_descriptor", f"action {name}: event_value must appear in profile.event_values")
        # Two actions may share an event code (a toggle): the wire distinguishes
        # them by allowed_states, so only names and capability ids must be unique.
        _state_list(action["allowed_states"], f"action {name}.allowed_states")
        _state_list(action["target_states"], f"action {name}.target_states")
        capability = action.get("capability", f"action.{name}")
        if not isinstance(capability, str) or not CAPABILITY_PATTERN.match(capability) \
                or capability in RESERVED_CAPABILITIES:
            raise ControlError("invalid_descriptor", f"action {name}: invalid capability id {capability!r}")
        if capability in seen_capabilities:
            raise ControlError("invalid_descriptor", f"Duplicate capability id: {capability}")
        seen_capabilities.add(capability)
        if "idempotent" in action and not isinstance(action["idempotent"], bool):
            raise ControlError("invalid_descriptor", f"action {name}: idempotent must be a boolean")
        action["capability"] = capability
        action.setdefault("idempotent", False)

    states = payload.get("states")
    if states is not None:
        if not isinstance(states, dict) or not states:
            raise ControlError("invalid_descriptor", "states must map state codes to names")
        for code, label in states.items():
            if not isinstance(code, str) or not code.isdigit() \
                    or not 0 <= int(code) <= MAX_STATE_CODE or str(int(code)) != code:
                raise ControlError("invalid_descriptor", "states keys must be canonical decimal state codes")
            _text(label, f"states[{code}]", 64)

    experience = payload.get("experience")
    if experience is not None:
        if not isinstance(experience, str):
            raise ControlError("invalid_descriptor", "experience must be a string")
        if len(experience.encode("utf-8")) > MAX_EXPERIENCE_BYTES:
            raise ControlError("descriptor_too_large",
                               f"experience exceeds {MAX_EXPERIENCE_BYTES} bytes")
    return payload


def revision_of(descriptor: dict) -> str:
    """Conservative revision: any descriptor change forces clients to re-read capabilities."""
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"gen-{digest[:16]}"


class DescriptorEngine:
    """Adapter surface compiled from a validated descriptor (microduck parity)."""

    def __init__(self, descriptor: dict):
        self.descriptor = descriptor
        self.profile = descriptor["profile"]
        self.motion = descriptor["motion"]
        self.actions = {action["name"]: action for action in descriptor["actions"]}
        self.by_capability = {action["capability"]: action for action in descriptor["actions"]}
        self.states = {int(code): name for code, name in (descriptor.get("states") or {}).items()}
        self.experience = descriptor.get("experience")
        self.revision = revision_of(descriptor)
        self._capabilities = self._compile_capabilities()

    def _compile_capabilities(self) -> list[dict]:
        entries = [{
            "id": "motion.twist", "version": 1, "aliases": ["move"],
            "input_schema": {"type": "object", "additionalProperties": False,
                "properties": {
                    "vx": {"type": "number", "minimum": self.motion["vx"]["min"],
                           "maximum": self.motion["vx"]["max"], "default": 0},
                    "vy": {"type": "number", "const": 0, "default": 0},
                    "yaw_rate": {"type": "number", "minimum": self.motion["yaw_rate"]["min"],
                                 "maximum": self.motion["yaw_rate"]["max"], "default": 0},
                    "duration_s": {"type": "number", "minimum": MIN_DURATION_S, "maximum": MAX_DURATION_S},
                }, "required": ["duration_s"]},
            "units": {"vx": self.motion["vx"]["unit"], "vy": "m/s",
                      "yaw_rate": self.motion["yaw_rate"]["unit"], "duration_s": "s"},
            "frame": "body", "max_duration_s": MAX_DURATION_S,
            "completion_semantics": MOTION_COMPLETION, "stop_semantics": STOP_SEMANTICS,
        }]
        for action in self.actions.values():
            entries.append({
                "id": action["capability"], "version": 1, "aliases": [action["name"]],
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                "units": {}, "frame": None, "max_duration_s": MAX_DURATION_S,
                "completion_semantics": ACTION_COMPLETION, "stop_semantics": STOP_SEMANTICS,
                "preconditions": {"allowed_state_codes": action["allowed_states"],
                                  "target_state_codes": action["target_states"]},
            })
        entries.append({
            "id": "control.stop", "version": 1, "aliases": ["stop"],
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "units": {}, "frame": None, "max_duration_s": 15,
            "completion_semantics": STOP_COMPLETION, "stop_semantics": STOP_SEMANTICS,
        })
        # Self-check: the compiled table must satisfy the frozen capability contract.
        from .protocol import validate_capabilities
        validate_capabilities({"api_version": API_VERSION, "capabilities_revision": self.revision,
                               "capabilities": entries})
        return entries

    def capabilities(self) -> list[dict]:
        return json.loads(json.dumps(self._capabilities))

    def _bounded(self, value, name: str, axis: dict) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ControlError("invalid_parameter", f"{name} must be a number")
        number = float(value)
        if not axis["min"] <= number <= axis["max"]:
            raise ControlError("out_of_range",
                               f"{name} must be in [{axis['min']}, {axis['max']}]")
        return number

    def normalize(self, body: dict) -> dict:
        kind = body.get("kind")
        if kind == "move":
            return {"kind": kind,
                    "vx": self._bounded(body.get("vx", 0), "vx", self.motion["vx"]),
                    "vy": 0.0,
                    "yaw_rate": self._bounded(body.get("yaw_rate", 0), "yaw_rate", self.motion["yaw_rate"]),
                    "duration_s": float(_duration(body.get("duration_s")))}
        if kind == "action":
            name = body.get("name")
            if not isinstance(name, str) or name not in self.actions:
                raise ControlError("invalid_action",
                                   f"Action must be one of: {', '.join(sorted(self.actions))}")
            return {"kind": kind, "name": name}
        if kind == "stop":
            return {"kind": kind}
        raise ControlError("invalid_request", "kind must be move, action or stop")

    def from_capability(self, capability: str, args) -> dict:
        if not isinstance(args, dict):
            raise ControlError("invalid_parameter", "args must be a JSON object")
        if capability == "motion.twist":
            if set(args) - {"vx", "vy", "yaw_rate", "duration_s"}:
                raise ControlError("invalid_parameter", "Unknown motion argument")
            return self.normalize({"kind": "move", **args})
        action = self.by_capability.get(capability)
        if action is None and capability != "control.stop":
            raise ControlError("unsupported_capability", f"Unsupported capability: {capability}")
        if args:
            raise ControlError("invalid_parameter", "This capability takes no arguments")
        return {"kind": "action", "name": action["name"]} if action else {"kind": "stop"}

    def public_command(self, command: dict) -> tuple[str, dict]:
        if command["kind"] == "move":
            return "motion.twist", {key: command[key] for key in ("vx", "vy", "yaw_rate", "duration_s")}
        if command["kind"] == "action":
            return self.actions[command["name"]]["capability"], {}
        return "control.stop", {}

    def _state_name(self, code) -> str:
        return self.states.get(code, "UNKNOWN")

    def prepare(self, command: dict, state: dict) -> tuple[dict, float, bool]:
        kind = command["kind"]
        if kind == "move":
            code = state.get("policy_state_code")
            if code not in self.motion["allowed_states"]:
                raise ControlError("invalid_state",
                                   f"Movement requires {self._state_list_label(self.motion['allowed_states'])}")
            return ({"kind": "move",
                     "values": {self.profile["motion_parameters"]["vx"]: command["vx"],
                                self.profile["motion_parameters"]["yaw_rate"]: command["yaw_rate"]},
                     "duration_ms": command["duration_s"] * 1000, "ttl_ms": MOVE_TTL_MS,
                     "allowed_states": list(self.motion["allowed_states"])},
                    command["duration_s"] + 10, False)
        if kind == "action":
            action = self.actions[command["name"]]
            code = state.get("policy_state_code")
            satisfied = action["idempotent"] and code in action["target_states"]
            if not satisfied and code not in action["allowed_states"]:
                raise ControlError("invalid_state",
                                   f"{command['name']} is not allowed from {self._state_name(code)}")
            return ({"kind": "event", "event_value": action["event_value"],
                     "allowed_states": list(action["allowed_states"]),
                     "target_states": list(action["target_states"]),
                     "timeout_ms": EVENT_TIMEOUT_MS}, 15, satisfied)
        return {"kind": "stop"}, 15, False

    def _state_list_label(self, codes: list[int]) -> str:
        return " or ".join(self._state_name(code) for code in codes)

    def state_available(self, snapshot: dict) -> bool:
        return type(snapshot.get("policy_state_code")) is int

    def decode_state(self, snapshot: dict) -> dict:
        result = dict(snapshot)
        result["policy_state"] = self._state_name(result.get("policy_state_code"))
        return result

    def observations(self, state: dict) -> dict:
        return {
            "base_pose": {"position_m": state.get("position_world_m"),
                          "orientation_wxyz": state.get("orientation_wxyz"), "frame": "world"},
            "base_velocity": {"values": state.get("base_velocity"),
                              "layout": "MuJoCo free-joint qvel: world linear xyz then body angular xyz",
                              "units": ["m/s"] * 3 + ["rad/s"] * 3},
            "policy": {"code": state.get("policy_state_code"), "name": state.get("policy_state"),
                       "steps": state.get("policy_state_steps")},
            "effective_parameters": dict(state.get("effective_parameters", {})),
        }


def _duration(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not MIN_DURATION_S <= value <= MAX_DURATION_S:
        raise ControlError("out_of_range", f"duration_s must be in [{MIN_DURATION_S}, {MAX_DURATION_S}]")
    return float(value)
