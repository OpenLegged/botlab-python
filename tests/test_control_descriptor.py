"""Descriptor validation and the compiled engine, microduck-equivalent fixture."""
from __future__ import annotations

import copy
import json

import pytest

from botlab.control.descriptor import (
    MAX_EXPERIENCE_BYTES,
    DescriptorEngine,
    revision_of,
    validate_descriptor,
)
from botlab.control.protocol import ControlError, validate_capabilities


def sample() -> dict:
    """The microduck control surface as a page would declare it."""
    return {
        "v": 1, "type": "descriptor",
        "profile": {"robot_id": "MicroDuck", "state_node_id": "state-machine",
                    "motion_parameters": {"vx": "forward_velocity", "yaw_rate": "yaw_rate"},
                    "event_parameter": "policy_command", "event_values": [1, 2, 3, 4, 5, 6]},
        "motion": {"vx": {"min": -0.2, "max": 0.25, "unit": "m/s"},
                   "yaw_rate": {"min": -1, "max": 1, "unit": "rad/s"},
                   "allowed_states": [0]},
        "actions": [
            {"name": "sit", "event_value": 1, "allowed_states": [0], "target_states": [2],
             "capability": "posture.sit", "idempotent": True},
            {"name": "stand_up", "event_value": 1, "allowed_states": [2], "target_states": [0],
             "capability": "posture.stand_up", "idempotent": True},
            {"name": "kick_left", "event_value": 2, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.kick_left"},
            {"name": "kick_right", "event_value": 3, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.kick_right"},
            {"name": "roll", "event_value": 4, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.roll"},
            {"name": "ground_pick", "event_value": 5, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.ground_pick"},
            {"name": "recover", "event_value": 6, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.recover"},
        ],
        "states": {"0": "WALK", "1": "SIT_ENTER", "2": "SIT", "3": "SIT_EXIT", "4": "KICK_LEFT",
                   "5": "KICK_RIGHT", "6": "POST_KICK", "7": "ROLL", "8": "ROLL_TIPPED",
                   "9": "GROUND_PICK", "10": "WALK_RESET", "11": "RECOVER_ENTER",
                   "12": "RECOVER", "13": "RECOVER_UPRIGHT"},
        "experience": "# MicroDuck control experience\n\nWalk threshold vx=0.25.",
    }


def mutate(**overrides):
    payload = sample()
    payload.update(overrides)
    return payload


class TestValidateDescriptor:
    def test_sample_is_valid(self):
        assert validate_descriptor(sample()) is not None

    def test_unknown_top_level_field_is_rejected(self):
        with pytest.raises(ControlError, match="known fields"):
            validate_descriptor(mutate(project_ref="hosted"))

    def test_wrong_version_or_type_is_rejected(self):
        with pytest.raises(ControlError):
            validate_descriptor(mutate(v=2))
        with pytest.raises(ControlError):
            validate_descriptor(mutate(type="state"))

    def test_non_finite_values_are_rejected(self):
        payload = sample()
        payload["motion"]["vx"]["min"] = float("nan")
        with pytest.raises(ControlError, match="finite"):
            validate_descriptor(payload)

    def test_profile_must_be_complete(self):
        for field in ("robot_id", "state_node_id", "motion_parameters", "event_parameter", "event_values"):
            payload = sample()
            del payload["profile"][field]
            with pytest.raises(ControlError):
                validate_descriptor(payload)

    def test_motion_parameters_must_be_exactly_two_distinct_axes(self):
        payload = sample()
        payload["profile"]["motion_parameters"]["vy"] = "lateral_velocity"
        with pytest.raises(ControlError):
            validate_descriptor(payload)
        payload = sample()
        payload["profile"]["motion_parameters"]["vx"] = payload["profile"]["motion_parameters"]["yaw_rate"]
        with pytest.raises(ControlError, match="distinct"):
            validate_descriptor(payload)

    def test_event_values_must_be_nonzero_unique_integers(self):
        for bad in ([], [0], [1, 1], ["1"], [True]):
            payload = sample()
            payload["profile"]["event_values"] = bad
            with pytest.raises(ControlError):
                validate_descriptor(payload)

    def test_axis_bounds_must_bound_a_range(self):
        payload = sample()
        payload["motion"]["vx"]["min"] = 0.25
        with pytest.raises(ControlError, match="min < max"):
            validate_descriptor(payload)
        payload = sample()
        payload["motion"]["yaw_rate"]["unit"] = ""
        with pytest.raises(ControlError):
            validate_descriptor(payload)

    def test_state_lists_must_be_nonempty_unique_codes(self):
        for bad in ([], [0, 0], [-1], ["0"], [True]):
            payload = sample()
            payload["motion"]["allowed_states"] = bad
            with pytest.raises(ControlError):
                validate_descriptor(payload)

    def test_toggle_actions_may_share_an_event_value(self):
        assert validate_descriptor(sample())  # sit and stand_up both use event 1

    def test_action_names_and_capabilities_must_be_unique(self):
        payload = sample()
        payload["actions"][2]["name"] = "sit"
        with pytest.raises(ControlError, match="Duplicate action name"):
            validate_descriptor(payload)
        payload = sample()
        payload["actions"][2]["capability"] = "posture.sit"
        with pytest.raises(ControlError, match="Duplicate capability id"):
            validate_descriptor(payload)

    def test_reserved_capability_ids_are_rejected(self):
        payload = sample()
        payload["actions"][2]["capability"] = "motion.twist"
        with pytest.raises(ControlError, match="invalid capability id"):
            validate_descriptor(payload)

    def test_action_event_value_must_be_declared(self):
        payload = sample()
        payload["actions"][2]["event_value"] = 9
        with pytest.raises(ControlError, match="profile.event_values"):
            validate_descriptor(payload)

    def test_default_capability_and_idempotent_are_filled_in(self):
        payload = mutate(actions=[{"name": "wave", "event_value": 1,
                                   "allowed_states": [0], "target_states": [1]}],
                         states={"0": "WALK", "1": "WAVE"})
        validated = validate_descriptor(payload)
        assert validated["actions"][0]["capability"] == "action.wave"
        assert validated["actions"][0]["idempotent"] is False

    def test_states_keys_must_be_canonical_codes(self):
        for bad in ({"00": "WALK"}, {"-1": "NEG"}, {"x": "LABEL"}, {"0": ""}):
            payload = sample()
            payload["states"] = bad
            with pytest.raises(ControlError):
                validate_descriptor(payload)

    def test_oversized_experience_is_rejected(self):
        payload = sample()
        payload["experience"] = "x" * (MAX_EXPERIENCE_BYTES + 1)
        with pytest.raises(ControlError, match="exceeds"):
            validate_descriptor(payload)


class TestRevision:
    def test_revision_is_stable_and_descriptor_sensitive(self):
        first = revision_of(validate_descriptor(sample()))
        assert first.startswith("gen-") and len(first) == 20
        assert revision_of(validate_descriptor(sample())) == first
        changed = sample()
        changed["experience"] = "different"
        assert revision_of(validate_descriptor(changed)) != first


class TestDescriptorEngine:
    def engine(self) -> DescriptorEngine:
        return DescriptorEngine(validate_descriptor(sample()))

    def test_capabilities_satisfy_the_frozen_contract(self):
        engine = self.engine()
        validate_capabilities({"api_version": 1, "capabilities_revision": engine.revision,
                               "capabilities": engine.capabilities()})

    def test_capability_surface(self):
        capabilities = {entry["id"]: entry for entry in self.engine().capabilities()}
        assert set(capabilities) == {"motion.twist", "control.stop", "posture.sit",
                                     "posture.stand_up", "behavior.kick_left",
                                     "behavior.kick_right", "behavior.roll",
                                     "behavior.ground_pick", "behavior.recover"}
        twist = capabilities["motion.twist"]
        assert twist["input_schema"]["properties"]["vx"]["minimum"] == -0.2
        assert twist["input_schema"]["properties"]["vx"]["maximum"] == 0.25
        assert twist["input_schema"]["properties"]["vy"] == {"type": "number", "const": 0, "default": 0}
        assert twist["units"]["vx"] == "m/s" and twist["frame"] == "body"
        assert capabilities["control.stop"]["aliases"] == ["stop"]
        assert capabilities["behavior.kick_left"]["aliases"] == ["kick_left"]
        assert capabilities["behavior.kick_left"]["preconditions"] == {
            "allowed_state_codes": [0], "target_state_codes": [0]}

    def test_normalize_bounds_and_roundtrip(self):
        engine = self.engine()
        command = engine.normalize({"kind": "move", "vx": 0.2, "yaw_rate": -0.5, "duration_s": 3})
        assert command == {"kind": "move", "vx": 0.2, "vy": 0.0, "yaw_rate": -0.5, "duration_s": 3.0}
        assert engine.public_command(command) == (
            "motion.twist", {"vx": 0.2, "vy": 0.0, "yaw_rate": -0.5, "duration_s": 3.0})
        with pytest.raises(ControlError, match="vx must be in"):
            engine.normalize({"kind": "move", "vx": 0.5, "duration_s": 1})
        with pytest.raises(ControlError):
            engine.normalize({"kind": "move", "duration_s": 20})
        with pytest.raises(ControlError):
            engine.normalize({"kind": "action", "name": "fly"})
        assert engine.normalize({"kind": "stop"}) == {"kind": "stop"}

    def test_from_capability_resolves_declared_ids(self):
        engine = self.engine()
        assert engine.from_capability("posture.sit", {}) == {"kind": "action", "name": "sit"}
        assert engine.from_capability("motion.twist", {"vx": 0.1, "duration_s": 1})["kind"] == "move"
        assert engine.from_capability("control.stop", {}) == {"kind": "stop"}
        with pytest.raises(ControlError):
            engine.from_capability("nav.goto", {})

    def test_prepare_compiles_the_wire_payload(self):
        engine = self.engine()
        payload, timeout, satisfied = engine.prepare(
            {"kind": "move", "vx": 0.15, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 2.0},
            {"policy_state_code": 0})
        assert payload == {"kind": "move", "values": {"forward_velocity": 0.15, "yaw_rate": 0.0},
                           "duration_ms": 2000.0, "ttl_ms": 500, "allowed_states": [0]}
        assert timeout == pytest.approx(12.0)
        assert satisfied is False
        with pytest.raises(ControlError, match="Movement requires"):
            engine.prepare({"kind": "move", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration_s": 1},
                           {"policy_state_code": 2})

    def test_prepare_action_semantics(self):
        engine = self.engine()
        payload, timeout, satisfied = engine.prepare({"kind": "action", "name": "sit"},
                                                     {"policy_state_code": 0})
        assert payload["event_value"] == 1 and payload["timeout_ms"] == 10000
        assert satisfied is False
        _, _, already = engine.prepare({"kind": "action", "name": "sit"}, {"policy_state_code": 2})
        assert already is True  # idempotent sit from SIT
        # stand_up from WALK is already satisfied (idempotent, target=WALK); a
        # non-idempotent action outside its allowed set must be refused instead.
        assert engine.prepare({"kind": "action", "name": "stand_up"}, {"policy_state_code": 0})[2] is True
        with pytest.raises(ControlError, match="not allowed from"):
            engine.prepare({"kind": "action", "name": "kick_left"}, {"policy_state_code": 2})

    def test_decode_and_observations(self):
        engine = self.engine()
        snapshot = {"policy_state_code": 2, "policy_state_steps": 7, "position_world_m": [1, 2, 3],
                    "orientation_wxyz": [1, 0, 0, 0], "base_velocity": [0] * 6,
                    "effective_parameters": {"forward_velocity": 0.1}}
        state = engine.decode_state(snapshot)
        assert state["policy_state"] == "SIT"
        observations = engine.observations(state)
        assert observations["base_pose"]["frame"] == "world"
        assert observations["policy"] == {"code": 2, "name": "SIT", "steps": 7}
        assert engine.decode_state({"policy_state_code": 99})["policy_state"] == "UNKNOWN"
        assert engine.state_available({"policy_state_code": 0}) is True
        assert engine.state_available({}) is False

    def test_deep_copies_keep_the_descriptor_frozen(self):
        engine = self.engine()
        frozen = copy.deepcopy(engine.capabilities())
        engine.capabilities()[0]["input_schema"]["properties"]["vx"]["minimum"] = 99
        assert engine.capabilities() == frozen
        json.dumps(engine.capabilities())  # serializable
