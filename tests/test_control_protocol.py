"""Frozen wire contract of the shared Control protocol core (stdlib only)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from botlab.control import protocol
from botlab.control.discovery import endpoint


def envelope(**overrides) -> dict:
    body = {"request_id": "req-1", "instance_id": "inst-1", "session_id": "sess-1",
            "episode_id": "ep-1", "capabilities_revision": "rev-1",
            "capability": "motion.twist", "args": {"vx": 0.25}}
    body.update(overrides)
    return body


class TestFrozenConstants:
    def test_command_statuses_are_the_v1_wire_set(self):
        assert protocol.COMMAND_STATUSES == {"accepted", "applied", "completed",
                                             "cancelled", "failed", "expired"}

    def test_terminal_statuses_stay_a_subset_of_command_statuses(self):
        assert protocol.TERMINAL < protocol.COMMAND_STATUSES

    def test_envelope_fields_are_frozen_at_exactly_eight(self):
        assert protocol.ENVELOPE_FIELDS == {"request_id", "instance_id", "session_id",
                                            "episode_id", "capabilities_revision",
                                            "capability", "args"}
        assert protocol.IDENTITY_FIELDS == protocol.ENVELOPE_FIELDS - {"args"}

    def test_local_v1_declares_no_protocol_extensions(self):
        assert isinstance(protocol.PROTOCOL_EXTENSIONS, tuple)
        assert protocol.PROTOCOL_EXTENSIONS == ()


class TestValidateEnvelope:
    def test_complete_envelope_passes_through(self):
        body = envelope()
        assert protocol.validate_envelope(body) is body

    def test_missing_and_unknown_fields_are_rejected(self):
        with pytest.raises(protocol.ControlError) as missing:
            incomplete = envelope()
            incomplete.pop("request_id")
            protocol.validate_envelope(incomplete)
        assert missing.value.code == "invalid_parameter"
        with pytest.raises(protocol.ControlError) as unknown:
            protocol.validate_envelope(envelope(project_ref="hosted-1"))
        assert unknown.value.code == "invalid_parameter"

    @pytest.mark.parametrize("key", sorted(protocol.IDENTITY_FIELDS))
    def test_identity_fields_must_be_nonempty_strings(self, key):
        for bad in ("", None, 7, []):
            with pytest.raises(protocol.ControlError):
                protocol.validate_envelope(envelope(**{key: bad}))

    def test_request_id_is_bounded_by_the_wire_limit(self):
        with pytest.raises(protocol.ControlError) as error:
            protocol.validate_envelope(envelope(request_id="x" * (protocol.MAX_REQUEST_ID + 1)))
        assert error.value.code == "invalid_parameter"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_numbers_anywhere_in_args_are_rejected(self, value):
        with pytest.raises(protocol.ControlError):
            protocol.validate_envelope(envelope(args={"vx": value}))
        with pytest.raises(protocol.ControlError):
            protocol.validate_envelope(envelope(args={"nested": {"list": [[value]]}}))

    def test_server_integrations_raise_their_own_exception_type(self):
        class ServiceError(Exception):
            pass

        def factory(code: str, message: str) -> ServiceError:
            return ServiceError(f"{code}: {message}")

        with pytest.raises(ServiceError, match="invalid_parameter"):
            protocol.validate_envelope({}, error=factory)
        assert protocol.validate_envelope(envelope(), error=factory)


class TestEndpoint:
    @pytest.mark.parametrize("origin", [
        "http://127.0.0.1:8776", "http://localhost:8776",          # local middleware
        "https://control.example.com", "https://control.example.com:8443",
        "https://127.0.0.1:8443",                                  # loopback TLS (local testing)
    ])
    def test_accepts_loopback_http_and_any_https(self, origin):
        assert endpoint(origin) == origin

    def test_trailing_slashes_normalize_away(self):
        assert endpoint("https://control.example.com/") == "https://control.example.com"

    @pytest.mark.parametrize("origin", [
        "http://control.example.com:8443",   # plaintext never leaves the loopback
        "http://127.0.0.1",                  # local http keeps the explicit-port rule
        "https://user:secret@control.example.com",
        "https://control.example.com/runtime",
        "https://control.example.com?token=x",
        "https://control.example.com#frag",
        "https://control.example.com:99999",
        "ws://control.example.com",
        "https://",
    ])
    def test_rejects_plaintext_remote_and_structured_suffixes(self, origin):
        with pytest.raises(protocol.ControlError):
            endpoint(origin)


class TestProjectKey:
    def test_key_is_a_stable_short_digest_of_the_resolved_path(self, tmp_path: Path):
        first = protocol.project_key(tmp_path)
        assert first == protocol.project_key(tmp_path)
        assert len(first) == 16 and int(first, 16) >= 0

    def test_relative_and_absolute_references_agree(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path.parent)
        assert protocol.project_key(tmp_path.name) == protocol.project_key(tmp_path)

    def test_distinct_locations_get_distinct_keys(self, tmp_path: Path):
        assert protocol.project_key(tmp_path) != protocol.project_key(tmp_path.parent)
