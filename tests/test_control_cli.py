"""Control API contract against an unrelated, joint-only stdlib middleware."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from botlab.control import client as transport
from botlab.control.cli import main
from botlab.control.discovery import endpoint, project_key
from botlab.control.protocol import ControlError
from botlab.control.receipts import Receipts


class JointMiddleware:
    """No base pose, velocity, sit action, policy state or MicroDuck imports."""

    def __init__(self, project):
        self.project = str(project.resolve())
        self.key = project_key(project)
        self.instance = "joint-instance-1"
        self.session = "joint-session"
        self.episode = "joint-episode"
        self.revision = "joints-v1"
        self.api_version = 1
        self.commands = {}
        self.received = []
        self.gets = []
        self.evicted = set()
        self.redirect = False
        self.descriptor = None
        self.capability = {"id": "joint.position", "version": 1, "aliases": ["home"],
                           "input_schema": {"type": "object", "required": ["positions"],
                                            "additionalProperties": False,
                                            "properties": {"positions": {"type": "array", "minItems": 2,
                                                                        "maxItems": 2, "items": {"type": "number", "minimum": -2, "maximum": 2}}}},
                           "units": {"positions": "rad"}, "frame": "joint",
                           "completion_semantics": "all joints within declared tolerance",
                           "max_duration_s": 5, "stop_semantics": "hold current joint positions"}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, data, status=200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def fail(self, code, status=409):
                self.reply({"ok": False, "error": {"code": code, "message": code}}, status)

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer test-token":
                    return self.fail("unauthorized", 401)
                parsed = urlsplit(self.path)
                owner.gets.append(self.path)
                if owner.redirect:
                    self.send_response(302)
                    self.send_header("Location", "/should-not-be-requested")
                    self.end_headers()
                    return
                response = {"ok": True, "instance_id": owner.instance}
                if parsed.path.endswith("/info"):
                    response.update(api_version=owner.api_version, protocol="botlab-control", project=owner.project,
                                    project_key=owner.key, backend={"id": "joint-test", "kind": "simulation"})
                elif parsed.path.endswith("/capabilities"):
                    response.update(api_version=owner.api_version, capabilities_revision=owner.revision,
                                    capabilities=[owner.capability])
                elif parsed.path.endswith("/sessions"):
                    response["sessions"] = [{"session_id": owner.session, "episode_id": owner.episode}]
                elif parsed.path.endswith("/state"):
                    response.update(project_key=owner.key, state=owner.state())
                elif "/commands/" in parsed.path:
                    command_id = unquote(parsed.path.rsplit("/", 1)[1])
                    if parse_qs(parsed.query).get("instance_id") != [owner.instance]:
                        return self.fail("wrong_instance")
                    if command_id in owner.evicted:
                        return self.fail("result_evicted")
                    if command_id not in owner.commands:
                        return self.fail("not_found", 404)
                    response["command"] = owner.commands[command_id]
                    response["pending"] = not response["command"]["terminal"]
                else:
                    return self.fail("not_found", 404)
                self.reply(response)

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer test-token":
                    return self.fail("unauthorized", 401)
                envelope = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if envelope["instance_id"] != owner.instance:
                    return self.fail("wrong_instance")
                if envelope["session_id"] != owner.session or envelope["episode_id"] != owner.episode:
                    return self.fail("wrong_session")
                if envelope["capabilities_revision"] != owner.revision:
                    return self.fail("capabilities_changed")
                command_id = envelope["request_id"]
                if command_id not in owner.commands:
                    owner.received.append(envelope)
                    owner.commands[command_id] = {"command_id": command_id, "status": "accepted", "terminal": False,
                                                  "instance_id": owner.instance, "session_id": owner.session,
                                                  "episode_id": owner.episode, "initial_observation": {"capture_stage": "accepted", "positions": [0, 0]},
                                                  "completion_observation": None, "current_state": owner.state()}
                self.reply({"ok": True, "command": owner.commands[command_id]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def state(self):
        state = {"ready": True, "connected": True, "external_enabled": True, "run_state": "running",
                 "session_id": self.session, "episode_id": self.episode, "state_age_ms": 1,
                 "progress_age_ms": 1, "observations": {"positions": [0.2, -0.2]}}
        if getattr(self, "generation", None) is not None:
            state["control_generation"] = self.generation
        return state

    def publish(self, root):
        directory = root / self.key
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        root.chmod(0o700)
        directory.chmod(0o700)
        self.descriptor = directory / f"{self.instance}.json"
        self.descriptor.write_text(json.dumps({"schema": 1, "api_version": 1, "protocol": "botlab-control",
                                               "project": self.project, "project_key": self.key, "instance_id": self.instance,
                                               "endpoint": self.endpoint, "token": "test-token", "pid": os.getpid()}))
        self.descriptor.chmod(0o600)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.fixture
def middleware(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("BOTLAB_CONTROL_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("BOTLAB_CONTROL_RECEIPTS_DIR", str(tmp_path / "receipts"))
    monkeypatch.chdir(project)
    server = JointMiddleware(project)
    server.publish(runtime)
    try:
        yield server
    finally:
        server.close()


def invoke(*args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main([*args, "--json"])
    return code, json.loads(output.getvalue())


def call(request_id="joint-request"):
    return invoke("call", "joint.position", "--args", '{"positions":[0.5,-0.5]}', "--request-id", request_id)


def test_joint_only_middleware_end_to_end(middleware):
    assert invoke("state")[1]["state"]["observations"] == {"positions": [0.2, -0.2]}
    assert "policy_state" not in invoke("state")[1]["state"]
    assert invoke("sessions")[0] == 0
    code, submitted = call()
    assert code == 0
    assert submitted["command"]["status"] == "accepted"
    code, pending = invoke("wait", "--id", "joint-request", "--timeout", "0")
    assert code == 0 and pending["pending"] is True
    middleware.commands["joint-request"].update(status="completed", terminal=True,
                                               completion_observation={"positions": [0.5, -0.5], "step": 42})
    code, completed = invoke("wait", "--id", "joint-request")
    assert code == 0 and completed["command"]["completion_observation"]["step"] == 42
    assert completed["command"]["current_state"]["observations"]["positions"] == [0.2, -0.2]
    assert invoke("stop", "--request-id", "joint-stop")[0] == 0
    assert middleware.received[-1]["capability"] == "control.stop"


def test_unknown_robot_shortcuts_are_not_guessed(middleware):
    assert invoke("move", "--vx", ".25", "--duration", "1")[1]["error"]["code"] == "unsupported_capability"
    assert invoke("action", "sit")[1]["error"]["code"] == "unsupported_capability"
    assert not middleware.received


def test_alias_resolves_declared_capability(middleware):
    middleware.capability["input_schema"] = {"type": "object"}
    assert invoke("action", "home")[0] == 0
    assert middleware.received[-1]["capability"] == "joint.position"


def test_arguments_file_and_schema_validation(middleware, tmp_path):
    arguments = tmp_path / "args.json"
    arguments.write_text('{"positions":[0.1,0.2]}')
    assert invoke("call", "joint.position", "--args-file", str(arguments))[0] == 0
    assert invoke("call", "joint.position", "--args", '{"positions":[3,0]}')[1]["error"]["code"] == "invalid_args"
    assert len(middleware.received) == 1


@pytest.mark.parametrize("args", [("call", "joint.position", "--args", '{"positions":[NaN,0]}'),
                                  ("move", "--vx", "nan", "--duration", "1"),
                                  ("wait", "--id", "x", "--timeout", "nan"), ("not-a-command",),
                                  ("call",), ("move", "--vx", "bad")])
def test_json_parse_errors_are_single_json_object(middleware, args):
    code, result = invoke(*args)
    assert code == 2 and result["ok"] is False
    assert not middleware.received


def test_retry_is_durable_private_and_no_credentials(middleware):
    assert call()[0] == 0
    receipt = Receipts(middleware.project).load("joint-request")
    assert receipt["envelope"]["session_id"] == middleware.session
    raw = Receipts(middleware.project).path("joint-request").read_text()
    assert "test-token" not in raw and '"token"' not in raw
    if os.name == "posix":
        assert Receipts(middleware.project).path("joint-request").stat().st_mode & 0o777 == 0o600
    assert call()[0] == 0
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert len(middleware.received) == 1


def test_retry_from_fresh_process(middleware):
    assert call()[0] == 0
    process = subprocess.run([sys.executable, "-m", "botlab.control", "retry", "--id", "joint-request", "--json"],
                             env={**os.environ, "PYTHONPATH": str(Path(transport.__file__).parents[2])},
                             text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["command"]["command_id"] == "joint-request"
    assert len(middleware.received) == 1


def test_duplicate_request_intent_cannot_change(middleware):
    call()
    code, error = invoke("call", "joint.position", "--args", '{"positions":[0,0]}', "--request-id", "joint-request")
    assert code == 2 and error["error"]["code"] == "request_conflict"
    assert len(middleware.received) == 1


def test_explicit_empty_request_id_is_not_replaced_with_generated_identity(middleware):
    for _ in range(2):
        code, result = call("")
        assert code == 2 and result["error"]["code"] == "invalid_identity"
    assert not middleware.received


def test_missing_request_id_generates_an_identity(middleware):
    code, result = invoke("call", "joint.position", "--args", '{"positions":[0.5,-0.5]}')
    assert code == 0
    assert len(result["command"]["command_id"]) == 36
    assert middleware.received[0]["request_id"] == result["command"]["command_id"]


def test_lost_submit_response_is_queried_not_replayed(middleware, monkeypatch):
    original = transport.request

    def lose_response(descriptor, path, body=None, timeout=5):
        result = original(descriptor, path, body, timeout)
        if body is not None:
            raise ControlError("network_timeout", "response lost")
        return result

    monkeypatch.setattr(transport, "request", lose_response)
    code, unknown = call()
    assert code == 2 and unknown["request_id"] == "joint-request"
    assert unknown["instance_id"] == middleware.instance
    monkeypatch.setattr(transport, "request", original)
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert len(middleware.received) == 1


def test_original_identity_is_reused_only_when_unknown(middleware):
    call()
    original = middleware.received[0].copy()
    middleware.commands.clear()
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert middleware.received == [original, original]


@pytest.mark.parametrize("new_generation", [2, None])
def test_unreceived_request_cannot_resume_changed_control_generation(middleware, monkeypatch, new_generation):
    middleware.generation = 1
    original = transport.request

    def fail_before_delivery(descriptor, path, body=None, timeout=5):
        if body is not None:
            raise ControlError("connection_error", "request never reached middleware")
        return original(descriptor, path, body, timeout)

    monkeypatch.setattr(transport, "request", fail_before_delivery)
    assert call()[0] == 2
    receipt = Receipts(middleware.project).load("joint-request")
    assert receipt["control_generation"] == 1
    assert "control_generation" not in receipt["envelope"]
    middleware.generation = new_generation
    monkeypatch.setattr(transport, "request", original)
    code, result = invoke("retry", "--id", "joint-request")
    assert code == 2 and result["error"]["code"] == "state_changed"
    assert not middleware.received


def test_unchanged_control_generation_allows_unknown_retry(middleware):
    middleware.generation = 1
    call()
    original = middleware.received[0].copy()
    middleware.commands.clear()
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert middleware.received == [original, original]


def test_changed_control_generation_still_allows_known_result_query(middleware):
    middleware.generation = 1
    call()
    middleware.generation = 2
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert len(middleware.received) == 1


@pytest.mark.parametrize("field", ["session", "episode", "revision"])
def test_unknown_request_is_not_rebound(middleware, field):
    call()
    middleware.commands.clear()
    setattr(middleware, field, "changed")
    code, result = invoke("retry", "--id", "joint-request")
    assert code == 2 and result["error"]["code"] == "original_identity_changed"
    assert len(middleware.received) == 1


def test_restart_never_replays_original_request(middleware):
    call()
    middleware.instance = "joint-instance-2"
    code, result = invoke("retry", "--id", "joint-request")
    assert code == 2 and result["error"]["code"] == "wrong_instance"
    assert len(middleware.received) == 1


def test_missing_receipt_and_evicted_result_never_replay(middleware):
    assert invoke("retry", "--id", "missing")[1]["error"]["code"] == "missing_receipt"
    call()
    middleware.evicted.add("joint-request")
    assert invoke("retry", "--id", "joint-request")[1]["error"]["code"] == "result_evicted"
    assert len(middleware.received) == 1


@pytest.mark.parametrize("status", ["cancelled", "failed", "expired"])
def test_terminal_failure_exit_code(middleware, status):
    call()
    middleware.commands["joint-request"].update(status=status, terminal=True)
    assert invoke("wait", "--id", "joint-request")[0] == 3


def test_wrong_project_and_explicit_wrong_session(middleware, tmp_path):
    assert invoke("--project", str(tmp_path / "other"), "--endpoint", middleware.endpoint, "state")[0] == 2
    assert invoke("--session", "wrong", "state")[1]["error"]["code"] == "wrong_session"
    middleware.project = str(tmp_path / "wrong")
    assert invoke("state")[1]["error"]["code"] == "wrong_project"


def test_multi_instance_requires_explicit_selection(middleware):
    second = JointMiddleware(Path(middleware.project))
    second.instance = "second-instance"
    second.publish(Path(os.environ["BOTLAB_CONTROL_RUNTIME_DIR"]))
    try:
        assert invoke("state")[1]["error"]["code"] == "ambiguous_instance"
        assert invoke("--instance", second.instance, "state")[1]["instance_id"] == second.instance
        code, result = invoke("instances")
        assert code == 0 and len(result["instances"]) == 2
        assert "test-token" not in json.dumps(result)
    finally:
        second.close()


def test_versions_metadata_and_redirect_rejected(middleware):
    middleware.api_version = 2
    assert invoke("state")[1]["error"]["code"] == "unsupported_version"
    middleware.api_version = 1
    middleware.capability["version"] = 2
    assert invoke("capabilities")[1]["error"]["code"] == "unsupported_capability_version"
    middleware.capability["version"] = 1
    del middleware.capability["units"]
    assert invoke("capabilities")[1]["error"]["code"] == "invalid_protocol"
    middleware.redirect = True
    assert invoke("state")[1]["error"]["code"] == "redirect_refused"
    assert not any("should-not" in path for path in middleware.gets)


def test_connection_token_requires_explicit_flag(middleware):
    assert "token" not in invoke("connection")[1]
    assert invoke("connection", "--show-token")[1]["token"] == "test-token"


def test_connection_embeds_agent_prompt_without_token(middleware):
    result = invoke("connection", "--show-token")[1]
    prompt = result["agent_prompt"]
    assert result["project"] in prompt
    assert "state --json" in prompt
    assert "test-token" not in prompt
    # The prompt assumes a prepared environment: it must never teach installs.
    for banned in ("pip install", "uv tool", "npm install", "git clone"):
        assert banned not in prompt


@pytest.mark.parametrize("origin", ["http://example.com:80", "http://10.0.0.5:8776", "http://127.0.0.1:80/path",
                                    "http://user:secret@127.0.0.1:80", "http://127.0.0.1:80?token=x",
                                    "http://127.0.0.1:80#x", "https://user:secret@example.com",
                                    "ftp://127.0.0.1:80", "http://127.0.0.1", "https://:443"])
def test_endpoint_never_exposes_auth_to_remote_or_redirect(origin):
    with pytest.raises(ControlError):
        endpoint(origin)


def test_unsafe_discovery_file_is_not_trusted(middleware):
    middleware.descriptor.chmod(0o644)
    if os.name == "posix":
        assert invoke("state")[1]["error"]["code"] == "no_service"


def test_receipt_write_failure_prevents_submit(middleware, monkeypatch):
    def fail(*args, **kwargs):
        raise ControlError("receipt_unavailable", "disk full")
    monkeypatch.setattr(Receipts, "save", fail)
    assert call()[1]["error"]["code"] == "receipt_unavailable"
    assert not middleware.received


def test_explicit_selection_must_not_rebind_receipt(middleware):
    call()
    code, result = invoke("--instance", "other-instance", "retry", "--id", "joint-request")
    assert code == 2 and result["error"]["code"] == "identity_conflict"
    assert len(middleware.received) == 1


def test_bad_schema_returns_json_protocol_error(middleware):
    middleware.capability["input_schema"]["properties"] = None
    assert call()[1]["error"]["code"] == "invalid_protocol"
    assert not middleware.received


def test_invalid_command_identity_or_terminal_flag_is_rejected(middleware):
    call()
    middleware.commands["joint-request"]["command_id"] = "other-command"
    assert invoke("wait", "--id", "joint-request")[1]["error"]["code"] == "wrong_command"
    middleware.commands["joint-request"]["command_id"] = "joint-request"
    middleware.commands["joint-request"]["terminal"] = True
    assert invoke("wait", "--id", "joint-request")[1]["error"]["code"] == "invalid_protocol"


def test_changed_episode_does_not_prevent_querying_known_task(middleware):
    call()
    middleware.episode = "later-episode"
    assert invoke("retry", "--id", "joint-request")[0] == 0
    assert len(middleware.received) == 1


def test_descriptor_replacement_refuses_retry(middleware):
    call()
    middleware.descriptor.unlink()
    middleware.instance = "replacement-instance"
    middleware.publish(Path(os.environ["BOTLAB_CONTROL_RUNTIME_DIR"]))
    assert invoke("retry", "--id", "joint-request")[1]["error"]["code"] == "original_instance_unavailable"
    assert len(middleware.received) == 1


def test_cli_dispatch_and_base_install_without_optional_packages(tmp_path):
    source = Path(transport.__file__).parents[2]
    script = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"aiohttp", "numpy", "mujoco", "onnxruntime", "microduck_cli", "microduck_control"}:
            raise ImportError("optional package forbidden: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from botlab._cli import main
raise SystemExit(main(["control", "instances", "--json"]))
'''
    process = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                             env={**os.environ, "PYTHONPATH": str(source)}, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["instances"] == []
