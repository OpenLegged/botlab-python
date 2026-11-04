"""Remote middleware flows over real TLS: enrollment, pinned identity, restart
refusal, re-enrollment and multi-instance selection against a dev CA."""

from __future__ import annotations

import json
import os
import shutil
import ssl
import stat
import subprocess
import threading
from pathlib import Path

import pytest

from botlab.control.discovery import project_key
from tests.test_control_cli import JointMiddleware, invoke


@pytest.fixture(scope="module")
def authority(tmp_path_factory) -> Path:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required for TLS middleware tests")
    root = tmp_path_factory.mktemp("pki")

    def run(*args: str) -> None:
        subprocess.run(["openssl", *args], check=True, capture_output=True, cwd=root)

    # Python 3.13 verifies with VERIFY_X509_STRICT by default: a CA without
    # basicConstraints/keyUsage is rejected ("CA cert does not include key usage
    # extension"), so the fixture CA carries the extensions a real CA has.
    run("req", "-x509", "-newkey", "rsa:2048", "-keyout", "ca.key", "-out", "ca.crt",
        "-days", "2", "-nodes", "-subj", "/CN=BotLab Test CA",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    run("req", "-newkey", "rsa:2048", "-keyout", "server.key", "-out", "server.csr",
        "-nodes", "-subj", "/CN=localhost")
    (root / "san.ext").write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
        "-CAcreateserial", "-out", "server.crt", "-days", "2", "-extfile", "san.ext")
    return root


class RemoteMiddleware(JointMiddleware):
    """The joint stub served over TLS, as a remote middleware would be."""

    def __init__(self, project: Path, authority: Path):
        super().__init__(project)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(authority / "server.crt", authority / "server.key")
        self.server.shutdown()
        self.thread.join()
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"https://127.0.0.1:{self.server.server_port}"


def prepare(tmp_path: Path, monkeypatch, authority: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("BOTLAB_CONTROL_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("BOTLAB_CONTROL_RECEIPTS_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("BOTLAB_CONTROL_CA_BUNDLE", str(authority / "ca.crt"))
    monkeypatch.setenv("BOTLAB_CONTROL_TOKEN", "test-token")
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def remote(tmp_path, monkeypatch, authority):
    project = prepare(tmp_path, monkeypatch, authority)
    server = RemoteMiddleware(project, authority)
    yield server
    server.close()


def enrolled_descriptor(instance_id: str) -> tuple[Path, dict]:
    runtime = Path(os.environ["BOTLAB_CONTROL_RUNTIME_DIR"])
    directory = runtime / project_key(Path.cwd())
    descriptor = directory / f"{instance_id}.json"
    return descriptor, json.loads(descriptor.read_text())


def test_enroll_pins_identity_and_controls_over_tls(remote):
    code, result = invoke("enroll", "--endpoint", remote.endpoint)
    assert code == 0
    assert result["ok"] and result["instance_id"] == remote.instance
    assert result["websocket"] == f"wss://127.0.0.1:{remote.server.server_port}/runtime"

    descriptor, payload = enrolled_descriptor(remote.instance)
    assert payload["remote"] is True and payload["token"] == "test-token"
    assert payload["project_key"] == remote.key and payload["endpoint"] == remote.endpoint
    assert stat.S_IMODE(descriptor.stat().st_mode) & 0o077 == 0

    code, state = invoke("state")
    assert code == 0 and state["state"]["session_id"] == remote.session
    code, command = invoke("call", "joint.position", "--args", '{"positions":[0.5,-0.5]}',
                           "--request-id", "remote-1")
    assert code == 0 and command["command"]["status"] == "accepted"
    assert remote.received[0]["instance_id"] == remote.instance


def test_remote_restart_refuses_replay_until_reenrolled(remote):
    assert invoke("enroll", "--endpoint", remote.endpoint)[0] == 0
    assert invoke("call", "joint.position", "--args", '{"positions":[0.5,-0.5]}',
                  "--request-id", "remote-2")[0] == 0

    remote.instance = "joint-instance-restarted"
    code, result = invoke("wait", "--id", "remote-2")
    assert code == 2 and result["error"]["code"] == "wrong_instance"
    code, result = invoke("retry", "--id", "remote-2")
    assert code == 2 and result["error"]["code"] == "wrong_instance"

    code, result = invoke("enroll", "--endpoint", remote.endpoint)
    assert code == 0 and result["instance_id"] == "joint-instance-restarted"
    # Re-enrollment replaced this endpoint's previous descriptor, not stacked it.
    descriptor, payload = enrolled_descriptor("joint-instance-restarted")
    assert payload["remote"] is True
    code, state = invoke("state")
    assert code == 0 and state["state"]["session_id"] == remote.session


def test_identity_pin_mismatch_is_refused(remote):
    assert invoke("enroll", "--endpoint", remote.endpoint)[0] == 0
    remote.key = "a-different-project"
    code, result = invoke("state")
    assert code == 2 and result["error"]["code"] == "wrong_project"


def test_self_signed_tls_is_rejected_without_the_dev_ca(remote, monkeypatch):
    monkeypatch.delenv("BOTLAB_CONTROL_CA_BUNDLE")
    code, result = invoke("enroll", "--endpoint", remote.endpoint)
    assert code == 2 and result["error"]["code"] == "connection_error"
    runtime = Path(os.environ["BOTLAB_CONTROL_RUNTIME_DIR"])
    assert not runtime.exists() or not any(runtime.rglob("*.json"))


def test_multiple_enrolled_instances_require_explicit_selection(tmp_path, monkeypatch, authority):
    project = prepare(tmp_path, monkeypatch, authority)
    first = RemoteMiddleware(project, authority)
    second = RemoteMiddleware(project, authority)
    try:
        assert invoke("enroll", "--endpoint", first.endpoint)[0] == 0
        # Both stubs report the same instance identity until told apart; a
        # second endpoint reusing it must fail cleanly instead of clobbering.
        code, result = invoke("enroll", "--endpoint", second.endpoint)
        assert code == 2 and result["error"]["code"] == "instance_conflict"
        second.instance = "joint-instance-second"
        assert invoke("enroll", "--endpoint", second.endpoint)[0] == 0
        code, result = invoke("state")
        assert code == 2 and result["error"]["code"] == "ambiguous_instance"
        assert {item["instance_id"] for item in result["instances"]} == {first.instance, second.instance}
        code, state = invoke("state", "--instance", second.instance)
        assert code == 0 and state["state"]["session_id"] == second.session
    finally:
        first.close()
        second.close()
