"""Recoverable SSH authentication without guessing keys or persisting passwords."""
import copy
import io
import json
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from botlab._server import start_project_server
from botlab.hil import devices
from botlab.hil.host import HilHost
from botlab.hil.protocol import HilError, error_details
from botlab.hil.targets import save_target


def profile():
    return {"id": "linux", "name": "Linux board", "transport": "ssh",
            "ssh": {"host": "board.local", "user": "robot"}}


def rejection():
    return b"robot@board.local: Permission denied (publickey,password).\n"


@pytest.mark.parametrize("password", [None, "incorrect-secret"])
def test_openssh_auth_rejection_carries_recovery_metadata_and_redacts(password, monkeypatch):
    credentials = {"password": password} if password else {}
    device = devices.LinuxDevice(profile(), credentials)
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, env=kwargs["env"])
        detail = rejection() + (password.encode() if password else b"")
        return subprocess.CompletedProcess(argv, 255, b"", detail)

    monkeypatch.setattr(devices.subprocess, "run", run)
    with pytest.raises(HilError) as caught:
        device._command("python3 -c 'pass'")
    assert error_details(caught.value) == {"error_code": "ssh_auth_required", "target_id": "linux",
                                          "password_attempted": password is not None}
    assert seen["env"]["LC_ALL"] == "C"
    if password:
        assert password not in str(caught.value)
        assert password not in repr(seen["argv"])
    else:
        assert "BatchMode=yes" in seen["argv"]
        # Default key files and ssh-agent remain OpenSSH's responsibility.
        assert "IdentitiesOnly=yes" not in seen["argv"]


@pytest.mark.parametrize(("code", "stderr"), [
    (1, rejection()),
    (255, b"python3: Permission denied\n"),
    (255, b"Permission denied (publickey,password).\n"),
    (255, b"another@board.local: Permission denied (publickey).\n"),
    (255, b"ssh: connect to host board.local port 22: Operation timed out\n"),
    (255, b"Host key verification failed.\n"),
    (1, b"sudo: a password is required\n"),
])
def test_transport_host_key_and_remote_permissions_never_request_ssh_password(code, stderr, monkeypatch):
    monkeypatch.setattr(devices.subprocess, "run", lambda argv, **_kwargs:
                        subprocess.CompletedProcess(argv, code, b"", stderr))
    with pytest.raises(HilError) as caught:
        devices.LinuxDevice(profile())._command("python3 -c 'pass'")
    assert error_details(caught.value) == {}


@pytest.mark.parametrize(("password", "padding"), [
    ("synthetic-review-password-0123456789", "x" * 1790),
    ("  synthetic-review-password-0123456789  ", ""),
])
def test_ssh_error_redacts_before_truncation_or_whitespace_trimming(password, padding, monkeypatch):
    stderr = (password + padding).encode()
    monkeypatch.setattr(devices.subprocess, "run", lambda argv, **_kwargs:
                        subprocess.CompletedProcess(argv, 1, b"", stderr))
    with pytest.raises(HilError) as caught:
        devices.LinuxDevice(profile(), {"password": password})._command("python3 -c 'pass'")
    error = str(caught.value)
    assert "0123456789" not in error
    assert "[redacted]" in error
    assert len(error) <= 1800


def test_successful_key_auth_needs_no_password(monkeypatch):
    monkeypatch.setattr(devices.subprocess, "run", lambda argv, **_kwargs:
                        subprocess.CompletedProcess(argv, 0, b"key-auth-ok", b""))
    assert devices.LinuxDevice(profile())._command("python3 -c 'pass'") == b"key-auth-ok"


def test_tunnel_auth_rejection_preserves_metadata_and_cleans_temp_files(monkeypatch):
    class Process:
        stdin = io.BytesIO()
        stderr = io.BytesIO(rejection())

        def poll(self):
            return 255

    process = Process()
    captured = {}

    def popen(argv, **kwargs):
        captured.update(environment=kwargs["env"])
        return process

    monkeypatch.setattr(devices.subprocess, "Popen", popen)
    with pytest.raises(HilError) as caught:
        devices.LinuxDevice(profile(), {"password": "wrong-secret"}).open_tunnel()
    assert error_details(caught.value) == {"error_code": "ssh_auth_required", "target_id": "linux",
                                          "password_attempted": True}
    assert process.stdin.closed
    from pathlib import Path
    assert not Path(captured["environment"]["SSH_ASKPASS"]).exists()


@pytest.fixture
def auth_http(project_dir, tmp_path, monkeypatch):
    target_file = tmp_path / "targets.json"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(target_file))
    target = save_target(profile())
    host = HilHost(project_dir)
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html><head></head><body>test</body></html>")
    httpd, base = start_project_server(project_dir, None, "127.0.0.1", 0, ui_dir=ui,
                                      release_provider=lambda: b"zip", hil=host)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def api(method, path, body=None):
        headers = {"X-Botlab-Hil-Token": httpd.botlab_hil_token, "Origin": base}
        data = None
        if method == "POST":
            headers["Content-Type"] = "application/json"
            data = json.dumps(body or {}).encode()
        with urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=5) as response:
            return json.load(response)

    try:
        yield host, target, target_file, api
    finally:
        host.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def finish(api, response):
    for _ in range(200):
        result = api("GET", "/hil/operations/" + response["operation"]["id"])
        if result["operation"]["status"] != "running":
            return result
        time.sleep(.01)
    pytest.fail("Authentication operation did not finish")


@pytest.mark.parametrize("route", ["/hil/devices/inspect", "/hil/devices/provision",
                                    "/hil/connect", "/hil/targets/projects"])
def test_async_auth_challenge_is_recoverable_without_terminal_state_error(auth_http, monkeypatch, route):
    host, target, _, api = auth_http

    def rejected(self, *_args, **_kwargs):
        raise self._ssh_failure(255, rejection())

    monkeypatch.setattr(devices.LinuxDevice, "_python", rejected)
    monkeypatch.setattr(devices.LinuxDevice, "open_tunnel", rejected)
    first = finish(api, api("POST", route, {"target": target["id"]}))
    assert first["operation"]["error_code"] == "ssh_auth_required"
    assert first["operation"]["target_id"] == target["id"]
    assert first["operation"]["password_attempted"] is False
    assert host.state()["error"] is None
    response = api("POST", "/hil/credentials", {
        "target": target["id"], "expected_target": target, "password": "incorrect-secret"})
    assert response["credential_status"][target["id"]] == {"password": True}
    second = finish(api, api("POST", route, {"target": target["id"]}))
    assert second["operation"]["password_attempted"] is True
    assert "incorrect-secret" not in json.dumps(second)
    assert host.state()["error"] is None


def test_credentials_are_snapshot_bound_memory_only_and_do_not_change_main_state(auth_http):
    host, target, target_file, api = auth_http
    original = target_file.read_bytes()
    host.selected_target = "another"
    host.application = {"built": {"id": "built"}, "loaded": {"id": "loaded"}, "projects": []}
    previous_application = copy.deepcopy(host.application)
    host.error = "A separate error"
    response = api("POST", "/hil/credentials", {
        "target": target["id"], "expected_target": target, "password": "new-secret"})
    assert response["capabilities"]["credentialPrompt"] == 1
    assert response["selectedTarget"] == "another"
    assert response["application"] == previous_application
    assert not response["connected"]
    assert response["error"] == "A separate error"
    assert "new-secret" not in json.dumps(response)
    assert target_file.read_bytes() == original
    assert host._credentials_for(target)["password"] == "new-secret"
    fresh = HilHost(host.project_dir)
    try:
        assert fresh.state()["credential_status"][target["id"]] == {"password": False}
    finally:
        fresh.close()


@pytest.mark.parametrize("invalid", ["changed", "missing", "wrong-id", "empty", "not-string", "extra"])
def test_credentials_reject_changed_or_missing_snapshot_and_invalid_password(auth_http, invalid):
    host, target, target_file, api = auth_http
    original = target_file.read_bytes()
    body = {"target": target["id"], "expected_target": copy.deepcopy(target), "password": "secret"}
    if invalid == "changed":
        body["expected_target"]["ssh"]["host"] = "old-host.local"
    elif invalid == "missing":
        del body["expected_target"]
    elif invalid == "wrong-id":
        body["target"] = "missing"
    elif invalid == "empty":
        body["password"] = ""
    elif invalid == "not-string":
        body["password"] = 123
    else:
        body["save_password"] = True
    with pytest.raises(HTTPError) as caught:
        api("POST", "/hil/credentials", body)
    assert caught.value.code == 400
    assert host.credentials == {}
    assert target_file.read_bytes() == original


def test_synchronous_http_error_retains_auth_metadata_and_redacts(auth_http, monkeypatch):
    host, target, _, api = auth_http
    host._store_credentials(target, {"password": "hidden-secret"})

    def reject(*_args):
        raise HilError("Rejected hidden-secret", error_code="ssh_auth_required",
                       target_id=target["id"], password_attempted=True)

    monkeypatch.setattr(host, "_handle", reject)
    with pytest.raises(HTTPError) as caught:
        api("POST", "/hil/credentials", {})
    response = json.load(caught.value)
    assert response == {"error": "Rejected [redacted]", "error_code": "ssh_auth_required",
                        "target_id": "linux", "password_attempted": True}
    assert host.error is None
