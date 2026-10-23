"""Browser-facing management routes with a real CPU Agent and loopback HTTP."""
import base64
import copy
import json
import math
import threading
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from botlab._server import start_project_server
from botlab.hil.agent import AgentServer, AgentState
from botlab.hil.host import HilHost
from botlab.hil.protocol import HilError
from botlab.hil.targets import read_targets, save_target
from botlab.hil.worker import WorkerBackend
from tests.test_hil_projects import artifact


def wait_job(api, response, *, timeout=10):
    operation = response["operation"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = api("GET", "/hil/operations/" + operation["id"])
        if result["operation"]["status"] != "running":
            return result
        time.sleep(0.01)
    pytest.fail("Management operation did not finish: " + json.dumps(result))


def completed(api, response):
    result = wait_job(api, response)
    assert result["operation"]["status"] == "succeeded", result
    return result["result"]


@pytest.fixture
def management(project_dir, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    worker = WorkerBackend("onnxruntime")
    state = AgentState(tmp_path / "cache", worker, root=tmp_path / "target")
    agent = AgentServer(("127.0.0.1", 0), state)
    agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
    agent_thread.start()
    target = save_target({"id": "cpu", "name": "Generic CPU", "backend": "onnxruntime",
                          "endpoint": f"tcp://127.0.0.1:{agent.server_address[1]}"})
    host = HilHost(project_dir, target["id"])
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html><head></head><body>test</body></html>")
    httpd, base = start_project_server(project_dir, None, "127.0.0.1", 0, ui_dir=ui,
                                      release_provider=lambda: b"zip", hil=host)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def api(method, path, body=None, *, timeout=5):
        headers = {"X-Botlab-Hil-Token": httpd.botlab_hil_token, "Origin": base}
        data = None
        if method == "POST":
            headers["Content-Type"] = "application/json"
            data = json.dumps(body or {}).encode()
        with urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=timeout) as result:
            return json.load(result)

    try:
        yield host, state, api
    finally:
        host.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        agent.shutdown()
        agent.server_close()
        agent_thread.join()
        worker.reset()


def test_full_build_deploy_load_run_stop_unload_management(management):
    host, agent, api = management
    assert api("POST", "/hil/connect", {"target": "cpu"})["connected"]
    built = completed(api, api("POST", "/hil/build"))["built"]
    assert built["execution_kind"] == "model-service"
    assert api("GET", "/hil/state")["application"]["built"] == built
    deployed = completed(api, api("POST", "/hil/deploy"))
    assert deployed["project"]["id"] == built["project"]["id"]
    loaded = completed(api, api("POST", "/hil/projects/load"))["loaded"]
    assert loaded["build_id"] == built["build_id"]
    assert host.session is None
    running = completed(api, api("POST", "/hil/start"))
    session = running["session"]
    metadata = api("POST", "/hil/models", {"session": session,
                    "sha256": built["models"][0]["sha256"]})
    port = metadata["inputs"][0]
    data = [float(index) for index in range(math.prod(port["shape"]))]
    inferred = api("POST", "/hil/infer", {"session": session, "seq": 0,
        "model": metadata["sha256"], "inputs": {port["name"]: {
            "dtype": "float32", "shape": port["shape"], "data": data}}})
    assert inferred["outputs"][metadata["outputs"][0]["name"]]["data"] == data
    assert inferred["calls"] == agent.calls == 1
    stopped = api("POST", "/hil/stop", {"session": session})
    assert stopped["session"] is None
    assert stopped["application"]["loaded"] is not None
    assert agent.models == {}
    assert api("POST", "/hil/projects/unload")["loaded"] is None
    assert len(api("GET", "/hil/projects")["projects"]) == 1
    api("POST", "/hil/projects/remove", {"project_id": built["project"]["id"]})
    assert api("GET", "/hil/projects")["projects"] == []
    assert not api("POST", "/hil/disconnect")["connected"]


def browser_sources(host):
    return {name: (host.project_dir / name).read_text()
            for name in ("main.py", "controller.py", "robot.preset.yaml")}


def prepare_loaded_project(api):
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/build"))
    completed(api, api("POST", "/hil/deploy"))
    return completed(api, api("POST", "/hil/projects/load"))["loaded"]


def loaded_start_body(host, loaded):
    return {"request_id": uuid.uuid4().hex, "sources": browser_sources(host),
            "project_id": loaded["project_id"], "build_id": loaded["build_id"]}


def test_deploy_requires_existing_build_without_remote_changes(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    def unexpected(*args, **kwargs):
        pytest.fail("Deploy must not build or issue remote requests without an artifact")
    monkeypatch.setattr(host, "_build", unexpected)
    monkeypatch.setattr(host, "_rpc", unexpected)
    result = wait_job(api, api("POST", "/hil/deploy"))
    assert result["operation"]["status"] == "failed"
    assert "Build the project before deploying" in result["operation"]["error"]
    assert host.artifact is None
    assert host.application["built"] is None
    assert agent.store.list() == []
    assert not (host.project_dir / ".botlab").exists()


def test_deploy_keeps_built_artifact_after_local_sources_and_model_change(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    built = completed(api, api("POST", "/hil/build"))["built"]
    artifact_before = copy.deepcopy(host.artifact)
    (host.project_dir / "controller.py").write_text("invalid python!\n")
    (host.project_dir / "assets/models/policy.onnx").write_bytes(b"changed after build")
    def unexpected(*args, **kwargs):
        pytest.fail("Deploy must consume the existing build")
    monkeypatch.setattr(host, "_build", unexpected)
    deployed = completed(api, api("POST", "/hil/deploy"))
    assert deployed["project"]["current_build_id"] == built["build_id"]
    assert host.artifact == artifact_before
    assert agent.store.manifest(built["project"]["id"], verify_files=True) == built


def test_cli_deploy_builds_and_installs_without_running(management, capsys):
    from botlab._cli import main

    host, agent, _ = management
    assert main(["deploy", str(host.project_dir), "--target", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["project"]["id"] == agent.store.list()[0]["id"]
    assert agent.session is agent.loaded is None


def test_stepwise_run_uses_exact_selected_release_without_build_deploy_or_load(management, monkeypatch):
    host, agent, api = management
    first = prepare_loaded_project(api)
    source = host.project_dir / "controller.py"
    original = source.read_text()
    source.write_text(original + "\n# Newer local version\n")
    newer = completed(api, api("POST", "/hil/build"))["built"]
    completed(api, api("POST", "/hil/deploy"))
    completed(api, api("POST", "/hil/projects/load", {
        "project_id": first["project_id"], "build_id": first["build_id"]}))
    source.write_text(original)
    assert newer["build_id"] != first["build_id"]
    before = copy.deepcopy(host.application)
    def unexpected(*args, **kwargs):
        pytest.fail("Stepwise run must not build, deploy, or select a project")
    for name in ("_build", "_deploy", "_load_project"):
        monkeypatch.setattr(host, name, unexpected)
    from botlab.hil import artifacts
    monkeypatch.setattr(artifacts, "build_project", unexpected)
    monkeypatch.setattr(artifacts, "_project_identity", unexpected)
    calls = []
    original_rpc = host._rpc
    def record_rpc(request, *args, **kwargs):
        calls.append(request["op"])
        return original_rpc(request, *args, **kwargs)
    monkeypatch.setattr(host, "_rpc", record_rpc)
    result = completed(api, api("POST", "/hil/start-loaded", loaded_start_body(host, first)))
    assert result["session"] == agent.session
    assert agent.loaded == first
    assert host.application == before
    assert calls == ["start"]
    assert api("GET", "/hil/state")["capabilities"]["stepwiseExecution"] == 1
    metadata = api("POST", "/hil/models", {"session": result["session"],
                                            "sha256": first["models"][0]["sha256"]})
    assert metadata["sha256"] == first["models"][0]["sha256"]
    api("POST", "/hil/stop", {"session": result["session"]})
    assert agent.loaded == first


def test_stepwise_run_does_not_require_an_in_memory_build(management):
    host, agent, api = management
    selected = prepare_loaded_project(api)
    # A fresh host can connect a persistent release without building locally.
    host.artifact = None
    host.application["built"] = None
    result = completed(api, api("POST", "/hil/start-loaded", loaded_start_body(host, selected)))
    assert agent.session == result["session"]
    assert host.artifact is None
    assert host.application["built"] is None
    assert host.application["loaded"] == agent.loaded == selected
    api("POST", "/hil/stop", {"session": result["session"]})


@pytest.mark.parametrize("change", ["browser", "disk", "model", "identity", "missing_identity", "missing_sources"])
def test_stepwise_run_rejects_unmatched_local_project_without_changing_selection(management, change):
    host, agent, api = management
    selected = prepare_loaded_project(api)
    body = loaded_start_body(host, selected)
    before = copy.deepcopy(host.application)
    identity_file = host.project_dir / ".botlab/project.json"
    if change == "browser":
        body["sources"]["controller.py"] += "\n# Unsaved edit\n"
    elif change == "disk":
        source = host.project_dir / "controller.py"
        source.write_text(source.read_text() + "\n# Saved edit\n")
        body["sources"] = browser_sources(host)
    elif change == "model":
        model = host.project_dir / "assets/models/policy.onnx"
        model.write_bytes(model.read_bytes() + b"\x12\x07changed")
    elif change == "identity":
        identity_file.write_text(json.dumps({"schema": 1, "id": "another-project"}))
    elif change == "missing_identity":
        identity_file.unlink()
    else:
        del body["sources"]
    result = wait_job(api, api("POST", "/hil/start-loaded", body))
    assert result["operation"]["status"] == "failed"
    assert "Open the matching local project" in result["operation"]["error"]
    assert host.application == before
    assert agent.loaded == selected
    assert agent.session is host.session is None
    if change == "missing_identity":
        assert not identity_file.exists()


@pytest.mark.parametrize("change", ["project_id", "build_id", "unloaded"])
def test_stepwise_run_rejects_stale_or_missing_selection(management, change):
    host, agent, api = management
    selected = prepare_loaded_project(api)
    body = loaded_start_body(host, selected)
    if change == "unloaded":
        api("POST", "/hil/projects/unload")
    else:
        body[change] = "another-project" if change == "project_id" else "f" * 64
    before = copy.deepcopy(host.application)
    result = wait_job(api, api("POST", "/hil/start-loaded", body))
    assert result["operation"]["status"] == "failed"
    assert host.application == before
    assert agent.session is None
    assert agent.loaded == (None if change == "unloaded" else selected)


@pytest.mark.parametrize("stage", ["validation", "started", "completed"])
def test_cancel_stepwise_run_preserves_preselected_project(management, monkeypatch, stage):
    host, agent, api = management
    selected = prepare_loaded_project(api)
    body = loaded_start_body(host, selected)
    entered, proceed = threading.Event(), threading.Event()
    if stage == "validation":
        from botlab.hil import artifacts
        original = artifacts.validate_loaded_project
        def pause_validation(*args, **kwargs):
            original(*args, **kwargs)
            entered.set()
            assert proceed.wait(5)
        monkeypatch.setattr(artifacts, "validate_loaded_project", pause_validation)
    elif stage == "started":
        original = host._handle
        def pause_start(method, path, request):
            result = original(method, path, request)
            if path == "/hil/start":
                entered.set()
                assert proceed.wait(5)
            return result
        monkeypatch.setattr(host, "_handle", pause_start)
    job = api("POST", "/hil/start-loaded", body)
    try:
        if stage == "completed":
            completed(api, job)
        else:
            assert entered.wait(3)
        api("POST", "/hil/cancel-start", {"request_id": body["request_id"]}, timeout=1)
    finally:
        proceed.set()
    result = wait_job(api, job)
    assert result["operation"]["phase"] == "cancelled"
    assert host.session is agent.session is None
    assert agent.loaded == host.application["loaded"] == selected
    assert host.sock is not None
    assert len(agent.store.list()[0]["releases"]) == 1


def test_async_build_progress_endpoint_is_responsive(management, monkeypatch):
    host, _, api = management
    entered, proceed = threading.Event(), threading.Event()
    original = host._build
    def slow_build():
        host._progress({"stage": "validation", "message": "Checking local project"})
        entered.set()
        assert proceed.wait(5)
        return original()
    monkeypatch.setattr(host, "_build", slow_build)
    response = api("POST", "/hil/build")
    try:
        assert entered.wait(2)
        started = time.monotonic()
        state = api("GET", "/hil/state", timeout=1)
        op = api("GET", "/hil/operations/" + response["operation"]["id"], timeout=1)
        assert time.monotonic() - started < 1
        assert state["operation"]["phase"] == op["operation"]["phase"] == "validation"
        assert "Checking local project" in op["operation"]["messages"]
        with pytest.raises(HTTPError) as rejected:
            api("POST", "/hil/build")
        assert rejected.value.code == 400
    finally:
        proceed.set()
    assert completed(api, response)["built"]


def test_progress_count_tracks_repeated_messages_beyond_retained_tail(management, monkeypatch):
    host, _, api = management
    emit_tail, tail_ready = threading.Event(), threading.Event()
    emit_more, more_ready = threading.Event(), threading.Event()
    finish = threading.Event()

    def chatty_build():
        assert emit_tail.wait(5)
        for index in range(5):
            host._progress({"stage": "build", "message": f"Preparing {index}"})
        for _ in range(105):
            host._progress({"stage": "build", "message": "Repeated download progress"})
        tail_ready.set()
        assert emit_more.wait(5)
        host._progress({"stage": "build", "message": "Repeated download progress"})
        more_ready.set()
        assert finish.wait(5)
        return {"built": True}

    monkeypatch.setattr(host, "_build", chatty_build)
    response = api("POST", "/hil/build")
    try:
        assert response["operation"]["messages"] == []
        assert response["operation"]["message_count"] == 0
        emit_tail.set()
        assert tail_ready.wait(2)
        first = api("GET", "/hil/operations/" + response["operation"]["id"])["operation"]
        assert first["message_count"] == 110
        assert first["messages"] == ["Repeated download progress"] * 100
        emit_more.set()
        assert more_ready.wait(2)
        second = api("GET", "/hil/state")["operation"]
        assert second["message_count"] == 111
        assert second["messages"] == first["messages"]
        assert first["message_count"] == 110  # Earlier snapshots remain immutable.
    finally:
        emit_tail.set()
        emit_more.set()
        finish.set()
    assert completed(api, response)["built"]
    assert api("GET", "/hil/state")["operation"]["message_count"] == 111

    monkeypatch.setattr(host, "_build", lambda: {"built": True})
    next_response = api("POST", "/hil/build")
    assert next_response["operation"]["id"] != response["operation"]["id"]
    assert completed(api, next_response)["built"]
    next_operation = api("GET", "/hil/state")["operation"]
    assert next_operation["message_count"] == 0
    assert next_operation["messages"] == []


def test_failed_build_preserves_deployed_project_and_next_run_recovers(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    started = completed(api, api("POST", "/hil/start"))
    api("POST", "/hil/stop", {"session": started["session"]})
    previous = agent.loaded.copy()
    source = host.project_dir / "controller.py"
    original = source.read_text()
    source.write_text("not valid python!\n")
    try:
        failed = wait_job(api, api("POST", "/hil/start"))
        assert failed["operation"]["status"] == "failed"
        assert agent.session is None
        assert agent.loaded == previous
        assert len(agent.store.list()[0]["releases"]) == 1
    finally:
        source.write_text(original)
    assert completed(api, api("POST", "/hil/start"))["session"]
    api("POST", "/hil/stop")


def test_browser_source_mismatch_does_not_deploy_or_start(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    sources = {name: (host.project_dir / name).read_text()
               for name in ("main.py", "controller.py", "robot.preset.yaml")}
    unsaved = {**sources, "controller.py": sources["controller.py"] + "\n# Browser draft\n"}
    failed = wait_job(api, api("POST", "/hil/start", {"sources": unsaved}))
    assert failed["operation"]["status"] == "failed"
    assert "Browser sources differ" in failed["operation"]["error"]
    assert agent.store.list() == []
    assert agent.session is None
    assert agent.loaded is None
    assert not (host.project_dir / ".botlab/project.json").exists()
    started = completed(api, api("POST", "/hil/start", {"sources": sources}))
    assert started["session"]
    assert len(agent.store.list()) == 1
    api("POST", "/hil/stop")


def test_one_click_run_selects_current_project_not_other_loaded_project(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    # Deploy a different application's model via actual TCP on the same owner.
    other = artifact("another-project", "Another project")
    with host.lock:
        host._rpc({"op": "projects.deploy", "artifact": other})
    completed(api, api("POST", "/hil/projects/load", {"project_id": "another-project"}))
    assert agent.loaded["project_id"] == "another-project"
    result = completed(api, api("POST", "/hil/start"))
    current = api("GET", "/hil/state")["application"]
    assert current["loaded"]["project_id"] == current["built"]["project"]["id"]
    assert agent.loaded["project_id"] != "another-project"
    assert len(agent.store.list()) == 2
    api("POST", "/hil/stop", {"session": result["session"]})


def test_target_failure_cleans_host_session_and_can_reconnect(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    started = completed(api, api("POST", "/hil/start"))
    with pytest.raises(HTTPError):
        api("POST", "/hil/models", {"session": started["session"], "sha256": "a" * 64,
                                     "data": base64.b64encode(b"undeployed").decode()})
    assert host.session is None
    assert not api("GET", "/hil/state")["connected"]
    api("POST", "/hil/connect", {"target": "cpu"})
    restarted = completed(api, api("POST", "/hil/start"))
    assert restarted["session"] != started["session"]
    api("POST", "/hil/stop")


def ssh_profile():
    return {"id": "linux", "name": "Linux board", "transport": "ssh", "profile": "generic-linux",
            "ssh": {"host": "192.0.2.10", "user": "robot"}, "root": "~/botlab"}


def test_target_inventory_never_persists_credentials_and_removal_clears_state(project_dir, tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(path))
    host = HilHost(project_dir)
    profile = ssh_profile()
    try:
        host.handle("POST", "/hil/targets", profile)
        assert read_targets()[0]["transport"] == "ssh"
        for secret in ({"password": "private-password"},
                       {"ssh": {**profile["ssh"], "password": "private-password"}}):
            with pytest.raises(HilError):
                host.handle("POST", "/hil/targets", {**profile, **secret})
        with pytest.raises(HilError):
            save_target({**profile, "credentials": {"password": "private-password"}})
        assert "private-password" not in path.read_text()
        host.credentials["linux"] = {"password": "private-password"}
        host.device_info["linux"] = {"machine": "aarch64"}
        assert "private-password" not in json.dumps(host.state())
        host.handle("POST", "/hil/targets/remove", {"target": "linux"})
        assert read_targets() == []
        assert host.credentials == host.device_info == {}
        assert host.selected_target is None
    finally:
        host.close()


def test_editing_target_does_not_reuse_another_host_credentials(project_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "profiles.json"))
    host = HilHost(project_dir)
    profile = ssh_profile()
    try:
        host.handle("POST", "/hil/targets", profile)
        host.credentials["linux"] = {"password": "old-host-password"}
        host.device_info["linux"] = {"hostname": "old-host"}
        host.handle("POST", "/hil/targets", {**profile, "ssh": {**profile["ssh"], "host": "192.0.2.11"}})
        assert "linux" not in host.credentials
        assert "linux" not in host.device_info
    finally:
        host.close()


def test_invalid_transient_password_completes_failed_job(project_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "profiles.json"))
    save_target(ssh_profile())
    host = HilHost(project_dir, "linux")
    try:
        job = host.handle("POST", "/hil/devices/inspect", {"target": "linux", "password": 123})
        result = wait_job(host.handle, job, timeout=2)
        assert result["operation"]["status"] == "failed"
        assert host.credentials.get("linux", {}).get("password") is None
    finally:
        host.close()


def test_async_device_errors_are_redacted_and_device_closed(project_dir, tmp_path, monkeypatch):
    from botlab.hil import devices
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "profiles.json"))
    save_target(ssh_profile())
    closed = []
    class FailedDevice:
        def __init__(self, target, credentials, progress):
            self.target = target
            self.credentials = credentials
            self.progress = progress
        def inspect(self):
            self.progress({"stage": "inspect", "message": "Remote progress mentions private-password"})
            raise HilError("Remote failure mentions private-password")
        def close(self):
            closed.append(True)
    monkeypatch.setattr(devices, "LinuxDevice", FailedDevice)
    host = HilHost(project_dir, "linux")
    try:
        response = host.handle("POST", "/hil/devices/inspect", {"password": "private-password"})
        result = wait_job(host.handle, response)
        assert result["operation"]["status"] == "failed"
        assert "private-password" not in json.dumps(result)
        assert "private-password" not in json.dumps(host.state())
        assert "[redacted]" in result["operation"]["error"]
        assert result["operation"]["message_count"] == 1
        assert result["operation"]["messages"] == ["Remote progress mentions [redacted]"]
        assert closed == [True]
    finally:
        host.close()


def test_synchronous_device_errors_are_redacted(management, monkeypatch):
    host, _, api = management
    host.credentials["cpu"] = {"password": "private-password"}

    def failed_command(*args):
        raise HilError("Remote failure mentions private-password")

    monkeypatch.setattr(host, "_handle", failed_command)
    with pytest.raises(HTTPError) as failed:
        api("POST", "/hil/disconnect")
    response = failed.value.read().decode()
    assert "private-password" not in response
    assert "[redacted]" in response
    assert host.state()["error"] == "Remote failure mentions [redacted]"


def test_cancel_before_start_request_arrives_never_builds_or_leases(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    request_id = uuid.uuid4().hex
    assert api("POST", "/hil/cancel-start", {"request_id": request_id})["cancelled"]
    result = wait_job(api, api("POST", "/hil/start", {"request_id": request_id}))
    assert result["operation"]["status"] == "failed"
    assert result["operation"]["phase"] == "cancelled"
    assert agent.store.list() == []
    assert agent.session is agent.loaded is None
    assert not (host.project_dir / ".botlab/project.json").exists()


def test_cancel_pending_build_is_responsive_and_does_not_deploy(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    entered, proceed = threading.Event(), threading.Event()
    original = host._build
    def pause_build(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert proceed.wait(5)
        return result
    monkeypatch.setattr(host, "_build", pause_build)
    request_id = uuid.uuid4().hex
    job = api("POST", "/hil/start", {"request_id": request_id})
    try:
        assert entered.wait(2)
        started = time.monotonic()
        cancelled = api("POST", "/hil/cancel-start", {"request_id": request_id}, timeout=1)
        assert time.monotonic() - started < 1
        assert cancelled["operation"]["phase"] == "cancelling"
        assert api("GET", "/hil/state", timeout=1)["session"] is None
    finally:
        proceed.set()
    result = wait_job(api, job)
    assert result["operation"]["phase"] == "cancelled"
    assert agent.store.list() == []
    assert agent.session is agent.loaded is None


@pytest.mark.parametrize("stage", ["loaded", "started"])
def test_cancel_during_load_or_start_response_releases_all_owned_resources(management, monkeypatch, stage):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    entered, proceed = threading.Event(), threading.Event()
    if stage == "loaded":
        original = host._load_project
        def pause_load(*args, **kwargs):
            result = original(*args, **kwargs)
            entered.set()
            assert proceed.wait(5)
            return result
        monkeypatch.setattr(host, "_load_project", pause_load)
    else:
        original = host._handle
        def pause_start(method, path, body):
            result = original(method, path, body)
            if path == "/hil/start":
                entered.set()
                assert proceed.wait(5)
            return result
        monkeypatch.setattr(host, "_handle", pause_start)
    request_id = uuid.uuid4().hex
    job = api("POST", "/hil/start", {"request_id": request_id})
    try:
        assert entered.wait(3)
        assert agent.loaded is not None
        if stage == "started":
            assert agent.session is not None
        api("POST", "/hil/cancel-start", {"request_id": request_id}, timeout=1)
    finally:
        proceed.set()
    result = wait_job(api, job)
    assert result["operation"]["phase"] == "cancelled"
    assert "result" not in result
    assert host.session is agent.session is agent.loaded is None
    assert agent.models == {}
    assert host.application["loaded"] is None
    assert len(agent.store.list()) == 1  # Cancellation preserves installed files.
    assert all(not key.startswith("_") for key in result["operation"])


def test_cancel_completed_start_and_late_cancel_cannot_stop_new_session(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    first_id = uuid.uuid4().hex
    first = completed(api, api("POST", "/hil/start", {"request_id": first_id}))
    assert agent.session == first["session"]
    response = api("POST", "/hil/cancel-start", {"request_id": first_id})
    assert response["operation"]["phase"] == "cancelled"
    assert agent.session is agent.loaded is None
    second_id = uuid.uuid4().hex
    second = completed(api, api("POST", "/hil/start", {"request_id": second_id}))
    assert second["session"] != first["session"]
    api("POST", "/hil/cancel-start", {"request_id": first_id})
    assert agent.session == host.session == second["session"]
    assert api("GET", "/hil/state")["operation"]["request_id"] == second_id
    api("POST", "/hil/stop")


def test_cancel_old_start_does_not_unload_a_new_manual_selection(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    request_id = uuid.uuid4().hex
    started = completed(api, api("POST", "/hil/start", {"request_id": request_id}))
    api("POST", "/hil/stop", {"session": started["session"]})
    selected = completed(api, api("POST", "/hil/projects/load"))["loaded"]
    api("POST", "/hil/cancel-start", {"request_id": request_id})
    assert host.application["loaded"] == agent.loaded == selected


def test_cancelling_other_id_does_not_cancel_pending_start(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    entered, proceed = threading.Event(), threading.Event()
    original = host._build
    def pause_build(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(host, "_build", pause_build)
    request_id = uuid.uuid4().hex
    job = api("POST", "/hil/start", {"request_id": request_id})
    try:
        assert entered.wait(2)
        assert api("POST", "/hil/cancel-start", {"request_id": uuid.uuid4().hex})["operation"] is None
    finally:
        proceed.set()
    result = completed(api, job)
    assert host.session == agent.session == result["session"]
    api("POST", "/hil/stop")


@pytest.mark.parametrize("replacement_password", [None, "corrected-password"])
def test_network_confirmation_persists_verified_host_key_not_password(project_dir, tmp_path, monkeypatch, replacement_password):
    from botlab.hil import devices
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(profile_path))
    target = save_target(ssh_profile())
    host = HilHost(project_dir, "linux")
    transaction = uuid.uuid4().hex
    fingerprint = "SHA256:" + "a" * 43
    host.credentials["linux"] = {"password": "transient-password", "host_key": fingerprint}
    expected_password = replacement_password or "transient-password"
    host.network_pending[transaction] = {"target": target, "new_host": "192.0.2.11"}
    checked = []
    class VerifiedDevice:
        def __init__(self, config, credentials, progress):
            assert config["ssh"]["host"] == "192.0.2.11"
            assert credentials == {"password": expected_password, "host_key": fingerprint}
        def confirm_network(self, request):
            checked.append(request)
            return {"confirmed": True}
        def close(self):
            pass
    monkeypatch.setattr(devices, "LinuxDevice", VerifiedDevice)
    try:
        result = completed(host.handle, host.handle("POST", "/hil/devices/network/confirm", {
            "target": "linux", "transaction": transaction, "host": "192.0.2.11",
            **({"password": replacement_password} if replacement_password else {})}))
        assert result["confirmed"]
        assert checked == [transaction]
        saved = read_targets()[0]
        assert saved["ssh"]["host"] == "192.0.2.11"
        assert saved["ssh"]["host_key"] == fingerprint
        assert "transient-password" not in profile_path.read_text()
        assert expected_password not in profile_path.read_text()
        assert host.credentials["linux"]["password"] == expected_password
        assert transaction not in host.network_pending
    finally:
        host.close()


def test_delete_target_guard_rejects_other_connected_target_without_rpc(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/build"))
    deployed = completed(api, api("POST", "/hil/deploy"))
    project_id = deployed["project"]["id"]
    calls = []
    original = host._rpc
    def record_rpc(request, *args, **kwargs):
        calls.append(request["op"])
        return original(request, *args, **kwargs)
    monkeypatch.setattr(host, "_rpc", record_rpc)
    with pytest.raises(HTTPError) as rejected:
        api("POST", "/hil/projects/remove", {"target": "other-target", "project_id": project_id})
    assert rejected.value.code == 400
    assert "Target changed or disconnected" in json.load(rejected.value)["error"]
    assert calls == []
    assert len(agent.store.list()) == 1
    assert api("POST", "/hil/projects/remove", {"target": "cpu", "project_id": project_id})["removed"]
    assert calls == ["projects.remove", "projects.list"]
    assert agent.store.list() == []


def test_delete_target_guard_rejects_disconnected_target_without_rpc(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/build"))
    deployed = completed(api, api("POST", "/hil/deploy"))
    api("POST", "/hil/disconnect")
    calls = []
    monkeypatch.setattr(host, "_rpc", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(HTTPError) as rejected:
        api("POST", "/hil/projects/remove", {"target": "cpu", "project_id": deployed["project"]["id"]})
    assert rejected.value.code == 400
    assert "Target changed or disconnected" in json.load(rejected.value)["error"]
    assert calls == []
    assert len(agent.store.list()) == 1


@pytest.mark.parametrize("operation", ["inspect", "provision"])
def test_running_project_rejects_device_management_without_releasing_session(management, monkeypatch, operation):
    from botlab.hil import devices
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    started = completed(api, api("POST", "/hil/start"))
    session = started["session"]
    connection = host.sock
    loaded = host.application["loaded"].copy()
    host.device_info["cpu"] = {"agent_installed": True}
    def no_device(*args, **kwargs):
        pytest.fail("Device management must reject a running project before SSH")
    monkeypatch.setattr(devices, "LinuxDevice", no_device)
    rejected = wait_job(api, api("POST", "/hil/devices/" + operation, {"target": "cpu"}))
    assert rejected["operation"]["status"] == "failed"
    assert "Stop the project" in rejected["operation"]["error"]
    assert host.sock is connection
    assert host.session == agent.session == session
    assert host.application["loaded"] == agent.loaded == loaded
    assert host.device_info["cpu"]["agent_installed"]
    # The same real Agent lease still accepts a model request after rejection.
    model = host.application["built"]["models"][0]["sha256"]
    metadata = api("POST", "/hil/models", {"session": session, "sha256": model})
    port = metadata["inputs"][0]
    values = [float(index) for index in range(math.prod(port["shape"]))]
    result = api("POST", "/hil/infer", {"session": session, "seq": 0, "model": model,
        "inputs": {port["name"]: {"dtype": "float32", "shape": port["shape"], "data": values}}})
    assert result["calls"] == 1
    api("POST", "/hil/stop")


@pytest.mark.parametrize("inspection_fails", [False, True])
def test_idle_inspection_preserves_connection_and_loaded_project(management, monkeypatch, inspection_fails):
    from botlab.hil import devices
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/start"))
    api("POST", "/hil/stop")
    connection = host.sock
    loaded = host.application["loaded"].copy()
    # Represent the already-established loopback connection as an SSH target;
    # only the independent OS inspection is stubbed, never the Agent protocol.
    host.targets[0] = {**ssh_profile(), "id": "cpu"}
    host.device_info["cpu"] = {"hostname": "stale-device", "agent_installed": True}
    closed = []
    class InspectedDevice:
        def __init__(self, target, credentials, progress):
            self.target = target
        def inspect(self):
            assert "cpu" not in host.device_info
            if inspection_fails:
                raise HilError("SSH inspection failed")
            return {"hostname": "current-device", "agent_installed": False}
        def close(self):
            closed.append(True)
    monkeypatch.setattr(devices, "LinuxDevice", InspectedDevice)
    checked = wait_job(api, api("POST", "/hil/devices/inspect", {"target": "cpu"}))
    assert checked["operation"]["status"] == ("failed" if inspection_fails else "succeeded")
    assert host.sock is connection
    assert host.selected_target == "cpu"
    assert host.application["loaded"] == agent.loaded == loaded
    assert host.session is agent.session is None
    if inspection_fails:
        assert "cpu" not in api("GET", "/hil/state")["device_info"]
    else:
        assert host.device_info["cpu"]["hostname"] == "current-device"
        assert not host.device_info["cpu"]["agent_installed"]
    assert closed == [True]
    assert api("GET", "/hil/projects")["loaded"] == loaded


def test_inspecting_another_target_preserves_current_connection(management, monkeypatch):
    from botlab.hil import devices
    host, _, api = management
    api("POST", "/hil/targets", ssh_profile())
    api("POST", "/hil/connect", {"target": "cpu"})
    connection = host.sock
    def no_device(*args, **kwargs):
        pytest.fail("Checking another target must be refused before SSH")
    monkeypatch.setattr(devices, "LinuxDevice", no_device)
    rejected = wait_job(api, api("POST", "/hil/devices/inspect", {"target": "linux"}))
    assert rejected["operation"]["status"] == "failed"
    assert "Disconnect before checking another target" in rejected["operation"]["error"]
    assert host.selected_target == "cpu"
    assert host.sock is connection
    assert api("GET", "/hil/projects")["projects"] == []


def test_idle_provision_disconnects_before_installation(management, monkeypatch):
    from botlab.hil import devices
    host, _, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/start"))
    api("POST", "/hil/stop")
    host.targets[0] = {**ssh_profile(), "id": "cpu"}
    installed = []
    class ProvisionedDevice:
        def __init__(self, target, credentials, progress):
            self.target = target
        def provision(self):
            assert host.sock is None
            assert host.session is None
            assert host.application["loaded"] is None
            installed.append(True)
            return {"installed": True, "device": {"agent_installed": False}}
        def close(self):
            pass
    monkeypatch.setattr(devices, "LinuxDevice", ProvisionedDevice)
    result = completed(api, api("POST", "/hil/devices/provision", {"target": "cpu"}))
    assert result["installed"]
    assert installed == [True]
    assert not api("GET", "/hil/state")["connected"]


def test_saved_password_is_transient_and_survives_fingerprint_acceptance(project_dir, tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(profile_path))
    host = HilHost(project_dir)
    try:
        profile = ssh_profile()
        saved = host.handle("POST", "/hil/targets", {
            **profile, "credentials": {"password": "private-password"}})
        assert saved["credential_status"]["linux"] == {"password": True}
        assert saved["capabilities"]["transientCredentials"] == 1
        assert saved["capabilities"]["targetProjects"] == 1
        assert "private-password" not in json.dumps(saved)
        assert "credentials" not in profile_path.read_text()
        assert "private-password" not in profile_path.read_text()
        profile = {**read_targets()[0], "name": "Renamed board"}
        profile["ssh"] = {**profile["ssh"], "host_key": "SHA256:" + "a" * 43}
        updated = host.handle("POST", "/hil/targets", profile)
        assert updated["credential_status"]["linux"] == {"password": True}
        assert host.credentials["linux"]["password"] == "private-password"
        fresh = HilHost(project_dir)
        assert fresh.state()["credential_status"]["linux"] == {"password": False}
        fresh.close()
    finally:
        host.close()
    assert host.credentials == host.credential_targets == {}


@pytest.mark.parametrize("credentials", [None, [], {"password": 123}, {"password": "bad\x00value"},
                                         {"token": "secret"}, {"host_key": "bad"}])
def test_invalid_saved_transient_credentials_do_not_write_target(project_dir, tmp_path, monkeypatch, credentials):
    profile_path = tmp_path / "profiles.json"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(profile_path))
    host = HilHost(project_dir)
    try:
        with pytest.raises(HilError, match="credential|fingerprint"):
            host.handle("POST", "/hil/targets", {**ssh_profile(), "credentials": credentials})
        assert not profile_path.exists()
        assert host.credentials == {}
    finally:
        host.close()


@pytest.mark.parametrize("change", [
    {"ssh": {"host": "192.0.2.11", "user": "robot"}},
    {"ssh": {"host": "192.0.2.10", "user": "other"}},
    {"ssh": {"host": "192.0.2.10", "user": "robot", "port": 2222}},
    {"ssh": {"host": "192.0.2.10", "user": "robot", "identity_file": "~/.ssh/another"}},
    {"root": "~/another-botlab"}, {"backend": "another"}, {"agent_port": 50053},
])
def test_connection_identity_edits_invalidate_saved_password(project_dir, tmp_path, monkeypatch, change):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "profiles.json"))
    host = HilHost(project_dir)
    try:
        host.handle("POST", "/hil/targets", {**ssh_profile(), "credentials": {"password": "old-secret"}})
        response = host.handle("POST", "/hil/targets", {**ssh_profile(), **change})
        assert response["credential_status"]["linux"] == {"password": False}
        assert "linux" not in host.credentials
        response = host.handle("POST", "/hil/targets", {
            **ssh_profile(), **change, "credentials": {"password": "new-secret"}})
        assert response["credential_status"]["linux"] == {"password": True}
        assert host.credentials["linux"] == {"password": "new-secret"}
    finally:
        host.close()


def test_saving_unconnected_target_or_same_connected_password_preserves_main(management):
    host, _, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    connection = host.sock
    api("POST", "/hil/targets", {**ssh_profile(), "credentials": {"password": "other-password"}})
    assert host.sock is connection
    assert host.selected_target == "cpu"
    # Model the existing main connection's SSH configuration without new I/O.
    target = save_target({**ssh_profile(), "id": "cpu"})
    host.targets = read_targets()
    response = api("POST", "/hil/targets", {**target, "credentials": {"password": "corrected-password"}})
    assert response["connected"]
    assert response["credential_status"]["cpu"]["password"]
    assert host.sock is connection
    assert host.selected_target == "cpu"
    with pytest.raises(HTTPError):
        api("POST", "/hil/targets", {**target, "ssh": {**target["ssh"], "host": "192.0.2.11"}})
    assert host.sock is connection
    assert host.credentials["cpu"]["password"] == "corrected-password"
    assert "corrected-password" not in json.dumps(api("GET", "/hil/state"))


def test_scoped_scan_disconnected_does_not_select_or_connect_main(management):
    host, agent, api = management
    agent.store.deploy(artifact())
    host.selected_target = None
    inventory = completed(api, api("POST", "/hil/targets/projects", {"target": "cpu"}))
    assert inventory["target"] == "cpu"
    assert [project["id"] for project in inventory["projects"]] == ["project-one"]
    assert inventory["loaded"] is None
    assert host.selected_target is None
    assert host.sock is None
    assert host.application == {"built": None, "loaded": None, "projects": []}
    missing = wait_job(api, api("POST", "/hil/targets/projects"))
    assert missing["operation"]["status"] == "failed"
    assert "Choose a configured target" in missing["operation"]["error"]


def test_scoped_other_target_inventory_and_delete_preserve_main(management, tmp_path):
    from tests.test_hil_projects import Backend
    host, main_agent, api = management
    other_state = AgentState(tmp_path / "other-cache", Backend(), root=tmp_path / "other-target")
    other_agent = AgentServer(("127.0.0.1", 0), other_state)
    thread = threading.Thread(target=other_agent.serve_forever, daemon=True)
    thread.start()
    try:
        other = save_target({"id": "other", "name": "Other board", "backend": "onnxruntime",
                             "endpoint": f"tcp://127.0.0.1:{other_agent.server_address[1]}"})
        host.targets = read_targets()
        other_state.store.deploy(artifact("other-project"))
        other_state.store.deploy(artifact("other-project", revision="second"))
        api("POST", "/hil/connect", {"target": "cpu"})
        completed(api, api("POST", "/hil/start"))
        api("POST", "/hil/stop")
        original_connection = host.sock
        original_application = copy.deepcopy(host.application)
        original_loaded = copy.deepcopy(main_agent.loaded)
        scanned = completed(api, api("POST", "/hil/targets/projects", {"target": "other"}))
        assert [project["id"] for project in scanned["projects"]] == ["other-project"]
        assert len(scanned["projects"][0]["releases"]) == 2
        deleted = completed(api, api("POST", "/hil/targets/projects/remove", {
            "target": "other", "project_id": "other-project", "expected_target": other}))
        assert deleted == {"target": "other", "projects": [], "loaded": None}
        assert other_state.store.list() == []
        assert host.selected_target == "cpu"
        assert host.sock is original_connection
        assert host.application == original_application
        assert main_agent.loaded == original_loaded
        assert host.session is None
    finally:
        other_agent.shutdown()
        other_agent.server_close()
        thread.join()


@pytest.mark.parametrize("stale", ["missing", "wrong-id", "old-config"])
def test_scoped_delete_requires_confirmed_target_config(management, monkeypatch, stale):
    host, agent, api = management
    agent.store.deploy(artifact())
    expected = copy.deepcopy(host.targets[0])
    if stale == "wrong-id":
        expected["id"] = "other"
    elif stale == "old-config":
        expected["endpoint"] = "tcp://127.0.0.1:1"
    def unexpected_connection(*args, **kwargs):
        pytest.fail("Stale deletion must fail before connecting")
    monkeypatch.setattr("botlab.hil.host.socket.create_connection", unexpected_connection)
    response = wait_job(host.handle, host.handle("POST", "/hil/targets/projects/remove", {
        "target": "cpu", "project_id": "project-one",
        **({"expected_target": expected} if stale != "missing" else {})}))
    assert response["operation"]["status"] == "failed"
    assert "Scan the target and confirm" in response["operation"]["error"]
    assert len(agent.store.list()) == 1


def test_scoped_loaded_project_is_protected_without_unloading(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/start"))
    api("POST", "/hil/stop")
    loaded = copy.deepcopy(agent.loaded)
    connection = host.sock
    inventory = completed(api, api("POST", "/hil/targets/projects", {"target": "cpu"}))
    assert inventory["loaded"] == loaded
    result = wait_job(api, api("POST", "/hil/targets/projects/remove", {
        "target": "cpu", "project_id": loaded["project_id"], "expected_target": host.targets[0]}))
    assert result["operation"]["status"] == "failed"
    assert "Unload the project before deleting its loaded release" in result["operation"]["error"]
    assert host.sock is connection
    assert host.application["loaded"] == agent.loaded == loaded
    assert len(agent.store.list()) == 1


def test_scoped_scan_refuses_active_hil_before_network_io(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    session = completed(api, api("POST", "/hil/start"))["session"]
    connection = host.sock
    result = wait_job(api, api("POST", "/hil/targets/projects", {"target": "cpu"}))
    assert result["operation"]["status"] == "failed"
    assert "Stop the project" in result["operation"]["error"]
    assert host.sock is connection
    assert host.session == agent.session == session


@pytest.mark.parametrize("missing_agent", [False, True])
def test_scoped_ssh_scan_uses_transient_auth_and_closes_temporary_tunnel(management, monkeypatch, missing_agent):
    from botlab.hil import devices
    host, agent, api = management
    endpoint = host.targets[0]["endpoint"]
    from botlab.hil.protocol import endpoint_address
    address = endpoint_address(endpoint)
    api("POST", "/hil/targets", {**ssh_profile(), "credentials": {"password": "scan-password"}})
    closed = []
    class Tunnel:
        def __init__(self):
            self.address = address
        def close(self):
            closed.append("tunnel")
    class Device:
        def __init__(self, target, credentials, progress):
            assert target["id"] == "linux"
            assert credentials["password"] == "scan-password"
            self.target = target
            self.progress = progress
        def open_tunnel(self):
            if missing_agent:
                self.progress({"stage": "connect", "message": "scan-password"})
                raise HilError("Agent is not installed; scan-password")
            return Tunnel()
        def close(self):
            closed.append("device")
    monkeypatch.setattr(devices, "LinuxDevice", Device)
    result = wait_job(api, api("POST", "/hil/targets/projects", {"target": "linux"}))
    assert result["operation"]["status"] == ("failed" if missing_agent else "succeeded")
    assert closed == (["device"] if missing_agent else ["tunnel", "device"])
    assert "scan-password" not in json.dumps(result)
    assert "scan-password" not in json.dumps(api("GET", "/hil/state"))
    if missing_agent:
        assert "Agent is not installed" in result["operation"]["error"]
    else:
        assert result["result"] == {"target": "linux", "projects": [], "loaded": None}
    assert host.sock is None
    assert host.selected_target == "linux"


def test_scoped_failed_inventory_does_not_release_main_loaded_project(management, monkeypatch):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    completed(api, api("POST", "/hil/start"))
    api("POST", "/hil/stop")
    loaded = copy.deepcopy(agent.loaded)
    connection = host.sock
    def failed_inventory():
        raise HilError("Invalid project inventory")
    monkeypatch.setattr(agent.store, "list", failed_inventory)
    response = wait_job(api, api("POST", "/hil/targets/projects", {"target": "cpu"}))
    assert response["operation"]["kind"] == "projects-scan"
    assert response["operation"]["status"] == "failed"
    assert "Invalid project inventory" in response["operation"]["error"]
    assert host.sock is connection
    assert host.application["loaded"] == agent.loaded == loaded
    assert host.selected_target == "cpu"
    assert host.session is None


def scoped_selection(host, value):
    manifest = value["manifest"]
    return {"target": "cpu", "expected_target": copy.deepcopy(host.targets[0]),
            "project_id": manifest["project"]["id"], "build_id": manifest["build_id"]}


def test_scoped_load_switches_exact_versions_and_projects(management, monkeypatch):
    host, agent, api = management
    old, new, other = artifact(), artifact(revision="second"), artifact("other-project")
    for value in (old, new, other):
        agent.store.deploy(value)
    api("POST", "/hil/connect", {"target": "cpu"})
    connection = host.sock
    def unexpected_build_or_deploy(*args, **kwargs):
        pytest.fail("Selecting a deployed version must not build or deploy")
    monkeypatch.setattr(host, "_build", unexpected_build_or_deploy)
    monkeypatch.setattr(host, "_deploy", unexpected_build_or_deploy)
    for value in (new, old, other):
        selection = scoped_selection(host, value)
        loaded = completed(api, api("POST", "/hil/projects/load", selection))["loaded"]
        assert loaded["project_id"] == selection["project_id"]
        assert loaded["build_id"] == selection["build_id"]
        assert host.application["loaded"] == agent.loaded == loaded
        project = next(item for item in host.application["projects"] if item["id"] == loaded["project_id"])
        assert project["current_build_id"] == loaded["build_id"]
        assert project["state"] == "loaded"
        assert agent.store.current(loaded["project_id"]) == loaded["build_id"]
        assert host.sock is connection
        assert host.session is None
    assert host.artifact is None
    assert len(agent.store.list()) == 2
    assert host.state()["capabilities"]["deploymentVersions"] == 1
    assert agent.device()["capabilities"]["deploymentVersions"] == 1


@pytest.mark.parametrize("stale", ["missing", "different-target", "endpoint", "name", "extra-setting"])
def test_scoped_load_requires_exact_connected_target_snapshot(management, stale):
    host, agent, api = management
    loaded = prepare_loaded_project(api)
    value = artifact("other-project")
    agent.store.deploy(value)
    body = scoped_selection(host, value)
    if stale == "missing":
        body.pop("expected_target")
    elif stale == "different-target":
        body["target"] = "other"
    elif stale == "endpoint":
        body["expected_target"]["endpoint"] = "tcp://127.0.0.1:1"
    elif stale == "name":
        body["expected_target"]["name"] = "Old display name"
    else:
        body["expected_target"]["unexpected"] = True
    connection, owner = host.sock, agent.project_owner
    result = wait_job(api, api("POST", "/hil/projects/load", body))
    assert result["operation"]["status"] == "failed"
    assert "Target " in result["operation"]["error"]
    assert host.sock is connection
    assert agent.project_owner is owner
    assert host.application["loaded"] == agent.loaded == loaded


def test_scoped_load_requires_target_connection(management):
    host, agent, api = management
    value = artifact()
    agent.store.deploy(value)
    result = wait_job(api, api("POST", "/hil/projects/load", scoped_selection(host, value)))
    assert result["operation"]["status"] == "failed"
    assert host.sock is None
    assert agent.loaded is None


@pytest.mark.parametrize("stale", ["project", "version", "empty-version", "null-version"])
def test_scoped_load_stale_selection_preserves_existing_lease(management, stale):
    host, agent, api = management
    loaded = prepare_loaded_project(api)
    value = artifact("other-project")
    agent.store.deploy(value)
    body = scoped_selection(host, value)
    if stale == "project":
        body["project_id"] = "removed-project"
    elif stale == "version":
        body["build_id"] = "a" * 64
    else:
        body["build_id"] = "" if stale == "empty-version" else None
    connection, owner = host.sock, agent.project_owner
    result = wait_job(api, api("POST", "/hil/projects/load", body))
    assert result["operation"]["status"] == "failed"
    assert host.sock is connection
    assert agent.project_owner is owner
    assert host.application["loaded"] == agent.loaded == loaded
    assert agent.store.current(loaded["project_id"]) == loaded["build_id"]
    assert api("GET", "/hil/projects")["loaded"] == loaded


@pytest.mark.parametrize("path", ["/hil/projects/load", "/hil/targets/projects/remove"])
def test_scoped_version_mutation_rejects_running_session(management, path):
    host, agent, api = management
    value = artifact("inactive-project")
    agent.store.deploy(value)
    api("POST", "/hil/connect", {"target": "cpu"})
    session = completed(api, api("POST", "/hil/start"))["session"]
    connection, loaded = host.sock, copy.deepcopy(agent.loaded)
    result = wait_job(api, api("POST", path, scoped_selection(host, value)))
    assert result["operation"]["status"] == "failed"
    assert "Stop" in result["operation"]["error"]
    assert host.sock is connection
    assert host.session == agent.session == session
    assert host.application["loaded"] == agent.loaded == loaded
    assert len(agent.store.list()) == 2


def test_scoped_remove_inactive_version_preserves_loaded_version_and_refreshes_inventory(management):
    host, agent, api = management
    old, new = artifact(), artifact(revision="second")
    for value in (old, new):
        agent.store.deploy(value)
    api("POST", "/hil/connect", {"target": "cpu"})
    loaded = completed(api, api("POST", "/hil/projects/load", scoped_selection(host, new)))["loaded"]
    connection, owner = host.sock, agent.project_owner
    result = completed(api, api("POST", "/hil/targets/projects/remove", scoped_selection(host, old)))
    assert [item["build_id"] for item in result["projects"][0]["releases"]] == [new["manifest"]["build_id"]]
    assert host.application["projects"] == result["projects"]
    assert result["loaded"] == host.application["loaded"] == agent.loaded == loaded
    assert host.sock is connection
    assert agent.project_owner is owner
    assert agent.store.current("project-one") == loaded["build_id"]


@pytest.mark.parametrize("whole_project", [False, True])
def test_scoped_remove_last_version_or_whole_inactive_project(management, whole_project):
    host, agent, api = management
    loaded = prepare_loaded_project(api)
    other = artifact("inactive-project")
    agent.store.deploy(other)
    if whole_project:
        agent.store.deploy(artifact("inactive-project", revision="second"))
    body = scoped_selection(host, other)
    if whole_project:
        body.pop("build_id")
    connection = host.sock
    result = completed(api, api("POST", "/hil/targets/projects/remove", body))
    assert [item["id"] for item in result["projects"]] == [loaded["project_id"]]
    assert host.application["projects"] == result["projects"]
    assert host.application["loaded"] == agent.loaded == loaded
    assert host.sock is connection
    assert not agent.store.project_path("inactive-project").exists()


def test_scoped_remove_loaded_version_is_rejected_without_unloading(management):
    host, agent, api = management
    loaded = prepare_loaded_project(api)
    connection, owner = host.sock, agent.project_owner
    result = wait_job(api, api("POST", "/hil/targets/projects/remove", {
        "target": "cpu", "expected_target": host.targets[0],
        "project_id": loaded["project_id"], "build_id": loaded["build_id"]}))
    assert result["operation"]["status"] == "failed"
    assert "Unload" in result["operation"]["error"]
    assert host.sock is connection
    assert agent.project_owner is owner
    assert host.application["loaded"] == agent.loaded == loaded
    assert len(agent.store.list()) == 1


@pytest.mark.parametrize("build_id", [None, "", " ", 1, "invalid", "../release"])
def test_scoped_remove_invalid_version_never_deletes_whole_project(management, build_id):
    host, agent, api = management
    value = artifact()
    agent.store.deploy(value)
    result = wait_job(api, api("POST", "/hil/targets/projects/remove", {
        **scoped_selection(host, value), "build_id": build_id}))
    assert result["operation"]["status"] == "failed"
    assert "valid deployed version" in result["operation"]["error"]
    assert len(agent.store.list()) == 1


def test_scoped_remove_inactive_current_version_keeps_loaded_older_version(management):
    host, agent, api = management
    old, new = artifact(), artifact(revision="second")
    agent.store.deploy(old)
    api("POST", "/hil/connect", {"target": "cpu"})
    loaded = completed(api, api("POST", "/hil/projects/load", scoped_selection(host, old)))["loaded"]
    # Deployment can update the project's default without changing its loaded version.
    agent.store.deploy(new)
    assert agent.store.current("project-one") == new["manifest"]["build_id"]
    result = completed(api, api("POST", "/hil/targets/projects/remove", scoped_selection(host, new)))
    assert result["projects"][0]["current_build_id"] == loaded["build_id"]
    assert host.application["loaded"] == agent.loaded == loaded
    assert len(result["projects"][0]["releases"]) == 1


def test_scoped_load_deleted_after_preflight_preserves_previous_selection(management, monkeypatch):
    host, agent, api = management
    loaded = prepare_loaded_project(api)
    candidate = artifact("inactive-project")
    agent.store.deploy(candidate)
    original_scan = host._target_projects
    def delete_after_scan(*args, **kwargs):
        inventory = original_scan(*args, **kwargs)
        agent.store.remove("inactive-project", candidate["manifest"]["build_id"])
        return inventory
    monkeypatch.setattr(host, "_target_projects", delete_after_scan)
    connection, owner = host.sock, agent.project_owner
    result = wait_job(api, api("POST", "/hil/projects/load", scoped_selection(host, candidate)))
    assert result["operation"]["status"] == "failed"
    assert "not deployed" in result["operation"]["error"]
    assert host.sock is connection
    assert agent.project_owner is owner
    assert host.application["loaded"] == agent.loaded == loaded
    assert host.session is agent.session is None
    assert api("GET", "/hil/projects")["loaded"] == loaded
    # The preserved connection remains usable, including its execution lease.
    session = completed(api, api("POST", "/hil/start-loaded", loaded_start_body(host, loaded)))["session"]
    assert host.session == agent.session == session


@pytest.mark.parametrize("single_version", [False, True])
def test_scoped_single_version_delete_requires_agent_capability(management, monkeypatch, single_version):
    host, agent, api = management
    value = artifact()
    agent.store.deploy(value)
    original_device = agent.device
    def legacy_device():
        result = original_device()
        result["capabilities"].pop("deploymentVersions")
        return result
    monkeypatch.setattr(agent, "device", legacy_device)
    body = scoped_selection(host, value)
    if not single_version:
        body.pop("build_id")
    result = wait_job(api, api("POST", "/hil/targets/projects/remove", body))
    if single_version:
        assert result["operation"]["status"] == "failed"
        assert "Update" in result["operation"]["error"]
        assert len(agent.store.list()) == 1
    else:
        assert result["operation"]["status"] == "succeeded"
        assert agent.store.list() == []


def test_agent_load_refusal_preserves_active_session_and_connection(management):
    host, agent, api = management
    api("POST", "/hil/connect", {"target": "cpu"})
    session = completed(api, api("POST", "/hil/start"))["session"]
    connection, loaded = host.sock, copy.deepcopy(agent.loaded)
    with pytest.raises(HilError, match="Stop"):
        host._rpc({"op": "projects.load", "project_id": loaded["project_id"],
                   "build_id": loaded["build_id"]})
    assert host.sock is connection
    assert host.session == agent.session == session
    assert host.application["loaded"] == agent.loaded == loaded
    assert api("POST", "/hil/stop", {"session": session})["session"] is None


def test_lease_preserved_response_cannot_suppress_other_rpc_failures(management, monkeypatch):
    host, agent, api = management
    prepare_loaded_project(api)
    original_dispatch = agent.dispatch
    def invalid_unload_response(owner, request):
        if request["op"] == "projects.unload":
            return {"ok": False, "error": "Unexpected unload failure", "lease_preserved": True}
        return original_dispatch(owner, request)
    monkeypatch.setattr(agent, "dispatch", invalid_unload_response)
    with pytest.raises(HilError, match="Unexpected unload failure"):
        host._rpc({"op": "projects.unload"})
    assert host.sock is None
    assert host.application["loaded"] is None


def test_scoped_load_requires_agent_capability_without_changing_selection(management, monkeypatch):
    host, agent, api = management
    original_device = agent.device
    def legacy_device():
        result = original_device()
        result["capabilities"].pop("deploymentVersions")
        return result
    monkeypatch.setattr(agent, "device", legacy_device)
    loaded = prepare_loaded_project(api)
    candidate = artifact("inactive-project")
    agent.store.deploy(candidate)
    connection, owner = host.sock, agent.project_owner
    result = wait_job(api, api("POST", "/hil/projects/load", scoped_selection(host, candidate)))
    assert result["operation"]["status"] == "failed"
    assert "Update" in result["operation"]["error"]
    assert host.sock is connection
    assert agent.project_owner is owner
    assert host.application["loaded"] == agent.loaded == loaded
