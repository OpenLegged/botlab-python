"""Persistent, robot-independent deployment and inference isolation contracts."""
import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from botlab._project import STANDARD_PYTHON_MAIN
from botlab.hil.agent import AgentState
from botlab.hil.artifacts import ProjectStore, build_digest, build_project, validate_artifact
from botlab.hil.protocol import HilError
from botlab.hil.worker import WorkerBackend

MODEL = (Path(__file__).parent / "assets" / "identity.onnx").read_bytes()


class Backend:
    def describe(self):
        return {"backend": "onnxruntime", "provider": "test"}

    def load(self, path):
        assert path.read_bytes() == MODEL
        return object(), [{"name": "input", "dtype": "float32", "shape": [1]}], [
            {"name": "output", "dtype": "float32", "shape": [1]}]

    def infer(self, handle, inputs):
        return {"output": inputs["input"]}

    def close(self, handle):
        pass


class HangingBackend(Backend):
    def infer(self, handle, inputs):
        time.sleep(10)
        return super().infer(handle, inputs)


def hanging_backend():
    return HangingBackend()


def artifact(project_id="project-one", name="Example", revision="first"):
    digest = hashlib.sha256(MODEL).hexdigest()
    model_path = f"models/{digest}.onnx"
    manifest = {"schema": 1, "project": {"id": project_id, "name": name},
                "execution_kind": "model-service", "backend": "onnxruntime",
                "project_digest": hashlib.sha256(revision.encode()).hexdigest(),
                "control_hz": 50, "physics_hz": 200,
                "models": [{"id": "policy", "sha256": digest, "path": model_path,
                            "size": len(MODEL), "config": {"inputName": "input"}}]}
    manifest["build_id"] = build_digest(manifest)
    return {"manifest": manifest, "files": {model_path: base64.b64encode(MODEL).decode()}}


@pytest.fixture
def state(tmp_path):
    return AgentState(tmp_path / "cache", Backend(), root=tmp_path / "botlab")


def deploy(state, owner, value):
    return state.dispatch(owner, {"op": "projects.deploy", "artifact": value})


def load(state, owner, value):
    return state.dispatch(owner, {"op": "projects.load", "project_id": value["manifest"]["project"]["id"],
                                  "build_id": value["manifest"]["build_id"]})


def test_persistent_multiple_projects_and_idempotent_deployment(state):
    owner = object()
    first, second = artifact(), artifact("second", "Example")
    assert not deploy(state, owner, first)["already_deployed"]
    assert deploy(state, owner, first)["already_deployed"]
    deploy(state, owner, second)
    assert len(state.dispatch(owner, {"op": "projects.list"})["projects"]) == 2
    restarted = AgentState(state.models_dir, Backend(), root=state.store.root)
    assert restarted.loaded is None
    assert len(restarted.dispatch(owner, {"op": "projects.list"})["projects"]) == 2
    assert not list(state.store.staging.iterdir())
    result = subprocess.run([sys.executable, "-c",
        "import json,sys; from pathlib import Path; from botlab.hil.artifacts import ProjectStore; "
        "print(json.dumps(ProjectStore(Path(sys.argv[1])).list()))", str(state.store.root)],
        capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)) == 2


def test_retry_repairs_initial_selection_after_interrupted_publish(state):
    owner, value = object(), artifact()
    deploy(state, owner, value)
    project = state.store.project_path("project-one")
    (project / "current").unlink()
    (project / "project.json").unlink()
    assert deploy(state, owner, value)["already_deployed"]
    assert state.store.current("project-one") == value["manifest"]["build_id"]


def test_stop_unload_remove_and_run_log_are_separate(state):
    owner, value = object(), artifact()
    deploy(state, owner, value)
    selection = load(state, owner, value)["loaded"]
    session = state.dispatch(owner, {"op": "start"})["session"]
    # A deployed model loads from the selected release, without browser bytes.
    digest = value["manifest"]["models"][0]["sha256"]
    state.dispatch(owner, {"op": "load", "session": session, "sha256": digest})
    result = state.dispatch(owner, {"op": "infer", "session": session, "seq": 0,
        "model": digest, "inputs": {"input": {"dtype": "float32", "shape": [1], "data": [3]}}})
    assert result["outputs"]["output"]["data"] == [3]
    state.dispatch(owner, {"op": "stop", "session": session})
    assert state.loaded == selection
    assert state.models == {}
    with pytest.raises(HilError, match="Unload"):
        state.dispatch(owner, {"op": "projects.remove", "project_id": "project-one"})
    state.dispatch(owner, {"op": "projects.unload"})
    assert len(state.store.list()) == 1
    records = [json.loads(line) for line in next(state.store.projects.rglob("*.jsonl")).read_text().splitlines()]
    assert [item["event"] for item in records] == ["started", "stopped"]
    state.dispatch(owner, {"op": "projects.remove", "project_id": "project-one"})
    assert state.store.list() == []


def test_load_rejects_undeployed_model_and_changed_release(state):
    owner, value = object(), artifact()
    deploy(state, owner, value)
    load(state, owner, value)
    session = state.dispatch(owner, {"op": "start"})["session"]
    with pytest.raises(HilError, match="not part of the loaded deployment"):
        state.dispatch(owner, {"op": "load", "session": session, "sha256": "a" * 64,
                               "data": base64.b64encode(b"different").decode()})
    state.dispatch(owner, {"op": "stop", "session": session})
    model_file = next(state.store.projects.rglob("*.onnx"))
    model_file.write_bytes(bytes([MODEL[0] ^ 1]) + MODEL[1:])
    with pytest.raises(HilError, match="SHA-256"):
        load(state, owner, value)


def test_rollback_and_connection_ownership(state):
    owner, other = object(), object()
    old, new = artifact(), artifact(revision="second")
    deploy(state, owner, old)
    deploy(state, owner, new)
    assert state.store.current("project-one") == new["manifest"]["build_id"]
    load(state, owner, old)
    assert state.store.current("project-one") == old["manifest"]["build_id"]
    for operation in ({"op": "projects.load", "project_id": "project-one"},
                      {"op": "projects.unload"}, {"op": "start"},
                      {"op": "projects.deploy", "artifact": new}):
        with pytest.raises(HilError, match="another connection"):
            state.dispatch(other, operation)
    assert state.dispatch(other, {"op": "projects.list"})["loaded"] is not None
    session = state.dispatch(owner, {"op": "start"})["session"]
    with pytest.raises(HilError, match="Stop"):
        load(state, owner, new)
    state.dispatch(owner, {"op": "stop", "session": session})
    state.disconnect(owner)
    assert state.loaded is None
    assert load(state, other, new)["loaded"]["build_id"] == new["manifest"]["build_id"]


@pytest.mark.parametrize("mutation", ["path", "project", "hash", "extra", "external", "base64"])
def test_invalid_artifact_never_changes_existing_deployment(state, mutation):
    owner, valid = object(), artifact()
    deploy(state, owner, valid)
    value = copy.deepcopy(artifact(revision="second"))
    manifest, files = value["manifest"], value["files"]
    model = manifest["models"][0]
    if mutation == "path":
        model["path"] = "../../escape.onnx"
    elif mutation == "project":
        manifest["project"]["id"] = "../escape"
    elif mutation == "hash":
        files[model["path"]] = base64.b64encode(b"x" * len(MODEL)).decode()
    elif mutation == "extra":
        files["startup.py"] = base64.b64encode(b"unrelated").decode()
    elif mutation == "base64":
        files[model["path"]] = "!"
    else:
        external = b"\x3a\x03\x2a\x01\x68"
        digest = hashlib.sha256(external).hexdigest()
        model.update(sha256=digest, path=f"models/{digest}.onnx", size=len(external))
        value["files"] = {model["path"]: base64.b64encode(external).decode()}
    manifest["build_id"] = build_digest(manifest)
    with pytest.raises(HilError):
        deploy(state, owner, value)
    assert state.store.current("project-one") == valid["manifest"]["build_id"]
    assert len(state.store.list()[0]["releases"]) == 1
    assert not list(state.store.staging.iterdir())


def test_symlink_directory_is_not_followed(tmp_path):
    store = ProjectStore(tmp_path / "botlab")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.projects / "project-one").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HilError, match="symlinks"):
        store.deploy(artifact())
    assert not list(outside.iterdir())


def test_build_only_models_stable_identity_and_controller_sensitive(tmp_path):
    from tests.test_project import CONTROLLER, PRESET
    project = tmp_path / "original"
    project.mkdir()
    (project / "main.py").write_text(STANDARD_PYTHON_MAIN)
    (project / "controller.py").write_text(CONTROLLER)
    (project / "robot.preset.yaml").write_text(PRESET)
    (project / "assets/robot").mkdir(parents=True)
    (project / "assets/models").mkdir()
    (project / "assets/robot/scene.xml").write_text("<mujoco/>")
    (project / "assets/models/policy.onnx").write_bytes(MODEL)
    (project / "credentials.txt").write_text("never deploy")
    (project / "assets/models/unrelated.onnx").write_bytes(b"not declared")
    first = build_project(project)
    assert build_project(project) == first
    assert len(first["files"]) == 1
    renamed = tmp_path / "renamed"
    shutil.move(project, renamed)
    assert build_project(renamed) == first
    (renamed / "controller.py").write_text(CONTROLLER.replace("Test Duck", "Another name"))
    changed = build_project(renamed)
    assert changed["manifest"]["project"]["id"] == first["manifest"]["project"]["id"]
    assert changed["manifest"]["build_id"] != first["manifest"]["build_id"]
    assert changed["files"] == first["files"]
    validate_artifact(changed)


def test_build_accepts_matching_browser_sources_and_crlf(project_dir):
    sources = {name: (project_dir / name).read_text()
               for name in ("main.py", "controller.py", "robot.preset.yaml")}
    expected = build_project(project_dir)
    assert build_project(project_dir, expected_sources=sources) == expected
    windows = {name: source.replace("\n", "\r\n") for name, source in sources.items()}
    assert build_project(project_dir, expected_sources=windows) == expected


def test_build_reads_same_size_source_edits_despite_stale_bytecode(project_dir):
    from botlab._project import load_project
    # Simulate bytecode created by another SDK command before a rapid edit.
    load_project(project_dir)
    source = project_dir / "controller.py"
    metadata = source.stat()
    first = build_project(project_dir)
    previous_prefix, previous_write = sys.pycache_prefix, sys.dont_write_bytecode
    source.write_text(source.read_text().replace("control_hz=50", "control_hz=25"))
    os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    second = build_project(project_dir)
    assert first["manifest"]["control_hz"] == 50
    assert second["manifest"]["control_hz"] == 25
    assert second["manifest"]["build_id"] != first["manifest"]["build_id"]
    assert (sys.pycache_prefix, sys.dont_write_bytecode) == (previous_prefix, previous_write)


def test_failed_build_restores_interpreter_bytecode_settings(project_dir):
    source = project_dir / "controller.py"
    source.write_text("invalid python!\n")
    previous = sys.pycache_prefix, sys.dont_write_bytecode
    with pytest.raises(SyntaxError):
        build_project(project_dir)
    assert (sys.pycache_prefix, sys.dont_write_bytecode) == previous


@pytest.mark.parametrize("first_build", [True, False])
def test_hil_build_and_release_compile_share_import_lock(project_dir, monkeypatch, first_build):
    from botlab import _project
    entered, proceed, second_started, second_entered = (threading.Event() for _ in range(4))
    original = _project._import_controller
    prefixes = []
    errors, results = [], []
    def gated_import(*args):
        prefixes.append((sys.pycache_prefix, sys.dont_write_bytecode))
        if len(prefixes) == 1:
            entered.set()
            assert proceed.wait(5)
        else:
            second_entered.set()
        return original(*args)
    monkeypatch.setattr(_project, "_import_controller", gated_import)
    def compile_project(build, second=False):
        if second:
            second_started.set()
        try:
            results.append(build_project(project_dir) if build else _project.load_project(project_dir))
        except Exception as exc:
            errors.append(exc)
    first = threading.Thread(target=compile_project, args=(first_build,))
    second = threading.Thread(target=compile_project, args=(not first_build, True))
    previous = sys.pycache_prefix, sys.dont_write_bytecode
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        assert second_started.wait(2)
        assert not second_entered.wait(0.1)
        # A waiting HIL build must not modify interpreter settings while a
        # release compilation still owns the shared import/declaration state.
        assert (sys.pycache_prefix, sys.dont_write_bytecode) == prefixes[0]
    finally:
        proceed.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert len(results) == 2
    assert (sys.pycache_prefix, sys.dont_write_bytecode) == previous


@pytest.mark.parametrize("change", ["controller", "preset", "extra", "missing"])
def test_build_rejects_browser_source_mismatch_before_identity_creation(project_dir, change):
    sources = {name: (project_dir / name).read_text()
               for name in ("main.py", "controller.py", "robot.preset.yaml")}
    if change == "controller":
        sources["controller.py"] += "\n# Unsaved browser change\n"
    elif change == "preset":
        sources["robot.preset.yaml"] = sources["robot.preset.yaml"].replace("actionScale: 1.0", "actionScale: 2.0")
    elif change == "extra":
        sources["helper.py"] = "# Browser-only file"
    else:
        del sources["robot.preset.yaml"]
    with pytest.raises(HilError, match="Browser sources differ"):
        build_project(project_dir, expected_sources=sources)
    assert not (project_dir / ".botlab/project.json").exists()


@pytest.mark.parametrize("sources", [[], {}, {"main.py": 123}, {"main.py": "x" * (4 * 1024 * 1024 + 1)}])
def test_browser_source_contract_is_bounded(project_dir, sources):
    with pytest.raises(HilError, match="Browser sources"):
        build_project(project_dir, expected_sources=sources)


def test_worker_hang_times_out_and_next_session_recovers(tmp_path):
    worker = WorkerBackend("tests.test_hil_projects:hanging_backend", inference_timeout=0.1)
    model = tmp_path / "model.onnx"
    model.write_bytes(MODEL)
    try:
        handle, _, _ = worker.load(model)
        with pytest.raises(HilError, match="timed out"):
            worker.infer(handle, {"input": {"dtype": "float32", "shape": [1], "data": [2]}})
        assert worker.process is None
        assert worker.load(model)[0] != handle
    finally:
        worker.reset()


def test_isolated_worker_runs_real_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    import math
    worker = WorkerBackend("onnxruntime")
    model = tmp_path / "identity.onnx"
    model.write_bytes(MODEL)
    try:
        handle, inputs, outputs = worker.load(model)
        shape = inputs[0]["shape"]
        data = [float(index) for index in range(math.prod(shape))]
        result = worker.infer(handle, {inputs[0]["name"]: {
            "dtype": "float32", "shape": shape, "data": data}})
        assert result[outputs[0]["name"]]["data"] == data
    finally:
        worker.reset()


def test_stop_terminates_hung_worker_without_blocking_management(tmp_path):
    worker = WorkerBackend("tests.test_hil_projects:hanging_backend", inference_timeout=10)
    state = AgentState(tmp_path / "models", worker, root=tmp_path / "botlab")
    owner, value = object(), artifact()
    deploy(state, owner, value)
    load(state, owner, value)
    session = state.dispatch(owner, {"op": "start"})["session"]
    digest = value["manifest"]["models"][0]["sha256"]
    state.dispatch(owner, {"op": "load", "session": session, "sha256": digest})
    errors = []
    def infer():
        try:
            state.dispatch(owner, {"op": "infer", "session": session, "seq": 0, "model": digest,
                "inputs": {"input": {"dtype": "float32", "shape": [1], "data": [2]}}})
        except HilError as exc:
            errors.append(exc)
    thread = threading.Thread(target=infer)
    thread.start()
    try:
        time.sleep(0.1)
        started = time.monotonic()
        assert state.dispatch(object(), {"op": "hello"})["device"]["state"] == "running"
        state.dispatch(owner, {"op": "stop", "session": session})
        assert time.monotonic() - started < 2
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors
        assert state.calls == 0
        assert state.loaded is not None
    finally:
        worker.reset()
        thread.join(timeout=2)


def test_independent_management_connection_can_remove_only_inactive_deployments(state):
    owner, manager = object(), object()
    old, new, other = artifact(), artifact(revision="second"), artifact("other-project")
    for value in (old, new, other):
        deploy(state, owner, value)
    selection = load(state, owner, new)["loaded"]
    for build in (None, selection["build_id"]):
        with pytest.raises(HilError, match="Unload"):
            state.dispatch(manager, {"op": "projects.remove", "project_id": "project-one", "build_id": build})
    assert state.loaded == selection
    assert state.project_owner is owner
    state.dispatch(manager, {"op": "projects.remove", "project_id": "project-one",
                             "build_id": old["manifest"]["build_id"]})
    assert [item["build_id"] for item in state.store.summary("project-one")["releases"]] == [selection["build_id"]]
    state.dispatch(manager, {"op": "projects.remove", "project_id": "other-project"})
    state.disconnect(manager)
    assert state.loaded == selection
    assert state.project_owner is owner
    assert len(state.store.list()) == 1


def test_running_owner_protects_even_inactive_release_from_other_management(state):
    owner, manager = object(), object()
    active, inactive = artifact(), artifact("inactive-project")
    for value in (active, inactive):
        deploy(state, owner, value)
    selection = load(state, owner, active)["loaded"]
    session = state.dispatch(owner, {"op": "start"})["session"]
    with pytest.raises(HilError, match="Stop"):
        state.dispatch(manager, {"op": "projects.remove", "project_id": "inactive-project"})
    state.disconnect(manager)
    assert state.session == session
    assert state.owner is state.project_owner is owner
    assert state.loaded == selection
    assert len(state.store.list()) == 2
