"""Runtime JSON results stay in their project and preserve ordinary inputs."""

import json

import pytest

from botlab import Simulation, _host
from botlab._runtime._host_runtime import LocalRuntimeSession
from botlab._runtime._results import write_result_file


def result(value):
    return json.dumps({"format": "botlab.runtime-result", "version": 1, "data": value})


class Physics:
    def step_substeps(self, count, observer=None):
        pass

    def after_control_step(self):
        pass


def test_sdk_writes_and_replaces_a_result_during_real_custom_steps(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    session = LocalRuntimeSession(Physics(), project_directory=project)
    _host.set_runtime_session(session)
    try:
        sim = Simulation.from_project()
        session._step_callback = lambda dt, time, index: sim.write_result(
            "results/run-1/handeye-result.json", {"count": index, "label": "标定"},
        )
        session.custom_step(0)
        monkeypatch.chdir(other)
        session.custom_step(1)
        saved = project / "results/run-1/handeye-result.json"
        assert json.loads(saved.read_text())["data"] == {"count": 1, "label": "标定"}
        assert not (other / "results").exists()
        assert not list(saved.parent.glob(".result-*"))
    finally:
        _host.set_runtime_session(None)
        session.close()


def test_result_writes_are_rejected_during_declaration_and_after_close(tmp_path):
    _host.set_runtime_session(None)
    with pytest.raises(_host.BotLabError, match="active control step"):
        Simulation.from_project().write_result("results/result.json", {})
    session = LocalRuntimeSession(Physics(), project_directory=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="active control step"):
            session.writeResult("results/result.json", result({}))
        session.close()
        session._custom_step_active = True
        with pytest.raises(RuntimeError, match="active control step"):
            session.writeResult("results/result.json", result({}))
        assert not (tmp_path / "results").exists()
    finally:
        session.close()


@pytest.mark.parametrize("existing", ['{"calibration": 12}', 'not JSON', '\xff'])
def test_does_not_overwrite_ordinary_project_inputs(tmp_path, existing):
    target = tmp_path / "results/input.json"
    target.parent.mkdir()
    target.write_text(existing)
    with pytest.raises(ValueError, match="Cannot overwrite project input"):
        write_result_file(tmp_path, "results/input.json", result({}))
    assert target.read_text() == existing


@pytest.mark.parametrize("path", [
    "main.py", "results/../main.json", "/results/data.json", "results//data.json",
    "results/data.json\n", "results/数据.json", "results\\data.json",
])
def test_rejects_unsafe_result_paths_without_writes(tmp_path, path):
    with pytest.raises(ValueError, match="Runtime result path"):
        write_result_file(tmp_path, path, result({}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("content", [
    '{}', '{"format":"botlab.runtime-result","version":true,"data":{}}',
    '{"format":"botlab.runtime-result","version":1,"data":NaN}',
    '{"format":"botlab.runtime-result","version":1,"data":1e999}',
])
def test_rejects_invalid_result_documents_without_writes(tmp_path, content):
    with pytest.raises(ValueError):
        write_result_file(tmp_path, "results/data.json", content)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("linked_file", [False, True])
def test_rejects_symlinks_without_modifying_their_target(tmp_path, linked_file):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "data.json"
    external.write_text(result({"original": True}))
    results = project / "results"
    if linked_file:
        results.mkdir()
        (results / "data.json").symlink_to(external)
    else:
        results.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        write_result_file(project, "results/data.json", result({"new": True}))
    assert json.loads(external.read_text())["data"] == {"original": True}


def test_open_headless_writes_results_into_its_project_and_can_run_again(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    from botlab import open_headless

    project = tmp_path / "calibration"
    project.mkdir()
    (project / "robot.xml").write_text('''<mujoco>
      <worldbody><body name="pole"><joint name="hinge" type="hinge"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size=".05" mass="1"/>
      </body></worldbody><actuator><motor joint="hinge"/></actuator>
    </mujoco>''')
    (project / "controller.py").write_text('''from botlab import Controller, Robot, project
project("Result persistence", control_hz=50, physics_hz=200)
robot = Robot("robot.xml", config={"actionDimension": 1,
    "jointQposOffset": 0, "jointQvelOffset": 0})

class ResultController(Controller):
    def __init__(self, sim):
        super().__init__(robot)
        self.sim = sim

    def step(self, observation, frame):
        self.sim.write_result("results/handeye-result.json", {"index": frame.index})
        return [0]

def create_controller(sim):
    return ResultController(sim)
''')
    (project / "main.py").write_text(
        "from botlab import Simulation\nfrom controller import create_controller\n"
        "sim = Simulation.from_project()\nsim.run(create_controller(sim))\n"
    )
    saved = project / "results/handeye-result.json"
    with open_headless(project) as run:
        run.steps(2)
        assert json.loads(saved.read_text())["data"] == {"index": 1}
        run.reset()
        run.step()
        assert json.loads(saved.read_text())["data"] == {"index": 0}
    with open_headless(project) as run:
        run.step()
    assert json.loads(saved.read_text())["data"] == {"index": 0}
