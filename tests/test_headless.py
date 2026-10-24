"""The public headless rollout API: open, step, record, inject parameters.

Porting a policy into BotLab used to mean re-implementing ``botlab-run``'s
private assembly and control loop (session internals, physics fill order,
payload extraction) just to measure tracking quality — the verification
script for the GMTrack CASBOT02 port touched eight private attributes. These
tests pin the supported surface: the same loop, one control step at a time,
with parameter overrides that take effect mid-rollout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from botlab import HeadlessRun, StepRecord, load_motion, open_headless  # noqa: E402
from botlab._host import BotLabError  # noqa: E402
from tests.conftest import write_motion_folder, write_motion_npz  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
IDENTITY_ONNX = (TESTS_DIR / "assets" / "identity.onnx").read_bytes()
NEGATE_ONNX = (TESTS_DIR / "assets" / "negate.onnx").read_bytes()

PENDULUM_MJCF = """<mujoco>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# The identity model passes its observation through, so the recorded action IS
# the commands observation — the declared parameter's current value.
CONTROLLER = '''\
from botlab import Model, PolicyController, Robot, concat, project

project("Headless Duck", control_hz=50, physics_hz=200,
        parameters={"gain": {"default": 1.0, "range": [0.0, 2.0]}})
robot = Robot("robot.xml", config={
    "actionDimension": 1,
    "jointQposOffset": 0,
    "jointQvelOffset": 0,
    "policyJointOrder": [0],
    "motorJointOrder": [0],
    "defaultJointPositions": [0.0],
    "actionScale": 1.0,
})
actor = Model("policy.onnx")

def create_controller(sim):
    observation = concat([robot.commands("gain")])
    action = actor(observation).action()
    control = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=control)
'''

MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)


def _pendulum_project(directory: Path, model_bytes: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "controller.py").write_text(CONTROLLER, encoding="utf-8")
    (directory / "main.py").write_text(MAIN, encoding="utf-8")
    (directory / "robot.xml").write_text(PENDULUM_MJCF, encoding="utf-8")
    (directory / "policy.onnx").write_bytes(model_bytes)
    return directory


@pytest.fixture()
def pendulum_project(tmp_path: Path) -> Path:
    return _pendulum_project(tmp_path, IDENTITY_ONNX)


def test_headless_run_steps_and_records(pendulum_project: Path):
    import numpy as np

    original_cwd = Path().resolve()
    with open_headless(pendulum_project) as run:
        assert isinstance(run, HeadlessRun)
        assert run.control_hz == 50
        assert run.physics_hz == 200
        assert run.manifest["parameters"]["gain"]["default"] == 1.0

        records = run.steps(5)
        assert len(records) == 5
        assert [record.frame.index for record in records] == list(range(5))
        assert all(isinstance(record, StepRecord) for record in records)
        assert all(record.action.dtype == np.float32 for record in records)
        # The identity model passes the commands observation through: the
        # declared default gain is what the policy saw every step.
        assert [float(record.action[0]) for record in records] == [1.0] * 5
        assert np.isfinite(run.physics.data.qpos).all()

    # Teardown restores the caller's process state and clears the runtime.
    assert Path().resolve() == original_cwd
    import botlab_host

    with pytest.raises(BotLabError, match="offline packager never steps"):
        botlab_host.observe("Robot")


def test_set_parameter_takes_effect_from_the_next_step(pendulum_project: Path):
    with open_headless(pendulum_project) as run:
        first = run.step()
        assert float(first.action[0]) == 1.0

        run.set_parameter("gain", 2.0)
        second = run.step()
        assert float(second.action[0]) == 2.0

        # Range-checked like the declaration, through the same surface.
        with pytest.raises(BotLabError, match="outside its range"):
            run.set_parameter("gain", 3.0)
        with pytest.raises(BotLabError, match="not declared"):
            run.set_parameter("heading", 0.5)


def test_reset_restarts_the_rollout_with_overrides_intact(pendulum_project: Path):
    with open_headless(pendulum_project) as run:
        run.steps(3)
        run.set_parameter("gain", 0.5)
        run.reset()
        record = run.step()
        assert record.frame.index == 0
        assert float(record.action[0]) == 0.5


def test_open_headless_requires_a_project_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        open_headless(tmp_path / "missing")


def test_step_after_close_is_rejected(pendulum_project: Path):
    run = open_headless(pendulum_project)
    run.close()
    with pytest.raises(RuntimeError, match="closed"):
        run.step()
    run.close()  # idempotent


def test_load_motion_reads_both_layouts(tmp_path: Path):
    write_motion_npz(tmp_path / "walk.npz")
    npz = load_motion(tmp_path / "walk.npz")
    assert npz["frame_count"] == 4
    assert npz["joint_names"] == ["joint_0", "joint_1", "joint_2"]

    write_motion_folder(tmp_path / "folder")
    folder = load_motion(tmp_path / "folder")
    assert folder["frame_count"] == npz["frame_count"]
    assert folder["dof_count"] == npz["dof_count"]
    assert folder["fps"] == npz["fps"] == 50.0


@pytest.mark.parametrize("with_metadata", [False, True])
def test_folder_names_match_sdk_and_public_motion_loading(pendulum_project: Path, with_metadata):
    import asyncio
    import importlib
    import json

    import numpy as np

    folder = pendulum_project / "folder"
    write_motion_folder(folder)
    expected = {"body_names": ["root", "foot"], "joint_names": ["hip", "knee", "ankle"]}
    if with_metadata:
        (folder / "metadata.json").write_text(json.dumps(expected), encoding="utf-8")
    controller = pendulum_project / "controller.py"
    controller.write_text(controller.read_text() + '\nfrom botlab import Motion\nreference = Motion("folder")\n')
    with open_headless(pendulum_project) as run:
        sdk_motion = asyncio.run(importlib.import_module("controller").reference.load())
        public_motion = load_motion(folder)
        run_motion = run.load_motion("folder")
        for key, names in expected.items():
            if with_metadata:
                assert sdk_motion[key] == public_motion[key] == run_motion[key] == names
            else:
                assert key not in sdk_motion and key not in public_motion and key not in run_motion
        np.testing.assert_array_equal(sdk_motion["joint_position"], public_motion["joint_position"])
        np.testing.assert_array_equal(sdk_motion["body_quaternion"], run_motion["body_quaternion"])


def test_close_evicts_cached_model_sessions(tmp_path: Path):
    """The session pool keys by resource id, not file contents — a run that
    closes must leave nothing behind, or the next project with the same
    resource id silently re-runs the previous model."""
    project_a = _pendulum_project(tmp_path / "a", IDENTITY_ONNX)
    project_b = _pendulum_project(tmp_path / "b", NEGATE_ONNX)

    with open_headless(project_a) as run:
        assert float(run.step().action[0]) == 1.0  # identity: passes gain through
    with open_headless(project_b) as run:
        assert float(run.step().action[0]) == -1.0  # negate: stale session would say +1.0


def test_close_evicts_sessions_for_a_reexported_model(tmp_path: Path):
    project = _pendulum_project(tmp_path / "proj", IDENTITY_ONNX)
    with open_headless(project) as run:
        assert float(run.step().action[0]) == 1.0
    (project / "policy.onnx").write_bytes(NEGATE_ONNX)
    with open_headless(project) as run:
        assert float(run.step().action[0]) == -1.0


def test_load_motion_resolves_against_project_directory(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    write_motion_npz(project / "walk.npz")
    monkeypatch.chdir(tmp_path)  # CWD has no motions anywhere

    data = load_motion("walk.npz", project=project)
    assert data["frame_count"] == 4


def test_load_motion_errors_name_what_is_missing(tmp_path: Path):
    empty = tmp_path / "folder"
    empty.mkdir()
    with pytest.raises(BotLabError, match=r"missing required file\(s\).*body_pos_w\.bin"):
        load_motion(empty)

    with pytest.raises(BotLabError, match=r"was not found \(tried"):
        load_motion(tmp_path / "nope.npz")


def test_run_load_motion_works_after_close_and_cwd_moves(tmp_path: Path, monkeypatch):
    project = _pendulum_project(tmp_path / "proj", IDENTITY_ONNX)
    write_motion_npz(project / "walk.npz")
    run = open_headless(project)
    run.close()
    monkeypatch.chdir(tmp_path)  # away from the project directory

    assert run.load_motion("walk.npz")["frame_count"] == 4


def test_close_warns_when_pool_cleanup_fails(tmp_path, monkeypatch):
    from botlab._runtime._processors import _model

    class _Pool:
        def clear(self):
            raise RuntimeError("pool boom")

    monkeypatch.setattr(_model, "shared_session_pool", lambda: _Pool())
    run = HeadlessRun.__new__(HeadlessRun)
    run._closed = False
    run._directory = tmp_path
    run._original_directory = str(Path.cwd())
    run._original_path = list(__import__("sys").path)

    class _Session:
        def close(self):
            pass

    run.session = _Session()
    with pytest.warns(UserWarning, match="pool cleanup failed"):
        run.close()
