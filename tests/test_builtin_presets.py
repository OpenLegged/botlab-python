"""Built-in preset resolution: the vendored BotLab registry, offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._presets import PRESETS
from botlab._project import _preset_lookup, load_project

BUILT_IN_ROBOT = "g1-hiking-robot"


def test_registry_carries_calibration_presets():
    assert PRESETS[BUILT_IN_ROBOT]["kind"] == "robot"
    config = PRESETS[BUILT_IN_ROBOT]["config"]
    assert config["actionDimension"] == 29
    assert "policyJointOrder" in config and "kp" in config


def test_lookup_resolves_built_in_ids(tmp_path: Path):
    lookup = _preset_lookup(tmp_path)
    kind, config, allow_overrides = lookup(BUILT_IN_ROBOT)
    assert kind == "robot"
    assert config["actionDimension"] == 29
    assert allow_overrides is True


def test_lookup_returns_none_for_unknown_built_in(tmp_path: Path):
    lookup = _preset_lookup(tmp_path)
    assert lookup("no-such-preset") is None


def test_lookup_keeps_project_local_files_first(tmp_path: Path):
    (tmp_path / "robot.preset.yaml").write_text(
        "kind: robot\nactionDimension: 1\n", encoding="utf-8"
    )
    lookup = _preset_lookup(tmp_path)
    kind, config, _ = lookup("robot.preset.yaml")
    assert kind == "robot"
    assert config == {"actionDimension": 1}


BUILT_IN_CONTROLLER = f'''
from botlab import Model, PolicyController, Robot, concat, project

project("G1 Local", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="{BUILT_IN_ROBOT}")
actor = Model("assets/models/policy.onnx")

def create_controller(sim):
    observation = concat([
        robot.projected_gravity(),
        robot.joint_position(default=robot.default_angles),
    ])
    action = actor(observation).action()
    control = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=control)
'''


STANDARD_MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)


def _write_project(tmp_path: Path, controller: str) -> Path:
    (tmp_path / "controller.py").write_text(controller, encoding="utf-8")
    (tmp_path / "main.py").write_text(STANDARD_MAIN, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def built_in_project(tmp_path: Path) -> Path:
    (tmp_path / "controller.py").write_text(BUILT_IN_CONTROLLER, encoding="utf-8")
    (tmp_path / "main.py").write_text(STANDARD_MAIN, encoding="utf-8")
    (tmp_path / "assets" / "robot").mkdir(parents=True)
    (tmp_path / "assets" / "robot" / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    (tmp_path / "assets" / "models").mkdir(parents=True)
    (tmp_path / "assets" / "models" / "policy.onnx").write_bytes(b"fake-onnx-bytes")
    return tmp_path


def test_load_project_compiles_a_built_in_preset_project(built_in_project: Path):
    project = load_project(built_in_project)
    robot = next(r for r in project.manifest["resources"] if r["kind"] == "robot")
    assert robot["preset"] == BUILT_IN_ROBOT
    # The preset's calibration merged into the resource config.
    assert robot["config"]["actionDimension"] == 29
    assert len(robot["config"]["policyJointOrder"]) == 29
    assert project.compiled_plan is not None


def test_built_in_preset_kind_mismatch_rejected(tmp_path: Path):
    sensor_preset = next(pid for pid, item in PRESETS.items() if item["kind"] == "sensor")
    controller = (
        "from botlab import Robot\n"
        f'Robot("assets/robot/scene.xml", preset="{sensor_preset}")\n'
    )
    with pytest.raises(BotLabError, match="is sensor, not robot"):
        load_project(_write_project(tmp_path, controller))


def test_unknown_preset_still_rejected(tmp_path: Path):
    controller = (
        "from botlab import Robot\n"
        'Robot("assets/robot/scene.xml", preset="mystery-robot")\n'
    )
    with pytest.raises(BotLabError, match="Unknown BotLab resource preset"):
        load_project(_write_project(tmp_path, controller))
