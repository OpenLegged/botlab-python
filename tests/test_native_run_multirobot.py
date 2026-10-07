"""`botlab run` with multi-robot projects: composed scenes, both controller kinds."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("mujoco") is None,
    reason="runtime extras not installed",
)

PENDULUM_MJCF = """<mujoco>
  <worldbody>
    <body name="base" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

CUSTOM_CONTROLLER = '''\
from botlab import Controller, Robot, project

project("Two Pendulums", control_hz=50, physics_hz=200)
first = Robot(
    "robot.xml",
    name="first",
    config={
        "actionDimension": 1, "jointQposOffset": 7, "jointQvelOffset": 6,
        "defaultJointPositions": [0.0], "actionScale": 1.0,
    },
)
second = Robot(
    "robot.xml",
    name="second",
    config={
        "actionDimension": 1, "jointQposOffset": 7, "jointQvelOffset": 6,
        "defaultJointPositions": [0.0], "actionScale": 1.0,
        "spawnPosition": [1.0, 0.0, 2.0],
    },
)


class Hold(Controller):
    def __init__(self, robot, sim):
        super().__init__(robot)
        self._sim = sim

    def step(self, observation, frame):
        if frame.index == 10:
            self._sim.request_reset("probe reset")
        return self.robot.position_pd(
            self.robot.default_joint_positions(), kp=5.0, kd=1.0, effort_limit=10.0
        )


def create_controller(sim):
    return [Hold(first, sim), Hold(second, sim)]
'''

NATIVE_GROUP_CONTROLLER = '''\
from botlab import PositionController, Robot, project

project("Two Pendulums Native", control_hz=50, physics_hz=200)
first = Robot(
    "robot.xml",
    name="first",
    config={
        "actionDimension": 1, "jointQposOffset": 7, "jointQvelOffset": 6,
        "defaultJointPositions": [0.0], "actionScale": 1.0,
    },
)
second = Robot(
    "robot.xml",
    name="second",
    config={
        "actionDimension": 1, "jointQposOffset": 7, "jointQvelOffset": 6,
        "defaultJointPositions": [0.0], "actionScale": 1.0,
        "spawnPosition": [1.0, 0.0, 2.0],
    },
)


def create_controller(sim):
    return [
        PositionController(first, kp=5.0, kd=1.0, effort_limit=10.0),
        PositionController(second, kp=5.0, kd=1.0, effort_limit=10.0),
    ]
'''

STANDARD_MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)


def _project(tmp_path: Path, controller: str, extra_files: dict | None = None) -> Path:
    (tmp_path / "robot.xml").write_text(PENDULUM_MJCF, encoding="utf-8")
    (tmp_path / "controller.py").write_text(controller, encoding="utf-8")
    (tmp_path / "main.py").write_text(STANDARD_MAIN, encoding="utf-8")
    for name, content in (extra_files or {}).items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


def _run_native(project: Path, **env_overrides) -> subprocess.CompletedProcess:
    env = {**os.environ, "BOTLAB_STEPS": "50", **env_overrides}
    return subprocess.run(
        [sys.executable, "-m", "botlab._runtime.run", str(project)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_multirobot_custom_controllers_run(tmp_path: Path):
    result = _run_native(_project(tmp_path, CUSTOM_CONTROLLER))
    assert result.returncode == 0, result.stderr
    assert "completed 50 control steps" in result.stderr


def test_multirobot_native_group_runs(tmp_path: Path):
    result = _run_native(_project(tmp_path, NATIVE_GROUP_CONTROLLER))
    assert result.returncode == 0, result.stderr
    assert "completed 50 control steps" in result.stderr


SUBSTEP_OBSERVER_CONTROLLER = '''\
import os

from botlab import Controller, Robot, project

project("Substep Probe", control_hz=50, physics_hz=200)
robot = Robot(
    "robot.xml",
    config={
        "actionDimension": 1, "jointQposOffset": 7, "jointQvelOffset": 6,
        "defaultJointPositions": [0.0], "actionScale": 1.0,
    },
)


class Hold(Controller):
    def step(self, observation, frame):
        return [0.0]


def create_controller(sim):
    hits = []
    probe_path = os.environ["SUBSTEP_PROBE"]

    def observer(frame):
        hits.append((frame.dt, frame.index))
        if len(hits) == 200:
            with open(probe_path, "w", encoding="utf-8") as handle:
                handle.write(f"{len(hits)} {hits[-1][0]} {hits[-1][1]}")

    sim.on_substep(observer)
    return Hold(robot)
'''


def test_substep_observer_fires_per_physics_step(tmp_path: Path):
    probe = tmp_path / "probe.txt"
    result = subprocess.run(
        [sys.executable, "-m", "botlab._runtime.run", str(_project(tmp_path, SUBSTEP_OBSERVER_CONTROLLER))],
        capture_output=True,
        text=True,
        env={**os.environ, "BOTLAB_STEPS": "50", "SUBSTEP_PROBE": str(probe)},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # 50 控制步 × 200/50 = 4 物理子步：第 200 次回调时 dt=0.005、index=200
    assert probe.read_text(encoding="utf-8") == "200 0.005 200"


def test_substep_observer_error_aborts_the_run(tmp_path: Path):
    controller = SUBSTEP_OBSERVER_CONTROLLER.replace(
        "        hits.append((frame.dt, frame.index))",
        "        hits.append((frame.dt, frame.index))\n        raise ValueError('boom')",
    )
    result = subprocess.run(
        [sys.executable, "-m", "botlab._runtime.run", str(_project(tmp_path, controller))],
        capture_output=True,
        text=True,
        env={**os.environ, "BOTLAB_STEPS": "50", "SUBSTEP_PROBE": str(tmp_path / "p.txt")},
        timeout=120,
    )
    assert result.returncode != 0
    assert "on_substep observer failed at physics step 1: boom" in result.stderr


def test_multirobot_report_records_poses_and_resets(tmp_path: Path):
    import json

    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "botlab._runtime.run",
            str(_project(tmp_path, CUSTOM_CONTROLLER)),
            "--report", str(report),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "BOTLAB_STEPS": "50"},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["steps"] == 50
    assert data["robots"] == ["first", "second"]
    assert len(data["base_position"]["first"]) == 50
    assert len(data["base_position"]["second"][0]) == 3
    # 两只摆的 spawn 不同：x 坐标分列可分辨
    assert data["base_position"]["second"][0][0] > 0.9
    assert data["resets"] and data["resets"][0]["reason"] == "probe reset"


def test_multirobot_urdf_extra_is_an_explicit_error(tmp_path: Path):
    controller = CUSTOM_CONTROLLER.replace(
        'second = Robot(\n    "robot.xml",',
        'second = Robot(\n    "extra.urdf",',
    )
    assert "extra.urdf" in controller  # 替换必须生效，否则测试失真
    project = _project(
        tmp_path,
        controller,
        {"extra.urdf": '<robot name="arm"><link name="base"/></robot>'},
    )
    result = _run_native(project)
    assert result.returncode != 0
    assert "composed robots require MJCF" in result.stderr
