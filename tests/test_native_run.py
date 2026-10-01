"""The `botlab-run` native entry: wiring, dependency errors, and a real run."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    <motor joint="hinge" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

PENDULUM_CONTROLLER = '''\
from botlab import Controller, Robot, project

project("Pendulum Smoke", control_hz=50, physics_hz=200)
robot = Robot(
    "robot.xml",
    config={
        "actionDimension": 1,
        "jointQposOffset": 7,
        "jointQvelOffset": 6,
        "defaultJointPositions": [0.0],
        "actionScale": 1.0,
    },
)


class Hold(Controller):
    def reset(self):
        pass

    def step(self, observation, frame):
        return [0.0]


def create_controller(sim):
    return Hold(robot)
'''

STANDARD_MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)


def _pendulum_project(tmp_path: Path) -> Path:
    (tmp_path / "robot.xml").write_text(PENDULUM_MJCF, encoding="utf-8")
    (tmp_path / "controller.py").write_text(PENDULUM_CONTROLLER, encoding="utf-8")
    (tmp_path / "main.py").write_text(STANDARD_MAIN, encoding="utf-8")
    return tmp_path


def _run_native(project: Path, *extra_env: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    for item in extra_env:
        key, _, value = item.partition("=")
        env[key] = value
    return subprocess.run(
        [sys.executable, "-m", "botlab.sdk._runtime.run", str(project)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_pyproject_wires_the_native_entry_point():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["botlab-run"] == "botlab.sdk._runtime.run:main"
    extras = data["project"]["optional-dependencies"]["runtime"]
    assert {"numpy", "mujoco", "onnxruntime", "websockets"} <= {
        requirement.split(">=")[0].split("[")[0] for requirement in extras
    }


def test_usage_error_without_arguments():
    result = subprocess.run(
        [sys.executable, "-m", "botlab.sdk._runtime.run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_missing_directory_is_reported(tmp_path: Path):
    result = _run_native(tmp_path / "nope")
    assert result.returncode == 2
    assert "project directory not found" in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is not None,
    reason="runtime extras installed; the missing-dependency error path needs them absent",
)
def test_missing_runtime_extras_lists_packages(tmp_path: Path):
    result = _run_native(_pendulum_project(tmp_path))
    assert result.returncode != 0
    assert "mujoco" in (result.stderr + result.stdout)


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="runtime extras not installed",
)
def test_native_run_completes_control_steps(tmp_path: Path):
    result = _run_native(_pendulum_project(tmp_path), "BOTLAB_STEPS=3")
    assert result.returncode == 0, result.stderr
    assert "completed 3 control steps" in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="runtime extras not installed",
)
def test_native_run_rejects_nonexistent_robot_source(tmp_path: Path):
    project = _pendulum_project(tmp_path)
    (project / "robot.xml").unlink()
    result = _run_native(project, "BOTLAB_STEPS=3")
    assert result.returncode != 0
    assert "was not found" in (result.stderr + result.stdout)
