"""Shared fixtures: a minimal standard BotLab project on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

CONTROLLER = '''
from botlab import Model, PolicyController, Robot, concat, project

project("Test Duck", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")
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

PRESET = """
kind: robot
actionDimension: 3
jointQposOffset: 7
jointQvelOffset: 6
policyJointOrder: [0, 1, 2]
motorJointOrder: [0, 1, 2]
actionScale: 1.0
defaultJointPositions: [0.1, 0.2, 0.3]
"""

MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "controller.py").write_text(CONTROLLER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(b"fake-onnx-bytes")
    return tmp_path
