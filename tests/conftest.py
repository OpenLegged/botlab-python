"""Shared fixtures: a minimal standard BotLab project on disk, plus motion
layout writers for tests that exercise both supported asset formats."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
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


def _encode_bin(array, *, integer=False) -> bytes:
    """One .bin motion array: NPZ\\0 magic, ndims, shape, dtype, payload."""
    array = np.asarray(array)
    if integer:
        array = array.astype(np.int64)
        dtype_size, dtype_code = 8, b"l"
    else:
        array = array.astype(np.float32)
        dtype_size, dtype_code = 4, b"f"
    shape = array.shape
    header = struct.pack("<4sI", b"NPZ\x00", len(shape))
    header += b"".join(struct.pack("<I", int(dim)) for dim in shape)
    header += struct.pack("<I4s", dtype_size, dtype_code)
    return header + array.ravel().tobytes()


def write_motion_npz(path: Path, frames: int = 4, bodies: int = 2, dofs: int = 3) -> None:
    """A valid single-file motion archive with names, via numpy's savez."""
    rng = np.random.default_rng(7)
    np.savez(
        path,
        joint_pos=rng.normal(size=(frames, dofs)).astype(np.float32),
        joint_vel=rng.normal(size=(frames, dofs)).astype(np.float32),
        body_pos_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
        body_quat_w=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, bodies, 1)
        ),
        body_lin_vel_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
        body_ang_vel_w=rng.normal(size=(frames, bodies, 3)).astype(np.float32),
        fps=np.array(50, dtype=np.int64),
        dof_names=np.array([f"joint_{index}" for index in range(dofs)]),
        body_names=np.array([f"body_{index}" for index in range(bodies)]),
    )


def write_motion_folder(path: Path, frames: int = 4, bodies: int = 2, dofs: int = 3) -> None:
    """The seven-file .bin folder layout with the same random content."""
    rng = np.random.default_rng(7)
    path.mkdir(parents=True, exist_ok=True)
    (path / "joint_pos.bin").write_bytes(_encode_bin(rng.normal(size=(frames, dofs))))
    (path / "joint_vel.bin").write_bytes(_encode_bin(rng.normal(size=(frames, dofs))))
    (path / "body_pos_w.bin").write_bytes(_encode_bin(rng.normal(size=(frames, bodies, 3))))
    (path / "body_quat_w.bin").write_bytes(
        _encode_bin(np.tile([1.0, 0.0, 0.0, 0.0], (frames, bodies, 1)))
    )
    (path / "body_lin_vel_w.bin").write_bytes(
        _encode_bin(rng.normal(size=(frames, bodies, 3)))
    )
    (path / "body_ang_vel_w.bin").write_bytes(
        _encode_bin(rng.normal(size=(frames, bodies, 3)))
    )
    (path / "fps.bin").write_bytes(_encode_bin([50], integer=True))
