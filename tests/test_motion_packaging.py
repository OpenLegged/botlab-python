"""Motion asset packaging: single-file .npz archives and .bin folders.

The native runtime accepts both motion layouts, but the packager only read
the seven-file .bin folder — an .npz project ran headless and then failed at
``botlab build`` with a misleading "demos.npz/body_pos_w.bin does not exist"
error. The engine's restore path has accepted a single packaged .npz entry
all along (it re-registers the archive under a "<prefix>.npz" registry key
and keeps the dof/body names the folder layout drops), so these tests pin
the writer to that contract.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._project import load_project
from botlab._release import _is_npz_motion_source, create_release_package
from botlab.sdk._runtime._motion import MOTION_REQUIRED_FILES, load_motion_npz
from tests.conftest import write_motion_npz

MOTION_CONTROLLER = '''
from botlab import Model, Motion, PolicyController, Robot, concat, project

project("Motion Duck", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")
actor = Model("assets/models/policy.onnx")
motion = Motion("{source}")

def create_controller(sim):
    observation = concat([
        robot.projected_gravity(),
        robot.joint_position(default=robot.default_angles),
    ])
    action = actor(observation).action()
    control = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=control)
'''


def _motion_project(tmp_path: Path, source: str) -> Path:
    from tests.conftest import MAIN, PRESET

    (tmp_path / "controller.py").write_text(
        MOTION_CONTROLLER.format(source=source), encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(b"fake-onnx-bytes")
    return tmp_path


def test_npz_motion_packages_as_one_asset(tmp_path: Path):
    motion_dir = tmp_path / "assets" / "motions"
    motion_dir.mkdir(parents=True)
    write_motion_npz(motion_dir / "demos.npz")
    project = load_project(_motion_project(tmp_path, "assets/motions/demos.npz"))

    package = create_release_package(project)
    archive = zipfile.ZipFile(io.BytesIO(package))
    config = json.loads(archive.read("release.json"))
    motion_entries = [a for a in config["assets"] if a["role"] == "motion"]
    assert len(motion_entries) == 1
    entry = motion_entries[0]
    assert entry["name"] == "demos.npz"
    data = archive.read(entry["path"])
    assert entry["integrity"]["byteLength"] == len(data)
    # The packaged bytes stay a loadable motion archive with its names.
    motion = load_motion_npz(data, "release:demos.npz")
    assert motion["frame_count"] == 4
    assert motion["dof_count"] == 3
    assert motion["joint_names"] == ["joint_0", "joint_1", "joint_2"]


def test_bin_folder_motion_still_packages_seven_files(tmp_path: Path):
    folder = tmp_path / "assets" / "motions" / "demos"
    folder.mkdir(parents=True)
    for name in MOTION_REQUIRED_FILES:
        (folder / name).write_bytes(b"\x00" * 24)
    project = load_project(_motion_project(tmp_path, "assets/motions/demos"))

    package = create_release_package(project)
    archive = zipfile.ZipFile(io.BytesIO(package))
    config = json.loads(archive.read("release.json"))
    names = {a["name"] for a in config["assets"] if a["role"] == "motion"}
    assert names == set(MOTION_REQUIRED_FILES)


def test_missing_folder_files_error_mentions_npz(tmp_path: Path):
    (tmp_path / "assets" / "motions" / "demos").mkdir(parents=True)
    project = load_project(_motion_project(tmp_path, "assets/motions/demos"))

    with pytest.raises(BotLabError, match=r"single \.npz archive"):
        create_release_package(project)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("assets/motions/demos.npz", True),
        ("assets/motions/DEMOS.NPZ", True),
        ("assets/motions/demos.npz?raw=1", True),
        ("assets/motions/demos.npz/", True),
        ("https://example.com/motions/walk.npz", True),
        ("registry://release-abc/demos.npz", True),
        ("assets/motions/demos", False),
        ("assets/motions/demos.bin", False),
        ("https://example.com/motions/walk.npz.zip", False),
    ],
)
def test_is_npz_motion_source(source: str, expected: bool):
    assert _is_npz_motion_source(source) is expected
