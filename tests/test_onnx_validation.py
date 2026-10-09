"""Packaging-time ONNX model validation.

A torch 2.x dynamo export writes weights to a sibling ``.data`` file; the
model loads fine next to where it was exported and then breaks every runtime
that reads the packaged bytes alone — the GMTrack CASBOT02 export hit this
and the failure surfaced deep inside onnxruntime, far from the asset.
``create_release_package`` now parses model assets (when the ``onnx`` package
is installed) and rejects, with actionable messages: unparseable files,
externalized weights, and plan-declared input/output names the graph does
not define.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._project import load_project
from botlab._release import _validate_onnx_model, create_release_package

TESTS_DIR = Path(__file__).resolve().parent
IDENTITY_ONNX = (TESTS_DIR / "assets" / "identity.onnx").read_bytes()
EXTERNAL_WEIGHTS_ONNX = (TESTS_DIR / "assets" / "external_weights.onnx").read_bytes()

CONTROLLER = '''
from botlab import Model, PolicyController, Robot, concat, project

project("Validate Duck", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")
actor = Model("assets/models/policy.onnx")

def create_controller(sim):
    observation = concat([
        robot.projected_gravity(),
        robot.joint_position(default=robot.default_angles),
    ])
    action = actor(observation, {call_args}).action()
    control = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=control)
'''


def _project(tmp_path: Path, model_bytes: bytes, call_args: str = "") -> Path:
    from tests.conftest import MAIN, PRESET

    (tmp_path / "controller.py").write_text(
        CONTROLLER.format(call_args=call_args), encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(model_bytes)
    return tmp_path


@pytest.fixture()
def onnx_module():
    return pytest.importorskip("onnx")


def test_valid_model_packages(tmp_path: Path, onnx_module):
    project = load_project(_project(tmp_path, IDENTITY_ONNX))
    package = create_release_package(project)
    assert package[:2] == b"PK"  # zip magic: packaging succeeded


def test_garbage_bytes_are_rejected_with_an_actionable_error(tmp_path: Path, onnx_module):
    project = load_project(_project(tmp_path, b"not-an-onnx-model"))
    with pytest.raises(BotLabError, match="not a valid ONNX model"):
        create_release_package(project)


def test_external_data_weights_are_rejected(tmp_path: Path, onnx_module):
    project = load_project(_project(tmp_path, EXTERNAL_WEIGHTS_ONNX))
    # The error embeds a multi-line inline recipe; `.` must cross newlines.
    with pytest.raises(
        BotLabError, match=r"(?s)external file.*convert_model_from_external_data"
    ):
        create_release_package(project)


def test_declared_input_name_must_exist_in_the_graph(tmp_path: Path, onnx_module):
    project = load_project(
        _project(tmp_path, IDENTITY_ONNX, call_args='input_name="wrong"')
    )
    with pytest.raises(BotLabError, match=r'input_name "wrong".*inputs are \[.obs.\]'):
        create_release_package(project)


def test_declared_output_name_must_exist_in_the_graph(tmp_path: Path, onnx_module):
    project = load_project(
        _project(tmp_path, IDENTITY_ONNX, call_args='output_name="wrong"')
    )
    with pytest.raises(BotLabError, match=r'output_name "wrong".*outputs are \[.action.\]'):
        create_release_package(project)


def test_matching_names_pass(tmp_path: Path, onnx_module):
    project = load_project(
        _project(tmp_path, IDENTITY_ONNX, call_args='input_name="obs", output_name="action"')
    )
    assert create_release_package(project)[:2] == b"PK"


def test_validation_is_skipped_without_the_onnx_package(
    tmp_path: Path, monkeypatch, onnx_module
):
    import botlab._release as release

    monkeypatch.setattr(release, "_optional_onnx_module", lambda: None)
    project = load_project(_project(tmp_path, b"not-an-onnx-model"))
    # Without onnx installed, packaging trusts the bytes (today's behavior).
    assert create_release_package(project)[:2] == b"PK"


def test_validator_reports_the_external_file_name(onnx_module):
    with pytest.raises(BotLabError, match=r"tiny\.onnx\.data"):
        _validate_onnx_model(
            onnx_module, EXTERNAL_WEIGHTS_ONNX, "policy", []
        )
