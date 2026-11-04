"""Unit tests for the static host: declaration collection and validation."""

from __future__ import annotations

import pytest

from botlab import _host
from botlab._host import BotLabError


@pytest.fixture()
def session():
    return _host.begin_project_declaration(lambda preset_id: None)


def test_declare_robot_without_preset_requires_calibration(session):
    with pytest.raises(BotLabError, match="actionDimension"):
        session.declare_resource("robot", "Robot", "assets/robot/scene.xml", None, None, {})


def test_declare_project_requires_integer_frequency_ratio(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 14, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    with pytest.raises(BotLabError, match="integer multiple"):
        session.declare_project("Test", 50, 201, {})


def test_declare_project_rejects_unknown_parameter_fields(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 14, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    with pytest.raises(BotLabError, match="unknown field"):
        session.declare_project("Test", None, None, {"speed": {"default": 0, "mystery": 1}})


def test_duplicate_resource_id_rejected(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 14, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    with pytest.raises(BotLabError, match="already declared"):
        session.declare_resource("model", "Robot", "assets/models/a.onnx", None, None, {})


def test_resource_source_rejects_unsafe_paths(session):
    with pytest.raises(BotLabError, match="safe project-relative"):
        session.declare_resource("model", "M", "../escape.onnx", None, None, {})
    with pytest.raises(BotLabError, match="safe project-relative"):
        session.declare_resource("model", "M2", "/abs/path.onnx", None, None, {})


def test_manifest_defaults_match_browser_shape(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 14, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    session.declare_project("Named", None, None, {})
    manifest = session.project_manifest("Fallback")
    assert manifest == {
        "botlab": 1,
        "project": {"name": "Named", "entry": "main.py"},
        "runtime": {
            "language": "python",
            "simulator": "mujoco",
            "version": "3",
            "controlHz": 50,
            "physicsHz": 200,
        },
        "resources": [
            {
                "id": "Robot",
                "kind": "robot",
                "source": "assets/robot/scene.xml",
                "config": {
                    "actionDimension": 14,
                    "jointQposOffset": 7,
                    "jointQvelOffset": 6,
                },
            }
        ],
        "parameters": {},
        "outputs": {"variables": [], "curves": []},
    }


def test_project_requires_a_robot(session):
    session.declare_project("Empty", None, None, {})
    with pytest.raises(BotLabError, match="must declare a robot"):
        session.project_manifest("Fallback")


def test_parameter_range_feeds_sdk_queries(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 3, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    session.declare_project(
        "P", None, None,
        {"speed": {"default": 0.0, "range": [-1.0, 1.0], "step": 0.1}},
    )
    session.finish_declaration()
    assert _host.parameterRange("speed") == [-1.0, 1.0]
    assert _host.parameterRange("missing") is None
    assert _host.parameter("speed") == 0.0
    assert _host.actionDimension("Robot") == 3
    _host.end_project_declaration()


def test_declaration_channel_closes_after_finish(session):
    session.declare_resource(
        "robot", "Robot", "assets/robot/scene.xml", None, None,
        {"actionDimension": 3, "jointQposOffset": 7, "jointQvelOffset": 6},
    )
    session.finish_declaration()
    with pytest.raises(BotLabError, match="being applied"):
        session.declare_resource("model", "M", "assets/models/a.onnx", None, None, {})


def test_json_value_normalizes_integer_floats_and_rejects_nan():
    with pytest.raises(BotLabError, match="finite"):
        _host._json_value(float("nan"), "x")
    assert _host._json_value(1.0, "x") == 1
    assert isinstance(_host._json_value(1.0, "x"), int)
    assert _host._json_value({"b": 1, "a": [True, None]}, "x") == {"a": [True, None], "b": 1}


def test_option_parameter_membership(session):
    options = [{"label": "Walk", "value": 2}, {"label": "Jump", "value": 7}]
    session.declare_project("Options", 50, 200, {"mode": {"default": 2, "options": options}})
    session.set_parameter("mode", 7)
    assert session.parameter("mode") == 7
    for value in (3, 2.5, True):
        with pytest.raises(BotLabError):
            session.set_parameter("mode", value)


@pytest.mark.parametrize("spec", [
    {"default": 1, "options": []},
    {"default": 1, "options": [{"label": "A", "value": 2}]},
    {"default": 1, "options": [{"label": "", "value": 1}]},
    {"default": 1, "options": [{"label": "A", "value": 1}] * 2},
    {"default": 1.5, "options": [{"label": "A", "value": 1.5}]},
    {"default": 1, "range": [0, 2], "options": [{"label": "A", "value": 1}]},
    {"default": 1, "step": 1, "options": [{"label": "A", "value": 1}]},
    {"default": 1, "options": [{"label": "A", "value": 1, "extra": True}]},
])
def test_invalid_option_parameters(session, spec):
    with pytest.raises(BotLabError):
        session.declare_project("Options", 50, 200, {"mode": spec})
def test_read_project_file_needs_a_directory(session):
    with pytest.raises(BotLabError, match="needs a project directory"):
        _host.read_project_file("config/runtime.yaml")


def test_read_config_feeds_every_declaration_chain(project_dir):
    """The four config-driven chains end to end: project(), Sensor,
    commands(parameters=...) and control_profile(...) all read their
    declaration data from config/*.yaml through the vendored SDK."""
    from botlab._project import load_project

    config = project_dir / "config"
    config.mkdir()
    (config / "runtime.yaml").write_text(
        "name: Config Demo\ncontrol_hz: 50\nphysics_hz: 200\n", encoding="utf-8"
    )
    (config / "parameters.yaml").write_text(
        "forward_velocity: {default: 0.0, range: [-0.5, 0.5]}\n"
        "yaw_rate: {default: 0.0, range: [-1, 1]}\n"
        "policy_command: {default: 0, range: [0, 4]}\n",
        encoding="utf-8",
    )
    (config / "observation.yaml").write_text(
        "depth_camera:\n"
        "  source: assets/robot/scene.xml\n"
        "  config: {outputMode: depth}\n"
        "  name: DepthCamera\n",
        encoding="utf-8",
    )
    (config / "io.yaml").write_text(
        "mode_command:\n"
        "  parameters: [forward_velocity, yaw_rate]\n"
        "  input_mode: keyboard\n",
        encoding="utf-8",
    )
    (config / "control.yaml").write_text(
        "state_node: state-machine\n"
        "vx: forward_velocity\n"
        "yaw_rate: yaw_rate\n"
        "event: policy_command\n"
        "motion_states: [0]\n"
        'states: {"0": WALK}\n'
        "actions:\n"
        "  - {name: sit, event_value: 1, allowed_states: [0], target_states: [3]}\n",
        encoding="utf-8",
    )
    (project_dir / "controller.py").write_text(
        "from botlab import (PolicyController, Robot, Sensor, control_profile,\n"
        "                    declare_control, project, read_config)\n"
        "\n"
        'runtime = read_config("config/runtime.yaml")\n'
        "project(**runtime, parameters=read_config(\"config/parameters.yaml\"))\n"
        'robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")\n'
        'Sensor(**read_config("config/observation.yaml")["depth_camera"])\n'
        "declare_control(control_profile(robot, **read_config(\"config/control.yaml\")))\n"
        "\n"
        "\n"
        "def create_controller(sim):\n"
        '    command = robot.commands(**read_config("config/io.yaml")["mode_command"])\n'
        "    return PolicyController(robot, control=command.action())\n",
        encoding="utf-8",
    )

    loaded = load_project(project_dir)
    manifest = loaded.manifest
    assert manifest["project"]["name"] == "Config Demo"
    assert manifest["runtime"]["controlHz"] == 50
    sensor = next(r for r in manifest["resources"] if r["kind"] == "sensor")
    assert sensor["id"] == "DepthCamera"
    assert sensor["config"] == {"outputMode": "depth"}
    command = next(n for n in loaded.compiled_plan["nodes"] if n["type"] == "obs-commands")["data"]
    assert command["parameterNames"] == ["forward_velocity", "yaw_rate"]
    assert command["cmdMin"] == [-0.5, -1]
    assert command["cmdMax"] == [0.5, 1]

    # load_project closes its session on the way out; the path sandbox holds
    # inside a live declaration window.
    session = _host.begin_project_declaration(lambda preset_id: None, project_dir)
    assert _host.read_project_file("config/runtime.yaml").startswith("name: Config Demo")
    with pytest.raises(BotLabError, match="inside the project directory"):
        _host.read_project_file("../escape.yaml")
    with pytest.raises(BotLabError, match="was not found"):
        _host.read_project_file("config/missing.yaml")
    session.finish_declaration()
    _host.end_project_declaration()
