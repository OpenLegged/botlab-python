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
