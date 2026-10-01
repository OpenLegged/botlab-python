"""Unit tests for the host's runtime-session hook and project_state view."""

from __future__ import annotations

import pytest

from botlab import _host
from botlab._host import BotLabError

ROBOT_CONFIG = {"actionDimension": 3, "jointQposOffset": 7, "jointQvelOffset": 6}


class FakeSession:
    def __init__(self):
        self.calls = []

    def observe(self, rid):
        self.calls.append(("observe", rid))
        return {"qpos": [0.0] * 10}

    def applyAction(self, rid, action):
        self.calls.append(("applyAction", rid, list(action)))

    async def predictModel(self, rid, values):
        self.calls.append(("predictModel", rid, list(values)))
        return [0.5]

    def registerStep(self, proxy, robot_ids=None):
        self.calls.append(("registerStep", robot_ids))

    def registerReset(self, proxy):
        self.calls.append(("registerReset",))

    def requestReset(self, reason, spec=None):
        self.calls.append(("requestReset", reason, spec))


@pytest.fixture()
def clean_state():
    _host.end_project_declaration()
    _host.set_runtime_session(None)
    yield
    _host.set_runtime_session(None)
    _host.end_project_declaration()


def test_runtime_calls_raise_without_session(clean_state):
    with pytest.raises(BotLabError, match="offline packager never steps"):
        _host.observe("Robot")
    with pytest.raises(BotLabError, match="offline packager never infers"):
        _host.predictModel("Actor", [0.0])
    with pytest.raises(BotLabError, match="Custom Controllers require"):
        _host.registerStep(lambda: None)
    with pytest.raises(BotLabError, match="live simulation runtime"):
        _host.updateMetrics({"x": 1.0})
    with pytest.raises(BotLabError, match="live simulation runtime"):
        _host.loadMotion("walk")
    with pytest.raises(BotLabError, match="live simulation runtime"):
        _host.readMujoco("Robot", "qpos")
    with pytest.raises(BotLabError, match="live simulation runtime"):
        _host.requestReset("why")


def test_runtime_calls_delegate_to_installed_session(clean_state):
    session = FakeSession()
    _host.set_runtime_session(session)

    assert _host.observe("Robot") == {"qpos": [0.0] * 10}
    _host.applyAction("Robot", [1.0, 2.0, 3.0])
    _host.requestReset("tilted", {"motionId": "walk"})
    _host.registerStep(lambda: None, ["Robot"])

    # The session owns the async shape; the delegation must pass it through.
    import asyncio

    assert asyncio.run(_host.predictModel("Actor", [0.25])) == [0.5]

    assert session.calls == [
        ("observe", "Robot"),
        ("applyAction", "Robot", [1.0, 2.0, 3.0]),
        ("requestReset", "tilted", {"motionId": "walk"}),
        ("registerStep", ["Robot"]),
        ("predictModel", "Actor", [0.25]),
    ]


def test_uninstalling_session_restores_offline_errors(clean_state):
    _host.set_runtime_session(FakeSession())
    _host.set_runtime_session(None)
    with pytest.raises(BotLabError, match="offline packager never steps"):
        _host.observe("Robot")


def test_project_state_reset_collects_a_flat_manifest(clean_state, tmp_path):
    state = _host.project_state()
    state.reset(str(tmp_path))
    _host.declareResource("robot", "Robot", "assets/robot/scene.xml", None, None, ROBOT_CONFIG)
    _host.declareProject("Native Duck", 50, 200, {"speed": {"default": 0.4, "range": [0, 1]}})

    manifest = state.manifest()
    assert manifest["name"] == "Native Duck"
    assert manifest["control_hz"] == 50
    assert manifest["physics_hz"] == 200
    assert manifest["parameters"]["speed"]["default"] == 0.4
    assert manifest["resources"][0]["id"] == "Robot"
    assert state.directory == str(tmp_path)

    resource = state.resource("Robot")
    assert resource["kind"] == "robot"
    assert state.resource("Robot", "robot") is resource
    with pytest.raises(BotLabError, match="expected model"):
        state.resource("Robot", "model")


def test_project_state_defaults_frequencies_and_falls_back_to_directory_name(
    clean_state, tmp_path
):
    # project() is optional in the two-file contract: no declaration means the
    # 50/200 Hz defaults and the directory name as the project name.
    state = _host.project_state()
    state.reset(str(tmp_path))
    _host.declareResource("robot", "Robot", "assets/robot/scene.xml", None, None, ROBOT_CONFIG)

    manifest = state.manifest()
    assert manifest["name"] == tmp_path.name
    assert manifest["control_hz"] == 50
    assert manifest["physics_hz"] == 200


def test_project_state_without_declarations_raises(clean_state):
    _host.project_state().reset(None)
    with pytest.raises(BotLabError, match="reset\\(directory\\) first"):
        _host.project_state().manifest()
    with pytest.raises(BotLabError, match="reset\\(directory\\) first"):
        _host.project_state().resource("Robot")


def test_reset_replaces_previous_declarations(clean_state, tmp_path):
    state = _host.project_state()
    state.reset(str(tmp_path))
    _host.declareResource("robot", "Robot", "assets/robot/scene.xml", None, None, ROBOT_CONFIG)
    state.reset(str(tmp_path))
    with pytest.raises(BotLabError, match="Unknown resource"):
        state.resource("Robot")
