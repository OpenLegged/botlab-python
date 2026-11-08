"""Explicit project panels survive the CLI without entering the control plan."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import pytest

from botlab import _host
from botlab._cli import main
from botlab._debug import build_debug_release
from botlab._host import BotLabError
from botlab._project import load_project, python_release_project_digest
from botlab._release import create_release_package

DEFAULT_PANELS = {
    "watch": None,
    "watchCandidates": None,
    "tiles": None,
    "tileCandidates": None,
    "metrics": {},
}
EXPECTED_PANELS = {
    "watch": ["metric:phase"],
    "watchCandidates": ["metric:phase", "telemetry:baseSpeed"],
    "tiles": ["metric:phase"],
    "tileCandidates": ["metric:phase"],
    "metrics": {
        "phase": {
            "label": {"en": "Phase", "zh": "阶段"},
            "digits": 0,
            "values": {"0": "Rest", "1": {"en": "Walk", "zh": "行走"}},
        },
    },
}


def _declare_panels(project_dir, declaration):
    controller = project_dir / "controller.py"
    source = controller.read_text(encoding="utf-8")
    source = source.replace("concat, project\n", "concat, project, read_config\n")
    source = source.replace(
        'project("Test Duck", control_hz=50, physics_hz=200)',
        f'project("Test Duck", control_hz=50, physics_hz=200, panels={declaration})',
    )
    controller.write_text(source, encoding="utf-8")


@pytest.fixture()
def panels_project(project_dir):
    config = project_dir / "config"
    config.mkdir()
    (config / "curves.yaml").write_text(
        "_label: Curves\n"
        "watch: [metric:phase]\n"
        "candidates: [metric:phase, telemetry:baseSpeed]\n",
        encoding="utf-8",
    )
    (config / "readouts.yaml").write_text(
        "tiles: [metric:phase]\n"
        "candidates: [metric:phase]\n"
        "metrics:\n"
        "  phase:\n"
        "    label: {en: Phase, zh: 阶段}\n"
        "    digits: 0\n"
        "    values: {'0': Rest, '1': {en: Walk, zh: 行走}}\n",
        encoding="utf-8",
    )
    # A familiar filename is ordinary unused data once panels is declared.
    (config / "watch.yaml").write_text("watch: [telemetry:targetSpeed]\n", encoding="utf-8")
    _declare_panels(project_dir, '{"watch": read_config("config/curves.yaml"), '
                    '"metrics": read_config("config/readouts.yaml")}')
    return project_dir


def _package_config(package):
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        return json.loads(archive.read("release.json"))


def test_declared_panels_are_in_manifest_debug_and_release(panels_project):
    project = load_project(panels_project)
    assert project.manifest["panels"] == EXPECTED_PANELS
    for package in (build_debug_release(project), create_release_package(project)):
        published = _package_config(package)["project"]
        assert published["manifest"]["panels"] == EXPECTED_PANELS
        assert published["projectDigest"] == python_release_project_digest(
            project.manifest, project.files
        )
        assert "config/curves.yaml" in {entry["path"] for entry in published["files"]}


def test_panels_do_not_change_control_plan(project_dir):
    before = load_project(project_dir)
    assert "panels" not in before.manifest
    _declare_panels(project_dir, '{"watch": {"watch": ["qpos:0"]}}')
    after = load_project(project_dir)
    assert after.compiled_plan == before.compiled_plan
    assert after.project_digest != before.project_digest


def test_explicit_empty_panels_disable_legacy_filename_discovery(project_dir):
    (project_dir / "config").mkdir()
    (project_dir / "config/watch.yaml").write_text("watch: [qpos:0]\n", encoding="utf-8")
    _declare_panels(project_dir, "{}")
    assert load_project(project_dir).manifest["panels"] == DEFAULT_PANELS


def test_missing_panel_file_fails_with_its_path(project_dir):
    _declare_panels(project_dir, '{"watch": read_config("config/missing.yaml")}')
    with pytest.raises((ValueError, BotLabError), match="config/missing.yaml"):
        load_project(project_dir)


@pytest.mark.parametrize("command", ["run", "build", "validate"])
def test_cli_commands_accept_declared_panels(panels_project, command):
    args = [command, str(panels_project)]
    if command == "run":
        args.append("--no-server")
    assert main(args) == 0
    if command == "build":
        config = _package_config((panels_project / "release.zip").read_bytes())
        assert config["project"]["manifest"]["panels"] == EXPECTED_PANELS


def test_native_host_copies_panel_metadata():
    session = _host.begin_project_declaration(lambda preset_id: None)
    try:
        session.declare_resource("robot", "Robot", "robot.xml", None, None, {
            "actionDimension": 1, "jointQposOffset": 0, "jointQvelOffset": 0,
        })
        panels = {**DEFAULT_PANELS, "watch": ["qpos:0"]}
        _host.declareProject("Panel project", 50, 200, {}, None, None, panels)
        panels["watch"].clear()
        first = session.project_manifest("Fallback")
        assert first["panels"]["watch"] == ["qpos:0"]
        first["panels"]["watch"].clear()
        assert session.project_manifest("Fallback")["panels"]["watch"] == ["qpos:0"]
        assert _host.project_state().manifest()["panels"]["watch"] == ["qpos:0"]
    finally:
        _host.end_project_declaration()


def test_release_rejects_invalid_panels_even_with_valid_digest(panels_project):
    project = load_project(panels_project)
    project.manifest["panels"]["metrics"]["phase"]["digits"] = 10
    project = replace(project, project_digest=python_release_project_digest(project.manifest, project.files))
    with pytest.raises(BotLabError, match="digits"):
        create_release_package(project)
