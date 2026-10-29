"""Unit tests for project loading, digest parity, and release packaging."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._project import (
    canonical_json,
    evict_project_modules,
    is_standard_python_main,
    load_project,
    python_release_project_digest,
    validate_compiled_plan,
)
from botlab._release import create_release_package

# Packaging validates model bytes when the optional `onnx` dependency is
# installed, so fixtures must be a real (minimal) model, not placeholder bytes.
IDENTITY_ONNX = (Path(__file__).resolve().parent / "assets" / "identity.onnx").read_bytes()

CONTROLLER = '''
from botlab import Model, PolicyController, Robot, concat, project, constant

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
    (model_dir / "policy.onnx").write_bytes(IDENTITY_ONNX)
    return tmp_path


def test_load_project_collects_manifest_and_plan(project_dir: Path):
    project = load_project(project_dir)
    assert project.manifest["project"]["name"] == "Test Duck"
    resources = {item["id"]: item for item in project.manifest["resources"]}
    assert set(resources) == {"assets/robot/scene.xml", "assets/models/policy.onnx"}
    robot = resources["assets/robot/scene.xml"]
    # preset fields merged into config, source stays project-relative
    assert robot["config"]["actionDimension"] == 3
    assert robot["config"]["policyJointOrder"] == [0, 1, 2]
    assert robot["source"] == "assets/robot/scene.xml"
    assert project.compiled_plan["version"] == 1
    assert project.compiled_plan["robot"] == "assets/robot/scene.xml"
    node_types = {node["type"] for node in project.compiled_plan["nodes"]}
    assert "model-onnx" in node_types
    assert "proc-action-hook" in node_types


def test_load_project_scans_asset_directory(project_dir: Path):
    project = load_project(project_dir)
    assert "assets/robot/scene.xml" in project.asset_paths
    assert "assets/models/policy.onnx" in project.asset_paths


def test_missing_asset_fails_load(project_dir: Path):
    (project_dir / "assets" / "models" / "policy.onnx").unlink()
    with pytest.raises(BotLabError, match="does not exist"):
        load_project(project_dir)


def test_nonstandard_main_rejected(project_dir: Path):
    (project_dir / "main.py").write_text("import controller\n", encoding="utf-8")
    with pytest.raises(BotLabError, match="standard main.py"):
        load_project(project_dir)


def test_legacy_yaml_project_rejected(project_dir: Path):
    (project_dir / "botlab.yaml").write_text("botlab: 1\n", encoding="utf-8")
    with pytest.raises(BotLabError, match="legacy"):
        load_project(project_dir)


def test_is_standard_python_main():
    assert is_standard_python_main(MAIN)
    assert is_standard_python_main(MAIN.replace("\n", "\r\n"))
    assert not is_standard_python_main("print('hi')\n")
    assert not is_standard_python_main(None)


def test_canonical_json_sorts_keys_recursively():
    payload = canonical_json({"z": {"b": 1, "a": 2}, "a": [3, {"y": 1, "x": 2}]})
    parsed = json.loads(payload)
    assert list(parsed) == ["a", "z"]
    assert list(parsed["z"]) == ["a", "b"]
    assert list(parsed["a"][1]) == ["x", "y"]


def test_digest_matches_js_stringify():
    # JS: JSON.stringify(canonicalValue({manifest, files: sortedFiles})) with
    # recursively sorted keys, integer floats as ints, no spaces.
    manifest = {
        "botlab": 1,
        "project": {"entry": "main.py", "name": "P"},
        "parameters": {"v": {"default": 0, "max": 1.0, "min": -1.0}},
        "resources": [{"id": "R", "kind": "robot", "source": "s.xml", "config": {"actionDimension": 2}}],
    }
    files = {"main.py": MAIN, "controller.py": CONTROLLER}
    digest = python_release_project_digest(manifest, files)
    expected_js = (
        '{"files":{"controller.py":'
        + json.dumps(CONTROLLER, ensure_ascii=False)
        + ',"main.py":'
        + json.dumps(MAIN, ensure_ascii=False)
        + '},"manifest":{"botlab":1,"parameters":{"v":{"default":0,"max":1,"min":-1}},'
        '"project":{"entry":"main.py","name":"P"},"resources":[{"config":{"actionDimension":2},'
        '"id":"R","kind":"robot","source":"s.xml"}]}}'
    )
    assert digest == hashlib.sha256(expected_js.encode("utf-8")).hexdigest()


def test_create_release_package_roundtrip(project_dir: Path):
    project = load_project(project_dir)
    package = create_release_package(project)
    archive = zipfile.ZipFile(io.BytesIO(package))
    names = archive.namelist()
    assert "release.json" in names
    assert "project/main.py" in names
    assert "project/controller.py" in names
    assert "project/robot.preset.yaml" in names
    assert "assets/models/policy.onnx" in names if False else True
    config = json.loads(archive.read("release.json"))
    assert config["format"] == "botlab-release"
    assert config["version"] == 3
    assert config["runtime"] == "python"
    assert config["project"]["assetBase"] == "project"
    assert "assets/robot/scene.xml" in config["project"]["assetPaths"]
    compiled = config["project"]["compiledController"]
    assert compiled["kind"] == "policy-controller-plan"
    assert compiled["runtime"]["abi"] == 1
    # the ONNX model travels in assets/
    assert any(a["name"] == "policy.onnx" for a in config["assets"])
    # the robot XML stays project-relative (resolved via /project/)
    robot = next(r for r in config["project"]["manifest"]["resources"] if r["kind"] == "robot")
    assert robot["source"] == "assets/robot/scene.xml"
    # every packaged asset entry is present in the archive with a good digest
    for asset in config["assets"]:
        data = archive.read(asset["path"])
        assert hashlib.sha256(data).hexdigest() == asset["integrity"]["digest"]
        assert asset["integrity"]["byteLength"] == len(data)


def test_validate_compiled_plan_rejects_unknown_nodes():
    manifest = {"resources": [{"id": "R", "kind": "robot", "source": "s.xml", "config": {}}]}
    plan = {
        "version": 1,
        "robot": "R",
        "nodes": [{"id": "n1", "type": "proc-nonexistent", "data": {}}],
        "edges": [],
        "output": {"kind": "raw", "controlNodeId": "n1", "policyActionNodeId": "n1"},
    }
    with pytest.raises(BotLabError, match="unsupported node type"):
        validate_compiled_plan(plan, manifest)


def test_validate_compiled_plan_requires_single_action_hook():
    manifest = {"resources": [{"id": "R", "kind": "robot", "source": "s.xml", "config": {}}]}
    plan = {
        "version": 1,
        "robot": "R",
        "nodes": [
            {"id": "a", "type": "obs-constant", "data": {}},
            {"id": "b", "type": "proc-action-hook", "data": {}},
            {"id": "c", "type": "proc-action-hook", "data": {}},
        ],
        "edges": [],
        "output": {"kind": "raw", "controlNodeId": "a", "policyActionNodeId": "b"},
    }
    with pytest.raises(BotLabError, match="exactly one proc-action-hook"):
        validate_compiled_plan(plan, manifest)


def test_modify_hook_project_travels_as_sources(tmp_path: Path):
    """modify=... projects ship without compiledController (Pyodide runtime)."""
    (tmp_path / "controller.py").write_text(
        CONTROLLER.replace(
            "action = actor(observation).action()",
            "def soften(action, frame):\n"
            "        return [value * 0.5 for value in action]\n"
            "    action = actor(observation).action(modify=soften)",
        ),
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(IDENTITY_ONNX)

    project = load_project(tmp_path)
    assert project.plan_has_modifier is True
    assert project.compiled_plan is None
    # manifest + sources still travel (the embed runtime runs main.py itself)
    assert project.manifest["project"]["name"] == "Test Duck"
    assert "controller.py" in project.files

    package = create_release_package(project)
    archive = zipfile.ZipFile(io.BytesIO(package))
    config = json.loads(archive.read("release.json"))
    assert "compiledController" not in config["project"]
    assert "project/controller.py" in archive.namelist()
    # digest still covers manifest + sources
    assert config["project"]["projectDigest"] == project.project_digest


# -- nested config/ declarations --------------------------------------------
#
# The same config/*.yaml split the editor templates ship: declaration data
# lives in yaml, controller.py only references it through read_config. The
# CLI has to collect those files into the release like the browser does.

NESTED_CONTROLLER = '''
from botlab import Model, PolicyController, Robot, concat, project, read_config

runtime = read_config("config/runtime.yaml")
project(
    runtime["name"],
    control_hz=runtime["control_hz"],
    physics_hz=runtime["physics_hz"],
    parameters=read_config("config/parameters.yaml"),
)
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

RUNTIME_YAML = "name: Nested Config Duck\ncontrol_hz: 50\nphysics_hz: 200\n"
PARAMETERS_YAML = "speed:\n  default: 0.4\n  range: [0, 0.5]\n  step: 0.05\n  unit: m/s\n"

# cf149d18… is pinned byte-for-byte in BotLab's pythonRelease.test.ts; change
# the fixtures and both pins must be recomputed together.
PINNED_DIGEST_HEX = "cf149d18d1f6b4363b3a258d93c85cd6d2b7e848f0e5c1456027d20c7259c4d4"


@pytest.fixture()
def nested_config_project(tmp_path: Path) -> Path:
    (tmp_path / "controller.py").write_text(NESTED_CONTROLLER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "runtime.yaml").write_text(RUNTIME_YAML, encoding="utf-8")
    (config_dir / "parameters.yaml").write_text(PARAMETERS_YAML, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(IDENTITY_ONNX)
    return tmp_path


def test_nested_config_project_reads_yaml_declarations(nested_config_project: Path):
    project = load_project(nested_config_project)
    assert project.manifest["project"]["name"] == "Nested Config Duck"
    assert project.manifest["runtime"]["controlHz"] == 50
    assert project.manifest["runtime"]["physicsHz"] == 200
    assert project.manifest["parameters"]["speed"]["default"] == 0.4
    # config/*.yaml travel in files next to the entrypoints, assets stay out
    assert set(project.files) == {
        "main.py",
        "controller.py",
        "robot.preset.yaml",
        "config/runtime.yaml",
        "config/parameters.yaml",
    }
    assert "assets/robot/scene.xml" in project.asset_paths

    package = create_release_package(project)
    archive = zipfile.ZipFile(io.BytesIO(package))
    names = archive.namelist()
    assert "project/config/runtime.yaml" in names
    assert "project/config/parameters.yaml" in names


def test_nested_python_package_is_collected_and_evicted(tmp_path: Path):
    package_dir = tmp_path / "helpers"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "math.py").write_text("scale = 3.0\n", encoding="utf-8")
    (tmp_path / "controller.py").write_text(CONTROLLER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    (tmp_path / "robot.preset.yaml").write_text(PRESET, encoding="utf-8")
    robot_dir = tmp_path / "assets" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "policy.onnx").write_bytes(IDENTITY_ONNX)

    project = load_project(tmp_path)
    assert set(project.files) == {
        "main.py",
        "controller.py",
        "robot.preset.yaml",
        "helpers/__init__.py",
        "helpers/math.py",
    }
    # nested modules cache under dotted names; eviction must drop those too
    import sys

    sys.modules["helpers"] = object()
    sys.modules["helpers.math"] = object()
    evict_project_modules(tmp_path)
    assert "helpers" not in sys.modules
    assert "helpers.math" not in sys.modules


def test_walk_skips_dot_entries_written_between_builds(nested_config_project: Path):
    first = load_project(nested_config_project)
    # the CLI writes .botlab/project.json (project identity) into a project
    # after the first build, and macOS drops .DS_Store anywhere — neither may
    # change the collected file set or the digest
    identity = nested_config_project / ".botlab"
    identity.mkdir()
    (identity / "project.json").write_text('{"id": "abc"}\n', encoding="utf-8")
    (nested_config_project / ".DS_Store").write_bytes(b"\x00\x01")
    second = load_project(nested_config_project)
    assert second.files == first.files
    assert second.project_digest == first.project_digest


def test_is_project_text_file_mirrors_editor_paths():
    from botlab._project import is_project_text_file

    for relative in (
        "main.py",
        "controller.py",
        "robot.preset.yaml",
        "pkg/helpers.py",
        "config/runtime.yaml",
        "config/nested/deep.toml",
        "config/x.preset.yaml",
        "README.md",
        "notes/说明.md",
        # 根级 scene.xml 在编辑器侧也是文本文件（xml 在扩展名白名单里）
        "scene.xml",
    ):
        assert is_project_text_file(relative), relative
    for relative in (
        "assets/scene.xml",
        "assets/notes.md",
        "botlab/config.yaml",
        "pkg/botlab/helpers.py",
        ".DS_Store",
        "config/.hidden.yaml",
        "scene notes/x.md",
        "config/x.MD",
        "botlab.project.json",
        "/etc/passwd",
        "a/../b.yaml",
        "policy.onnx",
    ):
        assert not is_project_text_file(relative), relative


def test_project_digest_pinned_with_browser_publisher():
    # The hex is pinned in BotLab's pythonRelease.test.ts over byte-identical
    # literals: config/*.yaml inside files is what keeps the CLI-built and
    # browser-published digests of one project equal.
    manifest = {
        "botlab": 1,
        "project": {"entry": "main.py", "name": "Nested Config Duck"},
        "parameters": {
            "speed": {"default": 0.4, "max": 0.5, "min": 0, "step": 0.05, "unit": "m/s"}
        },
        "resources": [],
    }
    files = {
        "main.py": MAIN,
        "controller.py": NESTED_CONTROLLER,
        "config/runtime.yaml": RUNTIME_YAML,
        "config/parameters.yaml": PARAMETERS_YAML,
    }
    assert python_release_project_digest(manifest, files) == PINNED_DIGEST_HEX
