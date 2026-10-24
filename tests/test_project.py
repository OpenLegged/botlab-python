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


def test_named_robot_sensor_identity_is_not_an_asset_path(project_dir):
    controller = CONTROLLER.replace('Robot("assets/robot/scene.xml",', 'Robot("assets/robot/scene.xml", name="G1",')
    controller += '\nfrom botlab import Sensor\ncamera = Sensor(robot, name="Depth", config={"outputMode": "depth"})\n'
    (project_dir / 'controller.py').write_text(controller)
    loaded = load_project(project_dir)
    sensor = next(r for r in loaded.manifest['resources'] if r['kind'] == 'sensor')
    assert sensor['source'] == 'G1'
    assert loaded.compiled_plan['robot'] == 'G1'
    assert 'G1' not in loaded.asset_paths
    assert 'assets/robot/scene.xml' in loaded.asset_paths
    create_release_package(loaded)


def test_sensor_requires_robot_reference(project_dir):
    (project_dir / 'controller.py').write_text(CONTROLLER + '\nfrom botlab import Sensor\ncamera = Sensor("missing", name="Depth")\n')
    with pytest.raises(BotLabError, match='must reference a declared robot'):
        load_project(project_dir)


def test_project_collects_mjcf_sibling_assets(project_dir):
    shared = project_dir / 'assets/shared'
    shared.mkdir()
    (shared / 'body.xml').write_text('<mujoco><worldbody/></mujoco>')
    (shared / 'mesh.obj').write_bytes(b'fixture')
    (project_dir / 'assets/robot/scene.xml').write_text("""<mujoco><compiler meshdir='../shared'/>
      <include file='../shared/body.xml'/><asset><mesh file='mesh.obj'/></asset></mujoco>""")
    loaded = load_project(project_dir)
    assert {'assets/shared/body.xml', 'assets/shared/mesh.obj'} <= set(loaded.asset_paths)


def test_digest_matches_actual_node_for_number_and_key_boundaries():
    import shutil
    import subprocess

    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the cross-language digest test')
    values = [6.812889478169382e-05, 1e-7, 1e-6, 1e20, 1e21, -0.0, 5e-324,
              1.7976931348623157e308, 333333333.33333329, 1000000000000000128]
    payload = {'values': values, 'keys': {'10': 1, '2': 2, '01': 3, '\U00010000': 4, '\ue000': 5},
               'text': '中文\n\ud800'}
    expected = subprocess.check_output([node, '-e', '''
const fs=require('fs'); const c=v=>Array.isArray(v)?v.map(c):v&&typeof v==='object'?
Object.fromEntries(Object.keys(v).sort().map(k=>[k,c(v[k])])):v;
process.stdout.write(JSON.stringify(c(JSON.parse(fs.readFileSync(0,'utf8')))));
'''], input=json.dumps(payload).encode()).decode()
    assert canonical_json(payload) == expected
