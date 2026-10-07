"""Site assembly tests: closure scan, release validation, and layout."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._project import load_project
from botlab._release import create_release_package
from botlab._site import (
    _collect_runtime_closure,
    _scan_chunk_references,
    build_site_directory,
    collect_site_asset_roots,
    read_release_config,
)


def _release_dict(**overrides):
    config = {
        "format": "botlab-release",
        "version": 3,
        "runtime": "python",
        "project": {
            "manifest": {
                "project": {"name": "T"},
                "resources": [
                    {"id": "robot", "kind": "robot", "source": "assets/robot/scene.xml"},
                    {"id": "policy", "kind": "model", "source": "assets/models/policy.onnx"},
                ],
            },
            "assetBase": "project",
            "assetPaths": ["assets/robot/scene.xml"],
            "compiledController": {"kind": "policy-controller-plan"},
        },
        "assets": [
            {
                "resourceId": "policy",
                "path": "assets/models/policy-abc.onnx",
                "role": "model",
                "target": {"kind": "resourceSource"},
            }
        ],
    }
    config.update(overrides)
    return config


def _zip_bytes(config: dict) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("release.json", json.dumps(config))
        archive.writestr("project/main.py", "print('hi')\n")
    return buffer.getvalue()


@pytest.fixture()
def mini_snapshot(tmp_path: Path) -> Path:
    """A fake UI snapshot with the shapes the closure scanner understands."""
    tmp_path = tmp_path / "ui-snapshot"
    tmp_path.mkdir()
    (tmp_path / "embed.html").write_text(
        '<script type="module" src="./assets/embed-abc.js"></script>'
        '<link rel="manifest" href="./embed.webmanifest">'
        '<link rel="icon" href="favicon.svg">',
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "embed-abc.js").write_text(
        'import("./chunk-def.js"); fetch("ort-wasm-simd-threaded-X1.wasm");'
        'const ignored = `${root}/dyn.js`; const remote = "https://example.com/x.js";'
        'new URL("mujoco-Q2.wasm", import.meta.url)',
        encoding="utf-8",
    )
    (assets / "chunk-def.js").write_text('load("obj-mesh-Y2.wasm")', encoding="utf-8")
    for name in (
        "ort-wasm-simd-threaded-X1.wasm",
        "mujoco-Q2.wasm",
        "obj-mesh-Y2.wasm",
        "editor-main-unused.js",
    ):
        (assets / name).write_bytes(b"wasm-or-js-bytes")
    (tmp_path / "embed.webmanifest").write_text("{}", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    obj_parser = tmp_path / "wasm" / "obj-parser"
    obj_parser.mkdir(parents=True)
    (obj_parser / "objParser.js").write_bytes(b"worker")
    (obj_parser / "objParser.wasm").write_bytes(b"wasm")
    return tmp_path


def test_closure_scan_collects_quoted_references_only(mini_snapshot: Path):
    files = {
        item.relative_to(mini_snapshot).as_posix()
        for item in mini_snapshot.rglob("*")
        if item.is_file()
    }
    closure = _collect_runtime_closure(mini_snapshot, files)
    assert "index.html" not in closure
    assert "assets/embed-abc.js" in closure
    assert "assets/chunk-def.js" in closure
    assert "assets/ort-wasm-simd-threaded-X1.wasm" in closure
    assert "assets/mujoco-Q2.wasm" in closure
    assert "assets/obj-mesh-Y2.wasm" in closure
    assert "assets/editor-main-unused.js" not in closure
    assert "wasm/obj-parser/objParser.js" in closure
    assert "wasm/obj-parser/objParser.wasm" in closure
    assert "embed.webmanifest" in closure
    assert "favicon.svg" in closure


def test_scan_chunk_references_resolves_relative_to_chunk_dir(mini_snapshot: Path):
    index = {"assets/chunk-def.js", "assets/obj-mesh-Y2.wasm"}
    refs = _scan_chunk_references('load("obj-mesh-Y2.wasm")', "assets/chunk-def.js", index)
    assert refs == {"assets/obj-mesh-Y2.wasm"}


def test_collect_site_asset_roots_rules():
    roots = collect_site_asset_roots(_release_dict())
    # The packaged model is restored from release.zip; only the project root remains.
    assert roots == {"project"}

    http = _release_dict()
    http["project"]["manifest"]["resources"][1]["source"] = "https://cdn.example.com/p.onnx"
    http["assets"] = []
    with pytest.raises(BotLabError, match="unarchived https://"):
        collect_site_asset_roots(http)

    unsafe = _release_dict()
    unsafe["project"]["assetPaths"] = ["../escape.xml"]
    with pytest.raises(BotLabError, match="assetPaths entry"):
        collect_site_asset_roots(unsafe)


def test_root_level_resource_uses_the_project_asset_base():
    config = _release_dict()
    config["project"]["manifest"]["resources"][0]["source"] = "robot.xml"
    config["project"]["assetPaths"] = ["robot.xml"]
    assert collect_site_asset_roots(config) == {"project"}


def test_read_release_config_requires_compiled_controller():
    config = read_release_config(_zip_bytes(_release_dict()))
    assert config["runtime"] == "python"

    no_controller = _release_dict()
    del no_controller["project"]["compiledController"]
    with pytest.raises(BotLabError, match="compiled controller plan"):
        read_release_config(_zip_bytes(no_controller))

    wrong_version = _release_dict()
    wrong_version["version"] = 4
    with pytest.raises(BotLabError, match="version 4"):
        read_release_config(_zip_bytes(wrong_version))


def test_build_site_directory_layout(
    project_dir: Path, mini_snapshot: Path, tmp_path: Path, monkeypatch
):
    from botlab import _site

    monkeypatch.setattr(_site, "ensure_browser_runtime", lambda: mini_snapshot)
    package = create_release_package(
        load_project(project_dir), metadata={"name": "Test Duck"}
    )
    out_dir = tmp_path / "site"
    stats = build_site_directory(project_dir, package, out_dir)

    assert stats["project"] == "Test Duck"
    assert (out_dir / "index.html").read_text(encoding="utf-8").startswith("<script")
    assert (out_dir / "assets" / "embed-abc.js").is_file()
    assert (out_dir / "wasm" / "obj-parser" / "objParser.wasm").is_file()
    assert not (out_dir / "assets" / "editor-main-unused.js").exists()
    # The whole project directory is mirrored under /project/ like the local server graft.
    assert (out_dir / "project" / "assets" / "robot" / "scene.xml").is_file()
    assert (out_dir / "project" / "controller.py").is_file()
    assert (out_dir / "release.zip").read_bytes() == package

    with pytest.raises(BotLabError, match="already exists"):
        build_site_directory(project_dir, package, out_dir)


def test_build_site_directory_rejects_shared_site_assets(
    project_dir: Path, mini_snapshot: Path, tmp_path: Path, monkeypatch
):
    from botlab import _site

    monkeypatch.setattr(_site, "ensure_browser_runtime", lambda: mini_snapshot)
    package = create_release_package(
        load_project(project_dir), metadata={"name": "Test Duck"}
    )
    config = json.loads(
        zipfile.ZipFile(BytesIO(package)).read("release.json")
    )
    config["project"]["manifest"]["resources"].append(
        {"id": "builtin-robot", "kind": "robot", "source": "models/go2/scene.xml"}
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release.json", json.dumps(config))
        archive.writestr("project/main.py", "print('hi')\n")
    with pytest.raises(BotLabError, match="models/go2"):
        build_site_directory(project_dir, buffer.getvalue(), tmp_path / "site")


def test_build_site_with_a_root_level_robot_asset(
    project_dir: Path, mini_snapshot: Path, tmp_path: Path, monkeypatch
):
    from botlab import _site

    monkeypatch.setattr(_site, "ensure_browser_runtime", lambda: mini_snapshot)
    (project_dir / "assets/robot/scene.xml").rename(project_dir / "robot.xml")
    controller = project_dir / "controller.py"
    controller.write_text(
        controller.read_text(encoding="utf-8").replace("assets/robot/scene.xml", "robot.xml"),
        encoding="utf-8",
    )
    package = create_release_package(load_project(project_dir), metadata={"name": "Root Robot"})
    out = tmp_path / "root-site"
    build_site_directory(project_dir, package, out)
    assert (out / "project/robot.xml").is_file()
    config = read_release_config((out / "release.zip").read_bytes())
    assert config["project"]["manifest"]["resources"][0]["source"] == "robot.xml"
