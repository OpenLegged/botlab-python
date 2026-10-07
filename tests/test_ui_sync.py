"""Small native viewer snapshot tests."""

from pathlib import Path

import pytest

from botlab._host import BotLabError
from botlab._ui_sync import sync_ui_snapshot, validate_viewer_snapshot


def _write(path: Path, content: str = "asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sync_copies_only_display_closure_and_replaces_old_runtime(tmp_path: Path):
    checkout = tmp_path / "BotLab"
    dist = checkout / "dist-viewer"
    _write(dist / "viewer.html", '<script type="module" src="/assets/viewer.js"></script>')
    _write(dist / "assets/viewer.js", 'import "./chunk.js"; const icon="/favicon.svg"')
    _write(dist / "assets/chunk.js", "const scene = 'Three.js'")
    _write(dist / "favicon.svg", "<svg/>")
    _write(dist / "assets/editor-unused.js", "unused")
    target = tmp_path / "ui"
    _write(target / "wasm/obj-parser/objParser.wasm", "stale")

    copied = sync_ui_snapshot(checkout, target)

    assert copied == 5
    assert (target / "index.html").read_bytes() == (target / "embed.html").read_bytes()
    assert (target / "assets/viewer.js").is_file()
    assert (target / "assets/chunk.js").is_file()
    assert (target / "favicon.svg").is_file()
    assert not (target / "assets/editor-unused.js").exists()
    assert not (target / "wasm").exists()
    validate_viewer_snapshot(target)


def test_sync_rejects_wasm_reference(tmp_path: Path):
    dist = tmp_path / "dist-viewer"
    _write(dist / "viewer.html", '<script src="/assets/viewer.js"></script>')
    _write(dist / "assets/viewer.js", 'fetch("/assets/mujoco.wasm")')
    with pytest.raises(BotLabError, match="WASM|simulation runtime"):
        sync_ui_snapshot(tmp_path, tmp_path / "ui")


def test_stale_full_snapshot_cannot_enter_wheel(tmp_path: Path):
    _write(tmp_path / "index.html", "<html/>")
    _write(tmp_path / "embed.html", "<html/>")
    _write(tmp_path / "assets/mujoco.wasm")
    with pytest.raises(BotLabError, match="WASM"):
        validate_viewer_snapshot(tmp_path)
