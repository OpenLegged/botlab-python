"""UI snapshot packaging tests."""

from pathlib import Path

from botlab._ui_sync import sync_ui_snapshot


def _write(path: Path, content: str = "asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sync_excludes_catalog_assets_but_keeps_runtime_files(tmp_path: Path):
    checkout = tmp_path / "BotLab"
    dist = checkout / "dist"
    _write(dist / "index.html", "<html><head></head><body>editor</body></html>")
    _write(dist / "embed.html", "<html><head></head><body>embed</body></html>")
    _write(dist / "assets" / "main.js", "runtime")
    _write(dist / "assets" / "models" / "runtime.bin", "nested runtime asset")
    _write(dist / "templates" / "microduck" / "policy.onnx", "template")
    _write(dist / "models" / "g1" / "scene.xml", "shared model")
    _write(dist / "d" / "microduck" / "index.html", "demo")

    target = tmp_path / "ui"
    _write(target / "templates" / "stale.bin", "stale")

    copied = sync_ui_snapshot(checkout, target)

    assert copied == 4
    assert (target / "index.html").is_file()
    assert "BOTLAB_LOCAL_PROJECT=true" in (target / "index.html").read_text()
    assert (target / "embed.html").is_file()
    assert (target / "assets" / "main.js").is_file()
    assert (target / "assets" / "models" / "runtime.bin").is_file()
    assert not (target / "templates").exists()
    assert not (target / "models").exists()
    assert not (target / "d").exists()
