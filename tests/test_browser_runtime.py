"""Offline browser runtime archive and cache tests."""

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from botlab import __version__
from botlab._browser_runtime import (
    MANIFEST,
    browser_runtime_cache_dir,
    ensure_browser_runtime,
    export_browser_runtime,
    install_browser_runtime,
)
from botlab._host import BotLabError


def _dist(tmp_path: Path) -> Path:
    dist = tmp_path / "BotLab/dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html><head></head><body>editor</body></html>")
    (dist / "embed.html").write_text("<html>embed</html>")
    (dist / "assets").mkdir()
    (dist / "assets/mujoco.wasm").write_bytes(b"runtime")
    for name in ("d", "models", "templates"):
        (dist / name).mkdir()
        (dist / name / "example.bin").write_bytes(b"skip")
    return dist


def test_export_install_and_validate_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BOTLAB_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    _dist(tmp_path)
    archive = tmp_path / "runtime.zip"
    assert export_browser_runtime(tmp_path / "BotLab", archive) == 3
    with zipfile.ZipFile(archive) as packed:
        assert set(packed.namelist()) == {MANIFEST, "index.html", "embed.html", "assets/mujoco.wasm"}
        assert json.loads(packed.read(MANIFEST))["version"] == __version__
    installed = install_browser_runtime(archive)
    assert installed == browser_runtime_cache_dir()
    assert "BOTLAB_LOCAL_PROJECT=true" in (installed / "index.html").read_text()
    assert ensure_browser_runtime() == installed


def test_install_rejects_wrong_version_and_unsafe_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BOTLAB_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    dist = _dist(tmp_path)
    archive = tmp_path / "runtime.zip"
    export_browser_runtime(dist.parent, archive)
    with zipfile.ZipFile(archive) as packed:
        payload = {name: packed.read(name) for name in packed.namelist()}
    manifest = json.loads(payload[MANIFEST])
    manifest["version"] = "0.0.0"
    payload[MANIFEST] = json.dumps(manifest).encode()
    with zipfile.ZipFile(archive, "w") as packed:
        for name, data in payload.items():
            packed.writestr(name, data)
    with pytest.raises(BotLabError, match="does not match"):
        install_browser_runtime(archive)
    assert not browser_runtime_cache_dir().exists()

    manifest["version"] = __version__
    payload[MANIFEST] = json.dumps(manifest).encode()
    with zipfile.ZipFile(archive, "w") as packed:
        for name, data in payload.items():
            packed.writestr(name, data)
        packed.writestr("../escape", "bad")
    with pytest.raises(BotLabError, match="unsafe"):
        install_browser_runtime(archive)
    assert not (tmp_path / "escape").exists()


def test_checksum_symlink_and_drive_path_rejected_without_replacing_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BOTLAB_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    _dist(tmp_path)
    archive = tmp_path / "runtime.zip"
    export_browser_runtime(tmp_path / "BotLab", archive)
    installed = install_browser_runtime(archive)
    with zipfile.ZipFile(archive) as packed:
        payload = {name: packed.read(name) for name in packed.namelist()}

    damaged = dict(payload)
    damaged["embed.html"] = b"bad"
    with zipfile.ZipFile(archive, "w") as packed:
        for name, data in damaged.items():
            packed.writestr(name, data)
    with pytest.raises(BotLabError, match="size mismatch"):
        install_browser_runtime(archive)
    assert (installed / "embed.html").read_bytes() == payload["embed.html"]

    for bad_name in ("C:/escape", "assets/link"):
        with zipfile.ZipFile(archive, "w") as packed:
            for name, data in payload.items():
                if name == "assets/mujoco.wasm" and bad_name == "assets/link":
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    packed.writestr(info, data)
                else:
                    packed.writestr(name, data)
            if bad_name == "C:/escape":
                packed.writestr(bad_name, b"bad")
        with pytest.raises(BotLabError, match="unsafe|non-file"):
            install_browser_runtime(archive)
        assert (installed / "embed.html").read_bytes() == payload["embed.html"]


def test_ensure_downloads_once_and_reuses_verified_cache(tmp_path: Path, monkeypatch):
    from botlab import _browser_runtime

    monkeypatch.setenv("BOTLAB_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    _dist(tmp_path)
    archive = tmp_path / "runtime.zip"
    export_browser_runtime(tmp_path / "BotLab", archive)
    downloaded = []

    def fake_open(url, timeout):
        downloaded.append((url, timeout))
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(_browser_runtime.urllib.request, "urlopen", fake_open)
    target = ensure_browser_runtime()
    assert (target / "assets/mujoco.wasm").read_bytes() == b"runtime"
    assert ensure_browser_runtime() == target
    assert len(downloaded) == 1
    assert f"v{__version__}" in downloaded[0][0]


def test_failed_cache_swap_restores_previous_runtime(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BOTLAB_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    _dist(tmp_path)
    archive = tmp_path / "runtime.zip"
    export_browser_runtime(tmp_path / "BotLab", archive)
    installed = install_browser_runtime(archive)
    original_rename = Path.rename

    def fail_staging_swap(path, target):
        if path.name.startswith(f".{__version__}-") and not path.name.startswith(f".{__version__}-backup"):
            raise OSError("simulated swap failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_swap)
    with pytest.raises(OSError, match="simulated swap failure"):
        install_browser_runtime(archive)
    assert (installed / "assets/mujoco.wasm").read_bytes() == b"runtime"
