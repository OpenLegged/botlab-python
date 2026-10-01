"""Debug-mode ``botlab run`` tests: stored in-memory release + rebuild loop.

The debug release must be payload-identical to the publishable package (the
browser verifies the same SHA-256 integrity manifest either way), never write
a zip file, and rebuild in a subprocess when project source changes.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from botlab._cli import main
from botlab._debug import DebugReleaseProvider, _project_signature, build_debug_release
from botlab._host import BotLabError
from botlab._project import load_project
from botlab._release import create_release_package


def _zip_entries(package: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _assert_clean_zip(package: bytes) -> None:
    """The browser's parser: EOCD must sit exactly at the end of the blob.

    Python's zipfile tolerates trailing garbage (it scans backwards for the
    EOCD); the TS runtime rejects it — any stray stdout bytes after the zip
    break the browser, so the subprocess contract is "stdout is the zip and
    nothing else" and this check enforces it.
    """
    eocd = package.rfind(b"PK\x05\x06")
    assert eocd != -1, "no end-of-central-directory record"
    comment_length = int.from_bytes(package[eocd + 20:eocd + 22], "little")
    assert eocd + 22 + comment_length == len(package), "trailing bytes after the zip"


def _release_config(package: bytes) -> dict:
    return json.loads(_zip_entries(package)["release.json"])


def _provider_for(project_dir) -> DebugReleaseProvider:
    project = load_project(project_dir)
    package = build_debug_release(project)
    return DebugReleaseProvider(project_dir, package, _project_signature(project_dir))


def test_stored_release_matches_deflate_payload(project_dir):
    project = load_project(project_dir)
    stored = create_release_package(project, metadata={"name": "Test Duck"}, compression="stored")
    deflated = create_release_package(
        project, metadata={"name": "Test Duck"}, compression="deflate"
    )

    stored_entries = _zip_entries(stored)
    deflated_entries = _zip_entries(deflated)
    assert set(stored_entries) == set(deflated_entries)
    for name, content in deflated_entries.items():
        if name == "release.json":
            continue
        assert stored_entries[name] == content

    stored_config = _release_config(stored)
    deflated_config = _release_config(deflated)
    stored_config.pop("createdAt")
    deflated_config.pop("createdAt")
    assert stored_config == deflated_config


def test_compression_methods_match_selection(project_dir):
    project = load_project(project_dir)
    with zipfile.ZipFile(io.BytesIO(create_release_package(project, compression="stored"))) as a:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in a.infolist())
    with zipfile.ZipFile(io.BytesIO(create_release_package(project, compression="deflate"))) as a:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in a.infolist())


def test_unknown_compression_is_rejected(project_dir):
    project = load_project(project_dir)
    with pytest.raises(BotLabError):
        create_release_package(project, compression="lzma")


def test_signature_ignores_build_outputs(project_dir):
    before = _project_signature(project_dir)
    (project_dir / "release.zip").write_bytes(b"stale artifact")
    (project_dir / ".botlab.zip").write_bytes(b"stale artifact")
    assert _project_signature(project_dir) == before


def test_provider_serves_initial_bytes_without_rebuild(project_dir):
    provider = _provider_for(project_dir)
    assert provider() == provider()


def test_provider_rebuilds_after_source_edit(project_dir):
    provider = _provider_for(project_dir)
    original = provider()
    assert _release_config(original)["project"]["manifest"]["project"]["name"] == "Test Duck"

    controller = project_dir / "controller.py"
    controller.write_text(controller.read_text().replace("Test Duck", "Test Duck Two"))
    rebuilt = provider()

    rebuilt_name = _release_config(rebuilt)["project"]["manifest"]["project"]["name"]
    assert rebuilt_name == "Test Duck Two"
    assert provider() == rebuilt


def test_rebuilt_package_is_a_clean_zip(project_dir):
    provider = _provider_for(project_dir)
    controller = project_dir / "controller.py"
    controller.write_text(controller.read_text().replace("Test Duck", "Clean Zip Duck"))
    _assert_clean_zip(provider())


def test_rebuild_stdout_carries_only_the_zip(project_dir):
    """Regression: host log lines and user print()s must not trail the zip.

    The subprocess used to write the package through ``sys.stdout.buffer``
    while a buffered text-layer log line (``[info] [Python] sim.run(...)
    started``) flushed after it at interpreter exit; Python's zipfile
    tolerated the 50 trailing bytes, the browser runtime did not.
    """
    controller = project_dir / "controller.py"
    controller.write_text(
        controller.read_text().replace("Test Duck", "Noisy Duck")
        + "\nprint('user debug noise')\n"
    )
    provider = _provider_for(project_dir)
    controller.write_text(controller.read_text().replace("Noisy Duck", "Noisier Duck"))

    rebuilt = provider()
    _assert_clean_zip(rebuilt)
    assert _release_config(rebuilt)["project"]["manifest"]["project"]["name"] == "Noisier Duck"


def test_provider_raises_when_rebuild_fails(project_dir):
    provider = _provider_for(project_dir)
    (project_dir / "controller.py").write_text("def broken syntax((")
    with pytest.raises(BotLabError):
        provider()


def test_run_no_server_writes_no_zip(project_dir, capsys):
    assert main(["run", str(project_dir), "--no-server"]) == 0
    assert list(project_dir.glob("*.zip")) == []
    out = capsys.readouterr().out
    assert "debug release" not in out
    assert "plan:" not in out


def test_run_verbose_prints_compile_summary(project_dir, capsys, monkeypatch):
    monkeypatch.setenv("BOTLAB_VERBOSE", "1")
    assert main(["run", str(project_dir), "--no-server"]) == 0
    out = capsys.readouterr().out
    assert "plan:" in out
    assert "project" in out


def test_run_browser_verbose_prints_debug_release(project_dir, capsys, monkeypatch):
    monkeypatch.setenv("BOTLAB_VERBOSE", "1")
    assert main(["run", str(project_dir), "--browser", "--no-server"]) == 0
    out = capsys.readouterr().out
    assert "debug release" in out
