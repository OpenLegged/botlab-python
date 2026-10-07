"""CLI tests for build/validate/run argument handling (project logic is
covered by test_project.py)."""

from __future__ import annotations

import json
import zipfile

import pytest

from botlab._cli import main


def test_version_reports_package_version(capsys):
    from botlab import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"botlab {__version__}"


def test_build_writes_default_release_zip(project_dir, capsys, monkeypatch):
    monkeypatch.chdir(project_dir)
    assert main(["build"]) == 0
    output = project_dir / "release.zip"
    assert output.is_file()
    assert "release.zip" in capsys.readouterr().out
    archive = zipfile.ZipFile(output)
    assert "release.json" in archive.namelist()
    config = json.loads(archive.read("release.json"))
    assert config["format"] == "botlab-release"
    assert config["version"] == 3


def test_build_supports_explicit_out_path(project_dir, tmp_path):
    target = tmp_path / "exports" / "my-release.zip"
    target.parent.mkdir()
    assert main(["build", str(project_dir), "--out", str(target)]) == 0
    assert target.is_file()
    assert (project_dir / "release.zip").exists() is False


def test_build_rejects_missing_output_directory(project_dir):
    target = project_dir / "nope" / "release.zip"
    assert main(["build", str(project_dir), "--out", str(target)]) == 2


def test_validate_reports_package_size(project_dir, capsys):
    assert main(["validate", str(project_dir)]) == 0
    assert "bytes" in capsys.readouterr().out


def test_run_no_server_writes_out(project_dir, tmp_path):
    target = tmp_path / "exported.zip"
    assert main(["run", str(project_dir), "--no-server", "--out", str(target)]) == 0
    assert target.is_file()


def test_run_headless_rejects_browser_execution():
    with pytest.raises(SystemExit) as error:
        main(["run", "--headless", "--browser"])
    assert error.value.code == 2


@pytest.mark.parametrize("option", [["--out", "release.zip"], ["--no-server"]])
def test_run_headless_rejects_release_only_options(option, tmp_path, capsys):
    assert main(["run", str(tmp_path), "--headless", *option]) == 2
    assert "--headless cannot be combined" in capsys.readouterr().err
    assert not (tmp_path / "release.zip").exists()


def test_docs_prints_bundled_docs_location(capsys):
    assert main(["docs"]) == 0
    out = capsys.readouterr().out
    assert "component-reference.md" in out
    assert "decision-table.md" in out
    assert "error-guide.md" in out
