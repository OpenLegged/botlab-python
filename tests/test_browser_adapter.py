"""CLI adapter contract: project imports, runtime ownership, and isolation."""

import sys
from pathlib import Path

import pytest

from botlab._cli import main


def test_adapter_receives_installed_runtime_without_compiling_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    # No controller.py: source-only browser controllers bypass host compilation.
    (project / "adapter_example.py").write_text(
        "from pathlib import Path\n"
        "def run(context):\n"
        "    assert context.project_dir == Path.cwd()\n"
        "    assert context.ui_dir.name == 'installed-runtime'\n"
        "    assert context.host == '127.0.0.1'\n"
        "    assert context.port == 12345\n"
        "    assert context.inference_endpoint == 'board:50051'\n"
        "    assert not context.open_browser\n"
        "    return 7\n"
    )
    monkeypatch.setattr("botlab._cli.ensure_browser_runtime", lambda: tmp_path / "installed-runtime")
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    try:
        assert main([
            "run", str(project), "--browser-adapter", "adapter_example:run",
            "--inference-endpoint", "board:50051", "--port", "12345", "--no-browser",
        ]) == 7
        assert Path.cwd() == previous_cwd
        assert sys.path == previous_path
    finally:
        sys.modules.pop("adapter_example", None)


@pytest.mark.parametrize("options", [
    ["--native"], ["--headless"], ["--out", "release.zip"],
    ["--no-server"], ["--report", "report.json"],
])
def test_incompatible_modes_rejected_before_loading_adapter(options, tmp_path, capsys):
    assert main(["run", str(tmp_path), "--browser-adapter", "absent:run", *options]) == 2
    assert "requires a live browser run" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["bad", "bad:path:run", "../bad:run", "absent:run"])
def test_invalid_adapter_is_actionable_error(entry, tmp_path):
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    assert main(["run", str(tmp_path), "--browser-adapter", entry]) == 2
    assert Path.cwd() == previous_cwd
    assert sys.path == previous_path


def test_endpoint_does_not_silently_run_local_inference(tmp_path, capsys):
    assert main(["run", str(tmp_path), "--inference-endpoint", "board:50051"]) == 2
    assert "requires --browser-adapter" in capsys.readouterr().err
