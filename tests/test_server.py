"""Server tests: the local-project endpoint, quiet access logs, startup links.

The ``botlab run`` debug loop serves the editor at ``/?local=1``; the editor
loads the project through ``/local-project.json`` and resolves project assets
under ``/project/``. These tests pin that contract plus the log-noise
regression (per-file access logs) without starting a full server thread.
"""

from __future__ import annotations

import json
import socketserver
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

import pytest

from botlab._server import (
    BotlabUIRequestHandler,
    local_project_payload,
    print_startup_banner,
    serve_project,
)


class _QuietServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


@contextmanager
def _running_server(project_dir: Path, tmp_path: Path):
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<html>editor</html>", encoding="utf-8")

    handler = BotlabUIRequestHandler
    handler.directory = str(ui_dir)
    server = _QuietServer(("127.0.0.1", 0), handler)
    server.botlab_ui_dir = ui_dir
    server.botlab_release_zip = None
    server.botlab_release_provider = None
    server.botlab_project_dir = project_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_local_project_payload_splits_sources_and_assets(project_dir: Path):
    payload = local_project_payload(project_dir)
    assert payload["format"] == "botlab-local-project"
    assert payload["version"] == 1
    assert payload["name"] == project_dir.name
    assert sorted(payload["files"]) == ["controller.py", "main.py", "robot.preset.yaml"]
    assert "from botlab import Model" in payload["files"]["controller.py"]
    assert payload["assets"] == [
        "assets/models/policy.onnx",
        "assets/robot/scene.xml",
    ]


def test_local_project_payload_ignores_build_outputs_and_caches(project_dir: Path):
    (project_dir / "release.zip").write_bytes(b"stale")
    (project_dir / ".botlab.zip").write_bytes(b"stale")
    (project_dir / ".git").mkdir()
    (project_dir / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (project_dir / "__pycache__").mkdir()
    (project_dir / "__pycache__" / "controller.cpython-312.pyc").write_bytes(b"\x00")
    payload = local_project_payload(project_dir)
    assert payload["assets"] == [
        "assets/models/policy.onnx",
        "assets/robot/scene.xml",
    ]
    assert sorted(payload["files"]) == ["controller.py", "main.py", "robot.preset.yaml"]


def test_local_project_endpoint_serves_json(project_dir: Path, tmp_path: Path):
    with _running_server(project_dir, tmp_path) as base:
        with urlopen(f"{base}/local-project.json") as response:
            assert response.status == 200
            assert "application/json" in response.headers["Content-Type"]
            payload = json.loads(response.read())
        assert payload["files"]["main.py"].startswith("from botlab import Simulation")
        assert "assets/robot/scene.xml" in payload["assets"]


def test_project_urls_serve_files_from_disk(project_dir: Path, tmp_path: Path):
    with _running_server(project_dir, tmp_path) as base:
        with urlopen(f"{base}/project/assets/robot/scene.xml") as response:
            assert response.read() == b"<mujoco/>"
        with urlopen(f"{base}/project/main.py") as response:
            assert b"Simulation" in response.read()


def test_project_urls_reject_escape(project_dir: Path, tmp_path: Path):
    from urllib.error import HTTPError

    with _running_server(project_dir, tmp_path) as base:
        with pytest.raises(HTTPError):
            urlopen(f"{base}/project/../../etc/passwd")


def test_quiet_access_log(project_dir: Path, tmp_path: Path, capsys):
    with _running_server(project_dir, tmp_path) as base:
        with capsys.disabled():
            with urlopen(f"{base}/local-project.json") as response:
                response.read()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_startup_banner_prints_local_link(capsys):
    print_startup_banner("microduck-project", "http://localhost:40501", "/")
    captured = capsys.readouterr()
    assert "[botlab] microduck-project ready" in captured.out
    assert "  ➜  Local:   " in captured.out
    assert "http://localhost:40501/" in captured.out
    assert "\x1b]8;;" in captured.out  # OSC 8 clickable links


def test_startup_banner_prints_network_link_when_lan_host(capsys, monkeypatch):
    from botlab import _server

    monkeypatch.setattr(_server, "_lan_ip", lambda: "192.168.1.42")
    print_startup_banner("duck", "http://localhost:40501", "/", host="0.0.0.0")
    captured = capsys.readouterr()
    assert "http://localhost:40501/" in captured.out
    assert "http://192.168.1.42:40501/" in captured.out


def test_startup_banner_omits_network_without_lan(capsys, monkeypatch):
    from botlab import _server

    monkeypatch.setattr(_server, "_lan_ip", lambda: None)
    print_startup_banner("duck", "http://localhost:40501", "/", host="0.0.0.0")
    captured = capsys.readouterr()
    assert "Network" not in captured.out


def test_serve_project_requires_exactly_one_release_source(project_dir: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        serve_project(project_dir, None, "127.0.0.1", 0, ui_dir=tmp_path)
    with pytest.raises(ValueError):
        serve_project(
            project_dir, tmp_path / "a.zip", "127.0.0.1", 0,
            ui_dir=tmp_path, release_provider=lambda: b"",
        )
