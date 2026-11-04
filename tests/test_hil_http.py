"""Local HIL HTTP boundary: bootstrap, origin isolation, and request limits."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from botlab._server import start_project_server


class Host:
    def handle(self, method, path, body=None):
        return {"selectedTarget": "x5", "method": method, "path": path, "body": body}


@pytest.fixture
def server(tmp_path):
    (tmp_path / "index.html").write_text("<html><head></head><body>editor</body></html>")
    httpd, base = start_project_server(tmp_path, None, "127.0.0.1", 0,
                                      ui_dir=tmp_path, release_provider=lambda: b"zip", hil=Host())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, base
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_bootstrap_and_authorized_state(server):
    httpd, base = server
    with urlopen(base) as response:
        source = response.read().decode()
        assert "window.BOTLAB_HIL=" in source
        assert httpd.botlab_hil_token in source
        assert response.headers["Cache-Control"] == "no-store"
    request = Request(base + "/hil/state", headers={"X-Botlab-Hil-Token": httpd.botlab_hil_token})
    with urlopen(request) as response:
        assert json.load(response)["selectedTarget"] == "x5"


@pytest.mark.parametrize("override", [{}, {"Origin": "https://other.example"}, {"Host": "other.example"}])
def test_rejects_untrusted_calls(server, override):
    httpd, base = server
    headers = {"Content-Type": "application/json"}
    if override:
        headers["X-Botlab-Hil-Token"] = httpd.botlab_hil_token
    headers.update(override)
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/hil/start", data=b"{}", headers=headers))
    assert error.value.code == 403


def test_rejects_oversized_body_before_read(server):
    httpd, base = server
    headers = {"Content-Type": "application/json", "X-Botlab-Hil-Token": httpd.botlab_hil_token,
               "Content-Length": str(49 * 1024 * 1024)}
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/hil/models", data=b"{}", headers=headers), timeout=2)
    assert error.value.code == 400


def test_valid_post_and_stale_token(server):
    httpd, base = server
    headers = {"Content-Type": "application/json", "X-Botlab-Hil-Token": httpd.botlab_hil_token,
               "Origin": base}
    with urlopen(Request(base + "/hil/start", data=b"{}", headers=headers)) as response:
        assert json.load(response)["body"] == {}
    headers["X-Botlab-Hil-Token"] = "old-token"
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/hil/start", data=b"{}", headers=headers))
    assert error.value.code == 403

@pytest.mark.parametrize("option", ["--native", "--headless", "--no-server"])
def test_hil_requires_live_browser_mode(project_dir, option, capsys):
    from botlab._cli import main
    assert main(["run", str(project_dir), "--hil", "x5", option]) == 2
    assert "standard live browser" in capsys.readouterr().err


def test_hil_rejects_old_browser_runtime(project_dir, tmp_path, monkeypatch, capsys):
    from botlab import _cli
    monkeypatch.setattr(_cli, "ensure_browser_runtime", lambda: tmp_path)
    assert _cli.main(["run", str(project_dir), "--hil", "x5"]) == 2
    assert "does not support HIL v1" in capsys.readouterr().err


def test_hil_cli_preselects_without_connecting(project_dir, tmp_path, monkeypatch):
    from botlab import _cli
    (tmp_path / "botlab-capabilities.json").write_text('{"hilProtocol": 1}')
    monkeypatch.setattr(_cli, "ensure_browser_runtime", lambda: tmp_path)
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    calls = []
    def serve(*args, **kwargs):
        state = kwargs["hil"].state()
        calls.append(state)
        assert args[2] == "127.0.0.1"
        assert state["selectedTarget"] == "command-line"
        assert not state["connected"] and state["session"] is None
        kwargs["hil"].close()
    monkeypatch.setattr(_cli, "serve_project", serve)
    assert _cli.main(["run", str(project_dir), "--hil", "127.0.0.1:50052"]) == 0
    assert len(calls) == 1
