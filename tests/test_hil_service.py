"""systemd user-service installation for the resident agent and the gateway.

Covers the pure pieces (token policy, unit body) and the CLI install/uninstall
flows with a recording systemctl/loginctl fake — no real systemd needed.
"""

from __future__ import annotations

import hashlib
import stat
import subprocess
from pathlib import Path

import pytest

from botlab.hil import service
from botlab.hil.agent import main as agent_main
from botlab.hil.gateway import main as gateway_main
from botlab.hil.protocol import HilError

GATEWAY = "wss://boards.example.com"


class FakeLogin:
    """Records systemctl/loginctl invocations; simulates their outcomes."""

    def __init__(self, home: Path, *, available=True, linger="no", enable_linger_ok=True):
        self.home = home
        self.available = available
        self.linger = linger
        self.enable_linger_ok = enable_linger_ok
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "systemctl" and not self.available:
            return subprocess.CompletedProcess(argv, 1, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def check_output(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:2] == ["loginctl", "enable-linger"]:
            if not self.enable_linger_ok:
                raise subprocess.CalledProcessError(1, argv)
            self.linger = "yes"
            return ""
        assert argv[:2] == ["loginctl", "show-user"]
        return self.linger + "\n"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Isolated HOME + unit directory with a working fake systemctl."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(service, "UNIT_TARGET", tmp_path / ".config/systemd/user")
    fake = FakeLogin(tmp_path)
    monkeypatch.setattr(service.subprocess, "run", fake.run)
    monkeypatch.setattr(service.subprocess, "check_output", fake.check_output)
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/systemctl")
    return fake


def _agent_install(fake: FakeLogin, *extra) -> int:
    return agent_main(["install", "--gateway", GATEWAY, "--device", "desk-x5",
                       "--root", str(fake.home / "botlab"), *extra])


def _agent_unit(fake: FakeLogin) -> Path:
    name = "botlab-agent-" + service.unit_hash(str((fake.home / "botlab").resolve())) + ".service"
    return fake.home / ".config/systemd/user" / name


def test_check_env_token_rejects_unsafe_values():
    assert service.check_env_token("T", "abc-_123.XYZ") == "abc-_123.XYZ"
    for bad in ("", "has space", 'quo"te', "apo'strophe", "back\\slash", "tab\tchar"):
        with pytest.raises(HilError):
            service.check_env_token("T", bad)


def test_unit_body_quotes_and_escapes():
    body = service.unit_body(
        "BotLab test", ["/usr/bin/python3", "-m", "botlab.hil.agent", "connect",
                        "--gateway", "wss://a/b%20c"],
        Path("/tmp/token dir/env"), network_online=True)
    assert "Description=BotLab test\n" in body
    assert "After=network-online.target\nWants=network-online.target\n" in body
    assert 'EnvironmentFile="/tmp/token dir/env"\n' in body
    assert 'ExecStart="/usr/bin/python3" "-m" "botlab.hil.agent" "connect" ' \
           '"--gateway" "wss://a/b%%20c"\n' in body
    assert "Restart=on-failure\nRestartSec=3\nKillMode=control-group" in body
    assert body.rstrip().endswith("WantedBy=default.target")


def test_unit_body_keeps_non_ascii_paths_literal():
    # systemd unquoting has no \uXXXX escapes; such paths must stay UTF-8.
    body = service.unit_body("t", ["/opt/产品/python"], Path("/opt/产品/env"))
    assert '"/opt/产品/python"' in body
    assert '"/opt/产品/env"' in body
    assert "\\u" not in body


def test_agent_install_writes_files_and_enables(fake, monkeypatch):
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "token-1")
    assert _agent_install(fake) == 0
    unit, env = _agent_unit(fake), fake.home / "botlab/agent/gateway-token.env"
    body = unit.read_text()
    assert "EnvironmentFile=" in body and str(env) in body
    assert '"--device" "desk-x5"' in body and f'"{GATEWAY}"' in body
    assert env.read_text() == "BOTLAB_HIL_TOKEN=token-1\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert ["systemctl", "--user", "daemon-reload"] in fake.calls
    assert ["systemctl", "--user", "enable", "--now", unit.name] in fake.calls


def test_agent_install_rejects_invalid_options(fake, monkeypatch):
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "token-1")
    assert agent_main(["install", "--gateway", "ws://192.168.1.5:50060",
                       "--device", "desk-x5", "--root", str(fake.home / "botlab")]) == 1
    assert agent_main(["install", "--gateway", GATEWAY, "--device", "bad id!",
                       "--root", str(fake.home / "botlab")]) == 1
    assert not (fake.home / "botlab").exists()


def test_agent_install_requires_a_token(fake, monkeypatch, capsys):
    monkeypatch.delenv("BOTLAB_HIL_TOKEN", raising=False)
    monkeypatch.setattr("botlab.hil.agent.getpass.getpass", lambda *a: "")
    assert _agent_install(fake) == 1
    assert "BOTLAB_HIL_TOKEN" in capsys.readouterr().err


def test_install_requires_systemd(fake, monkeypatch, capsys):
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "token-1")
    assert _agent_install(fake) == 1
    assert "systemd" in capsys.readouterr().err


def test_enable_linger_failure_reports_hint(fake, monkeypatch, capsys):
    failed = FakeLogin(fake.home, enable_linger_ok=False)
    monkeypatch.setattr(service.subprocess, "run", failed.run)
    monkeypatch.setattr(service.subprocess, "check_output", failed.check_output)
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "token-1")
    assert _agent_install(fake, "--enable-linger") == 1
    out = capsys.readouterr().out
    assert "autostart at boot: no" in out
    assert "loginctl enable-linger" in out


def test_uninstall_removes_files_and_reports_missing(fake, monkeypatch, capsys):
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "token-1")
    assert _agent_install(fake) == 0
    assert agent_main(["uninstall", "--root", str(fake.home / "botlab")]) == 0
    assert not _agent_unit(fake).exists()
    assert not (fake.home / "botlab/agent/gateway-token.env").exists()
    assert agent_main(["uninstall", "--root", str(fake.home / "botlab")]) == 1


def test_gateway_install_pins_token(fake, monkeypatch):
    monkeypatch.setenv("BOTLAB_HIL_GATEWAY_TOKEN", "pinned-1")
    assert gateway_main(["install", "--host", "127.0.0.1", "--port", "50061"]) == 0
    unit = fake.home / ".config/systemd/user/botlab-hil-gateway.service"
    env = fake.home / ".config/botlab/gateway-token.env"
    body = unit.read_text()
    assert '"botlab.hil.gateway"' in body and '"--port" "50061"' in body
    assert "network-online.target" not in body
    assert env.read_text() == "BOTLAB_HIL_GATEWAY_TOKEN=pinned-1\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_gateway_uninstall_removes_pinned_token(fake, monkeypatch):
    monkeypatch.setenv("BOTLAB_HIL_GATEWAY_TOKEN", "pinned-1")
    assert gateway_main(["install"]) == 0
    assert gateway_main(["uninstall"]) == 0
    assert not (fake.home / ".config/systemd/user/botlab-hil-gateway.service").exists()
    assert not (fake.home / ".config/botlab/gateway-token.env").exists()


def test_unit_hash_matches_provisioning_formula():
    assert service.unit_hash("/home/dev/botlab") == \
        hashlib.sha256(b"/home/dev/botlab").hexdigest()[:12]
