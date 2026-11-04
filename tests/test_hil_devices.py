"""Device adapters must not trust unknown hosts, leak credentials, or reuse host binaries."""
import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from botlab.hil import devices
from botlab.hil.protocol import HilError


def target(**updates):
    return {"transport": "ssh", "ssh": {"host": "board.local", "user": "robot"}, **updates}


def result(stdout=b"", code=0, stderr=b""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


@pytest.mark.parametrize("config", [
    {"ssh": {"host": "-oProxyCommand=bad", "user": "robot"}},
    {"ssh": {"host": "board", "user": "root;whoami"}},
    {"ssh": {"host": "board", "user": "robot", "port": True}},
    {"root": "/"}, {"root": "~/"}, {"root": "~/../other"},
])
def test_reject_invalid_device_before_any_subprocess(config, monkeypatch):
    monkeypatch.setattr(devices, "_run", lambda *_args, **_kwargs: pytest.fail("executed command"))
    with pytest.raises(HilError):
        devices.LinuxDevice(target(**config))


def test_password_transient_and_never_in_argv(tmp_path):
    device = devices.LinuxDevice(target(), {"password": "a secret $() password"})
    argv, env = device._ssh(tmp_path)
    assert "StrictHostKeyChecking=yes" in argv
    assert "a secret" not in repr(argv)
    assert env["SSH_ASKPASS_REQUIRE"] == "force"
    secret = tmp_path / "password"
    assert secret.stat().st_mode & 0o777 == 0o600
    assert secret.read_text() == "a secret $() password"
    assert "secret $()" not in (tmp_path / "askpass").read_text()
    assert "password" not in repr(device.target)
    device.close()
    assert not device.credentials


def test_ssh_fingerprint_mismatch_fails_before_auth(tmp_path, monkeypatch):
    key = base64.b64encode(b"observed key").decode()
    monkeypatch.setattr(devices, "_run", lambda *_args, **_kwargs:
                        result(f"board.local ssh-ed25519 {key}\n".encode()))
    device = devices.LinuxDevice(target(), {"password": "secret", "host_key": "SHA256:wrong"})
    with pytest.raises(HilError, match="host key changed"):
        device._ssh(tmp_path)
    assert not (tmp_path / "password").exists()


def test_accepted_fingerprint_is_pinned_in_private_known_hosts(tmp_path, monkeypatch):
    key = base64.b64encode(b"observed key").decode()
    monkeypatch.setattr(devices, "_run", lambda *_args, **_kwargs:
                        result(f"board.local ssh-ed25519 {key}\n".encode()))
    profile = target()
    profile["ssh"]["host_key"] = devices._fingerprint(key)
    device = devices.LinuxDevice(profile)
    argv, _ = device._ssh(tmp_path)
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in argv
    assert "GlobalKnownHostsFile=/dev/null" in argv
    assert key in (tmp_path / "known_hosts").read_text()


def test_download_matches_target_not_host():
    tags = devices._platforms({"machine": "aarch64", "libc": ["glibc", "2.31"]})
    assert tags[0] == "manylinux_2_31_aarch64"
    assert tags[-1] == "manylinux2014_aarch64"
    assert all("x86" not in tag and "macos" not in tag for tag in tags)
    with pytest.raises(HilError, match="musl"):
        devices._platforms({"machine": "aarch64", "libc": ["musl", "1.2"]})


def test_download_uses_cross_platform_binary_only_and_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("BOTLAB_WHEEL_CACHE", str(tmp_path))
    def run(argv, **kwargs):
        calls.append(argv)
        if "download" in argv:
            folder = Path(argv[argv.index("--dest") + 1])
            (folder / "numpy-test.whl").write_bytes(b"wheel")
        return result()
    monkeypatch.setattr(devices, "_run", run)
    info = {"machine": "aarch64", "libc": ["glibc", "2.35"], "python": [3, 10, 12]}
    first = devices.download_wheels(info)
    again = devices.download_wheels(info)
    assert first == again
    command, = [call for call in calls if "download" in call]
    assert "--only-binary=:all:" in command
    assert command[command.index("--python-version") + 1] == "3.10"
    assert "manylinux_2_35_aarch64" in command
    assert not any("rdk" in arg or "microduck" in arg for arg in command)


def test_bundle_contains_sdk_source_and_linux_wheels_not_host_binaries(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "numpy-1-cp310-manylinux2014_aarch64.whl"
    wheel.write_bytes(b"linux binary")
    bundle, digest = devices.build_agent_bundle(tmp_path, wheels)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
        manifest = archive.extractfile("manifest.json").read()
        assert hashlib.sha256(manifest).hexdigest() == digest
        assert "package/botlab/hil/agent.py" in names
        assert not any(name.endswith((".dylib", ".so", ".pyc")) for name in names)
        assert f"wheels/{wheel.name}" in names
        for name, checksum in json.loads(manifest).items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == checksum


def execute_install(tmp_path, entries, release=None):
    buffer = io.BytesIO()
    manifest = json.dumps({name: hashlib.sha256(content).hexdigest()
                           for name, content in entries.items()}, sort_keys=True).encode()
    digest = release or hashlib.sha256(manifest).hexdigest()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in {**entries, "manifest.json": manifest}.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    root = tmp_path.resolve() / "botlab"
    source = devices._INSTALL.replace("__PARAMS_JSON__", repr(json.dumps({"root": str(root), "release": digest})))
    run = subprocess.run([sys.executable, "-c", source], input=buffer.getvalue(), capture_output=True)
    return run, root, digest


def test_install_rejects_tar_traversal_before_write(tmp_path):
    run, root, _ = execute_install(tmp_path, {"../escaped": b"bad"})
    assert run.returncode != 0
    assert b"Invalid Agent bundle member" in run.stderr
    assert not (root / "escaped").exists()


def test_install_checks_digest_before_switching_release(tmp_path):
    run, root, _ = execute_install(tmp_path, {"package/a.py": b"x=1"}, release="0" * 64)
    assert run.returncode != 0
    assert b"checksum mismatch" in run.stderr
    assert not (root / "agent" / "current").exists()


def test_install_unpacks_wheel_into_separate_environment(tmp_path):
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example/module.py", "answer=42")
        archive.writestr("sample-1.data/purelib/extra.py", "answer=1")
        archive.writestr("sample-1.data/scripts/unneeded", "not installed")
    run, root, digest = execute_install(tmp_path, {
        "package/botlab/hil/agent.py": b"", "wheels/example.whl": wheel.getvalue()})
    assert run.returncode == 0, run.stderr.decode()
    release = root / "agent" / "releases" / digest
    assert (release / "deps" / "example" / "module.py").read_text() == "answer=42"
    assert (release / "deps" / "extra.py").exists()
    assert not (release / "deps" / "unneeded").exists()
    assert not (root / "agent" / "current").exists()  # Activated only after import preflight.
    assert (root / "projects").is_dir()
    assert not (root / "projects").is_symlink()


@pytest.mark.parametrize("entry", ["../escape.py", "/absolute.py"])
def test_install_rejects_malformed_dependency_wheel(tmp_path, entry):
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(entry, "bad")
    run, root, _ = execute_install(tmp_path, {"wheels/example.whl": wheel.getvalue()})
    assert run.returncode != 0
    assert b"Invalid dependency wheel member" in run.stderr
    assert not (root / "agent" / "current").exists()


def test_network_values_reject_unusable_address():
    for address in ["127.0.0.1", "0.0.0.0", "224.1.1.1", "192.168.1.0/24", "192.168.1.255/24"]:
        with pytest.raises(HilError):
            devices._network_values("static", address)
    assert devices._network_values("static", "192.168.9.100", "255.255.255.0") == (
        "192.168.9.100/24", "255.255.255.0", "")
    with pytest.raises(HilError):
        devices._network_values("static", "192.168.9.100/24", gateway="192.168.8.1")


def test_network_change_uses_nm_checkpoint_and_explicit_confirmation(monkeypatch):
    device = devices.LinuxDevice(target())
    commands = []
    def command(source, **kwargs):
        commands.append(source)
        return b'{"transaction":"0123456789abcdef0123456789abcdef", "requires_confirmation":true}'
    monkeypatch.setattr(device, "_command", command)
    changed = device.configure_network("usb0", "static", "192.168.9.10/24")
    assert changed["requires_confirmation"]
    assert "CheckpointCreate" in commands[0]
    assert "start_new_session=True" in commands[0]
    assert "CheckpointDestroy" not in commands[0]
    device.confirm_network(changed["transaction"])
    assert "CheckpointDestroy" in commands[1]
    with pytest.raises(HilError):
        device.confirm_network("../../etc/passwd")


def test_local_network_touches_only_selected_service_with_os_authorization(monkeypatch):
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(devices, "local_interfaces", lambda: [{"name": "en9", "addresses": []}])
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if "-listnetworkserviceorder" in argv:
            return result(b"(1) Wi-Fi\n(Hardware Port: Wi-Fi, Device: en0)\n(2) USB Board\n(Hardware Port: USB, Device: en9)\n")
        return result(b"Manual Configuration\nIP address: 192.168.9.100\n")
    monkeypatch.setattr(devices, "_run", run)
    state = devices.configure_local_network("en9", "static", "192.168.9.100/24")
    assert state["service"] == "USB Board"
    auth, = [call for call in calls if call[0] == "osascript"]
    assert "with administrator privileges" in auth[2]
    assert "USB Board" in auth[2]
    assert "Wi-Fi" not in auth[2]


def test_discovery_does_not_probe_neighbors_without_interface(monkeypatch):
    monkeypatch.setattr(devices, "local_interfaces", lambda: [{"name": "en9", "addresses": [], "active": True}])
    monkeypatch.setattr(devices, "_run", lambda *_args, **_kwargs: pytest.fail("read unscoped neighbors"))
    found = devices.discover_devices(profile="rdk-x5")
    assert found["candidates"] == []
    assert found["requires_interface"]


def test_device_inspection_rejects_unsupported_python(monkeypatch):
    device = devices.LinuxDevice(target())
    monkeypatch.setattr(device, "_python", lambda *_args, **_kwargs:
                        json.dumps({"system": "Linux", "python": [3, 9], "machine": "aarch64"}))
    with pytest.raises(HilError, match="Python 3.10"):
        device.inspect()


@pytest.mark.parametrize("probe_result, expected", [
    (0, True),
    (1, False),
    (subprocess.TimeoutExpired("systemctl", 5), False),
    (OSError("user manager unavailable"), False),
])
@pytest.mark.parametrize("agent_installed", [False, True])
def test_inspection_checks_systemd_user_manager_independently_of_agent(
    tmp_path, monkeypatch, capsys, probe_result, expected, agent_installed,
):
    root = tmp_path / "botlab"
    if agent_installed:
        (root / "agent" / "current").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None)
    monkeypatch.setattr(devices.subprocess, "check_output", lambda *_args, **_kwargs: b"[]")
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if isinstance(probe_result, Exception):
            raise probe_result
        return result(code=probe_result)
    monkeypatch.setattr(devices.subprocess, "run", run)
    exec(devices._INSPECT.replace("__ROOT_JSON__", repr(str(root))), {})
    info = json.loads(capsys.readouterr().out)
    assert info["systemd_user_available"] is expected
    assert info["agent_installed"] is agent_installed
    assert info["service_manager"] == "systemd"  # Existing compatibility field.
    assert calls == [(["systemctl", "--user", "show-environment"], {"capture_output": True, "timeout": 5})]


def test_inspection_without_systemctl_reports_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(devices.subprocess, "check_output", lambda *_args, **_kwargs: b"[]")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("no systemctl installed"))
    exec(devices._INSPECT.replace("__ROOT_JSON__", repr(str(tmp_path / "botlab"))), {})
    info = json.loads(capsys.readouterr().out)
    assert info["systemd_user_available"] is False
    assert info["service_manager"] == "process"


def test_failed_dependency_preflight_keeps_previous_release(tmp_path):
    root = tmp_path.resolve() / 'botlab'
    agent = root / 'agent'
    old = agent / 'releases' / ('1' * 64)
    new = agent / 'releases' / ('2' * 64)
    old.mkdir(parents=True)
    (new / 'package').mkdir(parents=True)
    (new / 'deps').mkdir()
    (new / 'deps' / 'numpy.py').write_text('raise ImportError("incompatible new runtime")')
    (agent / 'current').symlink_to(old)
    params = {'root': str(root), 'release': '2' * 64, 'port': 50052, 'backend': 'onnxruntime'}
    source = devices._START.replace('__PARAMS_JSON__', repr(json.dumps(params)))
    run = subprocess.run([sys.executable, '-c', source], capture_output=True)
    assert run.returncode != 0
    assert b'incompatible new runtime' in run.stderr
    assert (agent / 'current').resolve() == old
    assert not (agent / 'agent.pid').exists()


def test_agent_launch_isolated_from_global_site_packages():
    assert "[sys.executable, '-S', '-c'" in devices._START
    assert "args = [sys.executable, '-S', '-m', 'botlab.hil.agent'" in devices._START


@pytest.mark.parametrize('name', ['_INSPECT', '_INSTALL', '_START', '_CONNECT', '_NETWORK_BEGIN', '_NETWORK_APPLY', '_NETWORK_CONFIRM'])
def test_embedded_remote_scripts_are_valid_python(name):
    source = (getattr(devices, name).replace('__ROOT_JSON__', repr('~/botlab'))
              .replace('__PARAMS_JSON__', repr(json.dumps({'root': '~/botlab', 'release': '0' * 64})))
              .replace('__APPLY_SOURCE__', repr(devices._NETWORK_APPLY))
              .replace('__TRANSACTION__', repr('0' * 32)))
    compile(source, name, 'exec')


def test_open_tunnel_holds_exec_session_and_waits_for_real_hello(monkeypatch):
    import socket
    import threading

    from botlab.hil.protocol import receive, send

    calls = []
    seen = {}
    finished = threading.Event()
    class Process:
        stdin = io.BytesIO()
        stderr = io.BytesIO()
        def poll(self):
            return 0 if self.stdin.closed else None
        def wait(self, timeout=None):
            assert self.stdin.closed
            return 0
    process = Process()
    def popen(argv, **kwargs):
        kwargs['stdout'].write(b'BOTLAB_TARGET_READY\n')
        kwargs['stdout'].flush()
        seen.update(argv=argv, kwargs=kwargs)
        local_port = int(argv[argv.index('-L')+1].split(':')[1])
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', local_port))
        listener.listen()
        def server():
            try:
                # Forwarding is ready before the remote Agent: an accepted local
                # socket alone must not make open_tunnel return success.
                first, _ = listener.accept()
                first.close()
                with listener.accept()[0] as stream:
                    calls.append(receive(stream))
                    send(stream, {'ok': True, 'device': {'backend': 'onnxruntime'}})
            finally:
                listener.close()
                finished.set()
        threading.Thread(target=server, daemon=True).start()
        return process
    device = devices.LinuxDevice(target())
    monkeypatch.setattr(device, '_ssh', lambda _directory: (['ssh', 'robot@board.local'], {}))
    monkeypatch.setattr(devices.subprocess, 'Popen', popen)
    tunnel = device.open_tunnel()
    assert len(calls) == 1 and calls[0]['op'] == 'hello'
    assert '-N' not in seen['argv']
    assert 'StdinNull=no' in seen['argv']
    assert seen['argv'][-1].startswith('python3 -c ')
    assert seen['kwargs']['stdin'] == subprocess.PIPE
    assert not process.stdin.closed
    tunnel.close()
    assert process.stdin.closed
    assert finished.wait(1)


def test_invalid_agent_hello_closes_exec_and_credentials(monkeypatch):
    import socket
    import threading

    from botlab.hil.protocol import receive, send

    seen = {}
    class Process:
        stdin = io.BytesIO()
        stderr = io.BytesIO()
        def poll(self):
            return 0 if self.stdin.closed else None
        def wait(self, timeout=None):
            return 0
    process = Process()
    def popen(argv, **kwargs):
        kwargs['stdout'].write(b'BOTLAB_TARGET_READY\n')
        kwargs['stdout'].flush()
        listener = socket.socket()
        listener.bind(('127.0.0.1', int(argv[argv.index('-L')+1].split(':')[1])))
        listener.listen()
        def server():
            try:
                with listener.accept()[0] as stream:
                    receive(stream)
                    send(stream, {'ok': False, 'error': 'not ready'})
            finally:
                listener.close()
        threading.Thread(target=server, daemon=True).start()
        return process
    device = devices.LinuxDevice(target())
    def ssh(directory):
        seen['directory'] = directory
        return ['ssh', 'robot@board.local'], {}
    monkeypatch.setattr(device, '_ssh', ssh)
    monkeypatch.setattr(devices.subprocess, 'Popen', popen)
    with pytest.raises(HilError, match='valid BotLab Agent hello'):
        device.open_tunnel()
    assert process.stdin.closed
    assert not Path(seen['directory']).exists()


def _start_local_connection_helper(root, port):
    source = devices._CONNECT.replace('__PARAMS_JSON__', repr(json.dumps({'root': str(root), 'port': port})))
    return subprocess.Popen([sys.executable, '-c', source], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _make_fake_agent(root, port):
    agent = root / 'agent'
    agent.mkdir(parents=True)
    source = '''import json, os, socketserver, struct
from pathlib import Path
folder = Path(__file__).parent
with (folder/'starts').open('a') as stream:
    stream.write(str(os.getpid())+'\\n')
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        def read(size):
            data = b''
            while len(data) < size:
                part = self.request.recv(size-len(data))
                if not part:
                    raise EOFError
                data += part
            return data
        try:
            message = json.loads(read(struct.unpack('!I', read(4))[0]))
            payload = json.dumps({'version':1,'ok':True,'device':{'backend':'onnxruntime'}}).encode()
            self.request.sendall(struct.pack('!I',len(payload))+payload)
        except (EOFError, OSError):
            pass
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
with Server(('127.0.0.1', PORT), Handler) as server:
    server.serve_forever()
'''.replace('PORT', str(port))
    (agent / 'start-agent.py').write_text(source)
    return agent


def _wait_agent_port(port, children, timeout=5):
    import socket
    import time

    from botlab.hil.protocol import receive, send
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in children:
            assert process.poll() is None, process.stderr.read().decode()
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2) as stream:
                stream.settimeout(.2)
                send(stream, {'op': 'hello'})
                if receive(stream).get('ok'):
                    return
        except (OSError, EOFError):
            pass
        time.sleep(.05)
    pytest.fail('Local fake Agent did not become ready')


def test_connection_helper_requires_installation(tmp_path):
    child = _start_local_connection_helper(tmp_path / 'absent', 50052)
    _, error = child.communicate(timeout=5)
    assert child.returncode != 0
    assert b'Install and connect first' in error


def test_connection_helper_starts_once_and_keeps_login_until_eof(tmp_path):
    import os
    import signal
    import socket
    import time
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    root = tmp_path / 'board with spaces'
    agent = _make_fake_agent(root, port)
    children = [_start_local_connection_helper(root, port), _start_local_connection_helper(root, port)]
    try:
        _wait_agent_port(port, children)
        # Both logins remain active after readiness; their locks must have been
        # released so the second connection reuses the same running Agent.
        time.sleep(.2)
        assert all(child.poll() is None for child in children)
        assert len((agent / 'starts').read_text().splitlines()) == 1
        children[0].stdin.close()
        children[0].wait(timeout=3)
        assert children[0].returncode == 0
        assert children[1].poll() is None
        _wait_agent_port(port, [children[1]])
        # A fresh login must not restart an Agent used by another connection.
        third = _start_local_connection_helper(root, port)
        children.append(third)
        time.sleep(.2)
        assert third.poll() is None
        assert len((agent / 'starts').read_text().splitlines()) == 1
    finally:
        for child in children:
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        if (agent / 'agent.pid').exists():
            try:
                os.kill(int((agent / 'agent.pid').read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
