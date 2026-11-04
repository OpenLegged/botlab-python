"""Linux target bootstrap over verified SSH; no board or project-specific runtime.

The local machine downloads Linux wheels, so targets need Python and SSH but do
not need pip, virtualenv, Internet access or a preinstalled BotLab service.
Passwords are transient and never enter an argv, target profile or remote file.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import ipaddress
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .protocol import HilError, error_details, receive, send

Progress = Callable[[dict], None]
_AGENT_PORT = 50052
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:%_-]{0,252}$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}\$?$")
_REQUIREMENTS = ("numpy>=1.24,<3", "onnxruntime>=1.18,<2", "PyYAML>=6.0,<7")


def _run(argv, *, data=None, env=None, timeout=30, check=True):
    try:
        result = subprocess.run(argv, input=data, capture_output=True,
                                env=env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HilError(f"Device command failed: {type(exc).__name__}") from exc
    if check and result.returncode:
        # Never include argv: it may contain a command being sent to the target.
        detail = result.stderr.decode("utf-8", "replace").strip()[-1800:]
        raise HilError(detail or f"Device command exited with status {result.returncode}")
    return result


def _ssh_config(target):
    config = target.get("ssh", {})
    if not isinstance(config, dict):
        raise HilError("SSH settings must be an object")
    host, user = config.get("host", ""), config.get("user", "")
    port = config.get("port", 22)
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        raise HilError("A valid SSH hostname or IP address is required")
    if not isinstance(user, str) or not _USER.fullmatch(user):
        raise HilError("A valid SSH username is required")
    if type(port) is not int or not 1 <= port <= 65535:
        raise HilError("SSH port must be between 1 and 65535")
    root = target.get("root", "~/botlab")
    if (not isinstance(root, str) or len(root) > 512 or "\x00" in root
            or "\n" in root or not (root.startswith("~/") or root.startswith("/"))
            or root.rstrip("/") in ("", "~", "/home", "/usr", "/etc", "/opt", "/var")
            or ".." in Path(root).parts):
        raise HilError("Target root must be an absolute or ~/ path to a dedicated BotLab directory")
    identity = config.get("identity_file")
    if identity is not None and (not isinstance(identity, str) or not identity or "\x00" in identity):
        raise HilError("SSH identity_file must be a file path")
    return host, user, port, root


def _fingerprint(key):
    try:
        raw = base64.b64decode(key, validate=True)
    except ValueError as exc:
        raise HilError("Invalid SSH host key") from exc
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


def inspect_host_key(host: str, port: int = 22) -> dict:
    """Return observed fingerprints, without trusting or writing any of them."""
    _ssh_config({"ssh": {"host": host, "user": "probe", "port": port}})
    result = _run(["ssh-keyscan", "-T", "4", "-p", str(port), "-t", "ed25519,ecdsa,rsa", host],
                  timeout=16, check=False)
    keys = []
    for line in result.stdout.decode().splitlines():
        parts = line.split()
        if len(parts) == 3 and not line.startswith("#"):
            keys.append({"type": parts[1], "fingerprint": _fingerprint(parts[2]), "key": parts[2]})
    if not keys:
        raise HilError("No SSH host key received; check the device address and SSH service")
    lookup = host if port == 22 else f"[{host}]:{port}"
    known = _run(["ssh-keygen", "-F", lookup], check=False)
    known_keys = {parts[2] for line in known.stdout.decode().splitlines()
                  if len(parts := line.split()) >= 3 and not line.startswith("#")}
    return {"host": host, "port": port, "known": any(k["key"] in known_keys for k in keys),
            "keys": [{k: v for k, v in item.items() if k != "key"} for item in keys]}


class SSHTunnel:
    def __init__(self, process, port: int, credentials_dir=None):
        self.process = process
        self.address = ("127.0.0.1", port)
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self._credentials_dir = credentials_dir

    def close(self):
        # EOF ends the remote exec helper and its PAM session. -N forwarding
        # alone would not keep a non-lingering systemd user manager alive.
        stream = getattr(self.process, "stdin", None)
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        if self._credentials_dir:
            self._credentials_dir.cleanup()
            self._credentials_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class LinuxDevice:
    def __init__(self, target: dict, credentials: dict | None = None,
                 progress: Progress | None = None):
        self.host, self.user, self.port, self.root = _ssh_config(target)
        self.target = dict(target)
        self.credentials = dict(credentials or {})
        if set(self.credentials) - {"password", "host_key"}:
            raise HilError("Unknown SSH credential field")
        for value in self.credentials.values():
            if not isinstance(value, str) or len(value) > 4096 or "\x00" in value:
                raise HilError("Invalid SSH credential")
        self.progress = progress or (lambda _: None)

    def close(self):
        self.credentials.clear()

    def _notify(self, stage, message):
        self.progress({"stage": stage, "message": message})

    def _ssh(self, directory):
        argv = ["ssh", "-T", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                "-o", "LogLevel=ERROR", "-p", str(self.port)]
        identity = self.target["ssh"].get("identity_file")
        if identity:
            argv += ["-i", str(Path(identity).expanduser()), "-o", "IdentitiesOnly=yes"]
        env = os.environ.copy()
        env["LC_ALL"] = "C"  # Stable OpenSSH diagnostics for recoverable authentication failures.
        fingerprint = self.credentials.get("host_key") or self.target["ssh"].get("host_key")
        if fingerprint:
            # Explicitly accepted fingerprint is compared against a fresh scan.
            scan = _run(["ssh-keyscan", "-T", "4", "-p", str(self.port),
                         "-t", "ed25519,ecdsa,rsa", self.host], timeout=16, check=False)
            accepted = []
            for line in scan.stdout.decode().splitlines():
                fields = line.split()
                if len(fields) == 3 and not line.startswith("#") and _fingerprint(fields[2]) == fingerprint:
                    accepted.append(line)
            if not accepted:
                raise HilError("SSH host key changed or does not match the accepted fingerprint")
            known = Path(directory) / "known_hosts"
            known.write_text("\n".join(accepted) + "\n")
            known.chmod(0o600)
            argv += ["-o", f"UserKnownHostsFile={known}", "-o", "GlobalKnownHostsFile=/dev/null"]
        password = self.credentials.get("password")
        if password is not None:
            secret = Path(directory) / "password"
            secret.write_text(password)
            secret.chmod(0o600)
            askpass = Path(directory) / "askpass"
            askpass.write_text("#!/bin/sh\ncase \"$1\" in *assword*) exec cat "
                               + shlex.quote(str(secret)) + ";; *) exit 1;; esac\n")
            askpass.chmod(0o700)
            env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE="force", DISPLAY="botlab:0")
            argv += ["-o", "NumberOfPasswordPrompts=1"]
        else:
            argv += ["-o", "BatchMode=yes"]
        argv += [f"{self.user}@{self.host}"]
        return argv, env

    def _ssh_failure(self, returncode, stderr):
        detail = stderr.decode("utf-8", "replace")
        # Match OpenSSH's final authentication diagnostic, not a remote Python,
        # sudo or filesystem permission error. Ordinary transport failures also
        # exit 255, so the status alone cannot justify asking for a password.
        peer = re.escape(f"{self.user}@{self.host}")
        auth_required = returncode == 255 and re.search(
            rf"(?m)^{peer}: Permission denied \([A-Za-z0-9@._,-]+\)\.$", detail.strip()[-1800:]
        ) is not None
        if self.credentials.get("password"):
            detail = detail.replace(self.credentials["password"], "[redacted]")
        # Redact before trimming or truncating: either can split a credential
        # and leave a fragment that later full-password replacement misses.
        detail = detail.strip()[-1800:]
        metadata = ({"error_code": "ssh_auth_required", "target_id": self.target.get("id"),
                     "password_attempted": bool(self.credentials.get("password"))}
                    if auth_required else {})
        return HilError(detail or f"SSH command exited with status {returncode}", **metadata)

    def _command(self, command: str, *, data=None, timeout=30):
        with tempfile.TemporaryDirectory(prefix="botlab-ssh-") as directory:
            argv, env = self._ssh(directory)
            try:
                result = _run([*argv, command], data=data, env=env, timeout=timeout, check=False)
                if result.returncode:
                    raise self._ssh_failure(result.returncode, result.stderr)
                return result.stdout
            except HilError as exc:
                detail = str(exc)
                if self.credentials.get("password"):
                    detail = detail.replace(self.credentials["password"], "[redacted]")
                raise HilError(detail, **error_details(exc)) from exc

    def _python(self, source: str, *, data=None, timeout=30):
        return self._command("python3 -c " + shlex.quote(source), data=data, timeout=timeout)

    def inspect(self) -> dict:
        self._notify("inspect", "Checking target Linux, Python, architecture and available storage")
        source = _INSPECT.replace("__ROOT_JSON__", repr(self.root))
        result = json.loads(self._python(source))
        if result.get("system") != "Linux":
            raise HilError("This device adapter requires Linux")
        if tuple(result["python"][:2]) < (3, 10):
            raise HilError("The target requires Python 3.10 or newer")
        if result["machine"] not in ("aarch64", "x86_64"):
            raise HilError("Built-in ONNX CPU provisioning currently supports Linux aarch64 and x86_64")
        return result

    def provision(self) -> dict:
        info = self.inspect()
        backend = self.target.get("backend", "onnxruntime")
        if backend != "onnxruntime":
            raise HilError("Automatic provisioning currently supports onnxruntime; install custom backend dependencies separately")
        agent_port = self.target.get("agent_port", _AGENT_PORT)
        if type(agent_port) is not int or not 1 <= agent_port <= 65535:
            raise HilError("Invalid target Agent port")
        with tempfile.TemporaryDirectory(prefix="botlab-provision-") as directory:
            folder = Path(directory)
            self._notify("dependencies", "Downloading matching Linux wheels on this computer")
            wheel_dir = download_wheels(info, progress=self.progress)
            self._notify("package", "Packaging installed BotLab Agent and isolated dependencies")
            bundle, digest = build_agent_bundle(folder, wheel_dir)
            self._notify("upload", "Uploading Agent and dependencies over SSH")
            params = {"root": self.root, "release": digest, "port": agent_port, "backend": backend}
            source = _INSTALL.replace("__PARAMS_JSON__", repr(json.dumps(params)))
            result = json.loads(self._python(source, data=bundle.read_bytes(), timeout=180))
        self._notify("start", "Starting target Agent on loopback")
        source = _START.replace("__PARAMS_JSON__", repr(json.dumps(params)))
        started = json.loads(self._python(source, timeout=35))
        self._notify("ready", "Target Agent is ready; project deployment can begin")
        return {**result, **started, "device": info, "endpoint": f"tcp://127.0.0.1:{agent_port}"}

    def configure_network(self, interface: str, method: str, address: str | None = None,
                          gateway: str | None = None, netmask: str | None = None) -> dict:
        """Stage a NetworkManager change; expires after 90 s unless confirmed.

        Rollback belongs to NetworkManager, so closing SSH cannot disable it.
        Reconnect to the returned new_host and call confirm_network(transaction).
        """
        if not isinstance(interface, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", interface):
            raise HilError("Invalid target network interface")
        value, _, route = _network_values(method, address, netmask, gateway)
        params = {"interface": interface, "method": method, "address": value or "", "gateway": route or ""}
        source = (_NETWORK_BEGIN.replace("__PARAMS_JSON__", repr(json.dumps(params)))
                  .replace("__APPLY_SOURCE__", repr(_NETWORK_APPLY)))
        prefix = "" if self.user == "root" else "sudo -n "
        return json.loads(self._command(prefix + "python3 -c " + shlex.quote(source), timeout=30))

    def confirm_network(self, transaction: str) -> dict:
        if not isinstance(transaction, str) or not re.fullmatch(r"[a-f0-9]{32}", transaction):
            raise HilError("Invalid network transaction")
        source = _NETWORK_CONFIRM.replace("__TRANSACTION__", repr(transaction))
        prefix = "" if self.user == "root" else "sudo -n "
        return json.loads(self._command(prefix + "python3 -c " + shlex.quote(source), timeout=40))

    def open_tunnel(self) -> SSHTunnel:
        agent_port = self.target.get("agent_port", _AGENT_PORT)
        if type(agent_port) is not int or not 1 <= agent_port <= 65535:
            raise HilError("Invalid target Agent port")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        directory = tempfile.TemporaryDirectory(prefix="botlab-tunnel-")
        tunnel = None
        try:
            argv, env = self._ssh(directory.name)
            argv[-1:-1] = ["-o", "ExitOnForwardFailure=yes", "-o", "StdinNull=no", "-L",
                           f"127.0.0.1:{port}:127.0.0.1:{agent_port}"]
            source = _CONNECT.replace("__PARAMS_JSON__", repr(json.dumps({
                "root": self.root, "port": agent_port,
            })))
            # A persistent exec channel holds a genuine SSH/PAM login session.
            # Its stdin stays open until SSHTunnel.close; no linger/sudo needed.
            argv.append("python3 -c " + shlex.quote(source))
            ready_path = Path(directory.name) / "ready"
            with ready_path.open("wb") as output:
                process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=output,
                                           stderr=subprocess.PIPE, env=env)
            tunnel = SSHTunnel(process, port, directory)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                status = process.poll()
                if status is not None:
                    raise self._ssh_failure(status, process.stderr.read())
                with ready_path.open("rb") as output:
                    ready = b"BOTLAB_TARGET_READY\n" in output.read(4096)
                if not ready:
                    time.sleep(.1)
                    continue
                try:
                    with socket.create_connection(tunnel.address, timeout=.5) as stream:
                        stream.settimeout(.5)
                        send(stream, {"op": "hello"})
                        response = receive(stream)
                        if response.get("ok") is True and isinstance(response.get("device"), dict):
                            return tunnel
                        raise HilError("Target did not return a valid BotLab Agent hello")
                except (OSError, EOFError):
                    time.sleep(.1)
            raise HilError("Target Agent did not become ready; inspect target installation and Agent log")
        except Exception:
            if tunnel:
                tunnel.close()
            else:
                directory.cleanup()
            raise


def _platforms(info):
    machine = info["machine"]
    libc = info.get("libc", ["", ""])
    if libc[0] != "glibc":
        raise HilError("Built-in provisioning requires glibc Linux; musl needs a target plugin")
    try:
        major, minor = map(int, libc[1].split(".")[:2])
    except (ValueError, IndexError) as exc:
        raise HilError("Could not determine target glibc compatibility") from exc
    if major != 2 or minor < 17:
        raise HilError("Target glibc 2.17 or newer is required")
    return [f"manylinux_2_{n}_{machine}" for n in range(min(minor, 50), 16, -1)] + [f"manylinux2014_{machine}"]


def download_wheels(info: dict, *, progress: Progress | None = None) -> Path:
    platforms = _platforms(info)
    version = ".".join(str(n) for n in info["python"][:2])
    signature = hashlib.sha256(json.dumps([version, platforms, _REQUIREMENTS]).encode()).hexdigest()[:20]
    cache = Path(os.environ.get("BOTLAB_WHEEL_CACHE", "~/.cache/botlab/target-wheels")).expanduser()
    destination = cache / signature
    if (destination / ".complete").is_file() and list(destination.glob("*.whl")):
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as temp:
        folder = Path(temp)
        downloader = [sys.executable, "-m", "pip"]
        if _run([*downloader, "--version"], check=False).returncode:
            environment = folder / "pip-env"
            _run([sys.executable, "-m", "venv", str(environment)], timeout=60)
            downloader = [str(environment / "bin" / "python"), "-m", "pip"]
        wheels = folder / "wheels"
        wheels.mkdir()
        argv = [*downloader, "download", "--disable-pip-version-check", "--only-binary=:all:",
                "--python-version", version, "--implementation", "cp", "--abi", "cp" + version.replace(".", ""),
                "--dest", str(wheels)]
        for item in platforms:
            argv += ["--platform", item]
        _run([*argv, *_REQUIREMENTS], timeout=600)
        (wheels / ".complete").write_text(json.dumps({"requirements": _REQUIREMENTS, "python": version}))
        try:
            wheels.rename(destination)
        except FileExistsError:
            pass  # Another device job downloaded the same environment.
    return destination


def build_agent_bundle(folder: Path, wheels: Path) -> tuple[Path, str]:
    """Package pure Python from this SDK, not the host's installed binary modules."""
    package = Path(__file__).resolve().parents[1]
    payloads = {}
    for path in sorted(package.rglob("*.py")):
        if path.is_symlink() or "__pycache__" in path.parts:
            continue
        payloads["package/botlab/" + path.relative_to(package).as_posix()] = path.read_bytes()
    for path in sorted(wheels.glob("*.whl")):
        if path.is_symlink():
            raise HilError("Dependency wheels must not be symlinks")
        payloads["wheels/" + path.name] = path.read_bytes()
    if not any(name.startswith("wheels/") for name in payloads):
        raise HilError("No Linux dependency wheels available")
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    bundle = folder / "agent.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, content in {**payloads, "manifest.json": encoded}.items():
            meta = tarfile.TarInfo(name)
            meta.size, meta.mode, meta.mtime = len(content), 0o600, 0
            archive.addfile(meta, io.BytesIO(content))
    return bundle, digest


_INSPECT = r'''
import json, os, platform, shutil, subprocess, sys
from pathlib import Path
root = Path(__ROOT_JSON__).expanduser()
parent = root
while not parent.exists():
    parent = parent.parent
network = []
try:
    network = json.loads(subprocess.check_output(['ip', '-j', 'address'], timeout=5))
except Exception:
    pass
identity = ''
try:
    identity = Path('/etc/machine-id').read_text().strip()
except OSError:
    pass
os_release = {}
try:
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os_release[key] = value.strip('"')
except OSError:
    pass
network_manager = None
try:
    if shutil.which('nmcli') and subprocess.check_output(['nmcli', '-t', '-f', 'RUNNING', 'general'], timeout=5).decode().strip() == 'running':
        network_manager = 'NetworkManager'
except Exception:
    pass
systemd_user_available = False
try:
    if shutil.which('systemctl'):
        systemd_user_available = subprocess.run(['systemctl', '--user', 'show-environment'],
            capture_output=True, timeout=5).returncode == 0
except (OSError, subprocess.TimeoutExpired):
    pass
network_interfaces = [{'name': item.get('ifname'), 'active': item.get('operstate') == 'UP',
    'addresses': [{'address': a['local'], 'prefix': a['prefixlen']} for a in item.get('addr_info', []) if a.get('family') == 'inet']}
    for item in network if item.get('ifname') != 'lo']
memory = 0
try:
    memory = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))) * 1024
except (OSError, StopIteration, ValueError):
    pass
print(json.dumps({'system': platform.system(), 'machine': platform.machine(), 'hostname': platform.node(),
    'python': list(sys.version_info[:3]), 'python_executable': sys.executable, 'libc': platform.libc_ver(),
    'os': os_release, 'id': identity, 'memory_bytes': memory, 'free_bytes': shutil.disk_usage(parent).free,
    'root': str(root), 'network': network, 'network_interfaces': network_interfaces,
    'network_manager': network_manager, 'capabilities': {'provision': True,
    'network_config': 'networkmanager' if network_manager else None}, 'agent_installed': (root/'agent'/'current').exists(),
    'service_manager': 'systemd' if shutil.which('systemctl') else 'process',
    'systemd_user_available': systemd_user_available}))
'''

_INSTALL = r'''
import hashlib, io, json, os, shutil, sys, tarfile, tempfile, zipfile
from pathlib import Path, PurePosixPath
params = json.loads(__PARAMS_JSON__)
root = Path(params['root']).expanduser()
if root.is_symlink() or any(p.is_symlink() for p in root.parents):
    raise ValueError('BotLab root must not be a symlink')
root.mkdir(parents=True, exist_ok=True)
agent = root/'agent'
agent.mkdir(exist_ok=True)
releases = agent/'releases'
releases.mkdir(exist_ok=True)
release = releases/params['release']
if any(p.is_symlink() for p in (agent, releases, release)):
    raise ValueError('Agent paths must not be symlinks')
with tempfile.TemporaryDirectory(prefix='.install-', dir=agent) as temp:
    stage = Path(temp)
    raw = sys.stdin.buffer.read(512*1024*1024+1)
    if len(raw) > 512*1024*1024:
        raise ValueError('Agent bundle exceeds 512 MiB')
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        names = set()
        for item in archive:
            path = PurePosixPath(item.name)
            if not item.isfile() or path.is_absolute() or '..' in path.parts or item.name in names:
                raise ValueError('Invalid Agent bundle member')
            names.add(item.name)
            total += item.size
            if total > 768*1024*1024:
                raise ValueError('Unpacked Agent bundle exceeds limit')
            destination = stage/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(item).read())
    encoded = (stage/'manifest.json').read_bytes()
    if hashlib.sha256(encoded).hexdigest() != params['release']:
        raise ValueError('Agent manifest checksum mismatch')
    manifest = json.loads(encoded)
    if set(manifest) != names - {'manifest.json'}:
        raise ValueError('Agent bundle does not match manifest')
    for name, checksum in manifest.items():
        if hashlib.sha256((stage/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Agent file checksum mismatch')
    deps = stage/'deps'
    deps.mkdir()
    unpacked = 0
    for wheel in (stage/'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('Invalid dependency wheel member')
                unpacked += member.file_size
                if unpacked > 1024*1024*1024:
                    raise ValueError('Dependency wheels exceed 1 GiB unpacked limit')
                if member.is_dir():
                    continue
                parts = list(path.parts)
                if parts[0].endswith('.data'):
                    if len(parts) < 3 or parts[1] not in ('purelib', 'platlib'):
                        continue
                    parts = parts[2:]
                target = deps.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    shutil.rmtree(stage/'wheels')
    if not release.exists():
        shutil.copytree(stage, release)
for name in ('projects', 'staging', 'environments'):
    destination = root/name
    if destination.is_symlink():
        raise ValueError('BotLab data paths must not be symlinks')
    destination.mkdir(exist_ok=True)
print(json.dumps({'root': str(root), 'release': params['release'], 'installed': True}))
'''

_START = r'''
import hashlib, json, os, shutil, signal, socket, subprocess, sys, time
from pathlib import Path
params = json.loads(__PARAMS_JSON__)
root = Path(params['root']).expanduser()
agent = root/'agent'
release = agent/'releases'/params['release']
package, deps = release/'package', release/'deps'
env = dict(os.environ, PYTHONPATH=str(package)+os.pathsep+str(deps), PYTHONNOUSERSITE='1')
# Validate dependencies before touching a healthy previous Agent or current link.
subprocess.run([sys.executable, '-S', '-c', 'import numpy, onnxruntime; import botlab.hil.agent'],
               env=env, check=True, stdout=subprocess.DEVNULL)
unit_name = 'botlab-agent-'+hashlib.sha256(str(root).encode()).hexdigest()[:12]+'.service'
systemd = False
try:
    if shutil.which('systemctl'):
        systemd = subprocess.run(['systemctl', '--user', 'show-environment'],
            capture_output=True, timeout=5).returncode == 0
except (OSError, subprocess.TimeoutExpired):
    pass
if systemd:
    subprocess.run(['systemctl', '--user', 'stop', unit_name], capture_output=True)
pid_file = agent/'agent.pid'
if pid_file.exists():
    try:
        old = int(pid_file.read_text())
        command = Path('/proc')/str(old)/'cmdline'
        data = command.read_bytes()
        if b'botlab.hil.agent' in data and str(root).encode() in data:
            os.kill(old, signal.SIGTERM)
            for _ in range(50):
                if not command.exists():
                    break
                time.sleep(.1)
            if command.exists():
                raise RuntimeError('Previous Agent did not stop')
    except (ValueError, FileNotFoundError, ProcessLookupError):
        pass
try:
    with socket.create_connection(('127.0.0.1', params['port']), timeout=.3):
        raise RuntimeError('Agent port is occupied by an unmanaged process')
except (ConnectionRefusedError, TimeoutError, socket.timeout):
    pass
current = agent/'current'
temporary = agent/'.next'
temporary.unlink(missing_ok=True)
temporary.symlink_to(release)
os.replace(temporary, current)
launcher = agent/'start-agent.py'
args = [sys.executable, '-S', '-m', 'botlab.hil.agent', 'serve', '--host', '127.0.0.1',
        '--port', str(params['port']), '--root', str(root), '--backend', params['backend']]
# A small launcher also allows the process fallback to be restarted after a reboot.
launcher.write_text('import os\nfrom pathlib import Path\nroot = Path('+repr(str(root))+')\n'
    +'env = dict(os.environ, PYTHONPATH=str(root/"agent/current/package")+os.pathsep+str(root/"agent/current/deps"), PYTHONNOUSERSITE="1")\n'
    +'log = open(root/"agent/agent.log", "ab", buffering=0)\nos.dup2(log.fileno(), 1)\nos.dup2(log.fileno(), 2)\n'
    +'os.chdir(root)\nos.execve('+repr(sys.executable)+', '+repr(args)+', env)\n')
launcher.chmod(0o700)
service = 'managed-process'
service_error = None
process = None
if systemd:
    unit = Path.home()/'.config/systemd/user'/unit_name
    unit.parent.mkdir(parents=True, exist_ok=True)
    quote = lambda value: json.dumps(str(value).replace('%', '%%'))
    unit.write_text('[Unit]\nDescription=BotLab target management Agent\nAfter=network.target\n'
        +'[Service]\nType=simple\nExecStart='+quote(sys.executable)+' '+quote(launcher)+'\n'
        +'Restart=on-failure\nRestartSec=3\nKillMode=control-group\n[Install]\nWantedBy=default.target\n')
    try:
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, capture_output=True, timeout=10)
        subprocess.run(['systemctl', '--user', 'enable', '--now', unit_name], check=True, capture_output=True, timeout=15)
        service = 'systemd-user'
    except Exception:
        subprocess.run(['systemctl', '--user', 'stop', unit_name], capture_output=True)
        service_error = 'User service unavailable; using managed process'
if service == 'managed-process':
    log = open(agent/'agent.log', 'ab', buffering=0)
    process = subprocess.Popen([sys.executable, str(launcher)], cwd=root, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid = process.pid
else:
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', '-p', 'MainPID', '--value', unit_name]).decode().strip())
pid_file.write_text(str(pid))
for _ in range(100):
    if process and process.poll() is not None:
        raise RuntimeError('Agent failed to start; inspect '+str(agent/'agent.log'))
    try:
        with socket.create_connection(('127.0.0.1', params['port']), timeout=.2):
            break
    except OSError:
        time.sleep(.1)
else:
    if process:
        process.terminate()
    elif service == 'systemd-user':
        subprocess.run(['systemctl', '--user', 'stop', unit_name], capture_output=True)
    raise RuntimeError('Agent start timed out; inspect '+str(agent/'agent.log'))
linger = False
if service == 'systemd-user':
    try:
        linger = subprocess.check_output(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'], timeout=5).decode().strip() == 'yes'
    except Exception:
        pass
print(json.dumps({'running': True, 'pid': pid, 'port': params['port'], 'service': service,
    'service_error': service_error, 'autostart': service == 'systemd-user' and linger,
    'starts_on_login': service == 'systemd-user', 'log': str(agent/'agent.log')}))
'''


_CONNECT = r'''
import fcntl, hashlib, json, os, socket, struct, subprocess, sys, time
from pathlib import Path
params = json.loads(__PARAMS_JSON__)
root = Path(params['root']).expanduser()
agent = root/'agent'
launcher = agent/'start-agent.py'
if not launcher.is_file() or launcher.is_symlink():
    raise RuntimeError('BotLab Agent is not installed; choose Install and connect first')
if root.is_symlink() or agent.is_symlink():
    raise RuntimeError('BotLab Agent installation paths must not be symlinks')

def hello():
    try:
        connection = socket.create_connection(('127.0.0.1', params['port']), timeout=.5)
    except OSError:
        return False
    with connection:
        connection.settimeout(1)
        payload = json.dumps({'version': 1, 'op': 'hello'}).encode()
        connection.sendall(struct.pack('!I', len(payload))+payload)
        def read(size):
            data = bytearray()
            while len(data) < size:
                part = connection.recv(size-len(data))
                if not part:
                    raise RuntimeError('Agent closed its readiness connection')
                data.extend(part)
            return data
        try:
            size = struct.unpack('!I', read(4))[0]
            if not 0 < size <= 1024*1024:
                raise RuntimeError('Target port is occupied by an incompatible service')
            result = json.loads(read(size))
            if result.get('version') != 1 or result.get('ok') is not True or not isinstance(result.get('device'), dict):
                raise RuntimeError('Target port is occupied by an incompatible service')
            return True
        except (socket.timeout, OSError) as exc:
            raise RuntimeError('Target Agent is accepting connections but did not answer hello') from exc

# Serialize fallback startup across concurrent SDK hosts. A configured systemd
# unit is started idempotently; an already-ready Agent is never restarted.
lock_path = agent/'connect.lock'
if lock_path.is_symlink():
    raise RuntimeError('Unsafe Agent connection lock')
lock = open(lock_path, 'a')
os.chmod(lock_path, 0o600)
deadline = time.monotonic()+30
while True:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() >= deadline:
            raise RuntimeError('Another connection is still starting the target Agent')
        time.sleep(.1)
started = None
try:
    if not hello():
        unit_name = 'botlab-agent-'+hashlib.sha256(str(root).encode()).hexdigest()[:12]+'.service'
        unit = Path.home()/'.config/systemd/user'/unit_name
        if unit.is_file() and Path('/usr/bin/systemctl').exists():
            # PAM has created the user manager for this exec session. Starting
            # an enabled/already-starting unit is safe and never restarts it.
            result = subprocess.run(['systemctl', '--user', 'start', unit_name], capture_output=True, timeout=20)
            if result.returncode:
                raise RuntimeError('Could not start installed Agent user service; inspect Agent log or reinstall')
        else:
            log = open(agent/'agent.log', 'ab', buffering=0)
            started = subprocess.Popen([sys.executable, str(launcher)], cwd=root,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            (agent/'agent.pid').write_text(str(started.pid))
        while time.monotonic() < deadline:
            if started is not None and started.poll() is not None:
                raise RuntimeError('Installed Agent failed to start; inspect Agent log or reinstall')
            if hello():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Installed Agent did not become ready; inspect Agent log or reinstall')
except BaseException:
    # Only clean up a process started by this failed connection; never kill an
    # existing Agent or alter the state of a configured systemd service.
    if started is not None and started.poll() is None:
        started.terminate()
        try:
            started.wait(timeout=3)
        except subprocess.TimeoutExpired:
            started.kill()
            started.wait(timeout=3)
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
# Signal helper readiness before the Host validates the forwarded Agent hello.
print("BOTLAB_TARGET_READY", flush=True)
# EOF arrives when the SDK closes the tunnel, crashes, or loses SSH transport.
# Keep the login alive while the data forwarding channel is in use.
while sys.stdin.buffer.read(1):
    pass
'''


def local_interfaces() -> list[dict]:
    """Read adapters; this function never changes host network settings."""
    if platform.system() != "Darwin":
        result = _run(["ip", "-j", "address"], check=False)
        if result.returncode:
            return []
        return [{"name": item["ifname"], "label": item["ifname"],
                 "addresses": [{"address": a["local"], "prefix": a["prefixlen"]}
                               for a in item.get("addr_info", []) if a.get("family") == "inet"],
                 "active": item.get("operstate") == "UP"}
                for item in json.loads(result.stdout) if item["ifname"] != "lo"]
    labels = {}
    ports = _run(["networksetup", "-listallhardwareports"], check=False).stdout.decode()
    for block in ports.split("\n\n"):
        label = re.search(r"Hardware Port: (.+)", block)
        device = re.search(r"Device: (.+)", block)
        if label and device:
            labels[device[1]] = label[1]
    output = _run(["ifconfig", "-a"]).stdout.decode()
    results = []
    for block in re.split(r"(?=^[A-Za-z][\w.-]*: flags=)", output, flags=re.M):
        name = re.match(r"([\w.-]+): flags=", block)
        if not name or name[1] == "lo0":
            continue
        addresses = []
        for address, mask in re.findall(r"\binet ([\d.]+) netmask (0x[0-9a-f]+|[\d.]+)", block):
            mask = str(ipaddress.IPv4Address(int(mask, 16))) if mask.startswith("0x") else mask
            addresses.append({"address": address, "prefix": ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen})
        results.append({"name": name[1], "label": labels.get(name[1], name[1]),
                        "addresses": addresses, "active": "status: active" in block})
    return results


def discover_devices(interface: str | None = None, profile: str | None = None,
                     host: str | None = None) -> dict:
    """Probe only neighbors on the selected interface plus an explicit hostname.

    Board profiles contribute documented factory-address hints, never credentials.
    This is discovery, not an unrestricted subnet/port scanner.
    """
    interfaces = local_interfaces()
    if interface and interface not in {item["name"] for item in interfaces}:
        raise HilError("Selected network interface no longer exists")
    selected = [item for item in interfaces if not interface or item["name"] == interface]
    networks = [ipaddress.ip_network(f"{a['address']}/{a['prefix']}", strict=False)
                for item in selected for a in item["addresses"]]
    candidates = set()
    if host:
        _ssh_config({"ssh": {"host": host, "user": "probe"}})
        candidates.add(host)
    if interface:
        neighbors = _run(["arp", "-an"], check=False).stdout.decode()
        for line in neighbors.splitlines():
            match = re.search(r"\(([\d.]+)\).*\bon\s+(\S+)", line)
            if match and match[2] == interface and "incomplete" not in line:
                candidates.add(match[1])
        if profile == "rdk-x5":
            for hint in ("192.168.127.10", "192.168.128.10"):
                if any(ipaddress.ip_address(hint) in network for network in networks):
                    candidates.add(hint)
    local = {a["address"] for item in interfaces for a in item["addresses"]}
    def probe(address):
        try:
            with socket.create_connection((address, 22), timeout=.6) as stream:
                stream.settimeout(.6)
                banner = stream.recv(256).decode("ascii", "replace").strip()
                if banner.startswith("SSH-"):
                    return {"host": address, "port": 22, "transport": "ssh", "banner": banner[:120]}
        except OSError:
            pass
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        found = [item for item in pool.map(probe, sorted(candidates - local)[:64]) if item]
    return {"interfaces": interfaces, "candidates": found,
            "profile": profile, "requires_interface": not bool(interface or host)}


def _network_values(method, address=None, netmask=None, gateway=None):
    if method not in ("dhcp", "static"):
        raise HilError("Network method must be dhcp or static")
    if method == "dhcp":
        return None, None, None
    try:
        if not isinstance(address, str):
            raise ValueError
        value = ipaddress.IPv4Interface(address if "/" in address else f"{address}/{netmask or '24'}")
        route = str(ipaddress.IPv4Address(gateway)) if gateway else ""
        if value.ip.is_loopback or value.ip.is_multicast or value.ip.is_unspecified:
            raise ValueError
        if route and ipaddress.IPv4Address(route) not in value.network:
            raise ValueError
        if value.network.prefixlen < 31 and value.ip in (value.network.network_address, value.network.broadcast_address):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise HilError("Provide a usable IPv4 address, subnet mask and optional gateway in the same subnet") from exc
    return str(value), str(value.netmask), route


def configure_local_network(interface: str, method: str, address: str | None = None,
                            netmask: str | None = None, gateway: str | None = None) -> dict:
    """Configure one Mac adapter, using the native administrator authorization UI.

    Only selected network service IPv4 settings are touched. The previous state
    is returned so the caller can offer Restore. No administrator password is
    collected by BotLab.
    """
    if platform.system() != "Darwin":
        raise HilError("Automatic host network configuration currently supports macOS")
    interfaces = local_interfaces()
    if interface not in {item["name"] for item in interfaces}:
        raise HilError("Selected network interface no longer exists")
    value, mask, route = _network_values(method, address, netmask, gateway)
    listing = _run(["networksetup", "-listnetworkserviceorder"]).stdout.decode()
    matches = re.findall(r"^\(\d+\) (.+)\n\(Hardware Port: .*?, Device: ([^)]+)\)", listing, re.M)
    services = [name for name, device in matches if device == interface and not name.startswith("*")]
    if len(services) != 1:
        raise HilError("Select an adapter with exactly one enabled macOS network service")
    service = services[0]
    previous = _run(["networksetup", "-getinfo", service]).stdout.decode()
    if method == "dhcp":
        command = ["/usr/sbin/networksetup", "-setdhcp", service]
    else:
        command = ["/usr/sbin/networksetup", "-setmanual", service, value.split("/")[0], mask, route]
    # JSON strings and AppleScript strings share the needed quote/backslash escapes.
    script = "do shell script " + json.dumps(shlex.join(command)) + " with administrator privileges"
    _run(["osascript", "-e", script], timeout=120)
    current = _run(["networksetup", "-getinfo", service]).stdout.decode()
    return {"interface": interface, "service": service, "method": method,
            "previous": previous, "current": current, "interfaces": local_interfaces()}


_NETWORK_BEGIN = r'''
import json, os, re, shutil, subprocess, sys, time, uuid
from pathlib import Path
params = json.loads(__PARAMS_JSON__)
if os.geteuid() != 0:
    raise RuntimeError('Network configuration requires passwordless sudo or a root SSH account')
if not shutil.which('nmcli') or not shutil.which('busctl'):
    raise RuntimeError('This network adapter requires NetworkManager; keep existing configuration or install a platform plugin')
def call(*args):
    return subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=15).decode().strip()
base = ['busctl', '--system', 'call', 'org.freedesktop.NetworkManager', '/org/freedesktop/NetworkManager', 'org.freedesktop.NetworkManager']
result = call(*base, 'GetDeviceByIpIface', 's', params['interface'])
match = re.fullmatch(r'o "(/org/freedesktop/NetworkManager/Devices/[0-9]+)"', result)
if not match:
    raise RuntimeError('Interface is not managed by NetworkManager')
connection = call('nmcli', '-g', 'GENERAL.CON-UUID', 'device', 'show', params['interface'])
if not re.fullmatch('[a-fA-F0-9-]{36}', connection):
    raise RuntimeError('Interface has no active NetworkManager connection')
result = call(*base, 'CheckpointCreate', 'aouu', '1', match[1], '90', '0')
checkpoint = re.fullmatch(r'o "(/org/freedesktop/NetworkManager/Checkpoint/[0-9]+)"', result)
if not checkpoint:
    raise RuntimeError('Could not create network rollback checkpoint')
transaction = uuid.uuid4().hex
folder = Path('/run/botlab-network')
if folder.is_symlink():
    raise RuntimeError('Unsafe network transaction directory')
folder.mkdir(mode=0o700, exist_ok=True)
state = dict(params, transaction=transaction, connection=connection, checkpoint=checkpoint[1], expires=time.time()+90, applied=False)
path = folder/(transaction+'.json')
path.write_text(json.dumps(state))
path.chmod(0o600)
source = __APPLY_SOURCE__
log = open(folder/(transaction+'.log'), 'ab', buffering=0)
subprocess.Popen([sys.executable, '-c', source, str(path)], stdin=subprocess.DEVNULL,
                 stdout=log, stderr=log, start_new_session=True)
print(json.dumps({'transaction': transaction, 'rollback_seconds': 90,
                  'new_host': params.get('address', '').split('/')[0] or None,
                  'requires_confirmation': True, 'interface': params['interface']}))
'''

_NETWORK_APPLY = r'''
import json, subprocess, sys, time
from pathlib import Path
path = Path(sys.argv[1])
state = json.loads(path.read_text())
time.sleep(2)
args = ['nmcli', '--wait', '15', 'device', 'modify', state['interface']]
if state['method'] == 'dhcp':
    args += ['ipv4.method', 'auto', 'ipv4.addresses', '', 'ipv4.gateway', '']
else:
    args += ['ipv4.method', 'manual', 'ipv4.addresses', state['address'], 'ipv4.gateway', state['gateway']]
try:
    subprocess.run(args, timeout=20, check=True)
    state['applied'] = True
except Exception as exc:
    state['error'] = str(exc)
path.write_text(json.dumps(state))
'''

_NETWORK_CONFIRM = r'''
import json, os, subprocess, time
from pathlib import Path
transaction = __TRANSACTION__
if os.geteuid() != 0:
    raise RuntimeError('Network configuration requires administrator privileges')
path = Path('/run/botlab-network')/(transaction+'.json')
state = json.loads(path.read_text())
if time.time() >= state['expires'] or not state['applied']:
    raise RuntimeError(state.get('error') or 'Network transaction expired or is not ready; previous settings will be restored')
base = ['busctl', '--system', 'call', 'org.freedesktop.NetworkManager', '/org/freedesktop/NetworkManager', 'org.freedesktop.NetworkManager']
# Atomically verify the checkpoint still exists and reserve time to commit.
# The daemon retains rollback responsibility even if this SSH connection fails.
subprocess.run(base+['CheckpointAdjustRollbackTimeout', 'ou', state['checkpoint'], '60'],
               check=True, timeout=10, stdout=subprocess.DEVNULL)
args = ['nmcli', '--wait', '10', 'connection', 'modify', 'uuid', state['connection']]
if state['method'] == 'dhcp':
    args += ['ipv4.method', 'auto', 'ipv4.addresses', '', 'ipv4.gateway', '']
else:
    args += ['ipv4.method', 'manual', 'ipv4.addresses', state['address'], 'ipv4.gateway', state['gateway']]
subprocess.run(args, check=True, timeout=15, stdout=subprocess.DEVNULL)
subprocess.run(base+['CheckpointDestroy', 'o', state['checkpoint']], check=True, timeout=10, stdout=subprocess.DEVNULL)
path.unlink()
print(json.dumps({'confirmed': True, 'interface': state['interface'], 'method': state['method']}))
'''
