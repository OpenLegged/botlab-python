"""Private, project-scoped local middleware discovery (never imports a project)."""

from __future__ import annotations

import getpass
import hashlib
import ipaddress
import os
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .protocol import API_VERSION, PROTOCOL, ControlError, identifier, parse_json, project_key


def runtime_dir() -> Path:
    user = str(os.getuid()) if hasattr(os, "getuid") else hashlib.sha256(getpass.getuser().encode()).hexdigest()[:12]
    return Path(os.environ.get("BOTLAB_CONTROL_RUNTIME_DIR", Path(tempfile.gettempdir()) / f"botlab-control-{user}"))


def endpoint(value: str) -> str:
    """Accept loopback HTTP origins (local middleware) and HTTPS origins
    (remote middleware behind TLS); plaintext HTTP never leaves the loopback."""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if not host:
            raise ValueError()
        try:
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if (parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError()
        if parsed.scheme == "http":
            if not loopback or not parsed.port:
                raise ValueError()
        elif parsed.scheme == "https":
            port = parsed.port  # Accessing .port rejects malformed ports.
            if port is not None and not 1 <= port <= 65535:
                raise ValueError()
        else:
            raise ValueError()
    except (ValueError, TypeError) as error:
        raise ControlError("invalid_endpoint",
                           "Endpoint must be a loopback HTTP origin or an HTTPS origin, without "
                           "credentials, path, query or fragment.") from error
    return value.rstrip("/")


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ControlError("unsafe_file", "Control metadata must be a regular file owned by this user.")
    if os.name == "posix" and info.st_mode & 0o077:
        raise ControlError("unsafe_file", "Control metadata must have private file permissions (0600).")


def private_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ControlError("unsafe_directory", "Control metadata directory must be owned by this user and not a symlink.")
    if os.name == "posix" and info.st_mode & 0o077:
        raise ControlError("unsafe_directory", "Control metadata directory must have private permissions (0700).")


def descriptors(project: str | Path) -> list[dict]:
    root = runtime_dir()
    directory = root / project_key(project)
    if not directory.exists():
        return []
    private_directory(root)
    private_directory(directory)
    result = []
    for path in sorted(directory.glob("*.json")):
        try:
            private_file(path)
            item = parse_json(path.read_text())
            if not isinstance(item, dict):
                continue
            if item.get("schema") != 1 or item.get("api_version") != API_VERSION or item.get("protocol") != PROTOCOL:
                continue
            remote = item.get("remote") is True
            if remote:
                # Enrolled remote descriptors pin the hosted identity that
                # /info reported at enrollment; the client verifies the pin.
                if (not isinstance(item.get("project_key"), str) or not item["project_key"]
                        or not isinstance(item.get("project"), str) or not item["project"]):
                    continue
            elif item.get("project_key") != project_key(project) or item.get("project") != str(Path(project).resolve()):
                continue
            identifier(item.get("instance_id"), "instance_id")
            item["endpoint"] = endpoint(item.get("endpoint"))
            if remote and urlsplit(item["endpoint"]).scheme != "https":
                continue
            token = item.get("token")
            if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
                continue
            result.append(item)
        except (OSError, ControlError):
            # Partial, stale and insecure descriptors cannot grant credentials.
            continue
    return result


def public_descriptor(item: dict) -> dict:
    return {key: item.get(key) for key in ("instance_id", "project", "project_key", "endpoint", "pid")}
