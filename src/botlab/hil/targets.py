"""User-owned target inventory; it contains connection settings, never credentials."""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from .protocol import HilError, endpoint_address, gateway_endpoint


def targets_path() -> Path:
    override = os.environ.get("BOTLAB_TARGETS_FILE")
    return Path(override).expanduser() if override else Path.home() / ".config/botlab/targets.json"


def read_targets(path: Path | None = None) -> list[dict]:
    path = path or targets_path()
    if not path.exists():
        return []
    try:
        document = json.loads(path.read_text())
        if document.get("version") != 1 or not isinstance(document.get("targets"), list):
            raise ValueError
        result = [validate_target(item) for item in document["targets"]]
        if len(result) > 100 or len({item["id"] for item in result}) != len(result):
            raise ValueError
        return result
    except (ValueError, TypeError, AttributeError) as exc:
        raise HilError(f"Invalid target configuration: {path}") from exc


def validate_target(value: dict) -> dict:
    if not isinstance(value, dict):
        raise HilError("Target must be an object")
    target_id = value.get("id") or uuid.uuid4().hex[:12]
    if not isinstance(target_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", target_id):
        raise HilError("Target ID must use letters, digits, underscores or hyphens")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise HilError("Target name must be 1–100 characters")
    if "password" in value or "credentials" in value:
        raise HilError("Credentials must not be saved in target settings")
    transport = value.get("transport", "tcp")
    if transport not in ("tcp", "ssh", "gateway"):
        raise HilError("Target transport must be tcp, ssh or gateway")
    endpoint = value.get("endpoint")
    extras = {}
    if transport == "gateway":
        # The gateway token never lives here; it arrives via BOTLAB_HIL_TOKEN.
        endpoint = gateway_endpoint(endpoint)
        device = value.get("device")
        if not isinstance(device, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", device):
            raise HilError("Gateway target requires a device ID (letters, digits, _ or -)")
        extras = {"transport": "gateway", "device": device}
    elif transport == "ssh":
        from .devices import _ssh_config
        config = value.get("ssh")
        if not isinstance(config, dict) or set(config) - {"host", "user", "port", "identity_file", "host_key"}:
            raise HilError("Invalid SSH settings; passwords are transient")
        host, user, port, root = _ssh_config(value)
        config = {**config, "host": host, "user": user, "port": port}
        fingerprint = config.get("host_key")
        if fingerprint is not None and (not isinstance(fingerprint, str) or not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint)):
            raise HilError("Invalid SSH host key fingerprint")
        agent_port = value.get("agent_port", 50052)
        if type(agent_port) is not int or not 1 <= agent_port <= 65535:
            raise HilError("Invalid target agent port")
        endpoint = f"tcp://127.0.0.1:{agent_port}"
        profile = value.get("profile", "generic-linux")
        if profile not in ("generic-linux", "rdk-x5"):
            raise HilError("Unsupported device profile")
        extras = {"transport": "ssh", "ssh": config, "root": root,
                  "agent_port": agent_port, "profile": profile}
    else:
        endpoint_address(endpoint)
    backend = value.get("backend") or "onnxruntime"
    if not isinstance(backend, str) or len(backend) > 200:
        raise HilError("Invalid backend name")
    return {"id": target_id, "name": name.strip(), "endpoint": endpoint, "backend": backend, **extras}


def save_target(value: dict, path: Path | None = None) -> dict:
    target = validate_target(value)
    path = path or targets_path()
    items = read_targets(path)
    items = [item for item in items if item["id"] != target["id"]] + [target]
    if len(items) > 100:
        raise HilError("At most 100 targets may be configured")
    _write_targets(items, path)
    return target


def remove_target(target_id: str, path: Path | None = None) -> None:
    path = path or targets_path()
    items = read_targets(path)
    if not any(item["id"] == target_id for item in items):
        raise HilError("Unknown target")
    _write_targets([item for item in items if item["id"] != target_id], path)


def _write_targets(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".targets-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"version": 1, "targets": items}, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
