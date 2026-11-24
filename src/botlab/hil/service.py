"""systemd user-service installation for resident HIL processes.

``botlab-agent install`` and ``botlab-hil-gateway install`` write one user
unit plus a 0600 token EnvironmentFile, then enable it with systemctl. A user
unit starts at login; boot without a login additionally needs lingering.
Lingering is only ever enabled through the explicit ``--enable-linger`` flag
and its result is reported honestly, never implied. Without a systemd user
manager the install fails loudly — a plain process cannot survive a reboot,
so there is no process fallback here (unlike SSH provisioning, whose managed
process only has to outlive one connection).

Tokens reach the unit through the EnvironmentFile, never through the command
line: process listings and shell history must stay credential-free, the same
stance as ``BOTLAB_HIL_TOKEN`` handling elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from .protocol import HilError

UNIT_TARGET = Path.home() / ".config/systemd/user"


def unit_hash(key: str) -> str:
    """Same 12-hex identity the SSH provisioning uses for a storage root."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def check_env_token(name: str, value: str) -> str:
    """One line, printable ASCII, no whitespace/quotes/backslash: values
    systemd's EnvironmentFile parser would mangle (C-style escapes, quote
    stripping) or that would break "Bearer <token>" semantics are rejected."""
    for char in value or "":
        if ord(char) < 0x21 or ord(char) > 0x7e or char in "\"'\\":
            raise HilError(f"{name} must be non-empty printable ASCII without "
                           "spaces, quotes or backslashes")
    if not value:
        raise HilError(f"{name} must not be empty")
    return value


def user_manager_available() -> bool:
    if not shutil.which("systemctl"):
        return False
    try:
        return subprocess.run(["systemctl", "--user", "show-environment"],
                              capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def linger_enabled() -> bool:
    try:
        return subprocess.check_output(
            ["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"],
            timeout=5, text=True).strip() == "yes"
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def try_enable_linger() -> str | None:
    """Enable lingering for the current user; ``None`` on success."""
    try:
        result = subprocess.run(["loginctl", "enable-linger"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0 and linger_enabled():
        return None
    return result.stderr.strip() or "loginctl enable-linger failed"


def unit_body(description: str, exec_argv: list[str],
              env_file: Path | None = None, *, network_online: bool = False) -> str:
    def quote(value):
        # ensure_ascii=False: systemd units are UTF-8 and its unquoting has no
        # \uXXXX escapes, so non-ASCII paths must stay literal.
        return json.dumps(str(value).replace("%", "%%"), ensure_ascii=False)

    lines = ["[Unit]", f"Description={description}"]
    if network_online:
        lines += ["After=network-online.target", "Wants=network-online.target"]
    lines += ["[Service]", "Type=simple"]
    if env_file is not None:
        lines.append(f"EnvironmentFile={quote(env_file)}")
    lines.append("ExecStart=" + " ".join(quote(part) for part in exec_argv))
    lines += ["Restart=on-failure", "RestartSec=3", "KillMode=control-group"]
    lines += ["", "[Install]", "WantedBy=default.target", ""]
    return "\n".join(lines)


def install_service(name: str, description: str, exec_argv: list[str],
                    env_file: Path, env: dict[str, str], *, network_online: bool = False,
                    enable_linger: bool = False) -> dict:
    """Write env file + unit, then enable --now. Returns a summary.

    On a systemctl failure the written files stay in place (uninstall works)
    and the error is raised; there is no partial-success fallback mode.
    """
    if not user_manager_available():
        raise HilError("No systemd user manager available; boot autostart needs one. "
                       "Run the process manually or set up a service with your "
                       "system's process supervisor instead.")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("".join(f"{key}={value}\n" for key, value in env.items()))
    env_file.chmod(0o600)
    unit_path = UNIT_TARGET / name
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_body(description, exec_argv, env_file,
                                   network_online=network_online))
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                       check=True, capture_output=True, timeout=15)
        subprocess.run(["systemctl", "--user", "enable", "--now", name],
                       check=True, capture_output=True, timeout=20)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise HilError(f"systemctl failed to enable {name}: {exc}") from exc
    linger = linger_enabled()
    linger_error = None
    if enable_linger and not linger:
        linger_error = try_enable_linger()
        linger = linger_enabled()
    return {"unit": name, "unit_file": unit_path, "env_file": env_file,
            "starts_on_login": True, "linger": linger,
            "autostart": linger, "linger_error": linger_error}


def uninstall_service(name: str, env_file: Path | None = None) -> dict:
    unit_path = UNIT_TARGET / name
    if not unit_path.exists():
        raise HilError(f"{name} is not installed for this user")
    try:
        subprocess.run(["systemctl", "--user", "disable", "--now", name],
                       check=True, capture_output=True, timeout=20)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise HilError(f"systemctl failed to disable {name}: {exc}") from exc
    unit_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   check=False, capture_output=True, timeout=15)
    if env_file is not None:
        env_file.unlink(missing_ok=True)
    return {"unit": name, "removed": unit_path, "env_file": env_file}
