"""The native ``botlab run`` flow: local execution plus a browser viewer.

The simulation runs in a CPython subprocess — native MuJoCo physics and
ONNX Runtime inference through the package-owned ``_runtime`` runner. This
process compiles the project, serves the UI snapshot so the browser can open
the read-only ``?ws=`` viewer, and forwards the runner's output.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

from ._debug import DebugReleaseProvider, _project_signature, build_debug_release
from ._errors import BotLabError
from ._server import start_project_server, ui_snapshot_dir

_DISPLAY_READY_MARKER = "display ready: "
_DISPLAY_STARTUP_TIMEOUT_SECONDS = 30.0


def require_native_runtime() -> None:
    """Fail up front when the runtime extras are missing — no silent fallback."""
    from ._runtime import _deps

    probe = _deps.probe_runtime_deps()
    if probe.missing:
        raise BotLabError(
            f"{_deps.describe_missing(probe.missing)} "
            "(or drop --native to run the project in the browser editor instead)"
        )


def _free_loopback_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_native(
    project,
    project_dir: Path,
    *,
    host: str,
    port: int,
    open_browser: bool,
    report_path: str | None = None,
) -> int:
    """Serve the viewer surface and drive the native runner until it exits."""
    package = build_debug_release(project)
    provider = DebugReleaseProvider(project_dir, package, _project_signature(project_dir))
    httpd, base_url = start_project_server(
        project_dir, None, host, port, release_provider=provider, ui_dir=ui_snapshot_dir()
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        return _drive_runner(project, project_dir, base_url, open_browser=open_browser,
                             report_path=report_path)
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join(timeout=5.0)


def _drive_runner(project, project_dir: Path, base_url: str, *, open_browser: bool,
                  report_path: str | None = None) -> int:
    display_port = _free_loopback_port()
    env = {**os.environ, "BOTLAB_DISPLAY": "1", "BOTLAB_DISPLAY_PORT": str(display_port)}
    if report_path is not None:
        env["BOTLAB_REPORT"] = report_path
    runner = subprocess.Popen(
        [sys.executable, "-m", "botlab._runtime.run", str(project_dir)],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    display_url: str | None = None

    def forward_stderr():
        nonlocal display_url
        stream = runner.stderr
        assert stream is not None
        for line in stream:
            sys.stderr.write(line)
            sys.stderr.flush()
            if _DISPLAY_READY_MARKER in line and display_url is None:
                display_url = line.split(_DISPLAY_READY_MARKER, 1)[1].strip()

    pump = threading.Thread(target=forward_stderr, daemon=True)
    pump.start()
    try:
        deadline = time.monotonic() + _DISPLAY_STARTUP_TIMEOUT_SECONDS
        while (
            display_url is None
            and runner.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if display_url is None:
            exit_code = runner.poll()
            if exit_code is not None:
                return exit_code  # the runner failed on its own; output is forwarded
            raise BotLabError(
                "the native runner did not open its display stream within "
                f"{_DISPLAY_STARTUP_TIMEOUT_SECONDS:.0f}s"
            )

        viewer_url = f"{base_url}/embed.html?ws={quote(display_url, safe='')}"
        # flush: stdout is block-buffered when redirected — the links must be
        # visible immediately, not at process exit.
        print(
            f"[botlab] {project.manifest['project']['name']} is running natively; "
            "the browser is a viewer",
            flush=True,
        )
        print(f"  ➜  Viewer: {viewer_url}", flush=True)
        print(f"  ➜  Stream: {display_url}", flush=True)
        if open_browser:
            webbrowser.open(viewer_url)

        try:
            return runner.wait()
        except KeyboardInterrupt:
            print("\n[botlab] stopping the native run", file=sys.stderr)
            runner.send_signal(signal.SIGINT)
            try:
                return runner.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                return 130
    finally:
        if runner.poll() is None:
            runner.kill()
        pump.join(timeout=1.0)
