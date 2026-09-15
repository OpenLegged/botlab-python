"""The ``botlab run`` debug loop: a release built straight from source.

The browser runtime always fetches ``./release.zip``, so "run from source"
means the debug server answers that request with a package it builds in
memory: zip entries STORED (no deflate pass), no file ever written to the
project, and a rebuild in a fresh subprocess whenever any project file
changed since the last fetch — reload the browser page after an edit and the
new code runs, without restarting the CLI. Publishable packages stay the
domain of ``botlab build`` (DEFLATED, written to disk).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ._host import BotLabError
from ._project import LoadedProject, load_project
from ._release import create_release_package

# Directories that never describe project source state (caches, VCS, UI
# snapshot data of other tools).
_IGNORED_DIRECTORIES = {"__pycache__", "node_modules"}
# Build outputs: writing one must not trigger a debug rebuild.
_IGNORED_FILES = {"release.zip", ".botlab.zip"}

Signature = tuple[tuple[str, int, int], ...]


def build_debug_release(project: LoadedProject) -> bytes:
    """Package a loaded project for the debug server (STORED, in memory)."""
    return create_release_package(
        project,
        metadata={"name": project.manifest["project"]["name"]},
        compression="stored",
    )


def _project_signature(project_dir: Path) -> Signature:
    """Stat every source file under the project directory.

    A file's (relative path, mtime_ns, size) triple identifies the inputs of
    a release build: the two-file contract, preset YAML, and every packaged
    asset. Hidden directories, caches, and build outputs are pruned; symlinked
    directories are skipped so a cycle cannot hang the server.
    """
    entries: list[tuple[str, int, int]] = []
    pending = [project_dir]
    while pending:
        for child in pending.pop().iterdir():
            name = child.name
            if child.is_dir():
                if child.is_symlink() or name.startswith(".") or name in _IGNORED_DIRECTORIES:
                    continue
                pending.append(child)
                continue
            if name in _IGNORED_FILES:
                continue
            try:
                stat = child.stat()
            except OSError:
                continue  # broken symlink or a file deleted mid-walk
            entries.append(
                (child.relative_to(project_dir).as_posix(), stat.st_mtime_ns, stat.st_size)
            )
    entries.sort()
    return tuple(entries)


class DebugReleaseProvider:
    """Serve ``/release.zip`` bytes, rebuilding from source when files change.

    The provider holds the current package bytes in memory and compares a
    stat signature of the project directory on every request. When source
    changed, it rebuilds in a subprocess (``python -m botlab._debug``) so the
    fresh import sees the edits, then swaps the bytes in. A failed rebuild
    keeps the server up and raises, so the error surfaces in both the CLI
    terminal and the browser.
    """

    def __init__(self, project_dir: Path, initial_bytes: bytes, initial_signature: Signature):
        self._project_dir = project_dir.resolve()
        self._bytes = initial_bytes
        self._signature = initial_signature
        self._lock = threading.Lock()

    def __call__(self) -> bytes:
        signature = _project_signature(self._project_dir)
        if signature != self._signature:
            with self._lock:
                if signature != self._signature:
                    self._rebuild(signature)
        return self._bytes

    def _rebuild(self, signature: Signature) -> None:
        started = time.monotonic()
        # stderr is inherited so the child's plan summary and any traceback
        # stream live into the `botlab run` terminal; only the zip bytes come
        # back through the pipe.
        result = subprocess.run(
            [sys.executable, "-m", "botlab._debug", str(self._project_dir)],
            stdout=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            print("[botlab] debug rebuild failed; fix the project and reload the page", file=sys.stderr)
            raise BotLabError("debug rebuild failed; see the botlab terminal output")
        self._bytes = result.stdout
        self._signature = signature
        print(
            f"[botlab] source changed — rebuilt release in {time.monotonic() - started:.1f}s",
            file=sys.stderr,
        )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m botlab._debug <project_dir>", file=sys.stderr)
        return 2
    project_dir = Path(argv[0]).resolve()

    # stdout of this process is a data channel: the zip bytes and nothing
    # else. load_project() logs through the host bridge, and user controller
    # code may print() freely — redirect fd 1 onto fd 2 so every stdout
    # write, at any layer (text buffer, binary buffer, fd), lands on stderr,
    # and deliver the package on a saved copy of the real pipe.
    package_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        project = load_project(project_dir)
        package = build_debug_release(project)
        if os.environ.get("BOTLAB_VERBOSE"):
            if project.compiled_plan is not None:
                print(
                    f"[botlab] plan: {len(project.compiled_plan['nodes'])} nodes, "
                    f"{len(project.compiled_plan['edges'])} edges (compiled, no Python at runtime)",
                    file=sys.stderr,
                )
            else:
                print(
                    "[botlab] plan: runs via in-browser Pyodide "
                    "(project uses Signal.action(modify=...))",
                    file=sys.stderr,
                )
        _write_all(package_fd, package)
        return 0
    finally:
        os.close(package_fd)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
