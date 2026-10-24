"""The ``botlab`` command line interface.

- ``botlab run [PROJECT_DIR]`` — run the project in the browser: serve the
  complete BotLab editor — the same web UI as the platform, with the project
  tree, editor, Console/Watch/Metrics panels — from live sources and assets
  straight from the directory, so edits land on reload. No zip file is
  written. ``--native`` executes the project locally instead (local MuJoCo +
  ONNX Runtime, needs the ``runtime`` extra) and opens the browser as a
  read-only ``?ws=`` viewer of the running simulation. ``--headless`` runs
  natively without a viewer or HTTP server.
- ``botlab build [PROJECT_DIR]`` — export release.zip locally (no server, no
  upload); the same package the platform's ``/d/<id>/`` demos are served from.
  ``--site <dir>`` assembles a self-contained static site around it instead
  (push the directory to GitHub Pages and the demo runs from any sub-path).
- ``botlab validate`` / ``botlab ui`` / ``botlab docs`` — packaging dry-run,
  UI snapshot tools, and the bundled docs location.

Any standard two-file BotLab project works; nothing here knows about a specific
robot or policy.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from . import __version__
from ._browser_runtime import (
    browser_runtime_cache_dir,
    ensure_browser_runtime,
    export_browser_runtime,
    fetch_browser_runtime,
)
from ._debug import DebugReleaseProvider, _project_signature, build_debug_release
from ._errors import BotLabError
from ._native import require_native_runtime, run_native
from ._project import load_project
from ._release import create_release_package
from ._server import serve_project, ui_snapshot_dir


def _print_project_summary(project) -> None:
    summary = ", ".join(
        f"{resource['kind']}:{resource['id']}" for resource in project.manifest["resources"]
    )
    print(f"[botlab] project {project.manifest['project']['name']!r} — {summary}")
    if project.compiled_plan is not None:
        print(
            f"[botlab] plan: {len(project.compiled_plan['nodes'])} nodes, "
            f"{len(project.compiled_plan['edges'])} edges (compiled, no Python at runtime)"
        )
    else:
        print("[botlab] plan: runs via in-browser Pyodide (project uses Signal.action(modify=...))")


def _set_verbose(args: argparse.Namespace) -> None:
    import os

    if getattr(args, "verbose", False):
        os.environ["BOTLAB_VERBOSE"] = "1"


# (applies, flag, message): `run` rejects `flag` when `applies(args)` holds.
# --out wins over --native/--browser; the editor debug loop is the remainder.
_RUN_CONFLICTS = (
    (lambda a: a.headless, "out", "--headless cannot be combined with --out or --no-server."),
    (lambda a: a.headless, "no_server", "--headless cannot be combined with --out or --no-server."),
    (
        lambda a: not a.headless and a.out,
        "report",
        "--report only applies to native runs, not --out.",
    ),
    (
        lambda a: not (a.headless or a.out or a.native),
        "report",
        "--report only applies to native runs, not the editor debug loop.",
    ),
)


def _check_run_options(args: argparse.Namespace) -> None:
    for applies, flag, message in _RUN_CONFLICTS:
        if applies(args) and getattr(args, flag):
            raise BotLabError(message)


def _cmd_run(args: argparse.Namespace) -> int:
    _set_verbose(args)
    _check_run_options(args)
    project_dir = Path(args.project_dir or ".").resolve()
    if args.headless:
        from ._runtime.run import main as run_headless

        argv = ["botlab run --headless", str(project_dir)]
        if args.report:
            argv += ["--report", args.report]
        try:
            return run_headless(argv)
        except KeyboardInterrupt:
            print("[botlab.run] interrupted", file=sys.stderr)
            return 130

    project = load_project(project_dir)
    if os.environ.get("BOTLAB_VERBOSE"):
        _print_project_summary(project)
    metadata = {"name": project.manifest["project"]["name"]}

    if args.out:
        package = create_release_package(project, metadata=metadata)
        release_zip = Path(args.out).resolve()
        release_zip.write_bytes(package)
        print(f"[botlab] wrote {release_zip} ({len(package)} bytes)")
        if args.no_server:
            return 0
        # --out serves the frozen artifact: the embed surface is what
        # reviews it, since the editor debug loop always reads live sources.
        serve_project(
            project_dir,
            release_zip,
            args.host,
            args.port,
            open_browser=not args.no_browser,
            entry="/embed.html",
            ui_dir=ensure_browser_runtime(),
        )
        return 0

    if args.native:
        # Native execution: physics and inference run in a local CPython
        # subprocess; the browser only renders the loopback display stream.
        if args.no_server:
            return 0
        require_native_runtime()
        return run_native(
            project,
            project_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            report_path=args.report,
        )

    # Default: the browser editor debug loop (the full web UI). The editor
    # itself executes the project (MuJoCo WASM + ONNX Runtime Web); this
    # server only feeds it sources and assets, rebuilding on change so edits
    # land on reload. `--browser` spells the default out explicitly.
    package = build_debug_release(project)
    provider = DebugReleaseProvider(project_dir, package, _project_signature(project_dir))
    if os.environ.get("BOTLAB_VERBOSE"):
        print(
            f"[botlab] debug release: {len(package)} bytes in memory (no zip written) — "
            "edit the project, reload the page, the release rebuilds"
        )
    if args.no_server:
        return 0
    serve_project(
        project_dir,
        None,
        args.host,
        args.port,
        open_browser=not args.no_browser,
        release_provider=provider,
        ui_dir=ensure_browser_runtime(),
    )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """Export the project as a BotLab release.zip (local, no upload)."""
    project_dir = Path(args.project_dir or ".").resolve()
    project = load_project(project_dir)
    package = create_release_package(
        project,
        metadata={"name": project.manifest["project"]["name"]},
    )
    if args.site:
        from ._site import build_site_directory

        stats = build_site_directory(project_dir, package, Path(args.site).resolve())
        print(
            f"[botlab] wrote {stats['out_dir']} ({stats['files']} files, "
            f"{stats['total_bytes']} bytes) for project {stats['project']!r}"
        )
        print("[botlab] serve the directory as-is (e.g. push it to GitHub Pages)")
        if not args.out:
            return 0
    output = Path(args.out).resolve() if args.out else project_dir / "release.zip"
    if output.parent != project_dir and not output.parent.exists():
        raise BotLabError(f"output directory does not exist: {output.parent}")
    output.write_bytes(package)
    print(f"[botlab] wrote {output} ({len(package)} bytes)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    project = load_project(Path(args.project_dir or ".").resolve())
    package = create_release_package(project, metadata={"name": project.manifest["project"]["name"]})
    print(f"OK: {project.manifest['project']['name']} — release.zip would be {len(package)} bytes")
    return 0


def _botlab_checkout(args: argparse.Namespace) -> Path:
    raw = args.botlab or os.environ.get("BOTLAB_REPO")
    if not raw:
        raise BotLabError(
            f"`botlab ui {args.action}` needs a BotLab checkout: pass --botlab <path> "
            "or set BOTLAB_REPO."
        )
    return Path(raw).expanduser().resolve()


def _cmd_ui(args: argparse.Namespace) -> int:
    from ._ui_sync import sync_ui_snapshot

    if args.action == "status":
        target = ui_snapshot_dir()
        print(f"UI snapshot: {target}")
        print(f"native viewer index.html present: {(target / 'index.html').is_file()}")
        print(f"embed.html present: {(target / 'embed.html').is_file()}")
        print(f"browser runtime cache: {browser_runtime_cache_dir()}")
        return 0
    if args.action == "sync":
        checkout = _botlab_checkout(args)
        synced = sync_ui_snapshot(checkout, ui_snapshot_dir())
        print(f"synced {synced} files into {ui_snapshot_dir()}")
        cache_root = browser_runtime_cache_dir().parent
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sync-", dir=cache_root) as temporary:
            archive = Path(temporary) / f"botlab-browser-runtime-{__version__}.zip"
            exported = export_browser_runtime(checkout, archive)
            runtime = fetch_browser_runtime(archive=archive)
        print(f"cached browser runtime ({exported} files) at {runtime}")
        return 0
    if args.action == "export":
        if not args.out:
            raise BotLabError("`botlab ui export` requires --out <archive.zip>.")
        archive = Path(args.out).expanduser().resolve()
        exported = export_browser_runtime(_botlab_checkout(args), archive)
        print(f"exported {exported} browser runtime files to {archive}")
        return 0
    if args.action == "fetch":
        if args.archive and args.url:
            raise BotLabError("Use only one of --archive and --url.")
        runtime = fetch_browser_runtime(
            archive=Path(args.archive).expanduser().resolve() if args.archive else None,
            url=args.url,
        )
        print(f"installed browser runtime at {runtime}")
        return 0
    raise BotLabError(f"unknown ui subcommand {args.action!r}")


def _cmd_docs(args: argparse.Namespace) -> int:
    """Print where the bundled docs live, for humans and AI coding agents."""
    package_dir = Path(__file__).resolve().parent
    # Editable checkouts keep the docs at the repo root (src/botlab/../docs);
    # wheels carry them inside the package (botlab/docs).
    docs_dir = next(
        (
            candidate
            for candidate in (package_dir / "docs", package_dir.parent.parent / "docs")
            if candidate.is_dir()
        ),
        None,
    )
    if docs_dir is None:
        raise BotLabError(
            "this install does not bundle the docs (botlab/docs/ is missing); "
            "read them at https://github.com/OpenLegged/botlab-python/tree/main/docs"
        )
    print(docs_dir)
    for path in sorted(docs_dir.glob("*.md")):
        print(f"  {path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botlab",
        description="Compile, run, and export BotLab Python projects.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run a project in the browser editor (the full web UI); "
        "--native executes locally with a read-only viewer",
    )
    run.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
    execution = run.add_mutually_exclusive_group()
    execution.add_argument(
        "--browser",
        action="store_true",
        help="execute in the browser editor — the default, spelled out "
        "explicitly (MuJoCo WASM + ONNX Runtime Web)",
    )
    execution.add_argument(
        "--native",
        action="store_true",
        help="execute natively in CPython (local MuJoCo + ONNX Runtime); the "
        "browser only renders a read-only ``?ws=`` viewer of the simulation",
    )
    execution.add_argument(
        "--headless",
        action="store_true",
        help="run natively without a viewer or HTTP server (supports custom Python controllers)",
    )
    run.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0, LAN-reachable)")
    run.add_argument("--port", type=int, default=0, help="bind port (default: random free port)")
    run.add_argument(
        "--out",
        help="serve a release.zip exported to this path instead of the in-memory debug "
        "release (disables rebuild-on-change)",
    )
    run.add_argument(
        "--no-server",
        action="store_true",
        help="build the release and exit without serving (smoke test; no file written "
        "unless --out is given)",
    )
    run.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    run.add_argument(
        "--report",
        metavar="PATH",
        help="write a machine-readable JSON run report (base-pose series per robot, "
        "resets with reasons, sim.metrics series) after the run",
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="print the compile summary (resources, plan size, release bytes)",
    )
    run.set_defaults(func=_cmd_run)

    build = subparsers.add_parser(
        "build",
        help="export the project as release.zip (default: <project>/release.zip)",
    )
    build.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
    build.add_argument("--out", help="output path for release.zip (default: <project>/release.zip)")
    build.add_argument(
        "--site",
        help="also assemble a self-contained static site directory at this path "
        "(serve it as-is, e.g. on GitHub Pages); without --out no separate "
        "release.zip is written",
    )
    build.set_defaults(func=_cmd_build)

    validate = subparsers.add_parser("validate", help="apply a project and build its release in memory")
    validate.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
    validate.set_defaults(func=_cmd_validate)

    ui = subparsers.add_parser("ui", help="manage the bundled BotLab UI snapshot")
    ui.add_argument("action", choices=["sync", "status", "export", "fetch"])
    ui.add_argument(
        "--botlab",
        help="path to a BotLab checkout for sync/export (default: $BOTLAB_REPO)",
    )
    ui.add_argument("--out", help="versioned runtime archive output for ui export")
    ui.add_argument("--archive", help="offline runtime archive to install for ui fetch")
    ui.add_argument("--url", help="runtime archive URL override for ui fetch")
    ui.set_defaults(func=_cmd_ui)

    docs = subparsers.add_parser(
        "docs",
        help="print where the bundled docs (component reference, decision table, error guide) live",
    )
    docs.set_defaults(func=_cmd_docs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (BotLabError, OSError) as error:
        print(f"botlab: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
