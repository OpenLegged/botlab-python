"""The ``botlab`` command line interface.

- ``botlab run [PROJECT_DIR]`` — debug-run the project from source: compile it,
  serve the full BotLab editor (plus the embed runtime) to the browser, and
  rebuild the release whenever source changes. The editor opens with the
  local project loaded: edit the code, hit Run (or reload) — the debug server
  always serves current sources and assets straight from the directory. No
  zip file is written.
- ``botlab build [PROJECT_DIR]`` — export release.zip locally (no server, no
  upload); the same package the platform's ``/d/<id>/`` demos are served from.
- ``botlab validate`` / ``botlab ui`` — packaging dry-run and UI snapshot tools.

Any standard two-file BotLab project works; nothing here knows about a specific
robot or policy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._debug import DebugReleaseProvider, _project_signature, build_debug_release
from ._host import BotLabError
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


def _cmd_run(args: argparse.Namespace) -> int:
    _set_verbose(args)
    project_dir = Path(args.project_dir or ".").resolve()
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
        )
        return 0

    # Debug loop: the provider rebuilds on source change, so the user edits
    # and reloads the browser without restarting this command.
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


def _cmd_ui(args: argparse.Namespace) -> int:
    from ._ui_sync import sync_ui_snapshot

    if args.action == "status":
        target = ui_snapshot_dir()
        print(f"UI snapshot: {target}")
        print(f"editor index.html present: {(target / 'index.html').is_file()}")
        print(f"embed.html present: {(target / 'embed.html').is_file()}")
        return 0
    if args.action == "sync":
        synced = sync_ui_snapshot(Path(args.botlab).resolve(), ui_snapshot_dir())
        print(f"synced {synced} files into {ui_snapshot_dir()}")
        return 0
    raise BotLabError(f"unknown ui subcommand {args.action!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botlab",
        description="Compile and run BotLab Python projects in the browser BotLab runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="debug-run a project from source in the browser simulation runtime",
    )
    run.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
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
    build.set_defaults(func=_cmd_build)

    validate = subparsers.add_parser("validate", help="apply a project and build its release in memory")
    validate.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
    validate.set_defaults(func=_cmd_validate)

    ui = subparsers.add_parser("ui", help="manage the bundled BotLab UI snapshot")
    ui.add_argument("action", choices=["sync", "status"])
    ui.add_argument("--botlab", default="~/Project/BotLab", help="path to a BotLab checkout")
    ui.set_defaults(func=_cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BotLabError as error:
        print(f"botlab: error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"botlab: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
