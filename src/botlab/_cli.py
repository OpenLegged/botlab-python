"""The ``botlab`` command line interface.

- ``botlab run [PROJECT_DIR]`` — compile the project, package release.zip, and
  open the browser BotLab simulation runtime.
- ``botlab build [PROJECT_DIR]`` — export release.zip locally (no server, no
  upload); the same package the platform's ``/d/<id>/`` demos are served from.
- ``botlab validate`` / ``botlab ui`` — packaging dry-run and UI snapshot tools.

Any standard two-file BotLab project works; nothing here knows about a specific
robot or policy.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ._host import BotLabError
from ._project import load_project
from ._release import create_release_package
from ._server import serve_project, ui_snapshot_dir


def _cmd_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir or ".").resolve()
    project = load_project(project_dir)
    package = create_release_package(
        project,
        metadata={
            "name": project.manifest["project"]["name"],
        },
    )

    release_zip: Path
    if args.out:
        release_zip = Path(args.out).resolve()
        release_zip.write_bytes(package)
        print(f"[botlab] wrote {release_zip} ({len(package)} bytes)")
    else:
        release_handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lives as long as the server
            prefix="botlab-release-", suffix=".zip", delete=False
        )
        release_handle.write(package)
        release_handle.flush()
        release_zip = Path(release_handle.name)

    if args.no_server:
        return 0

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
    try:
        serve_project(
            project_dir,
            release_zip,
            args.host,
            args.port,
            open_browser=not args.no_browser,
        )
    finally:
        if not args.out:
            release_zip.unlink(missing_ok=True)
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

    run = subparsers.add_parser("run", help="package a project and open the simulation runtime")
    run.add_argument("project_dir", nargs="?", default=".", help="project directory (default: cwd)")
    run.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    run.add_argument("--port", type=int, default=0, help="bind port (default: random free port)")
    run.add_argument("--out", help="write release.zip to this path instead of a temp file")
    run.add_argument("--no-server", action="store_true", help="only build release.zip (--out required)")
    run.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
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
