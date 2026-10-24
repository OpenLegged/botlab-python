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
import importlib
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
from ._host import BotLabError
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


def _run_browser_adapter(args: argparse.Namespace, project_dir: Path) -> int:
    from .browser import BrowserRunContext

    if args.native or args.headless or args.out or args.no_server or args.report:
        raise BotLabError(
            "--browser-adapter requires a live browser run; it cannot be combined "
            "with --native, --headless, --out, --no-server, or --report."
        )
    module_name, separator, function_name = args.browser_adapter.partition(":")
    if (not separator or not function_name.isidentifier()
            or not all(part.isidentifier() for part in module_name.split("."))):
        raise BotLabError("--browser-adapter must be MODULE:FUNCTION.")
    if not project_dir.is_dir():
        raise BotLabError(f"project directory does not exist: {project_dir}")
    previous_directory = Path.cwd()
    previous_path = sys.path[:]
    try:
        os.chdir(project_dir)
        sys.path.insert(0, str(project_dir))
        try:
            module = importlib.import_module(module_name)
            adapter = getattr(module, function_name)
        except (ImportError, AttributeError) as error:
            raise BotLabError(f"Cannot load browser adapter {args.browser_adapter}: {error}") from error
        if not callable(adapter):
            raise BotLabError(f"Browser adapter {args.browser_adapter} is not callable.")
        context = BrowserRunContext(
            project_dir=project_dir,
            ui_dir=ensure_browser_runtime(),
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            inference_endpoint=args.inference_endpoint,
        )
        return adapter(context)
    finally:
        sys.path[:] = previous_path
        os.chdir(previous_directory)


def _cmd_run(args: argparse.Namespace) -> int:
    _set_verbose(args)
    project_dir = Path(args.project_dir or ".").resolve()
    adapter = getattr(args, "browser_adapter", None)
    target = getattr(args, "hil", None)
    if target is not None and (adapter or args.native or args.headless or args.out or args.no_server):
        raise BotLabError("--hil requires the standard live browser run.")
    args.host = args.host or ("0.0.0.0" if args.native or args.headless else "127.0.0.1")
    if target is not None and args.host not in ("127.0.0.1", "localhost"):
        raise BotLabError("HIL requires --host 127.0.0.1 or localhost.")
    if adapter:
        return _run_browser_adapter(args, project_dir)
    if getattr(args, "inference_endpoint", None):
        raise BotLabError("--inference-endpoint requires --browser-adapter.")
    if args.headless:
        if args.out or args.no_server:
            raise BotLabError("--headless cannot be combined with --out or --no-server.")
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
        if args.report:
            raise BotLabError("--report only applies to native runs, not --out.")
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
    if args.report:
        raise BotLabError("--report only applies to native runs, not the editor debug loop.")
    package = build_debug_release(project)
    provider = DebugReleaseProvider(project_dir, package, _project_signature(project_dir))
    if os.environ.get("BOTLAB_VERBOSE"):
        print(
            f"[botlab] debug release: {len(package)} bytes in memory (no zip written) — "
            "edit the project, reload the page, the release rebuilds"
        )
    if args.no_server:
        return 0
    from .hil import HilHost

    runtime = ensure_browser_runtime()
    if target is not None:
        import json

        try:
            capability = json.loads((runtime / "botlab-capabilities.json").read_text())
        except (OSError, ValueError):
            capability = {}
        if capability.get("hilProtocol") != 1:
            raise BotLabError("The cached browser runtime does not support HIL v1. Install a matching runtime with `botlab ui fetch --archive <path>`.")
    try:
        hil = HilHost(project_dir, selected_target=target) if args.host in ("127.0.0.1", "localhost") else None
    except ValueError as error:
        raise BotLabError(str(error)) from error
    serve_project(
        project_dir,
        None,
        args.host,
        args.port,
        open_browser=not args.no_browser,
        release_provider=provider,
        ui_dir=runtime,
        hil=hil,
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
        checkout = Path(args.botlab).expanduser().resolve()
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
        exported = export_browser_runtime(Path(args.botlab).expanduser().resolve(), archive)
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


def _hil_result(host, result):
    """Wait for a CLI management job; the browser uses the same operations API."""
    import time
    while isinstance(result, dict) and result.get("operation"):
        operation = result["operation"]
        if operation["status"] == "failed":
            raise BotLabError(operation["error"])
        if operation["status"] == "succeeded":
            return result.get("result", result)
        time.sleep(0.2)
        result = host.handle("GET", "/hil/operations/" + operation["id"])
    return result


def _bench_target(with_host, args) -> dict:
    """Serial-inference latency report: the real-time acceptance tier.

    Functional acceptance (the model executes correctly) is covered by the
    automated suites; this measures whether the whole loop fits the wall-clock
    budget (e.g. 20 ms at control_hz=50). No receipts, no retry — inference
    frames are never re-sent.
    """
    import base64
    import hashlib
    import random
    import statistics
    if not args.model:
        raise BotLabError("target bench requires --model FILE")
    _hil_result(with_host, with_host.handle("POST", "/hil/connect", {"target": args.name}))
    session = with_host.handle("POST", "/hil/start", {})["session"]
    content = Path(args.model).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        model = with_host.handle("POST", "/hil/models", {
            "session": session, "sha256": digest, "data": base64.b64encode(content).decode()})

        def feed():
            inputs = {}
            for spec in model["inputs"]:
                shape = [n if type(n) is int and n > 0 else 1 for n in spec["shape"]]
                size = 1
                for n in shape:
                    size *= n
                inputs[spec["name"]] = {"dtype": "float32", "shape": shape,
                                        "data": [random.uniform(-1, 1) for _ in range(size)]}
            return inputs

        rtts, inference = [], []
        for seq in range(args.iterations):
            result = with_host.handle("POST", "/hil/infer", {
                "session": session, "seq": seq, "model": digest, "inputs": feed()})
            rtts.append(result["rtt_ms"])
            inference.append(result["inference_ms"])
    finally:
        with_host.handle("POST", "/hil/stop", {"session": session})
    rtts.sort()

    def percentile(fraction):
        return round(rtts[min(len(rtts) - 1, int(fraction * len(rtts)))], 3)

    mean = statistics.fmean(rtts)
    return {"target": args.name, "model": digest, "iterations": args.iterations,
            "rtt_ms": {"p50": percentile(0.50), "p95": percentile(0.95),
                       "p99": percentile(0.99), "max": round(rtts[-1], 3)},
            "inference_ms_mean": round(statistics.fmean(inference), 3),
            "effective_hz": round(1000 / mean, 1),
            "budget_ms": args.budget_ms,
            "real_time_budget_met": mean <= args.budget_ms}


def _cmd_target(args: argparse.Namespace) -> int:
    import json
    from .hil import HilHost

    with_host = HilHost(Path.cwd())
    try:
        action = args.action
        body = {"target": args.name}
        if getattr(args, "ask_password", False):
            import getpass
            body["password"] = getpass.getpass("SSH password: ")
        if getattr(args, "host_key", None):
            body["host_key"] = args.host_key
        if action == "list":
            result = with_host.handle("GET", "/hil/state")["targets"]
        elif action == "discover":
            result = with_host.handle("POST", "/hil/devices/discover", {
                "interface": args.interface, "profile": args.profile})
        elif action == "add":
            if not args.name or not (args.endpoint or args.ssh_host or args.gateway):
                raise BotLabError("target add requires NAME and (--ssh-host HOST --user USER "
                                  "| --endpoint tcp://host:port | --gateway wss://… --device ID)")
            settings = {"id": args.name, "name": args.name, "backend": args.backend}
            if args.gateway:
                settings.update(transport="gateway", endpoint=args.gateway, device=args.device)
            elif args.ssh_host:
                settings.update(transport="ssh", ssh={"host": args.ssh_host, "user": args.user,
                    "port": args.ssh_port}, root=args.root, profile=args.profile or "generic-linux")
                if args.identity_file:
                    settings["ssh"]["identity_file"] = args.identity_file
                if args.host_key:
                    settings["ssh"]["host_key"] = args.host_key
            else:
                settings["endpoint"] = args.endpoint
            result = with_host.handle("POST", "/hil/targets", settings)
        else:
            if not args.name:
                raise BotLabError("target " + action + " requires NAME")
            target = next((t for t in with_host.targets if t["id"] == args.name), None)
            if action == "remove":
                result = with_host.handle("POST", "/hil/targets/remove", body)
            elif action == "provision":
                result = with_host.handle("POST", "/hil/devices/provision", body)
            elif action == "identity":
                result = with_host.handle("POST", "/hil/devices/host-key", body)
            elif action == "inspect" and target and target.get("transport") == "ssh":
                result = with_host.handle("POST", "/hil/devices/inspect", body)
            elif action == "bench":
                result = _bench_target(with_host, args)
            else:
                result = _hil_result(with_host, with_host.handle("POST", "/hil/connect", body))
                if action == "projects":
                    result = with_host.handle("GET", "/hil/projects")
        print(json.dumps(_hil_result(with_host, result), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        raise BotLabError(str(error)) from error
    finally:
        with_host.close()
    return 0


def _cmd_deploy(args: argparse.Namespace) -> int:
    import json
    from .hil import HilHost
    host = HilHost(Path(args.project_dir or "."))
    body = {"target": args.target}
    if args.ask_password:
        import getpass
        body["password"] = getpass.getpass("SSH password: ")
    try:
        _hil_result(host, host.handle("POST", "/hil/connect", body))
        _hil_result(host, host.handle("POST", "/hil/build", {}))
        result = _hil_result(host, host.handle("POST", "/hil/deploy", {}))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as error:
        raise BotLabError(str(error)) from error
    finally:
        host.close()


def _cmd_agent(args: argparse.Namespace) -> int:
    from .hil.agent import main as agent_main

    return agent_main(args.agent_args)


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
    run.add_argument(
        "--host", help="bind address (browser default: 127.0.0.1; native: 0.0.0.0)",
    )
    run.add_argument(
        "--hil", metavar="TARGET",
        help="select a registered HIL target or tcp://host:port in the browser toolbar",
    )
    run.add_argument(
        "--browser-adapter", metavar="MODULE:FUNCTION",
        help="run a project browser adapter with the SDK-managed web runtime",
    )
    run.add_argument(
        "--inference-endpoint", metavar="HOST:PORT",
        help="remote inference address passed to the selected browser adapter",
    )
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
    ui.add_argument("--botlab", default="~/Project/BotLab", help="path to a BotLab checkout")
    ui.add_argument("--out", help="versioned runtime archive output for ui export")
    ui.add_argument("--archive", help="offline runtime archive to install for ui fetch")
    ui.add_argument("--url", help="runtime archive URL override for ui fetch")
    ui.set_defaults(func=_cmd_ui)

    target = subparsers.add_parser("target", help="manage HIL target devices")
    target.add_argument("action", choices=["list", "add", "inspect", "remove", "discover", "identity", "provision", "projects", "bench"])
    target.add_argument("name", nargs="?")
    target.add_argument("--endpoint")
    target.add_argument("--gateway", help="outbound gateway origin, ws://loopback:port or wss://host[:port]")
    target.add_argument("--device", help="device ID the board agent registered at the gateway")
    target.add_argument("--model", type=Path, help="model file for target bench")
    target.add_argument("--iterations", type=int, default=200, help="bench inference iterations")
    target.add_argument("--budget-ms", type=float, default=20.0,
                        help="wall-clock budget per inference (control_hz=50 → 20)")
    target.add_argument("--backend", default="onnxruntime")
    target.add_argument("--ssh-host")
    target.add_argument("--user")
    target.add_argument("--ssh-port", type=int, default=22)
    target.add_argument("--identity-file")
    target.add_argument("--host-key", help="explicitly trusted SSH SHA256 fingerprint")
    target.add_argument("--ask-password", action="store_true")
    target.add_argument("--root", default="~/botlab")
    target.add_argument("--interface")
    target.add_argument("--profile", choices=["generic-linux", "rdk-x5"])
    target.set_defaults(func=_cmd_target)

    deploy = subparsers.add_parser("deploy", help="build and install a HIL application without running it")
    deploy.add_argument("project_dir", nargs="?")
    deploy.add_argument("--target", required=True)
    deploy.add_argument("--ask-password", action="store_true")
    deploy.set_defaults(func=_cmd_deploy)

    agent = subparsers.add_parser("agent", help="run the target-side HIL agent")
    agent.add_argument("agent_args", nargs=argparse.REMAINDER)
    agent.set_defaults(func=_cmd_agent)

    control = subparsers.add_parser("control", help="control project middleware through the generic Control API")
    control.add_argument("control_args", nargs=argparse.REMAINDER)

    docs = subparsers.add_parser(
        "docs",
        help="print where the bundled docs (component reference, decision table, error guide) live",
    )
    docs.set_defaults(func=_cmd_docs)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "control":
        from .control.cli import main as control_main

        return control_main(argv[1:])
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
