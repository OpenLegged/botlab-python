"""Load a BotLab project directory and compile it into a release payload.

A project is the two-file contract from BotLab: ``main.py`` + ``controller.py``
(plus optional flat ``*.preset.yaml`` calibration files and project-relative
assets). Importing ``controller`` against the static host collects the
resource declarations; executing ``main.py`` registers the compiled
controller plan. The result is everything ``embed.html`` needs at runtime,
without the browser ever needing Pyodide for this project again.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _host
from ._contract import PRESET_FILE_SUFFIX, SUPPORTED_NODE_TYPES
from ._errors import BotLabError
from ._json import json_value
from ._preset_files import _preset_lookup

STANDARD_PYTHON_MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)

RELEASE_VERSION = 3
SIMULATION_ENGINE_ABI_VERSION = 1
# Compiled plan contract version (policyControllerPlanEnvelope.ts); group envelope,
# release envelope and single plans share it.
POLICY_CONTROLLER_PLAN_VERSION = 1
MAX_PYTHON_RELEASE_PACKAGE_BYTES = 200 * 1024 * 1024

# Flat project files a release may carry besides main.py / controller.py.
SAFE_PROJECT_SUFFIXES = (".py", PRESET_FILE_SUFFIX)


def is_standard_python_main(source: str | None) -> bool:
    return source is not None and source.replace("\r\n", "\n").replace("\r", "\n").strip() == STANDARD_PYTHON_MAIN.strip()


def _is_safe_archive_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path:
        return False
    if ":" in path.split("/")[0] and len(path.split("/")[0]) > 1:
        # scheme-ish prefix like "http:" — mirror the TS regex test.
        import re

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path):
            return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


@dataclass(frozen=True)
class LoadedProject:
    project_dir: Path
    manifest: dict
    files: dict[str, str]
    compiled_plan: dict | None
    project_digest: str
    plan_has_modifier: bool
    asset_paths: list[str] = field(default_factory=list)


def _project_source_files(project_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for entry in sorted(project_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == "botlab.yaml":
            if entry.stat().st_size > 0:
                raise BotLabError(
                    "A non-empty botlab.yaml is the legacy browser format; the offline CLI "
                    "only packages Python-declared projects (empty or missing YAML)."
                )
            continue
        if entry.name in ("main.py", "controller.py") or entry.name.endswith(SAFE_PROJECT_SUFFIXES):
            files[entry.name] = entry.read_text(encoding="utf-8")
    if "main.py" not in files:
        raise BotLabError('A BotLab project requires "main.py".')
    if "controller.py" not in files:
        raise BotLabError('A BotLab project requires "controller.py".')
    return files


def _project_relative_assets(project_dir: Path, source_files: dict[str, str], manifest: dict) -> list[str]:
    """Project-relative paths every declared resource resolves underneath.

    The manifest sources are project-relative (``assets/robot/scene.xml``);
    the CLI serves the whole project directory at ``/project/`` so the
    browser resolves them exactly like a workspace project.
    """
    assets: set[str] = set()
    for resource in manifest["resources"]:
        source = resource["source"]
        if source.startswith(("http://", "https://", "data:", "procedural://")):
            continue
        if not _is_safe_archive_path(source):
            raise BotLabError(f'Unsafe project asset path "{source}" in resource {resource["id"]!r}')
        candidate = project_dir / source
        if not candidate.exists():
            raise BotLabError(
                f'Resource {resource["id"]!r} points at "{source}", which does not exist in the project.'
            )
        if candidate.is_file():
            assets.add(source)
        elif candidate.is_dir():
            for item in sorted(candidate.rglob("*")):
                if item.is_file():
                    assets.add(item.relative_to(project_dir).as_posix())
        else:
            raise BotLabError(f'Resource path "{source}" is neither a file nor a directory.')
    return sorted(assets)


def _canonical_value(value: Any) -> Any:
    """Mirror the TS ``canonicalValue``: sort object keys recursively.

    JS ``Object.keys(value).sort()`` reorders every nested object, so the
    digest input is key-order-canonical on both sides.
    """
    if isinstance(value, dict):
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize like JS ``JSON.stringify`` of the canonicalized digest input.

    ``pythonReleaseProjectDigest`` canonicalizes (recursively sorted object
    keys, finite numbers only) then digests ``JSON.stringify`` output.
    Python's ``json`` with ``separators=(',', ':')`` produces the same bytes
    as JS as long as floats render identically; integer-valued floats were
    normalized to ``int`` upstream (JS ``JSON.stringify(1.0)`` is ``"1"``).
    """
    return json.dumps(_canonical_value(value), ensure_ascii=False, separators=(",", ":"))


def python_release_project_digest(manifest: dict, files: dict[str, str]) -> str:
    normalized = json_value({"manifest": manifest, "files": dict(sorted(files.items()))}, "release")
    payload = canonical_json(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evict_project_modules(project_dir: Path) -> None:
    """Drop cached top-level project modules from ``sys.modules``.

    A previous load_project or run in this process leaves ``controller``/
    ``main`` (and any sibling module) imported; re-importing the cache would
    run no declarations against a fresh session — the same eviction
    pyodideRuntime.stageProjectModules performs before staging sources.
    Every path that re-opens a project goes through this helper so the
    eviction set cannot drift between callers.
    """
    for name in ("controller", "main"):
        sys.modules.pop(name, None)
    for entry in project_dir.iterdir():
        if entry.is_file() and entry.suffix == ".py":
            sys.modules.pop(entry.stem, None)


def _import_controller(project_dir: Path, source_files: dict[str, str]) -> _host.HostSession:
    session = _host.begin_project_declaration(_preset_lookup(project_dir))
    sys.path.insert(0, str(project_dir))
    try:
        evict_project_modules(project_dir)
        importlib.import_module("controller")
        return session
    finally:
        pass


def load_project(project_dir: Path) -> LoadedProject:
    """Apply the project: collect declarations, run main.py, capture the plan."""
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise BotLabError(f"Project directory does not exist: {project_dir}")

    source_files = _project_source_files(project_dir)
    if not is_standard_python_main(source_files["main.py"]):
        raise BotLabError(
            "The offline CLI packages compiled controllers, which require the standard "
            "main.py entrypoint:\n" + STANDARD_PYTHON_MAIN
        )

    session = _import_controller(project_dir, source_files)
    manifest = session.project_manifest(project_dir.name)
    _host.finish_declaration()

    # main.py execution: the standard four lines register the controller plan.
    importlib.import_module("main")

    plan = session.policy_controller_plan
    has_modifier = session.plan_has_modifier
    _host.end_project_declaration()
    if plan is None:
        raise BotLabError(
            "main.py did not register a controller; a release requires "
            "Simulation.run(PolicyController(...))."
        )

    # A modify() hook needs a live Python runtime, which the embed runtime
    # provides via Pyodide when the package carries no compiled controller:
    # sources + manifest travel, the plan does not.
    compiled_plan = None if has_modifier else plan

    digest = python_release_project_digest(manifest, source_files)
    assets = _project_relative_assets(project_dir, source_files, manifest)
    return LoadedProject(
        project_dir=project_dir,
        manifest=manifest,
        files=source_files,
        compiled_plan=compiled_plan,
        project_digest=digest,
        plan_has_modifier=has_modifier,
        asset_paths=assets,
    )


def _required_node_types(plan: dict) -> list[str]:
    entries = _plan_entries(plan)
    types: set[str] = set()
    for entry in entries:
        for node in entry.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("type"), str):
                types.add(node["type"])
    return sorted(types)


def _plan_entries(plan: dict) -> list[dict]:
    """Expand the single/group controller envelope (policyControllerPlanEnvelope.ts).

    A group is ``{"version": 1, "controllers": [...]}`` — no nested groups, no
    other fields; anything else is a single plan.
    """
    if not isinstance(plan, dict) or "controllers" not in plan:
        return [plan]
    unknown = [field for field in plan if field not in ("version", "controllers")]
    if unknown:
        raise BotLabError(
            f"[botlab] PolicyController group plan has unknown fields {unknown}"
        )
    if plan.get("version") != POLICY_CONTROLLER_PLAN_VERSION:
        raise BotLabError(
            f"[botlab] PolicyController group plan.version must be {POLICY_CONTROLLER_PLAN_VERSION}"
        )
    controllers = plan["controllers"]
    if not isinstance(controllers, list) or not controllers:
        raise BotLabError(
            "[botlab] PolicyController group plan.controllers must be a non-empty list"
        )
    for index, entry in enumerate(controllers):
        if isinstance(entry, dict) and "controllers" in entry:
            raise BotLabError(
                f"[botlab] PolicyController group plan.controllers[{index}] "
                "cannot be another group"
            )
    return controllers


def validate_compiled_plan(plan: dict, manifest: dict) -> None:
    """Port of pythonReleaseControllerValidation.ts, plus the digest envelope."""
    entries = _plan_entries(plan)
    robot_ids: set[str] = set()
    for entry in entries:
        _validate_single_plan(entry, manifest)
        robot_id = entry.get("robot")
        if robot_id in robot_ids:
            raise BotLabError(
                f'compiled controller plan contains duplicate controller for robot "{robot_id}"'
            )
        robot_ids.add(robot_id)
    declared = [item for item in manifest["resources"] if item["kind"] == "robot"]
    if any(robot["id"] not in robot_ids for robot in declared):
        raise BotLabError(
            "compiled controller plan must provide exactly one controller for every declared robot"
        )


def _validate_single_plan(plan: dict, manifest: dict) -> None:
    robot = next(
        (item for item in manifest["resources"] if item["id"] == plan.get("robot")), None
    )
    if robot is None or robot["kind"] != "robot":
        raise BotLabError(f'compiled controller plan robot "{plan.get("robot")}" is not declared')

    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise BotLabError("compiled controller plan must contain at least one node")
    node_by_id: dict[str, dict] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise BotLabError(f"compiled controller plan contains an invalid node at index {index}")
        nid, ntype = node.get("id"), node.get("type")
        if not isinstance(nid, str) or not nid:
            raise BotLabError(f"compiled controller plan contains an invalid node id at index {index}")
        if not isinstance(ntype, str) or not ntype or not isinstance(node.get("data"), dict):
            raise BotLabError(f"compiled controller plan contains an invalid node at index {index}")
        if ntype not in SUPPORTED_NODE_TYPES:
            raise BotLabError(f'compiled controller plan requires unsupported node type "{ntype}"')
        if nid in node_by_id:
            raise BotLabError(f'compiled controller plan contains duplicate node id "{nid}"')
        node_by_id[nid] = node

    edge_ids: set[str] = set()
    for index, edge in enumerate(plan.get("edges", [])):
        if not isinstance(edge, dict):
            raise BotLabError(f"compiled controller plan contains an invalid edge at index {index}")
        ok = (
            isinstance(edge.get("id"), str) and edge["id"]
            and isinstance(edge.get("source"), str) and edge["source"] in node_by_id
            and isinstance(edge.get("target"), str) and edge["target"] in node_by_id
        )
        if not ok:
            raise BotLabError(f"compiled controller plan contains an invalid edge at index {index}")
        if edge["id"] in edge_ids:
            raise BotLabError(f'compiled controller plan contains duplicate edge id "{edge["id"]}"')
        edge_ids.add(edge["id"])

    output = plan.get("output")
    if not isinstance(output, dict):
        raise BotLabError("compiled controller plan output is missing")
    hooks = [node for node in node_by_id.values() if node["type"] == "proc-action-hook"]
    if len(hooks) != 1:
        raise BotLabError("compiled controller plan must contain exactly one proc-action-hook node")
    action_node = node_by_id.get(output.get("policyActionNodeId", ""))
    if output.get("policyActionNodeId") not in node_by_id or action_node["type"] != "proc-action-hook":
        raise BotLabError("compiled controller plan output.policyActionNodeId must reference the proc-action-hook node")
    output_kind = output.get("kind", "position-pd")
    if output_kind == "raw":
        if output.get("controlNodeId") not in node_by_id:
            raise BotLabError("compiled controller plan output.controlNodeId must reference a plan node")
    elif output_kind == "position-pd":
        pd_node = node_by_id.get(output.get("pdNodeId", ""))
        if output.get("pdNodeId") not in node_by_id or pd_node["type"] != "proc-pd":
            raise BotLabError("compiled controller plan output.pdNodeId must reference a proc-pd node")
    else:
        raise BotLabError('compiled controller plan output.kind must be "position-pd" or "raw"')


def compiled_controller_envelope(plan: dict, manifest: dict, files: dict[str, str]) -> dict:
    validate_compiled_plan(plan, manifest)
    return {
        "kind": "policy-controller-plan",
        "version": POLICY_CONTROLLER_PLAN_VERSION,
        "runtime": {
            "engine": "botlab-simulation-engine",
            "abi": SIMULATION_ENGINE_ABI_VERSION,
            "simulator": "mujoco",
            "requiredNodeTypes": _required_node_types(plan),
        },
        "projectDigest": python_release_project_digest(manifest, files),
        "plan": plan,
    }
