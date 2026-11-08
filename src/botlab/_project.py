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
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rfc8785
import yaml

from . import _host
from ._host import PRESET_FILE_SUFFIX, BotLabError

STANDARD_PYTHON_MAIN = (
    "from botlab import Simulation\n"
    "from controller import create_controller\n"
    "\n"
    "sim = Simulation.from_project()\n"
    "sim.run(create_controller(sim))\n"
)

RELEASE_VERSION = 3
SIMULATION_ENGINE_ABI_VERSION = 1
MAX_PYTHON_RELEASE_PACKAGE_BYTES = 200 * 1024 * 1024

# Declaration compilation uses process-global host state and imported modules.
# Browser release requests and asynchronous HIL builds must share this lock.
PROJECT_LOAD_LOCK = threading.RLock()


def is_standard_python_main(source: str | None) -> bool:
    return source is not None and source.replace("\r\n", "\n").replace("\r", "\n").strip() == STANDARD_PYTHON_MAIN.strip()


def _is_safe_archive_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path:
        return False
    if ":" in path.split("/")[0] and len(path.split("/")[0]) > 1:
        # scheme-ish prefix like "http:" — mirror the TS regex test.
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path):
            return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


# -- project text file paths -------------------------------------------------
#
# is_project_text_file mirrors the editor's isProjectTextFilePath
# (src/domain/low-code/projectFilePaths.ts in BotLab): which project files are
# text declarations shipped in ``files`` and which stay binary assets. The
# browser publish path and this CLI must cut the same file set out of the same
# project or the release digests diverge — change both sides together.

_NAME_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PRESET_BASE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_PRESET_SUFFIX = PRESET_FILE_SUFFIX
_RESERVED_DIRECTORY_SEGMENTS = frozenset({"botlab"})
_TEXT_FILE_EXTENSIONS = frozenset({
    "md", "markdown", "txt", "json", "jsonc", "yaml", "yml", "toml", "ini",
    "cfg", "csv", "xml", "html", "css", "js", "jsx", "ts", "tsx", "sh",
})


def _is_word_character(character: str) -> bool:
    # TS \p{L}\p{N}: str.isalpha/str.isnumeric are exactly the Unicode
    # letter and number categories.
    return character.isalpha() or character.isnumeric()


def _is_python_source_path(relative: str) -> bool:
    parts = relative.split("/")
    file_name = parts.pop()
    if file_name.endswith(".py"):
        if not _NAME_SEGMENT.fullmatch(file_name[: -len(".py")]):
            return False
    elif file_name.endswith(_PRESET_SUFFIX):
        if not _PRESET_BASE.fullmatch(file_name[: -len(_PRESET_SUFFIX)]):
            return False
    else:
        return False
    return all(
        _NAME_SEGMENT.fullmatch(segment) and segment not in _RESERVED_DIRECTORY_SEGMENTS
        for segment in parts
    )


def _is_text_directory_segment(segment: str) -> bool:
    body = segment[1:] if segment.startswith(".") else segment
    if not body or not (_is_word_character(body[0]) or body[0] == "_"):
        return False
    return all(_is_word_character(c) or c in ("_", "-") for c in body)


def _is_text_file_name(name: str) -> bool:
    stem, dot, extension = name.rpartition(".")
    if not dot or extension not in _TEXT_FILE_EXTENSIONS:
        return False
    if not (_is_word_character(name[0]) or name[0] == "_"):
        return False
    return all(_is_word_character(c) or c in ("_", " ", ".", "-") for c in stem)


def is_project_text_file(relative: str) -> bool:
    """Editor isProjectTextFilePath, mirrored: .py/.preset.yaml follow the
    Python module path rules, other text files a looser unicode name rule;
    assets/ and botlab/ stay out of the text set."""
    if relative.endswith(".py") or relative.endswith(_PRESET_SUFFIX):
        return _is_python_source_path(relative)
    if not _is_safe_archive_path(relative) or relative == "botlab.project.json":
        return False
    parts = relative.split("/")
    file_name = parts.pop()
    if not _is_text_file_name(file_name):
        return False
    if parts and not all(
        _is_text_directory_segment(segment) and segment not in _RESERVED_DIRECTORY_SEGMENTS
        for segment in parts
    ):
        return False
    return not parts or parts[0] not in ("assets", "botlab")


@dataclass(frozen=True)
class LoadedProject:
    project_dir: Path
    manifest: dict
    files: dict[str, str]
    compiled_plan: dict | None
    project_digest: str
    plan_has_modifier: bool
    asset_paths: list[str] = field(default_factory=list)


def _parse_preset_file(preset_path: Path) -> tuple[str, dict]:
    with preset_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise BotLabError(f"A BotLab resource preset must be a mapping: {preset_path}")
    kind = data.pop("kind", None)
    if not isinstance(kind, str) or kind not in _host.RESOURCE_KINDS:
        raise BotLabError(
            f"Resource preset kind must be one of: {', '.join(sorted(_host.RESOURCE_KINDS))} "
            f"in {preset_path}"
        )
    return kind, data


def _preset_lookup(project_dir: Path):
    cache: dict[str, tuple[str, dict, bool]] = {}

    def lookup(preset_id: str):
        if preset_id in cache:
            return cache[preset_id]
        if preset_id.endswith(PRESET_FILE_SUFFIX):
            # Project-relative like the browser host's os.path.join(directory,
            # name): config/*.preset.yaml works, escapes do not.
            if os.path.isabs(preset_id) or ".." in preset_id.replace("\\", "/").split("/"):
                raise BotLabError(
                    f'BotLab resource preset "{preset_id}" must stay inside the project directory.'
                )
            preset_path = project_dir / preset_id
            if not preset_path.is_file():
                raise BotLabError(
                    f'BotLab resource preset file "{preset_id}" is not part of this project.'
                )
            kind, config = _parse_preset_file(preset_path)
            entry = (kind, _host._json_value(config, f"preset {preset_id}"), True)
        else:
            raise BotLabError(
                f'resource preset "{preset_id}" must be a project-local '
                f"{PRESET_FILE_SUFFIX} file"
            )
        cache[preset_id] = entry
        return entry

    return lookup


def walk_project_files(project_dir: Path):
    """Yield ``(relative_posix, path)`` for every real file under the project.

    Skips symlinks and local noise — dot entries at any depth, a root
    ``release.zip``/``tmp``, ``__pycache__``/``node_modules`` anywhere — so
    VCS metadata, the ``.botlab/`` project identity written between builds,
    and editor scratch never reach classification. The source collector and
    the local payload server share this walk: whatever the editor sees as
    text files must be exactly what the release digest covers.
    """
    for entry in sorted(project_dir.rglob("*")):
        if entry.is_symlink() or not entry.is_file():
            continue
        relative = entry.relative_to(project_dir)
        parts = relative.parts
        if any(part.startswith(".") for part in parts):
            continue
        if parts[0] in ("release.zip", "tmp") or any(
            part in ("__pycache__", "node_modules") for part in parts
        ):
            continue
        yield relative.as_posix(), entry


def _project_source_files(project_dir: Path) -> dict[str, str]:
    """Every project text file, collected recursively like the editor does.

    ``config/*.yaml`` declarations, nested Python packages and docs travel in
    ``files`` next to main.py/controller.py; classification is
    is_project_text_file's job.
    """
    files: dict[str, str] = {}
    for posix, entry in walk_project_files(project_dir):
        if posix == "botlab.yaml":
            if entry.stat().st_size > 0:
                raise BotLabError(
                    "A non-empty botlab.yaml is the legacy browser format; the offline CLI "
                    "only packages Python-declared projects (empty or missing YAML)."
                )
            continue
        if is_project_text_file(posix):
            files[posix] = entry.read_text(encoding="utf-8")
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
        if resource["kind"] == "sensor":
            target = next((item for item in manifest["resources"] if item["id"] == source), None)
            if target is None or target["kind"] != "robot":
                raise BotLabError(f'Sensor {resource["id"]!r} must reference a declared robot, got {source!r}')
            continue
        if source.startswith(("http://", "https://", "data:", "procedural://")):
            continue
        if not _is_safe_archive_path(source):
            raise BotLabError(f'Unsafe project asset path "{source}" in resource {resource["id"]!r}')
        candidate = project_dir / source
        if not candidate.resolve().is_relative_to(project_dir.resolve()):
            raise BotLabError(f'Resource path "{source}" escapes the project.')
        if not candidate.exists():
            raise BotLabError(
                f'Resource {resource["id"]!r} points at "{source}", which does not exist in the project.'
            )
        if candidate.is_file():
            assets.add(source)
            if resource["kind"] in ("robot", "scene") and candidate.suffix.lower() in (".xml", ".mjcf"):
                from ._mjcf import mjcf_dependencies

                documents, dependencies = mjcf_dependencies(candidate, root=project_dir)
                assets.update(path.relative_to(project_dir).as_posix() for path in set(documents) | dependencies)
        elif candidate.is_dir():
            for item in sorted(candidate.rglob("*")):
                if item.is_file():
                    if not item.resolve().is_relative_to(project_dir.resolve()):
                        raise BotLabError(f'Resource path "{item}" escapes the project.')
                    assets.add(item.relative_to(project_dir).as_posix())
        else:
            raise BotLabError(f'Resource path "{source}" is neither a file nor a directory.')
    return sorted(assets)


def canonical_json(value: Any) -> str:
    """Match existing browser canonicalValue + JSON.stringify bytes (release v3).

    JCS supplies ECMAScript number rendering. Objects retain JSON.stringify's
    numeric-index-first enumeration, which differs from pure JCS key ordering.
    """
    if isinstance(value, dict):
        def key_order(key):
            index = int(key) if key.isascii() and key.isdecimal() else -1
            if 0 <= index < 2**32 - 1 and str(index) == key:
                return (0, index)
            return (1, key.encode("utf-16-be", errors="surrogatepass"))

        return "{" + ",".join(canonical_json(key) + ":" + canonical_json(value[key])
                              for key in sorted(value, key=key_order)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(map(canonical_json, value)) + "]"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return rfc8785.dumps(float(value)).decode("utf-8")
        except (ValueError, OverflowError) as error:
            raise BotLabError("Release digest contains a non-finite number") from error
    # Well-formed JSON.stringify escapes lone UTF-16 surrogates, not other Unicode.
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "".join(f"\\u{ord(char):04x}" if 0xD800 <= ord(char) <= 0xDFFF else char for char in text)


def python_release_project_digest(manifest: dict, files: dict[str, str]) -> str:
    from ._host import _json_value

    normalized = _json_value({"manifest": manifest, "files": dict(sorted(files.items()))}, "release")
    payload = canonical_json(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evict_project_modules(project_dir: Path) -> None:
    """Drop cached project modules (nested packages included) from ``sys.modules``.

    A previous load_project or run in this process leaves ``controller``/
    ``main`` (and any sibling module) imported; re-importing the cache would
    run no declarations against a fresh session — the same eviction
    pyodideRuntime.stageProjectModules performs before staging sources
    (``pkg/helpers.py`` caches under ``pkg.helpers``). Every path that
    re-opens a project goes through this helper so the eviction set cannot
    drift between callers.
    """
    for name in ("controller", "main"):
        sys.modules.pop(name, None)
    for entry in project_dir.rglob("*.py"):
        if entry.is_symlink() or not entry.is_file():
            continue
        parts = entry.relative_to(project_dir).parts
        # ``pkg/__init__.py`` caches under the package name ``pkg``.
        if entry.stem == "__init__" and len(parts) > 1:
            sys.modules.pop(".".join(parts[:-1]), None)
        else:
            sys.modules.pop(".".join((*parts[:-1], entry.stem)), None)


def _import_controller(project_dir: Path, source_files: dict[str, str]) -> _host.HostSession:
    session = _host.begin_project_declaration(_preset_lookup(project_dir), project_dir)
    sys.path.insert(0, str(project_dir))
    try:
        evict_project_modules(project_dir)
        importlib.import_module("controller")
        return session
    finally:
        pass


def load_project(project_dir: Path) -> LoadedProject:
    """Apply the project: collect declarations, run main.py, capture the plan."""
    with PROJECT_LOAD_LOCK:
        return _load_project(project_dir)


def _load_project(project_dir: Path) -> LoadedProject:
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
    if plan.get("version") != 1:
        raise BotLabError("[botlab] PolicyController group plan.version must be 1")
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
        if ntype not in _host.SUPPORTED_NODE_TYPES:
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
        "version": 1,
        "runtime": {
            "engine": "botlab-simulation-engine",
            "abi": SIMULATION_ENGINE_ABI_VERSION,
            "simulator": "mujoco",
            "requiredNodeTypes": _required_node_types(plan),
        },
        "projectDigest": python_release_project_digest(manifest, files),
        "plan": plan,
    }
