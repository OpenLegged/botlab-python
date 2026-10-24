"""BotLab Python release package writer (format version 3).

The browser embed runtime (``embed.html`` → ``fetchPublishedProject``) reads
``./release.zip``: ``release.json`` plus ``project/`` sources and ``assets/``
binaries, with SHA-256 integrity for every entry. This module writes the same
archive from a compiled :class:`botlab._project.LoadedProject` — asset
packaging differs from the browser writer only where the browser would fetch
remote URLs: the CLI packages project-local files by path instead.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from ._contract import MOTION_REQUIRED_FILES
from ._errors import BotLabError
from ._project import MAX_PYTHON_RELEASE_PACKAGE_BYTES, LoadedProject

ASSET_FOLDERS = {
    "model": "models",
    "motion": "motions",
    "mesh": "meshes",
    "robot": "robots",
    "scene": "scenes",
}

MIME_BY_ROLE_SUFFIX = {
    "model": "application/octet-stream",
    "robot": "application/xml",
    "scene": "application/xml",
}


def _sha256_integrity(data: bytes) -> dict:
    return {
        "algorithm": "SHA-256",
        "byteLength": len(data),
        "digest": hashlib.sha256(data).hexdigest(),
    }


def _sanitize_segment(value: str, fallback: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return safe or fallback


def _source_file_name(resource: dict, role: str) -> str:
    source = resource["source"]
    if source.startswith(("data:", "blob:")):
        extension = "onnx" if role == "model" else ("xml" if role in ("robot", "scene") else "bin")
        return f"{_sanitize_segment(resource['id'], role)}.{extension}"
    # Same URL stripping the npz detection uses, so a query-bearing source
    # still derives its plain file name ("demos.npz?raw=1" → "demos.npz") —
    # consumers key npz-ness off the packaged name.
    name = _strip_url_extras(source).rstrip("/").split("/")[-1]
    return _sanitize_segment(name, f"{_sanitize_segment(resource['id'], role)}.bin")


def _asset_path(role: str, resource_id: str, name: str, used: set[str]) -> str:
    base = _sanitize_segment(resource_id, role)
    path = f"assets/{ASSET_FOLDERS[role]}/{base}-{name}"
    suffix = 2
    while path in used:
        path = f"assets/{ASSET_FOLDERS[role]}/{base}-{suffix}-{name}"
        suffix += 1
    used.add(path)
    return path


def _should_package(resource: dict) -> bool:
    kind = resource["kind"]
    if kind == "sensor":
        return False
    if kind in ("model", "motion", "mesh"):
        return True
    if resource["source"].startswith("procedural://"):
        return False
    # Robot/scene XML keeps its project-relative source so the embed runtime
    # resolves it (and its relative mesh includes) through the asset inventory
    # served at /project/ — the same shape built-in demos use. Restoring an
    # XML from assets/ would rewrite its source to a blob: URL, whose opaque
    # base cannot resolve the MJCF's relative mesh paths.
    return False


def _strip_url_extras(source: str) -> str:
    """Drop the registry scheme and URL query/fragment — the parts of a
    source that never name a file on disk."""
    if source.startswith("registry://"):
        source = source[len("registry://"):]
    return source.split("?", 1)[0].split("#", 1)[0]


def _is_npz_motion_source(source: str) -> bool:
    """Mirror the engine's isNpzMotionSource: motion archives package as one
    asset; everything else is the seven-file .bin folder layout. A trailing
    slash means folder, exactly like the engine (URL pathname) and the local
    runtime's ``os.path.isdir`` dispatch."""
    return _strip_url_extras(source).lower().endswith(".npz")


def _pack_resource(project_dir: Path, resource: dict, used: set[str]) -> list[dict]:
    """Read one resource's bytes from the project directory.

    The compiled-controller embed runtime resolves ``assets/`` entries into
    object URLs and rewrites ``resource.source`` to them; the manifest source
    therefore only has to be resolvable at package time.
    """
    if not _should_package(resource):
        return []
    role = resource["kind"]
    if role not in ASSET_FOLDERS:
        return []
    entries: list[dict] = []

    def read_file(relative: str) -> bytes:
        path = project_dir / relative
        if not path.is_file():
            raise BotLabError(
                f'Resource {resource["id"]!r} asset "{relative}" does not exist in the project.'
            )
        return path.read_bytes()

    if role == "motion":
        if _is_npz_motion_source(resource["source"]):
            # Single-file NPZ motions travel as one asset; the embed runtime
            # re-registers them under a "<prefix>.npz" registry entry and keeps
            # their dof/body names, which the .bin folder layout drops.
            data = read_file(resource["source"])
            name = _source_file_name(resource, role)
            entries.append({
                "resourceId": resource["id"],
                "path": _asset_path(role, resource["id"], name, used),
                "name": name,
                "mimeType": "application/octet-stream",
                "role": role,
                "integrity": _sha256_integrity(data),
                "_buffer": data,
            })
            return entries
        base = resource["source"].rstrip("/")
        names = list(MOTION_REQUIRED_FILES)
        if (project_dir / base / "metadata.json").is_file():
            names.append("metadata.json")
        for required in names:
            try:
                data = read_file(f"{base}/{required}")
            except BotLabError as error:
                raise BotLabError(
                    f"{error} Motions package either as a single .npz archive "
                    "(source ending in .npz) or as a folder with the required "
                    f".bin files: {', '.join(MOTION_REQUIRED_FILES)}."
                ) from error
            entries.append({
                "resourceId": resource["id"],
                "path": _asset_path(role, resource["id"], required, used),
                "name": required,
                "mimeType": "application/json" if required == "metadata.json" else "application/octet-stream",
                "role": role,
                "integrity": _sha256_integrity(data),
                "_buffer": data,
            })
        return entries

    data = read_file(resource["source"])
    name = _source_file_name(resource, role)
    mime = MIME_BY_ROLE_SUFFIX.get(role, "application/octet-stream")
    entries.append({
        "resourceId": resource["id"],
        "path": _asset_path(role, resource["id"], name, used),
        "name": name,
        "mimeType": mime,
        "role": role,
        "integrity": _sha256_integrity(data),
        "_buffer": data,
    })
    return entries


def create_release_package(
    project: LoadedProject, metadata: dict | None = None, *, compression: str = "deflate"
) -> bytes:
    """Build release.zip bytes for a loaded project.

    Compiled projects carry a ``compiledController`` plan and run without
    Python. Projects whose action hook needs a live runtime (``modify=...``)
    travel as sources + manifest only: the embed runtime boots Pyodide and
    executes ``main.py`` itself.

    Robot/scene sources stay project-relative in the manifest; the embed
    runtime receives the project directory as its asset inventory (served at
    ``/project/`` by the CLI), so meshes referenced by the MJCF resolve through
    the inventory the same way they do for workspace projects.

    ``compression`` picks the zip method: ``"deflate"`` for publishable
    packages (``botlab build``) and ``"stored"`` for the debug server, which
    rebuilds the package in memory on every source change and cares about
    build latency, not package size. The archive payload and integrity
    digests are identical either way.
    """
    if compression not in ("deflate", "stored"):
        raise BotLabError(f"release compression must be 'deflate' or 'stored', got {compression!r}")
    compiled = None
    if project.compiled_plan is not None:
        from ._project import compiled_controller_envelope

        compiled = compiled_controller_envelope(
            project.compiled_plan, project.manifest, project.files
        )
        if compiled["projectDigest"] != project.project_digest:
            raise BotLabError(
                "Cannot create Python release package: compiled controller does not match "
                "the current manifest and source files."
            )

    used_paths = {f"project/{path}" for path in project.files}
    packed: list[dict] = []
    for resource in project.manifest["resources"]:
        packed.extend(_pack_resource(project.project_dir, resource, used_paths))
    _validate_onnx_assets(packed, compiled)
    total_declared = sum(entry["integrity"]["byteLength"] for entry in packed)
    if total_declared > MAX_PYTHON_RELEASE_PACKAGE_BYTES:
        raise BotLabError(
            "Cannot create Python release package: declared assets exceed the "
            f"{MAX_PYTHON_RELEASE_PACKAGE_BYTES}-byte limit."
        )

    source_manifest = [
        {
            "path": path,
            "archivePath": f"project/{path}",
            "integrity": _sha256_integrity(project.files[path].encode("utf-8")),
        }
        for path in sorted(project.files)
    ]

    normalized_metadata = {
        key: value.strip()
        for key, value in (metadata or {}).items()
        if isinstance(value, str) and value.strip()
    } or None

    config = {
        "format": "botlab-release",
        "version": 3,
        "runtime": "python",
        "createdAt": _release_timestamp(),
        "metadata": normalized_metadata,
        "project": {
            "manifest": project.manifest,
            "files": source_manifest,
            "projectDigest": project.project_digest,
            **({"compiledController": compiled} if compiled is not None else {}),
            "assetBase": "project",
            "assetPaths": project.asset_paths,
        },
        "assets": [
            {key: value for key, value in entry.items() if key != "_buffer"} for entry in packed
        ],
    }

    import io

    zip_method = zipfile.ZIP_STORED if compression == "stored" else zipfile.ZIP_DEFLATED
    zip_level = None if compression == "stored" else 6
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zip_method, compresslevel=zip_level) as archive:
        archive.writestr("release.json", json.dumps(config, indent=2))
        for path, source in sorted(project.files.items()):
            archive.writestr(f"project/{path}", source)
        for entry in packed:
            archive.writestr(entry["path"], entry["_buffer"])
    package = buffer.getvalue()
    if len(package) > MAX_PYTHON_RELEASE_PACKAGE_BYTES:
        raise BotLabError(
            f"Cannot create Python release package: package is {len(package)} bytes, "
            f"exceeding the {MAX_PYTHON_RELEASE_PACKAGE_BYTES}-byte limit."
        )
    return package


def _release_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _optional_onnx_module():
    """The onnx package when installed; validation degrades to a no-op without
    it (packaging itself only needs the model's bytes)."""
    try:
        import onnx
    except ImportError:
        return None
    return onnx


_GRAPH_FACTS_BY_DIGEST: dict[str, tuple[list[str], list[str]]] = {}


def _model_graph_facts(onnx, data: bytes, resource_id: str, digest: str | None):
    """Parse the model once per content digest: (input names, output names).

    The parse plus the external-weights check are the expensive part of
    validation (a full protobuf read); the debug server rebuilds packages on
    every source change, so successful facts are memoized by the digest the
    package entry already computed. Faults are never memoized — they raise
    again on the next build until the bytes change.
    """
    if digest is not None and digest in _GRAPH_FACTS_BY_DIGEST:
        return _GRAPH_FACTS_BY_DIGEST[digest]

    label = f'model resource "{resource_id}"'
    try:
        model = onnx.load_model_from_string(data)
    except Exception as error:
        raise BotLabError(
            f"[botlab] {label} is not a valid ONNX model ({error}). "
            "Export a single ONNX file from your training framework."
        ) from error

    # A model whose weights live in a sibling file loads fine wherever it was
    # exported but breaks every runtime that reads the packaged bytes alone.
    from onnx import external_data_helper

    external = [
        next(
            (item.value for item in tensor.external_data if item.key == "location"),
            tensor.name,
        )
        for tensor in model.graph.initializer
        if external_data_helper.uses_external_data(tensor)
    ]
    if external:
        raise BotLabError(
            f"[botlab] {label} stores weights in external file(s) "
            f"({', '.join(sorted(set(external)))}); the release ships only the "
            "model itself. Inline them before packaging, e.g.\n"
            "  model = onnx.load(path, load_external_data=False)\n"
            "  onnx.external_data_helper.convert_model_from_external_data(model)\n"
            "  onnx.save(model, path)"
        )

    facts = (
        [item.name for item in model.graph.input],
        [item.name for item in model.graph.output],
    )
    if digest is not None:
        _GRAPH_FACTS_BY_DIGEST[digest] = facts
    return facts


def _validate_onnx_model(
    onnx, data: bytes, resource_id: str, calls: list[dict], digest: str | None = None
) -> None:
    label = f'model resource "{resource_id}"'
    input_names, output_names = _model_graph_facts(onnx, data, resource_id, digest)
    if not calls:
        return
    for call in calls:
        declared_input = call.get("inputName")
        if declared_input is not None and declared_input not in input_names:
            raise BotLabError(
                f'[botlab] {label} call declares input_name "{declared_input}" but the '
                f'model inputs are {input_names}. Pass input_name matching the graph.'
            )
        declared_output = call.get("outputName")
        if declared_output is not None and declared_output not in output_names:
            raise BotLabError(
                f'[botlab] {label} call declares output_name "{declared_output}" but the '
                f'model outputs are {output_names}. Pass output_name matching the graph.'
            )


def _model_calls_in_node(data: dict):
    """Every (resourceId, call) pair a model-onnx node declares.

    Covers plain calls (the node's own resourceId plus inputName/outputName)
    and policy banks: each policy names its own resource and input/output
    names, and pipeline stages do the same per stage.
    """
    yield data
    policies = data.get("policies")
    if isinstance(policies, list):
        for policy in policies:
            if not isinstance(policy, dict):
                continue
            if isinstance(policy.get("resourceId"), str):
                yield policy
            pipeline = policy.get("pipeline")
            if isinstance(pipeline, list):
                for stage in pipeline:
                    if isinstance(stage, dict) and isinstance(stage.get("resourceId"), str):
                        yield stage


def _validate_onnx_assets(packed: list[dict], compiled: dict | None) -> None:
    """Deep-check packaged models when the onnx package is available.

    Faults caught here (unparseable files, externalized weights, graph/graph
    name mismatches) otherwise surface deep inside the browser runtime where
    they are much harder to attribute back to the asset.
    """
    onnx = _optional_onnx_module()
    if onnx is None:
        return
    calls: dict[str, list[dict]] = {}
    if compiled is not None:
        for node in compiled.get("plan", {}).get("nodes", []):
            if isinstance(node, dict) and node.get("type") == "model-onnx":
                data = node.get("data") or {}
                for call in _model_calls_in_node(data):
                    resource_id = call.get("resourceId")
                    if isinstance(resource_id, str):
                        calls.setdefault(resource_id, []).append(call)
    for entry in packed:
        if entry["role"] == "model":
            _validate_onnx_model(
                onnx,
                entry["_buffer"],
                entry["resourceId"],
                calls.get(entry["resourceId"], []),
                digest=entry["integrity"]["digest"],
            )
