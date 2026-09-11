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

from ._host import MOTION_REQUIRED_FILES, BotLabError
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
    name = source.rstrip("/").split("/")[-1]
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
        base = resource["source"].rstrip("/")
        for name in MOTION_REQUIRED_FILES:
            data = read_file(f"{base}/{name}")
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


def create_release_package(project: LoadedProject, metadata: dict | None = None) -> bytes:
    """Build release.zip bytes for a loaded project.

    Compiled projects carry a ``compiledController`` plan and run without
    Python. Projects whose action hook needs a live runtime (``modify=...``)
    travel as sources + manifest only: the embed runtime boots Pyodide and
    executes ``main.py`` itself.

    Robot/scene sources stay project-relative in the manifest; the embed
    runtime receives the project directory as its asset inventory (served at
    ``/project/`` by the CLI), so meshes referenced by the MJCF resolve through
    the inventory the same way they do for workspace projects.
    """
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

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
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
