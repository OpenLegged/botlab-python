"""Assemble a self-contained static site around a built release.zip.

``botlab build --site <dir>`` turns a project into a directory that can be
push to any static host (GitHub Pages included) with zero configuration:

    <dir>/
    ├── index.html            ← the UI snapshot's embed.html, renamed
    ├── assets/…              ← the embed runtime closure (static JS/WASM scan)
    ├── wasm/obj-parser/…     ← the module-string referenced terrain runtime
    ├── project/…             ← declared source files and asset dependencies
    └── release.zip           ← the release package, fetched by './release.zip'

Every runtime URL is page-relative, so the site works from a sub-path such as
``user.github.io/<repo>/`` and needs no COOP/COEP headers (the embed runtime
is single-threaded). The assembly rules mirror BotLab's
``scripts/build-embed.mjs`` — ported here so users do not need a BotLab
checkout.
"""

from __future__ import annotations

import io
import json
import posixpath
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from ._browser_runtime import ensure_browser_runtime
from ._host import BotLabError

MAX_RELEASE_ZIP_BYTES = 200 * 1024 * 1024
MAX_RELEASE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_QUOTED_STRING = re.compile(r"[\"'`]([^\"'`\s]{2,160})[\"'`]")
_HTML_REFERENCE = re.compile(r'(?:src|href)="([^"]+)"')


def _is_safe_site_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) > 0
        and not value.startswith("/")
        and "\\" not in value
        and not _SCHEME.match(value)
        and all(part not in ("", ".", "..") for part in value.split("/"))
    )


def _require_safe_site_path(value: Any, label: str) -> str:
    if not _is_safe_site_path(value):
        raise BotLabError(f'Release {label} is not a safe site-relative path: "{value}".')
    return value


def _resolve_relative_reference(raw: str, base_dir: str) -> str | None:
    clean = raw.split("?")[0].split("#")[0]
    if not clean or "${" in clean or _SCHEME.match(clean):
        return None
    if clean.startswith("/"):
        return clean.lstrip("/")
    resolved = base_dir.split("/") if base_dir else []
    for segment in clean.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not resolved:
                return None
            resolved.pop()
        else:
            resolved.append(segment)
    return "/".join(resolved)


def _scan_chunk_references(source: str, chunk_file: str, file_index: set[str]) -> set[str]:
    base_dir = posixpath.dirname(chunk_file)
    references: set[str] = set()
    for match in _QUOTED_STRING.finditer(source):
        resolved = _resolve_relative_reference(match.group(1), "" if base_dir == "." else base_dir)
        if resolved and resolved in file_index:
            references.add(resolved)
    return references


def _collect_runtime_closure(snapshot_dir: Path, file_index: set[str]) -> set[str]:
    html = (snapshot_dir / "embed.html").read_text(encoding="utf-8")
    closure: set[str] = set()
    queue: list[str] = []
    for match in _HTML_REFERENCE.finditer(html):
        resolved = _resolve_relative_reference(match.group(1), "")
        if not resolved:
            continue
        closure.add(resolved)
        if resolved.endswith(".js"):
            queue.append(resolved)
    while queue:
        file = queue.pop()
        path = snapshot_dir / file
        if not path.is_file():
            raise BotLabError(f'UI snapshot closure references missing file "{file}".')
        for reference in _scan_chunk_references(
            path.read_text(encoding="utf-8"), file, file_index
        ):
            if reference in closure:
                continue
            closure.add(reference)
            if reference.endswith(".js"):
                queue.append(reference)
    # The obj-parser worker is referenced by a module string constant
    # ('wasm/obj-parser/objParser.js'), not by the chunk closure; carry it
    # explicitly and fail loudly when the snapshot lacks it.
    if not (snapshot_dir / "wasm/obj-parser/objParser.js").is_file():
        raise BotLabError("UI snapshot is missing wasm/obj-parser/objParser.js (terrain runtime).")
    closure.add("wasm/obj-parser/objParser.js")
    closure.add("wasm/obj-parser/objParser.wasm")
    return closure


def _packaged_targets(config: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for asset in config.get("assets") or []:
        if not isinstance(asset, dict) or not isinstance(asset.get("resourceId"), str):
            continue
        target = asset.get("target") if isinstance(asset.get("target"), dict) else {}
        kind = target.get("kind", "resourceSource")
        if kind == "resourceConfig" and target.get("key") in (
            "terrainMeshUrl",
            "terrainAssetPath",
        ):
            targets.add(f'{asset["resourceId"]}\0resourceConfig\0{target["key"]}')
        else:
            targets.add(f'{asset["resourceId"]}\0resourceSource\0')
    return targets


def _add_local_source_root(roots: set[str], source: str, asset_base: str, label: str) -> None:
    scheme = _SCHEME.match(source)
    if scheme:
        raise BotLabError(
            f'Release {label} uses unarchived {scheme.group(0)}// source "{source}"; '
            "standalone output must not depend on external or page-lifetime URLs."
        )
    _require_safe_site_path(source, label)
    if source.startswith("assets/") and asset_base:
        return
    directory = posixpath.dirname(source)
    # A root-level project asset lives under assetBase itself. Python's
    # dirname('robot.xml') is '' (Node's is '.'), not an external site root.
    roots.add(directory or asset_base or source)


def collect_site_asset_roots(config: dict[str, Any]) -> set[str]:
    """Turn release resource declarations into site-relative directory roots.

    Packaged dependencies are skipped (release.zip restores them) and
    procedural sources are declarations, not directories; both mirror
    ``collectSiteAssetRoots`` in BotLab's build-embed-plan.mjs.
    """
    project = config.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("manifest"), dict):
        raise BotLabError("release.json project manifest is missing or invalid.")
    asset_base = project.get("assetBase", "")
    if asset_base:
        asset_base = _require_safe_site_path(asset_base.rstrip("/"), "assetBase")
    asset_paths = project.get("assetPaths")
    if asset_paths is not None:
        if not isinstance(asset_paths, list):
            raise BotLabError("Release assetPaths must be an array.")
        for path in asset_paths:
            _require_safe_site_path(path, "assetPaths entry")

    resources = project["manifest"].get("resources")
    if not isinstance(resources, list):
        raise BotLabError("release.json resources are missing or invalid.")
    packaged = _packaged_targets(config)
    roots: set[str] = set()
    if asset_base:
        roots.add(asset_base)

    for resource in resources:
        if (
            not isinstance(resource, dict)
            or not isinstance(resource.get("id"), str)
            or not resource["id"]
            or not isinstance(resource.get("kind"), str)
            or not isinstance(resource.get("source"), str)
            or not resource["source"]
        ):
            raise BotLabError("Release contains an invalid resource declaration.")
        source_packaged = f'{resource["id"]}\0resourceSource\0' in packaged
        scheme = _SCHEME.match(resource["source"])
        if not source_packaged and resource["kind"] != "sensor":
            if scheme and scheme.group(0)[:-1].lower() == "procedural":
                if resource["kind"] != "scene":
                    raise BotLabError(
                        f'Release resource "{resource["id"]}" uses procedural:// outside a scene.'
                    )
            else:
                _add_local_source_root(
                    roots, resource["source"], asset_base, f'resource "{resource["id"]}" source'
                )
        config_value = resource.get("config") if isinstance(resource.get("config"), dict) else {}
        if resource["kind"] != "scene" or config_value.get("terrainKind") != "custom-mesh":
            continue
        source_key = (
            "terrainMeshUrl"
            if isinstance(config_value.get("terrainMeshUrl"), str)
            and config_value["terrainMeshUrl"]
            else "terrainAssetPath"
        )
        mesh_source = config_value.get(source_key)
        if not isinstance(mesh_source, str) or not mesh_source:
            raise BotLabError(f'Release custom-mesh scene "{resource["id"]}" has no mesh source.')
        config_packaged = (
            f'{resource["id"]}\0resourceConfig\0terrainMeshUrl' in packaged
            or f'{resource["id"]}\0resourceConfig\0terrainAssetPath' in packaged
        )
        if not config_packaged:
            _add_local_source_root(
                roots, mesh_source, asset_base, f'scene "{resource["id"]}" mesh source'
            )
    return roots


def read_release_config(package: bytes) -> dict[str, Any]:
    if not 0 < len(package) <= MAX_RELEASE_ZIP_BYTES:
        raise BotLabError(f"Release package must be between 1 and {MAX_RELEASE_ZIP_BYTES} bytes.")
    try:
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            total_uncompressed = sum(info.file_size for info in archive.infolist())
            if total_uncompressed > MAX_RELEASE_UNCOMPRESSED_BYTES:
                raise BotLabError("Release total uncompressed content exceeds the release limit.")
            raw = archive.read("release.json")
    except BotLabError:
        raise
    except KeyError as error:
        raise BotLabError(
            "release.zip is missing release.json; it is not a BotLab Python release."
        ) from error
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise BotLabError(f"release.zip could not be read: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BotLabError(f"release.json is not valid JSON: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("format") != "botlab-release"
        or value.get("runtime") != "python"
    ):
        raise BotLabError("release.json is not a BotLab Python release package (format/runtime mismatch).")
    if value.get("version") != 3:
        raise BotLabError(
            f'release.json version {value.get("version")} is not supported for standalone builds.'
        )
    controller = (value.get("project") or {}).get("compiledController")
    if not isinstance(controller, dict) or controller.get("kind") != "policy-controller-plan":
        raise BotLabError(
            "release.zip carries no compiled controller plan. Re-run the project so a policy "
            "controller plan is compiled, then build again — the standalone runtime cannot "
            "execute a live-Python (modify=…) controller."
        )
    return value


def _contained_file(root: Path, relative: str, label: str) -> Path:
    if not _is_safe_site_path(relative):
        raise BotLabError(f'{label} is unsafe: "{relative}".')
    candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BotLabError(f'{label} escapes its root: "{relative}".')
    return resolved


def _iter_project_files(project_dir: Path) -> list[str]:
    return sorted(
        item.relative_to(project_dir).as_posix()
        for item in project_dir.rglob("*")
        if item.is_file()
    )



def _site_project_files(config: dict[str, Any]) -> list[str]:
    """Use the release inventory; never discover files by walking the project."""
    project = config["project"]
    sources = project.get("files")
    assets = project.get("assetPaths")
    if not isinstance(sources, list) or not isinstance(assets, list):
        raise BotLabError("Release project files and assetPaths must be arrays; rebuild the release.")
    paths = set()
    for entry in sources:
        if not isinstance(entry, dict):
            raise BotLabError("Release project files entry must be an object.")
        paths.add(_require_safe_site_path(entry.get("path"), "project files entry"))
    for path in assets:
        paths.add(_require_safe_site_path(path, "assetPaths entry"))
    return sorted(paths)


def build_site_directory(project_dir: Path, package: bytes, out_dir: Path) -> dict[str, Any]:
    """Write the standalone site for one already-built release package."""
    snapshot_dir = ensure_browser_runtime()
    if not (snapshot_dir / "embed.html").is_file():
        raise BotLabError(
            f"Browser runtime is missing embed.html under {snapshot_dir}; "
            "run `botlab ui fetch --archive <archive.zip>`."
        )
    if out_dir.exists():
        raise BotLabError(f"Output directory already exists: {out_dir} (remove it or pass another).")

    config = read_release_config(package)
    project_name = ((config.get("project") or {}).get("manifest") or {}).get("project", {}).get(
        "name", "(unnamed)"
    )
    asset_base = (config.get("project") or {}).get("assetBase") or "project"

    snapshot_files = _iter_project_files(snapshot_dir)
    file_index = set(snapshot_files)
    closure = _collect_runtime_closure(snapshot_dir, file_index)

    roots = collect_site_asset_roots(config)
    site_roots = sorted(roots - {asset_base})
    if site_roots:
        raise BotLabError(
            "Release references shared site assets outside the project "
            f"({', '.join(site_roots)}). Standalone output must be self-contained; "
            "copy those assets into the project (assets/…) and reference them "
            "project-relatively, then build again."
        )

    copies: list[tuple[Path, Path]] = []
    seen_targets: set[Path] = set()

    def add_copy(source: Path, target_relative: str) -> None:
        target = _contained_file(out_dir, target_relative, "Output path")
        if target in seen_targets:
            return
        seen_targets.add(target)
        copies.append((source, target))

    add_copy(snapshot_dir / "embed.html", "index.html")
    for file in sorted(closure):
        add_copy(_contained_file(snapshot_dir, file, "UI snapshot input"), file)
    # Preserve paths within the declared dependency closure so MJCF includes
    # resolve, without walking venvs, logs, downloads or previous site outputs.
    for relative in _site_project_files(config):
        source = _contained_file(project_dir, relative, "Project asset")
        if not source.is_file():
            raise BotLabError(f'Project asset is missing or is not a file: "{relative}".')
        add_copy(source, f"{asset_base}/{relative}")

    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    release_target = _contained_file(out_dir, "release.zip", "Output path")
    release_target.write_bytes(package)

    total_bytes = sum(target.stat().st_size for _, target in copies) + len(package)
    return {
        "out_dir": out_dir,
        "project": project_name,
        "files": len(copies) + 1,
        "total_bytes": total_bytes,
    }
