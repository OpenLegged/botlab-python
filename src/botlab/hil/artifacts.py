"""Deterministic project builds and atomic, content-addressed target deployments.

The HIL v1 deployment manifest declares the project's resources — models and
data assets alike — as content-addressed blobs. Blob bytes live exactly once in
a shared objects/ store; releases are manifest-only directories. Simulation and
the Python controller continue to execute on the host.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ._onnx import require_embedded_tensors
from .protocol import MAX_MODEL_BYTES, SHA256, HilError

PROJECT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
RESOURCE_KIND = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
STAGING_ID = re.compile(r"^[0-9a-f]{32}$")
# Models stay bounded per deployment because a session may load at most 32;
# data resources (scenes, meshes, sources, ...) are only counted, not loaded.
MAX_MODELS_PER_DEPLOYMENT = 32
MAX_RESOURCES_PER_DEPLOYMENT = 1024
MAX_CONFIG_BYTES = 16384
MANIFEST_KEYS = {
    "schema", "project", "execution_kind", "backend", "resources", "build_id",
    "project_digest", "control_hz", "physics_hz",
}
LEGACY_MANIFEST_KEYS = MANIFEST_KEYS - {"resources"} | {"models"}
MODEL_KINDS = ("model",)
# Declared-only sources (registry:// motions, procedural:// scene terrain) name
# data the declaring host generates or resolves; their parameters live in config,
# never in a packaged blob. Only project-local files ship.
PACKAGEABLE_SCHEME_SEP = "://"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def build_digest(manifest: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in manifest.items() if k != "build_id"})).hexdigest()


def _identity(value):
    if not isinstance(value, str) or not PROJECT_ID.fullmatch(value):
        raise HilError("Invalid project ID")
    return value


def _digest(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise HilError("Invalid build ID")
    return value


def _regular(path: Path):
    # Reject symlinks in every component, not just the final filename.
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise HilError(f"BotLab paths must not contain symlinks: {path.name}")
    return path


def _atomic_json(path: Path, value):
    _regular(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _project_identity(directory: Path) -> str:
    folder = _regular(directory / ".botlab")
    folder.mkdir(exist_ok=True)
    path = _regular(folder / "project.json")
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"schema": 1, "id": uuid.uuid4().hex}, stream)
    except FileExistsError:
        pass
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return _identity(value.get("id"))
    except (ValueError, AttributeError) as exc:
        raise HilError("Invalid .botlab/project.json; preserve or repair the project identity") from exc


def _check_expected_sources(expected_sources, actual_sources):
    if expected_sources is None:
        return
    if not isinstance(expected_sources, dict) or not 1 <= len(expected_sources) <= 128:
        raise HilError("Browser sources must contain 1–128 text files")
    total = 0
    for name, source in expected_sources.items():
        if not isinstance(name, str) or not 0 < len(name) <= 512 or not isinstance(source, str):
            raise HilError("Browser sources must map file names to text")
        try:
            total += len(source.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise HilError("Browser sources must be valid UTF-8 text") from exc
        if total > 4 * 1024 * 1024:
            raise HilError("Browser sources exceed the 4 MiB limit")
    def normalize(files):
        return {name: source.replace("\r\n", "\n").replace("\r", "\n")
                for name, source in files.items()}
    if normalize(expected_sources) != normalize(actual_sources):
        raise HilError("Browser sources differ from the project on disk. "
                       "Save changes to local project files and reload before continuing.")


def _load_declarations(project_dir: Path, expected_sources=None):
    """Read declarations without persisting bytecode or project identity."""
    from .._project import PROJECT_LOAD_LOCK, load_project

    project_dir = Path(project_dir).resolve()
    # The existing declaration compiler temporarily imports project modules.
    with PROJECT_LOAD_LOCK:
        # Timestamp .pyc caches can reuse an old declaration after same-size
        # edits within one second. Read through an empty cache namespace rather
        # than deleting project files, and restore interpreter settings even on
        # failed compilation. HIL declaration checks never persist bytecode.
        with tempfile.TemporaryDirectory(prefix="botlab-build-bytecode-") as bytecode:
            previous_prefix, previous_write = sys.pycache_prefix, sys.dont_write_bytecode
            try:
                sys.pycache_prefix = bytecode
                sys.dont_write_bytecode = True
                project = load_project(project_dir)
            finally:
                sys.pycache_prefix, sys.dont_write_bytecode = previous_prefix, previous_write
        _check_expected_sources(expected_sources, project.files)
    return project


def _resource_content(project_dir, resource):
    candidate = _regular(project_dir / resource["source"])
    if not candidate.resolve().is_relative_to(project_dir) or not candidate.is_file():
        raise HilError(f"Resource {resource['id']} must be a project-local file")
    return candidate.read_bytes()


def _packageable(resource):
    return PACKAGEABLE_SCHEME_SEP not in str(resource.get("source", ""))


def validate_loaded_project(project_dir: Path, loaded: dict, expected_sources: dict) -> None:
    """Check local execution against a selection without building or selecting it."""
    project_dir = Path(project_dir).resolve()
    try:
        identity = json.loads(_regular(project_dir / ".botlab" / "project.json").read_text())
        if not isinstance(identity, dict) or _identity(identity.get("id")) != loaded["project_id"]:
            raise HilError("The local project identity does not match the connected project")
        project = _load_declarations(project_dir, expected_sources)
        if project.project_digest != loaded["project_digest"]:
            raise HilError("The local sources do not match the connected project version")
        resources, _ = build_resource_entries(project_dir, project)
        if sorted(resources, key=lambda item: (item["kind"], item["id"])) != loaded["resources"]:
            raise HilError("The local resource files do not match the connected project version")
    except Exception as exc:
        raise HilError("Open the matching local project and sources for the connected version "
                       "before running it. " + str(exc)) from exc


def build_resource_entries(project_dir: Path, project) -> tuple[list[dict], dict]:
    """Content-addressed entries plus contents for every packageable resource.

    The same rule on both packaging channels: a resource ships when its source
    resolves to a project-local file; declared-only sources (registry://,
    procedural://) stay host-local.
    Source files never ship as blobs — the project digest already pins their
    content, and credentials must never ride a deployment.
    """
    project_dir = Path(project_dir).resolve()
    resources, contents = [], {}
    for resource in project.manifest["resources"]:
        if not _packageable(resource):
            continue
        content = _resource_content(project_dir, resource)
        if resource["kind"] in MODEL_KINDS:
            require_embedded_tensors(content)
        digest = hashlib.sha256(content).hexdigest()
        path = f"blobs/{digest}"
        contents[path] = content
        resources.append({"kind": resource["kind"], "id": resource["id"], "sha256": digest,
                          "path": path, "size": len(content),
                          "config": resource.get("config", {})})
    return resources, contents


def build_project(project_dir: Path, expected_sources: dict | None = None) -> dict:
    """Compile declarations and package every project-local resource."""
    project_dir = Path(project_dir).resolve()
    project = _load_declarations(project_dir, expected_sources)
    identity = _project_identity(project_dir)
    resources, contents = build_resource_entries(project_dir, project)
    resources = sorted(resources, key=lambda item: (item["kind"], item["id"]))
    manifest = {"schema": 1, "project": {"id": identity, "name": project.manifest["project"]["name"]},
                "execution_kind": "model-service", "backend": "onnxruntime",
                "project_digest": project.project_digest,
                "control_hz": project.manifest["runtime"]["controlHz"],
                "physics_hz": project.manifest["runtime"]["physicsHz"],
                "resources": resources}
    manifest["build_id"] = build_digest(manifest)
    artifact = {"manifest": manifest,
                "files": {path: base64.b64encode(content).decode("ascii")
                          for path, content in contents.items()}}
    validate_artifact(artifact)
    return artifact


def resources_of(manifest: dict) -> list[dict]:
    """Normalized resource entries; legacy model-only manifests gain a kind."""
    if "resources" in manifest:
        return manifest["resources"]
    return [{"kind": "model", **entry} for entry in manifest["models"]]


def validate_manifest(value):
    if not isinstance(value, dict):
        raise HilError("Invalid deployment manifest")
    legacy = "models" in value
    expected = LEGACY_MANIFEST_KEYS if legacy else MANIFEST_KEYS
    if set(value) != expected:
        raise HilError("Invalid deployment manifest")
    if value["schema"] != 1 or type(value["schema"]) is not int:
        raise HilError("Unsupported deployment schema")
    if value["execution_kind"] != "model-service" or value["backend"] != "onnxruntime":
        raise HilError("This deployment format supports model-service / onnxruntime only")
    _digest(value["project_digest"])
    for field in ("control_hz", "physics_hz"):
        number = value[field]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise HilError("Deployment frequencies must be finite positive numbers")
    project = value["project"]
    if not isinstance(project, dict) or set(project) != {"id", "name"}:
        raise HilError("Invalid deployment project")
    _identity(project["id"])
    if not isinstance(project["name"], str) or not 0 < len(project["name"]) <= 256:
        raise HilError("Project name must contain 1–256 characters")
    entries = value["models"] if legacy else value["resources"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= (
            MAX_MODELS_PER_DEPLOYMENT if legacy else MAX_RESOURCES_PER_DEPLOYMENT):
        raise HilError("A deployment requires 1–"
                       f"{MAX_MODELS_PER_DEPLOYMENT if legacy else MAX_RESOURCES_PER_DEPLOYMENT} declared resources")
    identifiers = set()
    paths = {}
    models = 0
    for entry in entries:
        keys = {"id", "sha256", "path", "size", "config"} if legacy else {
            "kind", "id", "sha256", "path", "size", "config"}
        if not isinstance(entry, dict) or set(entry) != keys:
            raise HilError("Invalid deployment resource")
        if not legacy and (not isinstance(entry["kind"], str)
                           or not RESOURCE_KIND.fullmatch(entry["kind"])):
            raise HilError("Deployment resource kind must be a lowercase identifier")
        if not isinstance(entry["config"], dict) or len(_canonical(entry["config"])) > MAX_CONFIG_BYTES:
            raise HilError("Invalid or oversized resource interface configuration")
        identity = entry["id"]
        if not isinstance(identity, str) or not 0 < len(identity) <= 1024 or identity in identifiers:
            raise HilError("Resource IDs must be nonempty and unique")
        identifiers.add(identity)
        digest = _digest(entry["sha256"])
        if not legacy:
            if entry["path"] != f"blobs/{digest}":
                raise HilError("Unsafe deployment resource path")
        elif entry["path"] != f"models/{digest}.onnx":
            raise HilError("Unsafe deployment model path")
        if not legacy and entry["kind"] in MODEL_KINDS:
            models += 1
            if models > MAX_MODELS_PER_DEPLOYMENT:
                raise HilError(f"A deployment declares at most {MAX_MODELS_PER_DEPLOYMENT} models")
        size = entry["size"]
        bounded = type(size) is int and 0 < size and (size <= MAX_MODEL_BYTES or not legacy)
        if not bounded:
            raise HilError("Invalid deployment resource size")
        if entry["path"] in paths and paths[entry["path"]] != size:
            raise HilError("Conflicting deployment resource sizes")
        paths[entry["path"]] = size
    if legacy and sum(paths.values()) > MAX_MODEL_BYTES:
        raise HilError("Legacy deployment exceeds the 32 MiB artifact limit")
    if _digest(value["build_id"]) != build_digest(value):
        raise HilError("Deployment build SHA-256 verification failed")
    return value


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != {"manifest", "files"}:
        raise HilError("Deployment requires a manifest and files")
    manifest = validate_manifest(artifact["manifest"])
    files = artifact["files"]
    specs = {item["path"]: item for item in resources_of(manifest)}
    if not isinstance(files, dict) or set(files) != set(specs):
        raise HilError("Deployment files must exactly match the declared resources")
    decoded = {}
    for path, item in specs.items():
        encoded = files[path]
        if not isinstance(encoded, str) or len(encoded) > ((item["size"] + 2) // 3) * 4:
            raise HilError("Deployment resource size exceeds manifest")
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise HilError("Invalid deployment base64") from exc
        if len(content) != item["size"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise HilError("Deployment resource SHA-256 verification failed")
        if item["kind"] in MODEL_KINDS:
            require_embedded_tensors(content)
        decoded[path] = content
    return manifest, decoded


class ProjectStore:
    """Files owned by this agent: shared content-addressed blobs, immutable
    manifest-only releases, and explicit selection. Blob bytes are stored once
    under objects/<sha256>.bin no matter how many releases reference them."""

    def __init__(self, root: Path):
        self.root = _regular(Path(root).expanduser().absolute())
        self.projects = _regular(self.root / "projects")
        self.objects = _regular(self.root / "objects")
        self.staging = _regular(self.root / "staging")
        for folder in (self.projects, self.objects, self.staging):
            folder.mkdir(parents=True, exist_ok=True)
        # A booting agent cannot have uploads in flight; abandoned staging from
        # a crashed process is garbage by definition.
        for leftover in self.staging.iterdir():
            shutil.rmtree(leftover, ignore_errors=True)
        self.stagings = {}
        self._migrate_legacy_releases()

    def _migrate_legacy_releases(self):
        """Move pre-objects release-local model files into the shared store.

        Filenames are their own content hashes (enforced at write time), so a
        rename completes the migration; anything else is left for verify_files
        to surface when that release is actually used.
        """
        for folder in self.projects.iterdir():
            if not PROJECT_ID.fullmatch(folder.name) or not folder.is_dir():
                continue
            for release in _safe_iterdir(folder / "releases"):
                for name in ("models", "blobs"):
                    payloads = release / name
                    if not payloads.is_dir():
                        continue
                    for payload in payloads.iterdir():
                        if payload.is_file() and SHA256.fullmatch(payload.stem):
                            destination = self.objects / f"{payload.stem}.bin"
                            if destination.exists():
                                payload.unlink()
                            else:
                                os.replace(payload, destination)
                    if not any(payloads.iterdir()):
                        payloads.rmdir()

    def project_path(self, project_id):
        return _regular(self.projects / _identity(project_id))

    def release_path(self, project_id, build_id):
        return _regular(self.project_path(project_id) / "releases" / _digest(build_id))

    def object_path(self, sha256):
        return _regular(self.objects / f"{_digest(sha256)}.bin")

    def blob_path(self, entry, release: Path) -> Path:
        """Resolve a manifest entry: shared object first, release-local fallback."""
        shared = self.object_path(entry["sha256"])
        if shared.is_file():
            return shared
        return _regular(release / entry["path"])

    def current(self, project_id):
        path = _regular(self.project_path(project_id) / "current")
        if not path.exists():
            raise HilError("Project has no deployed release")
        return _digest(json.loads(path.read_text()))

    def manifest(self, project_id, build_id=None, *, verify_files=False):
        build_id = build_id or self.current(project_id)
        release = self.release_path(project_id, build_id)
        path = _regular(release / "manifest.json")
        if not path.is_file():
            raise HilError("Project release is not deployed")
        manifest = validate_manifest(json.loads(path.read_text()))
        if manifest["project"]["id"] != project_id or manifest["build_id"] != build_id:
            raise HilError("Deployed manifest does not match its release directory")
        if verify_files:
            for entry in resources_of(manifest):
                item = self.blob_path(entry, release)
                if not item.is_file() or item.stat().st_size != entry["size"]:
                    raise HilError("Deployed resource is missing or has changed; redeploy the project")
                content = item.read_bytes()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise HilError("Deployed resource SHA-256 verification failed")
                if entry["kind"] in MODEL_KINDS:
                    require_embedded_tensors(content)
        return manifest

    def select(self, project_id, build_id):
        self.manifest(project_id, build_id, verify_files=True)
        _atomic_json(self.project_path(project_id) / "current", build_id)

    def _write_blob(self, content: bytes, sha256: str):
        digest = _digest(sha256)
        destination = self.object_path(digest)
        if destination.is_file() and destination.stat().st_size == len(content):
            return
        descriptor, temporary = tempfile.mkstemp(prefix=".blob-", dir=self.objects)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _repair_metadata(self, project_id, build_id, manifest):
        """A process can stop after the immutable directory rename and before
        selecting the initial release. Retrying repairs the metadata."""
        project = self.project_path(project_id)
        if not _regular(project / "project.json").exists():
            _atomic_json(project / "project.json", manifest["project"])
        if not _regular(project / "current").exists():
            _atomic_json(project / "current", build_id)

    def _finish_release(self, staging: Path, manifest):
        project_id, build_id = manifest["project"]["id"], manifest["build_id"]
        release = self.release_path(project_id, build_id)
        _atomic_json(staging / ".deployment.json", {"created_at": time.time()})
        release.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, release)
        _atomic_json(self.project_path(project_id) / "project.json", manifest["project"])
        _atomic_json(self.project_path(project_id) / "current", build_id)

    def stage(self, manifest, owner):
        """Open a staged deployment: validate the manifest, list missing blobs."""
        manifest = validate_manifest(manifest)
        if self.release_path(manifest["project"]["id"], manifest["build_id"]).exists():
            self.manifest(manifest["project"]["id"], manifest["build_id"], verify_files=True)
            self._repair_metadata(manifest["project"]["id"], manifest["build_id"], manifest)
            return {"already_deployed": True, "project": self.summary(manifest["project"]["id"])}
        missing = []
        for entry in resources_of(manifest):
            shared = self.object_path(entry["sha256"])
            if not shared.is_file() or shared.stat().st_size != entry["size"]:
                missing.append({"sha256": entry["sha256"], "size": entry["size"]})
        staging_id = uuid.uuid4().hex
        staging = _regular(self.staging / staging_id)
        staging.mkdir()
        _atomic_json(staging / "manifest.json", manifest)
        self.stagings[staging_id] = {"owner": owner, "manifest": manifest, "received": {}}
        return {"staging": staging_id, "missing": missing}

    def put_blob(self, staging_id, sha256, content: bytes, owner):
        """Append one chunk to a staged blob; completes it into objects/ on the
        final chunk. Blobs are verified before they ever enter the store."""
        record = self._staging_record(staging_id, owner)
        manifest, received = record["manifest"], record["received"]
        entry = next((item for item in resources_of(manifest) if item["sha256"] == sha256), None)
        if entry is None:
            raise HilError("Blob is not declared by the staged manifest")
        digest = _digest(sha256)
        done = received.get(digest, 0) + len(content)
        if done > entry["size"]:
            raise HilError("Blob chunk overflows the declared size")
        part = _regular(self.staging / staging_id / f"{digest}.part")
        with part.open("ab") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        received[digest] = done
        if done < entry["size"]:
            return {"received": done, "complete": False}
        if hashlib.sha256(part.read_bytes()).hexdigest() != digest:
            raise HilError("Blob SHA-256 verification failed")
        os.replace(part, self.object_path(digest))
        return {"received": done, "complete": True}

    def commit(self, staging_id, owner):
        """Turn a complete staging into an immutable release and select it."""
        record = self._staging_record(staging_id, owner)
        staging = _regular(self.staging / staging_id)
        manifest = validate_manifest(json.loads((staging / "manifest.json").read_text()))
        for entry in resources_of(manifest):
            shared = self.object_path(entry["sha256"])
            if not shared.is_file() or shared.stat().st_size != entry["size"]:
                raise HilError(f"Blob {entry['sha256']} was not fully uploaded")
        release = self.release_path(manifest["project"]["id"], manifest["build_id"])
        already = release.exists()
        if already:
            self.manifest(manifest["project"]["id"], manifest["build_id"], verify_files=True)
            shutil.rmtree(staging)
        else:
            self._finish_release(staging, manifest)
        self.stagings.pop(staging_id, None)
        return {"project": self.summary(manifest["project"]["id"]), "already_deployed": already}

    def cancel(self, staging_id, owner):
        self._staging_record(staging_id, owner)
        shutil.rmtree(_regular(self.staging / staging_id))
        self.stagings.pop(staging_id, None)
        return {"cancelled": True}

    def cancel_stagings(self, owner):
        for staging_id, record in list(self.stagings.items()):
            if record["owner"] is owner:
                shutil.rmtree(_regular(self.staging / staging_id), ignore_errors=True)
                self.stagings.pop(staging_id, None)

    def _staging_record(self, staging_id, owner):
        if not isinstance(staging_id, str) or not STAGING_ID.fullmatch(staging_id):
            raise HilError("Invalid staging ID")
        record = self.stagings.get(staging_id)
        if record is None or not (self.staging / staging_id).is_dir():
            raise HilError("Unknown or expired deployment staging; stage again")
        if record["owner"] is not owner:
            raise HilError("Deployment staging is owned by another connection")
        return record

    def read_blob(self, sha256, offset=0, length=None):
        """Serve blob bytes for a release-referenced sha to a connected host."""
        digest = _digest(sha256)
        if not self._is_referenced(digest):
            raise HilError("Blob is not referenced by any deployed release")
        path = self.object_path(digest)
        if not path.is_file():
            raise HilError("Blob is missing from the target store")
        total = path.stat().st_size
        if type(offset) is not int or offset < 0 or offset > total:
            raise HilError("Blob offset must be within the blob")
        if length is None:
            length = total - offset
        if type(length) is not int or length < 0 or offset + length > total:
            raise HilError("Blob length must fit within the blob")
        with path.open("rb") as stream:
            stream.seek(offset)
            return {"total": total, "data": stream.read(length)}

    def deploy(self, artifact):
        """Single-frame deployment: validate, store blobs, publish the release."""
        manifest, files = validate_artifact(artifact)
        for content in files.values():
            self._write_blob(content, hashlib.sha256(content).hexdigest())
        staging = Path(tempfile.mkdtemp(prefix="deploy-", dir=self.staging))
        try:
            _atomic_json(staging / "manifest.json", manifest)
            already = self.release_path(manifest["project"]["id"], manifest["build_id"]).exists()
            if already:
                self.manifest(manifest["project"]["id"], manifest["build_id"], verify_files=True)
                self._repair_metadata(manifest["project"]["id"], manifest["build_id"], manifest)
            else:
                self._finish_release(staging, manifest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {"project": self.summary(manifest["project"]["id"]), "already_deployed": already}

    def summary(self, project_id):
        folder = self.project_path(project_id)
        releases = []
        for release in sorted(_safe_iterdir(folder / "releases")):
            if not SHA256.fullmatch(release.name):
                continue
            manifest = self.manifest(project_id, release.name)
            deployed = _regular(release / ".deployment.json")
            releases.append({"build_id": release.name, "resources": resources_of(manifest),
                             "created_at": json.loads(deployed.read_text())["created_at"]})
        releases.sort(key=lambda item: item["created_at"], reverse=True)
        current = self.current(project_id)
        manifest = self.manifest(project_id, current)
        return {"id": project_id, "name": manifest["project"]["name"],
                "current_build_id": current, "releases": releases, "state": "deployed"}

    def list(self):
        return [self.summary(folder.name) for folder in sorted(self.projects.iterdir())
                if PROJECT_ID.fullmatch(folder.name) and folder.is_dir()]

    def remove(self, project_id, build_id=None):
        folder = self.project_path(project_id)
        if not folder.is_dir():
            raise HilError("Project is not deployed")
        if build_id is None:
            shutil.rmtree(folder)
        else:
            self.manifest(project_id, build_id)
            current = self.current(project_id)
            releases = [item for item in self.summary(project_id)["releases"]
                        if item["build_id"] != build_id]
            if not releases:
                shutil.rmtree(folder)
            else:
                if current == build_id:
                    _atomic_json(folder / "current", releases[0]["build_id"])
                shutil.rmtree(self.release_path(project_id, build_id))
        self._collect_objects()
        return {"removed": True}

    def _referenced_shas(self):
        referenced = set()
        for folder in self.projects.iterdir():
            if not PROJECT_ID.fullmatch(folder.name) or not folder.is_dir():
                continue
            for release in _safe_iterdir(folder / "releases"):
                manifest = _read_manifest(release)
                if manifest is not None:
                    referenced.update(item["sha256"] for item in resources_of(manifest))
        return referenced

    def _is_referenced(self, digest):
        return digest in self._referenced_shas()

    def _collect_objects(self):
        """Delete shared blobs no remaining release references."""
        referenced = self._referenced_shas()
        for blob in _safe_iterdir(self.objects):
            if blob.is_file() and blob.suffix == ".bin" and blob.stem not in referenced:
                blob.unlink()

    def log(self, project_id, session, event, **fields):
        path = _regular(self.project_path(project_id) / "runs" / f"{_identity(session)}.jsonl")
        path.parent.mkdir(exist_ok=True)
        with path.open("ab") as stream:
            stream.write(_canonical({"time": time.time(), "event": event, **fields}) + b"\n")


def _safe_iterdir(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(folder.iterdir())


def _read_manifest(release: Path):
    path = release / "manifest.json"
    if not path.is_file():
        return None
    try:
        return validate_manifest(json.loads(path.read_text()))
    except (HilError, ValueError):
        return None
