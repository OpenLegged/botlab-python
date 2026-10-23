"""Deterministic model-service builds and atomic, persistent target deployments.

The HIL v1 artifact contains declared, embedded-weight ONNX models only. The
simulation and Python controller continue to execute on the host.
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
MAX_ARTIFACT_BYTES = MAX_MODEL_BYTES
MAX_PROJECT_STORAGE_BYTES = 512 * 1024 * 1024


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


def _model_content(project_dir, resource):
    candidate = _regular(project_dir / resource["source"])
    if not candidate.resolve().is_relative_to(project_dir) or not candidate.is_file():
        raise HilError(f"Model {resource['id']} must be a project-local file")
    size = candidate.stat().st_size
    if not 0 < size <= MAX_MODEL_BYTES:
        raise HilError("Each model must be nonempty and at most 32 MiB")
    content = candidate.read_bytes()
    require_embedded_tensors(content)
    return content


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
        models = []
        for resource in project.manifest["resources"]:
            if resource["kind"] != "model":
                continue
            content = _model_content(project_dir, resource)
            digest = hashlib.sha256(content).hexdigest()
            models.append({"id": resource["id"], "sha256": digest,
                           "path": f"models/{digest}.onnx", "size": len(content),
                           "config": resource.get("config", {})})
        if sorted(models, key=lambda item: item["id"]) != loaded["models"]:
            raise HilError("The local model files do not match the connected project version")
    except Exception as exc:
        raise HilError("Open the matching local project and sources for the connected version "
                       "before running it. " + str(exc)) from exc


def build_project(project_dir: Path, expected_sources: dict | None = None) -> dict:
    """Compile declarations and package models; never package arbitrary source files."""
    project_dir = Path(project_dir).resolve()
    project = _load_declarations(project_dir, expected_sources)
    identity = _project_identity(project_dir)
    models, files = [], {}
    total = 0
    for resource in project.manifest["resources"]:
        if resource["kind"] != "model":
            continue
        content = _model_content(project_dir, resource)
        digest = hashlib.sha256(content).hexdigest()
        path = f"models/{digest}.onnx"
        if path not in files:
            total += len(content)
            if total > MAX_ARTIFACT_BYTES:
                raise HilError("HIL v1 deployment exceeds the 32 MiB artifact limit")
            files[path] = base64.b64encode(content).decode("ascii")
        models.append({"id": resource["id"], "sha256": digest, "path": path, "size": len(content),
                       "config": resource.get("config", {})})
    manifest = {"schema": 1, "project": {"id": identity, "name": project.manifest["project"]["name"]},
                "execution_kind": "model-service", "backend": "onnxruntime",
                "project_digest": project.project_digest,
                "control_hz": project.manifest["runtime"]["controlHz"],
                "physics_hz": project.manifest["runtime"]["physicsHz"],
                "models": sorted(models, key=lambda item: item["id"])}
    manifest["build_id"] = build_digest(manifest)
    artifact = {"manifest": manifest, "files": files}
    validate_artifact(artifact)
    return artifact


def validate_manifest(value):
    if not isinstance(value, dict) or set(value) != {
        "schema", "project", "execution_kind", "backend", "models", "build_id",
        "project_digest", "control_hz", "physics_hz"
    }:
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
    models = value["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= 32:
        raise HilError("A model-service deployment requires 1–32 declared models")
    identifiers = set()
    paths = {}
    for model in models:
        if not isinstance(model, dict) or set(model) != {"id", "sha256", "path", "size", "config"}:
            raise HilError("Invalid deployment model")
        if not isinstance(model["config"], dict) or len(_canonical(model["config"])) > 16384:
            raise HilError("Invalid or oversized model interface configuration")
        identity = model["id"]
        if not isinstance(identity, str) or not 0 < len(identity) <= 1024 or identity in identifiers:
            raise HilError("Model resource IDs must be nonempty and unique")
        identifiers.add(identity)
        digest = _digest(model["sha256"])
        if model["path"] != f"models/{digest}.onnx":
            raise HilError("Unsafe deployment model path")
        size = model["size"]
        if type(size) is not int or not 0 < size <= MAX_MODEL_BYTES:
            raise HilError("Invalid deployment model size")
        if model["path"] in paths and paths[model["path"]] != size:
            raise HilError("Conflicting deployment model sizes")
        paths[model["path"]] = size
    if sum(paths.values()) > MAX_ARTIFACT_BYTES:
        raise HilError("HIL v1 deployment exceeds the 32 MiB artifact limit")
    if _digest(value["build_id"]) != build_digest(value):
        raise HilError("Deployment build SHA-256 verification failed")
    return value


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != {"manifest", "files"}:
        raise HilError("Deployment requires a manifest and files")
    manifest = validate_manifest(artifact["manifest"])
    files = artifact["files"]
    specs = {item["path"]: item for item in manifest["models"]}
    if not isinstance(files, dict) or set(files) != set(specs):
        raise HilError("Deployment files must exactly match the declared models")
    decoded = {}
    for path, item in specs.items():
        encoded = files[path]
        if not isinstance(encoded, str) or len(encoded) > ((item["size"] + 2) // 3) * 4:
            raise HilError("Deployment model size exceeds manifest")
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise HilError("Invalid deployment base64") from exc
        if len(content) != item["size"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise HilError("Deployment model SHA-256 verification failed")
        require_embedded_tensors(content)
        decoded[path] = content
    return manifest, decoded


class ProjectStore:
    """Files owned by this agent, with immutable releases and explicit selection."""

    def __init__(self, root: Path):
        self.root = _regular(Path(root).expanduser().absolute())
        self.projects = _regular(self.root / "projects")
        self.staging = _regular(self.root / "staging")
        self.projects.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id):
        return _regular(self.projects / _identity(project_id))

    def release_path(self, project_id, build_id):
        return _regular(self.project_path(project_id) / "releases" / _digest(build_id))

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
            for model in manifest["models"]:
                item = _regular(release / model["path"])
                if not item.is_file() or item.stat().st_size != model["size"]:
                    raise HilError("Deployed model is missing or has changed; redeploy the project")
                content = item.read_bytes()
                if hashlib.sha256(content).hexdigest() != model["sha256"]:
                    raise HilError("Deployed model SHA-256 verification failed")
                require_embedded_tensors(content)
        return manifest

    def select(self, project_id, build_id):
        self.manifest(project_id, build_id, verify_files=True)
        _atomic_json(self.project_path(project_id) / "current", build_id)

    def deploy(self, artifact):
        manifest, files = validate_artifact(artifact)
        project_id, build_id = manifest["project"]["id"], manifest["build_id"]
        release = self.release_path(project_id, build_id)
        if release.exists():
            self.manifest(project_id, build_id, verify_files=True)
            # A process can stop after the immutable directory rename and
            # before selecting the initial release. Retrying repairs metadata.
            project = self.project_path(project_id)
            if not _regular(project / "project.json").exists():
                _atomic_json(project / "project.json", manifest["project"])
            if not _regular(project / "current").exists():
                _atomic_json(project / "current", build_id)
            return {"project": self.summary(project_id), "already_deployed": True}
        usage = sum(p.stat().st_size for p in self.projects.rglob("*")
                    if p.is_file() and not p.is_symlink())
        if usage + sum(map(len, files.values())) > MAX_PROJECT_STORAGE_BYTES:
            raise HilError("Project storage exceeds 512 MiB; remove unused releases")
        staging = Path(tempfile.mkdtemp(prefix="deploy-", dir=self.staging))
        try:
            for path, content in files.items():
                destination = staging / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            _atomic_json(staging / "manifest.json", manifest)
            _atomic_json(staging / ".deployment.json", {"created_at": time.time()})
            release.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, release)
            _atomic_json(self.project_path(project_id) / "project.json", manifest["project"])
            _atomic_json(self.project_path(project_id) / "current", build_id)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {"project": self.summary(project_id), "already_deployed": False}

    def summary(self, project_id):
        folder = self.project_path(project_id)
        releases = []
        for release in sorted(_regular(folder / "releases").iterdir()):
            if not SHA256.fullmatch(release.name):
                continue
            manifest = self.manifest(project_id, release.name)
            deployed = _regular(release / ".deployment.json")
            releases.append({"build_id": release.name, "models": manifest["models"],
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
        return {"removed": True}

    def log(self, project_id, session, event, **fields):
        path = _regular(self.project_path(project_id) / "runs" / f"{_identity(session)}.jsonl")
        path.parent.mkdir(exist_ok=True)
        with path.open("ab") as stream:
            stream.write(_canonical({"time": time.time(), "event": event, **fields}) + b"\n")
