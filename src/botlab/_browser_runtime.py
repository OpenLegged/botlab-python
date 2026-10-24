"""Versioned, separately distributed browser editor and simulation runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from . import __version__
from ._errors import BotLabError
from ._ui_sync import localize_index_html

MANIFEST = "botlab-browser-runtime.json"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_FILES = 10000
_EXCLUDED = {"d", "models", "templates"}


def browser_runtime_url() -> str:
    return (
        "https://github.com/OpenLegged/botlab-python/releases/download/"
        f"v{__version__}/botlab-browser-runtime-{__version__}.zip"
    )


def browser_runtime_cache_dir() -> Path:
    override = os.environ.get("BOTLAB_RUNTIME_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "botlab/browser"
    return root / __version__


def _safe_relative(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and ":" not in name
        and not any(part in ("", ".", "..") for part in name.split("/"))
        and path.as_posix() == name
        and not name.endswith("/")
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for(files: list[tuple[str, Path]]) -> dict:
    index = localize_index_html(next(source for name, source in files if name == "index.html").read_text(encoding="utf-8")).encode("utf-8")
    return {
        "format": "botlab-browser-runtime",
        "version": __version__,
        "files": {
            name: {
                "size": len(index) if name == "index.html" else source.stat().st_size,
                "sha256": hashlib.sha256(index).hexdigest() if name == "index.html" else _hash_file(source),
            }
            for name, source in files
        },
    }


def export_browser_runtime(botlab_dir: Path, archive_path: Path) -> int:
    """Archive a built BotLab editor/runtime outside the wheel."""
    dist = botlab_dir / "dist"
    if not (dist / "index.html").is_file() or not (dist / "embed.html").is_file():
        raise BotLabError(f"{dist} needs index.html and embed.html; run `npm run build` first.")
    files: list[tuple[str, Path]] = []
    for source in sorted(dist.rglob("*")):
        if not source.is_file() or source.relative_to(dist).parts[0] in _EXCLUDED:
            continue
        if source.is_symlink():
            raise BotLabError(f"Browser runtime source must not be a symlink: {source}")
        name = source.relative_to(dist).as_posix()
        if not _safe_relative(name) or name == MANIFEST:
            raise BotLabError(f"Invalid browser runtime path: {name}")
        files.append((name, source))
    manifest = _manifest_for(files)
    if len(files) > MAX_FILES or sum(info["size"] for info in manifest["files"].values()) > MAX_UNPACKED_BYTES:
        raise BotLabError("Browser runtime exceeds archive limits.")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        for name, source in files:
            if name == "index.html":
                archive.writestr(name, localize_index_html(source.read_text(encoding="utf-8")))
            else:
                archive.write(source, name)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        archive_path.unlink()
        raise BotLabError("Browser runtime archive exceeds download limit.")
    return len(files)


def _validate_manifest(raw: bytes) -> dict[str, dict]:
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BotLabError(f"Browser runtime manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != "botlab-browser-runtime":
        raise BotLabError("Archive is not a BotLab browser runtime.")
    if manifest.get("version") != __version__:
        raise BotLabError(
            f"Browser runtime version {manifest.get('version')!r} does not match "
            f"botlab-python {__version__}; fetch v{__version__}."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_FILES:
        raise BotLabError("Browser runtime manifest has invalid file list.")
    if "index.html" not in files or "embed.html" not in files:
        raise BotLabError("Browser runtime is missing index.html or embed.html.")
    total = 0
    for name, info in files.items():
        if (not isinstance(name, str) or not _safe_relative(name) or name == MANIFEST
            or not isinstance(info, dict) or type(info.get("size")) is not int
            or info["size"] < 0 or info["size"] > MAX_UNPACKED_BYTES
            or not isinstance(info.get("sha256"), str)
            or len(info["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in info["sha256"])):
            raise BotLabError(f"Browser runtime manifest contains invalid file: {name!r}.")
        total += info["size"]
    if total > MAX_UNPACKED_BYTES:
        raise BotLabError("Browser runtime exceeds unpacked size limit.")
    return files


def _validate_cache(path: Path) -> bool:
    try:
        if path.is_symlink() or (path / MANIFEST).is_symlink():
            return False
        files = _validate_manifest((path / MANIFEST).read_bytes())
        actual = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
        if actual != set(files) | {MANIFEST}:
            return False
        if any((path / name).is_symlink() for name in actual):
            return False
        return all((path / name).stat().st_size == info["size"]
                   and _hash_file(path / name) == info["sha256"]
                   for name, info in files.items())
    except (OSError, BotLabError):
        return False


def install_browser_runtime(archive_path: Path) -> Path:
    """Validate and atomically install an offline archive in the version cache."""
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise BotLabError("Browser runtime archive exceeds download limit.")
    target = browser_runtime_cache_dir()
    if target.is_symlink():
        raise BotLabError(f"Browser runtime cache path is a symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{__version__}-", dir=target.parent))
    try:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if (len(names) > MAX_FILES + 1 or len(names) != len(set(names))
                    or MANIFEST not in names or any(not _safe_relative(name) for name in names)):
                    raise BotLabError("Browser runtime archive contains unsafe or duplicate paths.")
                if archive.getinfo(MANIFEST).file_size > 2 * 1024 * 1024:
                    raise BotLabError("Browser runtime manifest exceeds size limit.")
                files = _validate_manifest(archive.read(MANIFEST))
                if set(names) != set(files) | {MANIFEST}:
                    raise BotLabError("Browser runtime archive differs from its manifest.")
                if sum(info.file_size for info in infos) > MAX_UNPACKED_BYTES + 2 * 1024 * 1024:
                    raise BotLabError("Browser runtime archive exceeds unpacked size limit.")
                for info in infos:
                    mode = info.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                        raise BotLabError(f"Browser runtime archive contains a non-file: {info.filename}")
                    if info.filename == MANIFEST:
                        continue
                    expected = files[info.filename]
                    if info.file_size != expected["size"]:
                        raise BotLabError(f"Browser runtime size mismatch: {info.filename}")
                    destination = staging / info.filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(info) as source, destination.open("wb") as output:
                        copied = 0
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            copied += len(chunk)
                            if copied > expected["size"]:
                                raise BotLabError(f"Browser runtime expanded beyond declared size: {info.filename}")
                            digest.update(chunk)
                            output.write(chunk)
                    if copied != expected["size"] or digest.hexdigest() != expected["sha256"]:
                        raise BotLabError(f"Browser runtime checksum mismatch: {info.filename}")
                (staging / MANIFEST).write_bytes(archive.read(MANIFEST))
        except (zipfile.BadZipFile, RuntimeError, OSError) as error:
            raise BotLabError(f"Could not install browser runtime archive {archive_path}: {error}") from error
        backup = target.parent / f".{__version__}-backup-{uuid.uuid4().hex}"
        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except OSError:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def fetch_browser_runtime(*, url: str | None = None, archive: Path | None = None) -> Path:
    if archive is not None:
        return install_browser_runtime(archive)
    source_url = url or browser_runtime_url()
    cache_root = browser_runtime_cache_dir().parent
    cache_root.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".download-", suffix=".zip", dir=cache_root)
    os.close(fd)
    download = Path(path)
    try:
        try:
            with urllib.request.urlopen(source_url, timeout=60) as source, download.open("wb") as output:
                copied = 0
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    copied += len(chunk)
                    if copied > MAX_ARCHIVE_BYTES:
                        raise BotLabError("Browser runtime download exceeds archive limit.")
                    output.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BotLabError(
                f"Could not download browser runtime v{__version__} from {source_url}: {error}. "
                "Use `botlab ui fetch --archive <path>` for an offline archive."
            ) from error
        return install_browser_runtime(download)
    finally:
        download.unlink(missing_ok=True)


def ensure_browser_runtime() -> Path:
    target = browser_runtime_cache_dir()
    if _validate_cache(target):
        return target
    try:
        return fetch_browser_runtime()
    except BotLabError as error:
        raise BotLabError(
            f"Browser runtime v{__version__} is unavailable: {error} "
            "Run `botlab ui fetch --archive <path>` or `botlab ui sync --botlab <checkout>`."
        ) from error
