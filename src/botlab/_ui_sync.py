"""Copy the pure Three.js native display into the small default wheel."""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath

from ._host import BotLabError

_HTML_REFERENCE = re.compile(r'''(?:src|href)=["']([^"']+)["']''')
_ASSET_REFERENCE = re.compile(r'''["'`]([^"'`\s]{2,240})["'`]''')
_CSS_URL = re.compile(r"url\(\s*['\"]?([^)'\"\s]+)")
_FORBIDDEN = re.compile(r"(?:\.wasm\b|WebAssembly|ort-wasm|pyodide|@mujoco/mujoco)", re.I)
_HEAVY_PATH = re.compile(r"(?:^|/)(?:wasm|pyodide|netron|monaco|ort(?:-|\.|/)|onnxruntime)", re.I)
_ANALYTICS_BLOCK = re.compile(
    r"[ \t]*<!-- Analytics: Umami -->[^\n]*\n[ \t]*<script[^>]*></script>\n?"
    r"|[ \t]*<!-- 百度自动推送[^\n]*-->\n[ \t]*<script>.*?</script>\n?",
    re.S,
)


def localize_index_html(html: str) -> str:
    """Keep the full editor's local-project mode when exported separately."""
    html = _ANALYTICS_BLOCK.sub("", html)
    if "BOTLAB_LOCAL_PROJECT" not in html:
        html = html.replace("</head>", "<script>window.BOTLAB_LOCAL_PROJECT=true</script></head>", 1)
    return html


def _resolve_reference(raw: str, current: str, available: set[str]) -> str | None:
    if (raw.startswith(("#", "//", "data:")) or
        re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw)):
        return None
    path = raw.split("?", 1)[0].split("#", 1)[0]
    if not path:
        return None
    parts = [] if path.startswith("/") else list(PurePosixPath(current).parent.parts)
    for part in path.lstrip("/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    candidate = "/".join(parts)
    return candidate if candidate in available else None


def _viewer_closure(dist: Path) -> list[str]:
    if not (dist / "viewer.html").is_file():
        raise BotLabError(
            f"{dist} is missing viewer.html. Run the BotLab viewer build first."
        )
    available = {
        path.relative_to(dist).as_posix()
        for path in dist.rglob("*") if path.is_file()
    }
    if any(_HEAVY_PATH.search(name) or ".wasm" in name.lower() for name in available):
        raise BotLabError("Viewer build contains WASM; only the pure Three.js display can enter the wheel.")
    seen: set[str] = set()
    pending = ["viewer.html"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        source = dist / name
        if source.is_symlink():
            raise BotLabError(f"Viewer build contains a symlink: {source}")
        seen.add(name)
        if source.suffix not in {".html", ".js", ".mjs", ".css"}:
            continue
        content = source.read_text(encoding="utf-8")
        if _FORBIDDEN.search(content):
            raise BotLabError(f"Viewer build references browser simulation runtime or WASM: {name}")
        refs = (_HTML_REFERENCE,) if source.suffix == ".html" else (
            (_CSS_URL,) if source.suffix == ".css" else (_ASSET_REFERENCE,)
        )
        for pattern in refs:
            for match in pattern.finditer(content):
                reference = _resolve_reference(match.group(1), name, available)
                if reference and reference not in seen:
                    pending.append(reference)
    return sorted(seen)


def validate_viewer_snapshot(path: Path) -> list[str]:
    """Validate a wheel input, including stale files from an old full snapshot."""
    if not (path / "index.html").is_file() or not (path / "embed.html").is_file():
        raise BotLabError(f"Viewer snapshot lacks index.html or embed.html under {path}.")
    files = [item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()]
    if any(_HEAVY_PATH.search(name) or ".wasm" in name.lower() for name in files):
        raise BotLabError("Viewer snapshot contains WASM; rerun `botlab ui sync`.")
    for name in files:
        item = path / name
        if item.is_symlink():
            raise BotLabError(f"Viewer snapshot contains a symlink: {name}")
        if item.suffix in {".html", ".js", ".mjs", ".css"} and _FORBIDDEN.search(item.read_text(encoding="utf-8")):
            raise BotLabError(f"Viewer snapshot references browser simulation runtime: {name}")
    return files


def sync_ui_snapshot(botlab_dir: Path, target_dir: Path) -> int:
    dist = botlab_dir / "dist-viewer"
    closure = _viewer_closure(dist)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for name in closure:
        source = dist / name
        destination = target_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    html = (target_dir / "viewer.html").read_bytes()
    (target_dir / "index.html").write_bytes(html)
    (target_dir / "embed.html").write_bytes(html)
    (target_dir / "viewer.html").unlink()
    validate_viewer_snapshot(target_dir)
    return len(closure) + 1
