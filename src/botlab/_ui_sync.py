"""Sync the BotLab UI snapshot from a BotLab build.

``botlab run`` serves the full BotLab site out of one snapshot: the editor
(``index.html``, the default debug surface with the complete IDE — console,
watch, metrics, program catalog) plus the embed runtime (``embed.html``) for
packaged releases. The snapshot is an exact mirror of ``dist/`` minus the
``d/`` demo apps, so stale chunks from older builds never survive a sync.

One deliberate local-only edit: the analytics snippets in ``index.html``
(Umami + Baidu push) are stripped — they would neither load nor report under
the debug server's cross-origin isolation headers, and localhost hits are
junk data anyway — and a ``window.BOTLAB_LOCAL_PROJECT`` flag is injected so
the editor at ``/`` boots straight into the local project with a bare URL.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ._host import BotLabError

# Demo apps shipped under dist/d/ have their own serving story; the CLI hosts
# the BotLab editor and embed runtimes only.
_SNAPSHOT_EXCLUDED_TOP_DIRS = {"d"}

# Analytics blocks in the built index.html, anchored on their stable HTML
# comments. Missing markers (a future index.html) leave the file untouched.
_ANALYTICS_BLOCK = re.compile(
    r"[ \t]*<!-- Analytics: Umami -->[^\n]*\n[ \t]*<script[^>]*></script>\n?"
    r"|[ \t]*<!-- 百度自动推送[^\n]*-->\n[ \t]*<script>.*?</script>\n?",
    re.S,
)

# Marks this site copy as the `botlab run` debug surface: the editor boots
# straight into the local project instead of the template catalog.
_LOCAL_FLAG_SCRIPT = '<script>window.BOTLAB_LOCAL_PROJECT=true</script>'


def strip_local_analytics(html: str) -> str:
    return _ANALYTICS_BLOCK.sub("", html)


def add_local_project_flag(html: str) -> str:
    if "BOTLAB_LOCAL_PROJECT" in html:
        return html
    return html.replace("</head>", f"{_LOCAL_FLAG_SCRIPT}</head>", 1)


def localize_index_html(html: str) -> str:
    return add_local_project_flag(strip_local_analytics(html))


def _collect_source_files(dist: Path) -> list[Path]:
    files: list[Path] = []
    pending = [dist]
    while pending:
        for child in pending.pop().iterdir():
            if child.is_dir():
                if child.name in _SNAPSHOT_EXCLUDED_TOP_DIRS:
                    continue
                pending.append(child)
                continue
            if child.is_file():
                files.append(child)
    return files


def sync_ui_snapshot(botlab_dir: Path, target_dir: Path) -> int:
    dist = botlab_dir / "dist"
    if not (dist / "index.html").is_file() or not (dist / "embed.html").is_file():
        raise BotLabError(
            f"{dist} is missing index.html / embed.html. Run `npm run build` in the "
            f"BotLab checkout first ({botlab_dir})."
        )

    sources = _collect_source_files(dist)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    copied = 0
    for source in sources:
        destination = target_dir / source.relative_to(dist)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "index.html" and source.parent == dist:
            destination.write_text(
                localize_index_html(source.read_text(encoding="utf-8")), encoding="utf-8"
            )
        else:
            shutil.copy2(source, destination)
        copied += 1
    return copied
