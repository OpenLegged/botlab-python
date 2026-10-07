"""Hatch build hook: include only the checked pure display snapshot.

``src/botlab/ui/`` (regenerated with ``botlab ui sync``) never enters
git, so fresh clones lack it. A static force-include makes hatchling fail the
whole build when the source is missing, which would block ``pip install -e .``
for CI and contributors. Instead, include what is present, and refuse only
builds that would silently ship a wheel without the snapshot (set
``BOTLAB_ALLOW_MISSING_UI=1`` to override, e.g. in CI).
"""

from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_UI_SNAPSHOT = ("src/botlab/ui", "botlab/ui")
_DOCS = ("docs", "botlab/docs")
_REQUIRED_DOCS = ("component-reference.md", "decision-table.md", "error-guide.md")


def _validate_viewer(path: Path) -> None:
    if not (path / "index.html").is_file() or not (path / "embed.html").is_file():
        raise ValueError("index.html or embed.html is missing")
    for source in path.rglob("*"):
        if source.is_symlink():
            raise ValueError(f"symlink in viewer: {source}")
        if not source.is_file():
            continue
        import re

        relative = source.relative_to(path).as_posix()
        if ".wasm" in relative.lower() or re.search(
            r"(?:^|/)(?:wasm|pyodide|netron|monaco|ort(?:-|\.|/)|onnxruntime)",
            relative, re.I,
        ):
            raise ValueError(f"WASM or heavyweight runtime in viewer: {source}")
        if source.suffix in {".html", ".js", ".mjs", ".css"}:
            if re.search(r"(?:\.wasm\b|WebAssembly|ort-wasm|pyodide|@mujoco/mujoco)",
                         source.read_text(encoding="utf-8"), re.I):
                raise ValueError(f"browser simulation runtime reference in viewer: {source}")


class BotlabBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        force_include = build_data.setdefault("force_include", {})
        if (root / _UI_SNAPSHOT[0]).is_dir():
            try:
                _validate_viewer(root / _UI_SNAPSHOT[0])
            except Exception as error:
                raise SystemExit(f"Invalid native viewer snapshot: {error}") from error
            destination = _UI_SNAPSHOT[0] if self.target_name == "sdist" else _UI_SNAPSHOT[1]
            force_include[_UI_SNAPSHOT[0]] = destination
        elif not os.environ.get("BOTLAB_ALLOW_MISSING_UI"):
            raise SystemExit(
                "src/botlab/ui is missing: this wheel would ship without the UI "
                "snapshot and every `botlab run` would fail. Regenerate it with "
                "`botlab ui sync --botlab <BotLab checkout>`, or set "
                "BOTLAB_ALLOW_MISSING_UI=1 when you only need an installable "
                "package (CI, contributor checkout)."
            )
        missing_docs = [name for name in _REQUIRED_DOCS if not (root / _DOCS[0] / name).is_file()]
        if missing_docs:
            raise SystemExit(
                "Required bundled docs are missing: " + ", ".join(missing_docs)
                + ". Restore docs/ before building; BOTLAB_ALLOW_MISSING_UI only "
                "permits a contributor/CI wheel without the UI snapshot."
            )
        force_include[_DOCS[0]] = _DOCS[0] if self.target_name == "sdist" else _DOCS[1]
