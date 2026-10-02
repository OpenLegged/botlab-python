"""Hatch build hook: force-include the gitignored UI snapshot when present.

``src/botlab/ui/`` (~199 MB, regenerated with ``botlab ui sync``) never enters
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


class BotlabBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        force_include = build_data.setdefault("force_include", {})
        if (root / _UI_SNAPSHOT[0]).is_dir():
            force_include[_UI_SNAPSHOT[0]] = _UI_SNAPSHOT[1]
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
        force_include[_DOCS[0]] = _DOCS[1]
