"""Sync the BotLab UI snapshot from a BotLab build.

The embed runtime is a static closure of files: ``embed.html``, everything it
references under ``assets/``, the Pyodide runtime, and the wasm helpers. The
BotLab build already computes that closure (``dist/publish-assets.json``), so
syncing is a verbatim copy of exactly those files. This keeps the Python
package's UI byte-identical to a real BotLab release.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ._host import BotLabError

SNAPSHOT_MANIFEST = "ui-manifest.json"


def sync_ui_snapshot(botlab_dir: Path, target_dir: Path) -> int:
    dist = botlab_dir / "dist"
    manifest_path = dist / "publish-assets.json"
    if not manifest_path.is_file():
        raise BotLabError(
            f"{manifest_path} is missing. Run `npm run build` in the BotLab checkout first "
            f"({botlab_dir})."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise BotLabError(f"{manifest_path} does not list any files.")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    copied = 0
    for relative in files:
        source = dist / relative
        if not source.is_file():
            raise BotLabError(f"publish asset {relative!r} listed but missing under {dist}.")
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    snapshot_manifest = {
        "version": 1,
        "sourceBotlabBuild": "dist",
        "files": sorted(files),
    }
    (target_dir / SNAPSHOT_MANIFEST).write_text(
        json.dumps(snapshot_manifest, indent=2), encoding="utf-8"
    )
    return copied
