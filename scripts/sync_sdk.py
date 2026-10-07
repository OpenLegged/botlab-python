#!/usr/bin/env python3
"""Mirror BotLab's browser Python SDK into this distribution.

The ``botlab`` package BotLab serves in the browser
(``src/domain/python/botlab/`` in the BotLab repository) is vendored verbatim
as this distribution's ``sdk`` subpackage. Only the flat browser SDK modules
are mirrored; the native ``botlab._runtime`` is owned by this repository and
must never be copied from BotLab. The generated built-in preset data module
(``src/domain/python/botlab_presets.py``) is copied alongside as
``src/botlab/_presets.py``. Re-sync after BotLab-side changes:

    python scripts/sync_sdk.py --botlab ../BotLab

The SDK copy is byte-for-byte with no local edits: the top-level
``botlab_host`` module the mirror imports is aliased to ``botlab._host`` at
import time in ``src/botlab/__init__.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_SUBDIR = Path("src/domain/python/botlab")
PRESETS_SOURCE = PACKAGE_SUBDIR.parent / "botlab_presets.py"
TARGET_DIR = REPO_ROOT / "src" / "botlab" / "sdk"
PRESETS_TARGET = REPO_ROOT / "src" / "botlab" / "_presets.py"


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--botlab",
        default=str(REPO_ROOT.parent / "BotLab"),
        help="BotLab monorepo checkout (default: sibling ../BotLab)",
    )
    args = parser.parse_args()

    source_root = Path(args.botlab).expanduser().resolve()
    source = source_root / PACKAGE_SUBDIR
    presets_source = source_root / PRESETS_SOURCE
    if not (source / "__init__.py").is_file():
        return fail(
            f"{source} is not the BotLab python package (missing __init__.py); "
            "pass --botlab <BotLab checkout>"
        )
    if not presets_source.is_file():
        return fail(
            f"{presets_source} is missing; run `npm run export:presets` in the "
            "BotLab checkout first"
        )

    files = sorted(source.glob("*.py"))
    if not files or "__init__.py" not in {p.name for p in files}:
        return fail(f"{source} contains no SDK modules")

    unexpected = [
        p
        for p in source.iterdir()
        # _runtime 是旧原生 runner 的滞留副本——归 botlab-python 所有，不镜像；
        # 本仓移除它之前 sync 不应被它卡住（见 docs/local-python-runtime.md 迁移约定）。
        if not (p.name in ("__pycache__", "_runtime") and p.is_dir())
        and ((p.is_file() and p.suffix != ".py") or p.is_dir())
    ]
    if unexpected:
        names = ", ".join(sorted(p.name for p in unexpected))
        return fail(
            f"{source} contains non-module entries ({names}); the mirror ships "
            "only .py files — give data files a packaging story on the BotLab "
            "side first"
        )

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, TARGET_DIR / path.name)

    shutil.copy2(presets_source, PRESETS_TARGET)
    print(f"synced {len(files)} SDK modules: {source} -> {TARGET_DIR}")
    print(f"synced preset data: {presets_source} -> {PRESETS_TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
