#!/usr/bin/env python3
"""Mirror the BotLab monorepo python package into this distribution.

The ``botlab`` package BotLab serves in the browser
(``src/domain/python/botlab/`` in the BotLab repository) is vendored verbatim
as this distribution's ``sdk`` subpackage — its flat modules are the browser
SDK, and its ``_runtime/`` subpackage is the native headless runtime
(``botlab-run``). The generated built-in preset data module
(``src/domain/python/botlab_presets.py``) is copied alongside as
``src/botlab/_presets.py``. Re-sync after BotLab-side changes:

    python scripts/sync_sdk.py --botlab ../BotLab

The copy is byte-for-byte with no local edits: the top-level ``botlab_host``
module the mirror imports is aliased to ``botlab._host`` at import time in
``src/botlab/__init__.py``, and ``_runtime/`` lands under ``sdk/`` so its
``from .._runtime_types import ...`` parent-relative imports resolve unchanged.
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
RUNTIME_SUBDIR = "_runtime"
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
        if p.name not in ("__pycache__", RUNTIME_SUBDIR)
        and ((p.is_file() and p.suffix != ".py") or p.is_dir())
    ]
    if unexpected:
        names = ", ".join(sorted(p.name for p in unexpected))
        return fail(
            f"{source} contains non-module entries ({names}); the mirror ships "
            "only .py files — give data files a packaging story on the BotLab "
            "side first"
        )

    runtime_source = source / RUNTIME_SUBDIR
    runtime_files = sorted(runtime_source.rglob("*.py")) if runtime_source.is_dir() else []
    if runtime_source.exists():
        if not (runtime_source / "__init__.py").is_file():
            return fail(f"{runtime_source} is not a package (missing __init__.py)")
        stray = [
            path
            for path in runtime_source.rglob("*")
            if "__pycache__" not in path.relative_to(runtime_source).parts
            and path.is_file()
            and path.suffix != ".py"
        ]
        if stray:
            names = ", ".join(str(path.relative_to(runtime_source)) for path in stray)
            return fail(f"{runtime_source} contains non-module entries ({names})")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, TARGET_DIR / path.name)
    for path in runtime_files:
        destination = TARGET_DIR / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    shutil.copy2(presets_source, PRESETS_TARGET)
    print(
        f"synced {len(files)} SDK modules + {len(runtime_files)} runtime modules: "
        f"{source} -> {TARGET_DIR}"
    )
    print(f"synced preset data: {presets_source} -> {PRESETS_TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
