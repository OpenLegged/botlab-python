#!/usr/bin/env python3
"""Mirror the BotLab monorepo SDK package into src/botlab/sdk/.

The ``botlab`` package BotLab serves in the browser
(``src/domain/python/botlab/`` in the BotLab repository) is vendored verbatim
as this distribution's ``sdk`` subpackage, so the same SDK source runs in
CPython and in the browser. Re-sync the mirror after BotLab-side SDK changes:

    python scripts/sync_sdk.py --botlab ../BotLab

The copy is byte-for-byte with no local edits: the top-level ``botlab_host``
module the mirror imports is aliased to ``botlab._host`` at import time in
``src/botlab/__init__.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUBDIR = Path("src/domain/python/botlab")
TARGET_DIR = REPO_ROOT / "src" / "botlab" / "sdk"


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

    source = Path(args.botlab).expanduser().resolve() / SOURCE_SUBDIR
    if not (source / "__init__.py").is_file():
        return fail(
            f"{source} is not the BotLab SDK package (missing __init__.py); "
            "pass --botlab <BotLab checkout>"
        )

    files = sorted(source.glob("*.py"))
    if not files or "__init__.py" not in {p.name for p in files}:
        return fail(f"{source} contains no SDK modules")

    unexpected = [
        p
        for p in source.iterdir()
        if p.name != "__pycache__" and ((p.is_file() and p.suffix != ".py") or p.is_dir())
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
    print(f"synced {len(files)} modules: {source} -> {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
