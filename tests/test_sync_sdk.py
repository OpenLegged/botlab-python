"""Unit tests for scripts/sync_sdk.py: what the mirror carries, verbatim."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SYNTHETIC_PACKAGE = {
    "src/domain/python/botlab/__init__.py": "# sdk init\n",
    "src/domain/python/botlab/_signal.py": "# sdk module\n",
    # read_config 解析器是镜像面的一部分：CLI 工程读 config/*.yaml 靠它。
    "src/domain/python/botlab/_config.py": "# config reader\n",
}

@pytest.fixture()
def upstream(tmp_path: Path) -> Path:
    for relative, text in SYNTHETIC_PACKAGE.items():
        path = tmp_path / "BotLab" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path / "BotLab"


def run_sync(upstream: Path, dist_root: Path) -> int:
    scripts = dist_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(__file__).resolve().parent.parent / "scripts" / "sync_sdk.py",
        scripts / "sync_sdk.py",
    )
    result = subprocess.run(
        [sys.executable, str(scripts / "sync_sdk.py"), "--botlab", str(upstream)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(result.stdout + result.stderr)
    return result.returncode


@pytest.fixture()
def dist_root(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "src" / "botlab").mkdir(parents=True)
    native = root / "src" / "botlab" / "_runtime" / "run.py"
    native.parent.mkdir(parents=True)
    native.write_text("# independent native runner\n", encoding="utf-8")
    return root


def test_mirror_carries_only_flat_sdk_verbatim(upstream, dist_root):
    assert run_sync(upstream, dist_root) == 0

    sdk = dist_root / "src" / "botlab" / "sdk"
    assert (sdk / "__init__.py").read_text(encoding="utf-8") == "# sdk init\n"
    assert (sdk / "_signal.py").read_text(encoding="utf-8") == "# sdk module\n"
    assert (sdk / "_config.py").read_text(encoding="utf-8") == "# config reader\n"
    assert not (sdk / "_runtime").exists()
    assert (dist_root / "src" / "botlab" / "_runtime" / "run.py").read_text(
        encoding="utf-8"
    ) == "# independent native runner\n"
    assert not (dist_root / "src" / "botlab" / "_presets.py").exists()


def test_mirror_wipes_stale_modules(upstream, dist_root):
    stale = dist_root / "src" / "botlab" / "sdk" / "_gone.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("# stale\n", encoding="utf-8")
    old_mirror_runtime = stale.parent / "_runtime" / "run.py"
    old_mirror_runtime.parent.mkdir(parents=True)
    old_mirror_runtime.write_text("# old mirrored runner\n", encoding="utf-8")
    assert run_sync(upstream, dist_root) == 0
    assert not stale.exists()
    assert not old_mirror_runtime.exists()
    assert (dist_root / "src" / "botlab" / "_runtime" / "run.py").read_text(
        encoding="utf-8"
    ) == "# independent native runner\n"



def test_unexpected_entries_still_rejected(upstream, dist_root):
    (upstream / "src/domain/python/botlab" / "data.json").write_text("{}", encoding="utf-8")
    assert run_sync(upstream, dist_root) == 1


def test_upstream_runtime_subdirectory_skipped_without_touching_native(upstream, dist_root):
    old_sdk = dist_root / "src" / "botlab" / "sdk" / "_signal.py"
    old_sdk.parent.mkdir()
    old_sdk.write_text("# previous sdk\n", encoding="utf-8")
    foreign_runtime = upstream / "src/domain/python/botlab/_runtime"
    foreign_runtime.mkdir()
    (foreign_runtime / "run.py").write_text("# wrong owner\n", encoding="utf-8")
    # 上游滞留的 _runtime 副本不镜像、不阻断（归本仓所有，见 scripts/sync_sdk.py）
    assert run_sync(upstream, dist_root) == 0
    assert (dist_root / "src" / "botlab" / "_runtime" / "run.py").read_text(
        encoding="utf-8"
    ) == "# independent native runner\n"
    # 上游的 _runtime 没有进镜像，但 SDK 文件同步了
    assert not (dist_root / "src" / "botlab" / "sdk" / "_runtime").exists()
    assert old_sdk.read_text(encoding="utf-8") != "# previous sdk\n"


def test_pycache_is_ignored(upstream, dist_root):
    cache = upstream / "src/domain/python/botlab/__pycache__"
    cache.mkdir()
    (cache / "_signal.cpython-312.pyc").write_bytes(b"cached")
    assert run_sync(upstream, dist_root) == 0
    assert not (dist_root / "src" / "botlab" / "sdk" / "__pycache__").exists()
