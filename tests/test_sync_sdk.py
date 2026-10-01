"""Unit tests for scripts/sync_sdk.py: what the mirror carries, verbatim."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import pytest

SYNTHETIC_PACKAGE = {
    "src/domain/python/botlab/__init__.py": "# sdk init\n",
    "src/domain/python/botlab/_signal.py": "# sdk module\n",
    "src/domain/python/botlab/_runtime/__init__.py": "# runtime init\n",
    "src/domain/python/botlab/_runtime/run.py": "from .._runtime_types import Frame\n",
    "src/domain/python/botlab/_runtime/_processors/__init__.py": "# processors\n",
    "src/domain/python/botlab/_runtime/_processors/_model.py": "# model processor\n",
}
PRESETS_MODULE = 'PRESETS = {"g1": {"kind": "robot", "config": {}}}\n'


@pytest.fixture()
def upstream(tmp_path: Path) -> Path:
    for relative, text in SYNTHETIC_PACKAGE.items():
        path = tmp_path / "BotLab" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "BotLab" / "src/domain/python/botlab_presets.py").write_text(
        PRESETS_MODULE, encoding="utf-8"
    )
    return tmp_path / "BotLab"


def run_sync(upstream: Path, dist_root: Path) -> int:
    scripts = dist_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(__file__).resolve().parent.parent / "scripts" / "sync_sdk.py",
        scripts / "sync_sdk.py",
    )
    result = runpy.run_path(str(scripts / "sync_sdk.py"), run_name="not_main")
    # The script resolves its target from its own location; point it at the
    # synthetic distribution before calling main.
    result["REPO_ROOT"] = dist_root
    result["TARGET_DIR"] = dist_root / "src" / "botlab" / "sdk"
    result["PRESETS_TARGET"] = dist_root / "src" / "botlab" / "_presets.py"
    return result["main"]()


@pytest.fixture()
def dist_root(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "src" / "botlab").mkdir(parents=True)
    return root


def test_mirror_carries_sdk_runtime_and_presets_verbatim(upstream, dist_root):
    assert run_sync(upstream, dist_root) == 0

    sdk = dist_root / "src" / "botlab" / "sdk"
    assert (sdk / "__init__.py").read_text(encoding="utf-8") == "# sdk init\n"
    assert (sdk / "_runtime" / "run.py").read_text(encoding="utf-8").startswith(
        "from .._runtime_types"
    )
    assert (sdk / "_runtime" / "_processors" / "_model.py").is_file()
    assert (
        dist_root / "src" / "botlab" / "_presets.py"
    ).read_text(encoding="utf-8") == PRESETS_MODULE


def test_mirror_wipes_stale_modules(upstream, dist_root):
    stale = dist_root / "src" / "botlab" / "sdk" / "_gone.py"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("# stale\n", encoding="utf-8")
    assert run_sync(upstream, dist_root) == 0
    assert not stale.exists()


def test_missing_presets_module_fails(upstream, dist_root):
    (upstream / "src/domain/python/botlab_presets.py").unlink()
    assert run_sync(upstream, dist_root) == 1


def test_unexpected_entries_still_rejected(upstream, dist_root):
    (upstream / "src/domain/python/botlab" / "data.json").write_text("{}", encoding="utf-8")
    assert run_sync(upstream, dist_root) == 1


def test_runtime_with_stray_data_file_rejected(upstream, dist_root):
    (upstream / "src/domain/python/botlab/_runtime" / "weights.bin").write_bytes(b"x")
    assert run_sync(upstream, dist_root) == 1
