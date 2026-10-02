"""The candidate gate must reject incomplete wheels before installing them."""

from __future__ import annotations

import runpy
import zipfile
from pathlib import Path

import pytest

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_wheel.py"))


def _wheel(tmp_path: Path, *, ui: bool = True, missing_doc: bool = False) -> Path:
    wheel = tmp_path / "candidate.whl"
    names = ["botlab/_cli.py", "botlab/sdk/_runtime/run.py"]
    names.extend(f"botlab/docs/{name}" for name in VERIFY["DOCS"])
    if missing_doc:
        names.remove("botlab/docs/error-guide.md")
    if ui:
        names.extend(["botlab/ui/index.html", "botlab/ui/embed.html", "botlab/ui/runtime.wasm"])
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in names:
            archive.writestr(name, "fixture")
    return wheel


def test_candidate_gate_requires_bundled_docs(tmp_path):
    with pytest.raises(ValueError, match="error-guide.md"):
        VERIFY["verify_archive"](_wheel(tmp_path, missing_doc=True), source_wheel=False)


def test_source_wheel_is_not_a_candidate_release(tmp_path):
    wheel = _wheel(tmp_path, ui=False)
    VERIFY["verify_archive"](wheel, source_wheel=True)
    with pytest.raises(ValueError, match="WASM"):
        VERIFY["verify_archive"](wheel, source_wheel=False)


def test_source_contract_requires_an_explicitly_ui_free_wheel(tmp_path):
    with pytest.raises(ValueError, match="without the UI"):
        VERIFY["verify_archive"](_wheel(tmp_path), source_wheel=True)


def test_existing_output_environment_is_protected(tmp_path):
    wheel = _wheel(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("existing environment")
    with pytest.raises(SystemExit) as error:
        VERIFY["main"]([str(wheel), "--out", str(out)])
    assert error.value.code == 2
    assert marker.read_text() == "existing environment"
