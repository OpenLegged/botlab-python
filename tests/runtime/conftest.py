"""Isolate the real native host and keep temporary native assets in pytest tmp."""

import tempfile

import pytest

from botlab import _host


@pytest.fixture(autouse=True)
def native_runtime_state(tmp_path, monkeypatch):
    _host.set_runtime_session(None)
    _host.project_state().reset(str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield
    _host.set_runtime_session(None)
    _host.project_state().reset()
