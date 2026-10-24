"""Broadcasts drop clients that disconnected but surface real bugs."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("websockets")

import numpy as np  # noqa: E402
from websockets.exceptions import ConnectionClosedError  # noqa: E402

from botlab._runtime._display import DisplayServer  # noqa: E402


class _Client:
    def __init__(self, error):
        self.error = error

    def send(self, _payload):
        raise self.error


def _server_with(client):
    server = DisplayServer.__new__(DisplayServer)
    import threading

    server._lock = threading.Lock()
    server._clients = {client}
    server._scene_clients = set()
    server._scene_provider = None
    server._next_scene_frame_at = 0.0
    return server


@pytest.mark.parametrize("error", [ConnectionClosedError(None, None), BrokenPipeError()])
def test_disconnected_client_is_dropped(error):
    client = _Client(error)
    server = _server_with(client)
    server.broadcast(0.0, np.zeros(1, np.float32), np.zeros(1, np.float32))
    assert server._clients == set()


def test_unexpected_send_error_propagates():
    client = _Client(RuntimeError("bug"))
    server = _server_with(client)
    with pytest.raises(RuntimeError, match="bug"):
        server.broadcast(0.0, np.zeros(1, np.float32), np.zeros(1, np.float32))
