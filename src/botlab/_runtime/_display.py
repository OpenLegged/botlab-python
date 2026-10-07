"""Local display server: streams robot state over a loopback-only WebSocket.

The token URL printed by the native runner retains its legacy metadata text frame
and little-endian float32 ``[time, *qpos, *qvel]`` control frames. An explicit
``?scene=1`` suffix selects a cached MuJoCo scene description and world-state
frames. Browser origins and exact token paths are checked, and binding to
anything but 127.0.0.1 is refused.
"""

import json
import secrets
import threading
import time
from urllib.parse import urlsplit

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

_LOOPBACK_HOST = "127.0.0.1"
_MAX_TEXT_FRAME_BYTES = 64 * 1024
_MAX_SCENE_BYTES = 128 * 1024 * 1024
_MAX_SCENE_FRAME_BYTES = 32 * 1024 * 1024
_SCENE_FRAME_INTERVAL_SECONDS = 1 / 60
_MAX_METRICS = 1024
_MAX_DIAGNOSTICS = 200
_MAX_DIAGNOSTIC_KEYS = 32


def _allowed_browser_origin(origin):
    # Native WebSocket clients don't send Origin. Browsers always do, so an
    # absent value remains useful for CLI diagnostics without letting an
    # arbitrary web page read the loopback simulation stream.
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and parsed.hostname in ("127.0.0.1", "localhost")
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and parsed.query == ""
        and parsed.fragment == ""
        and (port is None or 1 <= port <= 65535)
    )


class DisplayServer:
    def __init__(self, metadata, host=_LOOPBACK_HOST, port=None, scene_provider=None):
        if host != _LOOPBACK_HOST:
            raise ValueError(
                f"[display] host must be {_LOOPBACK_HOST} (loopback only), got \"{host}\""
            )
        self._metadata = dict(metadata)
        self._host = host
        self._port = port
        self._clients = set()
        self._scene_clients = set()
        self._scene_provider = scene_provider
        self._scene_payload = None
        self._next_scene_frame_at = 0.0
        if scene_provider is not None:
            self._scene_payload = scene_provider.payload
            if len(self._scene_payload.encode("utf-8")) > _MAX_SCENE_BYTES:
                raise ValueError(f"[display] scene exceeds {_MAX_SCENE_BYTES} bytes")
            if scene_provider.frame_size > _MAX_SCENE_FRAME_BYTES:
                raise ValueError(
                    f"[display] scene frame exceeds {_MAX_SCENE_FRAME_BYTES} bytes"
                )
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._token = secrets.token_urlsafe(24)

    @property
    def port(self):
        return self._port

    @property
    def url(self):
        if self._port is None:
            raise RuntimeError("[display] server has not started")
        return f"ws://{self._host}:{self._port}/display/{self._token}"

    def start(self):
        if self._server is not None:
            raise RuntimeError("[display] server already started")
        self._server = serve(
            self._handler,
            self._host,
            self._port if self._port is not None else 0,
            max_size=1024,
            max_queue=1,
        )
        self._port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="botlab-display", daemon=True
        )
        self._thread.start()

    def _handler(self, websocket):
        request = websocket.request
        origin = request.headers.get("Origin")
        legacy_path = f"/display/{self._token}"
        scene_path = f"{legacy_path}?scene=1"
        if (
            request.path not in (legacy_path, scene_path)
            or not _allowed_browser_origin(origin)
            or (request.path == scene_path and self._scene_payload is None)
        ):
            websocket.close(1008, "invalid local display client")
            return
        scene = request.path == scene_path
        clients = self._scene_clients if scene else self._clients
        if scene:
            try:
                websocket.send(self._scene_payload)
            except ConnectionClosed:
                return
            with self._lock:
                clients.add(websocket)
        else:
            # Atomic with broadcasts: a client that received meta must already
            # be registered before the stepping thread sends its next frame.
            with self._lock:
                try:
                    websocket.send(self._text_payload({"type": "meta", **self._metadata}))
                except ConnectionClosed:
                    return
                clients.add(websocket)
        try:
            try:
                for _message in websocket:
                    websocket.close(1008, "local display is read-only")
                    return
            except ConnectionClosed:
                # Browser tabs and CLI-driven browsers may disappear without a
                # closing frame. That is a normal viewer disconnect, not a
                # server failure worth logging from the handler thread.
                pass
        finally:
            with self._lock:
                clients.discard(websocket)

    @staticmethod
    def _text_payload(message):
        payload = json.dumps(message, allow_nan=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > _MAX_TEXT_FRAME_BYTES:
            raise ValueError(
                f"[display] text frame exceeds {_MAX_TEXT_FRAME_BYTES} bytes"
            )
        return payload

    def _broadcast_text(self, message):
        payload = self._text_payload(message)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.send(payload)
            except Exception:
                with self._lock:
                    self._clients.discard(client)

    def update_metrics(self, metrics):
        if not isinstance(metrics, dict):
            raise ValueError("[display] metrics must be a mapping")
        if len(metrics) > _MAX_METRICS:
            raise ValueError(f"[display] metrics must contain at most {_MAX_METRICS} values")
        normalized = {}
        for key, value in metrics.items():
            if len(str(key)) > 256:
                raise ValueError("[display] metric names must be at most 256 characters")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f'[display] metric "{key}" must be finite')
            normalized[str(key)] = number
        self._broadcast_text({"type": "metrics", "metrics": normalized})

    def update_diagnostics(self, diagnostics):
        if not isinstance(diagnostics, list) or any(
            not isinstance(item, dict) for item in diagnostics
        ):
            raise ValueError("[display] diagnostics must be a list of mappings")
        if len(diagnostics) > _MAX_DIAGNOSTICS:
            raise ValueError(
                f"[display] diagnostics must contain at most {_MAX_DIAGNOSTICS} items"
            )
        if any(len(item) > _MAX_DIAGNOSTIC_KEYS for item in diagnostics):
            raise ValueError(
                "[display] each diagnostic must contain at most "
                f"{_MAX_DIAGNOSTIC_KEYS} fields"
            )
        self._broadcast_text(
            {
                "type": "diagnostics",
                "diagnostics": [dict(item) for item in diagnostics],
            }
        )

    def broadcast(self, time_seconds, qpos, qvel):
        frame = (
            np.concatenate((np.asarray([time_seconds], dtype=np.float32), qpos, qvel))
            .astype("<f4")
            .tobytes()
        )
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.send(frame)
            except Exception:
                with self._lock:
                    self._clients.discard(client)
        with self._lock:
            scene_clients = list(self._scene_clients)
        now = time.monotonic()
        if scene_clients and now >= self._next_scene_frame_at:
            self._next_scene_frame_at = now + _SCENE_FRAME_INTERVAL_SECONDS
            scene_frame = self._scene_provider.frame(time_seconds)
            if len(scene_frame) > _MAX_SCENE_FRAME_BYTES:
                raise ValueError(
                    f"[display] scene frame exceeds {_MAX_SCENE_FRAME_BYTES} bytes"
                )
            for client in scene_clients:
                try:
                    client.send(scene_frame)
                except Exception:
                    with self._lock:
                        self._scene_clients.discard(client)

    def stop(self):
        server = self._server
        with self._lock:
            clients = list(self._clients | self._scene_clients)
            self._clients.clear()
            self._scene_clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        self._server = None
        if server is not None:
            server.shutdown()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("[display] server thread did not stop")
