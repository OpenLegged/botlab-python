"""WebSocket transport for HIL v1: a socket-shaped view over one connection.

The device gateway relays opaque protocol bytes, so both endpoints keep using
``protocol.receive``/``send`` unchanged. ``websockets`` is imported lazily —
plain TCP targets must not pay for gateway mode, and the CLI import chain
stays standard-library only.
"""
from __future__ import annotations

from .protocol import MAX_FRAME_BYTES, HilError, gateway_endpoint

GATEWAY_TIMEOUT_HEADER = 5


class WebSocketStream:
    """Adapter exposing ``recv``/``sendall``/``settimeout``/``close``.

    ``sendall`` sends exactly one binary message; ``recv`` serves bytes from an
    internal buffer refilled one message at a time, so partial reads behave
    like a stream socket for ``protocol.receive``. ``close_code`` records the
    gateway's delivered close code once the channel ends (``None`` before that
    and when no close frame arrived), so the resident agent can tell a
    delivered lease signal from an abrupt drop.
    """

    def __init__(self, websocket):
        self.websocket = websocket
        self._buffer = b""
        self._timeout: float | None = None
        self.close_code: int | None = None

    def settimeout(self, timeout):
        self._timeout = timeout

    def sendall(self, data):
        try:
            self.websocket.send(data)
        except Exception as exc:  # websockets raises its own hierarchy
            raise OSError(_describe(exc)) from exc

    def recv(self, bufsize):
        while not self._buffer:
            try:
                message = self.websocket.recv(timeout=self._timeout)
            except TimeoutError:
                raise
            except Exception as exc:
                # `rcvd.code` (not the deprecated ConnectionClosed.code) is
                # the delivered close code, or None when no frame arrived.
                close = getattr(exc, "rcvd", None)
                self.close_code = getattr(close, "code", None)
                raise OSError(_describe(exc)) from exc
            if not isinstance(message, (bytes, bytearray)):
                raise OSError("Gateway channel received a non-binary WebSocket frame")
            if not message:
                raise OSError("Gateway channel received an empty WebSocket message")
            self._buffer = bytes(message)
        result = self._buffer[:bufsize]
        self._buffer = self._buffer[bufsize:]
        return result

    def close(self):
        try:
            self.websocket.close()
        except Exception:
            pass

    def shutdown(self, how):
        """Socket-compatible teardown for HilHost._drop: no half-close exists."""
        self.close()


def _describe(exc: Exception) -> str:
    reason = getattr(exc, "recv_reason", None) or ""
    text = str(exc) or type(exc).__name__
    return f"{text} ({reason})" if reason else text


def gateway_url(endpoint: str, channel: str, device: str) -> str:
    """Build ``<endpoint>/<channel>/<device>`` from a validated bare origin."""
    from urllib.parse import quote
    return f"{endpoint.rstrip('/')}/{channel}/{quote(device, safe='')}"


def connect_gateway(endpoint: str, channel: str, device: str, token: str) -> WebSocketStream:
    """Dial a loopback WS gateway with bearer authentication.

    Cross-machine connections use SSH forwarding. Exactly one dial is attempted;
    the agent or coordinator owns reconnection after a failed attempt.
    """
    endpoint = gateway_endpoint(endpoint)
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise HilError("Gateway mode needs the websockets package; install botlab-python[target]") from exc
    url = gateway_url(endpoint, channel, device)
    # legacy=True keeps the long-lived connection object (no context
    # manager); compression=None skips permessage-deflate — tensor frames
    # pay CPU latency for negligible size wins; proxy=None dials the
    # gateway directly, never through ambient system proxy settings.
    try:
        websocket = connect(url, additional_headers=[("Authorization", f"Bearer {token}")],
                            open_timeout=GATEWAY_TIMEOUT_HEADER,
                            max_size=MAX_FRAME_BYTES, compression=None, legacy=True,
                            proxy=None)
    except Exception as exc:
        raise HilError(f"Gateway connection failed: {_describe(exc)}") from exc
    return WebSocketStream(websocket)
