"""BotLab HIL v1: bounded length-framed JSON and float32 tensor contracts."""
from __future__ import annotations

import json
import math
import re
import socket
import struct
from urllib.parse import urlsplit

VERSION = 1
MAX_MODEL_BYTES = 32 * 1024 * 1024
MAX_FRAME_BYTES = 48 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 1_000_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HilError(ValueError):
    """An actionable HIL error, optionally carrying a recoverable SSH challenge."""

    def __init__(self, message, *, error_code=None, target_id=None, password_attempted=None):
        super().__init__(message)
        self.error_code = error_code
        self.target_id = target_id
        self.password_attempted = password_attempted


def error_details(error) -> dict:
    """Only public recovery metadata; never expose credentials or exception internals."""
    if isinstance(error, HilError) and error.error_code == "ssh_auth_required":
        return {"error_code": error.error_code, "target_id": error.target_id,
                "password_attempted": error.password_attempted is True}
    return {}


def endpoint_address(endpoint: str) -> tuple[str, int]:
    if not isinstance(endpoint, str):
        raise HilError("Target endpoint must be tcp://host:port")
    try:
        url = urlsplit(endpoint if "://" in endpoint else "tcp://" + endpoint)
        if (url.scheme != "tcp" or not url.hostname or url.port is None
                or url.username or url.password or url.path or url.query or url.fragment):
            raise ValueError
        if not 1 <= url.port <= 65535:
            raise ValueError
        return url.hostname, url.port
    except ValueError as exc:
        raise HilError("Target endpoint must be tcp://host:port") from exc


def gateway_endpoint(endpoint: str) -> str:
    """Validate a gateway origin: ws:// stays loopback, wss:// goes anywhere.

    Mirrors the Control M1 endpoint rule — plaintext never leaves the loopback,
    and paths are derived from the device ID, never stored.
    """
    if not isinstance(endpoint, str):
        raise HilError("Gateway endpoint must be ws:// or wss://")
    try:
        url = urlsplit(endpoint)
        scheme = url.scheme.lower()
        if url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
            raise ValueError
        if not url.hostname:
            raise ValueError
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError
        if scheme == "ws":
            if url.hostname not in ("127.0.0.1", "localhost", "::1") or url.port is None:
                raise ValueError
        elif scheme != "wss":
            raise ValueError
        return f"{scheme}://{url.netloc.lower()}"
    except ValueError as exc:
        raise HilError("Gateway endpoint must be a bare ws://loopback:port or wss://host[:port] origin") from exc


def _read(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise EOFError("Target disconnected")
        result.extend(part)
    return bytes(result)


def receive(sock: socket.socket) -> dict:
    size = struct.unpack("!I", _read(sock, 4))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise HilError("HIL message exceeds the frame limit")
    try:
        result = json.loads(_read(sock, size), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError) as exc:
        raise HilError("Malformed HIL JSON") from exc
    if not isinstance(result, dict) or result.get("version") != VERSION:
        raise HilError("Unsupported HIL protocol version (expected 1)")
    return result


def send(sock: socket.socket, value: dict) -> None:
    payload = json.dumps({**value, "version": VERSION}, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME_BYTES:
        raise HilError("HIL message exceeds the frame limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def tensor(value: object, spec: dict | None = None) -> dict:
    if not isinstance(value, dict) or value.get("dtype") != "float32":
        raise HilError("HIL v1 accepts float32 tensors only")
    shape, data = value.get("shape"), value.get("data")
    if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
        raise HilError("Tensor shape must contain non-negative integer dimensions")
    size = math.prod(shape)
    if size > MAX_TENSOR_ELEMENTS or not isinstance(data, list) or len(data) != size:
        raise HilError("Tensor data length does not match shape or exceeds limit")
    if any(type(n) not in (int, float) or abs(n) > 3.4028234663852886e38 or not math.isfinite(n) for n in data):
        raise HilError("Tensor values must be finite float32 numbers")
    if spec:
        expected = spec["shape"]
        if spec["dtype"] != "float32" or len(shape) != len(expected) or any(
            isinstance(n, int) and n >= 0 and n != actual
            for n, actual in zip(expected, shape, strict=True)
        ):
            raise HilError(f"Tensor shape does not match port {spec['name']}: expected {expected}, got {shape}")
    return {"dtype": "float32", "shape": shape, "data": data}


def tensors(values: object, specs: list[dict]) -> dict:
    if not isinstance(values, dict) or set(values) != {s["name"] for s in specs}:
        raise HilError("Tensor names do not match the model ports")
    return {spec["name"]: tensor(values[spec["name"]], spec) for spec in specs}


def ports(value: object) -> list[dict]:
    """Validate backend-provided metadata before exposing it to consumers."""
    if not isinstance(value, list) or not value or len(value) > 128:
        raise HilError("Models require 1–128 named ports")
    names = set()
    for spec in value:
        if not isinstance(spec, dict):
            raise HilError("Invalid model port metadata")
        name, shape = spec.get("name"), spec.get("shape")
        if not isinstance(name, str) or not name or len(name) > 1024 or name in names:
            raise HilError("Model port names must be nonempty and unique")
        names.add(name)
        if spec.get("dtype") != "float32" or not isinstance(shape, list) or len(shape) > 16:
            raise HilError("HIL v1 requires float32 ports with valid shapes")
        if any(not (n is None or type(n) is int and n >= 0 or isinstance(n, str)) for n in shape):
            raise HilError("Invalid model port dimension")
    return value
