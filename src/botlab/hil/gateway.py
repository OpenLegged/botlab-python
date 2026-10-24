"""Device gateway for hosted HIL: relays opaque HIL v1 frames between the
local coordinator and resident board agents that dial out.

The gateway never parses HIL frames — pairing and byte relay only — so the
inference path inherits no Control-command semantics: no receipts, no retry,
no task table. There is exactly one shared bearer token (operator channel,
same restricted-credential stance as Control M1) and no tokenless mode;
``ws://`` is for loopback development, remote gateways must use ``wss://``.

Pairing rules:
- one agent per device ID: a re-registration takes over and closes the old
  agent connection together with its paired coordinator (device restart);
- one coordinator per device: a second coordinator is refused ("device busy");
- either side disappearing closes the other.
"""
from __future__ import annotations

import argparse
import os
import secrets
import ssl
import sys
import threading
from pathlib import Path

from .protocol import MAX_FRAME_BYTES, HilError

RECV_TIMEOUT = 600


def _parse_path(path: str) -> tuple[str, str] | None:
    if not path.startswith("/"):
        return None
    parts = path.split("/")
    if len(parts) != 3 or parts[1] not in ("agent", "device") or not parts[2]:
        return None
    return parts[1], parts[2]


class _Agent:
    def __init__(self, websocket):
        self.websocket = websocket
        self.host = None
        self.closed = threading.Event()


class Gateway:
    def __init__(self, token: str):
        self.token = token
        self.lock = threading.Lock()
        self.agents: dict[str, _Agent] = {}

    # -- websockets.sync.server plumbing -----------------------------------

    def handler(self, websocket):
        route = _parse_path(websocket.request.path)
        if route is None:
            websocket.close(1008, "unknown gateway path")
            return
        channel, device = route
        header = websocket.request.headers.get("Authorization", "")
        if not secrets.compare_digest(header, f"Bearer {self.token}"):
            websocket.close(1008, "invalid gateway token")
            return
        if channel == "agent":
            self._agent(websocket, device)
        else:
            self._device(websocket, device)

    # -- registration -------------------------------------------------------

    def _agent(self, websocket, device):
        agent = _Agent(websocket)
        with self.lock:
            previous = self.agents.get(device)
            if previous is not None and not previous.closed.is_set():
                # Device restart or redeploy: the new resident agent wins and
                # the displaced pairing dies with it.
                previous.closed.set()
                previous.websocket.close(1012, "device re-registered")
            self.agents[device] = agent
        try:
            while not agent.closed.is_set():
                try:
                    message = websocket.recv(timeout=RECV_TIMEOUT)
                except TimeoutError:
                    continue
                except Exception:
                    break
                if isinstance(message, str):
                    websocket.close(1008, "binary frames required")
                    break
                with self.lock:
                    host = agent.host
                if host is None:
                    # Coordinators speak first (hello); unsolicited agent
                    # traffic before pairing violates the contract.
                    websocket.close(1008, "agent spoke before pairing")
                    break
                try:
                    host.send(message)
                except Exception:
                    break
        finally:
            with self.lock:
                if self.agents.get(device) is agent:
                    del self.agents[device]
            agent.closed.set()
            host = agent.host
            if host is not None:
                host.close(1011, "device agent disconnected")

    def _device(self, websocket, device):
        with self.lock:
            agent = self.agents.get(device)
            if agent is None or agent.closed.is_set():
                websocket.close(1008, "unknown device")
                return
            if agent.host is not None:
                websocket.close(1008, "device busy")
                return
            agent.host = websocket
        try:
            while not agent.closed.is_set():
                try:
                    message = websocket.recv(timeout=RECV_TIMEOUT)
                except TimeoutError:
                    continue
                except Exception:
                    break
                if isinstance(message, str):
                    websocket.close(1008, "binary frames required")
                    break
                try:
                    agent.websocket.send(message)
                except Exception:
                    break
        finally:
            # A leaving coordinator must kill the device link too: the HIL
            # lease is per agent connection, exactly like the TCP transport.
            agent.closed.set()
            websocket.close()
            agent.websocket.close(1011, "coordinator disconnected")


def _ssl_context(certificate: Path | None, key: Path | None) -> ssl.SSLContext | None:
    if not certificate and not key:
        return None
    if not certificate or not key:
        raise HilError("Gateway TLS needs both --tls-cert and --tls-key")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    return context


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Relay HIL device traffic for board agents dialing out")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (use 0.0.0.0 only behind a TLS reverse proxy)")
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--tls-cert", type=Path, help="serve wss:// directly with this certificate")
    parser.add_argument("--tls-key", type=Path, help="private key matching --tls-cert")
    parser.add_argument("--print-token", action="store_true",
                        help="print the shared token at startup (operator channel only)")
    options = parser.parse_args(argv)
    try:
        from websockets.sync.server import serve
    except ImportError as exc:
        raise HilError("The gateway needs the websockets package; install botlab-python[gateway]") from exc
    token = os.environ.get("BOTLAB_HIL_GATEWAY_TOKEN") or secrets.token_urlsafe(32)
    gateway = Gateway(token)
    try:
        context = _ssl_context(options.tls_cert, options.tls_key)
    except (HilError, OSError, ssl.SSLError) as error:
        print(f"Gateway TLS setup failed: {error}", file=sys.stderr)
        return 2
    try:
        server = serve(gateway.handler, options.host, options.port, ssl=context,
                       max_size=MAX_FRAME_BYTES, compression=None,
                       ping_interval=20, ping_timeout=20)
    except OSError as error:
        print(f"Gateway cannot listen on {options.host}:{options.port}: {error}", file=sys.stderr)
        return 2
    port = server.socket.getsockname()[1]
    scheme = "wss" if context else "ws"
    print(f"BotLab HIL gateway: {scheme}://{options.host}:{port}", file=sys.stderr, flush=True)
    print(f"Agent enrollment: botlab-agent connect --gateway {scheme}://{options.host}:{port} "
          f"--device <id>", file=sys.stderr, flush=True)
    if options.print_token:
        print(f"Token: {token}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
