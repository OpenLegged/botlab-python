"""Device gateway for hosted HIL: relays opaque HIL v1 frames between the
local coordinator and resident board agents that dial out.

The gateway never parses HIL frames — pairing and byte relay only — so the
inference path inherits no Control-command semantics: no receipts, no retry,
no task table. There is exactly one shared bearer token (operator channel,
same restricted-credential stance as Control M1) and no tokenless mode; it
arrives in the ``Authorization`` header or, for browsers that cannot set
handshake headers, as one leading ``{"type": "gateway-auth", "token": ...}``
text control frame. The gateway binds only to loopback; cross-machine clients
connect through an SSH tunnel to its ``ws://`` endpoint.

Pairing rules:
- one agent per device ID: a re-registration takes over and closes the old
  agent connection together with its paired coordinator (device restart);
- one coordinator per device: a second coordinator is refused ("device busy");
- either side disappearing closes the other: a coordinator that completes its
  close handshake releases the agent for an immediate redial (close code
  ``LEASE_RELEASE_CLOSE_CODE``), while an abnormal drop leaves the agent on
  its retry-interval backoff.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
from pathlib import Path

from .protocol import GATEWAY_LOOPBACK_HOSTS, LEASE_RELEASE_CLOSE_CODE, MAX_FRAME_BYTES, HilError

RECV_TIMEOUT = 600
AUTH_TIMEOUT = 5


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
            # Browsers cannot set Authorization headers on a WebSocket
            # handshake; the same credential may arrive as one leading text
            # control frame instead.
            if not self._frame_auth(websocket):
                return
        if channel == "agent":
            self._agent(websocket, device)
        else:
            self._device(websocket, device)

    def _frame_auth(self, websocket) -> bool:
        """Authenticate a header-less client (the browser) with one text frame.

        The frame is gateway control traffic, not an HIL frame — the relay
        itself still never parses HIL traffic, and any text frame after
        authentication remains a protocol error exactly as before.
        """
        try:
            message = websocket.recv(timeout=AUTH_TIMEOUT)
        except TimeoutError:
            websocket.close(1008, "gateway authentication timeout")
            return False
        except Exception:
            return False
        token = None
        if isinstance(message, str):
            try:
                frame = json.loads(message)
            except ValueError:
                frame = None
            if isinstance(frame, dict) and frame.get("type") == "gateway-auth" \
                    and isinstance(frame.get("token"), str):
                token = frame["token"]
        if token is None or not secrets.compare_digest(
                token.encode("utf-8"), self.token.encode("utf-8")):
            websocket.close(1008, "invalid gateway token")
            return False
        return True

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
        # Imported as late as `serve` in main(): the module stays importable
        # without the websockets extra.
        from websockets.exceptions import ConnectionClosedOK

        with self.lock:
            agent = self.agents.get(device)
            if agent is None or agent.closed.is_set():
                websocket.close(1008, "unknown device")
                return
            if agent.host is not None:
                websocket.close(1008, "device busy")
                return
            agent.host = websocket
        released = False
        try:
            while not agent.closed.is_set():
                try:
                    message = websocket.recv(timeout=RECV_TIMEOUT)
                except TimeoutError:
                    continue
                except ConnectionClosedOK:
                    # A completed close handshake is a deliberate disconnect;
                    # the agent should redial now, not back off.
                    released = True
                    break
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
            # A leaving coordinator still ends the device link — the HIL lease
            # is per agent connection, exactly like the TCP transport. A clean
            # goodbye asks the agent to redial at once; anything abnormal
            # leaves the agent its retry-interval backoff.
            with self.lock:
                agent.host = None
                agent.closed.set()
            websocket.close()
            agent.websocket.close(
                LEASE_RELEASE_CLOSE_CODE if released else 1011,
                "coordinator disconnected")


GATEWAY_UNIT = "botlab-hil-gateway.service"


def _install(options) -> int:
    """Install the gateway as a systemd user service with a pinned token.

    A service must not generate a fresh random token per restart — that would
    silently invalidate every enrolled agent — so the token is pinned from
    BOTLAB_HIL_GATEWAY_TOKEN or an interactive prompt into a 0600 env file.
    """
    import getpass

    from .service import check_env_token, install_service

    token = check_env_token("BOTLAB_HIL_GATEWAY_TOKEN",
                            os.environ.get("BOTLAB_HIL_GATEWAY_TOKEN")
                            or getpass.getpass("Gateway token to pin: "))
    exec_argv = [sys.executable, "-m", "botlab.hil.gateway",
                 "--host", options.host, "--port", str(options.port)]
    try:
        summary = install_service(
            GATEWAY_UNIT, "BotLab HIL device gateway",
            exec_argv, Path.home() / ".config/botlab/gateway-token.env",
            {"BOTLAB_HIL_GATEWAY_TOKEN": token}, enable_linger=options.enable_linger)
    except HilError as exc:
        print(f"botlab-hil-gateway install: {exc}", file=sys.stderr)
        return 1
    print(f"Installed {summary['unit']} with a pinned token ({summary['env_file']}, 0600)")
    print(f"  unit:  {summary['unit_file']}")
    print(f"  starts at login; logs: journalctl --user -u {summary['unit']} -f")
    if summary["autostart"]:
        print("  autostart at boot: yes (lingering enabled)")
    else:
        print("  autostart at boot: no — a user service only starts at login.")
        print("  to start at boot without a login, run: sudo loginctl enable-linger $USER")
    return 0 if summary["linger_error"] is None else 1


def _uninstall(options) -> int:
    from .service import uninstall_service

    try:
        summary = uninstall_service(GATEWAY_UNIT, Path.home() / ".config/botlab/gateway-token.env")
    except HilError as exc:
        print(f"botlab-hil-gateway uninstall: {exc}", file=sys.stderr)
        return 1
    print(f"Removed {summary['unit']} and {summary['env_file']}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["install"], ["uninstall"]):
        install = argv[0] == "install"
        parser = argparse.ArgumentParser(
            description=("Install" if install else "Remove")
                      + " the gateway as a systemd user service with a pinned token")
        parser.add_argument("--host", default="127.0.0.1", choices=GATEWAY_LOOPBACK_HOSTS,
                            help="loopback bind address; use SSH forwarding for remote clients")
        parser.add_argument("--port", type=int, default=50060)
        if install:
            parser.add_argument("--enable-linger", action="store_true",
                                help="also enable user lingering so the service starts at "
                                     "boot without a login (may need sudo)")
        options = parser.parse_args(argv[1:])
        return (_install if install else _uninstall)(options)
    parser = argparse.ArgumentParser(
        description="Relay HIL device traffic for board agents dialing out")
    parser.add_argument("--host", default="127.0.0.1", choices=GATEWAY_LOOPBACK_HOSTS,
                        help="loopback bind address; use SSH forwarding for remote clients")
    parser.add_argument("--port", type=int, default=50060)
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
        server = serve(gateway.handler, options.host, options.port,
                       max_size=MAX_FRAME_BYTES, compression=None,
                       ping_interval=20, ping_timeout=20)
    except OSError as error:
        print(f"Gateway cannot listen on {options.host}:{options.port}: {error}", file=sys.stderr)
        return 2
    port = server.socket.getsockname()[1]
    host = f"[{options.host}]" if ":" in options.host else options.host
    print(f"BotLab HIL gateway: ws://{host}:{port}", file=sys.stderr, flush=True)
    print(f"Agent enrollment: botlab-agent connect --gateway ws://{host}:{port} "
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
