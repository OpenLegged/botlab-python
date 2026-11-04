"""HIL device-gateway flows: outbound agent registration, transparent relay,
exclusive pairing, device-restart takeover and stale-session rejection.

These cover the *functional* acceptance tier only: the model executes
correctly through the gateway. The wall-clock tier (loop rate at the
project's control_hz) is measured on real hardware via `botlab target bench`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import ssl
import subprocess
import threading
import time
from pathlib import Path

import pytest

from botlab.hil.agent import AgentState, serve_stream
from botlab.hil.gateway import Gateway
from botlab.hil.host import HilHost
from botlab.hil.protocol import HilError, gateway_endpoint, receive, send
from botlab.hil.targets import save_target
from botlab.hil.wsstream import WebSocketStream, connect_gateway

pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.sync.client import connect as ws_connect  # noqa: E402
from websockets.sync.server import serve as ws_serve  # noqa: E402

TOKEN = "gateway-test-token"
DEVICE = "bench-board"


class EchoBackend:
    """Deterministic stand-in backend: doubles every input port."""

    def describe(self):
        return {"backend": "echo", "provider": "test", "dtypes": ["float32"]}

    def load(self, path):
        spec = [{"name": "x", "dtype": "float32", "shape": [2]}]
        return object(), spec, spec

    def infer(self, handle, inputs):
        return {name: {"dtype": "float32", "shape": v["shape"],
                       "data": [value * 2 for value in v["data"]]}
                for name, v in inputs.items()}

    def close(self, handle):
        pass


class AgentThread:
    """Resident board agent mirroring `botlab-agent connect`: one AgentState,
    redialing the gateway whenever a lease terminates or the link drops."""

    def __init__(self, endpoint: str, device: str, models_dir: Path, token: str = TOKEN):
        self.endpoint, self.device, self.token = endpoint, device, token
        self.state = AgentState(models_dir, EchoBackend())
        self.stop = threading.Event()
        self.stream = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.stream = connect_gateway(self.endpoint, "agent", self.device, self.token)
            except HilError:
                self.stop.wait(0.05)
                continue
            try:
                serve_stream(self.state, self.stream)
            except (OSError, EOFError, ValueError):
                pass
            finally:
                self.state.disconnect(self.stream)
                self.stream.close()
                self.stream = None

    def close(self):
        self.stop.set()
        stream = self.stream
        if stream:
            stream.close()
        self.thread.join(timeout=5)


def wait_registered(server_gateway: Gateway, device: str = DEVICE, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while device not in server_gateway.agents:
        assert time.monotonic() < deadline, f"device {device} never registered"
        time.sleep(0.02)


def wait_unregistered(server_gateway: Gateway, device: str = DEVICE) -> None:
    deadline = time.monotonic() + 5
    while device in server_gateway.agents:
        assert time.monotonic() < deadline, f"device {device} never deregistered"
        time.sleep(0.02)


def wait_replaced(server_gateway: Gateway, previous, device: str = DEVICE) -> None:
    """Wait until `previous`'s registration has been displaced by a new one.

    The resident agent redials within milliseconds of a terminated lease, so
    polling for an empty registry races the re-registration; comparing the
    registered `_Agent` object is deterministic.
    """
    deadline = time.monotonic() + 5
    while server_gateway.agents.get(device) is previous or device not in server_gateway.agents:
        assert time.monotonic() < deadline, f"device {device} registration never replaced"
        time.sleep(0.02)


@pytest.fixture
def gateway():
    server_gateway = Gateway(TOKEN)
    server = ws_serve(server_gateway.handler, "127.0.0.1", 0,
                      max_size=64 * 1024 * 1024, ping_interval=5, ping_timeout=5)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", server_gateway
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def board(tmp_path, monkeypatch, gateway):
    """Registered device plus a gateway target the HilHost can connect to."""
    endpoint, server_gateway = gateway
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", TOKEN)
    agent = AgentThread(endpoint, DEVICE, tmp_path / "models")
    wait_registered(server_gateway)
    save_target({"id": "gw", "name": "Gateway board", "transport": "gateway",
                 "endpoint": endpoint, "device": DEVICE, "backend": "echo"})
    yield agent
    agent.close()


def load_echo_model(host: HilHost) -> str:
    content = b"echo-model-bytes"
    digest = hashlib.sha256(content).hexdigest()
    host.handle("POST", "/hil/models", {"session": host.session,
                                        "sha256": digest,
                                        "data": base64.b64encode(content).decode()})
    return digest


def run_inference(host: HilHost, digest: str, seq: int):
    return host.handle("POST", "/hil/infer", {
        "session": host.session, "seq": seq, "model": digest,
        "inputs": {"x": {"dtype": "float32", "shape": [2], "data": [1.0, 2.5]}}})


def dial(endpoint: str, device: str = DEVICE, token: str = TOKEN) -> WebSocketStream:
    return WebSocketStream(ws_connect(
        f"{endpoint}/device/{device}",
        additional_headers=[("Authorization", f"Bearer {token}")],
        open_timeout=5, max_size=64 * 1024 * 1024, legacy=True))


def test_gateway_end_to_end(board, tmp_path):
    host = HilHost(tmp_path / "project", selected_target="gw")
    try:
        state = host.handle("POST", "/hil/connect", {"target": "gw"})
        assert state["connected"] is True
        assert state["device"]["backend"] == "echo"
        session = host.handle("POST", "/hil/start", {})["session"]
        digest = load_echo_model(host)
        result = run_inference(host, digest, 0)
        assert result["outputs"]["x"]["data"] == [2.0, 5.0]
        assert result["rtt_ms"] >= 0 and result["inference_ms"] >= 0

        # Serial multi-model loop stands in for the functional in-loop tier.
        for seq in range(1, 25):
            run_inference(host, digest, seq)
        host.handle("POST", "/hil/stop", {"session": session})
        assert host.handle("GET", "/hil/state")["session"] is None
    finally:
        host.close()


def test_gateway_unknown_device(gateway, tmp_path, monkeypatch):
    endpoint, _ = gateway
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", TOKEN)
    save_target({"id": "ghost", "name": "Ghost", "transport": "gateway",
                 "endpoint": endpoint, "device": "nobody-registered", "backend": "echo"})
    host = HilHost(tmp_path / "project", selected_target="ghost")
    try:
        with pytest.raises(HilError, match="unknown device"):
            host.handle("POST", "/hil/connect", {"target": "ghost"})
    finally:
        host.close()


def test_gateway_exclusive_pairing(board, tmp_path):
    first = HilHost(tmp_path / "project", selected_target="gw")
    second = HilHost(tmp_path / "project2", selected_target="gw")
    try:
        assert first.handle("POST", "/hil/connect", {"target": "gw"})["connected"] is True
        with pytest.raises(HilError, match="device busy"):
            second.handle("POST", "/hil/connect", {"target": "gw"})
        # The first coordinator keeps its exclusive link.
        first.handle("POST", "/hil/start", {})
        digest = load_echo_model(first)
        assert run_inference(first, digest, 0)["outputs"]["x"]["data"] == [2.0, 5.0]
    finally:
        first.close()
        second.close()


def test_gateway_bad_token(gateway, tmp_path, monkeypatch):
    endpoint, _ = gateway
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", "wrong-token")
    save_target({"id": "gw", "name": "Gateway board", "transport": "gateway",
                 "endpoint": endpoint, "device": DEVICE, "backend": "echo"})
    host = HilHost(tmp_path / "project", selected_target="gw")
    try:
        with pytest.raises(HilError, match="invalid gateway token"):
            host.handle("POST", "/hil/connect", {"target": "gw"})
    finally:
        host.close()


def dial_unauthenticated(endpoint: str, device: str = DEVICE):
    """Browser-shaped dial: no Authorization header can be set on a WebSocket
    handshake, so the credential must travel in a leading text frame."""
    return ws_connect(f"{endpoint}/device/{device}", open_timeout=5,
                      max_size=64 * 1024 * 1024, legacy=True)


def expect_close(raw) -> tuple[int, str]:
    with pytest.raises(ConnectionClosed) as exc:
        raw.recv(timeout=5)
    close = exc.value.rcvd
    assert close is not None, "peer closed without a close frame"
    return close.code, close.reason


def test_gateway_text_frame_auth(board, gateway):
    """The browser coordinator authenticates with one gateway-auth text frame,
    then runs the full device-chain flow unchanged."""
    endpoint, _ = gateway
    raw = dial_unauthenticated(endpoint)
    raw.send(json.dumps({"type": "gateway-auth", "token": TOKEN}))
    stream = WebSocketStream(raw)
    try:
        send(stream, {"op": "hello"})
        assert receive(stream)["ok"] is True
        send(stream, {"op": "start"})
        session = receive(stream)["session"]
        content = b"echo-model-bytes"
        digest = hashlib.sha256(content).hexdigest()
        send(stream, {"op": "load", "session": session, "sha256": digest,
                      "data": base64.b64encode(content).decode()})
        assert receive(stream)["ok"] is True
        send(stream, {"op": "infer", "session": session, "seq": 0, "model": digest,
                      "inputs": {"x": {"dtype": "float32", "shape": [2], "data": [1.0, 2.5]}}})
        assert receive(stream)["outputs"]["x"]["data"] == [2.0, 5.0]
    finally:
        stream.close()


def test_gateway_text_frame_auth_bad_token(gateway):
    endpoint, _ = gateway
    raw = dial_unauthenticated(endpoint)
    raw.send(json.dumps({"type": "gateway-auth", "token": "not-the-token"}))
    assert expect_close(raw) == (1008, "invalid gateway token")


def test_gateway_text_frame_auth_malformed(gateway):
    endpoint, _ = gateway
    raw = dial_unauthenticated(endpoint)
    raw.send("not json at all")
    assert expect_close(raw) == (1008, "invalid gateway token")


def test_gateway_text_frame_auth_timeout(gateway, monkeypatch):
    monkeypatch.setattr("botlab.hil.gateway.AUTH_TIMEOUT", 0.2)
    endpoint, _ = gateway
    raw = dial_unauthenticated(endpoint)
    assert expect_close(raw) == (1008, "gateway authentication timeout")


def test_gateway_text_after_auth_still_rejected(board, gateway):
    """The gateway-auth frame is control traffic, not an HIL frame: once
    authenticated, text frames remain a protocol error exactly as before."""
    endpoint, _ = gateway
    stream = dial(endpoint)
    try:
        send(stream, {"op": "hello"})
        assert receive(stream)["ok"] is True
        stream.websocket.send("a second text frame")
        with pytest.raises(OSError):
            receive(stream)
    finally:
        stream.close()


def test_gateway_missing_token_refused_before_dial(board, tmp_path, monkeypatch):
    monkeypatch.delenv("BOTLAB_HIL_TOKEN")
    host = HilHost(tmp_path / "project", selected_target="gw")
    try:
        with pytest.raises(HilError, match="BOTLAB_HIL_TOKEN"):
            host.handle("POST", "/hil/connect", {"target": "gw"})
    finally:
        host.close()


def test_gateway_device_restart_discards_stale_results(board, gateway, tmp_path):
    endpoint, server_gateway = gateway
    host = HilHost(tmp_path / "project", selected_target="gw")
    try:
        host.handle("POST", "/hil/connect", {"target": "gw"})
        stale_session = host.handle("POST", "/hil/start", {})["session"]

        # Device restart: the old agent is replaced by a fresh process whose
        # AgentState knows nothing about the old session. Re-registration
        # displaces the paired coordinator connection as well.
        board.close()
        wait_unregistered(server_gateway)
        fresh = AgentThread(endpoint, DEVICE, tmp_path / "models2")
        wait_registered(server_gateway)
        deadline = time.monotonic() + 5
        while True:
            try:
                host.handle("POST", "/hil/start", {})
            except HilError:
                break  # old link is gone, as intended
            assert time.monotonic() < deadline, "old link stayed usable after device restart"

        # A new coordinator pairing starts clean, rejects stale inference and
        # terminates the lease — exactly like the TCP transport would.
        replaced = server_gateway.agents[DEVICE]
        stream = dial(endpoint)
        try:
            send(stream, {"op": "hello"})
            assert receive(stream)["ok"] is True
            send(stream, {"op": "start"})
            session = receive(stream)["session"]
            assert session != stale_session
            content = b"echo-model-bytes"
            digest = hashlib.sha256(content).hexdigest()
            send(stream, {"op": "load", "session": session, "sha256": digest,
                          "data": base64.b64encode(content).decode()})
            assert receive(stream)["ok"] is True
            send(stream, {"op": "infer", "session": stale_session, "seq": 99,
                          "model": digest,
                          "inputs": {"x": {"dtype": "float32", "shape": [2], "data": [0, 0]}}})
            result = receive(stream)
            assert result["ok"] is False and "Stale" in result["error"]
        finally:
            stream.close()

        # The rejection killed the device link; the resident agent redials and
        # serves a clean pairing again.
        wait_replaced(server_gateway, replaced)
        stream = dial(endpoint)
        try:
            send(stream, {"op": "hello"})
            assert receive(stream)["ok"] is True
            send(stream, {"op": "start"})
            session = receive(stream)["session"]
            assert session != stale_session
            content = b"echo-model-bytes"
            digest = hashlib.sha256(content).hexdigest()
            send(stream, {"op": "load", "session": session, "sha256": digest,
                          "data": base64.b64encode(content).decode()})
            assert receive(stream)["ok"] is True
            send(stream, {"op": "infer", "session": session, "seq": 0,
                          "model": digest,
                          "inputs": {"x": {"dtype": "float32", "shape": [2], "data": [1, 1]}}})
            assert receive(stream)["outputs"]["x"]["data"] == [2.0, 2.0]
        finally:
            stream.close()
            fresh.close()
    finally:
        host.close()


def test_gateway_endpoint_rules():
    assert gateway_endpoint("ws://127.0.0.1:50060") == "ws://127.0.0.1:50060"
    assert gateway_endpoint("wss://boards.example.com") == "wss://boards.example.com"
    assert gateway_endpoint("wss://Boards.Example.COM:8443/") == "wss://boards.example.com:8443"
    for rejected in ("ws://10.0.0.5:50060", "wss://gw.example.com/device/x",
                     "http://127.0.0.1:50060", "ws://127.0.0.1",
                     "wss://user:pw@gw.example.com", "ftp://gw.example.com",
                     "wss://gw.example.com:99999", "ws://[::1]:50060/x", ""):
        with pytest.raises(HilError):
            gateway_endpoint(rejected)


def test_gateway_target_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    target = save_target({"id": "x5", "name": "Desk X5", "transport": "gateway",
                          "endpoint": "wss://Boards.Example.com", "device": "rdk-x5-desk"})
    assert target["endpoint"] == "wss://boards.example.com"
    assert target["device"] == "rdk-x5-desk"
    # Credentials never persist in target settings.
    assert "password" not in target and "token" not in target
    with pytest.raises(HilError):
        save_target({"id": "bad", "name": "Bad", "transport": "gateway",
                     "endpoint": "ws://192.168.1.10:50060", "device": "x"})
    with pytest.raises(HilError):
        save_target({"id": "bad2", "name": "Bad", "transport": "gateway",
                     "endpoint": "wss://boards.example.com", "device": "no spaces!"})


@pytest.fixture(scope="module")
def tls_authority(tmp_path_factory) -> Path:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required for gateway TLS tests")
    root = tmp_path_factory.mktemp("pki")

    def run(*args: str) -> None:
        subprocess.run(["openssl", *args], check=True, capture_output=True, cwd=root)

    # Python 3.13 verifies with VERIFY_X509_STRICT by default: a CA without
    # basicConstraints/keyUsage is rejected ("CA cert does not include key usage
    # extension"), so the fixture CA carries the extensions a real CA has.
    run("req", "-x509", "-newkey", "rsa:2048", "-keyout", "ca.key", "-out", "ca.crt",
        "-days", "2", "-nodes", "-subj", "/CN=BotLab HIL Test CA",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    run("req", "-newkey", "rsa:2048", "-keyout", "server.key", "-out", "server.csr",
        "-nodes", "-subj", "/CN=localhost")
    (root / "san.ext").write_text("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
        "-CAcreateserial", "-out", "server.crt", "-days", "2", "-extfile", "san.ext")
    return root


def test_gateway_tls_verified(tls_authority, tmp_path, monkeypatch):
    # websockets' sync TLS opening handshake stalls with small probability
    # (upstream race, reproduced without any BotLab code). A stalled dial
    # surfaces as a clean single-attempt HilError; production absorbs it at
    # the layers that own retrying (the agent redials, the host user
    # reconnects). This test retries the whole scenario once because the
    # server side of a stalled attempt recovers as soon as the client goes
    # away — anything else failing still fails the test.
    for attempt in (1, 2):
        try:
            _tls_scenario(tls_authority, tmp_path, monkeypatch)
        except HilError as exc:
            if attempt == 2 or "timed out" not in str(exc):
                raise
        else:
            return


def _tls_scenario(tls_authority, tmp_path, monkeypatch):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tls_authority / "server.crt", tls_authority / "server.key")
    server_gateway = Gateway(TOKEN)
    server = ws_serve(server_gateway.handler, "127.0.0.1", 0, ssl=context,
                      max_size=64 * 1024 * 1024, ping_interval=5, ping_timeout=5)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"wss://localhost:{server.socket.getsockname()[1]}"
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    monkeypatch.setenv("BOTLAB_HIL_TOKEN", TOKEN)
    monkeypatch.delenv("BOTLAB_HIL_CA_BUNDLE", raising=False)
    try:
        # Without the dev CA bundle the TLS handshake must fail outright.
        with pytest.raises(HilError, match="Gateway connection failed"):
            connect_gateway(endpoint, "agent", DEVICE, TOKEN)

        monkeypatch.setenv("BOTLAB_HIL_CA_BUNDLE", str(tls_authority / "ca.crt"))
        agent = AgentThread(endpoint, DEVICE, tmp_path / "models")
        # A stalled TLS opening handshake burns its whole 5 s timeout before
        # the agent's next dial; leave room for a few of those under load.
        wait_registered(server_gateway, timeout=25)
        save_target({"id": "tls", "name": "TLS gateway", "transport": "gateway",
                     "endpoint": endpoint, "device": DEVICE, "backend": "echo"})
        host = HilHost(tmp_path / "project", selected_target="tls")
        try:
            assert host.handle("POST", "/hil/connect", {"target": "tls"})["connected"] is True
            host.handle("POST", "/hil/start", {})
            digest = load_echo_model(host)
            assert run_inference(host, digest, 0)["outputs"]["x"]["data"] == [2.0, 5.0]
        finally:
            host.close()
            agent.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
