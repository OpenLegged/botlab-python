"""Contract tests use an arbitrary multi-input target, independent of any robot."""
import base64
import hashlib
import socket
import struct
import threading
from pathlib import Path

import pytest

from botlab.hil import HilError, HilHost
from botlab.hil.agent import AgentServer, AgentState
from botlab.hil.protocol import MAX_FRAME_BYTES, endpoint_address, receive, tensor
from botlab.hil.targets import read_targets, save_target


class ArithmeticBackend:
    closed = 0

    def describe(self):
        return {"backend": "arithmetic", "provider": "test"}

    def load(self, path):
        assert path.read_bytes() == b"test-model"
        def spec(name):
            return {"name": name, "dtype": "float32", "shape": [2, 3]}
        return object(), [spec("left"), spec("right")], [spec("sum"), spec("difference")]

    def infer(self, handle, inputs):
        left, right = inputs["left"]["data"], inputs["right"]["data"]
        return {"sum": values([a + b for a, b in zip(left, right, strict=True)]),
                "difference": values([a - b for a, b in zip(left, right, strict=True)])}

    def close(self, handle):
        self.closed += 1


def values(data=None):
    return {"dtype": "float32", "shape": [2, 3], "data": data or [1, 2, 3, 4, 5, 6]}


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    backend = ArithmeticBackend()
    state = AgentState(tmp_path / "models", backend)
    server = AgentServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = save_target({"id": "test", "name": "Test", "backend": "arithmetic",
                          "endpoint": f"tcp://127.0.0.1:{server.server_address[1]}"})
    host = HilHost(tmp_path, target["id"])
    try:
        yield host, state, target
    finally:
        host.close()
        server.shutdown()
        server.server_close()
        thread.join()


def prepare(host):
    assert host.handle("POST", "/hil/connect", {"target": "test"})["connected"]
    session = host.handle("POST", "/hil/start")["session"]
    content = b"test-model"
    digest = hashlib.sha256(content).hexdigest()
    model = host.handle("POST", "/hil/models", {"session": session, "sha256": digest,
                                                "data": base64.b64encode(content).decode()})
    return session, model


def infer(host, session, model, seq=0):
    return host.handle("POST", "/hil/infer", {"session": session, "seq": seq,
        "model": model["sha256"], "inputs": {"left": values(), "right": values([2] * 6)}})


def test_multi_input_multi_output_live_transport(live):
    host, state, _ = live
    session, model = prepare(host)
    assert [p["name"] for p in model["inputs"]] == ["left", "right"]
    result = infer(host, session, model)
    assert result["outputs"]["sum"]["data"] == [3, 4, 5, 6, 7, 8]
    assert result["outputs"]["difference"]["data"] == [-1, 0, 1, 2, 3, 4]
    assert result["calls"] == state.calls == 1
    assert result["rtt_ms"] >= result["inference_ms"] >= 0
    host.handle("POST", "/hil/stop", {"session": session})
    assert state.backend.closed == 1
    assert state.owner is None


def test_rejects_replayed_and_stale_steps(live):
    host, state, _ = live
    session, model = prepare(host)
    infer(host, session, model, 7)
    with pytest.raises(HilError, match="sequence"):
        infer(host, session, model, 7)
    assert state.calls == 1
    host.handle("POST", "/hil/stop", {"session": session})
    new_session = host.handle("POST", "/hil/start")["session"]
    assert new_session != session
    with pytest.raises(HilError, match="Stale"):
        infer(host, session, model, 8)
    with pytest.raises(HilError, match="Stale"):
        host.handle("POST", "/hil/stop", {"session": session})
    assert state.session == new_session


def test_agent_enforces_exclusive_lease(live):
    host, state, target = live
    prepare(host)
    other = HilHost(host.project_dir, target["id"])
    try:
        other.handle("POST", "/hil/connect", {"target": target["id"]})
        with pytest.raises(HilError, match="active session"):
            other.handle("POST", "/hil/start")
        assert state.session == host.session
    finally:
        other.close()


def test_hash_mismatch_closes_and_releases_session(live):
    host, state, _ = live
    session, _ = prepare(host)
    with pytest.raises(HilError, match="SHA-256"):
        host.handle("POST", "/hil/models", {"session": session, "sha256": "0" * 64,
                                            "data": base64.b64encode(b"wrong").decode()})
    assert not host.state()["connected"]
    # Joining the state lock waits for a potentially finishing server request.
    with state.lock:
        pass
    # Reconnect and acquire a fresh session verifies release on the server.
    host.handle("POST", "/hil/connect", {"target": "test"})
    assert host.handle("POST", "/hil/start")["session"] != session


def test_inference_failure_is_not_retried(live):
    host, state, _ = live
    session, model = prepare(host)
    calls = []
    def fail(handle, feeds):
        calls.append(1)
        raise RuntimeError("backend failure")
    state.backend.infer = fail
    with pytest.raises(HilError, match="backend failure"):
        infer(host, session, model)
    assert calls == [1]
    assert state.calls == 0
    assert not host.state()["connected"]


@pytest.mark.parametrize("shape,data", [([3], [1, 2]), ([2, 3], [float("nan")] * 6),
                                      ([2, 3], [1e40] * 6), ([True], [1]), ([-1], [])])
def test_invalid_tensors(shape, data):
    with pytest.raises(HilError):
        tensor({"dtype": "float32", "shape": shape, "data": data})


def test_shape_and_names_checked_before_execution(live):
    host, state, _ = live
    session, model = prepare(host)
    with pytest.raises(HilError, match="names"):
        host.handle("POST", "/hil/infer", {"session": session, "seq": 1,
            "model": model["sha256"], "inputs": {"wrong": values()}})
    assert state.calls == 0
    with pytest.raises(HilError, match="shape"):
        tensor({"dtype": "float32", "shape": [6], "data": list(range(6))}, model["inputs"][0])


def test_frame_limit_rejected_before_payload_read():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with pytest.raises(HilError, match="frame limit"):
            receive(right)
    finally:
        left.close()
        right.close()


def test_target_settings_atomic_and_endpoint_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("BOTLAB_TARGETS_FILE", str(tmp_path / "targets.json"))
    target = save_target({"id": "board", "name": "Device", "endpoint": "tcp://192.0.2.1:50051"})
    save_target({**target, "endpoint": "tcp://192.0.2.2:50051"})
    assert read_targets() == [{**target, "endpoint": "tcp://192.0.2.2:50051"}]
    assert not list(tmp_path.glob(".targets-*"))
    for endpoint in ("https://host:4", "tcp://host", "tcp://user:pass@host:4", "tcp://host:4/path"):
        with pytest.raises(HilError):
            endpoint_address(endpoint)


def test_agent_validates_sequence_independently(tmp_path):
    state = AgentState(tmp_path, ArithmeticBackend())
    owner = object()
    session = state.dispatch(owner, {"op": "start"})["session"]
    data = b"test-model"
    digest = hashlib.sha256(data).hexdigest()
    state.dispatch(owner, {"op": "load", "session": session, "sha256": digest,
                           "data": base64.b64encode(data).decode()})
    request = {"op": "infer", "session": session, "seq": 0, "model": digest,
               "inputs": {"left": values(), "right": values()}}
    state.dispatch(owner, request)
    with pytest.raises(HilError, match="sequence"):
        state.dispatch(owner, request)
    with pytest.raises(HilError, match="Stale"):
        state.dispatch(object(), {**request, "seq": 1})
    state.disconnect(owner)
    assert state.backend.closed == 1
    assert state.owner is None


def test_builtin_ort_uses_real_inference_and_rejects_external_weights(tmp_path):
    pytest.importorskip("onnxruntime")
    from botlab.hil.backends import create_backend
    backend = create_backend("onnxruntime")
    assets = Path(__file__).parent / "assets"
    handle, inputs, outputs = backend.load(assets / "identity.onnx")
    shape = inputs[0]["shape"]
    import math
    data = [float(i) for i in range(math.prod(shape))]
    result = backend.infer(handle, {inputs[0]["name"]: {"dtype": "float32", "shape": shape, "data": data}})
    assert result[outputs[0]["name"]]["data"] == data
    backend.close(handle)
    # External tensors are intentionally unsupported; only the uploaded artifact is readable.
    with pytest.raises(HilError, match="External ONNX tensor"):
        backend.load(assets / "external_weights.onnx")


@pytest.mark.parametrize("message", [b"\x3a\x03\x2a\x01\x68", b"\x3a\xff", b"\x00"])
def test_embedded_onnx_validation_rejects_malformed_or_external(message):
    from botlab.hil._onnx import require_embedded_tensors
    with pytest.raises(HilError):
        require_embedded_tensors(message)


def test_embedded_onnx_validation_checks_nested_attribute_tensors():
    from botlab.hil._onnx import require_embedded_tensors
    def message(tag, child):
        return bytes([tag * 8 + 2, len(child)]) + child
    # Model.graph -> Graph.node -> Node.attribute -> Attribute.t -> external_data.
    external = message(7, message(1, message(5, message(5, b"\x6a\x00"))))
    with pytest.raises(HilError, match="External"):
        require_embedded_tensors(external)
