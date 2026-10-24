"""Generic target agent for a trusted, isolated development network.

This MVP transport has no network authentication or encryption. Bind to the
explicit development interface; do not expose the agent to untrusted networks.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import platform
import socket
import socketserver
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .artifacts import ProjectStore
from .protocol import MAX_MODEL_BYTES, SHA256, HilError, ports, receive, send, tensors


class AgentState:
    def __init__(self, models_dir: Path, backend, *, root: Path | None = None):
        self.models_dir = Path(models_dir).expanduser().resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.lock = threading.RLock()
        self.owner = None
        self.session = None
        self.sequence = -1
        self.models = {}
        self.calls = 0
        self.store = ProjectStore(root if root is not None else self.models_dir.parent / "botlab")
        self.loaded = None
        self.project_owner = None

    def device(self):
        return {"hostname": platform.node(), "machine": platform.machine(),
                "protocol": 1, "calls": self.calls, **self.backend.describe(),
                "capabilities": {"projects": 1, "deploymentVersions": 1}, "loaded": self.loaded,
                "state": "running" if self.session else "loaded" if self.loaded else "idle"}

    def _release(self):
        session = self.session
        models, self.models = self.models, {}
        self.owner = self.session = None
        self.sequence = -1
        if session and self.loaded:
            try:
                self.store.log(self.loaded["project_id"], session, "stopped", calls=self.calls)
            except OSError:
                pass  # Disk-full logs must never prevent releasing the runtime.
        if callable(getattr(self.backend, "reset", None)):
            self.backend.reset()
            return
        for handle, _, _ in models.values():
            try:
                self.backend.close(handle)
            except Exception:
                pass

    def disconnect(self, owner):
        with self.lock:
            if self.owner is owner:
                self._release()
            if self.project_owner is owner:
                self.loaded = self.project_owner = None

    def dispatch(self, owner, request):
        if request.get("op") == "infer":
            return self._infer(owner, request)
        with self.lock:
            op = request.get("op")
            if op == "hello":
                return {"device": self.device()}
            if op == "projects.list":
                projects = self.store.list()
                for project in projects:
                    if self.loaded and project["id"] == self.loaded["project_id"]:
                        project["state"] = "running" if self.session else "loaded"
                return {"projects": projects, "loaded": self.loaded}
            if isinstance(op, str) and op.startswith("projects."):
                return self._project_operation(owner, request)
            if op == "start":
                if self.owner is not None:
                    raise HilError("Target already has an active session; stop it before starting another")
                if self.project_owner is not None and self.project_owner is not owner:
                    raise HilError("Target project is loaded by another connection")
                self.owner, self.session = owner, uuid.uuid4().hex
                self.sequence = -1
                if self.loaded:
                    self.store.log(self.loaded["project_id"], self.session, "started",
                                   build_id=self.loaded["build_id"])
                return {"session": self.session, "device": self.device()}
            if self.owner is not owner or not self.session or request.get("session") != self.session:
                raise HilError("Stale or inactive HIL session")
            if op == "stop":
                self._release()
                return {"stopped": True}
            if op == "load":
                return self._load(request)
            raise HilError("Unknown HIL operation")

    def _project_operation(self, owner, request):
        if self.owner is not None:
            raise HilError("Stop the active session before managing target projects")
        op = request["op"]
        # Independent inventory connections may clean up unused deployments.
        # Only removal gets this exception; selection and execution keep their
        # existing ownership lease. The loaded release is protected below.
        if self.project_owner is not None and self.project_owner is not owner and op != "projects.remove":
            raise HilError("Target project is loaded by another connection")
        if op == "projects.deploy":
            backend = self.backend.describe().get("backend")
            manifest = request.get("artifact", {}).get("manifest", {})
            if manifest.get("backend") != backend:
                raise HilError(f"Deployment backend does not match target backend {backend}")
            return self.store.deploy(request.get("artifact"))
        if op == "projects.load":
            manifest = self.store.manifest(request.get("project_id"), request.get("build_id"),
                                           verify_files=True)
            if manifest["backend"] != self.backend.describe().get("backend"):
                raise HilError("Deployment backend does not match target backend")
            selection = {"project_id": manifest["project"]["id"], "build_id": manifest["build_id"],
                         "name": manifest["project"]["name"], "models": manifest["models"],
                         "project_digest": manifest["project_digest"]}
            self.store.select(selection["project_id"], selection["build_id"])
            self.loaded, self.project_owner = selection, owner
            return {"loaded": self.loaded}
        if op == "projects.unload":
            self.loaded = self.project_owner = None
            return {"loaded": None}
        if op == "projects.remove":
            project_id, build_id = request.get("project_id"), request.get("build_id")
            if self.loaded and self.loaded["project_id"] == project_id and (
                build_id is None or self.loaded["build_id"] == build_id
            ):
                raise HilError("Unload the project before deleting its loaded release")
            return self.store.remove(project_id, build_id)
        raise HilError("Unknown project operation")

    def _infer(self, owner, request):
        with self.lock:
            if self.owner is not owner or not self.session or request.get("session") != self.session:
                raise HilError("Stale or inactive HIL session")
            seq = request.get("seq")
            if type(seq) is not int or seq < 0 or seq <= self.sequence:
                raise HilError("Inference sequence must increase within a session")
            model = request.get("model")
            if model not in self.models:
                raise HilError("Model is not loaded in this session")
            handle, inputs, outputs = self.models[model]
            feed = tensors(request.get("inputs"), inputs)
            self.sequence = seq
            session = self.session
        # Native backend execution must never hold the management lock. The CLI
        # backend is a killable process with a bounded inference deadline.
        started = time.perf_counter()
        result = self.backend.infer(handle, feed)
        elapsed = (time.perf_counter() - started) * 1000
        with self.lock:
            if self.session != session or self.owner is not owner:
                raise HilError("Inference finished after its session was stopped")
            result = tensors(result, outputs)
            self.calls += 1
            return {"session": session, "seq": seq, "outputs": result,
                    "inference_ms": elapsed, "calls": self.calls}

    def _load(self, request):
        digest, encoded = request.get("sha256"), request.get("data")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise HilError("Model identity must be a SHA-256 digest")
        if digest in self.models:
            _, inputs, outputs = self.models[digest]
            return {"sha256": digest, "inputs": inputs, "outputs": outputs}
        if len(self.models) >= 32:
            raise HilError("A session may load at most 32 models")
        if self.loaded:
            entry = next((item for item in self.loaded["models"] if item["sha256"] == digest), None)
            if entry is None:
                raise HilError("Model is not part of the loaded deployment; build and deploy this project again")
            self.store.manifest(self.loaded["project_id"], self.loaded["build_id"], verify_files=True)
            path = self.store.release_path(self.loaded["project_id"], self.loaded["build_id"]) / entry["path"]
            return self._load_path(digest, path)
        if not isinstance(encoded, str) or len(encoded) > ((MAX_MODEL_BYTES + 2) // 3) * 4:
            raise HilError("Model exceeds the 32 MiB limit")
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise HilError("Invalid model base64") from exc
        if not content or len(content) > MAX_MODEL_BYTES or hashlib.sha256(content).hexdigest() != digest:
            raise HilError("Model SHA-256 verification failed")
        path = self.models_dir / (digest + ".onnx")
        if path.is_symlink():
            raise HilError("Model cache path must not be a symlink")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            usage = sum(p.stat().st_size for p in self.models_dir.glob("*.onnx") if not p.is_symlink())
            if usage + len(content) > 256 * 1024 * 1024:
                raise HilError("Model cache exceeds 256 MiB; remove unused models on the target")
            fd, temporary = tempfile.mkstemp(dir=self.models_dir, prefix=".upload-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return self._load_path(digest, path)

    def _load_path(self, digest, path):
        handle, inputs, outputs = self.backend.load(path)
        try:
            ports(inputs)
            ports(outputs)
        except HilError:
            self.backend.close(handle)
            raise
        self.models[digest] = (handle, inputs, outputs)
        return {"sha256": digest, "inputs": inputs, "outputs": outputs}


class AgentServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, state: AgentState):
        self.state = state
        super().__init__(address, AgentHandler)


def serve_stream(state: AgentState, stream) -> None:
    """Run the dispatch loop over any socket-shaped stream (TCP or gateway)."""
    stream.settimeout(120)
    while True:
        message = receive(stream)
        try:
            result = state.dispatch(stream, message)
            send(stream, {"ok": True, **result})
        except Exception as exc:
            # Selection validates its candidate before changing ownership.
            # A stale or invalid candidate can therefore be rejected
            # without unloading the previously selected project. All
            # other execution/contract failures terminate the lease.
            preserved = message.get("op") == "projects.load" and isinstance(exc, HilError)
            send(stream, {"ok": False, "error": str(exc),
                          **({"lease_preserved": True} if preserved else {})})
            if not preserved:
                return


class AgentHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(120)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            # serve_stream keys the lease on the stream itself.
            serve_stream(self.server.state, self.request)
        except (OSError, EOFError, ValueError):
            pass
        finally:
            self.server.state.disconnect(self.request)


def _connect(options) -> int:
    """Resident outbound mode: dial the gateway and serve frames over WebSocket."""
    from .worker import WorkerBackend
    from .wsstream import connect_gateway

    root = options.root.expanduser()
    state = AgentState(options.models_dir or root / "cache" / "models",
                       WorkerBackend(options.backend, inference_timeout=options.inference_timeout), root=root)
    token = os.environ.get("BOTLAB_HIL_TOKEN") or getpass.getpass("Gateway token: ")
    print(f"BotLab HIL agent: dialing out to {options.gateway} as device {options.device}", flush=True)
    while True:
        try:
            stream = connect_gateway(options.gateway, "agent", options.device, token)
        except HilError as exc:
            print(f"Gateway connection failed: {exc}", file=sys.stderr, flush=True)
            if options.retry_interval <= 0:
                return 1
        else:
            print(f"BotLab HIL agent: registered as {options.device} ({options.backend})", flush=True)
            try:
                serve_stream(state, stream)
            except (OSError, EOFError, ValueError):
                print("BotLab HIL agent: gateway link dropped", file=sys.stderr, flush=True)
            finally:
                state.disconnect(stream)
                stream.close()
        if options.retry_interval <= 0:
            return 0
        time.sleep(options.retry_interval)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["connect"]:
        parser = argparse.ArgumentParser(
            description="Dial out to a BotLab HIL gateway and serve the device")
        parser.add_argument("--gateway", required=True, help="wss:// gateway origin (ws:// loopback only)")
        parser.add_argument("--device", required=True, help="device ID to register as")
        parser.add_argument("--root", type=Path, default=Path("~/botlab"), help="Agent and project storage root")
        parser.add_argument("--models-dir", type=Path, help="Legacy model cache (default: ROOT/cache/models)")
        parser.add_argument("--backend", default="onnxruntime", help="onnxruntime or module:factory")
        parser.add_argument("--inference-timeout", type=float, default=3.0,
                            help="Backend inference deadline in seconds")
        parser.add_argument("--retry-interval", type=float, default=5.0,
                            help="seconds between gateway reconnection attempts; 0 exits on loss")
        options = parser.parse_args(argv[1:])
        return _connect(options)
    if argv[:1] == ["serve"]:
        argv.pop(0)
    parser = argparse.ArgumentParser(description="Serve a BotLab HIL target on a trusted network")
    parser.add_argument("--host", required=True, help="Explicit development-interface address")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--root", type=Path, default=Path("~/botlab"), help="Agent and project storage root")
    parser.add_argument("--models-dir", type=Path, help="Legacy model cache (default: ROOT/cache/models)")
    parser.add_argument("--backend", default="onnxruntime", help="onnxruntime or module:factory")
    parser.add_argument("--inference-timeout", type=float, default=3.0,
                        help="Backend inference deadline in seconds")
    options = parser.parse_args(argv)
    from .worker import WorkerBackend
    root = options.root.expanduser()
    state = AgentState(options.models_dir or root / "cache" / "models",
                       WorkerBackend(options.backend, inference_timeout=options.inference_timeout), root=root)
    with AgentServer((options.host, options.port), state) as server:
        print(f"BotLab HIL agent: {options.host}:{server.server_address[1]} ({options.backend})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            with state.lock:
                state._release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
