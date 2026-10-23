"""Stdlib-only HIL coordinator used by the local BotLab HTTP server."""
from __future__ import annotations

import copy
import re
import socket
import threading
import time
import uuid
from pathlib import Path

from .protocol import (
    SHA256,
    HilError,
    endpoint_address,
    error_details,
    ports,
    receive,
    send,
    tensors,
)
from .targets import read_targets, remove_target, save_target, validate_target


class _StartCancelled(HilError):
    pass


class HilHost:
    def __init__(self, project_dir: Path, selected_target: str | None = None):
        self.project_dir = Path(project_dir).resolve()
        self.lock = threading.RLock()
        self.targets = read_targets()
        self.selected_target = selected_target
        if selected_target and not any(t["id"] == selected_target for t in self.targets):
            endpoint_address(selected_target)
            target = validate_target({"id": "command-line", "name": "Command-line target",
                                      "endpoint": selected_target})
            self.targets.append(target)
            self.selected_target = target["id"]
        self.sock = None
        self.device = None
        self.session = None
        self.error = None
        self.metrics = {"calls": 0, "inference_ms": 0, "rtt_ms": 0}
        self.last_activity = 0.0
        self.sequence = -1
        self.models = {}
        self.tunnel = None
        self.credentials = {}
        self.credential_targets = {}
        self.device_info = {}
        self.network_pending = {}
        self.artifact = None
        self.application = {"built": None, "loaded": None, "projects": []}
        self.operation_lock = threading.RLock()
        self.operations = {}
        self.operation = None
        self.cancelled_starts = {}
        self.project_generation = 0
        self.closed = False

    def state(self):
        return {"available": True, "targets": [dict(item) for item in self.targets],
                "selectedTarget": self.selected_target, "connected": self.sock is not None,
                "device": self.device, "session": self.session,
                "metrics": dict(self.metrics), "error": self.error,
                "application": copy.deepcopy(self.application),
                "device_info": copy.deepcopy(self.device_info),
                "credential_status": {target["id"]: {"password": bool(
                    self._credentials_for(target).get("password"))} for target in self.targets},
                "capabilities": {"targetManagement": 1, "projects": 1, "ssh": True,
                                 "targetProjects": 1, "transientCredentials": 1, "credentialPrompt": 1,
                                 "stepwiseExecution": 1, "deploymentVersions": 1},
                "operation": self._operation_snapshot()}

    def _drop(self):
        sock, self.sock = self.sock, None
        self.session = None
        self.models = {}
        self.sequence = -1
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self.tunnel:
            self.tunnel.close()
            self.tunnel = None
        self.application["loaded"] = None
        self.project_generation += 1

    def _rpc(self, message, timeout=5):
        if not self.sock:
            raise HilError("Connect a target first")
        started = time.perf_counter()
        try:
            self.sock.settimeout(timeout)
            send(self.sock, message)
            result = receive(self.sock)
            if result.get("ok") is not True and not (
                message.get("op") == "projects.load" and result.get("lease_preserved") is True
            ):
                raise HilError(result.get("error") or "Target request failed")
            self.last_activity = time.monotonic()
        except (OSError, ValueError, EOFError) as exc:
            self.error = self._redact(str(exc) or type(exc).__name__)
            self._drop()
            raise HilError(self.error) from exc
        # A failed project selection is validated before the Agent changes its
        # lease. Only this explicit response may preserve the current selection;
        # transport failures and every other RPC error still drop the connection.
        if result.get("ok") is not True:
            raise HilError(result.get("error") or "Target project selection failed")
        return result, (time.perf_counter() - started) * 1000

    def _check_session(self, body):
        if not self.session or body.get("session") != self.session:
            raise HilError("Stale or inactive HIL session")

    def _operation_snapshot(self, operation_id=None):
        with self.operation_lock:
            item = self.operations.get(operation_id) if operation_id else self.operation
            if item is None:
                return None
            return copy.deepcopy({k: v for k, v in item.items()
                                  if k != "result" and not k.startswith("_")})

    def _redact(self, message):
        passwords = {credential["password"] for credential in self.credentials.values()
                     if isinstance(credential.get("password"), str) and credential["password"]}
        for password in sorted(passwords, key=len, reverse=True):
            message = message.replace(password, "[redacted]")
        return message

    def _progress(self, value):
        with self.operation_lock:
            if self.operation:
                self.operation["phase"] = value.get("stage", "working")
                self.operation["messages"].append(self._redact(str(value.get("message", ""))))
                # A total count lets polling clients distinguish repeated lines
                # and locate new messages after this bounded tail rolls over.
                self.operation["message_count"] += 1
                self.operation["messages"] = self.operation["messages"][-100:]

    def _job(self, kind, callback, *, request_id=None):
        with self.operation_lock:
            if self.operation and self.operation["status"] == "running":
                raise HilError("Wait for the current target operation to finish")
            operation = {"id": uuid.uuid4().hex, "kind": kind, "status": "running",
                         "phase": "starting", "messages": [], "message_count": 0, "error": None}
            if request_id is not None:
                self._validate_start_request(request_id)
                if any(item.get("request_id") == request_id for item in self.operations.values()):
                    raise HilError("This HIL start request already exists")
                operation["request_id"] = request_id
            self.operations[operation["id"]] = self.operation = operation
            while len(self.operations) > 20:
                del self.operations[next(iter(self.operations))]
        def run():
            try:
                with self.lock:
                    if self.closed:
                        raise HilError("HIL host is closed")
                    result = callback()
                    if self.closed:
                        self._drop()
                        raise HilError("HIL host was closed during the operation")
                    # Publish success under the same cancellation lock. A cancel
                    # arriving after this point uses the exact completed lease.
                    with self.operation_lock:
                        self._check_start_cancelled(request_id)
                        operation.update(status="succeeded", phase="complete", result=result)
            except Exception as exc:
                if isinstance(exc, _StartCancelled):
                    with self.lock:
                        self._release_cancelled_start(operation)
                message = self._redact(str(exc) or type(exc).__name__)
                details = error_details(exc)
                # Authentication is recoverable through the browser credential
                # prompt, not a terminal connection failure (including Cancel).
                self.error = None if details else message
                with self.operation_lock:
                    operation.update(status="failed", phase="cancelled" if isinstance(exc, _StartCancelled)
                                     else "failed", error=message, **details)
        threading.Thread(target=run, name="botlab-" + kind, daemon=True).start()
        return {"operation": self._operation_snapshot(operation["id"])}

    @staticmethod
    def _validate_start_request(request_id):
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", request_id):
            raise HilError("HIL start request_id must be a unique 16–128 character identifier")

    def _check_start_cancelled(self, request_id):
        if request_id is None:
            return  # Compatibility with older clients and direct CLI callers.
        with self.operation_lock:
            if request_id in self.cancelled_starts or (
                self.operation and self.operation.get("request_id") == request_id
                and self.operation.get("_cancel_requested")
            ):
                raise _StartCancelled("HIL start was cancelled")

    def _release_cancelled_start(self, operation):
        """Caller owns the Host lock; never release a subsequent run or load."""
        session = operation.get("_started_session")
        try:
            if session and self.session == session:
                self._handle("POST", "/hil/stop", {"session": session})
            if (self.session is None and self.sock is not None
                    and operation.get("_loaded_generation") == self.project_generation
                    and self.application["loaded"] is not None):
                self._rpc({"op": "projects.unload"})
                self.application["loaded"] = None
                self.project_generation += 1
                self._projects()
        except (OSError, ValueError, EOFError):
            self._drop()  # TCP lease release also clears the Agent's loaded project.

    def _cancel_start(self, body):
        request_id = body.get("request_id")
        self._validate_start_request(request_id)
        with self.operation_lock:
            now = time.monotonic()
            # Retain cancellation-before-start races without unbounded history.
            self.cancelled_starts = {key: stamp for key, stamp in self.cancelled_starts.items()
                                     if now - stamp < 1200}
            self.cancelled_starts[request_id] = now
            while len(self.cancelled_starts) > 128:
                oldest = next(iter(self.cancelled_starts))
                if self.operation and oldest == self.operation.get("request_id"):
                    self.cancelled_starts[oldest] = self.cancelled_starts.pop(oldest)
                    continue
                del self.cancelled_starts[oldest]
            operation = next((item for item in self.operations.values()
                              if item.get("request_id") == request_id), None)
            pending = operation is not None and operation["status"] == "running"
            if operation is not None:
                operation["_cancel_requested"] = True
                if pending:
                    operation["phase"] = "cancelling"
        if operation is not None and not pending:
            with self.lock:
                self._release_cancelled_start(operation)
                with self.operation_lock:
                    operation.pop("result", None)
                    operation.update(status="failed", phase="cancelled", error="HIL start was cancelled")
        return {"cancelled": True, "request_id": request_id,
                "operation": self._operation_snapshot(operation["id"]) if operation else None}

    def _target(self, target_id=None):
        target_id = target_id or self.selected_target
        target = next((t for t in self.targets if t["id"] == target_id), None)
        if target is None:
            raise HilError("Choose a configured target")
        return target

    @staticmethod
    def _credential_identity(target):
        # Fingerprint acceptance verifies the same host; it must not discard the
        # password that was just entered in target configuration.
        identity = {key: copy.deepcopy(target.get(key)) for key in
                    ("transport", "endpoint", "root", "backend", "agent_port")}
        identity["ssh"] = {key: value for key, value in target.get("ssh", {}).items()
                           if key != "host_key"}
        return identity

    def _credentials_for(self, target):
        target_id = target["id"]
        bound = self.credential_targets.get(target_id)
        if bound is not None and bound != self._credential_identity(target):
            self.credentials.pop(target_id, None)
            self.credential_targets.pop(target_id, None)
        return dict(self.credentials.get(target_id, {}))

    def _store_credentials(self, target, credentials):
        self.credentials[target["id"]] = dict(credentials)
        self.credential_targets[target["id"]] = self._credential_identity(target)

    @staticmethod
    def _validate_credentials(credentials):
        if not isinstance(credentials, dict) or set(credentials) - {"password", "host_key"}:
            raise HilError("Invalid transient SSH credentials")
        if any(not isinstance(value, str) or len(value) > 4096 or "\x00" in value
               for value in credentials.values()):
            raise HilError("Invalid transient SSH credential")
        fingerprint = credentials.get("host_key")
        if fingerprint is not None and not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint):
            raise HilError("Invalid SSH host key fingerprint")
        return dict(credentials)

    def _device(self, body):
        from .devices import LinuxDevice
        target = self._target(body.get("target"))
        if target.get("transport") != "ssh":
            raise HilError("This operation requires an SSH target")
        credentials = self._credentials_for(target)
        for field in ("password", "host_key"):
            if body.get(field) is not None:
                credentials[field] = body[field]
        device = LinuxDevice(target, credentials, self._progress)
        self._store_credentials(target, credentials)
        return device

    def _inspect_device(self, body, provision=False):
        # Management jobs hold self.lock. Refuse an active session before any
        # device or connection changes; inspecting an idle target is read-only.
        if self.session:
            raise HilError("Stop the project before checking or installing the target")
        target = self._target(body.get("target"))
        if not provision and self.sock and target["id"] != self.selected_target:
            raise HilError("Disconnect before checking another target")
        self.device_info.pop(target["id"], None)
        device = self._device(body)
        try:
            if provision:
                self._drop()
            result = device.provision() if provision else device.inspect()
            self.device_info[device.target["id"]] = result
            self.selected_target = device.target["id"]
            self.error = None
            return result
        finally:
            device.close()

    def _build(self, expected_sources=None):
        if self.session:
            raise HilError("Stop the project before building or deploying")
        from .artifacts import build_project
        self._progress({"stage": "build", "message": "Validating the project and building its target application"})
        self.artifact = build_project(self.project_dir, expected_sources=expected_sources)
        self.application["built"] = self.artifact["manifest"]
        self.error = None
        return {"built": self.artifact["manifest"]}

    def _supports_projects(self):
        return bool(self.device and self.device.get("capabilities", {}).get("projects") == 1)

    def _projects(self):
        if not self._supports_projects():
            raise HilError("This target Agent does not support application deployment; initialize or update it first")
        result, _ = self._rpc({"op": "projects.list"})
        self.application.update(projects=result["projects"], loaded=result.get("loaded"))
        return result

    def _target_projects(self, body, *, remove=False):
        if self.session:
            raise HilError("Stop the project before managing target projects")
        target_id = body.get("target")
        if not isinstance(target_id, str) or not target_id:
            raise HilError("Choose a configured target to scan")
        target = self._target(target_id)
        if remove:
            expected = body.get("expected_target")
            if not isinstance(expected, dict) or validate_target(expected) != target:
                raise HilError("Target settings changed. Scan the target and confirm project deletion again.")
            if "build_id" in body and (
                not isinstance(body["build_id"], str) or not SHA256.fullmatch(body["build_id"])
            ):
                raise HilError("Choose a valid deployed version to delete")
            if not isinstance(body.get("project_id"), str) or not body["project_id"]:
                raise HilError("Choose a deployed project to delete")

        # Agent errors terminate their TCP lease. All browsing/removal gets a
        # separate connection so even a malformed inventory or refusal cannot
        # unload the main project or change the simulation's target selection.
        stream = tunnel = device = None
        try:
            self._progress({"stage": "connect", "message": "Opening target project management connection"})
            address = endpoint_address(target["endpoint"])
            if target.get("transport") == "ssh":
                device = self._device({"target": target_id})
                tunnel = device.open_tunnel()
                address = tunnel.address
            stream = socket.create_connection(address, timeout=3)
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def request(message):
                stream.settimeout(10)
                send(stream, message)
                result = receive(stream)
                if result.get("ok") is not True:
                    raise HilError(result.get("error") or "Target project request failed")
                return result

            hello = request({"op": "hello"})["device"]
            if hello.get("capabilities", {}).get("projects") != 1:
                raise HilError("Target Agent does not support project management. Install or update it from the main Connect action.")
            if hello.get("backend") != target["backend"]:
                raise HilError(f"Target backend is {hello.get('backend')}, expected {target['backend']}")
            if remove:
                if "build_id" in body and hello.get("capabilities", {}).get("deploymentVersions") != 1:
                    raise HilError("Target Agent does not support deleting individual versions. Update it from the main Connect action.")
                self._progress({"stage": "remove", "message": (
                    "Deleting the selected deployed version" if "build_id" in body
                    else "Deleting the selected target project and all releases")})
                request({"op": "projects.remove", "project_id": body["project_id"],
                         **({"build_id": body["build_id"]} if "build_id" in body else {})})
            self._progress({"stage": "scan", "message": "Reading deployed target projects"})
            result = request({"op": "projects.list"})
            if remove and self.sock and target_id == self.selected_target:
                self.application.update(projects=result["projects"], loaded=result.get("loaded"))
            self.error = None
            return {"target": target_id, "projects": result["projects"], "loaded": result.get("loaded")}
        finally:
            if stream:
                stream.close()
            if tunnel:
                tunnel.close()
            if device:
                device.close()

    def _deploy(self):
        if not self._supports_projects():
            raise HilError("Connect a target with application deployment support first")
        if self.artifact is None:
            raise HilError("Build the project before deploying")
        if self.session:
            raise HilError("Stop the project before deploying")
        self._progress({"stage": "deploy", "message": "Installing the application on the target"})
        result, _ = self._rpc({"op": "projects.deploy", "artifact": self.artifact}, timeout=120)
        self._projects()
        return result

    def _load_project(self, body):
        if self.session:
            raise HilError("Stop before loading another project")
        if not self._supports_projects():
            raise HilError("This target does not support deployed projects")
        scoped = "target" in body or "expected_target" in body
        if scoped:
            target_id = body.get("target")
            if not self.sock or target_id != self.selected_target:
                raise HilError("Target changed or disconnected. Connect the selected target before loading a version.")
            expected = body.get("expected_target")
            if not isinstance(expected, dict) or expected != self._target(target_id):
                raise HilError("Target settings changed. Scan the target and select the deployed version again.")
            if self.device.get("capabilities", {}).get("deploymentVersions") != 1:
                raise HilError("Target Agent does not support safe version switching. Update it from the main Connect action.")
            if not isinstance(body.get("project_id"), str) or not body["project_id"]:
                raise HilError("Choose a deployed project to load")
            if "build_id" in body and (
                not isinstance(body["build_id"], str) or not SHA256.fullmatch(body["build_id"])
            ):
                raise HilError("Choose a valid deployed version to load")
        built = self.application["built"]
        project_id = body.get("project_id") or (built and built["project"]["id"])
        build_id = body.get("build_id") or (built and built["build_id"] if not body.get("project_id") else None)
        if not project_id:
            raise HilError("Build and deploy the project before connecting it")
        if scoped:
            # Browse on an independent connection before changing the main
            # ownership lease. A stale UI selection must not unload its project.
            inventory = self._target_projects({"target": body["target"]})
            project = next((item for item in inventory["projects"] if item["id"] == project_id), None)
            if project is None:
                raise HilError("Project is no longer deployed. Scan the target and select another project.")
            build_id = build_id or project["current_build_id"]
            if not any(item["build_id"] == build_id for item in project["releases"]):
                raise HilError("Version is no longer deployed. Scan the target and select another version.")
        self._progress({"stage": "load", "message": "Loading the selected target project"})
        result, _ = self._rpc({"op": "projects.load", "project_id": project_id, "build_id": build_id}, timeout=60)
        self.application["loaded"] = result["loaded"]
        self.project_generation += 1
        self._projects()
        return result

    def _start_project(self, body):
        request_id = body.get("request_id")
        self._check_start_cancelled(request_id)
        self._build(body.get("sources"))
        self._check_start_cancelled(request_id)
        manifest = self.artifact["manifest"]
        self._deploy()
        self._check_start_cancelled(request_id)
        self._load_project({"project_id": manifest["project"]["id"], "build_id": manifest["build_id"]})
        with self.operation_lock:
            self.operation["_loaded_generation"] = self.project_generation
        self._check_start_cancelled(request_id)
        self._progress({"stage": "run", "message": "Starting target inference"})
        return self._start_session(body)

    def _start_loaded_project(self, body):
        from .artifacts import validate_loaded_project

        request_id = body.get("request_id")
        self._check_start_cancelled(request_id)
        if self.session:
            raise HilError("Stop the current session before starting another")
        loaded = self.application["loaded"]
        if not self.sock or not self._supports_projects() or loaded is None:
            raise HilError("Connect a deployed project before running it")
        if (body.get("project_id") != loaded["project_id"]
                or body.get("build_id") != loaded["build_id"]):
            raise HilError("The connected project version changed. Refresh the selection before running it")
        if not isinstance(body.get("sources"), dict):
            raise HilError("Open the matching local project and provide its browser sources before running it")
        self._progress({"stage": "validate", "message": "Checking local sources against the connected project version"})
        validate_loaded_project(self.project_dir, loaded, body["sources"])
        self._check_start_cancelled(request_id)
        self._progress({"stage": "run", "message": "Starting the connected project version"})
        # The selection predates this start. Cancellation releases only the new
        # session; it must never unload the user's explicitly selected project.
        return self._start_session(body)

    def _start_session(self, body):
        request_id = body.get("request_id")
        self._check_start_cancelled(request_id)
        result = self._handle("POST", "/hil/start", {"prepared": True})
        with self.operation_lock:
            self.operation["_started_session"] = result["session"]
        self._check_start_cancelled(request_id)
        return result

    def handle(self, method: str, path: str, body: dict | None = None) -> dict:
        body = {} if body is None else body
        if not isinstance(body, dict):
            raise HilError("HIL request body must be an object")
        if method == "POST" and path == "/hil/cancel-start":
            return self._cancel_start(body)
        if method == "GET" and path.startswith("/hil/operations/"):
            operation_id = path.rsplit("/", 1)[-1]
            with self.operation_lock:
                op = self.operations.get(operation_id)
                if not op:
                    raise HilError("Unknown target operation")
                return {"operation": self._operation_snapshot(operation_id),
                        **({"result": copy.deepcopy(op["result"])} if "result" in op else {})}
        if self.operation and self.operation["status"] == "running":
            if method == "GET" and path == "/hil/state":
                return self.state()
            raise HilError("Wait for the current target operation to finish")
        if method == "POST":
            jobs = {
                "/hil/build": ("build", self._build),
                "/hil/deploy": ("deploy", self._deploy),
                "/hil/projects/load": ("load", lambda: self._load_project(body)),
                "/hil/targets/projects": ("projects-scan", lambda: self._target_projects(body)),
                "/hil/targets/projects/remove": ("projects-remove", lambda: self._target_projects(body, remove=True)),
                "/hil/devices/inspect": ("inspect", lambda: self._inspect_device(body)),
                "/hil/devices/provision": ("provision", lambda: self._inspect_device(body, True)),
            }
            if path in jobs:
                return self._job(*jobs[path])
            if path == "/hil/devices/host-key":
                from .devices import inspect_host_key
                config = self._target(body.get("target")).get("ssh", {})
                return self._job("identity", lambda: inspect_host_key(config.get("host"), config.get("port", 22)))
            if path == "/hil/devices/discover":
                from .devices import discover_devices
                return self._job("discover", lambda: discover_devices(body.get("interface"), body.get("profile"), body.get("host")))
            if path == "/hil/devices/local-network":
                from .devices import configure_local_network
                return self._job("network", lambda: configure_local_network(
                    body.get("interface"), body.get("method"), body.get("address"), body.get("netmask"), body.get("gateway")))
            if path in ("/hil/devices/network", "/hil/devices/network/confirm"):
                def network():
                    if path.endswith("/confirm"):
                        from .devices import LinuxDevice
                        transaction = body.get("transaction")
                        pending = self.network_pending.get(transaction)
                        if not pending or pending["target"]["id"] != body.get("target"):
                            raise HilError("Unknown network change for this target")
                        new_host = body.get("host") or pending["new_host"]
                        if not new_host:
                            raise HilError("Discover and supply the target's new address before confirming")
                        target = copy.deepcopy(pending["target"])
                        target["ssh"]["host"] = new_host
                        credentials = dict(self.credentials.get(target["id"], {}))
                        credentials.update({k: body[k] for k in ("password", "host_key") if body.get(k) is not None})
                        device = LinuxDevice(target, credentials, self._progress)
                        try:
                            result = device.confirm_network(transaction)
                            if credentials.get("host_key"):
                                target["ssh"]["host_key"] = credentials["host_key"]
                            save_target(target)
                            self.targets = read_targets()
                            self._store_credentials(target, credentials)
                            self.device_info.pop(target["id"], None)
                            self.network_pending.pop(transaction, None)
                            return result
                        finally:
                            device.close()
                    device = self._device(body)
                    try:
                        if self.session:
                            raise HilError("Stop the project before configuring the target network")
                        self._drop()
                        result = device.configure_network(body.get("interface"), body.get("method"), body.get("address"), body.get("gateway"), body.get("netmask"))
                        self.network_pending[result["transaction"]] = {"target": copy.deepcopy(device.target), "new_host": result.get("new_host")}
                        return result
                    finally:
                        device.close()
                return self._job("network", network)
            if path == "/hil/connect" and self._target(body.get("target")).get("transport") == "ssh":
                return self._job("connect", lambda: self._handle(method, path, body))
            if path == "/hil/start-loaded":
                self._validate_start_request(body.get("request_id"))
                return self._job("run", lambda: self._start_loaded_project(body), request_id=body["request_id"])
            if path == "/hil/start" and self._supports_projects() and self.device.get("backend") == "onnxruntime":
                return self._job("run", lambda: self._start_project(body), request_id=body.get("request_id"))
            if path == "/hil/start" and body.get("request_id") is not None:
                return self._job("run", lambda: self._start_session(body), request_id=body["request_id"])
        with self.lock:
            try:
                return self._handle(method, path, body)
            except (OSError, ValueError, EOFError) as exc:
                message = self._redact(str(exc) or type(exc).__name__)
                details = error_details(exc)
                self.error = None if details else message
                raise HilError(message, **details) from exc

    def _handle(self, method, path, body):
        if not isinstance(body, dict):
            raise HilError("HIL request body must be an object")
        if method == "GET" and path == "/hil/state":
            if self.sock and time.monotonic() - self.last_activity > 30:
                try:
                    self._rpc({"op": "hello"})
                except HilError:
                    pass
            return self.state()
        if method == "GET" and path == "/hil/projects":
            return self._projects()
        if method != "POST":
            raise HilError("Unknown HIL route")
        if path == "/hil/credentials":
            if set(body) != {"target", "expected_target", "password"}:
                raise HilError("Target, expected target settings and password are required")
            target = self._target(body.get("target"))
            if target.get("transport") != "ssh":
                raise HilError("Transient SSH credentials require an SSH target")
            if not isinstance(body.get("expected_target"), dict) or body["expected_target"] != target:
                raise HilError("Target settings changed. Retry the operation before entering its password.")
            credentials = self._validate_credentials({"password": body.get("password")})
            if not credentials["password"]:
                raise HilError("Enter the SSH password")
            credentials = {**self._credentials_for(target), **credentials}
            self._store_credentials(target, credentials)
            if (self.operation and self.operation.get("error_code") == "ssh_auth_required"
                    and self.operation.get("target_id") == target["id"]
                    and self.error == self.operation.get("error")):
                self.error = None
            return self.state()
        if path == "/hil/targets/remove":
            target_id = body.get("target")
            if target_id == self.selected_target and self.sock:
                raise HilError("Disconnect before removing this target")
            remove_target(target_id)
            self.targets = read_targets()
            self.credentials.pop(target_id, None)
            self.credential_targets.pop(target_id, None)
            self.device_info.pop(target_id, None)
            if self.selected_target == target_id:
                self.selected_target = None
            return self.state()
        if path in ("/hil/projects/unload", "/hil/projects/remove"):
            if path == "/hil/projects/remove" and "target" in body and (
                self.sock is None or body["target"] != self.selected_target
            ):
                raise HilError("Target changed or disconnected. "
                               "Refresh the target and confirm project deletion again.")
            if self.session:
                raise HilError("Stop the project first")
            if not self._supports_projects():
                raise HilError("Target does not support project management")
            result, _ = self._rpc({**body, "op": "projects.unload" if path.endswith("unload") else "projects.remove"})
            if path.endswith("unload"):
                self.project_generation += 1
            self._projects()
            return result
        if path == "/hil/targets":
            settings = dict(body)
            supplied = self._validate_credentials(settings.pop("credentials", {}))
            target = validate_target(settings)
            previous = next((item for item in self.targets if item["id"] == target["id"]), None)
            if self.sock and target["id"] == self.selected_target and target != previous:
                raise HilError("Disconnect before changing connected target settings")
            if supplied and target.get("transport") != "ssh":
                raise HilError("Transient SSH credentials require an SSH target")
            credentials = self._credentials_for(previous) if previous else {}
            if previous is None or self._credential_identity(previous) != self._credential_identity(target):
                credentials = {}
            if previous != target:
                self.device_info.pop(target["id"], None)
            if previous and previous.get("ssh", {}).get("host_key") != target.get("ssh", {}).get("host_key"):
                credentials.pop("host_key", None)
            credentials.update(supplied)
            target = save_target(target)
            if credentials:
                self._store_credentials(target, credentials)
            else:
                self.credentials.pop(target["id"], None)
                self.credential_targets.pop(target["id"], None)
            self.targets = read_targets()
            if not self.sock:
                self.selected_target = target["id"]
            self.error = None
            return self.state()
        if path == "/hil/connect":
            target_id = body.get("target", self.selected_target)
            target = next((t for t in self.targets if t["id"] == target_id), None)
            if target is None:
                raise HilError("Choose a configured target")
            self._drop()
            self.selected_target = target_id
            try:
                address = endpoint_address(target["endpoint"])
                if target.get("transport") == "ssh":
                    device = self._device(body)
                    try:
                        self.tunnel = device.open_tunnel()
                        address = self.tunnel.address
                    finally:
                        device.close()
                self.sock = socket.create_connection(address, timeout=3)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                result, _ = self._rpc({"op": "hello"})
                self.device = result["device"]
                if self.device.get("backend") != target["backend"]:
                    raise HilError(f"Target backend is {self.device.get('backend')}, expected {target['backend']}")
                self.error = None
                if self._supports_projects():
                    self._projects()
            except Exception:
                self._drop()
                raise
            return self.state()
        if path == "/hil/disconnect":
            if self.session:
                self._handle("POST", "/hil/stop", {"session": self.session})
            self._drop()
            self.device = None
            self.error = None
            return self.state()
        if path == "/hil/start":
            if self.session:
                raise HilError("Stop the current session before starting another")
            result, _ = self._rpc({"op": "start"})
            self.session = result["session"]
            self.sequence = -1
            self.models = {}
            self.error = None
            return {"session": self.session, "device": result["device"]}
        if path == "/hil/stop":
            if body.get("session") and body["session"] != self.session:
                raise HilError("Stale HIL session; current session was not stopped")
            if self.session:
                self._rpc({"op": "stop", "session": self.session})
                self.session = None
                self.models = {}
                self.sequence = -1
            return self.state()
        if path == "/hil/models":
            self._check_session(body)
            result, _ = self._rpc({**body, "op": "load"}, timeout=30)
            try:
                if result.get("sha256") != body.get("sha256"):
                    raise HilError("Target returned the wrong model identity")
                ports(result.get("inputs"))
                ports(result.get("outputs"))
            except HilError:
                self._drop()
                raise
            self.models[result["sha256"]] = result
            return {key: result[key] for key in ("sha256", "inputs", "outputs")}
        if path == "/hil/infer":
            self._check_session(body)
            seq = body.get("seq")
            if type(seq) is not int or seq < 0 or seq <= self.sequence:
                raise HilError("Inference sequence must increase within a session")
            model = self.models.get(body.get("model"))
            if not model:
                raise HilError("Model is not loaded in this session")
            tensors(body.get("inputs"), model["inputs"])
            self.sequence = seq
            result, rtt = self._rpc({**body, "op": "infer"})
            try:
                if result.get("session") != self.session or result.get("seq") != seq:
                    raise HilError("Target returned a stale inference response")
                outputs = result.get("outputs")
                if not isinstance(outputs, dict):
                    raise HilError("Target returned invalid outputs")
                tensors(outputs, model["outputs"])
            except HilError:
                self._drop()
                raise
            self.metrics = {"calls": result["calls"], "inference_ms": result["inference_ms"], "rtt_ms": rtt}
            return {"session": result["session"], "seq": seq, "outputs": outputs, **self.metrics}
        raise HilError("Unknown HIL route")

    def close(self):
        self.closed = True
        with self.lock:
            if self.sock and self.session:
                try:
                    self._rpc({"op": "stop", "session": self.session}, timeout=1)
                except HilError:
                    pass
            self._drop()
            self.credentials.clear()
            self.credential_targets.clear()
