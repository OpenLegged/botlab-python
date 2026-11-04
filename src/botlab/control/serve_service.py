"""Generic browser-pairing middleware: project semantics arrive at pairing time.

Adapted from microduck's proven service semantics (task table, fingerprint
deduplication with tombstones, lease monitoring, completion observations,
boundary middleware, long-poll waits). The differences are deliberate:

- the adapter starts empty; a validated project descriptor uploaded over the
  paired /runtime connection compiles into a DescriptorEngine;
- only the public /control/v1 routes exist (no project legacy routes);
- the browser is the only execution backend, so pairing is always open.

This module imports aiohttp at the top level and therefore must never enter
the client CLI import chain: ``botlab.control.cli`` imports it lazily in the
serve branch only (mirrors botlab.hil.gateway).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import sys
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from aiohttp import WSMsgType, web

from .descriptor import DESCRIPTOR_VERSION, DescriptorEngine, validate_descriptor
from .discovery import private_directory
from .discovery import runtime_dir as discovery_runtime_dir
from .protocol import (
    API_VERSION,
    COMMAND_STATUSES,
    PROTOCOL,
    TERMINAL,
    project_key,
    validate_envelope,
)
from .protocol import ControlError as ProtocolError

STATUSES = COMMAND_STATUSES
WIRED_EXTENSIONS = ("project-descriptor",)
# The engine raises protocol.ControlError (stdlib package, no HTTP layer);
# these codes mirror microduck's explicit 400s, everything else is a task
# refusal (409).
BAD_REQUEST_CODES = frozenset({"invalid_parameter", "invalid_request",
                                "invalid_action", "invalid_args", "unsupported_capability"})


class ControlError(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.status = code, status


def finite(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError("invalid_parameter", f"{name} must be a number", 400)
    if not math.isfinite(value) or not low <= value <= high:
        raise ControlError("out_of_range", f"{name} must be in [{low}, {high}]", 400)
    return float(value)


class GenericControlService:
    def __init__(self, token: str, origins: set[str], *, clock=time.monotonic,
                 project: str | Path | None = None, max_results: int = 256,
                 max_requests: int = 10000, public_hosts: set[str] | None = None):
        self.token, self.origins, self.clock = token, origins, clock
        self.public_hosts = frozenset(public_hosts or ())
        if max_results < 1 or max_requests < 1:
            raise ValueError("Task retention limits must be positive")
        self.project = Path(project or Path.cwd()).resolve()
        self.project_key = project_key(self.project)
        self.instance_id = str(uuid.uuid4())
        # The engine is compiled from the descriptor the page uploads at pairing;
        # until then the service has no capabilities and refuses tasks.
        self.adapter: DescriptorEngine | None = None
        self.max_results, self.max_requests = max_results, max_requests
        self.submit_lock = asyncio.Lock()
        self.socket = None
        self.snapshot: dict | None = None
        self.received_at = 0.0
        self.progress_at = 0.0
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.fingerprints: dict[str, str] = {}
        self.bindings: dict[str, tuple[str, str]] = {}
        self.observations: OrderedDict[tuple, dict] = OrderedDict()
        self.active_id: str | None = None
        self.revision = asyncio.Condition()
        self.monitor: asyncio.Task | None = None
        self.retired_episodes: set[str] = set()

    @property
    def connected(self) -> bool:
        return self.socket is not None and not self.socket.closed

    def info(self) -> dict:
        return {"ok": True, "api_version": API_VERSION, "protocol": PROTOCOL,
                "project": str(self.project), "project_key": self.project_key,
                "instance_id": self.instance_id,
                "backend": {"id": "browser", "kind": "simulation"},
                "extensions": list(WIRED_EXTENSIONS),
                "retention": {"results": self.max_results, "motion_requests": self.max_requests,
                              "deduplication": "instance_lifetime; stop remains available at motion capacity"}}

    def require_engine(self) -> DescriptorEngine:
        if self.adapter is None:
            raise ControlError("no_project_paired",
                               "Pair a project page over /runtime first; it uploads the control descriptor")
        return self.adapter

    def capabilities(self) -> dict:
        engine = self.require_engine()
        return {"ok": True, "api_version": API_VERSION, "instance_id": self.instance_id,
                "capabilities_revision": engine.revision, "capabilities": engine.capabilities()}

    def experience(self) -> dict:
        engine = self.require_engine()
        return {"ok": True, "instance_id": self.instance_id,
                "capabilities_revision": engine.revision, "experience": engine.experience}

    async def changed(self):
        async with self.revision:
            self.revision.notify_all()

    def state(self) -> dict:
        if self.snapshot is None:
            return {"connected": self.connected, "external_enabled": False,
                    "run_state": "idle", "ready": False, "active_command_id": None}
        result = self.adapter.decode_state(self.snapshot)
        result.pop("type", None)
        elapsed_ms = max(0, (self.clock() - self.received_at) * 1000)
        result["state_age_ms"] = round(result.get("state_age_ms", 0) + elapsed_ms)
        result["progress_age_ms"] = round(max(0, (self.clock() - self.progress_at) * 1000))
        result["connected"] = self.connected
        result["active_command_id"] = self.active_id
        result["ready"] = (result["connected"] and result.get("run_state") == "running"
                           and result.get("external_enabled") is True
                           and self.adapter.state_available(result)
                           and result["state_age_ms"] < 1000 and result["progress_age_ms"] < 1000)
        return result

    def api_state(self) -> dict:
        state = self.state()
        state["observations"] = self.adapter.observations(state) if self.adapter else {}
        return {"ok": True, "instance_id": self.instance_id, "project_key": self.project_key, "state": state}

    def require_state(self, requested_session: str | None = None) -> dict:
        state = self.state()
        if requested_session and requested_session != state.get("session_id"):
            raise ControlError("wrong_session", "The requested browser session is not connected")
        if not state["ready"]:
            raise ControlError("not_ready", "Connect external control and run the simulation; fresh physics state is required")
        return state

    async def send(self, message: dict):
        if self.socket is None or self.socket.closed:
            raise ControlError("disconnected", "Browser control connection is closed")
        await asyncio.wait_for(self.socket.send_json({"v": 1, **message}), 1.0)

    @staticmethod
    def envelope(state: dict) -> dict:
        return {name: state[name] for name in ("session_id", "episode_id", "control_generation", "basis_id")}

    async def finish(self, job: dict, status: str, reason: str):
        if job["status"] in TERMINAL:
            return
        job.update(status=status, reason=reason, finished_at=self.clock())
        self.capture_completion(job)
        if self.active_id == job["command_id"]:
            self.active_id = None
        await self.changed()

    async def cancel_active(self, reason: str):
        if self.active_id is not None:
            await self.finish(self.jobs[self.active_id], "cancelled", reason)

    def check_instance(self, instance_id: str | None):
        if instance_id is not None and instance_id != self.instance_id:
            raise ControlError("wrong_instance", "Middleware instance changed; do not replay an old request")

    async def submit_public(self, body: dict) -> dict:
        validate_envelope(body, error=lambda code, message: ControlError(code, message, 400))
        self.check_instance(body["instance_id"])
        engine = self.require_engine()
        if body["capabilities_revision"] != engine.revision:
            raise ControlError("capabilities_changed", "Read the current capabilities before a new task")
        current = self.state()
        if body["session_id"] != current.get("session_id"):
            raise ControlError("wrong_session", "Read the current session before submitting a command")
        if body["episode_id"] != current.get("episode_id"):
            raise ControlError("wrong_episode", "Simulation episode changed; do not replay an old request")
        command = engine.from_capability(body["capability"], body["args"])
        async with self.submit_lock:
            await self._submit(body, command, engine,
                               binding=(body["session_id"], body["episode_id"]))
        return self.api_job(self.jobs[body["request_id"]])

    async def _submit(self, body: dict, command: dict, engine: DescriptorEngine,
                      binding: tuple[str, str]) -> dict:
        request_id = body["request_id"]
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ControlError("invalid_request", "request_id must be a nonempty string (max 128)", 400)
        try:
            json.dumps(body, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ControlError("invalid_request", "Request must contain finite JSON values", 400) from error
        normalized = engine.normalize(command)
        fingerprint = json.dumps(normalized, sort_keys=True, allow_nan=False)
        if request_id in self.fingerprints:
            known = self.bindings[request_id]
            if (self.fingerprints[request_id] != fingerprint
                    or (body.get("session_id") is not None and body["session_id"] != known[0])
                    or binding != known):
                raise ControlError("request_conflict", "request_id already belongs to a different command or execution identity")
            if request_id not in self.jobs:
                raise ControlError("result_evicted", "Result was retired; request ID remains reserved and will not execute again")
            return self.public_job(self.jobs[request_id])
        state = self.require_state(body.get("session_id"))
        if binding[1] != state.get("episode_id"):
            raise ControlError("wrong_episode", "Simulation episode changed; read state before a new task")
        kind = normalized["kind"]
        if kind != "stop" and len(self.fingerprints) >= self.max_requests:
            raise ControlError("request_capacity", "Motion request capacity reached; stop remains available")
        if self.active_id and kind != "stop":
            raise ControlError("busy", f"Command {self.active_id} is still executing")
        payload, timeout, already_satisfied = engine.prepare(normalized, state)
        wire = {"type": "command", "command_id": request_id, **self.envelope(state), **payload}
        if kind == "stop":
            await self.cancel_active("stop_requested")
        capability, args = engine.public_command(normalized)
        job = {"command_id": request_id, "kind": kind, "request": deepcopy(body),
               "instance_id": self.instance_id, "capability": capability, "args": args,
               "capabilities_revision": engine.revision,
               "session_id": state["session_id"], "episode_id": state["episode_id"],
               "status": "accepted", "created_at": self.clock(), "deadline": self.clock() + timeout,
               "initial_state": deepcopy(state),
               "initial_observation": {**deepcopy(state), "capture_stage": "accepted"},
               "completion_observation": None, "completion_observation_reason": "not_terminal"}
        self.jobs[request_id], self.fingerprints[request_id] = job, fingerprint
        self.bindings[request_id] = (state["session_id"], state["episode_id"])
        if already_satisfied:
            await self.finish(job, "completed", "already_satisfied")
        else:
            self.active_id = request_id
            try:
                await self.send(wire)
            except (TimeoutError, ControlError, ConnectionError) as error:
                await self.finish(job, "failed", str(error))
        while len(self.jobs) > self.max_results:
            oldest = next((key for key, value in self.jobs.items()
                           if key != request_id and value["status"] in TERMINAL), None)
            if oldest is None:
                break
            self.jobs.pop(oldest)
            # Keep fingerprint and identity tombstones for this instance's lifetime.
        await self.changed()
        return self.public_job(job)

    def capture_completion(self, job: dict):
        if job["status"] not in TERMINAL or job["completion_observation"] is not None:
            return
        if job.get("reason") == "already_satisfied":
            job["completion_observation"] = {**deepcopy(job["initial_state"]),
                                              "capture_stage": "already_satisfied"}
            job["completion_observation_reason"] = None
            return
        result = job.get("browser_result", {})
        key = (job["session_id"], job["episode_id"], result.get("step"))
        observed = self.observations.get(key) if result.get("status") in TERMINAL else None
        if observed is not None:
            job["completion_observation"] = {**deepcopy(observed), "capture_stage": "completed"}
            job["completion_observation_reason"] = None
        else:
            job["completion_observation_reason"] = "no_matching_completion_snapshot"

    def public_job(self, job: dict) -> dict:
        result = deepcopy({key: value for key, value in job.items() if key not in {"wire", "deadline"}})
        result["terminal"] = job["status"] in TERMINAL
        result["elapsed_s"] = round((job.get("finished_at", self.clock()) - job["created_at"]), 3)
        result["state"] = self.state()
        return result

    def api_job(self, job: dict) -> dict:
        result = self.public_job(job)
        result.pop("state", None)
        result["current_state"] = self.api_state()["state"]
        return result

    def find_job(self, command_id: str) -> dict:
        if command_id in self.jobs:
            return self.jobs[command_id]
        if command_id in self.fingerprints:
            raise ControlError("result_evicted", "Result was retired; the request remains reserved")
        raise ControlError("not_found", "Unknown command ID", 404)

    async def accept_message(self, message: dict):
        if message.get("v") != 1:
            raise ProtocolError("protocol_version", "Expected control protocol v1")
        kind = message.get("type")
        if kind == "descriptor":
            descriptor = validate_descriptor(message)
            self.adapter = DescriptorEngine(descriptor)
            await self.send({"type": "descriptor_ack", "capabilities_revision": self.adapter.revision,
                             "profile": self.adapter.profile})
            await self.changed()
        elif kind == "state":
            if self.adapter is None:
                raise ProtocolError("invalid_message", "Upload the project descriptor before state frames")
            for name in ("session_id", "episode_id", "basis_id"):
                if not isinstance(message.get(name), str) or not message[name]:
                    raise ProtocolError("bad_state", f"Missing state {name}")
            if message.get("run_state") not in {"idle", "running", "paused", "error"}:
                raise ProtocolError("bad_state", "Invalid run_state")
            for name in ("step", "control_generation", "sim_time_s", "captured_mono_ms", "state_age_ms"):
                finite(message.get(name), name, 0, 1e16)
            if message["episode_id"] in self.retired_episodes:
                return
            previous = self.snapshot
            if previous and message["session_id"] != previous["session_id"]:
                raise ProtocolError("wrong_session", "Browser session changed within a connection")
            new_episode = previous is None or previous["episode_id"] != message["episode_id"]
            if previous and new_episode:
                self.retired_episodes.add(previous["episode_id"])
                await self.cancel_active("episode_changed")
            if not new_episode and message["step"] < previous["step"]:
                return
            now = self.clock()
            if new_episode or message["step"] > previous["step"]:
                self.progress_at = now
            self.snapshot, self.received_at = deepcopy(message), now
            key = (message["session_id"], message["episode_id"], message["step"])
            self.observations[key] = self.state()
            while len(self.observations) > 64:
                self.observations.popitem(last=False)
            for job in self.jobs.values():
                self.capture_completion(job)
            if message["run_state"] != "running":
                await self.cancel_active(message["run_state"])
            elif not message.get("external_enabled", False):
                await self.cancel_active("external_control_disabled")
            await self.changed()
        elif kind == "result":
            job = self.jobs.get(message.get("command_id"))
            if job is None or job["status"] in TERMINAL:
                return
            if (message.get("session_id") != job["session_id"]
                    or message.get("episode_id") != job["episode_id"]):
                return
            status = message.get("status")
            if status not in STATUSES:
                raise ProtocolError("bad_result", "Unknown command result")
            if job["status"] == "applied" and status == "accepted":
                return
            job["browser_result"] = dict(message)
            if status in TERMINAL:
                await self.finish(job, status, str(message.get("reason", status)))
            else:
                job["status"] = status
            await self.changed()
        else:
            raise ProtocolError("invalid_message", "Expected descriptor, state or result")

    async def monitor_jobs(self):
        while True:
            await asyncio.sleep(0.1)
            if self.active_id is None:
                continue
            job = self.jobs[self.active_id]
            if self.clock() > job["deadline"]:
                await self.finish(job, "expired", "No completion before command deadline")
                # Stop renewing: browser's independent lease will expire.
                continue
            if job["kind"] != "move":
                continue
            state = self.state()
            if (not state["ready"] or state.get("session_id") != job["session_id"]
                    or state.get("episode_id") != job["episode_id"]):
                await self.finish(job, "expired", "Physics feedback became stale")
                continue
            try:
                await self.send({"type": "renew", "command_id": job["command_id"],
                                 **self.envelope(state), "ttl_ms": 500})
            except (TimeoutError, ControlError, ConnectionError) as error:
                await self.finish(job, "failed", str(error))

    async def disconnected(self):
        self.socket = None
        await self.cancel_active("browser_disconnected")
        self.snapshot = None
        self.retired_episodes.clear()
        self.observations.clear()
        await self.changed()


def create_app(service: GenericControlService) -> web.Application:
    @web.middleware
    async def boundary(request, handler):
        origin = request.headers.get("Origin")
        cors = request.path.startswith("/control/") and origin in service.origins
        try:
            host = request.host.split(":")[0]
            if host not in {"127.0.0.1", "localhost"} | service.public_hosts:
                raise ControlError("forbidden_host", "Loopback Host required", 403)
            if origin is not None and origin not in service.origins:
                raise ControlError("forbidden_origin", "Browser Origin is not allowed", 403)
            # CORS preflight: a page on an allowlisted origin fetches /control/v1
            # cross-port with Authorization + JSON Content-Type (both non-simple
            # headers), so the browser sends OPTIONS first. The spec forbids
            # credentials on preflight, which must be answered before the token
            # comparison below.
            if request.method == "OPTIONS" and request.path.startswith("/control/"):
                if origin not in service.origins:
                    raise ControlError("forbidden_origin", "Browser Origin is not allowed", 403)
                return web.Response(status=204, headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept",
                    "Access-Control-Max-Age": "600",
                })
            credential = request.headers.get("Authorization", "")
            expected = "Bearer " + service.token
            if request.path != "/runtime" and not secrets.compare_digest(credential, expected):
                raise ControlError("unauthorized", "Valid local control token required", 401)
            response = await handler(request)
        except ControlError as error:
            response = web.json_response({"ok": False, "error": {"code": error.code, "message": str(error)}}, status=error.status)
        except ProtocolError as error:
            response = web.json_response({"ok": False, "error": {"code": error.code, "message": str(error)}},
                                         status=400 if error.code in BAD_REQUEST_CODES else 409)
        except (ValueError, TypeError, KeyError) as error:
            response = web.json_response({"ok": False, "error": {"code": "invalid_request", "message": str(error)}}, status=400)
        if cors:
            # Echo the allowlisted Origin on every /control answer (errors
            # included) so the page can read the body; originless CLI traffic
            # gets no header.
            response.headers["Access-Control-Allow-Origin"] = origin
        return response

    app = web.Application(middlewares=[boundary], client_max_size=65536)

    async def api_info(request):
        service.check_instance(request.query.get("instance_id"))
        return web.json_response(service.info())

    async def api_capabilities(request):
        service.check_instance(request.query.get("instance_id"))
        return web.json_response(service.capabilities())

    async def api_experience(request):
        service.check_instance(request.query.get("instance_id"))
        return web.json_response(service.experience())

    async def api_state(request):
        service.check_instance(request.query.get("instance_id"))
        return web.json_response(service.api_state())

    async def api_sessions(request):
        service.check_instance(request.query.get("instance_id"))
        return web.json_response({"ok": True, "instance_id": service.instance_id,
                                  "sessions": [service.api_state()["state"]] if service.snapshot else []})

    async def api_submit(request):
        body = await request.json()
        if not isinstance(body, dict):
            raise ControlError("invalid_request", "JSON object required", 400)
        if request.path.endswith("/stop") and body.get("capability") != "control.stop":
            raise ControlError("invalid_parameter", "Stop endpoint requires control.stop", 400)
        return web.json_response({"ok": True, "command": await service.submit_public(body)})

    async def command(request):
        service.check_instance(request.query.get("instance_id"))
        job = service.find_job(request.match_info["id"])
        timeout = finite(float(request.query.get("timeout", "0")), "timeout", 0, 30)
        if timeout and job["status"] not in TERMINAL:
            async with service.revision:
                try:
                    await asyncio.wait_for(service.revision.wait_for(lambda: job["status"] in TERMINAL), timeout)
                except TimeoutError:
                    pass
        return web.json_response({"ok": True, "pending": job["status"] not in TERMINAL,
                                  "command": service.api_job(job)})

    async def runtime(request):
        if request.headers.get("Origin") not in service.origins:
            raise ControlError("forbidden_origin", "Explicit browser Origin required", 403)
        ws = web.WebSocketResponse(max_msg_size=65536, heartbeat=10)
        await ws.prepare(request)
        owns_connection = False
        descriptor_seen = False
        try:
            hello = await asyncio.wait_for(ws.receive_json(), 5)
            if (not isinstance(hello, dict) or hello.get("v") != 1 or hello.get("type") != "hello"
                    or not isinstance(hello.get("token"), str)
                    or not secrets.compare_digest(hello["token"], service.token)):
                await ws.close(code=1008, message=b"Authentication failed")
                return ws
            if service.socket is not None:
                await ws.close(code=1008, message=b"A browser already owns this instance")
                return ws
            service.socket, owns_connection = ws, True
            # The negotiation bit asks the page for its project descriptor; the
            # profile round-trips back in descriptor_ack (or immediately here on
            # reconnect, when a descriptor from an earlier pairing is retained).
            await ws.send_json({"v": 1, "type": "welcome",
                                "profile": service.adapter.profile if service.adapter else None,
                                "descriptor": {"version": DESCRIPTOR_VERSION}})
            async for frame in ws:
                if frame.type == WSMsgType.TEXT:
                    message = json.loads(frame.data)
                    if not isinstance(message, dict):
                        raise ProtocolError("invalid_message", "JSON object required")
                    if message.get("type") == "descriptor":
                        if descriptor_seen:
                            raise ProtocolError("invalid_message",
                                                "A descriptor was already uploaded on this connection")
                        descriptor_seen = True
                    await service.accept_message(message)
                elif frame.type in {WSMsgType.ERROR, WSMsgType.BINARY}:
                    break
        except (TimeoutError, ProtocolError, ControlError, ValueError, TypeError) as error:
            await ws.close(code=1008, message=str(error).encode()[:120])
        finally:
            if owns_connection:
                await service.disconnected()
        return ws

    async def lifecycle(app):
        service.monitor = asyncio.create_task(service.monitor_jobs())
        yield
        service.monitor.cancel()
        try:
            await service.monitor
        except asyncio.CancelledError:
            pass
        if service.socket is not None:
            await service.socket.close(code=1001, message=b"Server stopping")

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/runtime", runtime)
    app.router.add_get("/control/v1/info", api_info)
    app.router.add_get("/control/v1/capabilities", api_capabilities)
    app.router.add_get("/control/v1/experience", api_experience)
    app.router.add_get("/control/v1/state", api_state)
    app.router.add_get("/control/v1/sessions", api_sessions)
    app.router.add_post("/control/v1/commands", api_submit)
    app.router.add_post("/control/v1/stop", api_submit)
    app.router.add_get("/control/v1/commands/{id}", command)
    return app


def serve(args) -> dict:
    """Run the generic middleware; blocks until shutdown. Returns a summary."""
    if not 1024 <= args.port <= 65535:
        raise ControlError("invalid_port", "Port must be 1024..65535")
    if not args.origin:
        raise ControlError("invalid_parameter",
                           "--origin is required: allowlist the BotLab page origin for browser pairing")
    if args.public_endpoint and not args.public_endpoint.startswith("https://"):
        raise ControlError("invalid_parameter", "--public-endpoint must be an HTTPS origin")
    service = GenericControlService(secrets.token_urlsafe(32), set(args.origin),
                                    project=args.project,
                                    public_hosts=set(args.public_host or []))
    app = create_app(service)
    directory = discovery_runtime_dir() / service.project_key
    private_directory(discovery_runtime_dir(), create=True)
    private_directory(directory, create=True)
    endpoint = args.public_endpoint or f"http://127.0.0.1:{args.port}"
    info = {"schema": 1, "api_version": API_VERSION, "protocol": PROTOCOL,
            "project": str(service.project), "project_key": service.project_key,
            "instance_id": service.instance_id,
            "endpoint": endpoint,
            "websocket": "ws" + endpoint[4:] + "/runtime",
            "token": service.token, "pid": os.getpid()}
    descriptor_path = directory / f"{service.instance_id}.json"
    created = False
    try:
        fd = os.open(descriptor_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "w") as stream:
            json.dump(info, stream)
        print(f"BotLab control: {info['endpoint']}\nInstance: {service.instance_id}",
              file=sys.stderr, flush=True)
        if args.public_endpoint:
            # Operator channel: the remote user enrolls with this origin plus
            # the token, shared out of band (revealed only with --print-token).
            print(f"Remote enrollment: botlab control enroll --endpoint {info['endpoint']}",
                  file=sys.stderr, flush=True)
        if args.print_token:
            print(f"Token: {service.token}", file=sys.stderr, flush=True)
        web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)
    finally:
        if created:
            descriptor_path.unlink(missing_ok=True)
    return {"ok": True, "service": "stopped", "instance_id": service.instance_id}
