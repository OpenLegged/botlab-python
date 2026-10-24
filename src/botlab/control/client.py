"""Standard-library HTTP transport and identity-preserving command submission."""

from __future__ import annotations

import json
import os
import ssl
import uuid
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from .discovery import descriptors, private_directory, project_key, public_descriptor, runtime_dir
from .discovery import endpoint as normalize_endpoint
from .protocol import (
    API_PREFIX,
    API_VERSION,
    COMMAND_STATUSES,
    PROTOCOL,
    ControlError,
    identifier,
    parse_json,
    validate_args,
    validate_capabilities,
)
from .receipts import Receipts


def agent_prompt(project: str) -> str:
    """Ready-to-paste instructions for the user's AI assistant.

    Assumes the environment is already prepared — botlab-python installed and
    the project middleware running — because this string only ships from a
    live, discovered middleware. The pairing token is deliberately absent:
    it is surfaced separately by `connection --show-token` and must never be
    embedded in prompts, command lines or logs.
    """
    return (
        "你通过 BotLab Control CLI 操控一个正在运行的浏览器仿真。环境已就绪：botlab-python 已安装，"
        "项目中间件正在运行，无需安装任何东西。\n"
        "\n"
        f"工作目录：{project}\n"
        "在其他目录调用时，为每条命令加 --project 指向该目录。\n"
        "\n"
        "先核实链路（应看到 connected 与 external_enabled 为 true、run_state 为 running、ready 为 true）：\n"
        "botlab control --project . state --json\n"
        "\n"
        "控制规则：\n"
        "1. 每个动作：读 state → 发命令 → wait --id 直到 terminal:true → 核对物理反馈，再发下一条。\n"
        "2. 工具输出解析失败不等于命令未执行：先查 state 的 active_command_id，确认未接受后才可重发。\n"
        "3. 参数按 capabilities 声明的范围发送；定时移动不是精确导航，实际效果以位置与姿态反馈为准。\n"
        "4. 网络超时只表示结果未知：先 wait 原请求，必要时对同一 request-id 显式 retry；新的运动决定用新 ID。\n"
        "5. 页面隐藏或仿真暂停时 ready=false，动作会被拒绝；恢复后从当前位置继续，旧动作不会自动重放。\n"
        "6. 项目若有 CONTROL_EXPERIENCE.md 或 AGENTS.md 等控制经验/说明文件，先读再动。\n"
        "\n"
        "现在：读取状态并汇报当前姿态，然后等待用户指令。"
    )


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(descriptor: dict, path: str, body: dict | None = None, timeout: float = 5) -> dict:
    origin = normalize_endpoint(descriptor["endpoint"])
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    headers = {"Authorization": f"Bearer {descriptor['token']}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(origin + API_PREFIX + path, data=data, headers=headers)
    handlers = [ProxyHandler({}), _NoRedirects()]
    cafile = os.environ.get("BOTLAB_CONTROL_CA_BUNDLE")
    if cafile:
        handlers.append(HTTPSHandler(context=ssl.create_default_context(cafile=cafile)))
    opener = build_opener(*handlers)
    status = 200
    try:
        try:
            response = opener.open(req, timeout=timeout)
        except HTTPError as error:
            if 300 <= error.code < 400:
                raise ControlError("redirect_refused", "Control API redirects are not permitted.") from None
            response = error
            status = error.code
        with response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ControlError("invalid_protocol", "Control response is too large.")
        payload = parse_json(raw.decode("utf-8"))
    except TimeoutError as error:
        raise ControlError("network_timeout", "Middleware did not respond; command outcome may be unknown.") from error
    except (URLError, OSError, HTTPException) as error:
        raise ControlError("connection_error", "Cannot reach the selected middleware; command outcome may be unknown.") from error
    except (UnicodeError, ValueError) as error:
        raise ControlError("invalid_protocol", "Middleware returned an invalid JSON response.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise ControlError("invalid_protocol", "Middleware response must contain boolean ok.")
    if payload["ok"] is False:
        detail = payload.get("error", {})
        if not isinstance(detail, dict):
            raise ControlError("invalid_protocol", "Malformed middleware error response.")
        raise ControlError(str(detail.get("code", "request_failed")), str(detail.get("message", "Request failed.")))
    if status >= 400:
        raise ControlError("invalid_protocol", "Middleware returned success with an error HTTP status.")
    return payload


class ControlClient:
    def __init__(self, project: str | Path = ".", *, endpoint: str | None = None,
                 instance: str | None = None, session: str | None = None):
        self.project = str(Path(project).resolve())
        self.project_key = project_key(project)
        self.endpoint = normalize_endpoint(endpoint) if endpoint else None
        self.instance = instance
        self.session = session
        self.receipts = Receipts(project)
        self._selected = None

    def _info(self, descriptor: dict) -> dict:
        info = request(descriptor, "/info")
        if info.get("api_version") != API_VERSION or info.get("protocol") != PROTOCOL:
            raise ControlError("unsupported_version", "Middleware does not implement Control API v1.")
        if descriptor.get("remote") is True:
            # Enrolled identity pin: compare against what /info reported at
            # enrollment, not against this checkout's local path.
            if (info.get("project_key") != descriptor.get("project_key")
                    or not isinstance(info.get("project"), str) or not info["project"]):
                raise ControlError("wrong_project", "Middleware identity no longer matches the enrolled descriptor.")
        elif info.get("project_key") != self.project_key or info.get("project") != self.project:
            raise ControlError("wrong_project", "Middleware belongs to a different project.")
        if info.get("instance_id") != descriptor["instance_id"]:
            raise ControlError("wrong_instance", "Middleware instance changed; the original command must not be replayed.")
        return info

    def instances(self) -> dict:
        items = []
        for descriptor in descriptors(self.project):
            item = public_descriptor(descriptor)
            try:
                item["info"] = self._info(descriptor)
                item["available"] = True
            except ControlError as error:
                item["available"] = False
                item["error"] = error.payload()["error"]
            items.append(item)
        return {"ok": True, "project_key": self.project_key, "instances": items}

    def enroll(self, target: str, token: str) -> dict:
        """Bind a remote HTTPS middleware to this project by pinning the
        identity /info reports into a private discovery descriptor."""
        origin = normalize_endpoint(target)
        if urlsplit(origin).scheme != "https":
            raise ControlError("invalid_endpoint", "Enrollment requires an HTTPS middleware origin.")
        if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
            raise ControlError("invalid_credentials", "A nonempty middleware token is required.")
        info = request({"endpoint": origin, "token": token}, "/info")
        if info.get("api_version") != API_VERSION or info.get("protocol") != PROTOCOL:
            raise ControlError("unsupported_version", "Middleware does not implement Control API v1.")
        identifier(info.get("instance_id"), "instance_id")
        pinned_key, pinned_project = info.get("project_key"), info.get("project")
        if not isinstance(pinned_key, str) or not pinned_key or not isinstance(pinned_project, str) or not pinned_project:
            raise ControlError("invalid_protocol", "Middleware did not report a project identity to pin.")
        descriptor = {"schema": 1, "api_version": API_VERSION, "protocol": PROTOCOL, "remote": True,
                      "endpoint": origin, "websocket": "wss" + origin[len("https"):] + "/runtime",
                      "project": pinned_project, "project_key": pinned_key,
                      "instance_id": info["instance_id"], "token": token}
        root, directory = runtime_dir(), runtime_dir() / self.project_key
        private_directory(root, create=True)
        private_directory(directory, create=True)
        for path in sorted(directory.glob("*.json")):
            try:
                existing = parse_json(path.read_text())
            except (OSError, ControlError):
                continue
            # Re-enrollment replaces this endpoint's previous instance pin.
            if isinstance(existing, dict) and existing.get("remote") is True and existing.get("endpoint") == origin:
                path.unlink()
        descriptor_path = directory / f"{descriptor['instance_id']}.json"
        try:
            fd = os.open(descriptor_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ControlError("instance_conflict",
                               "A different endpoint already enrolled this instance identity; "
                               "remove that descriptor before enrolling this endpoint.") from error
        with os.fdopen(fd, "w") as stream:
            json.dump(descriptor, stream)
        return {"ok": True, **public_descriptor(descriptor), "websocket": descriptor["websocket"],
                "pairing": "Remote middleware enrolled; identity pinned to this project.",
                # The AI drives the CLI from this checkout, not the remote host
                # whose path /info reported and the descriptor pinned.
                "agent_prompt": agent_prompt(self.project)}

    def select(self) -> dict:
        if self._selected is not None:
            return self._selected
        candidates = [item for item in descriptors(self.project)
                      if (self.endpoint is None or item["endpoint"] == self.endpoint)
                      and (self.instance is None or item["instance_id"] == self.instance)]
        live, errors = [], []
        for candidate in candidates:
            try:
                self._info(candidate)
                live.append(candidate)
            except ControlError as error:
                errors.append(error)
        if not live:
            if len(errors) == 1:
                raise errors[0]
            raise ControlError("no_service", "No matching live middleware was discovered for this project.")
        if len(live) != 1:
            raise ControlError("ambiguous_instance", "Multiple middleware instances are available; select --instance or --endpoint.",
                               instances=[public_descriptor(item) for item in live])
        self._selected = live[0]
        return self._selected

    def _read(self, path: str) -> dict:
        descriptor = self.select()
        payload = request(descriptor, path)
        if payload.get("instance_id") != descriptor["instance_id"]:
            raise ControlError("wrong_instance", "Middleware instance changed while reading state.")
        return payload

    def capabilities(self) -> dict:
        payload = self._read("/capabilities")
        validate_capabilities(payload)
        return payload

    def state(self) -> dict:
        payload = self._read("/state")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ControlError("invalid_protocol", "Middleware did not return state.")
        if self.session and state.get("session_id") != self.session:
            raise ControlError("wrong_session", "Selected session is not the current middleware session.")
        return payload

    def sessions(self) -> dict:
        return self._read("/sessions")

    def connection(self, show_token: bool = False) -> dict:
        descriptor = self.select()
        result = {"ok": True, **public_descriptor(descriptor), "websocket": descriptor.get("websocket"),
                  "pairing": "Open the project's runtime, pair with this middleware, then enable external control.",
                  # Self, not the descriptor: a remote descriptor pins the
                  # serving host's path, but the AI runs the CLI from here.
                  "agent_prompt": agent_prompt(self.project)}
        if show_token:
            result["token"] = descriptor["token"]
        return result

    def _query(self, descriptor: dict, command_id: str, timeout: float = 0) -> dict:
        identifier(command_id, "command_id")
        query = urlencode({"instance_id": descriptor["instance_id"], "timeout": timeout})
        payload = request(descriptor, f"/commands/{quote(command_id, safe='')}?{query}", timeout=timeout + 5)
        self._check_command(payload, descriptor, command_id)
        return payload

    @staticmethod
    def _check_command(payload: dict, descriptor: dict, command_id: str) -> None:
        command = payload.get("command")
        if not isinstance(command, dict):
            raise ControlError("invalid_protocol", "Middleware did not return a command.")
        if command.get("instance_id") != descriptor["instance_id"]:
            raise ControlError("wrong_instance", "Command response belongs to another middleware instance.")
        if command.get("command_id") != command_id:
            raise ControlError("wrong_command", "Response does not match the requested command ID.")
        if command.get("status") not in COMMAND_STATUSES:
            raise ControlError("invalid_protocol", "Middleware returned an unknown command status.")
        terminal = command["status"] in {"completed", "cancelled", "failed", "expired"}
        if command.get("terminal") is not terminal:
            raise ControlError("invalid_protocol", "Command terminal flag disagrees with its status.")

    def _original(self, receipt: dict) -> dict:
        envelope = receipt["envelope"]
        if ((self.instance and self.instance != envelope["instance_id"])
                or (self.endpoint and self.endpoint != receipt["endpoint"])
                or (self.session and self.session != envelope["session_id"])):
            raise ControlError("identity_conflict", "Explicit selection conflicts with the original request receipt.")
        candidates = [item for item in descriptors(self.project)
                      if item["instance_id"] == envelope["instance_id"] and item["endpoint"] == receipt["endpoint"]]
        if len(candidates) != 1:
            raise ControlError("original_instance_unavailable", "Original middleware instance is unavailable; reassess current state before issuing a new request.")
        self._info(candidates[0])
        return candidates[0]

    def wait(self, command_id: str, timeout: float = 1) -> dict:
        receipt = self.receipts.load(command_id)
        descriptor = self._original(receipt) if receipt else self.select()
        return self._query(descriptor, command_id, timeout)

    def retry(self, request_id: str, *, expected_intent: dict | None = None) -> dict:
        receipt = self.receipts.load(request_id)
        if receipt is None:
            raise ControlError("missing_receipt", "Original request receipt is missing; automatic rebinding is not permitted.", request_id=request_id)
        envelope = receipt["envelope"]
        try:
            if expected_intent is not None and receipt["intent"] != expected_intent:
                raise ControlError("request_conflict", "This request ID already belongs to a different intent.")
            descriptor = self._original(receipt)
            try:
                return self._query(descriptor, request_id)
            except ControlError as error:
                if error.code not in {"not_found", "unknown_command"}:
                    raise
            capabilities = request(descriptor, "/capabilities")
            validate_capabilities(capabilities)
            state_response = request(descriptor, "/state")
            state = state_response.get("state", {})
            if (capabilities.get("instance_id") != envelope["instance_id"]
                    or state_response.get("instance_id") != envelope["instance_id"]
                    or capabilities["capabilities_revision"] != envelope["capabilities_revision"]
                    or state.get("session_id") != envelope["session_id"]
                    or state.get("episode_id") != envelope["episode_id"]):
                raise ControlError("original_identity_changed", "Original session, episode or capabilities changed; the command was not resent.")
            if ("control_generation" in receipt
                    and state.get("control_generation") != receipt["control_generation"]):
                raise ControlError("state_changed", "Control generation changed since the original request; retry cannot resume cancelled control.")
            return self._send(descriptor, envelope)
        except ControlError as error:
            error.context.update(request_id=request_id, instance_id=envelope["instance_id"])
            raise

    def _send(self, descriptor: dict, envelope: dict) -> dict:
        try:
            payload = request(descriptor, "/commands", envelope)
            self._check_command(payload, descriptor, envelope["request_id"])
            command = payload["command"]
            if any(command.get(key) != envelope[key] for key in ("session_id", "episode_id")):
                raise ControlError("wrong_command", "Response does not match the submitted session/episode.")
            return payload
        except ControlError as error:
            error.context.update(request_id=envelope["request_id"], instance_id=envelope["instance_id"])
            raise

    def call(self, name: str, args: dict, *, request_id: str | None = None, alias: bool = False) -> dict:
        if not isinstance(args, dict):
            raise ControlError("invalid_args", "Capability arguments must be a JSON object.")
        validate_args(args, {"type": "object"})
        request_id = identifier(str(uuid.uuid4()) if request_id is None else request_id, "request_id")
        intent = {"name": name, "alias": alias, "args": args}
        if self.receipts.load(request_id) is not None:
            return self.retry(request_id, expected_intent=intent)
        descriptor = self.select()
        capabilities = self.capabilities()
        matches = [item for item in capabilities["capabilities"]
                   if (name in item.get("aliases", []) if alias else item["id"] == name)]
        if name == "control.stop" and not alias:
            capability = {"id": "control.stop", "input_schema": {"type": "object", "additionalProperties": False}}
        elif len(matches) != 1:
            raise ControlError("unsupported_capability", f"Middleware does not declare a unique capability for {name}.")
        else:
            capability = matches[0]
        validate_args(args, capability["input_schema"])
        state = self.state()["state"]
        envelope = {"request_id": request_id, "instance_id": descriptor["instance_id"],
                    "session_id": identifier(state.get("session_id"), "session_id"),
                    "episode_id": identifier(state.get("episode_id"), "episode_id"),
                    "capabilities_revision": capabilities["capabilities_revision"],
                    "capability": capability["id"], "args": args}
        receipt, created = self.receipts.save(
            envelope, descriptor["endpoint"], intent,
            control_generation=state.get("control_generation"),
        )
        if not created:
            return self.retry(request_id, expected_intent=intent)
        return self._send(descriptor, receipt["envelope"])
