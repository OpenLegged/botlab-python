"""Generic middleware tests: descriptor pairing, ported service semantics, HTTP face.

The task-table semantics (freshness, deduplication, leases, cancellation) are
ported from microduck's proven suite; here the adapter starts empty and a
descriptor uploaded over the paired connection compiles it.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from aiohttp import WSMsgType, WSServerHandshakeError
    from aiohttp.test_utils import TestClient, TestServer
except ImportError as error:  # pragma: no cover - without the serve extra
    raise unittest.SkipTest("aiohttp (botlab-python[serve]) is required for serve tests") from error

from botlab.control import serve_service
from botlab.control.descriptor import revision_of, validate_descriptor
from botlab.control.discovery import descriptors
from botlab.control.protocol import ControlError as ProtocolError
from botlab.control.serve_service import ControlError, GenericControlService, create_app, serve

TOKEN = "test-token-only"
ORIGIN = "http://127.0.0.1:8775"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Browser:
    """Records wire commands without pretending to execute a simulation."""

    def __init__(self):
        self.closed = False
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)

    async def close(self, **kwargs):
        self.closed = True


def descriptor():
    """The microduck control surface as a page would declare it."""
    return {
        "v": 1, "type": "descriptor",
        "profile": {"robot_id": "MicroDuck", "state_node_id": "state-machine",
                    "motion_parameters": {"vx": "forward_velocity", "yaw_rate": "yaw_rate"},
                    "event_parameter": "policy_command", "event_values": [1, 2, 3, 4, 5, 6]},
        "motion": {"vx": {"min": -0.2, "max": 0.25, "unit": "m/s"},
                   "yaw_rate": {"min": -1, "max": 1, "unit": "rad/s"},
                   "allowed_states": [0]},
        "actions": [
            {"name": "sit", "event_value": 1, "allowed_states": [0], "target_states": [2],
             "capability": "posture.sit", "idempotent": True},
            {"name": "stand_up", "event_value": 1, "allowed_states": [2], "target_states": [0],
             "capability": "posture.stand_up", "idempotent": True},
            {"name": "kick_left", "event_value": 2, "allowed_states": [0], "target_states": [0],
             "capability": "behavior.kick_left"},
        ],
        "states": {"0": "WALK", "2": "SIT"},
        "experience": "# MicroDuck control experience",
    }


def snapshot(**changes):
    return {
        "v": 1, "type": "state", "session_id": "browser-a",
        "episode_id": "episode-a", "control_generation": 2,
        "basis_id": "basis-a", "run_state": "running", "external_enabled": True, "step": 10,
        "sim_time_s": 0.2, "captured_mono_ms": 200, "state_age_ms": 0,
        "policy_state_code": 0, "policy_state_steps": 10,
        "position_world_m": [0, 0, 0.1],
        "orientation_wxyz": [1, 0, 0, 0],
        "effective_parameters": {"forward_velocity": 0, "yaw_rate": 0},
        **changes,
    }


def result(command_id="cmd-a", status="completed", **changes):
    return {"v": 1, "type": "result", "session_id": "browser-a",
            "episode_id": "episode-a", "command_id": command_id,
            "status": status, "step": 110, "sim_time_s": 2.2, **changes}


def move_args(**changes):
    return {"vx": 0.1, "vy": 0, "yaw_rate": 0, "duration_s": 2, **changes}


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = Clock()
        self.service = GenericControlService(TOKEN, {ORIGIN}, clock=self.clock)
        self.browser = Browser()
        self.service.socket = self.browser

    def envelope(self, capability, args, request_id="cmd-a", **changes):
        return {"request_id": request_id, "instance_id": self.service.instance_id,
                "session_id": "browser-a", "episode_id": "episode-a",
                "capabilities_revision": self.service.adapter.revision,
                "capability": capability, "args": args, **changes}

    async def pair(self):
        await self.service.accept_message(descriptor())

    async def submit(self, **changes):
        envelope = self.envelope("motion.twist", move_args())
        envelope.update(changes)
        return await self.service.submit_public(envelope)

    async def assert_rejected(self, code, coroutine):
        with self.assertRaises((ControlError, ProtocolError)) as caught:
            await coroutine
        self.assertEqual(caught.exception.code, code)

    def commands(self):
        return [message for message in self.browser.messages if message.get("type") == "command"]

    async def test_unpaired_service_refuses_project_semantics(self):
        with self.assertRaises(ControlError) as caught:
            self.service.capabilities()
        self.assertEqual(caught.exception.code, "no_project_paired")
        with self.assertRaises(ControlError) as caught:
            self.service.experience()
        self.assertEqual(caught.exception.code, "no_project_paired")
        envelope = {"request_id": "cmd-a", "instance_id": self.service.instance_id,
                    "session_id": "browser-a", "episode_id": "episode-a",
                    "capabilities_revision": "gen-x", "capability": "motion.twist",
                    "args": move_args()}
        await self.assert_rejected("no_project_paired", self.service.submit_public(envelope))
        info = self.service.info()
        self.assertEqual(info["extensions"], ["project-descriptor"])
        self.assertEqual(info["backend"], {"id": "browser", "kind": "simulation"})
        self.assertFalse(self.service.state()["ready"])

    async def test_state_frames_before_descriptor_are_rejected(self):
        with self.assertRaises(ProtocolError) as caught:
            await self.service.accept_message(snapshot())
        self.assertEqual(caught.exception.code, "invalid_message")
        await self.pair()
        await self.service.accept_message(snapshot())
        self.assertTrue(self.service.state()["ready"])

    async def test_descriptor_compiles_engine_and_acks(self):
        await self.pair()
        ack = self.browser.messages[0]
        self.assertEqual(ack["type"], "descriptor_ack")
        self.assertEqual(ack["capabilities_revision"],
                         revision_of(validate_descriptor(descriptor())))
        self.assertEqual(ack["profile"]["robot_id"], "MicroDuck")
        capabilities = self.service.capabilities()
        self.assertEqual({entry["id"] for entry in capabilities["capabilities"]},
                         {"motion.twist", "control.stop", "posture.sit",
                          "posture.stand_up", "behavior.kick_left"})
        self.assertEqual(self.service.experience()["experience"],
                         "# MicroDuck control experience")

    async def test_invalid_descriptor_rejected_without_pairing(self):
        broken = descriptor()
        broken["surprise"] = True
        with self.assertRaises(ProtocolError) as caught:
            await self.service.accept_message(broken)
        self.assertEqual(caught.exception.code, "invalid_descriptor")
        self.assertIsNone(self.service.adapter)

    async def test_move_wire_and_bounds(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        self.assertTrue(self.service.state()["ready"])
        for args in (move_args(vx=0.251), move_args(yaw_rate=-1.01),
                     move_args(duration_s=0.049), move_args(duration_s=10.001)):
            with self.subTest(args=args):
                await self.assert_rejected(
                    "out_of_range", self.service.submit_public(
                        self.envelope("motion.twist", args)))
        await self.assert_rejected("invalid_parameter", self.service.submit_public(
            self.envelope("motion.twist", move_args(vx=True))))
        await self.assert_rejected("invalid_parameter", self.service.submit_public(
            self.envelope("motion.twist", move_args(vx=float("nan")))))
        await self.assert_rejected("unsupported_capability", self.service.submit_public(
            self.envelope("nav.goto", {})))
        self.assertEqual(self.commands(), [])
        submitted = await self.submit()
        self.assertEqual(submitted["status"], "accepted")
        self.assertFalse(submitted["terminal"])
        wire = self.commands()[0]
        self.assertEqual(wire["values"], {"forward_velocity": 0.1, "yaw_rate": 0})
        self.assertEqual(wire["duration_ms"], 2000)
        self.assertEqual(wire["episode_id"], "episode-a")
        self.assertEqual(wire["allowed_states"], [0])

    async def test_envelope_guards(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        await self.assert_rejected("wrong_instance", self.submit(instance_id="other-instance"))
        await self.assert_rejected("capabilities_changed", self.submit(capabilities_revision="gen-stale"))
        await self.assert_rejected("wrong_session", self.submit(session_id="another-browser"))

    async def test_retry_deduplicates_and_conflicting_retry_is_rejected(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        original = await self.submit()
        retried = await self.submit()
        self.assertEqual(original["command_id"], retried["command_id"])
        self.assertEqual(len(self.commands()), 1)
        await self.assert_rejected("request_conflict", self.submit(args=move_args(vx=0.2)))
        await self.assert_rejected("busy", self.submit(request_id="cmd-b"))
        await self.service.accept_message(result())
        await self.service.disconnected()
        # The public envelope needs a live session, so a resubmission after
        # disconnect is refused; the terminal outcome stays reachable by ID.
        await self.assert_rejected("wrong_session", self.submit())
        self.assertEqual(self.service.api_job(self.service.jobs["cmd-a"])["status"], "completed")
        self.assertEqual(len(self.commands()), 1)

    async def test_stop_cancels_motion_but_waits_for_browser_commit(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        await self.submit()
        stop = await self.service.submit_public(
            self.envelope("control.stop", {}, request_id="stop-a"))
        self.assertEqual(self.service.jobs["cmd-a"]["reason"], "stop_requested")
        self.assertEqual(stop["status"], "accepted")
        self.assertEqual(self.commands()[-1]["kind"], "stop")
        await self.service.accept_message(result(command_id="stop-a"))
        self.assertEqual(self.service.jobs["stop-a"]["status"], "completed")

    async def test_episode_change_cancels_and_retires_results(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        await self.submit()
        await self.service.accept_message(result(episode_id="old-episode"))
        await self.service.accept_message(result(status="applied"))
        self.assertEqual(self.service.jobs["cmd-a"]["status"], "applied")
        await self.service.accept_message(snapshot(episode_id="episode-b", step=1))
        self.assertEqual(self.service.jobs["cmd-a"]["reason"], "episode_changed")
        await self.service.accept_message(result())
        self.assertEqual(self.service.jobs["cmd-a"]["status"], "cancelled")

    async def test_action_semantics_and_disallowed_states(self):
        await self.pair()
        await self.service.accept_message(snapshot(policy_state_code=2))
        sit = await self.service.submit_public(
            self.envelope("posture.sit", {}, request_id="sit-a"))
        self.assertEqual((sit["status"], sit["reason"]), ("completed", "already_satisfied"))
        self.assertEqual(self.commands(), [])
        # Refusals are checked while idle: a busy table answers busy first.
        await self.assert_rejected("invalid_state", self.service.submit_public(
            self.envelope("behavior.kick_left", {}, request_id="kick-a")))
        await self.assert_rejected("invalid_state", self.submit(request_id="move-b"))
        stand = await self.service.submit_public(
            self.envelope("posture.stand_up", {}, request_id="stand-a"))
        self.assertEqual(stand["status"], "accepted")
        wire = self.commands()[-1]
        self.assertEqual((wire["kind"], wire["event_value"],
                          wire["allowed_states"], wire["target_states"]),
                         ("event", 1, [2], [0]))

    async def test_monitor_renews_with_latest_basis_and_expires(self):
        await self.pair()
        await self.service.accept_message(snapshot())
        await self.submit()
        await self.service.accept_message(snapshot(step=11, basis_id="basis-b", control_generation=3))
        monitor = asyncio.create_task(self.service.monitor_jobs())
        try:
            await asyncio.sleep(0.12)
            renewals = [m for m in self.browser.messages if m["type"] == "renew"]
            self.assertEqual(len(renewals), 1)
            self.assertEqual(renewals[0]["basis_id"], "basis-b")
            self.assertEqual(renewals[0]["control_generation"], 3)
            self.clock.now += 1.01
            await asyncio.sleep(0.12)
            self.assertEqual(self.service.jobs["cmd-a"]["status"], "expired")
        finally:
            monitor.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await monitor


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = GenericControlService(TOKEN, {ORIGIN})
        self.client = TestClient(TestServer(create_app(self.service)))
        self.addAsyncCleanup(self.client.close)
        await self.client.start_server()
        self.headers = {"Authorization": "Bearer " + TOKEN}

    async def pair_over_ws(self):
        ws = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await ws.send_json({"v": 1, "type": "hello", "token": TOKEN})
        welcome = await ws.receive_json()
        assert welcome["type"] == "welcome"
        await ws.send_json(descriptor())
        ack = await ws.receive_json()
        assert ack["type"] == "descriptor_ack"
        await ws.send_json(snapshot())
        return ws, ack

    async def test_http_host_origin_token_and_preflight_boundary(self):
        cases = (
            ({}, 401, "unauthorized"),
            ({"Authorization": "Bearer wrong"}, 401, "unauthorized"),
            ({**self.headers, "Host": "malicious.example"}, 403, "forbidden_host"),
            ({**self.headers, "Origin": "https://malicious.example"}, 403, "forbidden_origin"),
        )
        for headers, status, code in cases:
            with self.subTest(headers=headers):
                response = await self.client.get("/control/v1/state", headers=headers)
                self.assertEqual(response.status, status)
                self.assertEqual((await response.json())["error"]["code"], code)
        preflight = await self.client.options("/control/v1/state",
                                              headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"})
        self.assertEqual(preflight.status, 204)
        self.assertEqual(preflight.headers["Access-Control-Allow-Origin"], ORIGIN)
        response = await self.client.get("/control/v1/state", headers=self.headers)
        self.assertEqual(response.status, 200)
        self.assertFalse((await response.json())["state"]["ready"])

    async def test_project_endpoints_refuse_before_pairing(self):
        for path in ("/control/v1/capabilities", "/control/v1/experience"):
            with self.subTest(path=path):
                response = await self.client.get(path, headers=self.headers)
                self.assertEqual(response.status, 409)
                self.assertEqual((await response.json())["error"]["code"], "no_project_paired")
        response = await self.client.post("/control/v1/commands", headers=self.headers,
                                          json={"request_id": "x", "instance_id": self.service.instance_id,
                                                "session_id": "s", "episode_id": "e",
                                                "capabilities_revision": "gen-x",
                                                "capability": "motion.twist", "args": {}})
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"]["code"], "no_project_paired")

    async def test_websocket_boundary_and_welcome_negotiation(self):
        for headers in ({}, {"Origin": "https://malicious.example"}):
            with self.assertRaises(WSServerHandshakeError) as caught:
                await self.client.ws_connect("/runtime", headers=headers)
            self.assertEqual(caught.exception.status, 403)
        bad = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await bad.send_json({"v": 1, "type": "hello", "token": "wrong"})
        self.assertEqual((await bad.receive()).type, WSMsgType.CLOSE)
        self.assertEqual(bad.close_code, 1008)
        owner = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await owner.send_json({"v": 1, "type": "hello", "token": TOKEN})
        welcome = await owner.receive_json()
        self.assertEqual(welcome["profile"], None)
        self.assertEqual(welcome["descriptor"], {"version": 1})
        second = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await second.send_json({"v": 1, "type": "hello", "token": TOKEN})
        self.assertEqual((await second.receive()).type, WSMsgType.CLOSE)
        await owner.close()
        await second.close()
        await bad.close()

    async def test_second_descriptor_on_a_connection_closes_it(self):
        ws = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await ws.send_json({"v": 1, "type": "hello", "token": TOKEN})
        await ws.receive_json()
        await ws.send_json(descriptor())
        self.assertEqual((await ws.receive_json())["type"], "descriptor_ack")
        await ws.send_json(descriptor())
        frame = await ws.receive()
        self.assertEqual(frame.type, WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, 1008)
        await ws.close()

    async def test_invalid_descriptor_closes_without_pairing(self):
        broken = descriptor()
        broken["surprise"] = True
        ws = await self.client.ws_connect("/runtime", headers={"Origin": ORIGIN})
        await ws.send_json({"v": 1, "type": "hello", "token": TOKEN})
        await ws.receive_json()
        await ws.send_json(broken)
        self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)
        self.assertEqual(ws.close_code, 1008)
        self.assertIsNone(self.service.adapter)
        await ws.close()

    async def test_full_pairing_flow_over_real_websocket(self):
        ws, ack = await self.pair_over_ws()
        self.addAsyncCleanup(ws.close)
        self.assertEqual(ack["profile"]["robot_id"], "MicroDuck")
        response = await self.client.get("/control/v1/capabilities", headers=self.headers)
        payload = await response.json()
        self.assertEqual(payload["capabilities_revision"], ack["capabilities_revision"])
        response = await self.client.get("/control/v1/experience", headers=self.headers)
        self.assertEqual((await response.json())["experience"], "# MicroDuck control experience")

        envelope = {"request_id": "cmd-a", "instance_id": self.service.instance_id,
                    "session_id": "browser-a", "episode_id": "episode-a",
                    "capabilities_revision": ack["capabilities_revision"],
                    "capability": "motion.twist", "args": move_args()}
        response = await self.client.post("/control/v1/commands", headers=self.headers, json=envelope)
        command = (await response.json())["command"]
        self.assertEqual(command["status"], "accepted")
        self.assertEqual(command["current_state"]["policy_state"], "WALK")

        wire = await ws.receive_json()
        self.assertEqual(wire["type"], "command")
        self.assertEqual(wire["values"], {"forward_velocity": 0.1, "yaw_rate": 0})
        pending = await self.client.get("/control/v1/commands/cmd-a?timeout=0.01", headers=self.headers)
        self.assertTrue((await pending.json())["pending"])

        await ws.send_json(snapshot(step=110))
        await ws.send_json(result())
        final = await self.client.get("/control/v1/commands/cmd-a?timeout=1", headers=self.headers)
        payload = await final.json()
        self.assertFalse(payload["pending"])
        self.assertEqual(payload["command"]["status"], "completed")
        self.assertEqual(payload["command"]["completion_observation"]["capture_stage"], "completed")

    async def test_stop_endpoint_requires_stop_capability(self):
        await self.pair_over_ws()
        envelope = {"request_id": "stop-a", "instance_id": self.service.instance_id,
                    "session_id": "browser-a", "episode_id": "episode-a",
                    "capabilities_revision": self.service.adapter.revision,
                    "capability": "motion.twist", "args": {}}
        response = await self.client.post("/control/v1/stop", headers=self.headers, json=envelope)
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "invalid_parameter")


class ServeTests(unittest.TestCase):
    def args(self, **changes):
        values = {"port": 8776, "host": "127.0.0.1", "origin": [ORIGIN],
                  "public_host": [], "public_endpoint": None, "print_token": False,
                  "project": Path(self.project)}
        values.update(changes)
        return SimpleNamespace(**values)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.project.mkdir()

    def test_argument_validation(self):
        for changes, code in (({"port": 80}, "invalid_port"),
                              ({"port": 70000}, "invalid_port"),
                              ({"origin": []}, "invalid_parameter"),
                              ({"public_endpoint": "http://127.0.0.1:9443"}, "invalid_parameter")):
            with self.subTest(changes=changes):
                with self.assertRaises(ControlError) as caught:
                    serve(self.args(**changes))
                self.assertEqual(caught.exception.code, code)

    def test_registers_private_discoverable_descriptor_and_cleans_up(self):
        runtime = Path(self.temporary.name) / "runtime"
        seen = {}

        def fake_run_app(app, **kwargs):
            found = descriptors(self.project)
            seen["found"] = found
            directory = runtime / found[0]["project_key"]
            seen["files"] = list(directory.glob("*.json"))
            seen["mode"] = seen["files"][0].stat().st_mode & 0o777
            seen["directory_mode"] = directory.stat().st_mode & 0o777
            seen["payload"] = json.loads(seen["files"][0].read_text())

        with patch.dict(os.environ, {"BOTLAB_CONTROL_RUNTIME_DIR": str(runtime)}), \
                patch.object(serve_service.web, "run_app", fake_run_app):
            summary = serve(self.args())
        item = seen["found"][0]
        self.assertEqual(summary["instance_id"], item["instance_id"])
        self.assertEqual(item["endpoint"], "http://127.0.0.1:8776")
        self.assertEqual(item["websocket"], "ws://127.0.0.1:8776/runtime")
        self.assertTrue(item["token"])
        self.assertEqual(seen["mode"], 0o600)
        self.assertEqual(seen["directory_mode"], 0o700)
        self.assertEqual(seen["payload"]["pid"], os.getpid())
        self.assertEqual(list((runtime / item["project_key"]).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
