"""Local display server tests (tier B: needs websockets)."""

import json
import threading
import unittest
from types import SimpleNamespace

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from botlab._runtime import _display  # noqa: E402
from botlab._runtime.mujoco_scene import MujocoSceneProvider  # noqa: E402


class DisplayServerTest(unittest.TestCase):
    def start_server(self, **kwargs):
        server = _display.DisplayServer(
            metadata={"controlHz": 50, "qposDim": 3, "qvelDim": 2}, **kwargs
        )
        server.start()
        self.addCleanup(server.stop)
        return server

    def test_meta_then_state_frames_arrive_in_order(self):
        server = self.start_server()
        with connect(server.url, origin="http://127.0.0.1:5173") as client:
            meta = client.recv(timeout=5)
            self.assertIsInstance(meta, str)
            parsed = json.loads(meta)
            self.assertEqual(parsed["controlHz"], 50)
            self.assertEqual(parsed["qposDim"], 3)

            server.broadcast(
                time_seconds=1.25, qpos=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                qvel=np.array([0.5, -0.5], dtype=np.float32),
            )
            frame = client.recv(timeout=5)
            values = np.frombuffer(frame, dtype="<f4")
            np.testing.assert_allclose(
                values, [1.25, 1.0, 2.0, 3.0, 0.5, -0.5], rtol=0, atol=0
            )

    def test_metrics_and_diagnostics_are_typed_json_messages(self):
        server = self.start_server()
        with connect(server.url) as client:
            client.recv(timeout=5)
            server.update_metrics({"speed": 1.5})
            self.assertEqual(
                json.loads(client.recv(timeout=5)),
                {"type": "metrics", "metrics": {"speed": 1.5}},
            )
            server.update_diagnostics([{"level": "warn", "message": "hot"}])
            self.assertEqual(
                json.loads(client.recv(timeout=5)),
                {
                    "type": "diagnostics",
                    "diagnostics": [{"level": "warn", "message": "hot"}],
                },
            )

    def test_invalid_output_payloads_fail_before_broadcast(self):
        server = self.start_server()
        with self.assertRaises(ValueError):
            server.update_metrics({"bad": float("nan")})
        with self.assertRaises(ValueError):
            server.update_diagnostics({"not": "a list"})
        with self.assertRaisesRegex(ValueError, "at most 200"):
            server.update_diagnostics([{} for _ in range(201)])
        with self.assertRaisesRegex(ValueError, "text frame exceeds"):
            server.update_diagnostics([{"message": "x" * 65536}])

    def test_port_defaults_to_ephemeral_and_is_exposed(self):
        server = self.start_server()
        self.assertIsInstance(server.port, int)
        self.assertGreater(server.port, 0)

    def test_non_loopback_host_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _display.DisplayServer(metadata={}, host="0.0.0.0")
        self.assertIn("127.0.0.1", str(ctx.exception))

    def test_token_path_and_browser_origin_are_required(self):
        server = self.start_server()
        with connect(
            f"ws://127.0.0.1:{server.port}/",
            origin="http://127.0.0.1:5173",
        ) as client:
            with self.assertRaises(Exception):
                client.recv(timeout=1)
        with connect(server.url, origin="https://example.com") as client:
            with self.assertRaises(Exception):
                client.recv(timeout=1)

    def test_client_messages_close_the_read_only_stream(self):
        server = self.start_server()
        with connect(server.url) as client:
            client.recv(timeout=5)
            client.send("not allowed")
            with self.assertRaises(Exception):
                client.recv(timeout=1)

    def test_stop_disconnects_clients_and_is_idempotent(self):
        server = self.start_server()
        client = connect(server.url)
        try:
            client.recv(timeout=5)
            server.stop()
            with self.assertRaises(Exception):
                client.recv(timeout=1)
        finally:
            client.close()
        server.stop()
        with self.assertRaises(OSError):
            connect(server.url, close_timeout=1)


class SceneDisplayTest(unittest.TestCase):
    @staticmethod
    def model_and_data():
        obj = b"""v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""
        model = mujoco.MjModel.from_xml_string(
            """<mujoco><asset><mesh name="tetra" file="tetra.obj"/></asset>
            <worldbody><body name="robot" pos="1 2 3"><freejoint/>
            <geom type="box" size=".1 .2 .3"/>
            <geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>""",
            assets={"tetra.obj": obj},
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_first_frame_is_not_lost_during_initial_message_send(self):
        for scene in (False, True):
            with self.subTest(scene=scene):
                self.assert_initial_message_keeps_first_frame(scene)

    def assert_initial_message_keeps_first_frame(self, scene):
        server = _display.DisplayServer(
            {}, scene_provider=SimpleNamespace(
                payload='{"type":"scene"}', frame_size=11, frame=lambda _: b"scene-frame",
            ),
        )
        messages = []
        errors = []
        finished = threading.Event()

        def broadcast():
            try:
                server.broadcast(0.5, np.zeros(1), np.zeros(1))
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=broadcast, daemon=True)
        test = self

        class Client:
            request = SimpleNamespace(
                path=f"/display/{server._token}" + ("?scene=1" if scene else ""),
                headers={},
            )

            def send(self, payload):
                messages.append(payload)
                if isinstance(payload, str):
                    # Model a receiver starting a step before send returns.
                    # Without handshake locking, force that step to finish
                    # before the handler can register this client.
                    worker.start()
                    if not server._lock.locked():
                        test.assertTrue(finished.wait(5))

            def __iter__(self):
                test.assertTrue(finished.wait(5))
                return iter(())

        server._handler(Client())
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], str)
        self.assertIsInstance(messages[1], bytes)

    def test_scene_exports_parser_model_and_world_transforms(self):
        model, data = self.model_and_data()
        provider = MujocoSceneProvider(model, data, mujoco, "Test scene")
        server = _display.DisplayServer(
            {"qposDim": model.nq, "qvelDim": model.nv},
            scene_provider=provider,
        )
        server.start()
        self.addCleanup(server.stop)
        with connect(server.url + "?scene=1", origin="http://localhost:5173") as scene:
            message = json.loads(scene.recv(timeout=5))
            self.assertEqual(message["type"], "scene")
            self.assertEqual(message["version"], 1)
            self.assertEqual(message["project"], "Test scene")
            self.assertEqual(message["model"]["ngeom"], 2)
            self.assertEqual(message["model"]["geom_type"], [6, 7])
            self.assertEqual(message["geomTypes"]["mjGEOM_MESH"], 7)
            self.assertEqual(message["model"]["mesh_facenum"], model.mesh_facenum.tolist())
            self.assertEqual(message["model"]["mesh_face"], model.mesh_face.ravel().tolist())
            self.assertEqual(message["model"]["names"], list(model.names))
            self.assertGreater(len(message["model"]["mesh_vert"]), 0)
            for name in (
                "geom_pos", "geom_quat", "geom_size", "geom_rgba", "mesh_vert",
                "mesh_face", "mesh_faceadr", "mesh_facenum", "mesh_normal",
                "mesh_facenormal", "mat_texid", "tex_data", "hfield_data",
            ):
                self.assertEqual(message["model"][name], np.asarray(getattr(model, name)).ravel().tolist())
            self.assertEqual(message["frameFields"][0], {"name": "xpos", "size": 6})
            self.assertEqual(message["frameFields"][1], {"name": "xquat", "size": 8})

            data.qpos[:3] = [4, 5, 6]
            mujoco.mj_forward(model, data)
            server.broadcast(1.25, data.qpos, data.qvel)
            values = np.frombuffer(scene.recv(timeout=5), dtype="<f4")
            self.assertEqual(len(values), 1 + sum(field["size"] for field in message["frameFields"]))
            self.assertEqual(values[0], 1.25)
            np.testing.assert_allclose(values[1:7], data.xpos.ravel())
            np.testing.assert_allclose(values[7:15], data.xquat.ravel())

    def test_scene_frame_includes_dynamic_heightfield_values(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><asset><hfield name="h" nrow="2" ncol="2" '
            'size="1 1 1 .1"/></asset><worldbody><geom type="hfield" '
            'hfield="h"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        provider = MujocoSceneProvider(model, data, mujoco, "Terrain")
        message = json.loads(provider.payload)
        self.assertEqual(message["model"]["hfield_data"], [0, 0, 0, 0])
        self.assertEqual(message["frameFields"][-1], {"name": "hfield_data", "size": 4})
        model.hfield_data[:] = [0.0, 0.25, 0.5, 1.0]
        values = np.frombuffer(provider.frame(0.5), dtype="<f4")
        np.testing.assert_array_equal(values[-4:], model.hfield_data)

    def test_scene_uses_integrated_pose_without_mutating_physics_data(self):
        model, data = self.model_and_data()
        model.opt.gravity[:] = 0
        model.opt.timestep = 0.01
        data.qvel[0] = 1
        mujoco.mj_forward(model, data)
        provider = MujocoSceneProvider(model, data, mujoco, "Motion")
        mujoco.mj_step(model, data)
        self.assertNotEqual(data.qpos[0], data.xpos[1, 0])
        stale_xpos = data.xpos.copy()
        values = np.frombuffer(provider.frame(data.time), dtype="<f4")
        self.assertAlmostEqual(values[1 + 3], data.qpos[0])
        np.testing.assert_array_equal(data.xpos, stale_xpos)

    def test_scene_lights_match_forward_kinematics(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".01" gravity="0 0 0"/>'
            '<worldbody><body name="robot"><freejoint/>'
            '<geom type="sphere" size=".1"/>'
            '<light pos="1 2 3" dir="1 0 -1"/>'
            '</body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        data.qvel[0] = 1
        mujoco.mj_forward(model, data)
        provider = MujocoSceneProvider(model, data, mujoco, "Light")
        mujoco.mj_step(model, data)
        stale_lights = data.light_xpos.copy()
        values = np.frombuffer(provider.frame(data.time), dtype="<f4")
        reference = mujoco.MjData(model)
        reference.qpos[:] = data.qpos
        mujoco.mj_forward(model, reference)
        fields = json.loads(provider.payload)["frameFields"]
        offset = 1
        for field in fields:
            if field["name"] == "light_xpos":
                np.testing.assert_allclose(values[offset:offset + 3], reference.light_xpos.ravel())
            if field["name"] == "light_xdir":
                np.testing.assert_allclose(values[offset:offset + 3], reference.light_xdir.ravel())
                self.assertAlmostEqual(np.linalg.norm(values[offset:offset + 3]), 1, places=6)
            offset += field["size"]
        np.testing.assert_array_equal(data.light_xpos, stale_lights)

    def test_scene_negotiation_does_not_change_legacy_stream(self):
        model, data = self.model_and_data()
        server = _display.DisplayServer(
            {"qposDim": model.nq, "qvelDim": model.nv},
            scene_provider=MujocoSceneProvider(model, data, mujoco, "Test scene"),
        )
        server.start()
        self.addCleanup(server.stop)
        with connect(server.url) as legacy, connect(server.url + "?scene=1") as scene:
            self.assertEqual(json.loads(legacy.recv(timeout=5))["type"], "meta")
            self.assertEqual(json.loads(scene.recv(timeout=5))["type"], "scene")
            server.broadcast(0.5, data.qpos, data.qvel)
            np.testing.assert_array_equal(
                np.frombuffer(legacy.recv(timeout=5), dtype="<f4"),
                np.concatenate(([0.5], data.qpos, data.qvel)).astype("<f4"),
            )
            self.assertEqual(
                len(scene.recv(timeout=5)), server._scene_provider.frame_size
            )
            server._next_scene_frame_at = float("inf")
            server.broadcast(0.6, data.qpos, data.qvel)
            self.assertIsInstance(legacy.recv(timeout=5), bytes)
            with self.assertRaises(TimeoutError):
                scene.recv(timeout=0.1)
            server._next_scene_frame_at = 0
            server.broadcast(0.7, data.qpos, data.qvel)
            self.assertIsInstance(legacy.recv(timeout=5), bytes)
            self.assertIsInstance(scene.recv(timeout=5), bytes)
            server.update_metrics({"speed": 1})
            self.assertEqual(json.loads(legacy.recv(timeout=5))["type"], "metrics")
            with self.assertRaises(TimeoutError):
                scene.recv(timeout=0.1)

    def test_scene_rejects_bad_token_query_and_origin(self):
        model, data = self.model_and_data()
        server = _display.DisplayServer(
            {}, scene_provider=MujocoSceneProvider(model, data, mujoco, "Test scene")
        )
        server.start()
        self.addCleanup(server.stop)
        for url, origin in (
            (server.url + "?scene=0", None),
            (server.url + "?scene=1&extra=1", None),
            (server.url.replace("/display/", "/display/wrong") + "?scene=1", None),
            (server.url + "?scene=1", "https://example.com"),
        ):
            with self.subTest(url=url, origin=origin):
                with connect(url, origin=origin) as client:
                    with self.assertRaises(Exception):
                        client.recv(timeout=1)


if __name__ == "__main__":
    unittest.main()
