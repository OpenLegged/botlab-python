"""Camera observation tests: pose algebra, frame shapes, config, crop, depth post-processing."""

import math
import unittest

import mujoco
import numpy as np

from botlab._runtime import _observations as obs
from botlab._runtime._observations import _camera as camera

from .obs_fixtures import HALF_SQRT2, MJCF, make_forward_model


class CameraPoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def pose(self, spec, model=None, data=None):
        return camera._camera_pose(model or self.model, data or self.data, spec)

    def test_default_euler_looks_along_body_x(self):
        position, forward, up = self.pose({"target": "torso", "cameraOffset": [0.2, 0, 0]})
        np.testing.assert_allclose(position, [0.2, 0, 1], atol=1e-6)
        np.testing.assert_allclose(forward, [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-6)

    def test_yaw_alignment_and_body_offset_frames(self):
        # 90 deg about x: yaw extraction gives 0, so the 'ray' frame keeps
        # world axes while the 'body' frame rotates the offset with the body.
        model, data = make_forward_model(quat=[HALF_SQRT2, HALF_SQRT2, 0, 0])
        ray_position, ray_forward, _ = self.pose(
            {"target": "torso", "rayAlignment": "yaw", "cameraOffset": [0, 0.2, 0]},
            model,
            data,
        )
        body_position, body_forward, _ = self.pose(
            {
                "target": "torso",
                "rayAlignment": "yaw",
                "cameraOffsetFrame": "body",
                "cameraOffset": [0, 0.2, 0],
            },
            model,
            data,
        )
        # Ray frame keeps world axes: offset stays [0, 0.2, 0]; body frame
        # applies Rx(90): [0, 0.2, 0] -> [0, 0, 0.2]. Forward stays yaw-only.
        np.testing.assert_allclose(ray_position, [0, 0.2, 1], atol=1e-6)
        np.testing.assert_allclose(body_position, [0, 0, 1.2], atol=1e-6)
        np.testing.assert_allclose(ray_forward, [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(body_forward, [1, 0, 0], atol=1e-6)

    def test_yaw_alignment_rotates_with_heading(self):
        model, data = make_forward_model(quat=[HALF_SQRT2, 0, 0, HALF_SQRT2])
        position, forward, up = self.pose(
            {"target": "torso", "rayAlignment": "yaw", "cameraOffset": [0.2, 0, 0]},
            model,
            data,
        )
        np.testing.assert_allclose(position, [0, 0.2, 1], atol=1e-6)
        np.testing.assert_allclose(forward, [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-6)

    def test_site_target_with_xyaxes_rotation(self):
        model, data = make_forward_model(quat=[HALF_SQRT2, 0, 0, HALF_SQRT2])
        position, forward, up = self.pose(
            {
                "target": "imu_site",
                "targetKind": "site",
                "cameraRotationMode": "xyaxes",
                "cameraXyaxes": [0, 0, -1, 1, 0, 0],
            },
            model,
            data,
        )
        np.testing.assert_allclose(position, [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(forward, [-1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(up, [0, 1, 0], atol=1e-6)

    def test_camera_target_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.pose({"target": "leg"})
        self.assertIn('body target "leg" not found. Available:', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.pose({"target": "42"})
        self.assertIn("body index 42 is out of bounds for count 5", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.pose(
                {
                    "target": "torso",
                    "cameraRotationMode": "xyaxes",
                }
            )
        self.assertIn(
            "cameraXyaxes is required when cameraRotationMode is xyaxes", str(ctx.exception)
        )


class _FakeCameraView:
    def __init__(self):
        self.pos = np.zeros(3)
        self.forward = np.zeros(3)
        self.up = np.zeros(3)
        self.frustum_near = 0.0
        self.frustum_far = 0.0
        self.frustum_bottom = 0.0
        self.frustum_top = 0.0
        self.frustum_center = 0.0
        self.frustum_width = 0.0
        self.orthographic = 1


class _FakeRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scene = type("Scene", (), {})()
        self.scene.camera = [_FakeCameraView(), _FakeCameraView()]
        self.updates = 0
        self.depth_enabled = False
        self.scene_options = []
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.rgb[0, 0] = [255, 128, 0]
        self.depth = np.full((height, width), 5.0, dtype=np.float32)
        self.depth[0, 1] = 20.0

    def update_scene(self, data, camera=None, scene_option=None):
        self.updates += 1
        self.scene_options.append(scene_option)

    def enable_depth_rendering(self):
        self.depth_enabled = True

    def disable_depth_rendering(self):
        self.depth_enabled = False

    def render(self):
        if self.depth_enabled:
            return self.depth
        return self.rgb

    def close(self):
        pass


class CameraFramesShapeTest(unittest.TestCase):
    def setUp(self):
        self.model, self.data = make_forward_model()
        self.renderer = None

    def tearDown(self):
        obs.dispose_camera_renderers()

    def fake_renderer(self, model, width, height):
        self.renderer = _FakeRenderer(width, height)
        return self.renderer

    def frames(self, spec):
        original = camera._create_renderer
        camera._create_renderer = self.fake_renderer
        try:
            return obs.camera_frames(self.model, self.data, spec)
        finally:
            camera._create_renderer = original

    def test_rgb_normalization_and_shape(self):
        frames = self.frames({"target": "torso", "width": 8, "height": 6, "outputMode": "rgb"})
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (8 * 6 * 3,))
        self.assertEqual(frames[0].dtype, np.float32)
        self.assertAlmostEqual(frames[0][0], 1.0, places=6)
        self.assertAlmostEqual(frames[0][1], 128 / 255, places=6)
        self.assertEqual(frames[0][2], 0.0)

    def test_depth_clamps_to_unit_range(self):
        frames = self.frames(
            {"target": "torso", "width": 8, "height": 6, "outputMode": "depth", "depthScale": 10}
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (8 * 6,))
        # 5 m / 10 = 0.5; 20 m / 10 clamps to 1.
        self.assertAlmostEqual(frames[0][0], 0.5, places=6)
        self.assertAlmostEqual(frames[0][1], 1.0, places=6)

    def test_rgbd_channel_order_and_age_channel(self):
        frames = self.frames(
            {"target": "torso", "width": 4, "height": 4, "outputMode": "rgbd"}
        )
        self.assertEqual([f.shape for f in frames], [(4 * 4 * 3,), (4 * 4,)])

        aged = self.frames(
            {
                "target": "torso",
                "width": 4,
                "height": 4,
                "outputMode": "depth",
                "includeAgeSteps": True,
            }
        )
        self.assertEqual([f.shape for f in aged], [(4 * 4,), (1,)])
        np.testing.assert_allclose(aged[1], [0.0])

    def test_pose_and_frustum_written_to_scene_camera(self):
        self.frames({"target": "torso", "cameraOffset": [0.2, 0, 0], "fov": 90, "near": 0.05, "far": 5})
        view = self.renderer.scene.camera[0]
        np.testing.assert_allclose(view.pos, [0.2, 0, 1], atol=1e-6)
        np.testing.assert_allclose(view.forward, [1, 0, 0], atol=1e-6)
        self.assertEqual(view.frustum_near, 0.05)
        self.assertEqual(view.frustum_far, 5)
        self.assertAlmostEqual(view.frustum_top, 0.05 * math.tan(math.radians(45)), places=6)
        # 0 hands the halfwidth to mjr_render's viewport-aspect fallback —
        # the full-sensor aspect projection of cameraObservationPose.ts.
        self.assertEqual(view.frustum_width, 0.0)
        self.assertEqual(self.renderer.updates, 1)


class CameraRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()
        try:
            probe = camera._create_renderer(cls.model, 4, 4)
        except Exception as error:
            raise unittest.SkipTest(
                "MuJoCo GL renderer unavailable on this machine "
                f"({error}); set MUJOCO_GL=egl to enable the real-render "
                "camera test"
            ) from error
        probe.close()

    def tearDown(self):
        obs.dispose_camera_renderers()

    def test_real_render_shapes_and_ranges(self):
        frames = obs.camera_frames(
            self.model,
            self.data,
            {
                "target": "torso",
                "width": 32,
                "height": 24,
                "outputMode": "rgbd",
                "cameraOffset": [0.2, 0, 0],
                "far": 12,
                "depthScale": 5,
            },
        )
        self.assertEqual([f.shape for f in frames], [(32 * 24 * 3,), (32 * 24,)])
        self.assertEqual(frames[0].dtype, np.float32)
        self.assertTrue(np.all(frames[0] >= 0.0) and np.all(frames[0] <= 1.0))
        self.assertTrue(np.all(frames[1] >= 0.0) and np.all(frames[1] <= 1.0))
        # The torso box fills part of the view (non-black pixels) and the
        # sky clamps to the depth ceiling.
        self.assertGreater(frames[0].max(), 0.0)
        self.assertAlmostEqual(frames[1].max(), 1.0, places=5)

    def test_real_render_full_sensor_crop_and_metric_depth(self):
        # The g1 parkour depth camera geometry: 64x36 full sensor cropped to
        # 32x18, with a block at a known distance in front of the camera.
        model = mujoco.MjModel.from_xml_string(
            MJCF.replace(
                "</worldbody>",
                '<body name="wall" pos="5 0 1"><geom type="box" size="1 1 1"/></body>'
                "</worldbody>",
            )
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        try:
            frames = obs.camera_frames(
                model,
                data,
                {
                    "target": "torso",
                    "width": 32,
                    "height": 18,
                    "fullWidth": 64,
                    "fullHeight": 36,
                    "cropX": 16,
                    "cropY": 18,
                    "outputMode": "depth",
                    "fov": 58.29,
                    "near": 0.1,
                    "far": 8,
                    "depthScale": 8,
                    "cameraOffset": [0.3, 0, 0],
                },
            )
            self.assertEqual(frames[0].shape, (32 * 18,))
            depth = frames[0].reshape(18, 32)
            # The wall's front face is a plane 3.7 m ahead of the camera, so
            # every wall pixel reads the same view-Z depth; the crop's row 1
            # sits inside the wall (row 0 keeps the sky at the depth ceiling).
            self.assertAlmostEqual(depth[1, 12], 3.7 / 8.0, delta=0.02)
            self.assertAlmostEqual(depth[0, 0], 1.0, delta=1e-4)
            # cropY 18 selects the bottom half of the sensor, so the crop's
            # centre looks below the horizon and meets the floor closer than
            # the wall.
            self.assertLess(depth[9, 16], 3.7 / 8.0)
        finally:
            obs.dispose_camera_renderers()

    def test_real_render_geom_group_filter(self):
        model = mujoco.MjModel.from_xml_string(
            MJCF.replace(
                "</worldbody>",
                '<body name="front" pos="0.8 0 1"><geom name="front_geom" '
                'type="box" size="0.2 0.2 0.2" group="2"/></body>'
                "</worldbody>",
            )
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        base = {
            "target": "torso",
            "width": 32,
            "height": 24,
            "outputMode": "depth",
            "far": 12,
            "depthScale": 12,
            "cameraOffset": [0.2, 0, 0],
        }
        try:
            with_group = obs.camera_frames(
                model, data, dict(base, raycastGeomGroups=[0, 2])
            )[0].reshape(24, 32)[12, 16]
            without_group = obs.camera_frames(
                model, data, dict(base, raycastGeomGroups=[0])
            )[0].reshape(24, 32)[12, 16]
            # The group-2 box occludes the torso only when its group renders.
            self.assertAlmostEqual(with_group, 0.4 / 12.0, delta=0.03)
            self.assertGreater(without_group, with_group)
        finally:
            obs.dispose_camera_renderers()


class CameraConfigTest(unittest.TestCase):
    """read_camera_config mirror of cameraObservationConfig.ts."""

    def test_defaults_match_camera_observation(self):
        config = obs.read_camera_config({"target": "torso"})
        self.assertEqual(config["width"], 64)
        self.assertEqual(config["height"], 64)
        self.assertEqual(config["fullWidth"], 64)
        self.assertEqual(config["fullHeight"], 64)
        self.assertEqual(config["cropX"], 0)
        self.assertEqual(config["cropY"], 0)
        self.assertEqual(config["fov"], 60)
        self.assertEqual(config["near"], 0.02)
        self.assertEqual(config["far"], 10)
        self.assertEqual(config["depthScale"], 10)
        self.assertEqual(config["depthGaussianKernelSize"], 0)
        self.assertEqual(config["depthGaussianSigma"], 1)
        self.assertEqual(config["depthNoiseStd"], 0)
        self.assertEqual(config["depthNoiseDistanceStd"], 0)
        self.assertEqual(config["depthDropoutProbability"], 0)
        self.assertEqual(config["depthDropoutPatchCount"], 0)
        self.assertEqual(config["depthDropoutPatchRadius"], 0)
        self.assertEqual(config["depthNoiseSeed"], 1)
        self.assertEqual(config["outputMode"], "rgbd")
        self.assertEqual(config["raycastGeomGroups"], [])
        self.assertEqual(config["excludeGeoms"], [])
        self.assertEqual(config["cameraOffset"], [0, 0, 0])
        self.assertEqual(config["cameraRotationMode"], "euler")
        self.assertIsNone(config["cameraXyaxes"])

    def test_full_and_crop_dimensions(self):
        config = obs.read_camera_config(
            {
                "target": "torso",
                "width": 32,
                "height": 18,
                "fullWidth": 64,
                "fullHeight": 36,
                "cropX": 16,
                "cropY": 18,
            }
        )
        self.assertEqual(
            (config["width"], config["height"], config["fullWidth"], config["fullHeight"]),
            (32, 18, 64, 36),
        )
        self.assertEqual((config["cropX"], config["cropY"]), (16, 18))

    def test_numeric_strings_parse_like_js_number(self):
        config = obs.read_camera_config(
            {"target": "torso", "width": "32", "height": 18.0, "depthNoiseSeed": 5.0}
        )
        self.assertEqual(config["width"], 32)
        self.assertEqual(config["height"], 18)
        self.assertEqual(config["depthNoiseSeed"], 5)

    def test_crop_window_validation(self):
        with self.assertRaises(ValueError) as ctx:
            obs.read_camera_config(
                {"target": "torso", "width": 49, "height": 18, "fullWidth": 64, "cropX": 16}
            )
        self.assertIn("cropX (16) + width (49) exceeds fullWidth (64)", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            obs.read_camera_config(
                {"target": "torso", "width": 32, "height": 20, "fullHeight": 36, "cropY": 18}
            )
        self.assertIn("cropY (18) + height (20) exceeds fullHeight (36)", str(ctx.exception))

    def test_field_validation_errors(self):
        cases = [
            ({"target": "torso", "width": 0}, "width must be a positive integer"),
            ({"target": "torso", "height": 2.5}, "height must be a positive integer"),
            ({"target": "torso", "fullWidth": -1}, "fullWidth must be a positive integer"),
            ({"target": "torso", "cropX": -2}, "cropX must be a non-negative integer"),
            ({"target": "torso", "fov": 0}, "fov must be a finite number > 0"),
            ({"target": "torso", "depthScale": 0}, "depthScale must be a finite number > 0"),
            (
                {"target": "torso", "depthGaussianSigma": 0},
                "depthGaussianSigma must be a finite number > 0",
            ),
            (
                {"target": "torso", "depthNoiseStd": -1},
                "depthNoiseStd must be a finite number >= 0",
            ),
            (
                {"target": "torso", "depthNoiseSeed": -3},
                "depthNoiseSeed must be a non-negative integer",
            ),
            (
                {"target": "torso", "depthDropoutProbability": 1.5},
                "depthDropoutProbability must be a finite number between 0 and 1",
            ),
            (
                {"target": "torso", "depthGaussianKernelSize": 2},
                "depthGaussianKernelSize must be 0/1 or an odd integer >= 3",
            ),
            (
                {"target": "torso", "depthGaussianKernelSize": 4},
                "depthGaussianKernelSize must be 0/1 or an odd integer >= 3",
            ),
            ({"target": "torso", "readbackPipelineDepth": 5}, "readbackPipelineDepth must be <= 4"),
            ({"target": "torso", "outputMode": "gray"}, "outputMode must be"),
            ({"target": "torso", "depthBackend": "cuda"}, "depthBackend must be render"),
            (
                {"target": "torso", "cameraOffset": [1, 2]},
                "cameraOffset must contain exactly 3 finite numbers",
            ),
            (
                {"target": "torso", "raycastGeomGroups": [0, 1.5]},
                "raycastGeomGroups must contain non-negative integers",
            ),
            (
                {"target": "torso", "excludeGeoms": [3]},
                "excludeGeoms must be a list of names",
            ),
            (
                {"target": "torso", "excludeGeoms": ["a", " "]},
                "excludeGeoms must not contain empty names",
            ),
            ({"target": "torso", "far": 0.02}, "far must be greater than near"),
        ]
        for spec, message in cases:
            with self.assertRaises(ValueError) as ctx:
                obs.read_camera_config(spec)
            self.assertIn(message, str(ctx.exception))

    def test_target_required(self):
        with self.assertRaises(ValueError) as ctx:
            obs.read_camera_config({"target": "  "})
        self.assertIn("target is required", str(ctx.exception))

    def test_enum_and_special_fields(self):
        config = obs.read_camera_config(
            {
                "target": "torso",
                "outputMode": "depth",
                "targetKind": "site",
                "rayAlignment": "yaw",
                "cameraOffsetFrame": "body",
                "depthBackend": "raycaster",
                "readbackMode": "sync",
                "depthReadbackFormat": "linear-float32",
                "raycastGeomGroups": [0, 2],
                "cameraXyaxes": [1, 0, 0, 0, 0, 1],
            }
        )
        self.assertEqual(config["outputMode"], "depth")
        self.assertEqual(config["targetKind"], "site")
        self.assertEqual(config["depthBackend"], "render")
        self.assertEqual(config["readbackMode"], "sync")
        self.assertEqual(config["depthReadbackFormat"], "linear-float32")
        self.assertEqual(config["raycastGeomGroups"], [0, 2])
        # cameraXyaxes presence infers the xyaxes rotation mode (TS rule).
        self.assertEqual(config["cameraRotationMode"], "xyaxes")
        self.assertEqual(config["cameraXyaxes"], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_resource_config_merges_under_node_data(self):
        import botlab_host

        state = botlab_host.project_state()
        botlab_host.declareResource(
            "sensor",
            "cam-res-1",
            "scene.xml",
            None,
            None,
            {
                "outputMode": "depth",
                "width": 32,
                "height": 18,
                "fullWidth": 64,
                "fullHeight": 36,
                "cropX": 16,
                "cropY": 18,
                "fov": 58.29,
            },
            None,
        )
        try:
            config = obs.read_camera_config(
                {"resourceId": "cam-res-1", "nodeId": "cam-node"}
            )
            self.assertEqual(config["outputMode"], "depth")
            self.assertEqual((config["width"], config["height"]), (32, 18))
            self.assertEqual((config["cropX"], config["cropY"]), (16, 18))
            self.assertAlmostEqual(config["fov"], 58.29)
            # Node-level keys win over the resource config.
            overridden = obs.read_camera_config(
                {"resourceId": "cam-res-1", "fov": 40}
            )
            self.assertEqual(overridden["fov"], 40)
            self.assertEqual(obs.camera_output_mode({"resourceId": "cam-res-1"}), "depth")
            self.assertEqual(
                obs.camera_frame_size({"resourceId": "cam-res-1"}), (32, 18)
            )
            processor = obs.ObservationProcessor()
            processor.init(
                {"nodeType": "obs-camera", "resourceId": "cam-res-1", "includeAgeSteps": True}, []
            )
            self.assertEqual([len(b) for b in processor.output_buffers], [32 * 18, 1])
        finally:
            state.reset(state.directory)

    def test_unknown_camera_resource_errors(self):
        with self.assertRaises(ValueError) as ctx:
            obs.read_camera_config({"resourceId": "no-such-sensor"})
        self.assertIn('camera resource "no-such-sensor" could not be resolved', str(ctx.exception))


class CameraCropTest(unittest.TestCase):
    """Full-sensor render + crop window through the mocked Renderer."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def setUp(self):
        self.renderer = None

    def tearDown(self):
        obs.dispose_camera_renderers()

    def fake_renderer(self, model, width, height):
        self.renderer = _FakeRenderer(width, height)
        return self.renderer

    def frames(self, spec):
        original = camera._create_renderer
        camera._create_renderer = self.fake_renderer
        try:
            return obs.camera_frames(self.model, self.data, spec)
        finally:
            camera._create_renderer = original

    def test_renderer_created_at_full_sensor_size(self):
        self.frames(
            {
                "target": "torso",
                "width": 32,
                "height": 18,
                "fullWidth": 64,
                "fullHeight": 36,
                "outputMode": "rgbd",
            }
        )
        self.assertEqual((self.renderer.width, self.renderer.height), (64, 36))

    def test_crop_selects_the_window(self):
        renderer = None

        def factory(model, width, height):
            nonlocal renderer
            renderer = _FakeRenderer(width, height)
            # Markers: (0,0) is outside the crop window, (20, 20) inside.
            renderer.rgb[:] = 0
            renderer.rgb[0, 0] = [255, 0, 0]
            renderer.rgb[20, 20] = [0, 255, 0]
            renderer.depth = np.arange(36 * 64, dtype=np.float32).reshape(36, 64) / 100.0
            return renderer

        original = camera._create_renderer
        camera._create_renderer = factory
        try:
            frames = obs.camera_frames(
                self.model,
                self.data,
                {
                    "target": "torso",
                    "width": 32,
                    "height": 18,
                    "fullWidth": 64,
                    "fullHeight": 36,
                    "cropX": 16,
                    "cropY": 18,
                    "outputMode": "rgbd",
                    "depthScale": 36.0,
                },
            )
        finally:
            camera._create_renderer = original
        self.assertEqual(frames[0].shape, (18 * 32 * 3,))
        self.assertEqual(frames[1].shape, (18 * 32,))
        rgb = frames[0].reshape(18, 32, 3)
        # Row 20, col 20 of the full frame lands at crop row 2, col 4.
        np.testing.assert_allclose(rgb[2, 4], [0.0, 1.0, 0.0], atol=1e-6)
        # The (0, 0) marker is outside the window: no pure red anywhere.
        self.assertFalse(np.any((rgb[..., 0] > 0.9) & (rgb[..., 1] < 0.1)))
        depth = frames[1].reshape(18, 32)
        np.testing.assert_allclose(
            depth,
            renderer.depth[18:36, 16:48] / 36.0,
            atol=1e-7,
        )
    def test_crop_window_overflow_errors_at_render(self):
        with self.assertRaises(ValueError) as ctx:
            self.frames(
                {
                    "target": "torso",
                    "width": 49,
                    "height": 18,
                    "fullWidth": 64,
                    "fullHeight": 36,
                    "cropX": 16,
                }
            )
        self.assertIn("cropX (16) + width (49) exceeds fullWidth (64)", str(ctx.exception))

    def test_geom_groups_reach_the_scene_option(self):
        self.frames(
            {"target": "torso", "width": 8, "height": 6, "raycastGeomGroups": [0, 2]}
        )
        option = self.renderer.scene_options[-1]
        self.assertEqual(list(option.geomgroup), [1, 0, 1, 0, 0, 0])
        self.frames({"target": "torso", "width": 8, "height": 6})
        option = self.renderer.scene_options[-1]
        # No filter keeps mujoco's own default visibility (groups 0-2 on).
        self.assertEqual(list(option.geomgroup), [1, 1, 1, 0, 0, 0])


def _reference_gaussian(frame, kernel_size, sigma):
    """Independent separable 'same' convolution with edge replication."""
    radius = kernel_size // 2
    weights = [
        math.exp(-((k * k) / (2.0 * sigma * sigma))) for k in range(-radius, radius + 1)
    ]
    weight_sum = sum(weights)
    height, width = frame.shape
    horizontal = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            accumulator = 0.0
            for k in range(-radius, radius + 1):
                sx = min(max(x + k, 0), width - 1)
                accumulator += float(frame[y, sx]) * weights[k + radius]
            horizontal[y, x] = accumulator / weight_sum
    horizontal = horizontal.astype(np.float32)
    vertical = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            accumulator = 0.0
            for k in range(-radius, radius + 1):
                sy = min(max(y + k, 0), height - 1)
                accumulator += float(horizontal[sy, x]) * weights[k + radius]
            vertical[y, x] = accumulator / weight_sum
    return vertical.astype(np.float32)


class DepthPostprocessTest(unittest.TestCase):
    """The depth post-processing chain of cameraObservationRuntime.ts."""

    def config(self, **overrides):
        spec = {
            "target": "torso",
            "width": 4,
            "height": 3,
            "outputMode": "depth",
            "far": 10,
            "depthScale": 10,
            "depthNoiseSeed": 1,
        }
        spec.update(overrides)
        return obs.read_camera_config(spec)

    def test_gaussian_blur_matches_reference(self):
        frame = np.array(
            [
                [0.1, 0.9, 0.3, 0.7],
                [0.5, 0.0, 1.0, 0.2],
                [0.8, 0.4, 0.6, 0.1],
            ],
            dtype=np.float32,
        )
        for size, sigma in ((3, 1.0), (3, 2.5), (5, 1.0)):
            blurred = frame.copy()
            camera._apply_gaussian_blur(blurred, size, sigma)
            np.testing.assert_allclose(
                blurred, _reference_gaussian(frame, size, sigma), atol=1e-7
            )

    def test_gaussian_kernel_size_zero_or_one_is_identity(self):
        frame = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        blurred = frame.copy()
        camera._apply_gaussian_blur(blurred, 1, 1.0)
        np.testing.assert_allclose(blurred, frame, atol=0.0)

    def test_scale_and_clip_happen_before_blur(self):
        model, data = make_forward_model()
        def factory(model_arg, width, height):
            renderer_box = _FakeRenderer(width, height)
            # One column of 15 m clamps to 1.0 after /10; the neighbours are 3 m.
            renderer_box.depth = np.full((3, 3), 3.0, dtype=np.float32)
            renderer_box.depth[:, 1] = 15.0
            return renderer_box

        original = camera._create_renderer
        camera._create_renderer = factory
        try:
            frames = obs.camera_frames(
                model,
                data,
                {
                    "target": "torso",
                    "width": 3,
                    "height": 3,
                    "outputMode": "depth",
                    "depthScale": 10,
                    "depthGaussianKernelSize": 3,
                    "depthGaussianSigma": 1,
                },
            )
        finally:
            camera._create_renderer = original
        scaled = np.clip(np.array([[0.3, 1.0, 0.3]] * 3, dtype=np.float32), 0.0, 1.0)
        expected = _reference_gaussian(scaled, 3, 1.0)
        np.testing.assert_allclose(frames[0].reshape(3, 3), expected, atol=1e-6)
        # Blur-then-scale would give 1.0 in the middle column instead.
        self.assertLess(frames[0][4], 1.0)

    def test_mulberry32_matches_js_semantics(self):
        def js_random(state):
            values = []
            for _ in range(32):
                state = (state + 0x6D2B79F5) & 0xFFFFFFFF
                value = state
                value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFFFFFF
                value = (
                    value ^ (value + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFFFFFF))
                ) & 0xFFFFFFFF
                values.append(((value ^ (value >> 14)) & 0xFFFFFFFF) / 4294967296)
            return values

        for state in (0, 1, 123456789, 0xFFFFFFFF):
            np.testing.assert_allclose(
                camera._mulberry32_uniforms(state, 32), js_random(state), atol=0.0
            )

    def test_fnv1a_hash(self):
        self.assertEqual(camera._fnv1a_hash(""), 2166136261)
        reference = 2166136261
        for char in "cam-node":
            reference = ((reference ^ ord(char)) * 16777619) & 0xFFFFFFFF
        self.assertEqual(camera._fnv1a_hash("cam-node"), reference)

    def test_noise_is_reproducible_and_frame_decorrelated(self):
        config = self.config(depthNoiseStd=0.05, depthNoiseDistanceStd=0.01)
        frame = np.full((3, 4), 0.5, dtype=np.float32)

        first = frame.copy()
        camera._corrupt_camera_depth(config, "cam", 0, first)
        replay = frame.copy()
        camera._corrupt_camera_depth(config, "cam", 0, replay)
        np.testing.assert_array_equal(first, replay)

        later = frame.copy()
        camera._corrupt_camera_depth(config, "cam", 1, later)
        self.assertFalse(np.array_equal(first, later))
        # Corrupted values stay in [0, 1] and differ from the input.
        self.assertTrue(np.all(later >= 0.0) and np.all(later <= 1.0))
        self.assertTrue(np.any(later != 0.5))

    def test_distance_std_scales_noise_with_depth(self):
        config = self.config(depthNoiseStd=0, depthNoiseDistanceStd=0.5)
        frame = np.zeros((1, 4), dtype=np.float32)
        # Zero scaled depth -> zero std -> untouched (mirrors TS `std <= 0` skip).
        output = frame.copy()
        camera._corrupt_camera_depth(config, "cam", 0, output)
        np.testing.assert_array_equal(output, frame)

    def test_pixel_dropout_writes_miss_depth(self):
        config = self.config(depthDropoutProbability=1.0)
        frame = np.full((3, 4), 0.5, dtype=np.float32)
        camera._corrupt_camera_depth(config, "cam", 7, frame)
        np.testing.assert_allclose(frame, 10.0 / 10.0, atol=1e-7)

    def test_patch_dropout_writes_miss_depth_disc(self):
        config = self.config(depthDropoutPatchCount=2, depthDropoutPatchRadius=1)
        frame = np.full((6, 6), 0.5, dtype=np.float32)
        camera._corrupt_camera_depth(config, "cam", 0, frame)
        # Every patch writes clamped far/depthScale; the untouched majority
        # proves only discs changed, and the corruption stays in range.
        self.assertTrue(np.all(frame >= 0.0) and np.all(frame <= 1.0))
        self.assertTrue(np.any(frame == 1.0))
        self.assertTrue(np.any(frame == 0.5))

    def test_disabled_corruption_is_identity(self):
        config = self.config()
        frame = np.full((3, 4), 0.42, dtype=np.float32)
        output = frame.copy()
        camera._corrupt_camera_depth(config, "cam", 0, output)
        np.testing.assert_array_equal(output, frame)
        self.assertFalse(camera._camera_depth_corruption_enabled(config))

    def test_full_chain_noise_and_blur_reproducible(self):
        model, data = make_forward_model()

        def factory(model_arg, width, height):
            renderer = _FakeRenderer(width, height)
            renderer.depth = np.linspace(1.0, 4.0, 3 * 4, dtype=np.float32).reshape(3, 4)
            return renderer

        spec = {
            "target": "torso",
            "width": 4,
            "height": 3,
            "outputMode": "depth",
            "depthScale": 5.0,
            "depthGaussianKernelSize": 3,
            "depthGaussianSigma": 1,
            "depthNoiseStd": 0.02,
            "nodeId": "cam",
        }
        original = camera._create_renderer
        camera._create_renderer = factory
        try:
            obs.reset_camera_observation_state()
            first = obs.camera_frames(model, data, spec)[0]
            obs.reset_camera_observation_state()
            replay = obs.camera_frames(model, data, spec)[0]
            third = obs.camera_frames(model, data, spec)[0]
        finally:
            camera._create_renderer = original
        self.assertEqual(first.shape, (12,))
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, third))
        self.assertTrue(np.all(first >= 0.0) and np.all(first <= 1.0))

    def test_processor_reset_reanchors_the_noise_stream(self):
        model, data = make_forward_model()

        def factory(model_arg, width, height):
            renderer = _FakeRenderer(width, height)
            renderer.depth = np.full((3, 4), 2.0, dtype=np.float32)
            return renderer

        processor = obs.ObservationProcessor()
        spec = {
            "target": "torso",
            "width": 4,
            "height": 3,
            "outputMode": "depth",
            "depthNoiseStd": 0.05,
        }
        processor.init(dict(spec, nodeType="obs-camera", nodeId="cam-reset"), [])
        original = camera._create_renderer
        camera._create_renderer = factory
        try:
            obs.reset_camera_observation_state()
            first = obs.camera_frames(model, data, dict(spec, nodeId="cam-reset"))[0]
            processor.reset()
            replay = obs.camera_frames(model, data, dict(spec, nodeId="cam-reset"))[0]
        finally:
            camera._create_renderer = original
        np.testing.assert_array_equal(first, replay)
