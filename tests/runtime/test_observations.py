"""Observation assembly tests (tier B: requires mujoco + numpy).

Mirrors the TypeScript observation pipeline semantics against a small MJCF
(free joint + hinge + framequat/gyro/accelerometer sensors + plane floor +
boxes), with hand-computed expectations from the deterministic initial pose:

- per obs-type extraction values (qpos/qvel/dof slices, imu, projected
  gravity, base height/velocities, constants, actions, commands, queries);
- the transform chain (indices / scale / default) of
  mujocoObservationTransforms.ts;
- poseVelocityCommand waypoint tracking (yaw forward assist, lookahead
  rewrites against the root pose, per-node state anchoring/rebuild/clear,
  axis index/bound remapping and the explicit validation errors);
- height_scan rays (flat floor, box hit, cutoff floor, yaw vs world frame);
- camera pose algebra (euler default, yaw alignment, xyaxes, offset frames),
  frame shapes and normalization with a mocked Renderer, plus one real GL
  render when MUJOCO_GL is available (skipped with a clear reason otherwise);
- the obs-* REGISTRY key set against the hardcoded SimulationEngine.ts list
  (drift guard);
- explicit error shapes for bad inputs (node id + offending value).
"""

import functools
import math
import unittest

import numpy as np  # noqa: E402

from botlab._runtime import _observations as obs  # noqa: E402

try:
    import mujoco  # noqa: E402
except ImportError as import_error:  # pragma: no cover - tier-B guard
    raise ImportError(
        "test_observations requires the mujoco runtime extra "
        "(pip install botlab-python[runtime])"
    ) from import_error


# torso at (0, 0, 1) with identity orientation, hinge at 0, block at
# (0.5, 0, 0.1) with its top at z = 0.2, yaw box at (0, 0.3) with its top at
# z = 0.15, shin capsule top at z = 0.72 below the torso.
MJCF = """
<mujoco model="t4obs">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.1 0.1 0.2" mass="1"/>
      <site name="imu_site" pos="0 0 0"/>
      <body name="shin" pos="0 0 -0.3">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.5"/>
      </body>
    </body>
    <body name="block" pos="0.5 0 0.1">
      <geom name="block_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="yaw_box" pos="0 0.3 0.075">
      <geom name="yaw_box_geom" type="box" size="0.075 0.075 0.075"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge"/>
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_accel" site="imu_site"/>
  </sensor>
</mujoco>
"""

# SimulationEngine.ts ProcessorRegistry obs-* keys, hardcoded as the drift
# guard the TS side cannot provide at test time.
EXPECTED_OBS_NODE_TYPES = [
    "obs-qpos",
    "obs-qvel",
    "obs-imu",
    "obs-constant",
    "obs-base-height",
    "obs-base-lin-vel",
    "obs-base-ang-vel",
    "obs-projected-gravity",
    "obs-commands",
    "obs-dof-pos",
    "obs-dof-vel",
    "obs-actions",
    "obs-camera",
    "obs-mujoco-query",
    "obs-height-scan",
]

HALF_SQRT2 = math.sqrt(2) / 2


def make_forward_model(quat=None, qvel=None, ctrl=None):
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    if quat is not None:
        data.qpos[3:7] = quat
    if qvel is not None:
        data.qvel[:] = qvel
    if ctrl is not None:
        data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return model, data


def context(model, data, node_data=None, **extra):
    base = {"model": model, "data": data, "nodeData": node_data}
    base.update(extra)
    return base


def binding(node_id, node_type, data=None, processor=None, **extra):
    prepared = {
        "nodeId": node_id,
        "type": node_type,
        "data": data if data is not None else {},
        "processor": processor if processor is not None else obs.ObservationProcessor(),
        "qposOffset": 0,
        "qvelOffset": 0,
        "namePrefix": "",
    }
    prepared.update(extra)
    return prepared


class ObservationProcessorTest(unittest.TestCase):
    def test_init_dim_allocates_float32_buffer(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 4}, [])
        self.assertEqual(processor.output_buffers[0].shape, (4,))
        self.assertEqual(processor.output_buffers[0].dtype, np.float32)

    def test_init_dim_must_be_positive(self):
        with self.assertRaises(ValueError) as ctx:
            obs.ObservationProcessor().init({"dim": 0}, [])
        self.assertIn("[ObservationProcessor] dim must be a positive number", str(ctx.exception))

    def test_init_output_dims_validates_each_entry(self):
        with self.assertRaises(ValueError) as ctx:
            obs.ObservationProcessor().init({"outputDims": [3, -1]}, [])
        self.assertIn("[ObservationProcessor] outputDims[1] must be a positive integer", str(ctx.exception))

    def test_camera_init_buffer_shapes(self):
        rgb = obs.ObservationProcessor()
        rgb.init({"nodeType": "obs-camera", "outputMode": "rgb", "width": 8, "height": 6}, [])
        self.assertEqual([len(b) for b in rgb.output_buffers], [8 * 6 * 3])

        depth = obs.ObservationProcessor()
        depth.init(
            {"nodeType": "obs-camera", "outputMode": "depth", "width": 8, "height": 6,
             "includeAgeSteps": True},
            [],
        )
        self.assertEqual([len(b) for b in depth.output_buffers], [8 * 6, 1])

        rgbd = obs.ObservationProcessor()
        rgbd.init({"nodeType": "obs-camera", "outputMode": "rgbd", "width": 8, "height": 6}, [])
        self.assertEqual([len(b) for b in rgbd.output_buffers], [8 * 6 * 3, 8 * 6])

    def test_camera_include_age_steps_requires_depth(self):
        with self.assertRaises(ValueError) as ctx:
            obs.ObservationProcessor().init(
                {"nodeType": "obs-camera", "outputMode": "rgb", "includeAgeSteps": True}, []
            )
        self.assertIn(
            "[ObservationProcessor] includeAgeSteps requires a depth camera observation",
            str(ctx.exception),
        )

    def test_set_data_aliases_float32_and_copies_float64(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 3}, [])
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        processor.set_data(source)
        self.assertIs(processor.output_buffers[0], source)

        float64 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        buffer = processor.output_buffers[0]
        processor.set_data(float64)
        self.assertIs(processor.output_buffers[0], buffer)
        np.testing.assert_allclose(processor.output_buffers[0], [4.0, 5.0, 6.0])

    def test_set_data_all_replaces_every_slot(self):
        processor = obs.ObservationProcessor()
        processor.init(
            {"nodeType": "obs-camera", "outputMode": "rgbd", "width": 4, "height": 4}, []
        )
        frames = [np.zeros(48, dtype=np.float32), np.ones(16, dtype=np.float32)]
        processor.set_data_all(frames)
        self.assertEqual([len(b) for b in processor.output_buffers], [48, 16])
        self.assertIs(processor.output_buffers[1], frames[1])

    def test_compute_is_noop(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 2}, [])
        processor.set_data(np.array([1.0, 2.0], dtype=np.float32))
        processor.compute([])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0])


class RegistryTest(unittest.TestCase):
    def test_registry_matches_simulation_engine_obs_types(self):
        self.assertEqual(sorted(obs.REGISTRY), sorted(EXPECTED_OBS_NODE_TYPES))
        for node_type, processor_class in obs.REGISTRY.items():
            self.assertTrue(node_type.startswith("obs-"))
            self.assertIs(processor_class, obs.ObservationProcessor)


class TransformsTest(unittest.TestCase):
    def test_parse_number_list_forms(self):
        self.assertEqual(obs.parse_number_list(None, "x"), None)
        self.assertEqual(obs.parse_number_list(2.5, "x"), [2.5])
        self.assertEqual(obs.parse_number_list("1, 2 3", "x"), [1.0, 2.0, 3.0])
        self.assertEqual(obs.parse_number_list([1, 2], "x"), [1.0, 2.0])
        self.assertEqual(obs.parse_number_list("  ", "x"), None)

    def test_parse_number_list_rejects_bad_input(self):
        with self.assertRaises(ValueError) as ctx:
            obs.parse_number_list("1 abc", "indices")
        self.assertIn("indices must contain finite numbers", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            obs.parse_number_list({"a": 1}, "indices")
        self.assertIn(
            "indices must be a number, list string, or numeric array", str(ctx.exception)
        )

    def test_configured_dim_fallback_and_errors(self):
        self.assertEqual(obs.configured_dim({"dim": 3.7}, 5), 3)
        self.assertEqual(obs.configured_dim({}, 4), 4)
        with self.assertRaises(ValueError) as ctx:
            obs.configured_dim({}, 0)
        self.assertIn("observation dimension could not be inferred", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            obs.configured_dim({"dim": -1}, 5)
        self.assertIn("observation dimension must be positive, got -1", str(ctx.exception))

    def test_vector_from_pads_and_truncates(self):
        np.testing.assert_allclose(obs.vector_from([1, 2, 3, 4], 3, "x"), [1, 2, 3])
        np.testing.assert_allclose(obs.vector_from([1], 3, "x", 9.0), [1, 9, 9])
        np.testing.assert_allclose(obs.vector_from(None, 2, "x"), [0, 0])

    def test_constant_vector_requires_matching_dim(self):
        np.testing.assert_allclose(obs.constant_vector({"values": [1, 2, 3]}), [1, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            obs.constant_vector({})
        self.assertIn("obs-constant requires a finite values array", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            obs.constant_vector({"values": [1, 2], "dim": 3})
        self.assertIn("obs-constant values length 2 does not match dim 3", str(ctx.exception))

    def test_transform_chain_indices_default_scale(self):
        # (values[idx] - default[idx]) * scale: with an index map the default
        # and scale vectors are indexed by SOURCE index, like the TS helper.
        output = obs.transform_vector(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            {"indices": [2, 0], "defaultAngles": [1, 1, 2], "scale": [9, 9, 2, 0.5]},
        )
        np.testing.assert_allclose(output, [(3 - 2) * 2, (1 - 1) * 9])

    def test_transform_scalar_scale_broadcasts(self):
        output = obs.transform_vector(
            np.array([1.0, 3.0], dtype=np.float32), {"scale": 0.5, "defaultAngles": [1, 1]}
        )
        np.testing.assert_allclose(output, [0.0, 1.0])

    def test_transform_source_index_out_of_bounds(self):
        with self.assertRaises(ValueError) as ctx:
            obs.transform_vector(
                np.array([1.0, 2.0], dtype=np.float32), {"indices": [0, 5]}
            )
        self.assertIn("source index 5 exceeds observation length 2", str(ctx.exception))


class ObservationValuesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def value(self, node_type, node_data=None, **extra):
        return obs.get_mujoco_observation(
            node_type, context(self.model, self.data, node_data, **extra)
        )

    def test_obs_qpos_full_and_offset_slice(self):
        np.testing.assert_allclose(
            self.value("obs-qpos"), [0, 0, 1, 1, 0, 0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            self.value("obs-qpos", {"indices": [2, 6]}), [1, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            obs.get_mujoco_observation(
                "obs-qpos",
                context(self.model, self.data, {"dim": 1}, qposOffset=7),
            ),
            [0],
            atol=1e-6,
        )

    def test_obs_qvel(self):
        output = self.value("obs-qvel", {"dim": 7})
        self.assertEqual(output.shape, (7,))
        np.testing.assert_allclose(output, np.zeros(7), atol=1e-6)

    def test_obs_dof_pos_and_vel_skip_root(self):
        np.testing.assert_allclose(self.value("obs-dof-pos", {"dim": 1}), [0], atol=1e-6)
        np.testing.assert_allclose(self.value("obs-dof-vel", {"dim": 1}), [0], atol=1e-6)

    def test_obs_base_height(self):
        np.testing.assert_allclose(self.value("obs-base-height"), [1.0], atol=1e-6)

    def test_obs_projected_gravity_upright(self):
        np.testing.assert_allclose(
            self.value("obs-projected-gravity"), [0, 0, -1], atol=1e-6
        )

    def test_obs_imu_kinematic(self):
        np.testing.assert_allclose(
            self.value("obs-imu"), [1, 0, 0, 0, 0, 0, 0], atol=1e-6
        )

    def test_obs_constant(self):
        np.testing.assert_allclose(
            self.value("obs-constant", {"values": [1, 2, 3]}), [1, 2, 3]
        )

    def test_obs_actions_from_last_action(self):
        output = obs.get_mujoco_observation(
            "obs-actions",
            context(
                self.model,
                self.data,
                None,
                lastAction=np.array([0.5, -0.25], dtype=np.float32),
            ),
        )
        np.testing.assert_allclose(output, [0.5, -0.25], atol=1e-7)

    def test_obs_actions_falls_back_to_ctrl_dim(self):
        output = self.value("obs-actions")
        self.assertEqual(output.shape, (1,))

    def test_transform_chain_applies_to_qpos(self):
        # qpos[3] is the quaternion w (= 1 at the initial pose); defaults are
        # source-indexed, so index 3 subtracts 1 and indices 5/7 subtract 0.
        defaults = [0.0] * 8
        defaults[3] = 1.0
        output = self.value(
            "obs-qpos",
            {"indices": [7, 5, 3], "scale": 2, "defaultAngles": defaults},
        )
        np.testing.assert_allclose(output, [0.0, 0.0, 0.0], atol=1e-6)


class RotatedStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model(
            quat=[HALF_SQRT2, 0, 0, HALF_SQRT2], qvel=[0.2, 0, 0, 0.1, 0, 0, 0]
        )

    def value(self, node_type, node_data=None, **extra):
        return obs.get_mujoco_observation(
            node_type, context(self.model, self.data, node_data, **extra)
        )

    def test_obs_base_ang_vel_inverse_rotates_root_qvel(self):
        # R^T @ [0.1, 0, 0] for a 90 deg z quaternion = [0, -0.1, 0].
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel"), [0, -0.1, 0], atol=1e-6
        )

    def test_obs_base_ang_vel_explicit_body_frame_matches_real_gyro(self):
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel", {"rootAngularVelocityFrame": "body"}),
            self.data.sensordata[4:7],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel", {"rootAngularVelocityFrame": "body", "scale": 0.25}),
            [0.025, 0, 0],
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "rootAngularVelocityFrame must be body"):
            self.value("obs-base-ang-vel", {"rootAngularVelocityFrame": "unknown"})

    def test_obs_base_lin_vel_inverse_rotates_root_qvel(self):
        np.testing.assert_allclose(
            self.value("obs-base-lin-vel"), [0, -0.2, 0], atol=1e-6
        )

    def test_obs_projected_gravity_tilted_90deg_about_x(self):
        model, data = make_forward_model(quat=[HALF_SQRT2, HALF_SQRT2, 0, 0])
        np.testing.assert_allclose(
            obs.get_mujoco_observation(
                "obs-projected-gravity", context(model, data)
            ),
            [0, -1, 0],
            atol=1e-6,
        )

    def test_obs_projected_gravity_body_target(self):
        np.testing.assert_allclose(
            self.value(
                "obs-projected-gravity",
                {"targetKind": "body", "target": "block"},
            ),
            [0, 0, -1],
            atol=1e-6,
        )

    def test_obs_imu_g1_reads_sensors(self):
        # With a g1 profile the quaternion comes from the imu_quat sensor and
        # the angular velocity from the imu_gyro sensor (body frame), which
        # differs from the kinematic inverse-rotation path.
        output = self.value("obs-imu", {"robot": "g1"})
        gyro = self.data.sensordata[4:7]
        np.testing.assert_allclose(output[:4], [HALF_SQRT2, 0, 0, HALF_SQRT2], atol=1e-6)
        np.testing.assert_allclose(output[4:], gyro, atol=1e-6)
        self.assertFalse(np.allclose(output[4:], [0, -0.1, 0]))

    def test_obs_base_ang_vel_g1_sensor_and_fallback(self):
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel", {"robot": "g1"}),
            self.data.sensordata[4:7],
            atol=1e-6,
        )
        # velocitySource 'sensor' with no matching linear-velocity sensor
        # falls through to the kinematic path (no silent zeros).
        np.testing.assert_allclose(
            self.value("obs-base-lin-vel", {"robot": "g1", "velocitySource": "sensor"}),
            [0, -0.2, 0],
            atol=1e-6,
        )

    def test_body_target_name_prefix_resolution(self):
        with_prefix = obs.get_mujoco_observation(
            "obs-projected-gravity",
            context(
                self.model,
                self.data,
                {"targetKind": "body", "target": "block"},
                namePrefix="zz/",
            ),
        )
        np.testing.assert_allclose(with_prefix, [0, 0, -1], atol=1e-6)
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-projected-gravity", {"targetKind": "body", "target": "nope"})
        self.assertIn('body "nope" was not found', str(ctx.exception))

    def test_obs_qvel_offset_with_dim_bounded_slice(self):
        output = obs.get_mujoco_observation(
            "obs-qvel", context(self.model, self.data, {"dim": 2}, qvelOffset=5)
        )
        np.testing.assert_allclose(output, [0, 0], atol=1e-6)


class MujocoQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model(ctrl=[0.7])

    def query(self, node_data, **extra):
        return obs.get_mujoco_observation(
            "obs-mujoco-query", context(self.model, self.data, node_data, **extra)
        )

    def test_raw_queries(self):
        np.testing.assert_allclose(
            self.query({"query": "qpos"}), self.data.qpos, atol=1e-6
        )
        np.testing.assert_allclose(self.query({"query": "ctrl"}), [0.7], atol=1e-6)
        np.testing.assert_allclose(
            self.query({"query": "actuator.force.all"}), self.data.actuator_force, atol=1e-6
        )

    def test_row_queries_by_name_index_and_prefix(self):
        np.testing.assert_allclose(
            self.query({"query": "body.xpos", "target": "block"}), [0.5, 0, 0.1], atol=1e-6
        )
        np.testing.assert_allclose(
            self.query({"query": "body.xpos", "target": "3"}),
            self.query({"query": "body.xpos", "target": "block"}),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.query({"query": "geom.xpos", "target": "block_geom"}, namePrefix="zz/"),
            [0.5, 0, 0.1],
            atol=1e-6,
        )

    def test_sensor_and_joint_queries(self):
        np.testing.assert_allclose(
            self.query({"query": "sensor.data", "target": "imu_gyro"}), [0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            self.query({"query": "joint.qpos", "target": "hinge"}), [0], atol=1e-6
        )
        np.testing.assert_allclose(
            self.query({"query": "joint.qvel", "target": "root"}),
            np.zeros(6),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.query({"query": "actuator.ctrl", "target": "hinge_motor"}), [0.7], atol=1e-6
        )

    def test_user_start_length_slice(self):
        np.testing.assert_allclose(
            self.query({"query": "sensordata", "start": 4, "length": 3}),
            self.data.sensordata[4:7],
            atol=1e-6,
        )

    def test_body_velocity_world_matches_cvel_angular_and_translation(self):
        model, data = make_forward_model(qvel=[0.3, 0, 0, 0, 0, 0, 0])
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        shin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin")
        # Pure translation: body-origin velocities equal the root translation.
        np.testing.assert_allclose(
            obs.read_mujoco_access_query(
                {"model": model, "data": data},
                {"query": "body.linvel.world", "target": "torso"},
            ),
            [0.3, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            obs.read_mujoco_access_query(
                {"model": model, "data": data},
                {"query": "body.linvel.world", "target": "shin"},
            ),
            [0.3, 0, 0],
            atol=1e-6,
        )

        model, data = make_forward_model(qvel=[0, 0, 0, 0.1, 0, 0, 0])
        # The angular rows are the cvel rotation part read directly.
        np.testing.assert_allclose(
            obs.read_mujoco_access_query(
                {"model": model, "data": data},
                {"query": "body.angvel.world", "target": "torso"},
            ),
            data.cvel[torso, 0:3],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            obs.read_mujoco_access_query(
                {"model": model, "data": data},
                {"query": "body.angvel.local", "target": "torso"},
            ),
            data.xmat[torso].reshape(3, 3).T @ data.cvel[torso, 0:3],
            atol=1e-6,
        )
        self.assertEqual(shin, 2)  # layout sanity for the MJCF above

    def test_contact_timing_snapshot_boundary(self):
        # Without the physics-layer snapshot the query fails explicitly (T6
        # owns the aggregation); with a published snapshot the row reads it.
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "body.air_time", "target": "torso"})
        self.assertIn("snapshot field body_current_air_time is unavailable", str(ctx.exception))

        output = obs.read_mujoco_access_query(
            {
                "model": self.model,
                "data": self.data,
                "snapshot": {"body_current_air_time": np.arange(4, dtype=np.float64)},
            },
            {"query": "body.air_time", "target": "torso"},
        )
        np.testing.assert_allclose(output, [1.0], atol=1e-6)

    def test_query_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "body.xposnope"})
        self.assertIn('unsupported MuJoCo query "body.xposnope"', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "body.xpos"})
        self.assertIn("body target is required", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "joint.qpos", "target": "nojoint"})
        self.assertIn('joint "nojoint" was not found', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "body.xpos", "target": "99"})
        self.assertIn("body index 99 is out of bounds (count 5)", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.query({"query": "qpos", "start": 900})
        self.assertIn(
            "MuJoCo query output slice length must be a non-negative integer",
            str(ctx.exception),
        )


class HeightScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def scan(self, spec, model=None, data=None):
        return obs.height_scan(model or self.model, data or self.data, {"nodeId": "scan", **spec})

    def test_flat_floor_reads_zero(self):
        np.testing.assert_allclose(
            self.scan({"body": "torso", "points": [0.3, 0, 0.5, 0.2]}), [0, 0], atol=1e-6
        )

    def test_known_box_hit(self):
        np.testing.assert_allclose(
            self.scan({"body": "torso", "points": [0.5, 0]}), [0.2], atol=1e-6
        )

    def test_miss_beyond_max_distance_reads_full_drop(self):
        np.testing.assert_allclose(
            self.scan({"body": "torso", "points": [0.3, 0], "maxDistance": 0.5}), [1.5], atol=1e-6
        )

    def test_only_anchor_body_geoms_are_excluded(self):
        # The shin capsule hangs directly under the torso and is a different
        # body, so the ray still reports it: hit at z = 0.72 from origin
        # z = 2 gives the terrain height 0.72.
        np.testing.assert_allclose(
            self.scan({"body": "torso", "points": [0, 0]}), [0.72], atol=1e-6
        )

    def test_yaw_frame_rotates_points_with_body(self):
        model, data = make_forward_model(quat=[HALF_SQRT2, 0, 0, HALF_SQRT2])
        yaw = self.scan(
            {"body": "torso", "points": [0.3, 0], "frame": "yaw"}, model, data
        )
        world = self.scan(
            {"body": "torso", "points": [0.3, 0], "frame": "world"}, model, data
        )
        np.testing.assert_allclose(yaw, [0.15], atol=1e-6)  # over the yaw box
        np.testing.assert_allclose(world, [0.0], atol=1e-6)  # over the floor

    def test_geom_groups_filter(self):
        # Group 0 off hides every test geom (all are in group 0), so the ray
        # reads the full documented drop: origin_z (2) - maxDistance (10).
        output = self.scan(
            {"body": "torso", "points": [0.3, 0], "geomGroups": [0, 1, 1, 1, 1, 1]}
        )
        np.testing.assert_allclose(output, [2.0 - 10.0], atol=1e-6)

    def test_height_scan_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "torso", "points": [0.3, 0, 1]})
        self.assertIn('node "scan" needs a non-empty (x, y) point list', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "torso", "points": [0.3, 0], "lift": 0})
        self.assertIn('node "scan" lift must be a positive number', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"points": [0.3, 0]})
        self.assertIn('node "scan" needs the body it scans around', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "leg", "points": [0.3, 0]})
        self.assertIn('node "scan" body "leg" was not found', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "99", "points": [0.3, 0]})
        self.assertIn("node \"scan\" body 99 is outside the model", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "torso", "points": [0.3, 0], "geomGroups": [1, 1]})
        self.assertIn('node "scan" geomGroups must contain 6 flags', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.scan({"body": "torso", "points": [0.3, float("nan")]})
        self.assertIn('node "scan" point 0 is not finite', str(ctx.exception))


class CameraPoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def pose(self, spec, model=None, data=None):
        return obs._camera_pose(model or self.model, data or self.data, spec)

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
        original = obs._create_renderer
        obs._create_renderer = self.fake_renderer
        try:
            return obs.camera_frames(self.model, self.data, spec)
        finally:
            obs._create_renderer = original

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
            probe = obs._create_renderer(cls.model, 4, 4)
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
        original = obs._create_renderer
        obs._create_renderer = self.fake_renderer
        try:
            return obs.camera_frames(self.model, self.data, spec)
        finally:
            obs._create_renderer = original

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

        original = obs._create_renderer
        obs._create_renderer = factory
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
            obs._create_renderer = original
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
            obs._apply_gaussian_blur(blurred, size, sigma)
            np.testing.assert_allclose(
                blurred, _reference_gaussian(frame, size, sigma), atol=1e-7
            )

    def test_gaussian_kernel_size_zero_or_one_is_identity(self):
        frame = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        blurred = frame.copy()
        obs._apply_gaussian_blur(blurred, 1, 1.0)
        np.testing.assert_allclose(blurred, frame, atol=0.0)

    def test_scale_and_clip_happen_before_blur(self):
        model, data = make_forward_model()
        def factory(model_arg, width, height):
            renderer_box = _FakeRenderer(width, height)
            # One column of 15 m clamps to 1.0 after /10; the neighbours are 3 m.
            renderer_box.depth = np.full((3, 3), 3.0, dtype=np.float32)
            renderer_box.depth[:, 1] = 15.0
            return renderer_box

        original = obs._create_renderer
        obs._create_renderer = factory
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
            obs._create_renderer = original
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
                obs._mulberry32_uniforms(state, 32), js_random(state), atol=0.0
            )

    def test_fnv1a_hash(self):
        self.assertEqual(obs._fnv1a_hash(""), 2166136261)
        reference = 2166136261
        for char in "cam-node":
            reference = ((reference ^ ord(char)) * 16777619) & 0xFFFFFFFF
        self.assertEqual(obs._fnv1a_hash("cam-node"), reference)

    def test_noise_is_reproducible_and_frame_decorrelated(self):
        config = self.config(depthNoiseStd=0.05, depthNoiseDistanceStd=0.01)
        frame = np.full((3, 4), 0.5, dtype=np.float32)

        first = frame.copy()
        obs._corrupt_camera_depth(config, "cam", 0, first)
        replay = frame.copy()
        obs._corrupt_camera_depth(config, "cam", 0, replay)
        np.testing.assert_array_equal(first, replay)

        later = frame.copy()
        obs._corrupt_camera_depth(config, "cam", 1, later)
        self.assertFalse(np.array_equal(first, later))
        # Corrupted values stay in [0, 1] and differ from the input.
        self.assertTrue(np.all(later >= 0.0) and np.all(later <= 1.0))
        self.assertTrue(np.any(later != 0.5))

    def test_distance_std_scales_noise_with_depth(self):
        config = self.config(depthNoiseStd=0, depthNoiseDistanceStd=0.5)
        frame = np.zeros((1, 4), dtype=np.float32)
        # Zero scaled depth -> zero std -> untouched (mirrors TS `std <= 0` skip).
        output = frame.copy()
        obs._corrupt_camera_depth(config, "cam", 0, output)
        np.testing.assert_array_equal(output, frame)

    def test_pixel_dropout_writes_miss_depth(self):
        config = self.config(depthDropoutProbability=1.0)
        frame = np.full((3, 4), 0.5, dtype=np.float32)
        obs._corrupt_camera_depth(config, "cam", 7, frame)
        np.testing.assert_allclose(frame, 10.0 / 10.0, atol=1e-7)

    def test_patch_dropout_writes_miss_depth_disc(self):
        config = self.config(depthDropoutPatchCount=2, depthDropoutPatchRadius=1)
        frame = np.full((6, 6), 0.5, dtype=np.float32)
        obs._corrupt_camera_depth(config, "cam", 0, frame)
        # Every patch writes clamped far/depthScale; the untouched majority
        # proves only discs changed, and the corruption stays in range.
        self.assertTrue(np.all(frame >= 0.0) and np.all(frame <= 1.0))
        self.assertTrue(np.any(frame == 1.0))
        self.assertTrue(np.any(frame == 0.5))

    def test_disabled_corruption_is_identity(self):
        config = self.config()
        frame = np.full((3, 4), 0.42, dtype=np.float32)
        output = frame.copy()
        obs._corrupt_camera_depth(config, "cam", 0, output)
        np.testing.assert_array_equal(output, frame)
        self.assertFalse(obs._camera_depth_corruption_enabled(config))

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
        original = obs._create_renderer
        obs._create_renderer = factory
        try:
            obs.reset_camera_observation_state()
            first = obs.camera_frames(model, data, spec)[0]
            obs.reset_camera_observation_state()
            replay = obs.camera_frames(model, data, spec)[0]
            third = obs.camera_frames(model, data, spec)[0]
        finally:
            obs._create_renderer = original
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
        original = obs._create_renderer
        obs._create_renderer = factory
        try:
            obs.reset_camera_observation_state()
            first = obs.camera_frames(model, data, dict(spec, nodeId="cam-reset"))[0]
            processor.reset()
            replay = obs.camera_frames(model, data, dict(spec, nodeId="cam-reset"))[0]
        finally:
            obs._create_renderer = original
        np.testing.assert_array_equal(first, replay)


class FillObservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def runtime_context(self, parameters=None, last_action=None, camera_frames=None):
        return {
            "model": self.model,
            "data": self.data,
            "last_action": last_action,
            "parameters": parameters,
            "height_scan": functools.partial(obs.height_scan, self.model, self.data),
            "camera_frames": camera_frames,
        }

    def test_height_scan_binding_fills_processor(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 2}, [])
        target = binding(
            "scan-node",
            "obs-height-scan",
            {"body": "torso", "points": [0.5, 0, 0.3, 0], "dim": 2},
            processor,
        )
        obs.fill_observation(target, self.runtime_context())
        np.testing.assert_allclose(processor.output_buffers[0], [0.2, 0.0], atol=1e-6)

    def test_height_scan_dim_mismatch_errors(self):
        target = binding(
            "scan-node",
            "obs-height-scan",
            {"body": "torso", "points": [0.5, 0], "dim": 3},
        )
        with self.assertRaises(ValueError) as ctx:
            obs.fill_observation(target, self.runtime_context())
        self.assertIn(
            'Height scan "scan-node" returned 1 samples, expected 3', str(ctx.exception)
        )

    def test_height_scan_missing_dim_errors(self):
        target = binding("scan-node", "obs-height-scan", {"body": "torso", "points": [0.5, 0]})
        with self.assertRaises(ValueError) as ctx:
            obs.fill_observation(target, self.runtime_context())
        self.assertIn("scan-node.dim must be a positive integer", str(ctx.exception))

    def test_camera_binding_receives_frame_list(self):
        frames = [np.zeros(12, dtype=np.float32), np.full(4, 0.5, dtype=np.float32)]

        def fake_camera_frames(spec):
            return frames

        processor = obs.ObservationProcessor()
        processor.init(
            {"nodeType": "obs-camera", "outputMode": "rgbd", "width": 2, "height": 2}, []
        )
        target = binding("cam-node", "obs-camera", {"outputMode": "rgbd"}, processor)
        obs.fill_observation(
            target, self.runtime_context(camera_frames=fake_camera_frames)
        )
        self.assertEqual([len(b) for b in processor.output_buffers], [12, 4])
        np.testing.assert_allclose(processor.output_buffers[1], [0.5] * 4)

    def test_camera_binding_requires_callable(self):
        target = binding("cam-node", "obs-camera", {})
        with self.assertRaises(ValueError) as ctx:
            obs.fill_observation(target, self.runtime_context())
        self.assertIn(
            'Camera node "cam-node" requires a camera_frames callable', str(ctx.exception)
        )

    def test_obs_commands_receives_parameters(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 2}, [])
        data = {"parameterNames": ["forward", "lateral"], "dim": 2}
        target = binding("cmd-node", "obs-commands", data, processor)
        parameters = {"forward": 0.3, "lateral": -0.2}
        obs.fill_observation(
            target,
            self.runtime_context(parameters=lambda name: parameters[name]),
        )
        np.testing.assert_allclose(processor.output_buffers[0], [0.3, -0.2], atol=1e-7)

    def test_obs_commands_requires_parameter_names(self):
        target = binding("cmd-node", "obs-commands", {"dim": 2})
        with self.assertRaises(ValueError) as ctx:
            obs.fill_observation(
                target, self.runtime_context(parameters=lambda name: 0.0)
            )
        self.assertIn(
            "obs-commands parameterNames must be a non-empty string list", str(ctx.exception)
        )

    def test_last_action_flows_to_obs_actions(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 2}, [])
        target = binding("act-node", "obs-actions", {"dim": 2}, processor)
        obs.fill_observation(
            target,
            self.runtime_context(
                last_action=np.array([0.5, -0.25], dtype=np.float32)
            ),
        )
        np.testing.assert_allclose(processor.output_buffers[0], [0.5, -0.25], atol=1e-7)

    def test_qpos_binding_with_offsets(self):
        processor = obs.ObservationProcessor()
        processor.init({"dim": 1}, [])
        target = binding("q-node", "obs-qpos", {"dim": 1}, processor, qposOffset=7)
        obs.fill_observation(target, self.runtime_context())
        np.testing.assert_allclose(processor.output_buffers[0], [0.0], atol=1e-6)


class PoseVelocityCommandTest(unittest.TestCase):
    """poseVelocityCommandVector mirror (mujocoObservationCommands.ts).

    The baseline vector entering the transform is the parameter/project
    value list — the browser's no-interactive-input behavior. Expectations
    are hand-computed against the free-joint MJCF (root x/y from qpos, yaw
    from the identity or 90-degree quaternion).
    """

    def setUp(self):
        obs.clear_pose_velocity_states()

    def tearDown(self):
        obs.clear_pose_velocity_states()

    @staticmethod
    def pose_model(x=0.0, y=0.0, quat=None):
        model, data = make_forward_model(quat=quat)
        data.qpos[0] = x
        data.qpos[1] = y
        mujoco.mj_forward(model, data)
        return model, data

    def command(self, node_data, model, data, node_id="pv-node"):
        return obs.get_mujoco_observation(
            "obs-commands", context(model, data, node_data, nodeId=node_id)
        )

    def config(self, **overrides):
        node_data = {
            "dim": 3,
            "values": [0.4, 0, 0],
            "cmdMin": [0, -1, -1],
            "cmdMax": [0.5, 1, 1],
            "poseVelocityCommand": True,
        }
        node_data.update(overrides)
        return node_data

    def test_no_forward_command_keeps_baseline(self):
        model, data = self.pose_model(x=0.3, y=-0.2)
        output = self.command(self.config(values=[0, 0.25, -0.4]), model, data)
        np.testing.assert_allclose(output, [0, 0.25, -0.4], atol=1e-7)
        # No yaw assist (yawMinForwardVelocity unset) and no forward command:
        # the idle state keeps re-anchoring at the current pose.
        state = obs._pose_velocity_state_by_node["pv-node"]
        self.assertAlmostEqual(state["centerY"], -0.2, places=6)
        self.assertAlmostEqual(state["targetX"], 0.3 + 2.0, places=6)
        self.assertFalse(state["wasForwardActive"])

    def test_yaw_forward_assist_clamps_minimum_velocity(self):
        model, data = self.pose_model()
        config = self.config(values=[0, 0, 0.5], yawMinForwardVelocity=0.1)
        np.testing.assert_allclose(
            self.command(config, model, data), [0.1, 0, 0.5], atol=1e-7
        )
        # The assist velocity still respects the axis upper bound.
        config["cmdMax"] = [0.05, 1, 1]
        np.testing.assert_allclose(
            self.command(config, model, data), [0.05, 0, 0.5], atol=1e-7
        )
        # The assist path returns before any state is created.
        self.assertNotIn("pv-node", obs._pose_velocity_state_by_node)
        # Zero yaw rate or zero minimum disables it (baseline kept).
        np.testing.assert_allclose(
            self.command(
                self.config(values=[0, 0.1, 0], yawMinForwardVelocity=0.1),
                model,
                data,
            ),
            [0, 0.1, 0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            self.command(self.config(values=[0, 0, 0.5]), model, data),
            [0, 0, 0.5],
            atol=1e-7,
        )

    def test_forward_tracking_reuses_state_across_steps(self):
        model, data = self.pose_model()
        config = self.config()
        # Step 1: anchor at (0, 0); the target 2 m ahead with stiffness 2
        # saturates at the commanded speed cap min(cmdMax, commandedVx).
        np.testing.assert_allclose(
            self.command(config, model, data), [0.4, 0, 0], atol=1e-6
        )
        # Step 2: robot advanced to (0.5, 0.3); centerY stays anchored at 0
        # so lateral pulls back toward the walked line while the target keeps
        # its 2 m lead (targetX 2.5, dx 2, dy -0.3).
        data.qpos[0] = 0.5
        data.qpos[1] = 0.3
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            self.command(config, model, data),
            [0.4, -0.6, 2 * math.atan2(-0.3, 2.0)],  # heading -2*atan(0.15)
            atol=1e-6,
        )

    def test_forward_tracking_rotates_target_into_body_frame(self):
        # 90-degree yaw: the world +x target sits due right of the robot, so
        # bodyX ~ 0 and the lateral/heading axes saturate rightward.
        model, data = self.pose_model(quat=[HALF_SQRT2, 0, 0, HALF_SQRT2])
        np.testing.assert_allclose(
            self.command(self.config(), model, data), [0, -1, -1], atol=1e-6
        )

    def test_state_rebuilds_after_forward_release(self):
        model, data = self.pose_model()
        forward = self.config()
        idle = self.config(values=[0, 0, 0])
        np.testing.assert_allclose(
            self.command(forward, model, data), [0.4, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            self.command(idle, model, data), [0, 0, 0], atol=1e-7
        )
        # wasForwardActive flipped to False, so the next forward step
        # re-anchors centerY at the current y (0.8): lateral is exactly 0
        # instead of steering back toward the old line.
        data.qpos[0] = 1.0
        data.qpos[1] = 0.8
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            self.command(forward, model, data), [0.4, 0, 0], atol=1e-6
        )

    def test_clear_pose_velocity_states_re_anchors(self):
        model, data = self.pose_model()
        config = self.config(values=[0.5, 0, 0])
        np.testing.assert_allclose(
            self.command(config, model, data, node_id="pv-clear"),
            [0.5, 0, 0],
            atol=1e-6,
        )
        data.qpos[1] = 1.0
        mujoco.mj_forward(model, data)
        # Old anchor: dy = 0 - 1 saturates the lateral axis leftward.
        self.assertLess(
            self.command(config, model, data, node_id="pv-clear")[1], 0
        )
        obs.clear_pose_velocity_states("pv-clear")
        # Fresh anchor at y = 1: back on the line it now walks.
        self.assertEqual(
            float(self.command(config, model, data, node_id="pv-clear")[1]), 0.0
        )
        # Clearing an unknown node id is a no-op, clearing all empties the map.
        obs.clear_pose_velocity_states("pv-unknown")
        self.assertIn("pv-clear", obs._pose_velocity_state_by_node)
        obs.clear_pose_velocity_states()
        self.assertEqual(obs._pose_velocity_state_by_node, {})

    def test_target_reached_zeroes_velocity_and_keeps_yaw(self):
        # The running target keeps its lookahead lead, so the threshold
        # branch only triggers when targetDisThreshold > targetLookahead.
        model, data = self.pose_model()
        config = self.config(values=[0.4, 0, 0.1], targetLookahead=0.2)
        np.testing.assert_allclose(
            self.command(config, model, data), [0, 0, 0.1], atol=1e-7
        )

    def test_remapped_command_indices_and_dim_bounds(self):
        model, data = self.pose_model()
        # dim-length cmdMin/cmdMax are gathered through commandIndices, like
        # commandAxisBounds in keyboardCommands.ts.
        axes = obs.command_axis_config(
            {
                "commandIndices": [2, 3, 4],
                "cmdMin": [-1, -1, -1, -1, -2],
                "cmdMax": [1, 1, 1, 1, 2],
            },
            5,
        )
        self.assertEqual(axes["indices"], [2, 3, 4])
        self.assertEqual(axes["min"], [-1.0, -1.0, -2.0])
        self.assertEqual(axes["max"], [1.0, 1.0, 2.0])

        node_data = {
            "dim": 5,
            "values": [0, 0, 0.4, 0.2, -0.3],
            "commandIndices": [2, 3, 4],
            "cmdMin": [-1, -1, -1, -1, -2],
            "cmdMax": [1, 1, 1, 1, 2],
            "poseVelocityCommand": True,
        }
        output = self.command(node_data, model, data, node_id="pv-remap")
        np.testing.assert_allclose(output, [0, 0, 0.4, 0, -0.3], atol=1e-6)

    def test_fill_observation_applies_pose_velocity_after_parameters(self):
        model, data = self.pose_model()
        processor = obs.ObservationProcessor()
        processor.init({"dim": 3}, [])
        target = binding(
            "pv-fill-node",
            "obs-commands",
            {
                "parameterNames": ["forward", "lateral", "yaw"],
                "dim": 3,
                "poseVelocityCommand": True,
                "cmdMin": [0, -1, -1],
                "cmdMax": [0.5, 1, 1],
            },
            processor,
        )
        parameters = {"forward": 0.4, "lateral": 0.0, "yaw": 0.0}
        obs.fill_observation(
            target,
            {
                "model": model,
                "data": data,
                "last_action": None,
                "parameters": lambda name: parameters[name],
            },
        )
        np.testing.assert_allclose(
            processor.output_buffers[0], [0.4, 0, 0], atol=1e-6
        )

    def test_dim_node_id_and_context_required(self):
        model, data = self.pose_model()
        with self.assertRaises(ValueError) as ctx:
            self.command(self.config(dim=2), model, data)
        self.assertIn(
            "poseVelocityCommand requires command dim >= 3", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            obs.get_mujoco_observation(
                "obs-commands", context(model, data, self.config())
            )
        self.assertIn(
            "poseVelocityCommand obs-commands requires nodeId", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            obs._command_vector(self.config(), "pv-node", None)
        self.assertIn(
            "poseVelocityCommand requires MuJoCo model/data context",
            str(ctx.exception),
        )

    def test_scalar_validation_errors(self):
        model, data = self.pose_model()
        for field, value, message in (
            ("targetLookahead", -1, "targetLookahead must be > 0"),
            ("targetLookahead", "abc", "targetLookahead must be finite"),
            ("velocityControlStiffness", 0, "velocityControlStiffness must be > 0"),
            ("targetDisThreshold", -0.1, "targetDisThreshold must be >= 0"),
            ("yawMinForwardVelocity", -1, "yawMinForwardVelocity must be >= 0"),
        ):
            with self.assertRaises(ValueError) as ctx:
                self.command(
                    self.config(**{field: value}), model, data, node_id="pv-err"
                )
            self.assertIn(message, str(ctx.exception))

    def test_axis_config_defaults_and_errors(self):
        axes = obs.command_axis_config({}, 4)
        self.assertEqual(axes["indices"], [0, 1, 2])
        self.assertEqual(axes["min"], [-1.0, -1.0, -1.0])
        self.assertEqual(axes["max"], [1.0, 1.0, 1.0])

        for config, dim, message in (
            ({"cmdMin": [0, 0]}, 5, "cmdMin must contain 3 or 5 numbers"),
            (
                {"commandIndices": [0, 1]},
                3,
                "commandIndices must contain exactly 3 numbers",
            ),
            (
                {"commandIndices": [0, 1, 1.5]},
                3,
                "commandIndices must contain integers",
            ),
            ({"commandIndices": [0, 1, 1]}, 3, "duplicate command index 1"),
            (
                {"commandIndices": [0, 1, 5]},
                3,
                "command index 5 is outside dim 3",
            ),
            (
                {"cmdMin": [1, -1, -1], "cmdMax": [0.5, 1, 1]},
                3,
                "cmdMin[0] must be <= cmdMax[0]",
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                obs.command_axis_config(config, dim)
            self.assertIn(message, str(ctx.exception))


class FixedBaseModelTest(unittest.TestCase):
    """Cover the non-free-joint fallback paths (xpos/xquat/cvel reads)."""

    XML = """
    <mujoco model="t4fixed">
      <worldbody>
        <body name="pillar" pos="0 0 0.5">
          <joint name="pitch" type="hinge" axis="0 1 0"/>
          <geom name="pillar_geom" type="box" size="0.05 0.05 0.5"/>
        </body>
      </worldbody>
    </mujoco>
    """

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(cls.XML)
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def value(self, node_type, node_data=None):
        return obs.get_mujoco_observation(
            node_type, context(self.model, self.data, node_data)
        )

    def test_base_height_and_gravity_from_xpos_xquat(self):
        np.testing.assert_allclose(self.value("obs-base-height"), [0.5], atol=1e-6)
        np.testing.assert_allclose(
            self.value("obs-projected-gravity"), [0, 0, -1], atol=1e-6
        )

    def test_base_velocities_from_cvel(self):
        np.testing.assert_allclose(
            self.value("obs-base-lin-vel"), [0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel"), [0, 0, 0], atol=1e-6
        )

    def test_imu_uses_xquat_plus_cvel_angular(self):
        np.testing.assert_allclose(
            self.value("obs-imu"), [1, 0, 0, 0, 0, 0, 0], atol=1e-6
        )

    def test_height_scan_supports_ndarray_points(self):
        heights = obs.height_scan(
            self.model,
            self.data,
            {"nodeId": "s", "body": "pillar", "points": np.array([[0.5, 0.0], [1.0, 0.0]])},
        )
        # No floor in this model: both rays read the full drop (1.5 - 10).
        np.testing.assert_allclose(heights, [1.5 - 10.0, 1.5 - 10.0], atol=1e-6)


class BadInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = make_forward_model()

    def value(self, node_type, node_data=None, **extra):
        return obs.get_mujoco_observation(
            node_type, context(self.model, self.data, node_data, **extra)
        )

    def test_unsupported_observation_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-nope")
        self.assertIn('unsupported observation type "obs-nope"', str(ctx.exception))

    def test_camera_type_rejected_by_extractor(self):
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-camera")
        self.assertIn(
            "obs-camera requires renderer-backed camera observation", str(ctx.exception)
        )

    def test_qpos_offset_beyond_state(self):
        with self.assertRaises(ValueError) as ctx:
            obs.get_mujoco_observation(
                "obs-qpos", context(self.model, self.data, {"dim": 4}, qposOffset=7)
            )
        self.assertIn("data.qpos slice 7:11 exceeds length 8", str(ctx.exception))

    def test_negative_indices_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-qpos", {"indices": [1, -2]})
        self.assertIn("indices must be non-negative integers", str(ctx.exception))

    def test_non_positive_dim_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-qpos", {"dim": 0})
        self.assertIn("observation dimension must be positive, got 0", str(ctx.exception))

    def test_headless_interactive_modes_use_explicit_neutral_input_state(self):
        for mode in ("keyboard", "gamepad", "keyboard-gamepad"):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(
                    self.value(
                        "obs-commands",
                        {"dim": 3, "commandMode": mode, "values": [0.2, -0.1, 0.3]},
                    ),
                    [0.2, -0.1, 0.3],
                    atol=1e-7,
                )
        for mode in (
            "keyboard-buttons",
            "gamepad-buttons",
            "keyboard-gamepad-buttons",
        ):
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(
                    self.value("obs-commands", {"commandMode": mode}), [0.0]
                )
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-commands", {"dim": 2, "poseVelocityCommand": True})
        self.assertIn(
            "poseVelocityCommand requires command dim >= 3",
            str(ctx.exception),
        )

    def test_mapped_encoding_key_gates_are_inactive_in_headless_input_state(self):
        node_data = {
            "dim": 2,
            "commandEncoding": "one-hot",
            "values": [0.25, -0.5],
            "commandMappings": [
                {"keys": ["KeyW"], "index": 0},
                {"anyKey": "KeyX, KeyY", "index": 1},
            ],
        }
        np.testing.assert_allclose(
            self.value("obs-commands", node_data), [0.25, -0.5], atol=1e-7
        )

    def test_mapped_encoding_still_validates_key_gate_shape(self):
        with self.assertRaises(ValueError) as ctx:
            self.value("obs-commands", {
                "dim": 2,
                "commandEncoding": "mapped",
                "commandMappings": [{"keys": 42, "index": 0}],
            })
        self.assertIn("commandMappings[0].keys must be a string or string array", str(ctx.exception))

    def test_mapped_encoding_with_values(self):
        node_data = {
            "dim": 2,
            "commandEncoding": "one-hot",
            "commandMappings": [
                {"rootXRange": [10, 20], "index": 0},
                {"values": [0.4, -0.1]},
            ],
        }
        np.testing.assert_allclose(
            self.value("obs-commands", node_data), [0.4, -0.1], atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
