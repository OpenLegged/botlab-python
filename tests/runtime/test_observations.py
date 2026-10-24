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
import unittest

import mujoco
import numpy as np

from botlab._runtime import _observations as obs

from .obs_fixtures import EXPECTED_OBS_NODE_TYPES, HALF_SQRT2, binding, context, make_forward_model


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
        output = self.value("obs-imu", {"imuSensors": True})
        gyro = self.data.sensordata[4:7]
        np.testing.assert_allclose(output[:4], [HALF_SQRT2, 0, 0, HALF_SQRT2], atol=1e-6)
        np.testing.assert_allclose(output[4:], gyro, atol=1e-6)
        self.assertFalse(np.allclose(output[4:], [0, -0.1, 0]))

    def test_obs_base_ang_vel_g1_sensor_and_fallback(self):
        np.testing.assert_allclose(
            self.value("obs-base-ang-vel", {"imuSensors": True}),
            self.data.sensordata[4:7],
            atol=1e-6,
        )
        # velocitySource 'sensor' with no matching linear-velocity sensor
        # falls through to the kinematic path (no silent zeros).
        np.testing.assert_allclose(
            self.value("obs-base-lin-vel", {"imuSensors": True, "velocitySource": "sensor"}),
            [0, -0.2, 0],
            atol=1e-6,
        )

    def test_imu_sensors_only_when_node_opts_in(self):
        gyro = self.data.sensordata[4:7]
        for node_data in ({"imuSensors": False}, {}, {"robot": "g1"}, {"resolvedRobot": "g1"},
                          {"robotProfileApplied": "g1"}):
            output = self.value("obs-base-ang-vel", node_data)
            self.assertFalse(np.allclose(output, gyro), node_data)
        np.testing.assert_allclose(self.value("obs-base-ang-vel", {"imuSensors": True}), gyro, atol=1e-6)

    def test_removed_alias_keys_are_ignored(self):
        model, data = make_forward_model()
        node_data = {"size": 1, "length": 1, "scales": 10, "defaultPos": [1] * 8,
                     "joint2motorIdx": [2]}
        output = obs.get_mujoco_observation("obs-qpos", context(model, data, node_data))
        np.testing.assert_allclose(output, [0, 0, 1, 1, 0, 0, 0, 0], atol=1e-6)
        with self.assertRaises(ValueError):
            obs.constant_vector({"value": [1]})

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
