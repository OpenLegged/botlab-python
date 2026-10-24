"""Native physics adapter tests (tier B: needs mujoco + numpy)."""

import os
import tempfile
import unittest
from unittest import mock

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from botlab._runtime import (
    _physics,  # noqa: E402
    _physics_ctrl,  # noqa: E402
)

HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""

FLOAT_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="bot" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom name="body_geom" type="sphere" size="0.05" mass="0.5"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_motor" joint="knee" gear="1"/>
  </actuator>
</mujoco>
"""

HEIGHTFIELD_XML = """
<mujoco>
  <asset>
    <hfield name="terrain_field_heightfield" nrow="2" ncol="2"
            size="1 1 1 0.1" elevation="0 0 0 0"/>
  </asset>
  <worldbody>
    <geom type="hfield" hfield="terrain_field_heightfield"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="hinge_motor" joint="hinge"/></actuator>
</mujoco>
"""

DUAL_HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="first" pos="0 0 0.5">
      <joint name="first_joint" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="second" pos="0 0 0.2">
        <joint name="second_joint" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.05" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="first_motor" joint="first_joint"/>
    <motor name="second_motor" joint="second_joint"/>
  </actuator>
</mujoco>
"""


def robot_resource(**config):
    return {
        "id": "robot",
        "kind": "robot",
        "source": "robot.xml",
        "config": {
            "actionDimension": 1,
            "jointQposOffset": 0,
            "jointQvelOffset": 0,
            **config,
        },
    }


class CtrlOffsetTest(unittest.TestCase):
    def test_floating_base_prefix_and_plain_match(self):
        self.assertEqual(_physics_ctrl.ctrl_offset(8, 1, 7, "qpos"), 7)
        self.assertEqual(_physics_ctrl.ctrl_offset(7, 1, 6, "qvel"), 6)
        self.assertEqual(_physics_ctrl.ctrl_offset(1, 1, 7, "qpos"), 0)

    def test_incompatible_lengths_are_explicit(self):
        with self.assertRaises(ValueError) as ctx:
            _physics_ctrl.ctrl_offset(9, 1, 7, "qpos")
        self.assertIn("incompatible", str(ctx.exception))


def make_pd(target, kps, kds, effort_min, effort_max, **indices):
    pd = {
        "target": target, "kps": kps, "kds": kds,
        "effort_min": effort_min, "effort_max": effort_max,
    }
    pd.update(indices)
    return pd


class SubstepPdMathTest(unittest.TestCase):
    def test_legacy_offset_math_clamps_and_writes_ctrl(self):
        # float-base robot: qpos=8 (7 + 1 hinge), qvel=7 (6 + 1), ctrl=1
        qpos = np.array([0, 0, 1, 1, 0, 0, 0, 0.3])
        qvel = np.zeros(7)
        qvel[6] = 2.0
        ctrl = np.zeros(1)
        prepared = _physics_ctrl.PreparedSubstepPd(
            make_pd([0.5], [40.0], [4.0], [-10.0], [10.0]), qpos, qvel, ctrl
        )
        prepared.apply(qpos, qvel, ctrl)
        # tau = (0.5 - 0.3) * 40 - 2.0 * 4 = 8 - 8 = 0
        np.testing.assert_allclose(ctrl, [0.0], rtol=0, atol=1e-12)

        prepared.target[0] = 1.0
        prepared.apply(qpos, qvel, ctrl)
        # tau = (1.0 - 0.3) * 40 - 8 = 20 -> clamped to 10
        np.testing.assert_allclose(ctrl, [10.0], rtol=0, atol=1e-12)

    def test_absolute_indices_skip_offsets(self):
        qpos = np.array([0, 0, 1, 1, 0, 0, 0, 0.3])
        qvel = np.zeros(7)
        ctrl = np.zeros(1)
        prepared = _physics_ctrl.PreparedSubstepPd(
            make_pd(
                [0.5], [40.0], [4.0], [-100.0], [100.0],
                qpos_indices=[7], qvel_indices=[6], ctrl_indices=[0],
            ),
            qpos, qvel, ctrl,
        )
        self.assertEqual(prepared.qpos_offset, 0)
        prepared.apply(qpos, qvel, ctrl)
        # offsets are zero with absolute indices: tau = (0.5 - qpos[7]) * 40 - 0
        np.testing.assert_allclose(ctrl, [8.0], rtol=0, atol=1e-12)

    def test_partial_index_sets_are_rejected(self):
        qpos = np.zeros(1)
        qvel = np.zeros(1)
        ctrl = np.zeros(1)
        with self.assertRaises(ValueError) as ctx:
            _physics_ctrl.PreparedSubstepPd(
                make_pd([0.0], [1.0], [1.0], [-1.0], [1.0], qpos_indices=[0]),
                qpos, qvel, ctrl,
            )
        self.assertIn("requires qposIndices, qvelIndices, and ctrlIndices", str(ctx.exception))

    def test_length_mismatch_and_out_of_range_are_explicit(self):
        qpos = np.zeros(1)
        qvel = np.zeros(1)
        ctrl = np.zeros(1)
        with self.assertRaises(ValueError) as ctx:
            _physics_ctrl.PreparedSubstepPd(
                make_pd([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]),
                qpos, qvel, ctrl,
            )
        self.assertIn("legacy substep PD buffers must match ctrl length 1", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _physics_ctrl.PreparedSubstepPd(
                make_pd(
                    [0.0], [1.0], [1.0], [-1.0], [1.0],
                    qpos_indices=[5], qvel_indices=[0], ctrl_indices=[0],
                ),
                qpos, qvel, ctrl,
            )
        self.assertIn("qpos index 5 is outside source length 1", str(ctx.exception))


class MujocoPhysicsTest(unittest.TestCase):
    def test_pd_control_pulls_hinge_toward_target(self):
        physics = _physics.MujocoPhysics.from_xml_string(HINGE_XML, mujoco)
        target = np.array([0.5])
        for _ in range(20):
            physics.step_substeps(
                4, substep_pd=make_pd(
                    target, [80.0], [8.0], [-50.0], [50.0]
                )
            )
        self.assertGreater(float(physics.data.qpos[0]), 0.05)
        self.assertLess(float(physics.data.qpos[0]), 0.5)

    def test_ctrl_path_matches_direct_ctrl_write(self):
        physics = _physics.MujocoPhysics.from_xml_string(HINGE_XML, mujoco)
        physics.step_substeps(10, ctrl=[2.0])
        self.assertEqual(float(physics.data.ctrl[0]), 2.0)

    def test_ctrl_length_mismatch_is_explicit(self):
        physics = _physics.MujocoPhysics.from_xml_string(HINGE_XML, mujoco)
        with self.assertRaises(ValueError) as ctx:
            physics.step_substeps(1, ctrl=[1.0, 2.0])
        self.assertIn("does not match MuJoCo ctrl length", str(ctx.exception))

    def test_reset_restores_initial_state(self):
        initial_qpos = [0, 0, 1.35, 1, 0, 0, 0, 0.2]
        initial_qvel = [0, 0, 0, 0, 0, 0, 0.15]
        physics = _physics.MujocoPhysics.from_xml_string(
            FLOAT_BASE_XML,
            mujoco,
            resources=[robot_resource(
                actionDimension=1, jointQposOffset=7, jointQvelOffset=6
            )],
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
        )
        physics.step_substeps(50)
        self.assertNotEqual(float(physics.data.qpos[2]), 1.35)
        physics.reset()
        np.testing.assert_allclose(physics.data.qpos, initial_qpos, atol=0, rtol=0)
        np.testing.assert_allclose(physics.data.qvel, initial_qvel, atol=0, rtol=0)
        # mj_forward must have refreshed derived state for the restored pose.
        self.assertAlmostEqual(float(physics.data.xpos[1][2]), 1.35, places=6)

    def test_float_base_legacy_pd_steps_without_error(self):
        physics = _physics.MujocoPhysics.from_xml_string(FLOAT_BASE_XML, mujoco)
        physics.step_substeps(
            4, substep_pd=make_pd([0.2], [20.0], [2.0], [-5.0], [5.0])
        )
        self.assertTrue(np.all(np.isfinite(physics.data.qpos)))

    def test_custom_controller_observation_action_and_read_use_real_model(self):
        physics = _physics.MujocoPhysics.from_xml_string(
            HINGE_XML,
            mujoco,
            resources=[robot_resource()],
            initial_qpos=[0.25],
            initial_qvel=[0.0],
        )
        observation = physics.robot_observation("robot")
        self.assertEqual(observation["joint_position"], [0.25])
        self.assertEqual(observation["joint_velocity"], [0.0])
        self.assertEqual(physics.read_mujoco("robot", "qpos", start=0, length=1), [0.25])

        physics.apply_action("robot", [2.0])
        physics.step_substeps(1)
        self.assertEqual(float(physics.data.ctrl[0]), 2.0)

        before = float(physics.data.qpos[0])
        physics.apply_action("robot", {
            "kind": "position-pd",
            "target": [0.5],
            "kp": [80.0],
            "kd": [8.0],
            "effort_limit": [50.0],
        })
        physics.step_substeps(20)
        self.assertGreater(float(physics.data.qpos[0]), before)

    def test_custom_position_pd_respects_actuator_joint_order(self):
        physics = _physics.MujocoPhysics.from_xml_string(
            DUAL_HINGE_XML,
            mujoco,
            resources=[robot_resource(
                actionDimension=2,
                actuatorJointOrder=[1, 0],
            )],
        )
        physics.apply_action("robot", {
            "kind": "position-pd",
            "target": [1.0, 2.0],
            "kp": [10.0, 100.0],
            "kd": [0.0, 0.0],
            "effort_limit": [1_000.0, 1_000.0],
        })
        physics.step_substeps(1)
        np.testing.assert_allclose(physics.data.ctrl, [200.0, 10.0])

    def test_custom_position_pd_rejects_invalid_actuator_joint_order(self):
        physics = _physics.MujocoPhysics.from_xml_string(
            DUAL_HINGE_XML,
            mujoco,
            resources=[robot_resource(
                actionDimension=2,
                actuatorJointOrder=[0, 0],
            )],
        )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            physics.apply_action("robot", {
                "kind": "position-pd",
                "target": [0.0, 0.0],
                "kp": [1.0, 1.0],
                "kd": [0.0, 0.0],
                "effort_limit": [1.0, 1.0],
            })

    def test_sensor_snapshot_routes_resource_config_through_real_physics(self):
        sensor = {
            "id": "depth",
            "kind": "sensor",
            "source": "robot",
            "config": {"outputMode": "depth", "width": 2, "height": 1},
        }
        physics = _physics.MujocoPhysics.from_xml_string(
            HINGE_XML, mujoco, resources=[robot_resource(), sensor]
        )
        with mock.patch(
            "botlab._runtime._observations.camera_frames",
            return_value=[np.array([0.25, 0.75], dtype=np.float32)],
        ):
            first = physics.sensor_observation("depth")
            second = physics.sensor_observation("depth")
        self.assertEqual(first["depth"]["shape"], [1, 2])
        self.assertEqual(first["depth"]["data"], [0.25, 0.75])
        self.assertEqual(first["depth"]["sequence"], 1)
        self.assertEqual(second["depth"]["sequence"], 2)

    def test_motion_resource_reset_is_one_shot_then_preset_reset_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            motion_path = os.path.join(directory, "motion.npz")
            np.savez(
                motion_path,
                joint_pos=np.array([[0.1], [0.7]], dtype=np.float32),
                joint_vel=np.array([[0.0], [0.2]], dtype=np.float32),
                body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
                body_quat_w=np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=np.float32),
                body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
                fps=np.array(50, dtype=np.float32),
            )
            motion = {
                "id": "walk",
                "kind": "motion",
                "source": "motion.npz",
                "config": {},
            }
            physics = _physics.MujocoPhysics.from_xml_string(
                HINGE_XML,
                mujoco,
                resources=[robot_resource(), motion],
                path_resolver=lambda source: os.path.join(directory, source),
                initial_qpos=[0.25],
                initial_qvel=[0.05],
            )
            reset_time = physics.request_reset(
                "motion", {"motionId": "walk", "frame": 1, "rootBody": 0}
            )
            self.assertAlmostEqual(reset_time, 0.02)
            np.testing.assert_allclose(physics.data.qpos, [0.7], atol=1e-6)
            np.testing.assert_allclose(physics.data.qvel, [0.2], atol=1e-6)
            physics.request_reset("preset")
            np.testing.assert_allclose(physics.data.qpos, [0.25], atol=0, rtol=0)
            np.testing.assert_allclose(physics.data.qvel, [0.05], atol=0, rtol=0)

    def test_dynamic_heightfield_updates_real_mujoco_model(self):
        scene = {
            "id": "terrain field",
            "kind": "scene",
            "source": "procedural",
            "config": {"terrainKind": "heightfield", "heightfieldGrid": [2, 2]},
        }
        physics = _physics.MujocoPhysics.from_xml_string(
            HEIGHTFIELD_XML, mujoco, resources=[robot_resource(), scene]
        )
        physics.update_heightfield("terrain field", [[0.0, 0.25], [0.5, 1.0]])
        np.testing.assert_allclose(
            physics.model.hfield_data[:4], [0.0, 0.25, 0.5, 1.0], atol=0, rtol=0
        )
        with self.assertRaises(ValueError) as ctx:
            physics.update_heightfield("terrain field", [0.0, 0.25, 0.5, 1.1])
        self.assertIn("between 0 and 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
