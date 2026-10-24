"""obs-commands poseVelocityCommand tests."""

import math
import unittest

import mujoco
import numpy as np

from botlab._runtime import _observations as obs
from botlab._runtime._observations import _commands as commands

from .obs_fixtures import HALF_SQRT2, binding, context, make_forward_model


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
        state = commands._pose_velocity_state_by_node["pv-node"]
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
        self.assertNotIn("pv-node", commands._pose_velocity_state_by_node)
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
        self.assertIn("pv-clear", commands._pose_velocity_state_by_node)
        obs.clear_pose_velocity_states()
        self.assertEqual(commands._pose_velocity_state_by_node, {})

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
        axes = commands.command_axis_config(
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
            commands._command_vector(self.config(), "pv-node", None)
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
        axes = commands.command_axis_config({}, 4)
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
                commands.command_axis_config(config, dim)
            self.assertIn(message, str(ctx.exception))
