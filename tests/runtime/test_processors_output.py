"""Output/action processor tests (numpy mirror of the TS output processors).

Covers the normal path, reset behaviour, and the explicit error shapes of the
processors registered for out-action / proc-action-hook / out-mujoco /
scene-robot / scene-terrain / out-scope / out-logger / proc-pd /
proc-motion-reference / proc-motion-sample.
"""

import unittest

import numpy as np  # noqa: E402

from botlab._runtime._processors import _output  # noqa: E402

TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


def f32(values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def reference_motion_data(frame_count=3, dof_count=29):
    """Legacy 29-DOF reference fixture with hand-checkable frames.

    Joint channels are linear in frame and joint index; the anchor body moves
    to (1, 2, 3) and rotates to the (0.5, 0.5, 0.5, 0.5) wxyz quaternion on
    the last frame while frames 0/1 sit at the identity pose.
    """
    frames = np.arange(frame_count, dtype=np.float32)
    joint_pos = (
        frames[:, None] * 100.0
        + np.arange(dof_count, dtype=np.float32)[None, :]
    )
    joint_vel = (
        frames[:, None] * 10.0 + np.arange(dof_count, dtype=np.float32)[None, :]
    )
    body_pos = np.zeros((frame_count, 1, 3), dtype=np.float32)
    body_pos[-1, 0] = (1.0, 2.0, 3.0)
    body_quat = np.zeros((frame_count, 1, 4), dtype=np.float32)
    body_quat[:, 0] = (1.0, 0.0, 0.0, 0.0)
    body_quat[-1, 0] = (0.5, 0.5, 0.5, 0.5)
    return {
        "frame_count": frame_count,
        "body_count": 1,
        "dof_count": dof_count,
        "fps": 2.0,
        "joint_position": joint_pos.reshape(-1),
        "joint_velocity": joint_vel.reshape(-1),
        "body_position": body_pos.reshape(-1),
        "body_quaternion": body_quat.reshape(-1),
        "body_linear_velocity": np.zeros(frame_count * 3, dtype=np.float32),
        "body_angular_velocity": np.zeros(frame_count * 3, dtype=np.float32),
    }


def sample_motion_data(with_names=False):
    """Two-DOF motion fixture mirroring the TS MotionSampleProcessor tests."""
    data = {
        "frame_count": 3,
        "body_count": 2,
        "dof_count": 2,
        "fps": 2.0,
        "joint_position": f32([0, 10, 20, 30, 40, 50]),
        "joint_velocity": f32([1, 2, 3, 4, 5, 6]),
        "body_position": f32([1, 2, 3, 9, 9, 9, 3, 4, 5, 8, 8, 8, 5, 6, 7, 7, 7, 7]),
        "body_quaternion": f32(
            [1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        ),
        "body_linear_velocity": f32(
            [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10]
        ),
        "body_angular_velocity": f32(
            [10, 20, 30, 40, 50, 60, 30, 40, 50, 60, 70, 80, 50, 60, 70, 80, 90, 100]
        ),
    }
    if with_names:
        data["joint_names"] = ["hip", "knee"]
        data["body_names"] = ["pelvis", "foot"]
    return data


# ---------------------------------------------------------------------------
# ActionProcessor
# ---------------------------------------------------------------------------


class ActionProcessorTest(unittest.TestCase):
    def test_copies_input_to_owned_buffer(self):
        processor = _output.ActionProcessor()
        processor.init({"nodeId": "action", "nodeType": "out-action"}, [3])
        source = f32([1, -2, 3])
        buffer = processor.output_buffers[0]

        processor.compute([source])

        self.assertIsNot(processor.output_buffers[0], source)
        self.assertIs(processor.output_buffers[0], buffer)
        np.testing.assert_allclose(processor.output_buffers[0], [1, -2, 3], **TOLERANCE)
        source[:] = 9
        np.testing.assert_allclose(processor.output_buffers[0], [1, -2, 3], **TOLERANCE)

    def test_reallocates_when_input_length_changes(self):
        processor = _output.ActionProcessor()
        processor.init({}, [3])
        processor.compute([f32([1, 2, 3])])
        processor.compute([f32([7, 8])])
        self.assertEqual(len(processor.output_buffers[0]), 2)
        np.testing.assert_allclose(processor.output_buffers[0], [7, 8], **TOLERANCE)

    def test_compute_without_input_is_a_noop(self):
        processor = _output.ActionProcessor()
        processor.init({}, [2])
        processor.output_buffers[0][:] = 5
        processor.compute([])
        self.assertEqual(len(processor.output_buffers[0]), 2)
        np.testing.assert_allclose(processor.output_buffers[0], [5, 5], **TOLERANCE)

    def test_init_without_dims_creates_empty_buffer(self):
        processor = _output.ActionProcessor()
        processor.init({}, [])
        self.assertEqual(len(processor.output_buffers[0]), 0)


# ---------------------------------------------------------------------------
# ActionHookProcessor
# ---------------------------------------------------------------------------


class ActionHookProcessorTest(unittest.TestCase):
    def test_passthrough_without_hook(self):
        processor = _output.ActionHookProcessor()
        processor.init({}, [3])
        source = f32([1, -2, 3])

        processor.compute([source])

        self.assertIsNot(processor.output_buffers[0], source)
        np.testing.assert_allclose(processor.output_buffers[0], [1, -2, 3], **TOLERANCE)
        source[:] = 4
        np.testing.assert_allclose(processor.output_buffers[0], [1, -2, 3], **TOLERANCE)

    def test_hook_result_is_copied_to_output(self):
        processor = _output.ActionHookProcessor()
        processor.init({"run": lambda action: np.asarray(action) * 2.0}, [2])

        processor.compute([f32([1, 2])])

        np.testing.assert_allclose(processor.output_buffers[0], [2, 4], **TOLERANCE)

    def test_hook_may_be_injected_as_attribute(self):
        processor = _output.ActionHookProcessor()
        processor.init({}, [2])
        processor.run = lambda action: [action[0] + 1, action[1] - 1]

        processor.compute([f32([10, 20])])

        np.testing.assert_allclose(processor.output_buffers[0], [11, 19], **TOLERANCE)

    def test_hook_result_length_must_match_without_fallback(self):
        processor = _output.ActionHookProcessor()
        processor.init({"run": lambda action: [1]}, [2])

        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([3, 4])])

        self.assertIn(
            "run result length 1 does not match input length 2", str(ctx.exception)
        )
        # The invalid return is rejected before any write; no passthrough fallback.
        np.testing.assert_allclose(processor.output_buffers[0], [0, 0], **TOLERANCE)

    def test_hook_result_must_be_finite(self):
        processor = _output.ActionHookProcessor()
        processor.init(
            {"run": lambda action: f32([1, float("nan")])}, [2]
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([0, 0])])
        self.assertIn("run result[1] must be finite", str(ctx.exception))

    def test_hook_must_return_array_like(self):
        processor = _output.ActionHookProcessor()
        processor.init({"run": lambda action: None}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([0, 0])])
        self.assertIn("run must return an array-like value", str(ctx.exception))

        processor.init({"run": lambda action: 42}, [2])
        with self.assertRaises(ValueError):
            processor.compute([f32([0, 0])])

    def test_hook_exception_terminates_the_step(self):
        def boom(_action):
            raise RuntimeError("boom")

        processor = _output.ActionHookProcessor()
        processor.init({"run": boom}, [2])
        with self.assertRaises(RuntimeError):
            processor.compute([f32([1, 2])])
        # The step died inside the hook; the output keeps its previous content.
        np.testing.assert_allclose(processor.output_buffers[0], [0, 0], **TOLERANCE)

    def test_init_rejects_non_callable_run(self):
        processor = _output.ActionHookProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"run": "not-a-function"}, [2])
        self.assertIn("run must be a function when provided", str(ctx.exception))

    def test_input_presence_and_length_are_validated(self):
        processor = _output.ActionHookProcessor()
        processor.init({}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("Missing input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1, 2, 3])])
        self.assertIn("input length changed from 2 to 3", str(ctx.exception))

    def test_init_requires_positive_first_dim(self):
        processor = _output.ActionHookProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({}, [])
        self.assertIn("inputDims[0] is required", str(ctx.exception))


# ---------------------------------------------------------------------------
# ControlMergeProcessor
# ---------------------------------------------------------------------------


class ControlMergeProcessorTest(unittest.TestCase):
    def test_concatenates_inputs_in_slot_order(self):
        processor = _output.ControlMergeProcessor()
        processor.init(
            {"nodeId": "merge", "nodeType": "out-mujoco"},
            [2, 1],
        )
        processor.compute([f32([10, 20]), f32([30])])
        np.testing.assert_allclose(
            processor.output_buffers[0], [10, 20, 30], **TOLERANCE
        )

    def test_skips_unconnected_zero_dim_slots(self):
        processor = _output.ControlMergeProcessor()
        processor.init({"nodeId": "merge", "nodeType": "out-mujoco"}, [2, 0, 1])
        processor.compute([f32([10, 20]), None, f32([30])])
        np.testing.assert_allclose(
            processor.output_buffers[0], [10, 20, 30], **TOLERANCE
        )

    def test_missing_connected_slot_raises(self):
        processor = _output.ControlMergeProcessor()
        processor.init({"nodeId": "merge", "nodeType": "out-mujoco"}, [2, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([10, 20]), None])
        self.assertIn("Missing input at slot 1", str(ctx.exception))

    def test_length_mismatch_raises(self):
        processor = _output.ControlMergeProcessor()
        processor.init({"nodeId": "merge", "nodeType": "out-mujoco"}, [2, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([10, 20]), f32([1, 2])])
        self.assertIn("Slot 1 produced 2 values, expected 1", str(ctx.exception))

    def test_scene_robot_maps_joint_order_into_motor_order(self):
        processor = _output.ControlMergeProcessor()
        processor.init(
            {
                "nodeId": "$robot",
                "nodeType": "scene-robot",
                "joint2motorIdx": [2, 0, 1],
            },
            [2, 1],
        )
        processor.compute([f32([10, 20]), f32([30])])
        np.testing.assert_allclose(
            processor.output_buffers[0], [20, 30, 10], **TOLERANCE
        )

    def test_scene_robot_rejects_non_finite_input(self):
        processor = _output.ControlMergeProcessor()
        processor.init(
            {
                "nodeId": "$robot",
                "nodeType": "scene-robot",
                "joint2motorIdx": [0, 1],
            },
            [2],
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([float("nan"), 1])])
        self.assertIn("$robot (scene-robot) input[0] is not finite", str(ctx.exception))

    def test_invalid_permutation_raises(self):
        processor = _output.ControlMergeProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {
                    "nodeId": "$robot",
                    "nodeType": "scene-robot",
                    "joint2motorIdx": [0, 0, 1],
                },
                [3],
            )
        self.assertIn("must be a zero-based permutation", str(ctx.exception))

    def test_permutation_length_must_match_total_dim(self):
        processor = _output.ControlMergeProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {
                    "nodeId": "$robot",
                    "nodeType": "scene-robot",
                    "joint2motorIdx": [0],
                },
                [2, 1],
            )
        self.assertIn("must contain exactly 3 values", str(ctx.exception))

    def test_control_node_types_ignore_joint2motor_idx(self):
        processor = _output.ControlMergeProcessor()
        processor.init(
            {
                "nodeId": "terrain",
                "nodeType": "scene-terrain",
                "joint2motorIdx": [1, 0],
            },
            [2],
        )
        processor.compute([f32([10, 20])])
        np.testing.assert_allclose(
            processor.output_buffers[0], [10, 20], **TOLERANCE
        )


# ---------------------------------------------------------------------------
# ScopeProcessor
# ---------------------------------------------------------------------------


class ScopeProcessorTest(unittest.TestCase):
    def setUp(self):
        _output.clear_scope_trace("scope")

    def test_init_requires_node_id(self):
        processor = _output.ScopeProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({}, [1])
        self.assertIn("nodeId is required", str(ctx.exception))

    def test_labels_follow_input_dims(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [1, 2])
        self.assertEqual(
            _output.scope_trace_snapshot("scope"),
            {"times": [], "series": [], "labels": [], "sample_count": 0},
        )
        processor.compute([f32([1]), f32([2, 3])])
        self.assertEqual(
            _output.scope_trace_snapshot("scope")["labels"],
            ["IN1", "IN2[0]", "IN2[1]"],
        )

        skipped = _output.ScopeProcessor()
        skipped.init({"nodeId": "scope"}, [0, 1])
        skipped.compute([None, f32([5])])
        self.assertEqual(
            _output.scope_trace_snapshot("scope")["labels"], ["IN2"]
        )

    def test_compute_appends_connected_channels(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [1, 2])
        processor.compute([f32([1]), f32([2, 3])])
        processor.compute([f32([4]), f32([5, 6])])
        snapshot = _output.scope_trace_snapshot("scope")
        self.assertEqual(snapshot["sample_count"], 2)
        self.assertEqual(len(snapshot["times"]), 2)
        self.assertEqual(snapshot["series"][0], [1.0, 4.0])
        self.assertEqual(snapshot["series"][1], [2.0, 5.0])
        self.assertEqual(snapshot["series"][2], [3.0, 6.0])

    def test_compute_without_connected_inputs_is_a_noop(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [1])
        processor.compute([None])
        self.assertEqual(_output.scope_trace_snapshot("scope")["sample_count"], 0)

    def test_non_finite_input_raises(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1, float("inf")])])
        self.assertIn(
            '[scopeTraceStore] Scope node "scope" received a non-finite value '
            "on channel 2",
            str(ctx.exception),
        )

    def test_rolling_window_keeps_last_samples(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [1])
        for index in range(_output.MAX_SCOPE_SAMPLES + 5):
            processor.compute([f32([index])])
        snapshot = _output.scope_trace_snapshot("scope")
        self.assertEqual(len(snapshot["times"]), _output.MAX_SCOPE_SAMPLES)
        self.assertEqual(snapshot["sample_count"], _output.MAX_SCOPE_SAMPLES + 5)
        self.assertEqual(snapshot["series"][0][0], 5.0)
        self.assertEqual(snapshot["series"][0][-1], float(_output.MAX_SCOPE_SAMPLES + 4))

    def test_dimension_change_resets_the_trace(self):
        first = _output.ScopeProcessor()
        first.init({"nodeId": "scope"}, [2])
        first.compute([f32([1, 2])])

        second = _output.ScopeProcessor()
        second.init({"nodeId": "scope"}, [1])
        second.compute([f32([7])])

        snapshot = _output.scope_trace_snapshot("scope")
        self.assertEqual(len(snapshot["series"]), 1)
        self.assertEqual(snapshot["series"][0], [7.0])
        self.assertEqual(snapshot["sample_count"], 1)

    def test_clear_scope_trace_removes_the_buffer(self):
        processor = _output.ScopeProcessor()
        processor.init({"nodeId": "scope"}, [1])
        processor.compute([f32([1])])
        _output.clear_scope_trace("scope")
        self.assertEqual(
            _output.scope_trace_snapshot("scope"),
            {"times": [], "series": [], "labels": [], "sample_count": 0},
        )


# ---------------------------------------------------------------------------
# PDProcessor
# ---------------------------------------------------------------------------


class PDProcessorPositionTest(unittest.TestCase):
    def config(self, **overrides):
        config = {
            "nodeId": "pd",
            "nodeType": "proc-pd",
            "pdType": "position",
            "kps": [10],
            "kds": [0],
            "effortLimit": [100],
        }
        config.update(overrides)
        return config

    def test_blends_targets_from_current_qpos_over_configured_steps(self):
        processor = _output.PDProcessor()
        processor.init(self.config(targetBlendSteps=3), [1, 1, 1])
        target = f32([10])
        qpos = f32([2])
        qvel = f32([0])

        processor.compute([target, qpos, qvel])
        np.testing.assert_allclose(processor.output_buffers[0], [0], **TOLERANCE)
        np.testing.assert_allclose(
            processor.get_substep_pd()["target"], [2], **TOLERANCE
        )

        processor.compute([target, qpos, qvel])
        np.testing.assert_allclose(processor.output_buffers[0], [40], **TOLERANCE)
        np.testing.assert_allclose(
            processor.get_substep_pd()["target"], [6], **TOLERANCE
        )

        processor.compute([target, qpos, qvel])
        np.testing.assert_allclose(processor.output_buffers[0], [80], **TOLERANCE)
        np.testing.assert_allclose(
            processor.get_substep_pd()["target"], [10], **TOLERANCE
        )

    def test_reset_restarts_target_blending(self):
        processor = _output.PDProcessor()
        processor.init(self.config(targetBlendSteps=2), [1, 1, 1])
        inputs = [f32([10]), f32([2]), f32([0])]
        processor.compute(inputs)
        processor.compute(inputs)
        np.testing.assert_allclose(
            processor.get_substep_pd()["target"], [10], **TOLERANCE
        )

        processor.reset()
        processor.compute(inputs)
        np.testing.assert_allclose(
            processor.get_substep_pd()["target"], [2], **TOLERANCE
        )
        np.testing.assert_allclose(processor.output_buffers[0], [0], **TOLERANCE)

    def test_position_pd_matches_the_torque_formula(self):
        processor = _output.PDProcessor()
        processor.init(
            self.config(kps=[60.5, 40.0], kds=[2.0, 1.5], effortLimit=[100, 100]),
            [2, 2, 2],
        )
        processor.compute(
            [f32([1.2, -0.7]), f32([0.3, 0.1]), f32([-0.2, 0.4])]
        )
        expected = [
            (1.2 - 0.3) * 60.5 - (-0.2) * 2.0,
            (-0.7 - 0.1) * 40.0 - 0.4 * 1.5,
        ]
        np.testing.assert_allclose(processor.output_buffers[0], expected, **TOLERANCE)

    def test_position_pd_clips_to_effort_limit(self):
        processor = _output.PDProcessor()
        processor.init(
            self.config(kps=[1000], kds=[0], effortLimit=[50]), [1, 1, 1]
        )
        processor.compute([f32([10]), f32([0]), f32([0])])
        np.testing.assert_allclose(processor.output_buffers[0], [50], **TOLERANCE)

    def test_substep_pd_view_is_live_and_snake_cased(self):
        processor = _output.PDProcessor()
        processor.init(
            self.config(
                kps=[10, 10],
                kds=[1, 1],
                effortLimit=[20, 20],
                qposIndices=[1, 0],
            ),
            [2, 2, 2],
        )
        view = processor.get_substep_pd()
        self.assertEqual(
            sorted(view),
            ["effort_max", "effort_min", "kds", "kps", "qpos_indices", "qvel_indices", "target"],
        )
        np.testing.assert_allclose(view["qpos_indices"], [1, 0])
        np.testing.assert_allclose(view["qvel_indices"], [1, 0])  # defaults to qpos
        self.assertNotIn("ctrl_indices", view)

        processor.compute([f32([5, 6]), f32([0, 0]), f32([0, 0])])
        np.testing.assert_allclose(view["target"], [5, 6], **TOLERANCE)
        np.testing.assert_allclose(view["kps"], [10, 10], **TOLERANCE)
        np.testing.assert_allclose(view["effort_min"], [-20, -20], **TOLERANCE)
        np.testing.assert_allclose(view["effort_max"], [20, 20], **TOLERANCE)

    def test_explicit_effort_bounds(self):
        processor = _output.PDProcessor()
        processor.init(
            self.config(
                kps=[1000],
                kds=[0],
                effortLimitMin=[-5],
                effortLimitMax=[5],
            ),
            [1, 1, 1],
        )
        processor.compute([f32([10]), f32([0]), f32([0])])
        np.testing.assert_allclose(processor.output_buffers[0], [5], **TOLERANCE)

    def test_missing_inputs_at_compute_raises(self):
        processor = _output.PDProcessor()
        processor.init(self.config(), [1, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("Missing required inputs for position PD", str(ctx.exception))


class PDProcessorValidationTest(unittest.TestCase):
    def dims(self):
        return [1, 1, 1]

    def test_unsupported_pd_type_raises(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"pdType": "torque"}, self.dims())
        self.assertIn('unsupported pdType "torque"', str(ctx.exception))

    def test_blank_pd_type_defaults_to_position(self):
        processor = _output.PDProcessor()
        processor.init({"pdType": "", "kps": [1], "kds": [1], "effortLimit": [1]}, self.dims())
        processor.compute([f32([1]), f32([0]), f32([0])])
        self.assertEqual(len(processor.output_buffers[0]), 1)

    def test_position_requires_three_matching_inputs(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"kps": [1], "kds": [1], "effortLimit": [1]}, [1, 0, 1])
        self.assertIn(
            "position type requires target, qpos, and qvel inputs", str(ctx.exception)
        )

        with self.assertRaises(ValueError) as ctx:
            processor.init({"kps": [1, 1], "kds": [1, 1], "effortLimit": [1, 1]}, [2, 2, 3])
        self.assertIn(
            "input dimensions must match; got target=2, qpos=2, qvel=3",
            str(ctx.exception),
        )

    def test_gain_vectors_must_match_dof_count(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"kps": [1, 2], "kds": [1], "effortLimit": [1]}, [1, 1, 1])
        self.assertIn("kps must contain exactly 1 finite numbers", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            processor.init({"kps": 10, "kds": [1], "effortLimit": [1]}, [1, 1, 1])
        self.assertIn("kps must be an array with 1 numbers", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            processor.init({"kps": [1, -1], "kds": [1, 1], "effortLimit": [1, 1]}, [2, 2, 2])
        self.assertIn("kps[1] must be >= 0", str(ctx.exception))

    def test_effort_bounds_must_be_set_together(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {"kps": [1], "kds": [1], "effortLimitMin": [-1]}, [1, 1, 1]
            )
        self.assertIn(
            "effortLimitMin and effortLimitMax must be set together", str(ctx.exception)
        )

    def test_target_blend_steps_must_be_non_negative_integer(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {"kps": [1], "kds": [1], "effortLimit": [1], "targetBlendSteps": -1},
                [1, 1, 1],
            )
        self.assertIn(
            "targetBlendSteps must be a non-negative integer", str(ctx.exception)
        )

    def test_optional_index_lists_must_match_dof_count(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {
                    "kps": [1, 1],
                    "kds": [1, 1],
                    "effortLimit": [1, 1],
                    "qposIndices": [0, -1],
                },
                [2, 2, 2],
            )
        self.assertIn(
            "qposIndices[1] must be a non-negative integer", str(ctx.exception)
        )


class PDProcessorVelocityEffortTest(unittest.TestCase):
    def test_velocity_pd_clips_torque(self):
        processor = _output.PDProcessor()
        processor.init(
            {"pdType": "velocity", "kps": [5, 5], "effortLimit": 10}, [2, 2]
        )
        processor.compute([f32([1, -2]), f32([0.5, -1])])
        np.testing.assert_allclose(processor.output_buffers[0], [2.5, -5], **TOLERANCE)

        processor.compute([f32([5, -5]), f32([0, 0])])
        np.testing.assert_allclose(processor.output_buffers[0], [10, -10], **TOLERANCE)

    def test_velocity_pd_requires_matching_inputs(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"pdType": "velocity", "kps": [1], "effortLimit": [1]}, [1, 0])
        self.assertIn(
            "velocity type requires vel_target and qvel inputs", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {"pdType": "velocity", "kps": [1, 1], "effortLimit": [1, 1]}, [2, 1]
            )
        self.assertIn(
            "vel_target and qvel dimensions must match; got 2 vs 1",
            str(ctx.exception),
        )

    def test_velocity_missing_input_at_compute_raises(self):
        processor = _output.PDProcessor()
        processor.init({"pdType": "velocity", "kps": [1], "effortLimit": [1]}, [1, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1])])
        self.assertIn(
            "Missing vel_target or qvel input for velocity PD", str(ctx.exception)
        )

    def test_effort_pd_clips_target_tau(self):
        processor = _output.PDProcessor()
        processor.init(
            {"pdType": "effort", "effortLimit": [10, 20, 30]}, [3]
        )
        processor.compute([f32([50, -50, 3])])
        np.testing.assert_allclose(
            processor.output_buffers[0], [10, -20, 3], **TOLERANCE
        )

    def test_effort_pd_requires_target_tau(self):
        processor = _output.PDProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"pdType": "effort", "effortLimit": [1]}, [])
        self.assertIn("effort type requires target_tau input", str(ctx.exception))
        processor.init({"pdType": "effort", "effortLimit": [1]}, [1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn(
            "Missing target_tau input for effort control", str(ctx.exception)
        )

    def test_substep_pd_is_none_for_velocity_and_effort(self):
        velocity = _output.PDProcessor()
        velocity.init({"pdType": "velocity", "kps": [1], "effortLimit": [1]}, [1, 1])
        self.assertIsNone(velocity.get_substep_pd())

        effort = _output.PDProcessor()
        effort.init({"pdType": "effort", "effortLimit": [1]}, [1])
        self.assertIsNone(effort.get_substep_pd())


# ---------------------------------------------------------------------------
# MotionReferenceProcessor
# ---------------------------------------------------------------------------


class MotionReferenceProcessorTest(unittest.TestCase):
    def processor(self, input_dims=None, **config_overrides):
        config = {
            "nodeId": "motion",
            "nodeType": "proc-motion-reference",
            "motionPath": "motions/legacy",
            "motionData": reference_motion_data(),
            "startIndex": 0,
            "frameInterval": 1,
        }
        config.update(config_overrides)
        processor = _output.MotionReferenceProcessor()
        processor.init(config, input_dims or [])
        return processor

    def test_writes_legacy_reference_increments_over_two_steps(self):
        processor = self.processor()

        processor.compute([])
        output = processor.output_buffers[0]
        self.assertEqual(len(output), _output.MOTION_REFERENCE_DIM)
        np.testing.assert_allclose(
            output[0:29], 100.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(
            output[29:58], 10.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(output[58:61], [0, 0, 0], **TOLERANCE)
        np.testing.assert_allclose(output[61:67], [1, 0, 0, 1, 0, 0], **TOLERANCE)

        processor.compute([])
        output = processor.output_buffers[0]
        np.testing.assert_allclose(
            output[0:29], 200.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(
            output[29:58], 20.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(output[58:61], [1, 2, 3], **TOLERANCE)
        # Rotation of (w,x,y,z)=(0.5,0.5,0.5,0.5): [[0,0,1],[1,0,0],[0,1,0]].
        np.testing.assert_allclose(output[61:67], [0, 0, 1, 0, 0, 1], **TOLERANCE)

    def test_command_pause_current_freezes_and_zeroes_velocity(self):
        processor = self.processor(
            input_dims=[1, 1],
            commandCodes={"pauseConfigured": 101, "pauseCurrent": 102, "resume": 103},
        )
        processor.compute([f32([0]), f32([0])])

        processor.compute([f32([0]), f32([102])])
        output = processor.output_buffers[0]
        np.testing.assert_allclose(
            output[0:29], 100.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(output[29:58], np.zeros(29), **TOLERANCE)

        processor.compute([f32([0]), f32([103])])
        np.testing.assert_allclose(
            processor.output_buffers[0][29:58], 20.0 + np.arange(29), **TOLERANCE
        )

    def test_pause_configured_holds_the_configured_index(self):
        processor = self.processor(
            input_dims=[1, 1],
            pauseIndex=1,
            commandCodes={"pauseConfigured": 101, "pauseCurrent": 102, "resume": 103},
        )
        processor.compute([f32([0]), f32([101])])
        output = processor.output_buffers[0]
        np.testing.assert_allclose(
            output[0:29], 100.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(output[29:58], np.zeros(29), **TOLERANCE)

    def test_inactive_mode_pauses_at_start_index(self):
        processor = self.processor(activeMode=0)
        processor.compute([f32([1])])
        output = processor.output_buffers[0]
        np.testing.assert_allclose(
            output[0:29], 100.0 + np.arange(29), **TOLERANCE
        )
        np.testing.assert_allclose(output[29:58], np.zeros(29), **TOLERANCE)

        processor.compute([f32([0])])
        np.testing.assert_allclose(
            processor.output_buffers[0][29:58], 20.0 + np.arange(29), **TOLERANCE
        )

    def test_command_codes_validation(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(input_dims=[1, 1])
        self.assertIn(
            "commandCodes is required when the command input is connected",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            self.processor(
                input_dims=[1, 1],
                commandCodes={
                    "pauseConfigured": 10,
                    "pauseCurrent": 10,
                    "resume": 12,
                },
            )
        self.assertIn("commandCodes values must be distinct", str(ctx.exception))

    def test_requires_motion_source_and_injected_data(self):
        processor = _output.MotionReferenceProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"motionData": reference_motion_data()}, [])
        self.assertIn("motionPath or motionSource is required", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            processor.init({"motionPath": "motions/legacy"}, [])
        self.assertIn("motionData is required", str(ctx.exception))

    def test_non_29_dof_motion_fails_the_legacy_contract(self):
        processor = self.processor(motionData=reference_motion_data(dof_count=30))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn(
            "legacy motion reference requires 29 DOFs, got 30", str(ctx.exception)
        )

    def test_mode_input_must_be_scalar(self):
        processor = self.processor()
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([0, 1])])
        self.assertIn("mode input must have dimension 1, got 2", str(ctx.exception))

    def test_frame_interval_and_anchor_validation(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(frameInterval=0)
        self.assertIn("frameInterval must be positive", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.processor(anchorLinkIndex=-1)
        self.assertIn("anchorLinkIndex must be non-negative", str(ctx.exception))


# ---------------------------------------------------------------------------
# MotionSampleProcessor
# ---------------------------------------------------------------------------


class MotionSampleProcessorTest(unittest.TestCase):
    def processor(self, input_dims=None, **config_overrides):
        config = {
            "nodeId": "sampler",
            "nodeType": "proc-motion-sample",
            "motionPath": "motions/two-dof",
            "motionData": sample_motion_data(),
            "channel": "joint_position",
            "indices": [0],
        }
        config.update(config_overrides)
        processor = _output.MotionSampleProcessor()
        processor.init(config, input_dims or [1])
        return processor

    def test_linear_interpolation_of_joint_and_body_channels(self):
        joints = self.processor(channel="joint_position", indices=[1, 0])
        body = self.processor(channel="body_position", indices=[0])

        joints.compute([f32([0.25])])
        body.compute([f32([0.25])])

        np.testing.assert_allclose(joints.output_buffers[0], [20, 10], **TOLERANCE)
        np.testing.assert_allclose(body.output_buffers[0], [2, 3, 4], **TOLERANCE)

    def test_slerp_uses_shortest_path_for_antipodal_quaternions(self):
        processor = self.processor(channel="body_quaternion", indices=[0])
        processor.compute([f32([0.25])])
        value = processor.output_buffers[0]
        self.assertAlmostEqual(abs(float(value[0])), 1.0, places=6)
        self.assertAlmostEqual(
            float(np.linalg.norm(value)), 1.0, places=6
        )
        self.assertAlmostEqual(float(value[1]), 0.0, places=6)

    def test_nearest_interpolation_picks_the_closest_frame(self):
        processor = self.processor(interpolation="nearest")
        processor.compute([f32([0.49])])
        np.testing.assert_allclose(processor.output_buffers[0], [20], **TOLERANCE)

    def test_nearest_quaternion_sampling_normalizes(self):
        processor = self.processor(channel="body_quaternion", indices=[0],
                                   interpolation="nearest")
        processor.compute([f32([0.49])])
        value = processor.output_buffers[0]
        np.testing.assert_allclose(value, [-1, 0, 0, 0], **TOLERANCE)
        self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=6)

    def test_non_loop_clamps_and_loop_wraps_the_frame_period(self):
        clamp = self.processor()
        clamp.compute([f32([100])])
        np.testing.assert_allclose(clamp.output_buffers[0], [40], **TOLERANCE)

        loop = self.processor(loop=True)
        loop.compute([f32([1.25])])
        np.testing.assert_allclose(loop.output_buffers[0], [20], **TOLERANCE)
        loop.compute([f32([1.5])])
        np.testing.assert_allclose(loop.output_buffers[0], [0], **TOLERANCE)

    def test_named_indices_resolve_against_motion_metadata(self):
        processor = self.processor(
            motionData=sample_motion_data(with_names=True),
            indices=["knee", "hip"],
        )
        processor.compute([f32([0])])
        np.testing.assert_allclose(processor.output_buffers[0], [10, 0], **TOLERANCE)

    def test_out_of_range_indices_raise(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(indices=[2])
        self.assertIn(
            "indices are outside the joint_position channel count 2",
            str(ctx.exception),
        )

    def test_named_selection_requires_name_metadata(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(channel="body_position", indices=["pelvis"])
        self.assertIn(
            "named body_position selection requires explicit motion name "
            "metadata",
            str(ctx.exception),
        )

        with self.assertRaises(ValueError) as ctx:
            self.processor(
                motionData=sample_motion_data(with_names=True),
                indices=["tail"],
            )
        self.assertIn(
            'joint_position name "tail" is not present in motion metadata',
            str(ctx.exception),
        )

    def test_time_input_validation(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(input_dims=[2])
        self.assertIn("time input must have dimension 1", str(ctx.exception))

        processor = self.processor()
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([float("nan")])])
        self.assertIn("time input must be finite", str(ctx.exception))

    def test_config_validation(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor(channel="tail_position")
        self.assertIn(
            "channel is required and must be a supported motion channel",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            self.processor(indices=[])
        self.assertIn(
            "indices must be a non-empty array of integers or names",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            self.processor(indices=[0, "hip"])
        self.assertIn(
            "indices must contain only integers or only names", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            self.processor(interpolation="cubic")
        self.assertIn("interpolation must be linear or nearest", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.processor(loop="yes")
        self.assertIn("loop must be a boolean", str(ctx.exception))

    def test_zero_norm_quaternion_is_rejected_at_init(self):
        data = sample_motion_data()
        data["body_quaternion"] = np.zeros(24, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.processor(channel="body_quaternion", indices=[0], motionData=data)
        self.assertIn(
            "body quaternion 0 must have a non-zero finite norm", str(ctx.exception)
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RegistryTest(unittest.TestCase):
    def test_registry_maps_output_node_types(self):
        registry = _output.REGISTRY
        self.assertIs(registry["out-action"], _output.ActionProcessor)
        self.assertIs(registry["proc-action-hook"], _output.ActionHookProcessor)
        self.assertIs(registry["out-mujoco"], _output.ControlMergeProcessor)
        self.assertIs(registry["scene-robot"], _output.ControlMergeProcessor)
        self.assertIs(registry["scene-terrain"], _output.ControlMergeProcessor)
        self.assertIs(registry["out-scope"], _output.ScopeProcessor)
        self.assertIs(registry["out-logger"], _output.ScopeProcessor)
        self.assertIs(registry["proc-pd"], _output.PDProcessor)
        self.assertIs(registry["proc-motion-reference"], _output.MotionReferenceProcessor)
        self.assertIs(registry["proc-motion-sample"], _output.MotionSampleProcessor)


if __name__ == "__main__":
    unittest.main()
