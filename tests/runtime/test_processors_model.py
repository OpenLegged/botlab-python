"""ModelProcessor + OnnxSessionPool tests (tier A fakes + tier B real models).

Mirrors the TypeScript ModelProcessor main path semantics
(src/domain/runtime/engine/processors/ModelProcessor.ts): session reuse by
resource id, CPU/CUDA provider selection with a visible fallback warning, the
inputTensorDimsFromMetadata shape rules, constants, recurrent state rolling,
activeModes gating, control decimation, reset, and explicit error shapes.
Real-model tests (tier B) run the shipped go2w-lab2mj ONNX assets.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from botlab._runtime import _executor  # noqa: E402
from botlab._runtime._processors import _model  # noqa: E402

try:
    import onnxruntime  # noqa: E402
except ImportError:
    onnxruntime = None

FIXTURE_MODELS = os.path.join(os.path.dirname(__file__), "..", "fixtures", "models")
STUDENT_ONNX = os.path.join(FIXTURE_MODELS, "student.onnx")
STUDENT_MEMORY_ONNX = os.path.join(FIXTURE_MODELS, "student_memory.onnx")


def metadata(shape, is_tensor=True, element_type="float32"):
    return {"name": "x", "is_tensor": is_tensor, "type": element_type, "shape": shape}


class TestShapeInference(unittest.TestCase):
    """Pure mirrors of the modelProcessorTensors.ts rules."""

    def test_dynamic_dim_resolves(self):
        label = 'ONNX input "obs" for node n'
        self.assertEqual(
            _model.input_tensor_dims_from_metadata(metadata(["batch", 47]), label, 94),
            [2, 47],
        )
        self.assertEqual(
            _model.input_tensor_dims_from_metadata(metadata(["batch", 47]), label, 47),
            [1, 47],
        )

    def test_fully_static_shape_matches(self):
        dims = _model.input_tensor_dims_from_metadata(metadata([1, 47]), "label", 47)
        self.assertEqual(dims, [1, 47])

    def test_fully_static_shape_mismatch_is_explicit(self):
        with self.assertRaises(ValueError) as ctx:
            _model.input_tensor_dims_from_metadata(metadata([1, 47]), "label", 46)
        self.assertIn("expects 47 values, got 46", str(ctx.exception))

    def test_not_divisible_dynamic_shape_fails(self):
        with self.assertRaises(ValueError) as ctx:
            _model.input_tensor_dims_from_metadata(metadata(["batch", 5]), "label", 12)
        self.assertIn("cannot fit 12 input values", str(ctx.exception))

    def test_ambiguous_dynamic_dimensions_fail(self):
        with self.assertRaises(ValueError) as ctx:
            _model.input_tensor_dims_from_metadata(
                metadata(["a", "b", "c"]), "label", 8
            )
        self.assertIn("ambiguous dynamic dimensions", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _model.input_tensor_dims_from_metadata(
                metadata([3, "a", "b"]), "label", 12
            )
        self.assertIn("ambiguous dynamic dimensions", str(ctx.exception))

    def test_leading_dynamic_pair_resolves(self):
        dims = _model.input_tensor_dims_from_metadata(metadata(["a", "b", 5]), "label", 15)
        self.assertEqual(dims, [1, 3, 5])

    def test_missing_metadata_feeds_row_vector(self):
        self.assertEqual(_model.input_tensor_dims_from_metadata(None, "label", 7), [1, 7])
        self.assertEqual(
            _model.input_tensor_dims_from_metadata(metadata(None), "label", 7), [1, 7]
        )
        self.assertEqual(
            _model.input_tensor_dims_from_metadata(
                metadata([1, 4], is_tensor=False), "label", 4
            ),
            [1, 4],
        )

    def test_state_dims_collapse_symbolic_to_one(self):
        self.assertEqual(
            _model.tensor_dims_from_metadata(metadata([1, "batch_size", 512]), "label"),
            [1, 1, 512],
        )

    def test_state_dims_reject_invalid_metadata(self):
        with self.assertRaises(ValueError) as ctx:
            _model.tensor_dims_from_metadata(metadata(None), "label")
        self.assertIn("shape metadata is required", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _model.tensor_dims_from_metadata(metadata([1, 0, 4]), "label")
        self.assertIn('invalid dimension "0"', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _model.tensor_dims_from_metadata(metadata([1, 4], is_tensor=False), "label")
        self.assertIn("shape metadata is required", str(ctx.exception))

    def test_tensor_element_count(self):
        self.assertEqual(_model.tensor_element_count([2, 3, 4]), 24)
        self.assertEqual(_model.tensor_element_count([]), 1)

    def test_numeric_tensor_data(self):
        array = np.zeros(2, np.float32)
        self.assertIs(_model.numeric_tensor_data(array, "label"), array)
        self.assertIsNotNone(_model.numeric_tensor_data(np.zeros(2, np.int64), "label"))
        self.assertIsNotNone(_model.numeric_tensor_data(np.zeros(2, bool), "label"))
        with self.assertRaises(ValueError) as ctx:
            _model.numeric_tensor_data(np.array(["a", "b"]), "label")
        self.assertIn("is not a numeric tensor", str(ctx.exception))
        with self.assertRaises(ValueError):
            _model.numeric_tensor_data(None, "label")


class TestSessionPool(unittest.TestCase):
    """Pool identity rules with an injected factory — no real models."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path_a = os.path.join(self._directory.name, "a.onnx")
        self.path_b = os.path.join(self._directory.name, "b.onnx")
        with open(self.path_a, "wb") as handle:
            handle.write(b"model-a")
        with open(self.path_b, "wb") as handle:
            handle.write(b"model-b")
        self.factory_calls = []

    def make_pool(self, available_providers=None):
        def factory(model_bytes, providers):
            self.factory_calls.append((model_bytes, list(providers)))
            return _FakeMarkerSession(model_bytes.decode("utf-8"))

        return _model.OnnxSessionPool(
            session_factory=factory, available_providers=available_providers
        )

    def test_same_resource_id_reuses_session(self):
        pool = self.make_pool()
        first = pool.get("res-1", self.path_a)
        second = pool.get("res-1", self.path_a)
        third = pool.get("res-1", self.path_b)  # id is the identity, not the path
        self.assertIs(first, second)
        self.assertIs(first, third)
        self.assertEqual(len(self.factory_calls), 1)

    def test_different_resource_ids_do_not_share(self):
        pool = self.make_pool()
        first = pool.get("res-1", self.path_a)
        second = pool.get("res-2", self.path_a)
        self.assertIsNot(first, second)
        self.assertEqual(len(self.factory_calls), 2)

    def test_factory_receives_file_bytes_and_providers(self):
        pool = self.make_pool(available_providers=["CPUExecutionProvider"])
        pool.get("res-1", self.path_a, "cpu")
        self.assertEqual(self.factory_calls, [(b"model-a", ["CPUExecutionProvider"])])

    def test_public_wasm_provider_maps_to_native_cpu(self):
        pool = self.make_pool(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        pool.get("res-1", self.path_a, "wasm")
        self.assertEqual(self.factory_calls, [(b"model-a", ["CPUExecutionProvider"])])

    def test_public_webgpu_provider_uses_visible_cuda_fallback(self):
        pool = self.make_pool(available_providers=["CPUExecutionProvider"])
        output = io.StringIO()
        with redirect_stdout(output):
            pool.get("res-1", self.path_a, "webgpu")
        self.assertEqual(self.factory_calls, [(b"model-a", ["CPUExecutionProvider"])])
        self.assertIn(
            "CUDAExecutionProvider requested but not available", output.getvalue()
        )

    def test_public_webnn_provider_uses_visible_native_auto_mapping(self):
        pool = self.make_pool(available_providers=["CPUExecutionProvider"])
        output = io.StringIO()
        with redirect_stdout(output):
            pool.get("res-1", self.path_a, "webnn")
        self.assertEqual(self.factory_calls, [(b"model-a", ["CPUExecutionProvider"])])
        self.assertIn("has no native onnxruntime equivalent", output.getvalue())

    def test_cuda_fallback_warning_is_printed(self):
        pool = self.make_pool(available_providers=["CPUExecutionProvider"])
        output = io.StringIO()
        with redirect_stdout(output):
            session = pool.get("res-1", self.path_a, "cuda")
        self.assertEqual(session.marker, "model-a")
        self.assertIn(
            "CUDAExecutionProvider requested but not available", output.getvalue()
        )
        self.assertIn("res-1", output.getvalue())
        self.assertEqual(self.factory_calls[0][1], ["CPUExecutionProvider"])

    def test_missing_model_file_raises_with_resource_id(self):
        pool = self.make_pool()
        missing = os.path.join(self._directory.name, "missing.onnx")
        with self.assertRaises(FileNotFoundError) as ctx:
            pool.get("res-9", missing)
        message = str(ctx.exception)
        self.assertIn("res-9", message)
        self.assertIn(missing, message)

    def test_invalid_requested_provider_is_rejected(self):
        pool = self.make_pool()
        with self.assertRaises(ValueError) as ctx:
            pool.get("res-1", self.path_a, "gpu")
        self.assertIn("executionProvider", str(ctx.exception))


class _FakeMarkerSession:
    """Opaque session stand-in for pool identity tests."""

    def __init__(self, marker):
        self.marker = marker


class _FakeNodeArg:
    def __init__(self, name, shape, arg_type="tensor(float)"):
        self.name = name
        self.type = arg_type
        self.shape = shape


class _FakeSession:
    """Duck-typed onnxruntime session: records feeds, answers via a callable."""

    def __init__(self, inputs, outputs, output_fn=None, results=None):
        self._inputs = inputs
        self._outputs = outputs
        self._output_fn = output_fn
        self._results = results or {}
        self.runs = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.runs.append(
            (
                list(output_names),
                {name: np.array(value, copy=True) for name, value in feeds.items()},
            )
        )
        if self._output_fn is not None:
            return self._output_fn(list(output_names), feeds)
        return [np.array(self._results[name], copy=True) for name in output_names]


class TestModelProcessorFakes(unittest.TestCase):
    """Main-path semantics against duck-typed sessions (no onnxruntime)."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._sessions = {}
        self._counter = 0
        self._factory_calls = []
        self.pool = _model.OnnxSessionPool(
            session_factory=self._factory,
            available_providers=["CPUExecutionProvider"],
        )

    def _factory(self, model_bytes, providers):
        self._factory_calls.append((model_bytes, list(providers)))
        return self._sessions[model_bytes.decode("utf-8")]

    def attach(self, session):
        """Write a marker file the factory maps to ``session``; return its path."""
        self._counter += 1
        marker = f"model-{self._counter}"
        path = os.path.join(self._directory.name, f"{marker}.onnx")
        with open(path, "wb") as handle:
            handle.write(marker.encode("utf-8"))
        self._sessions[marker] = session
        return path

    def init_processor(self, session, input_dims, resource_id=None, **overrides):
        path = self.attach(session)
        config = {
            "nodeId": "model-node",
            "resourceId": resource_id or f"resource-{self._counter}",
            "modelPath": path,
        }
        config.update(overrides)
        processor = _model.ModelProcessor(session_pool=self.pool)
        processor.init(config, input_dims)
        return processor

    def init_policy_processor(self, policies, input_dims, **overrides):
        config = {
            "nodeId": "policy-bank",
            "resourceId": "policy-bank",
            "policies": policies,
        }
        config.update(overrides)
        processor = _model.ModelProcessor(session_pool=self.pool)
        processor.init(config, input_dims)
        return processor

    def test_single_model_compute(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 4])],
            outputs=[_FakeNodeArg("out", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"].ravel()[:2] * 0.5],
        )
        processor = self.init_processor(session, [4])
        self.assertEqual(processor.output_buffers[0].size, 2)  # inferred from metadata

        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        processor.compute([obs])
        np.testing.assert_allclose(processor.output_buffers[0], [0.5, 1.0])
        names, feeds = session.runs[0]
        self.assertEqual(names, ["out"])
        self.assertEqual(feeds["obs"].shape, (1, 4))
        self.assertEqual(feeds["obs"].dtype, np.float32)

        processor.compute([obs.tolist()])  # plain list converts to float32
        processor.compute([obs.astype(np.float64)])  # non-float32 coerces
        self.assertEqual(len(session.runs), 3)

    def test_static_input_shape_mismatch_is_explicit(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 4])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"] * 2.0],
        )
        processor = self.init_processor(session, [4])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.zeros(3, np.float32)])
        self.assertIn("expects 4 values, got 3", str(ctx.exception))

    def test_constants_mapping_fed_every_step(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("scale", [2])],
            outputs=[_FakeNodeArg("y", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"] * feeds["scale"]],
        )
        processor = self.init_processor(session, [2], constants={"scale": [3.0, 4.0]})
        processor.compute([np.array([1.0, 1.0], np.float32)])
        processor.compute([np.array([2.0, 5.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [6.0, 20.0])
        for _, feeds in session.runs:
            np.testing.assert_allclose(feeds["scale"], [3.0, 4.0])
            self.assertEqual(feeds["scale"].shape, (2,))
            self.assertEqual(feeds["scale"].dtype, np.float32)

    def test_extra_inputs_array_form_matches_constants(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("scale", [2])],
            outputs=[_FakeNodeArg("y", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"] * feeds["scale"]],
        )
        processor = self.init_processor(
            session, [2], extraInputs=[{"name": "scale", "values": [3.0, 4.0]}]
        )
        processor.compute([np.array([1.0, 2.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [3.0, 8.0])

    def test_constant_targeting_primary_input_rejected(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("scale", [2])],
            outputs=[_FakeNodeArg("y", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2], constants={"obs": [1.0, 1.0]})
        self.assertIn("cannot target the primary observation input", str(ctx.exception))

    def test_constant_with_unknown_name_rejected(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2])],
            outputs=[_FakeNodeArg("y", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2], constants={"nope": [1.0]})
        self.assertIn('constants[0].name "nope" is not an ONNX input', str(ctx.exception))

    def test_constant_values_length_must_match_dims(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("scale", [2])],
            outputs=[_FakeNodeArg("y", ["batch", 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2], constants={"scale": [1.0]})
        self.assertIn("does not match tensor size 2", str(ctx.exception))

    def test_recurrent_state_rolls_across_steps(self):
        def roll(names, feeds):
            obs = feeds["obs"].ravel()
            state = feeds["state"].ravel()
            return [obs + state, obs + state]

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("state", [2])],
            outputs=[_FakeNodeArg("out", ["batch", 2]), _FakeNodeArg("next_state", [2])],
            output_fn=roll,
        )
        processor = self.init_processor(session, [2])
        processor.compute([np.array([1.0, 2.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0])
        _, first_feeds = session.runs[0]
        np.testing.assert_array_equal(first_feeds["state"], np.zeros(2, np.float32))

        processor.compute([np.array([10.0, 20.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [11.0, 22.0])
        _, second_feeds = session.runs[1]
        np.testing.assert_allclose(second_feeds["state"], [1.0, 2.0])
        self.assertEqual(second_feeds["state"].shape, (2,))

    def test_recurrent_state_length_mismatch_is_explicit(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("state", [2])],
            outputs=[_FakeNodeArg("out", ["batch", 2]), _FakeNodeArg("next_state", [2])],
            output_fn=lambda names, feeds: [feeds["obs"], np.zeros(3, np.float32)],
        )
        processor = self.init_processor(session, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.array([1.0, 2.0], np.float32)])
        self.assertIn(
            'Recurrent state output "next_state" length 3 does not match input '
            '"state" length 2',
            str(ctx.exception),
        )

    def test_mode_gate_filters_inactive_modes(self):
        def make_session():
            return _FakeSession(
                inputs=[_FakeNodeArg("obs", [1, 2])],
                outputs=[_FakeNodeArg("out", [1, 2])],
                output_fn=lambda names, feeds: [feeds["obs"] * 2.0],
            )

        obs = np.array([1.0, 2.0], np.float32)
        gated = self.init_processor(make_session(), [2, 1], activeModes=[1])
        self.assertEqual(len(self._last_session().runs), 0)
        gated.compute([obs, np.array([0.0], np.float32)])  # mode 0 is not active
        self.assertEqual(len(self._last_session().runs), 0)
        gated.compute([obs, np.array([1.0], np.float32)])  # mode 1 is active
        self.assertEqual(len(self._last_session().runs), 1)
        with self.assertRaises(ValueError) as ctx:
            gated.compute([obs])  # mode input is missing
        self.assertIn("mode gate input is missing", str(ctx.exception))

        ungated = self.init_processor(make_session(), [2, 1])  # no activeModes
        ungated.compute([obs, np.array([7.0], np.float32)])
        self.assertEqual(len(self._last_session().runs), 1)

    def _last_session(self):
        return self._sessions[f"model-{self._counter}"]

    def test_mode_gate_input_must_have_dimension_one(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2, 3])
        self.assertIn("mode gate input must have dimension 1", str(ctx.exception))

    def test_active_modes_must_be_integers(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2, 1], activeModes="1 x")
        self.assertIn("activeModes must contain integer mode codes", str(ctx.exception))

    def test_control_decimation_skips_intermediate_steps(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        processor = self.init_processor(session, [2], controlDecimation=3)
        obs = np.zeros(2, np.float32)
        for _ in range(7):
            processor.compute([obs])
        self.assertEqual(len(session.runs), 3)

    def test_reset_clears_state_and_step_count_but_keeps_config(self):
        def roll(names, feeds):
            obs = feeds["obs"].ravel()
            state = feeds["state"].ravel()
            return [obs + state, obs + state]

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2]), _FakeNodeArg("state", [2])],
            outputs=[_FakeNodeArg("out", ["batch", 2]), _FakeNodeArg("next_state", [2])],
            output_fn=roll,
        )
        processor = self.init_processor(session, [2], controlDecimation=2)
        obs = np.array([1.0, 1.0], np.float32)
        processor.compute([obs])  # step 0 runs
        processor.compute([obs])  # step 1 skipped
        processor.compute([obs])  # step 2 runs
        self.assertEqual(len(session.runs), 2)
        np.testing.assert_allclose(processor.output_buffers[0], [2.0, 2.0])
        state_buffer = processor._recurrent_states[0].buffer
        np.testing.assert_allclose(state_buffer, [2.0, 2.0])  # rolled twice

        processor.reset()
        np.testing.assert_array_equal(
            processor.output_buffers[0], np.zeros(2, np.float32)
        )
        np.testing.assert_array_equal(state_buffer, np.zeros(2, np.float32))

        processor.compute([obs])  # step counter restarted, zero state fed again
        self.assertEqual(len(session.runs), 3)
        _, third_feeds = session.runs[2]
        np.testing.assert_array_equal(third_feeds["state"], np.zeros(2, np.float32))
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 1.0])

    def test_non_numeric_output_tensor_is_rejected(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [np.array(["a", "b"])],
        )
        processor = self.init_processor(session, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.zeros(2, np.float32)])
        self.assertIn("is not a numeric tensor", str(ctx.exception))

    def test_output_buffer_adopts_actual_output_size(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2])],
            outputs=[_FakeNodeArg("out", ["batch", 2])],
            output_fn=lambda names, feeds: [np.array([1.0, 2.0, 3.0], np.float32)],
        )
        processor = self.init_processor(session, [2])
        self.assertEqual(processor.output_buffers[0].size, 2)
        processor.compute([np.zeros(2, np.float32)])
        self.assertEqual(processor.output_buffers[0].size, 3)
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0, 3.0])

    def test_configured_input_and_output_names_select_io(self):
        def select_io(names, feeds):
            values = {
                "z": feeds["b"] * 2.0,  # the configured primary output
                "y": np.zeros(2, np.float32),  # the rolled state output
            }
            return [values[name] for name in names]

        session = _FakeSession(
            inputs=[_FakeNodeArg("a", ["batch", 2]), _FakeNodeArg("b", ["batch", 2])],
            outputs=[_FakeNodeArg("y", ["batch", 2]), _FakeNodeArg("z", ["batch", 2])],
            output_fn=select_io,
        )
        processor = self.init_processor(session, [2], inputName="b", outputName="z")
        processor.compute([np.array([1.0, 2.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [2.0, 4.0])
        names, feeds = session.runs[0]
        self.assertEqual(names[0], "z")  # primary output fetched first
        self.assertIn("b", feeds)
        # "a" is rolled as a recurrent input: symbolic 'batch' collapses to 1
        self.assertEqual(feeds["a"].shape, (1, 2))
        np.testing.assert_array_equal(feeds["a"], np.zeros((1, 2), np.float32))

    def test_configured_output_dim_cross_checked_with_static_shape(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("y", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [2], outputDim=3)
        self.assertIn(
            'outputDim 3 does not match ONNX output "y" dimension 2', str(ctx.exception)
        )

    def test_policy_bank_selects_only_the_active_mode(self):
        walk = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"] * 2.0],
        )
        sit = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"] * 3.0],
        )
        walk_path = self.attach(walk)
        sit_path = self.attach(sit)
        processor = self.init_policy_processor(
            [
                {"key": "walk", "mode": 0, "inputSlot": 1,
                 "resourceId": walk_path},
                {"key": "sit", "mode": 1, "inputSlot": 2,
                 "resourceId": sit_path},
            ],
            [1, 2, 2],
        )

        processor.compute([
            np.array([0.0], np.float32),
            np.array([1.0, 2.0], np.float32),
            np.array([10.0, 20.0], np.float32),
        ])
        np.testing.assert_allclose(processor.output_buffers[0], [2.0, 4.0])
        self.assertEqual(len(walk.runs), 1)
        self.assertEqual(len(sit.runs), 0)

        processor.compute([
            np.array([1.0], np.float32),
            np.array([3.0, 4.0], np.float32),
            np.array([10.0, 20.0], np.float32),
        ])
        np.testing.assert_allclose(processor.output_buffers[0], [30.0, 60.0])
        self.assertEqual(len(walk.runs), 1)
        self.assertEqual(len(sit.runs), 1)

        processor.compute([
            np.array([99.0], np.float32),
            np.zeros(2, np.float32),
            np.zeros(2, np.float32),
        ])
        np.testing.assert_array_equal(
            processor.output_buffers[0], np.zeros(2, np.float32)
        )

    def test_policy_bank_reuses_one_session_for_shared_resource(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        path = self.attach(session)
        processor = self.init_policy_processor(
            [
                {"key": "a", "mode": 0, "inputSlot": 1, "resourceId": path},
                {"key": "b", "mode": 1, "inputSlot": 2, "resourceId": path},
            ],
            [1, 2, 2],
        )
        self.assertEqual(len(self._factory_calls), 1)
        processor.compute([
            np.array([0.0], np.float32),
            np.array([1.0, 2.0], np.float32),
            np.array([3.0, 4.0], np.float32),
        ])
        processor.compute([
            np.array([1.0], np.float32),
            np.array([1.0, 2.0], np.float32),
            np.array([3.0, 4.0], np.float32),
        ])
        self.assertEqual(len(session.runs), 2)
        self.assertEqual(len(self._factory_calls), 1)

        processor.dispose()
        self.assertEqual(processor._policy_runtimes, [])
        replacement = self.init_policy_processor(
            [{"key": "replacement", "mode": 0, "inputSlot": 1,
              "resourceId": path}],
            [1, 2],
        )
        replacement.compute([
            np.array([0.0], np.float32), np.array([5.0, 6.0], np.float32)
        ])
        np.testing.assert_allclose(replacement.output_buffers[0], [5.0, 6.0])
        self.assertEqual(len(self._factory_calls), 1)

    def test_policy_pipeline_runs_stages_in_order(self):
        encoder = _FakeSession(
            inputs=[_FakeNodeArg("depth", [1, 2])],
            outputs=[_FakeNodeArg("latent", [1, 3])],
            output_fn=lambda names, feeds: [np.array([
                feeds["depth"].sum(), feeds["depth"].ravel()[0],
                feeds["depth"].ravel()[1],
            ], np.float32)],
        )
        actor = _FakeSession(
            inputs=[_FakeNodeArg("latent", [1, 3]), _FakeNodeArg("bias", [2])],
            outputs=[_FakeNodeArg("actions", [1, 2])],
            output_fn=lambda names, feeds: [
                feeds["latent"].ravel()[:2] + feeds["bias"].ravel()
            ],
        )
        encoder_path = self.attach(encoder)
        actor_path = self.attach(actor)
        processor = self.init_policy_processor(
            [{
                "key": "depth-walk",
                "mode": 4,
                "inputSlot": 1,
                "pipeline": [
                    {"key": "encoder", "resourceId": encoder_path,
                     "inputName": "depth", "outputName": "latent"},
                    {"key": "actor", "resourceId": actor_path,
                     "inputName": "latent", "outputName": "actions",
                     "extraInputs": [{"name": "bias", "values": [0.5, -0.5]}]},
                ],
            }],
            [1, 2],
        )
        processor.compute([
            np.array([4.0], np.float32),
            np.array([2.0, 3.0], np.float32),
        ])
        np.testing.assert_allclose(processor.output_buffers[0], [5.5, 1.5])
        self.assertEqual(len(encoder.runs), 1)
        self.assertEqual(len(actor.runs), 1)
        np.testing.assert_allclose(actor.runs[0][1]["latent"], [[5.0, 2.0, 3.0]])

    def test_policy_stage_uses_resource_output_dim_for_dynamic_metadata(self):
        import botlab_host

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", ["batch", 2])],
            outputs=[_FakeNodeArg("latent", ["batch", "latent_dim"])],
            output_fn=lambda names, feeds: [np.array([1.0, 2.0, 3.0], np.float32)],
        )
        path = self.attach(session)
        resource_id = f"dynamic-policy-{id(self)}"
        botlab_host.project_state().reset(self._directory.name)
        botlab_host.declareResource(
            "model", resource_id, os.path.basename(path), None, "wasm", {"outputDim": 3}, None
        )
        try:
            processor = self.init_policy_processor(
                [{"key": "dynamic", "mode": 0, "inputSlot": 1,
                  "resourceId": resource_id}],
                [1, 2],
            )
            processor.compute([
                np.array([0.0], np.float32), np.zeros(2, np.float32)
            ])
            np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0, 3.0])
        finally:
            state = botlab_host.project_state()
            state.reset(state.directory)

    def test_policy_recurrent_state_resets_when_mode_is_reentered(self):
        def roll(names, feeds):
            output = feeds["obs"].ravel() + feeds["state"].ravel()
            return [output, output]

        recurrent = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2]), _FakeNodeArg("state", [2])],
            outputs=[_FakeNodeArg("out", [1, 2]), _FakeNodeArg("next_state", [2])],
            output_fn=roll,
        )
        other = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        recurrent_path = self.attach(recurrent)
        other_path = self.attach(other)
        processor = self.init_policy_processor(
            [
                {"key": "recurrent", "mode": 0, "inputSlot": 1,
                 "resourceId": recurrent_path},
                {"key": "other", "mode": 1, "inputSlot": 2,
                 "resourceId": other_path},
            ],
            [1, 2, 2],
        )
        one = np.ones(2, np.float32)
        mode_zero = np.array([0.0], np.float32)
        mode_one = np.array([1.0], np.float32)

        processor.compute([mode_zero, one, one])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 1.0])
        processor.compute([mode_zero, one, one])
        np.testing.assert_allclose(processor.output_buffers[0], [2.0, 2.0])
        processor.compute([mode_one, one, one])
        processor.compute([mode_zero, one, one])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 1.0])
        np.testing.assert_array_equal(
            recurrent.runs[-1][1]["state"], np.zeros(2, np.float32)
        )

        processor.reset()
        processor.compute([mode_zero, one, one])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 1.0])

    def test_policy_inference_failure_propagates(self):
        def fail(names, feeds):
            raise RuntimeError("policy exploded")

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=fail,
        )
        path = self.attach(session)
        processor = self.init_policy_processor(
            [{"key": "broken", "mode": 0, "inputSlot": 1,
              "resourceId": path}],
            [1, 2],
        )
        with self.assertRaisesRegex(RuntimeError, "policy exploded"):
            processor.compute([
                np.array([0.0], np.float32), np.ones(2, np.float32)
            ])

    def test_node_id_is_required(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        path = self.attach(session)
        processor = _model.ModelProcessor(session_pool=self.pool)
        with self.assertRaises(ValueError) as ctx:
            processor.init({"resourceId": "r", "modelPath": path}, [2])
        self.assertIn("nodeId is required", str(ctx.exception))

    def test_input_dims_required(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        with self.assertRaises(ValueError) as ctx:
            self.init_processor(session, [0])
        self.assertIn("inputDims[0] is required", str(ctx.exception))

    def test_missing_model_file_is_explicit(self):
        missing = os.path.join(self._directory.name, "gone.onnx")
        processor = _model.ModelProcessor(session_pool=self.pool)
        with self.assertRaises(FileNotFoundError) as ctx:
            processor.init(
                {"nodeId": "n", "resourceId": "res-missing", "modelPath": missing}, [2]
            )
        message = str(ctx.exception)
        self.assertIn("res-missing", message)
        self.assertIn(missing, message)

    def test_model_path_defaults_to_resource_id(self):
        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        default_path = os.path.join(self._directory.name, "default-id.onnx")
        with open(default_path, "wb") as handle:
            handle.write(b"model-default")
        self._sessions["model-default"] = session
        processor = _model.ModelProcessor(session_pool=self.pool)
        processor.init({"nodeId": "n", "resourceId": default_path}, [2])
        processor.compute([np.array([4.0, 5.0], np.float32)])
        np.testing.assert_allclose(processor.output_buffers[0], [4.0, 5.0])

    def test_named_resource_resolves_through_host(self):
        import botlab_host

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        path = self.attach(session)
        resource_id = f"named-model-{id(self)}"
        botlab_host.project_state().reset(self._directory.name)
        botlab_host.declareResource(
            "model", resource_id, os.path.basename(path), None, None, {}, None
        )
        try:
            processor = _model.ModelProcessor(session_pool=self.pool)
            processor.init({"nodeId": "n", "resourceId": resource_id}, [2])
            processor.compute([np.array([7.0, 8.0], np.float32)])
            np.testing.assert_allclose(processor.output_buffers[0], [7.0, 8.0])
        finally:
            state = botlab_host.project_state()
            state.reset(state.directory)

    def test_named_resource_uses_public_provider_contract(self):
        import botlab_host

        session = _FakeSession(
            inputs=[_FakeNodeArg("obs", [1, 2])],
            outputs=[_FakeNodeArg("out", [1, 2])],
            output_fn=lambda names, feeds: [feeds["obs"]],
        )
        path = self.attach(session)
        resource_id = f"wasm-model-{id(self)}"
        botlab_host.project_state().reset(self._directory.name)
        botlab_host.declareResource(
            "model", resource_id, os.path.basename(path), None, "wasm", {}, None
        )
        provider_calls = []

        def factory(model_bytes, providers):
            provider_calls.append(list(providers))
            return self._sessions[model_bytes.decode("utf-8")]

        pool = _model.OnnxSessionPool(
            session_factory=factory,
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        try:
            processor = _model.ModelProcessor(session_pool=pool)
            processor.init({"nodeId": "n", "resourceId": resource_id}, [2])
            processor.compute([np.array([7.0, 8.0], np.float32)])
            self.assertEqual(provider_calls, [["CPUExecutionProvider"]])
            np.testing.assert_allclose(processor.output_buffers[0], [7.0, 8.0])
        finally:
            state = botlab_host.project_state()
            state.reset(state.directory)


@unittest.skipUnless(
    onnxruntime is not None and os.path.isfile(STUDENT_ONNX),
    "onnxruntime or the student.onnx asset is unavailable (tier-B)",
)
class TestStudentOnnxIntegration(unittest.TestCase):
    """Real single-input model: dynamic batch dim, inferred output width."""

    def test_registry_exposes_model_onnx(self):
        self.assertEqual(_model.REGISTRY, {"model-onnx": _model.ModelProcessor})

    def test_inference_output_matches_session_metadata(self):
        processor = _model.ModelProcessor()
        processor.init({"nodeId": "student", "resourceId": STUDENT_ONNX}, [5449])
        # output shape ['batch_size', 16] infers width 16 (batch collapses to 1)
        self.assertEqual(processor.output_buffers[0].size, 16)

        obs = np.full(5449, 0.25, dtype=np.float32)
        processor.compute([obs])
        output = processor.output_buffers[0]
        self.assertEqual(output.size, 16)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(output)))

        session = onnxruntime.InferenceSession(
            STUDENT_ONNX, providers=["CPUExecutionProvider"]
        )
        expected = session.run(["actions"], {"obs": obs.reshape(1, 5449)})[0]
        self.assertEqual(expected.shape, (1, 16))  # consistent with metadata
        np.testing.assert_allclose(output, expected.ravel(), rtol=1e-6)

    def test_wrong_input_length_is_explicit(self):
        processor = _model.ModelProcessor()
        processor.init({"nodeId": "student", "resourceId": STUDENT_ONNX}, [5449])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.zeros(5448, np.float32)])
        # 5448 does not divide the static 5449 dim of ['batch_size', 5449]
        self.assertIn("cannot fit 5448 input values", str(ctx.exception))

    def test_reset_restores_first_step_output(self):
        processor = _model.ModelProcessor()
        processor.init({"nodeId": "student", "resourceId": STUDENT_ONNX}, [5449])
        obs = np.full(5449, 0.25, dtype=np.float32)
        processor.compute([obs])
        first = processor.output_buffers[0].copy()
        processor.reset()
        np.testing.assert_array_equal(
            processor.output_buffers[0], np.zeros(16, np.float32)
        )
        processor.compute([obs])
        np.testing.assert_allclose(processor.output_buffers[0], first, rtol=1e-6)


@unittest.skipUnless(
    onnxruntime is not None and os.path.isfile(STUDENT_MEMORY_ONNX),
    "onnxruntime or the student_memory.onnx asset is unavailable (tier-B)",
)
class TestStudentMemoryIntegration(unittest.TestCase):
    """Real recurrent model: hidden_state input rolled from next_hidden_state."""

    def test_recurrent_state_rolls_through_real_model(self):
        processor = _model.ModelProcessor()
        processor.init(
            {
                "nodeId": "memory",
                "resourceId": STUDENT_MEMORY_ONNX,
                "outputDim": 512,  # latent shape is fully dynamic in metadata
            },
            [181],
        )
        self.assertEqual(processor.output_buffers[0].size, 512)
        states = processor._recurrent_states
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].input_name, "hidden_state")
        self.assertEqual(states[0].output_name, "next_hidden_state")

        session = onnxruntime.InferenceSession(
            STUDENT_MEMORY_ONNX, providers=["CPUExecutionProvider"]
        )
        obs_one = np.full(181, 0.5, dtype=np.float32)
        obs_two = np.full(181, -0.25, dtype=np.float32)
        hidden = np.zeros((1, 1, 512), dtype=np.float32)
        first = session.run(
            ["latent", "next_hidden_state"],
            {"encoded_obs": obs_one.reshape(1, 181), "hidden_state": hidden},
        )
        second = session.run(
            ["latent", "next_hidden_state"],
            {"encoded_obs": obs_two.reshape(1, 181), "hidden_state": first[1]},
        )

        processor.compute([obs_one])
        np.testing.assert_allclose(
            processor.output_buffers[0], first[0].ravel(), rtol=1e-6
        )
        processor.compute([obs_two])
        np.testing.assert_allclose(
            processor.output_buffers[0], second[0].ravel(), rtol=1e-6
        )
        np.testing.assert_allclose(states[0].buffer, second[1].ravel(), rtol=1e-6)


@unittest.skipUnless(
    onnxruntime is not None and os.path.isfile(STUDENT_ONNX),
    "onnxruntime or the student.onnx asset is unavailable (tier-B)",
)
class TestExecutorIntegration(unittest.TestCase):
    """A full plan through PlanExecutor with the real model node."""

    def test_model_node_executes_inside_plan(self):
        class ConstantNumpyProcessor:
            def __init__(self):
                self.output_buffers = []

            def init(self, config, input_dims):
                self.output_buffers = [np.asarray(config["values"], dtype=np.float32)]

            def compute(self, inputs):
                pass

        class ScopeSinkProcessor:
            def __init__(self):
                self.output_buffers = []
                self.received = None

            def init(self, config, input_dims):
                pass

            def compute(self, inputs):
                self.received = [np.array(value, copy=True) for value in inputs]

        registry = {
            "obs-constant": ConstantNumpyProcessor,
            "model-onnx": _model.ModelProcessor,
            "out-scope": ScopeSinkProcessor,
        }
        nodes = [
            {"id": "obs", "type": "obs-constant", "data": {"values": [0.25] * 5449}},
            {"id": "policy", "type": "model-onnx", "data": {"resourceId": STUDENT_ONNX}},
            {"id": "sink", "type": "out-scope", "data": {}},
        ]
        edges = [
            {
                "id": "e1",
                "source": "obs",
                "sourcePort": "out-0",
                "target": "policy",
                "targetPort": "in-0",
            },
            {
                "id": "e2",
                "source": "policy",
                "sourcePort": "out-0",
                "target": "sink",
                "targetPort": "in-0",
            },
        ]
        executor = _executor.PlanExecutor(registry)
        executor.compile(nodes, edges)
        executor.step()
        received = executor.get_processor("sink").received[0]
        self.assertEqual(received.size, 16)
        self.assertTrue(np.all(np.isfinite(received)))

        session = onnxruntime.InferenceSession(
            STUDENT_ONNX, providers=["CPUExecutionProvider"]
        )
        expected = session.run(
            ["actions"], {"obs": np.full((1, 5449), 0.25, dtype=np.float32)}
        )[0].ravel()
        np.testing.assert_allclose(received, expected, rtol=1e-6)

        executor.reset()
        executor.step()
        np.testing.assert_allclose(
            executor.get_processor("sink").received[0], expected, rtol=1e-6
        )
        executor.dispose()


if __name__ == "__main__":
    unittest.main()
