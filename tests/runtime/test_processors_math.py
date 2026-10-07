"""Math-family processor tests (numpy tier).

Mirrors the TS cases in VectorMathProcessor.test.ts and
SignalSpatialProcessors.test.ts plus reset semantics and the explicit error
shapes. Expected values are hand-computed with the same formulas the TS
processors use.
"""

import math
import unittest

import numpy as np  # noqa: E402

from botlab._runtime import _executor  # noqa: E402
from botlab._runtime._processors import _math  # noqa: E402


def f32(*values):
    return np.array(values, dtype=np.float32)


def init_processor(cls, config, input_dims):
    processor = cls()
    processor.init(config, input_dims)
    return processor


class MathProcessorTest(unittest.TestCase):
    def test_add_applies_value_offset(self):
        processor = init_processor(
            _math.MathProcessor, {"operation": "add", "value": [1.0, -2.0, 3.0]}, [3]
        )
        processor.compute([f32(10.0, 20.0, 30.0)])
        np.testing.assert_allclose(
            processor.output_buffers[0], [11.0, 18.0, 33.0], rtol=1e-6, atol=1e-7
        )

    def test_multiply_scalar_and_affine_vectors(self):
        multiply = init_processor(
            _math.MathProcessor, {"operation": "multiply", "value": 0.5}, [2]
        )
        multiply.compute([f32(4.0, 5.0)])
        np.testing.assert_allclose(
            multiply.output_buffers[0], [2.0, 2.5], rtol=1e-6, atol=1e-7
        )

        affine = init_processor(
            _math.MathProcessor,
            {"operation": "affine", "scale": [2.0, 3.0], "offset": 1.0},
            [2],
        )
        affine.compute([f32(1.0, 1.0)])
        np.testing.assert_allclose(
            affine.output_buffers[0], [3.0, 4.0], rtol=1e-6, atol=1e-7
        )

    def test_compute_leaves_input_untouched(self):
        processor = init_processor(
            _math.MathProcessor, {"operation": "multiply", "value": 2.0}, [3]
        )
        source = f32(1.0, 2.0, 3.0)
        original = source.copy()
        processor.compute([source])
        np.testing.assert_array_equal(source, original)

    def test_unsupported_operation_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _math.MathProcessor().init({"operation": "pow"}, [2])
        self.assertIn('unsupported operation "pow"', str(ctx.exception))

    def test_operand_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.MathProcessor().init({"operation": "add", "value": [1.0]}, [2])
        self.assertIn("value must contain exactly 2 numbers, got 1", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.MathProcessor().init(
                {"operation": "add", "value": [1.0, float("nan")]}, [2]
            )
        self.assertIn("value contains non-finite numbers", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.MathProcessor().init(
                {"operation": "affine", "scale": 1.0, "offset": None}, [2]
            )
        self.assertIn(
            "offset must be a finite number or array of size 2", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            _math.MathProcessor().init({"operation": "add", "value": 1.0}, [0])
        self.assertIn("inputDims[0] is required", str(ctx.exception))

    def test_compute_rejects_missing_or_resized_input(self):
        processor = init_processor(
            _math.MathProcessor, {"operation": "add", "value": 1.0}, [2]
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("Missing input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(1.0)])
        self.assertIn("input length changed from 2 to 1", str(ctx.exception))


class VectorMathProcessorTest(unittest.TestCase):
    def run_operation(self, operation, left, right):
        processor = init_processor(
            _math.VectorMathProcessor, {"operation": operation}, [len(left), len(right)]
        )
        processor.compute([f32(*left), f32(*right)])
        return processor.output_buffers[0]

    def test_elementwise_operations(self):
        np.testing.assert_allclose(
            self.run_operation("add", [1, 2, 3], [10, 20, 30]), [11, 22, 33], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.run_operation("subtract", [1, 2, 3], [10, 20, 30]),
            [-9, -18, -27],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("multiply", [1, 2, 3], [10, 20, 30]),
            [10, 40, 90],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("divide", [10, 20, 30], [2, 4, 5]), [5, 5, 6], rtol=1e-6
        )

    def test_scalar_broadcasts_from_either_side(self):
        np.testing.assert_allclose(
            self.run_operation("multiply", [1, 2, 3], [2]), [2, 4, 6], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.run_operation("subtract", [10], [1, 2, 3]), [9, 8, 7], rtol=1e-6
        )

    def test_shape_and_operation_validation(self):
        with self.assertRaises(ValueError) as ctx:
            _math.VectorMathProcessor().init({"operation": "add"}, [3, 2])
        self.assertIn(
            "inputs must have equal dimensions or one scalar, got 3 and 2",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.VectorMathProcessor().init({"operation": "add"}, [3])
        self.assertIn("requires exactly two inputs", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.VectorMathProcessor().init({"operation": "modulo"}, [2, 2])
        self.assertIn(
            "operation must be one of add, subtract, multiply, divide",
            str(ctx.exception),
        )

    def test_input_dimensions_changed_after_init(self):
        processor = init_processor(
            _math.VectorMathProcessor, {"operation": "add"}, [2, 2]
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(1.0, 2.0), f32(3.0)])
        self.assertIn("input dimensions changed after init", str(ctx.exception))


class SignalMathProcessorTest(unittest.TestCase):
    def run_operation(self, operation, values, width=None, second=None):
        config = {"operation": operation}
        if width is not None:
            config["width"] = width
        dims = [len(values)] + ([len(second)] if second is not None else [])
        processor = init_processor(_math.SignalMathProcessor, config, dims)
        inputs = [f32(*values)] + ([f32(*second)] if second is not None else [])
        processor.compute(inputs)
        return processor.output_buffers[0]

    def test_grouped_normalize_sum_and_sqrt(self):
        np.testing.assert_allclose(
            self.run_operation("normalize", [3, 4, 0, 0], width=2),
            [0.6, 0.8, 0.0, 0.0],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            self.run_operation("sum", [1, 2, 3, 4], width=2), [3, 7], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.run_operation("sqrt", [4, 9]), [2, 3], rtol=1e-6, atol=1e-7
        )

    def test_elementwise_unary_operations(self):
        np.testing.assert_allclose(
            self.run_operation("abs", [-1.0, 2.0]), [1.0, 2.0], rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            self.run_operation("sin", [0.0, math.pi / 2]),
            [0.0, 1.0],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("cos", [0.0, math.pi]),
            [1.0, -1.0],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("exp", [0.0, 1.0]), [1.0, math.e], rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            self.run_operation("log", [1.0, math.e]), [0.0, 1.0], rtol=1e-6, atol=1e-7
        )

    def test_grouped_reductions(self):
        np.testing.assert_allclose(
            self.run_operation("min", [1, 5, 2, 3], width=2), [1, 2], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.run_operation("max", [1, 5, 2, 3], width=2), [5, 3], rtol=1e-6
        )
        np.testing.assert_allclose(
            self.run_operation("norm", [3, 4, 2, 3], width=2),
            [5.0, math.sqrt(13)],
            rtol=1e-6,
        )

    def test_binary_dot_with_broadcast_and_cross(self):
        np.testing.assert_allclose(
            self.run_operation("dot", [1, 0, 0, 0, 1, 0], width=3, second=[2, 3, 4]),
            [2, 3],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("cross", [1, 0, 0], width=3, second=[0, 1, 0]),
            [0, 0, 1],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_domain_and_configuration_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.run_operation("sqrt", [4.0, -1.0])
        self.assertIn("sqrt input[1] must be non-negative", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.run_operation("log", [1.0, 0.0])
        self.assertIn("log input[1] must be positive", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.run_operation("sum", [1, 2, 3, 4], width=3)
        self.assertIn(
            "width must be a positive divisor of input dimension 4", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            self.run_operation("cross", [1, 0, 1, 0], width=2, second=[1, 0, 1, 0])
        self.assertIn("cross requires width 3", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalMathProcessor().init(
                {"operation": "dot", "width": 3}, [3]
            )
        self.assertIn("dot requires exactly two inputs", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalMathProcessor().init({"operation": "abs"}, [1, 1])
        self.assertIn("abs requires exactly one input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalMathProcessor().init({"operation": "median"}, [2])
        self.assertIn("operation must be one of", str(ctx.exception))

    def test_non_finite_input_and_dimension_change(self):
        processor = init_processor(
            _math.SignalMathProcessor, {"operation": "abs"}, [2]
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(1.0, float("nan"))])
        self.assertIn("input[1] must be finite", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(1.0)])
        self.assertIn("input dimensions changed after init", str(ctx.exception))


class SpatialMathProcessorTest(unittest.TestCase):
    half_sqrt = math.sqrt(0.5)
    z_quarter = (half_sqrt, 0.0, 0.0, half_sqrt)

    def run_operation(self, operation, input_dims, *inputs):
        processor = init_processor(
            _math.SpatialMathProcessor, {"operation": operation}, input_dims
        )
        processor.compute([f32(*values) for values in inputs])
        return processor.output_buffers[0]

    def test_rotate_inverse_and_matrix_encodings(self):
        np.testing.assert_allclose(
            self.run_operation(
                "quat_rotate", [4, 3], self.z_quarter, (1, 0, 0)
            ),
            [0, 1, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation(
                "quat_rotate_inverse", [4, 3], self.z_quarter, (1, 0, 0)
            ),
            [0, -1, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("quat_to_matrix", [4], self.z_quarter),
            [0, -1, 0, 1, 0, 0, 0, 0, 1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("quat_to_rotation6d", [4], self.z_quarter),
            [0, -1, 1, 0, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("quat_to_tangent_normal", [4], self.z_quarter),
            [0, 1, 0, 0, 0, 1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("quat_heading", [4], self.z_quarter),
            self.z_quarter,
            atol=1e-6,
        )

    def test_multiply_inverse_and_relative_identities(self):
        np.testing.assert_allclose(
            self.run_operation(
                "quat_multiply", [4, 4], self.z_quarter, self.z_quarter
            ),
            [0, 0, 0, 1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation("quat_inverse", [4], self.z_quarter),
            (self.half_sqrt, 0, 0, -self.half_sqrt),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation(
                "relative_quaternion", [4, 4], self.z_quarter, self.z_quarter
            ),
            [1, 0, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation(
                "relative_position", [3, 3, 4], (1, 1, 0), (1, 0, 0), self.z_quarter
            ),
            [1, 0, 0],
            atol=1e-6,
        )

    def test_slerp_uses_shortest_arc(self):
        np.testing.assert_allclose(
            self.run_operation(
                "quat_slerp",
                [4, 4, 1],
                (1, 0, 0, 0),
                (-1, 0, 0, 0),
                (0.5,),
            ),
            [1, 0, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation(
                "quat_slerp",
                [4, 4, 1],
                (1, 0, 0, 0),
                self.z_quarter,
                (0.5,),
            ),
            [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)],
            atol=1e-6,
        )

    def test_unnormalized_and_batched_rotations(self):
        np.testing.assert_allclose(
            self.run_operation("quat_rotate", [4, 3], (2, 0, 0, 0), (1, 2, 3)),
            [1, 2, 3],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.run_operation(
                "quat_rotate",
                [8, 3],
                (1, 0, 0, 0) + self.z_quarter,
                (1, 2, 3),
            ),
            [1, 2, 3, -2, 1, 3],
            atol=1e-6,
        )

    def test_singleton_batch_broadcast(self):
        np.testing.assert_allclose(
            self.run_operation(
                "quat_rotate", [4, 6], self.z_quarter, (1, 0, 0, 0, 1, 0)
            ),
            [0, 1, 0, -1, 0, 0],
            atol=1e-6,
        )

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.SpatialMathProcessor().init({"operation": "quat_rotate"}, [5, 3])
        self.assertIn(
            "quaternion dimension must be a positive multiple of 4, got 5",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SpatialMathProcessor().init(
                {"operation": "relative_position"}, [3, 3, 3]
            )
        self.assertIn(
            "orientation dimension must be a positive multiple of 4, got 3",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SpatialMathProcessor().init({"operation": "quat_slerp"}, [4, 4])
        self.assertIn("quat_slerp requires exactly 3 inputs", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SpatialMathProcessor().init(
                {"operation": "quat_multiply"}, [8, 12]
            )
        self.assertIn(
            "quaternion batches must match or one input must contain one batch",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SpatialMathProcessor().init({"operation": "quat_look"}, [4])
        self.assertIn("operation must be one of", str(ctx.exception))
        inverse = init_processor(
            _math.SpatialMathProcessor, {"operation": "quat_inverse"}, [4]
        )
        with self.assertRaises(ValueError) as ctx:
            inverse.compute([f32(0, 0, 0, 0)])
        self.assertIn("must not contain a zero quaternion", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            inverse.compute([f32(1, float("nan"), 0, 0)])
        self.assertIn("quaternion must contain finite numbers", str(ctx.exception))
        slerp = init_processor(
            _math.SpatialMathProcessor, {"operation": "quat_slerp"}, [4, 4, 1]
        )
        with self.assertRaises(ValueError) as ctx:
            slerp.compute(
                [f32(1, 0, 0, 0), f32(0, 1, 0, 0), f32(1.5)]
            )
        self.assertIn("interpolation must be between 0 and 1", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            slerp.compute([f32(1, 0, 0, 0), f32(0, 1, 0, 0), f32(0.5, 0.5)])
        self.assertIn("input dimensions changed after init", str(ctx.exception))


class ClampProcessorTest(unittest.TestCase):
    def test_clamps_against_scalar_and_vector_limits(self):
        processor = init_processor(
            _math.ClampProcessor, {"min": -1.0, "max": [2.0, 0.5]}, [2]
        )
        source = f32(-5.0, 0.25)
        original = source.copy()
        processor.compute([source])
        np.testing.assert_allclose(
            processor.output_buffers[0], [-1.0, 0.25], rtol=1e-6, atol=1e-7
        )
        np.testing.assert_array_equal(source, original)

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.ClampProcessor().init({"min": [0.0, 2.0], "max": [1.0, 1.0]}, [2])
        self.assertIn("min[1] must be <= max[1]", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.ClampProcessor().init({"min": [1.0], "max": 2.0}, [2])
        self.assertIn(
            "min must contain exactly 2 finite numbers", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            _math.ClampProcessor().init({"min": float("nan"), "max": 2.0}, [2])
        self.assertIn("min must be finite", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.ClampProcessor().init({"min": "lo", "max": 2.0}, [2])
        self.assertIn(
            "min must be a finite number or a 2-value array", str(ctx.exception)
        )

    def test_compute_validation_errors(self):
        processor = init_processor(
            _math.ClampProcessor, {"min": 0.0, "max": 1.0}, [2]
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("Missing input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(1.0)])
        self.assertIn(
            "Input length 1 does not match output length 2", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(0.5, float("inf"))])
        self.assertIn("input[1] is not finite", str(ctx.exception))


class SignalFilterProcessorTest(unittest.TestCase):
    def test_ema_first_sample_then_decay_and_reset(self):
        ema = init_processor(
            _math.SignalFilterProcessor, {"kind": "ema", "alpha": 0.25}, [1]
        )
        ema.compute([f32(8.0)])
        np.testing.assert_allclose(ema.output_buffers[0], [8.0], rtol=1e-6)
        ema.compute([f32(0.0)])
        np.testing.assert_allclose(ema.output_buffers[0], [6.0], rtol=1e-6, atol=1e-7)
        ema.reset()
        ema.compute([f32(3.0)])
        np.testing.assert_allclose(ema.output_buffers[0], [3.0], rtol=1e-6)

    def test_moving_average_ring_and_reset(self):
        average = init_processor(
            _math.SignalFilterProcessor, {"kind": "moving_average", "window": 3}, [1]
        )
        outputs = []
        for sample in (3.0, 6.0, 9.0, 12.0):
            average.compute([f32(sample)])
            outputs.append(float(average.output_buffers[0][0]))
        np.testing.assert_allclose(outputs, [3.0, 4.5, 6.0, 9.0], rtol=1e-6, atol=1e-7)
        average.reset()
        average.compute([f32(4.0)])
        np.testing.assert_allclose(average.output_buffers[0], [4.0], rtol=1e-6)

    def test_difference_first_zero_then_derivative_and_reset(self):
        difference = init_processor(
            _math.SignalFilterProcessor, {"kind": "difference", "dt": 0.5}, [1]
        )
        difference.compute([f32(3.0)])
        np.testing.assert_allclose(
            difference.output_buffers[0], [0.0], rtol=1e-6, atol=1e-7
        )
        difference.compute([f32(4.0)])
        np.testing.assert_allclose(
            difference.output_buffers[0], [2.0], rtol=1e-6, atol=1e-7
        )
        difference.reset()
        difference.compute([f32(9.0)])
        np.testing.assert_allclose(
            difference.output_buffers[0], [0.0], rtol=1e-6, atol=1e-7
        )

    def test_configuration_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.SignalFilterProcessor().init({"kind": "kalman"}, [1])
        self.assertIn(
            'kind must be "ema", "moving_average", or "difference"',
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SignalFilterProcessor().init({"kind": "ema", "alpha": 0.0}, [1])
        self.assertIn(
            "alpha must be a finite number greater than 0 and at most 1",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SignalFilterProcessor().init(
                {"kind": "moving_average", "window": 2.5}, [1]
            )
        self.assertIn("window must be a positive integer", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalFilterProcessor().init({"kind": "difference", "dt": 0.0}, [1])
        self.assertIn(
            "dt must be a finite number greater than 0", str(ctx.exception)
        )

    def test_compute_errors(self):
        ema = init_processor(
            _math.SignalFilterProcessor, {"kind": "ema", "alpha": 0.5}, [2]
        )
        with self.assertRaises(ValueError) as ctx:
            ema.compute([f32(1.0)])
        self.assertIn("input dimensions changed after init", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            ema.compute([f32(1.0, float("nan"))])
        self.assertIn("input[1] must be finite", str(ctx.exception))


class SignalNoiseProcessorTest(unittest.TestCase):
    @staticmethod
    def reference_uniforms(seed, count):
        state = seed
        uniforms = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) % 2 ** 32
            uniforms.append(state / 2 ** 32)
        return uniforms

    def test_uniform_stream_matches_reference_lcg(self):
        uniforms = self.reference_uniforms(7, 4)
        expected = [value * 2.0 - 1.0 for value in uniforms]
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "uniform", "scale": 1.0, "bias": 0.0, "seed": 7},
            [4],
        )
        processor.compute([f32(0.0, 0.0, 0.0, 0.0)])
        np.testing.assert_allclose(
            processor.output_buffers[0], expected, rtol=1e-6, atol=1e-7
        )

    def test_uniform_output_stays_in_unit_range(self):
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "uniform", "scale": 1.0, "bias": 0.0, "seed": 3},
            [8],
        )
        processor.compute([np.zeros(8, dtype=np.float32)])
        values = processor.output_buffers[0]
        self.assertTrue(np.all(values >= -1.0) and np.all(values < 1.0))

    def test_gaussian_matches_reference_box_muller(self):
        u1, u2 = self.reference_uniforms(42, 2)
        radius = math.sqrt(-2.0 * math.log(1.0 - u1))
        angle = 2.0 * math.pi * u2
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "gaussian", "scale": 1.0, "bias": 0.0, "seed": 42},
            [2],
        )
        processor.compute([f32(0.0, 0.0)])
        np.testing.assert_allclose(
            processor.output_buffers[0],
            [radius * math.cos(angle), radius * math.sin(angle)],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_gaussian_spare_carries_across_computes(self):
        u1, u2 = self.reference_uniforms(9, 2)
        radius = math.sqrt(-2.0 * math.log(1.0 - u1))
        angle = 2.0 * math.pi * u2
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "gaussian", "scale": 1.0, "bias": 0.0, "seed": 9},
            [1],
        )
        processor.compute([f32(0.0)])
        np.testing.assert_allclose(
            processor.output_buffers[0], [radius * math.cos(angle)], rtol=1e-6
        )
        processor.compute([f32(0.0)])
        np.testing.assert_allclose(
            processor.output_buffers[0], [radius * math.sin(angle)], rtol=1e-6
        )

    def test_reset_restarts_the_seeded_stream(self):
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {
                "distribution": "gaussian",
                "scale": [0.5, 2.0],
                "bias": 1.0,
                "seed": 42,
            },
            [2],
        )
        processor.compute([f32(10.0, 20.0)])
        first = processor.output_buffers[0].copy()
        processor.compute([f32(10.0, 20.0)])
        self.assertFalse(np.array_equal(processor.output_buffers[0], first))
        processor.reset()
        processor.compute([f32(10.0, 20.0)])
        np.testing.assert_array_equal(processor.output_buffers[0], first)

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.SignalNoiseProcessor().init(
                {"distribution": "uniform", "scale": 1.0, "bias": 0.0, "seed": -1}, [1]
            )
        self.assertIn("seed must be a uint32 integer", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalNoiseProcessor().init(
                {
                    "distribution": "uniform",
                    "scale": 1.0,
                    "bias": 0.0,
                    "seed": 2 ** 32,
                },
                [1],
            )
        self.assertIn("seed must be a uint32 integer", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.SignalNoiseProcessor().init(
                {"distribution": "laplace", "scale": 1.0, "bias": 0.0, "seed": 0}, [1]
            )
        self.assertIn(
            'distribution must be "uniform" or "gaussian"', str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            _math.SignalNoiseProcessor().init(
                {"distribution": "uniform", "scale": [1.0], "bias": 0.0, "seed": 0},
                [2],
            )
        self.assertIn(
            "scale must contain exactly 2 finite numbers", str(ctx.exception)
        )

    def test_overflow_and_input_errors(self):
        overflow = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "uniform", "scale": 0.0, "bias": 1e39, "seed": 0},
            [1],
        )
        with self.assertRaises(ValueError) as ctx:
            overflow.compute([f32(0.0)])
        self.assertIn("output[0] must be finite", str(ctx.exception))
        processor = init_processor(
            _math.SignalNoiseProcessor,
            {"distribution": "uniform", "scale": 1.0, "bias": 0.0, "seed": 0},
            [1],
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32(float("nan"))])
        self.assertIn("input[0] must be finite", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("input dimensions changed after init", str(ctx.exception))


class _FakeFrame:
    def __init__(self, time, index):
        self.dt = 0.02
        self.time = time
        self.index = index


class ClockProcessorTest(unittest.TestCase):
    def test_steps_unit_reads_frame_index(self):
        clock = init_processor(_math.ClockProcessor, {"unit": "steps"}, [])
        values = []
        for index in range(3):
            clock.compute([], _FakeFrame(index * 0.02, index))
            values.append(float(clock.output_buffers[0][0]))
        self.assertEqual(values, [0.0, 1.0, 2.0])
        clock.reset()
        clock.compute([], _FakeFrame(0.0, 0))
        self.assertEqual(float(clock.output_buffers[0][0]), 0.0)

    def test_seconds_unit_reads_frame_time(self):
        clock = init_processor(_math.ClockProcessor, {"unit": "seconds"}, [])
        clock.compute([], _FakeFrame(0.0, 0))
        np.testing.assert_allclose(
            clock.output_buffers[0], [0.0], rtol=1e-6, atol=1e-7
        )
        clock.compute([], _FakeFrame(0.04, 2))
        np.testing.assert_allclose(
            clock.output_buffers[0], [np.float32(0.04)], rtol=1e-6
        )

    def test_missing_frame_is_explicit(self):
        clock = init_processor(_math.ClockProcessor, {"unit": "seconds"}, [])
        with self.assertRaises(ValueError) as ctx:
            clock.compute([])
        self.assertIn("must pass frame=", str(ctx.exception))

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.ClockProcessor().init({"unit": "minutes"}, [])
        self.assertIn(
            "expected no inputs and unit seconds or steps", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            _math.ClockProcessor().init({"unit": "steps"}, [1])
        self.assertIn(
            "expected no inputs and unit seconds or steps", str(ctx.exception)
        )


class PhaseClockProcessorTest(unittest.TestCase):
    def test_phase_tracks_step_input(self):
        clock = init_processor(
            _math.PhaseClockProcessor, {"periodSteps": 200}, [1]
        )
        clock.compute([f32(0.0)])
        np.testing.assert_allclose(
            clock.output_buffers[0], [1.0, 0.0], rtol=1e-6, atol=1e-7
        )
        clock.compute([f32(50.0)])
        np.testing.assert_allclose(
            clock.output_buffers[0], [0.0, 1.0], rtol=1e-6, atol=1e-6
        )
        clock.reset()  # stateless: phase derives from the input alone
        clock.compute([f32(50.0)])
        np.testing.assert_allclose(
            clock.output_buffers[0], [0.0, 1.0], rtol=1e-6, atol=1e-6
        )

    def test_fractional_step_interpolates(self):
        clock = init_processor(
            _math.PhaseClockProcessor, {"periodSteps": 4}, [1]
        )
        clock.compute([f32(0.5)])
        np.testing.assert_allclose(
            clock.output_buffers[0],
            [math.sqrt(0.5), math.sqrt(0.5)],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as ctx:
            _math.PhaseClockProcessor().init({"periodSteps": 4}, [2])
        self.assertIn("step input must have dimension 1, got 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _math.PhaseClockProcessor().init({"periodSteps": 0}, [1])
        self.assertIn("periodSteps must be a positive integer", str(ctx.exception))
        clock = init_processor(
            _math.PhaseClockProcessor, {"periodSteps": 4}, [1]
        )
        with self.assertRaises(ValueError) as ctx:
            clock.compute([])
        self.assertIn("Missing scalar step input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            clock.compute([f32(-1.0)])
        self.assertIn(
            "step input must be non-negative and finite", str(ctx.exception)
        )


class RegistryTest(unittest.TestCase):
    def test_registry_covers_the_math_family_node_types(self):
        self.assertEqual(
            set(_math.REGISTRY),
            {
                "proc-math",
                "proc-vector-math",
                "proc-clamp",
                "proc-phase-clock",
                "proc-clock",
                "proc-spatial-math",
                "proc-signal-math",
                "proc-signal-noise",
                "proc-signal-filter",
            },
        )
        self.assertIs(_math.REGISTRY["proc-math"], _math.MathProcessor)
        self.assertIs(_math.REGISTRY["proc-signal-noise"], _math.SignalNoiseProcessor)


class ConstantSource:
    def __init__(self):
        self.output_buffers = []

    def init(self, config, input_dims):
        self.output_buffers = [f32(*config["values"])]

    def compute(self, inputs):
        pass


class Sink:
    def __init__(self):
        self.output_buffers = []
        self.received = None

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        self.received = [values.copy() for values in inputs]


class ExecutorIntegrationTest(unittest.TestCase):
    def test_math_processors_run_inside_plan_executor(self):
        registry = dict(_math.REGISTRY)
        registry["obs-constant"] = ConstantSource
        registry["out-scope"] = Sink

        def node(node_id, node_type, **data):
            return {"id": node_id, "type": node_type, "data": data}

        executor = _executor.PlanExecutor(registry)
        executor.compile(
            [
                node("source", "obs-constant", values=[1.0, 2.0, 3.0]),
                node("math", "proc-math", operation="add", value=[10.0, 20.0, 30.0]),
                node("clamp", "proc-clamp", min=-5.0, max=25.0),
                node("sink", "out-scope"),
            ],
            [
                {"id": "e1", "source": "source", "target": "math"},
                {"id": "e2", "source": "math", "target": "clamp"},
                {"id": "e3", "source": "clamp", "target": "sink"},
            ],
        )
        executor.step()
        sink = executor.get_processor("sink")
        np.testing.assert_allclose(
            sink.received[0], [11.0, 22.0, 25.0], rtol=1e-6, atol=1e-7
        )
        executor.reset()
        executor.dispose()


if __name__ == "__main__":
    unittest.main()
