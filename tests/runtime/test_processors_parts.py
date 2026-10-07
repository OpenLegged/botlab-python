"""Tests for the part-type processors (botlab._runtime._processors._parts).

Mirrors the TypeScript processor tests: fixed configs, deterministic inputs,
strict assert_allclose on happy paths, and exact error-message fragments for
bad configs/inputs (no silent fallbacks). Expected values for resize and the
conditional processors come from ResizeProcessor.test.ts and
GenericSignalProcessors.test.ts.
"""

import math
import unittest

import numpy as np  # noqa: E402

from botlab._runtime import _executor  # noqa: E402
from botlab._runtime._processors import _parts  # noqa: E402


def f32(values):
    return np.array(values, dtype=np.float32)


class RerouteProcessorTest(unittest.TestCase):
    def test_output_aliases_input_buffer(self):
        processor = _parts.RerouteProcessor()
        processor.init({}, [3])
        source = f32([1.0, 2.0, 3.0])
        processor.compute([source])
        self.assertIs(processor.output_buffers[0], source)
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0, 3.0])
        self.assertEqual(processor.output_buffers[0].dtype, np.float32)

    def test_missing_input_raises(self):
        processor = _parts.RerouteProcessor()
        processor.init({}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("[RerouteProcessor] Missing input", str(ctx.exception))

    def test_length_change_raises(self):
        processor = _parts.RerouteProcessor()
        processor.init({}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0, 2.0, 3.0])])
        self.assertIn("input length changed from 2 to 3", str(ctx.exception))

    def test_init_requires_input_dim(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.RerouteProcessor().init({}, [])
        self.assertIn("[RerouteProcessor] inputDims[0] is required", str(ctx.exception))


class ReorderProcessorTest(unittest.TestCase):
    def test_gathers_configured_indices(self):
        processor = _parts.ReorderProcessor()
        processor.init({"indices": [2, 0, 1]}, [3])
        source = f32([10.0, 20.0, 30.0])
        processor.compute([source])
        np.testing.assert_allclose(processor.output_buffers[0], [30.0, 10.0, 20.0])
        np.testing.assert_allclose(source, [10.0, 20.0, 30.0])  # inputs untouched

    def test_accepts_duplicate_indices_and_numpy_arrays(self):
        processor = _parts.ReorderProcessor()
        processor.init({"indices": np.array([0, 0, 2])}, [3])
        processor.compute([f32([10.0, 20.0, 30.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [10.0, 10.0, 30.0])

    def test_missing_indices_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ReorderProcessor().init({}, [3])
        self.assertIn("indices must be a number list", str(ctx.exception))

    def test_empty_indices_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ReorderProcessor().init({"indices": []}, [3])
        self.assertIn("indices must not be empty", str(ctx.exception))

    def test_out_of_range_index_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ReorderProcessor().init({"indices": [0, 3]}, [3])
        self.assertIn("indices must be integers between 0 and 2", str(ctx.exception))

    def test_non_integer_index_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ReorderProcessor().init({"indices": [0.5]}, [3])
        self.assertIn("indices must be integers between 0 and 2", str(ctx.exception))

    def test_compute_length_change_raises(self):
        processor = _parts.ReorderProcessor()
        processor.init({"indices": [0]}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0, 2.0, 3.0])])
        self.assertIn("input length changed from 2 to 3", str(ctx.exception))


def reference_resize(flat, in_w, in_h, out_w, out_h, channels=1,
                     layout="hwc", flip_x=False, flip_y=False):
    """Straight scalar port of the TS resizeBilinear loops (test-only oracle)."""
    def in_index(x, y, c):
        if layout == "chw":
            return c * in_w * in_h + y * in_w + x
        return (y * in_w + x) * channels + c

    def out_index(x, y, c):
        if layout == "chw":
            return c * out_w * out_h + y * out_w + x
        return (y * out_w + x) * channels + c

    output = [0.0] * (out_w * out_h * channels)
    scale_x = in_w / out_w
    scale_y = in_h / out_h
    for y in range(out_h):
        sample_y = out_h - 1 - y if flip_y else y
        src_y = (sample_y + 0.5) * scale_y - 0.5
        y0 = max(0, min(in_h - 1, math.floor(src_y)))
        y1 = min(in_h - 1, y0 + 1)
        wy = float(np.float32(src_y - y0))
        for x in range(out_w):
            sample_x = out_w - 1 - x if flip_x else x
            src_x = (sample_x + 0.5) * scale_x - 0.5
            x0 = max(0, min(in_w - 1, math.floor(src_x)))
            x1 = min(in_w - 1, x0 + 1)
            wx = float(np.float32(src_x - x0))
            for c in range(channels):
                v00 = float(flat[in_index(x0, y0, c)])
                v10 = float(flat[in_index(x1, y0, c)])
                v01 = float(flat[in_index(x0, y1, c)])
                v11 = float(flat[in_index(x1, y1, c)])
                v0 = v00 * (1 - wx) + v10 * wx
                v1 = v01 * (1 - wx) + v11 * wx
                output[out_index(x, y, c)] = v0 * (1 - wy) + v1 * wy
    return output


class ResizeProcessorTest(unittest.TestCase):
    def test_bilinear_downscale_matches_ts_test(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1,
             "outputHeight": 1, "channels": 1},
            [4],
        )
        processor.compute([f32([1.0, 2.0, 3.0, 4.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [2.5])

    def test_flip_y_matches_ts_test(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 2,
             "outputHeight": 2, "channels": 1, "flipY": True},
            [4],
        )
        processor.compute([f32([1.0, 2.0, 3.0, 4.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [3.0, 4.0, 1.0, 2.0])

    def test_upscale_hand_computed_grid(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 3,
             "outputHeight": 3, "channels": 1},
            [4],
        )
        processor.compute([f32([1.0, 2.0, 3.0, 4.0])])
        # Hand-derived from the TS sampling math: half-pixel centers with
        # scale 2/3 give wx/wy of -1/6, 1/2, 1/6 and clamped x0==x1==1 on the
        # last column/row, so those output pixels fall on the input border.
        expected = [
            1 / 2, 7 / 6, 5 / 3,
            11 / 6, 5 / 2, 3.0,
            17 / 6, 7 / 2, 4.0,
        ]
        np.testing.assert_allclose(processor.output_buffers[0], expected, rtol=1e-6)

    def test_chw_layout_downscale(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1,
             "outputHeight": 1, "channels": 2, "layout": "chw"},
            [8],
        )
        # chw planes: c0=[1,2,3,4], c1=[5,6,7,8]; bilinear average per channel.
        processor.compute([f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [2.5, 6.5])

    def test_random_hwc_and_chw_match_reference_loops(self):
        rng = np.random.default_rng(20260928)
        cases = [
            (3, 4, 5, 2, 2, "hwc", False, False),
            (3, 4, 5, 2, 2, "chw", False, False),
            (4, 2, 2, 5, 3, "hwc", True, False),
            (2, 5, 6, 3, 3, "chw", True, True),
        ]
        for in_w, in_h, out_w, out_h, channels, layout, flip_x, flip_y in cases:
            with self.subTest(layout=layout, flip_x=flip_x, flip_y=flip_y):
                processor = _parts.ResizeProcessor()
                processor.init(
                    {"inputWidth": in_w, "inputHeight": in_h, "outputWidth": out_w,
                     "outputHeight": out_h, "channels": channels, "layout": layout,
                     "flipX": flip_x, "flipY": flip_y},
                    [in_w * in_h * channels],
                )
                source = rng.random(in_w * in_h * channels).astype(np.float32)
                snapshot = source.copy()
                processor.compute([source])
                expected = reference_resize(
                    source, in_w, in_h, out_w, out_h, channels, layout, flip_x, flip_y
                )
                np.testing.assert_allclose(
                    processor.output_buffers[0], expected, rtol=1e-6, atol=1e-7
                )
                np.testing.assert_allclose(source, snapshot)  # inputs untouched

    def test_identity_shape_without_flips_copies(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 2,
             "outputHeight": 2, "channels": 1},
            [4],
        )
        source = f32([1.0, 2.0, 3.0, 4.0])
        processor.compute([source])
        np.testing.assert_allclose(processor.output_buffers[0], source)
        self.assertIsNot(processor.output_buffers[0], source)  # copy, not alias

    def test_channels_inferred_from_input_dim(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1, "outputHeight": 1},
            [4],
        )
        self.assertEqual(len(processor.output_buffers[0]), 1)
        processor.compute([f32([1.0, 2.0, 3.0, 4.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [2.5])

    def test_missing_width_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ResizeProcessor().init({"inputHeight": 2}, [4])
        self.assertIn("inputWidth must be a positive integer", str(ctx.exception))

    def test_bad_layout_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ResizeProcessor().init(
                {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1,
                 "outputHeight": 1, "layout": "flat"},
                [4],
            )
        self.assertIn('layout must be "hwc" or "chw"', str(ctx.exception))

    def test_bad_channels_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ResizeProcessor().init(
                {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1,
                 "outputHeight": 1, "channels": 0},
                [4],
            )
        self.assertIn("channels must be a positive integer", str(ctx.exception))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.ResizeProcessor().init(
                {"inputWidth": 2, "inputHeight": 2, "outputWidth": 1,
                 "outputHeight": 1, "channels": 3},
                [4],
            )
        self.assertIn(
            "input length 4 does not match inputWidth*inputHeight*channels 12",
            str(ctx.exception),
        )

    def test_compute_length_change_raises(self):
        processor = _parts.ResizeProcessor()
        processor.init(
            {"inputWidth": 1, "inputHeight": 2, "outputWidth": 1,
             "outputHeight": 1, "channels": 1},
            [2],
        )
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0, 2.0, 3.0])])
        self.assertIn("input length changed from 2 to 3", str(ctx.exception))


class MuxProcessorTest(unittest.TestCase):
    def test_concatenates_inputs_and_exposes_segments(self):
        processor = _parts.MuxProcessor()
        processor.init({"channels": 3}, [2, 1, 3])
        first, second, third = f32([1.0, 2.0]), f32([30.0]), f32([4.0, 5.0, 6.0])
        processor.compute([first, second, third])
        np.testing.assert_allclose(
            processor.output_buffers[0], [1.0, 2.0, 30.0, 4.0, 5.0, 6.0]
        )
        self.assertEqual(processor.output_segments[0], [2, 1, 3])
        np.testing.assert_allclose(first, [1.0, 2.0])  # inputs untouched

    def test_channels_defaults_to_four(self):
        processor = _parts.MuxProcessor()
        processor.init({}, [1, 1, 1, 1])
        processor.compute([f32([1.0]), f32([2.0]), f32([3.0]), f32([4.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0, 3.0, 4.0])

    def test_channel_count_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.MuxProcessor().init({"channels": 2}, [1, 1, 1])
        self.assertIn("[MuxProcessor] expected 2 inputs, got 3", str(ctx.exception))

    def test_zero_channels_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.MuxProcessor().init({"channels": 0}, [])
        self.assertIn("channels must be a positive integer", str(ctx.exception))

    def test_missing_init_dim_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.MuxProcessor().init({"channels": 2}, [2, 0])
        self.assertIn("[MuxProcessor] Missing input at slot 1", str(ctx.exception))

    def test_compute_missing_input_raises(self):
        processor = _parts.MuxProcessor()
        processor.init({"channels": 2}, [1, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0]), None])
        self.assertIn("[MuxProcessor] Missing input at slot 1", str(ctx.exception))


class DemuxProcessorTest(unittest.TestCase):
    def test_splits_input_by_sizes(self):
        processor = _parts.DemuxProcessor()
        processor.init({"sizes": [2, 3]}, [5])
        source = f32([1.0, 2.0, 3.0, 4.0, 5.0])
        processor.compute([source])
        self.assertEqual(len(processor.output_buffers), 2)
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 2.0])
        np.testing.assert_allclose(processor.output_buffers[1], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(source, [1.0, 2.0, 3.0, 4.0, 5.0])  # untouched

    def test_defaults_to_four_unit_outputs(self):
        processor = _parts.DemuxProcessor()
        processor.init({}, [4])
        processor.compute([f32([9.0, 8.0, 7.0, 6.0])])
        self.assertEqual(len(processor.output_buffers), 4)
        np.testing.assert_allclose(processor.output_buffers[2], [7.0])

    def test_channels_scalar_config_splits_units(self):
        processor = _parts.DemuxProcessor()
        processor.init({"channels": 2}, [2])
        processor.compute([f32([5.0, 6.0])])
        self.assertEqual(
            [len(buf) for buf in processor.output_buffers], [1, 1]
        )

    def test_total_size_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.DemuxProcessor().init({"sizes": [1, 2]}, [5])
        self.assertIn(
            "output sizes total 3 does not match input dimension 5", str(ctx.exception)
        )

    def test_non_positive_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.DemuxProcessor().init({"sizes": [0, 5]}, [5])
        self.assertIn("output sizes must be positive integers", str(ctx.exception))

    def test_string_size_raises_without_coercion(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.DemuxProcessor().init({"sizes": ["2", 3]}, [5])
        self.assertIn("output sizes must be positive integers", str(ctx.exception))

    def test_compute_length_change_raises(self):
        processor = _parts.DemuxProcessor()
        processor.init({"sizes": [1, 1]}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0, 2.0, 3.0])])
        self.assertIn(
            "input length 3 does not match compiled dimension 2", str(ctx.exception)
        )


class OneHotProcessorTest(unittest.TestCase):
    def test_encodes_index(self):
        processor = _parts.OneHotProcessor()
        processor.init({"size": 3}, [1])
        processor.compute([f32([2])])
        np.testing.assert_allclose(processor.output_buffers[0], [0.0, 0.0, 1.0])
        processor.compute([f32([0])])  # previous one-hot bit must be cleared
        np.testing.assert_allclose(processor.output_buffers[0], [1.0, 0.0, 0.0])

    def test_non_unit_input_dim_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.OneHotProcessor().init({"size": 3}, [2])
        self.assertIn("input must have dimension 1, got 2", str(ctx.exception))

    def test_missing_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.OneHotProcessor().init({}, [1])
        self.assertIn("size must be a positive integer", str(ctx.exception))

    def test_out_of_range_index_raises(self):
        processor = _parts.OneHotProcessor()
        processor.init({"size": 3}, [1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([3])])
        self.assertIn("index must be an integer between 0 and 2", str(ctx.exception))

    def test_non_integer_index_raises(self):
        processor = _parts.OneHotProcessor()
        processor.init({"size": 3}, [1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.5])])
        self.assertIn("index must be an integer between 0 and 2", str(ctx.exception))


class OneHotIndexProcessorTest(unittest.TestCase):
    def test_returns_first_index_above_threshold(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({"threshold": 0.5, "defaultIndex": 7}, [3])
        processor.compute([f32([0.1, 0.9, 0.9])])
        self.assertEqual(processor.output_buffers[0][0], 1.0)  # tie -> first index

    def test_falls_back_to_default_index(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({"threshold": 0.5, "defaultIndex": 4}, [3])
        processor.compute([f32([0.5, 0.0, -1.0])])  # strict '>' keeps the default
        self.assertEqual(processor.output_buffers[0][0], 4.0)

    def test_defaults_are_zero_threshold_and_index(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({}, [2])
        processor.compute([f32([0.0, -1.0])])
        self.assertEqual(processor.output_buffers[0][0], 0.0)

    def test_negative_default_index_is_preserved(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({"threshold": 1.0, "defaultIndex": -1}, [2])
        processor.compute([f32([0.0, 0.0])])
        self.assertEqual(processor.output_buffers[0][0], -1.0)

    def test_bad_threshold_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.OneHotIndexProcessor().init({"threshold": "abc"}, [2])
        self.assertIn("threshold must be finite", str(ctx.exception))

    def test_bad_default_index_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.OneHotIndexProcessor().init({"defaultIndex": 1.5}, [2])
        self.assertIn("defaultIndex must be an integer", str(ctx.exception))

    def test_non_finite_input_reports_first_bad_index(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({"threshold": 0.0}, [3])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([0.1, np.nan, np.inf])])
        self.assertIn("input[1] must be finite", str(ctx.exception))

    def test_compute_length_change_raises(self):
        processor = _parts.OneHotIndexProcessor()
        processor.init({}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0])])
        self.assertIn("input length changed from 2 to 1", str(ctx.exception))


class WhereProcessorTest(unittest.TestCase):
    def init_where(self):
        processor = _parts.WhereProcessor()
        processor.init({}, [1, 2, 2])
        return processor

    def test_returns_true_branch_when_condition_nonzero(self):
        processor = self.init_where()
        processor.compute([f32([1]), f32([10.0, 20.0]), f32([-10.0, -20.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [10.0, 20.0])

    def test_returns_false_branch_when_condition_zero(self):
        processor = self.init_where()
        processor.compute([f32([0]), f32([10.0, 20.0]), f32([-10.0, -20.0])])
        np.testing.assert_allclose(processor.output_buffers[0], [-10.0, -20.0])

    def test_output_is_a_copy_not_branch_alias(self):
        processor = self.init_where()
        when_true = f32([10.0, 20.0])
        processor.compute([f32([-3.0]), when_true, f32([-10.0, -20.0])])
        self.assertIsNot(processor.output_buffers[0], when_true)
        np.testing.assert_allclose(when_true, [10.0, 20.0])  # inputs untouched

    def test_wrong_slot_count_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.WhereProcessor().init({}, [1, 2])
        self.assertIn(
            "requires condition, when_true, and when_false inputs", str(ctx.exception)
        )

    def test_non_scalar_condition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.WhereProcessor().init({}, [2, 1, 1])
        self.assertIn("condition input must be scalar, got 2", str(ctx.exception))

    def test_mismatched_branch_dims_raise(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.WhereProcessor().init({}, [1, 2, 3])
        self.assertIn(
            "branch inputs must have the same positive dimension, got 2 and 3",
            str(ctx.exception),
        )

    def test_non_finite_condition_raises_at_compute(self):
        processor = self.init_where()
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([np.nan]), f32([1.0, 2.0]), f32([3.0, 4.0])])
        self.assertIn("condition input must be a finite scalar", str(ctx.exception))

    def test_branch_length_change_raises_at_compute(self):
        processor = self.init_where()
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1]), f32([1.0]), f32([3.0, 4.0])])
        self.assertIn("branch input dimensions changed after init", str(ctx.exception))


class LogicProcessorTest(unittest.TestCase):
    def test_all_requires_every_input_truthy(self):
        processor = _parts.LogicProcessor()
        processor.init({"operator": "all"}, [1, 1, 1])
        processor.compute([f32([1]), f32([-2]), f32([0.5])])
        self.assertEqual(processor.output_buffers[0][0], 1.0)
        processor.compute([f32([1]), f32([0]), f32([0.5])])
        self.assertEqual(processor.output_buffers[0][0], 0.0)

    def test_any_needs_one_truthy_input(self):
        processor = _parts.LogicProcessor()
        processor.init({"operator": "any"}, [1, 1])
        processor.compute([f32([0]), f32([-2])])
        self.assertEqual(processor.output_buffers[0][0], 1.0)
        processor.compute([f32([0]), f32([0])])
        self.assertEqual(processor.output_buffers[0][0], 0.0)

    def test_not_inverts_single_input(self):
        processor = _parts.LogicProcessor()
        processor.init({"operator": "not"}, [1])
        processor.compute([f32([0])])
        self.assertEqual(processor.output_buffers[0][0], 1.0)
        processor.compute([f32([0.25])])
        self.assertEqual(processor.output_buffers[0][0], 0.0)

    def test_unsupported_operator_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.LogicProcessor().init({"operator": "xor"}, [1, 1])
        self.assertIn("operator must be one of all, any, not", str(ctx.exception))

    def test_missing_operator_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.LogicProcessor().init({}, [1])
        self.assertIn("operator must be one of all, any, not", str(ctx.exception))

    def test_not_with_two_inputs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.LogicProcessor().init({"operator": "not"}, [1, 1])
        self.assertIn("not requires exactly one input", str(ctx.exception))

    def test_non_scalar_input_dims_raise(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.LogicProcessor().init({"operator": "all"}, [1, 2])
        self.assertIn(
            "all inputs must be connected scalar signals", str(ctx.exception)
        )

    def test_no_inputs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.LogicProcessor().init({"operator": "all"}, [])
        self.assertIn(
            "all inputs must be connected scalar signals", str(ctx.exception)
        )

    def test_non_scalar_or_non_finite_compute_input_raises(self):
        processor = _parts.LogicProcessor()
        processor.init({"operator": "all"}, [1, 1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1]), f32([np.nan])])
        self.assertIn("input[1] must be a finite scalar", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1]), f32([1.0, 2.0])])
        self.assertIn("input[1] must be a finite scalar", str(ctx.exception))


class CompareProcessorTest(unittest.TestCase):
    def test_constant_comparisons(self):
        cases = [
            ("gt", 2.0, 1.0, 1.0),
            ("gte", 2.0, 2.0, 1.0),
            ("lt", 2.0, 3.0, 1.0),
            ("lte", 2.0, 2.0, 1.0),
            ("eq", 2.0, 2.0, 1.0),
            ("ne", 2.0, 2.0, 0.0),
        ]
        for operator, left, value, expected in cases:
            with self.subTest(operator=operator):
                processor = _parts.CompareProcessor()
                processor.init({"operator": operator, "value": value}, [1])
                processor.compute([f32([left])])
                self.assertEqual(processor.output_buffers[0][0], expected)

    def test_two_input_comparison(self):
        processor = _parts.CompareProcessor()
        processor.init({"operator": "lt"}, [1, 1])
        processor.compute([f32([1]), f32([2])])
        self.assertEqual(processor.output_buffers[0][0], 1.0)

    def test_unsupported_operator_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init({"operator": "between", "value": 1}, [1])
        self.assertIn("operator must be one of", str(ctx.exception))

    def test_non_finite_value_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init(
                {"operator": "gt", "value": float("nan")}, [1]
            )
        self.assertIn("value must be finite", str(ctx.exception))

    def test_string_value_raises_without_coercion(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init({"operator": "gt", "value": "3"}, [1])
        self.assertIn("value must be finite", str(ctx.exception))

    def test_constant_with_two_inputs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init({"operator": "gt", "value": 1}, [1, 1])
        self.assertIn(
            "constant comparisons require exactly one input", str(ctx.exception)
        )

    def test_two_input_mode_without_value_and_one_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init({"operator": "gt"}, [1])
        self.assertIn(
            "signal comparisons require exactly two inputs", str(ctx.exception)
        )

    def test_non_scalar_input_dim_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.CompareProcessor().init({"operator": "gt", "value": 1}, [2])
        self.assertIn("inputDims[0] must be 1, got 2", str(ctx.exception))

    def test_compute_rejects_non_finite_left(self):
        processor = _parts.CompareProcessor()
        processor.init({"operator": "gt", "value": 1}, [1])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([np.nan])])
        self.assertIn("left input must be finite", str(ctx.exception))


class SignalLabelProcessorTest(unittest.TestCase):
    def test_aliases_input_and_keeps_segments(self):
        processor = _parts.SignalLabelProcessor()
        processor.init(
            {"signalLabel": " imu ", "upstreamInputSegments": [[2]]}, [2]
        )
        source = f32([7.0, 8.0])
        processor.compute([source])
        self.assertIs(processor.output_buffers[0], source)
        self.assertEqual(processor.output_segments, [[2]])

    def test_no_segments_leaves_output_segments_none(self):
        processor = _parts.SignalLabelProcessor()
        processor.init({"signalLabel": "imu", "upstreamInputSegments": [None]}, [2])
        self.assertIsNone(processor.output_segments)

    def test_missing_input_signal_raises_with_label(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.SignalLabelProcessor().init(
                {"signalLabel": "imu", "upstreamInputSegments": [None]}, [0]
            )
        self.assertIn('"imu" requires an input signal', str(ctx.exception))

    def test_unlabeled_missing_input_signal_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parts.SignalLabelProcessor().init({}, [])
        self.assertIn(
            "unlabeled signal label node requires an input signal", str(ctx.exception)
        )

    def test_compute_missing_input_raises_with_label(self):
        processor = _parts.SignalLabelProcessor()
        processor.init({"signalLabel": "imu"}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn('"imu" missing input', str(ctx.exception))

    def test_compute_length_change_raises_with_label(self):
        processor = _parts.SignalLabelProcessor()
        processor.init({"signalLabel": "imu"}, [2])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([f32([1.0, 2.0, 3.0])])
        self.assertIn('"imu" input length changed from 2 to 3', str(ctx.exception))


class RegistryTest(unittest.TestCase):
    def test_registry_keys_match_engine_mapping(self):
        self.assertEqual(
            set(_parts.REGISTRY),
            {
                "proc-reroute",
                "proc-reorder",
                "proc-resize",
                "proc-mux",
                "proc-demux",
                "proc-one-hot",
                "proc-onehot-index",
                "proc-where",
                "proc-logic",
                "proc-compare",
                "proc-signal-send",
                "proc-signal-receive",
            },
        )

    def test_signal_nodes_share_the_label_processor(self):
        self.assertIs(_parts.REGISTRY["proc-signal-send"], _parts.SignalLabelProcessor)
        self.assertIs(
            _parts.REGISTRY["proc-signal-receive"], _parts.SignalLabelProcessor
        )

    def test_every_entry_is_instantiable_with_output_buffers(self):
        for node_type, processor_class in _parts.REGISTRY.items():
            with self.subTest(node_type=node_type):
                processor = processor_class()
                self.assertTrue(hasattr(processor, "output_buffers"))


class _ConstantSource:
    def __init__(self):
        self.output_buffers = []

    def init(self, config, input_dims):
        self.output_buffers = [np.array(config["values"], dtype=np.float32)]

    def compute(self, inputs):
        pass


class _ScopeSink:
    def __init__(self):
        self.output_buffers = []
        self.received = None

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        self.received = list(inputs)


class ExecutorIntegrationTest(unittest.TestCase):
    def test_parts_run_inside_the_plan_executor(self):
        registry = dict(_parts.REGISTRY)
        registry["obs-constant"] = _ConstantSource
        registry["out-scope"] = _ScopeSink
        executor = _executor.PlanExecutor(registry)
        executor.compile(
            [
                {"id": "src", "type": "obs-constant", "data": {"values": [1.0, 2.0, 3.0]}},
                {"id": "waypoint", "type": "proc-reroute", "data": {}},
                {"id": "split", "type": "proc-demux", "data": {"sizes": [1, 2]}},
                {"id": "join", "type": "proc-mux", "data": {"channels": 2}},
                {"id": "scope", "type": "out-scope", "data": {}},
            ],
            [
                {"id": "e1", "source": "src", "target": "waypoint",
                 "sourcePort": None, "targetPort": None},
                {"id": "e2", "source": "waypoint", "target": "split",
                 "sourcePort": None, "targetPort": None},
                {"id": "e3", "source": "split", "target": "join",
                 "sourcePort": "out-0", "targetPort": "in-0"},
                {"id": "e4", "source": "split", "target": "join",
                 "sourcePort": "out-1", "targetPort": "in-1"},
                {"id": "e5", "source": "join", "target": "scope",
                 "sourcePort": None, "targetPort": None},
            ],
        )
        executor.step()
        sink = executor.get_processor("scope")
        np.testing.assert_allclose(sink.received[0], [1.0, 2.0, 3.0])
        mux = executor.get_processor("join")
        self.assertEqual(mux.output_segments[0], [1, 2])


if __name__ == "__main__":
    unittest.main()
