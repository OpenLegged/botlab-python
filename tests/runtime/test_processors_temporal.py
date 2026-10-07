"""Temporal processor tests: history stacking and signal delay (numpy ports).

Mirrors the TypeScript processor tests
(src/domain/runtime/__tests__/HistoryProcessor.test.ts and
src/domain/runtime/engine/processors/GenericSignalProcessors.test.ts):
stacking layouts, sampling offsets, delay age compensation, priming, reset
semantics and the explicit error shapes.
"""

import re
import unittest

import numpy as np  # noqa: E402

from botlab._runtime._processors import _temporal  # noqa: E402


def f32(*values):
    return np.asarray(values, dtype=np.float32)


def expect_error(test_case, action, message):
    with test_case.assertRaisesRegex(ValueError, re.escape(message)):
        action()


class HistoryProcessorTest(unittest.TestCase):
    def output(self, processor):
        return processor.output_buffers[0].tolist()

    def test_fills_history_with_first_sample(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 3, "order": "oldest-first", "stackMode": "isaacgym"}, [2])

        processor.compute([f32(1, 10)])

        self.assertEqual(self.output(processor), [1, 10, 1, 10, 1, 10])

    def test_oldest_first_wraps_ring_across_steps(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 3, "order": "oldest-first", "stackMode": "isaacgym"}, [3])

        inputs = [
            f32(1, 2, 3),
            f32(4, 5, 6),
            f32(7, 8, 9),
            f32(10, 11, 12),
            f32(13, 14, 15),
            f32(16, 17, 18),
            f32(19, 20, 21),
        ]
        outputs = []
        for source in inputs:
            processor.compute([source])
            outputs.append(self.output(processor))

        self.assertEqual(outputs[1], [1, 2, 3, 1, 2, 3, 4, 5, 6])
        self.assertEqual(outputs[6], [13, 14, 15, 16, 17, 18, 19, 20, 21])

    def test_newest_first_order(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 3, "order": "newest-first", "stackMode": "isaacgym"}, [2])

        processor.compute([f32(1, 10)])
        processor.compute([f32(2, 20)])

        self.assertEqual(self.output(processor), [2, 20, 1, 10, 1, 10])

    def test_isaaclab_layout_uses_explicit_input_segments(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {"stackSize": 2, "order": "oldest-first", "stackMode": "isaaclab", "inputSegments": [1, 2]},
            [3],
        )

        processor.compute([f32(1, 2, 3)])
        self.assertEqual(self.output(processor), [1, 1, 2, 3, 2, 3])
        processor.compute([f32(4, 5, 6)])
        self.assertEqual(self.output(processor), [1, 4, 2, 3, 5, 6])

    def test_isaaclab_layout_falls_back_to_upstream_segments(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {
                "stackSize": 2,
                "order": "oldest-first",
                "stackMode": "isaaclab",
                "upstreamInputSegments": [[1, 2]],
            },
            [3],
        )

        processor.compute([f32(1, 2, 3)])
        processor.compute([f32(4, 5, 6)])

        self.assertEqual(self.output(processor), [1, 4, 2, 3, 5, 6])

    def test_sample_offsets_read_from_larger_source_buffer(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {
                "stackSize": 3,
                "sourceBufferSize": 5,
                "sampleOffsets": [-5, -3, -1],
                "order": "oldest-first",
                "stackMode": "isaacgym",
            },
            [1],
        )

        for value in range(1, 6):
            processor.compute([f32(value)])

        self.assertEqual(self.output(processor), [1, 3, 5])

    def test_sample_offsets_cover_g1_depth_window(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {
                "stackSize": 8,
                "sourceBufferSize": 37,
                "sampleOffsets": [-36, -31, -26, -21, -16, -11, -6, -1],
                "order": "oldest-first",
                "stackMode": "isaacgym",
            },
            [1],
        )

        for frame in range(1, 38):
            processor.compute([f32(frame)])

        self.assertEqual(self.output(processor), [2, 7, 12, 17, 22, 27, 32, 37])

    def test_sample_decimation_skips_unsampled_steps(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {"stackSize": 2, "sampleDecimation": 2, "order": "oldest-first", "stackMode": "isaacgym"},
            [1],
        )

        outputs = []
        for value in (1, 2, 3, 4):
            processor.compute([f32(value)])
            outputs.append(self.output(processor))

        self.assertEqual(outputs, [[1, 1], [1, 1], [1, 3], [1, 3]])

    def test_initial_fill_value_seeds_offset_history(self):
        processor = _temporal.HistoryProcessor()
        processor.init(
            {
                "stackSize": 1,
                "sourceBufferSize": 3,
                "sampleOffsets": [-3],
                "order": "oldest-first",
                "stackMode": "isaacgym",
                "initialFillValue": 0,
            },
            [1],
        )

        outputs = []
        for value in (1, 2, 3, 4):
            processor.compute([f32(value)])
            outputs.append(self.output(processor))

        self.assertEqual(outputs, [[0], [0], [0], [2]])

    def test_reset_returns_to_initial_state(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 2, "order": "oldest-first", "stackMode": "isaacgym"}, [1])
        processor.compute([f32(1)])
        processor.compute([f32(2)])
        self.assertEqual(self.output(processor), [1, 2])

        processor.reset()
        self.assertEqual(self.output(processor), [0, 0])

        processor.compute([f32(3)])
        self.assertEqual(self.output(processor), [3, 3])

    def test_stack_mode_defaults_to_isaacgym_like_typescript(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 2, "order": "oldest-first"}, [2])

        processor.compute([f32(1, 10)])
        processor.compute([f32(2, 20)])

        self.assertEqual(self.output(processor), [1, 10, 2, 20])

    def test_compute_does_not_mutate_inputs(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 2, "order": "oldest-first", "stackMode": "isaacgym"}, [2])
        source = f32(1, 10)
        snapshot = source.copy()

        processor.compute([source])
        processor.compute([f32(2, 20)])

        self.assertEqual(source.tolist(), snapshot.tolist())

    def test_missing_input_raises(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 2, "order": "oldest-first", "stackMode": "isaacgym"}, [2])

        expect_error(self, lambda: processor.compute([]), "[HistoryProcessor] Missing input")

    def test_input_length_mismatch_raises(self):
        processor = _temporal.HistoryProcessor()
        processor.init({"stackSize": 2, "order": "oldest-first", "stackMode": "isaacgym"}, [2])

        expect_error(
            self,
            lambda: processor.compute([f32(1, 2, 3)]),
            "[HistoryProcessor] input length 3 does not match itemSize 2",
        )

    def test_invalid_stack_size_raises(self):
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {"stackSize": 0, "order": "oldest-first"}, [2]
            ),
            "[HistoryProcessor] stackSize must be a positive integer",
        )
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {"stackSize": 1.5, "order": "oldest-first"}, [2]
            ),
            "[HistoryProcessor] stackSize must be a positive integer",
        )

    def test_source_buffer_size_must_cover_stack_size(self):
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {"stackSize": 3, "sourceBufferSize": 2, "order": "oldest-first"}, [2]
            ),
            "[HistoryProcessor] sourceBufferSize must be an integer greater than or equal to stackSize",
        )

    def test_invalid_order_and_stack_mode_raise(self):
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {"stackSize": 2, "order": "middle-out"}, [2]
            ),
            '[HistoryProcessor] order must be "oldest-first" or "newest-first"',
        )
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {"stackSize": 2, "stackMode": "isaacorchard"}, [2]
            ),
            '[HistoryProcessor] stackMode must be "isaacgym" or "isaaclab"',
        )

    def test_sample_offsets_validation_errors(self):
        def build(offsets):
            return lambda: _temporal.HistoryProcessor().init(
                {
                    "stackSize": 2,
                    "sourceBufferSize": 5,
                    "sampleOffsets": offsets,
                    "order": "oldest-first",
                },
                [1],
            )

        expect_error(
            self,
            build([-1]),
            "[HistoryProcessor] sampleOffsets length 1 must match stackSize 2",
        )
        expect_error(
            self,
            build([0, -1]),
            "[HistoryProcessor] sampleOffsets must contain non-zero integers",
        )
        expect_error(
            self,
            build([-6, -1]),
            "[HistoryProcessor] sampleOffsets must be within sourceBufferSize 5",
        )
        expect_error(
            self,
            build("soon"),
            "[HistoryProcessor] sampleOffsets must be a number array",
        )

    def test_input_segments_total_must_match_item_size(self):
        expect_error(
            self,
            lambda: _temporal.HistoryProcessor().init(
                {
                    "stackSize": 2,
                    "stackMode": "isaaclab",
                    "inputSegments": [1, 3],
                    "order": "oldest-first",
                },
                [3],
            ),
            "[HistoryProcessor] inputSegments total 4 does not match itemSize 3",
        )


class DelayProcessorTest(unittest.TestCase):
    def output(self, processor):
        return processor.output_buffers[0].tolist()

    def test_subtracts_clamped_age_from_total_delay(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 2, "fill": -1}, [1, 1])

        processor.compute([f32(10), f32(0)])
        self.assertEqual(self.output(processor), [-1])
        processor.compute([f32(20), f32(0)])
        self.assertEqual(self.output(processor), [-1])
        processor.compute([f32(30), f32(0)])
        self.assertEqual(self.output(processor), [10])

        processor.compute([f32(40), f32(1)])
        self.assertEqual(self.output(processor), [30])
        processor.compute([f32(50), f32(20)])
        self.assertEqual(self.output(processor), [50])

    def test_primes_missing_history_with_first_sample(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 7, "fill": 0, "primeWithFirstSample": True}, [1, 1])

        outputs = []
        for index in range(9):
            processor.compute([f32(index + 1), f32(0)])
            outputs.append(processor.output_buffers[0][0])
        self.assertEqual(outputs, [1, 1, 1, 1, 1, 1, 1, 1, 2])

        processor.reset()
        processor.compute([f32(10), f32(1)])
        self.assertEqual(self.output(processor), [10])

    def test_primes_multidimensional_history(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 2, "fill": 0, "primeWithFirstSample": True}, [2])

        processor.compute([f32(1, 2)])
        self.assertEqual(self.output(processor), [1, 2])
        processor.compute([f32(3, 4)])
        self.assertEqual(self.output(processor), [1, 2])
        processor.compute([f32(5, 6)])
        self.assertEqual(self.output(processor), [1, 2])
        processor.compute([f32(7, 8)])
        self.assertEqual(self.output(processor), [3, 4])

    def test_reset_fills_outputs_with_fill_value(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 1, "fill": [7, 8]}, [2])
        self.assertEqual(self.output(processor), [7, 8])

        processor.compute([f32(1, 2)])
        self.assertEqual(self.output(processor), [7, 8])
        processor.compute([f32(3, 4)])

        processor.reset()
        self.assertEqual(self.output(processor), [7, 8])
        self.assertEqual(processor.output_segments, [[2]])

    def test_zero_delay_passes_current_sample(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 0, "fill": 0}, [2])

        processor.compute([f32(1, 2)])
        self.assertEqual(self.output(processor), [1, 2])
        processor.compute([f32(3, 4)])
        self.assertEqual(self.output(processor), [3, 4])

    def test_output_segments_propagated_from_upstream(self):
        processor = _temporal.DelayProcessor()
        processor.init(
            {"totalSteps": 1, "fill": 0, "upstreamInputSegments": [[2, 3]]}, [5]
        )
        self.assertEqual(processor.output_segments, [[2, 3]])

        fallback = _temporal.DelayProcessor()
        fallback.init({"totalSteps": 1, "fill": 0}, [5])
        self.assertEqual(fallback.output_segments, [[5]])

    def test_compute_does_not_mutate_inputs(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 2, "fill": 0, "primeWithFirstSample": True}, [2])
        source = f32(1, 2)
        snapshot = source.copy()

        processor.compute([source])

        self.assertEqual(source.tolist(), snapshot.tolist())

    def test_total_steps_must_be_non_negative_integer(self):
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": -1, "fill": 0}, [2]),
            "[DelayProcessor] totalSteps must be a non-negative integer",
        )
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": 1.5, "fill": 0}, [2]),
            "[DelayProcessor] totalSteps must be a non-negative integer",
        )

    def test_fill_validation(self):
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": 1}, [2]),
            "[DelayProcessor] fill is required",
        )
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": 1, "fill": [0]}, [2]),
            "[DelayProcessor] fill must contain exactly 2 finite numbers",
        )
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": 1, "fill": "zero"}, [2]),
            "[DelayProcessor] fill must be a finite number or a 2-value array",
        )

    def test_prime_flag_must_be_boolean(self):
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init(
                {"totalSteps": 1, "fill": 0, "primeWithFirstSample": "yes"}, [2]
            ),
            "[DelayProcessor] primeWithFirstSample must be a boolean",
        )

    def test_age_input_must_be_non_negative_integer(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 1, "fill": 0}, [2, 1])

        expect_error(
            self,
            lambda: processor.compute([f32(1, 2), f32(0.5)]),
            "[DelayProcessor] age input must be a non-negative integer",
        )
        expect_error(
            self,
            lambda: processor.compute([f32(1, 2), f32(-1)]),
            "[DelayProcessor] age input must be a non-negative integer",
        )

    def test_age_input_dimension_validated_at_init(self):
        expect_error(
            self,
            lambda: _temporal.DelayProcessor().init({"totalSteps": 1, "fill": 0}, [2, 2]),
            "[DelayProcessor] age input must have dimension 1 when connected",
        )

    def test_source_length_change_and_missing_source_raise(self):
        processor = _temporal.DelayProcessor()
        processor.init({"totalSteps": 1, "fill": 0}, [2])

        expect_error(
            self,
            lambda: processor.compute([f32(1, 2, 3)]),
            "[DelayProcessor] source input length changed from 2 to 3",
        )
        expect_error(
            self,
            lambda: processor.compute([]),
            "[DelayProcessor] Missing source input",
        )


class RegistryTest(unittest.TestCase):
    def test_registry_exposes_temporal_node_types(self):
        self.assertEqual(
            _temporal.REGISTRY,
            {
                "proc-history": _temporal.HistoryProcessor,
                "proc-delay": _temporal.DelayProcessor,
            },
        )


if __name__ == "__main__":
    unittest.main()
