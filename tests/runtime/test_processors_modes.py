"""Mode-class processor tests (tier B: numpy required).

Mirrors the TypeScript processor semantics from
src/domain/runtime/engine/processors/ (ModeSwitchProcessor.test.ts,
StateMachineProcessor.test.ts, GenericSignalProcessors.test.ts): happy
paths with fixed configs and input sequences, reset behaviour, error
shapes, the historical proc-fsm / proc-select registry aliases, and the
plug-in contract with the local plan executor.
"""

import unittest

import numpy as np  # noqa: E402

from botlab._runtime import _executor  # noqa: E402
from botlab._runtime._processors import _modes  # noqa: E402


def scalar(value):
    return np.array([value], dtype=np.float32)


# ---------------------------------------------------------------------------
# ModeSwitchProcessor (proc-fsm / proc-mode-switch)
# ---------------------------------------------------------------------------


def make_mode_switch(transitions, initial=0):
    processor = _modes.ModeSwitchProcessor()
    processor.init({"initialMode": initial, "transitions": transitions}, [1])
    return processor


class ModeSwitchProcessorTest(unittest.TestCase):
    def test_follows_command_edges_from_filters_and_any(self):
        processor = make_mode_switch([
            {"from": [0], "command": 1, "to": 1},
            {"from": "1", "command": 2, "to": 2},  # string form equals [1]
            {"from": "any", "command": 3, "to": 0},
        ])
        command = np.zeros(1, dtype=np.float32)
        modes = []
        for code in (0, 1, 1, 2, 3, 2, 0, 1):
            command[0] = code
            processor.compute([command])
            modes.append(float(processor.output_buffers[0][0]))
        # Repeated commands are not edges; command 2 from mode 0 has no
        # matching from-filter; command 0 never transitions.
        self.assertEqual(modes, [0, 1, 1, 2, 0, 0, 0, 1])
        self.assertEqual(processor.output_buffers[0].dtype, np.float32)
        # The zero-copy command input is never written back.
        self.assertEqual(command.tolist(), [1.0])

    def test_reset_restores_initial_mode_and_rearms_command_edges(self):
        processor = make_mode_switch([{"from": "any", "command": 1, "to": 1}], initial=3)
        self.assertEqual(float(processor.output_buffers[0][0]), 3.0)
        processor.compute([scalar(1)])
        self.assertEqual(float(processor.output_buffers[0][0]), 1.0)
        processor.reset()
        self.assertEqual(float(processor.output_buffers[0][0]), 3.0)
        # lastCommandCode is cleared, so the same pulse is an edge again.
        processor.compute([scalar(1)])
        self.assertEqual(float(processor.output_buffers[0][0]), 1.0)

    def test_init_requires_scalar_command_input(self):
        processor = _modes.ModeSwitchProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init(
                {"initialMode": 0, "transitions": [{"command": 1, "to": 1}]}, [2]
            )
        self.assertIn("[ModeSwitchProcessor]", str(ctx.exception))
        self.assertIn("command input must have dimension 1, got 2", str(ctx.exception))

    def test_init_rejects_bad_initial_mode_and_command_codes(self):
        with self.assertRaises(ValueError) as ctx:
            make_mode_switch([{"command": 1, "to": 1}], initial=1.5)
        self.assertIn("initialMode must be an integer mode code", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_mode_switch([{"from": "any", "command": 0, "to": 1}])
        self.assertIn("transitions[0].command must be non-zero", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_mode_switch([{"from": "any", "to": 1}])
        self.assertIn("transitions[0].command is required", str(ctx.exception))

    def test_init_requires_non_empty_transitions(self):
        with self.assertRaises(ValueError) as ctx:
            make_mode_switch([])
        self.assertIn("transitions must be a non-empty array", str(ctx.exception))

    def test_compute_requires_command_input(self):
        processor = make_mode_switch([{"from": "any", "command": 1, "to": 1}])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("[ModeSwitchProcessor] Missing command input", str(ctx.exception))


# ---------------------------------------------------------------------------
# ModeSelectProcessor (proc-select / proc-mode-select)
# ---------------------------------------------------------------------------


def make_mode_select(slot_modes, dims=(1, 3, 3), passive=0):
    processor = _modes.ModeSelectProcessor()
    processor.init(
        {"passiveMode": passive, "slotModes": list(slot_modes)}, list(dims)
    )
    return processor


class ModeSelectProcessorTest(unittest.TestCase):
    def test_passive_zeros_mode_match_copies_and_unknown_falls_back_to_hold(self):
        processor = make_mode_select([1, 2])
        mode = np.zeros(1, dtype=np.float32)
        hold = np.array([7.0, 7.0, 7.0], dtype=np.float32)
        first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        second = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        for mode_code, expected in [
            (0, [0.0, 0.0, 0.0]),  # passive mode
            (1, [1.0, 2.0, 3.0]),  # first data slot
            (2, [4.0, 5.0, 6.0]),  # second data slot
            (9, [7.0, 7.0, 7.0]),  # unknown mode falls back to hold
        ]:
            mode[0] = mode_code
            processor.compute([mode, hold, first, second])
            self.assertEqual(processor.output_buffers[0].tolist(), expected)

        # Inputs are zero-copy aliases and stay untouched.
        self.assertEqual(hold.tolist(), [7.0, 7.0, 7.0])
        self.assertEqual(first.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(second.tolist(), [4.0, 5.0, 6.0])

    def test_missing_data_slot_breaks_to_hold(self):
        processor = make_mode_select([1], dims=(1, 2))
        processor.compute([
            scalar(1),
            np.array([5.0, 5.0], dtype=np.float32),
            None,
        ])
        self.assertEqual(processor.output_buffers[0].tolist(), [5.0, 5.0])

    def test_init_validates_input_dimensions(self):
        with self.assertRaises(ValueError) as ctx:
            make_mode_select([1], dims=(2, 3))
        self.assertIn("mode input must have dimension 1, got 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_mode_select([1], dims=(1,))
        self.assertIn("hold input must have a positive dimension; got 0", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_mode_select([1], dims=(1, 3, 4))
        self.assertIn("data inputs must all have dimension 3; slot 2 has 4", str(ctx.exception))

    def test_init_requires_slot_modes_for_every_connected_data_input(self):
        with self.assertRaises(ValueError) as ctx:
            make_mode_select([1], dims=(1, 3, 3, 3))
        self.assertIn(
            "slotModes has 1 entries but 2 data inputs are connected",
            str(ctx.exception),
        )

    def test_init_rejects_non_list_slot_modes(self):
        processor = _modes.ModeSelectProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.init({"passiveMode": 0, "slotModes": 5}, [1, 2])
        self.assertIn("slotModes must be a non-empty integer list", str(ctx.exception))

    def test_compute_detects_dimension_change(self):
        processor = make_mode_select([1], dims=(1, 2))
        processor.compute([scalar(0), np.zeros(2, dtype=np.float32)])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([scalar(0), np.zeros(3, dtype=np.float32)])
        self.assertIn("input dimensions changed after init", str(ctx.exception))

    def test_stateless_contract_has_no_reset(self):
        processor = make_mode_select([1])
        self.assertFalse(hasattr(processor, "reset"))


# ---------------------------------------------------------------------------
# StateMachineProcessor (proc-state-machine)
# ---------------------------------------------------------------------------


def make_state_machine(transitions, dims=(1,), initial=0):
    processor = _modes.StateMachineProcessor()
    processor.init({"initialState": initial, "transitions": transitions}, list(dims))
    return processor


class StateMachineProcessorTest(unittest.TestCase):
    @staticmethod
    def output(processor):
        return [
            float(processor.output_buffers[0][0]),
            float(processor.output_buffers[1][0]),
        ]

    def test_edge_triggered_timed_one_shot(self):
        processor = make_state_machine([
            {"from": [0], "command": 1, "to": 1},
            {"from": [1], "afterSteps": 3, "to": 0},
        ])
        for command, expected in [
            (0, [0, 0]),  # first compute does not tick elapsed steps
            (1, [1, 0]),
            (0, [1, 1]),
            (0, [1, 2]),
            (0, [0, 0]),  # afterSteps reached: automatic transition
        ]:
            processor.compute([scalar(command)])
            self.assertEqual(self.output(processor), expected)

    def test_conditions_and_preserved_steps_and_reset(self):
        processor = make_state_machine([
            {"from": [0], "command": 1, "to": 1},
            {"from": [1], "conditionSlot": 1, "to": 2, "resetSteps": False},
            {"from": [2], "afterSteps": 3, "conditionSlot": 2, "to": 0},
        ], dims=(1, 1, 1))

        def run(command, tipped, upright):
            processor.compute([
                scalar(command),
                scalar(tipped),
                scalar(upright),
            ])
            return self.output(processor)

        self.assertEqual(run(1, 0, 0), [1, 0])
        self.assertEqual(run(0, 0, 0), [1, 1])
        self.assertEqual(run(0, 1, 0), [2, 2])  # resetSteps=False keeps steps
        self.assertEqual(run(0, 0, 1), [0, 0])

        processor.reset()
        self.assertEqual(self.output(processor), [0, 0])
        self.assertEqual(run(0, 0, 0), [0, 0])  # no tick on the first step

    def test_command_edges_take_priority_over_automatic_transitions(self):
        processor = make_state_machine([
            {"from": [0], "command": 1, "to": 1},
            {"from": [0], "afterSteps": 0, "to": 9},
            {"from": [1], "command": 1, "to": 8},
            {"from": [1], "command": 2, "to": 2},
        ])
        processor.compute([scalar(1)])
        self.assertEqual(self.output(processor), [1, 0])
        processor.compute([scalar(1)])
        self.assertEqual(self.output(processor), [1, 1])
        processor.compute([scalar(2)])
        self.assertEqual(self.output(processor), [2, 0])

    def test_reset_rearms_command_edges(self):
        processor = make_state_machine([{"from": "any", "command": 1, "to": 1}])
        processor.compute([scalar(1)])
        processor.compute([scalar(1)])
        self.assertEqual(self.output(processor), [1, 1])
        processor.reset()
        processor.compute([scalar(1)])
        self.assertEqual(self.output(processor), [1, 0])

    def test_init_validates_command_and_condition_dimensions(self):
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"command": 1, "to": 1}], dims=(2,))
        self.assertIn("command input must have dimension 1, got 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"conditionSlot": 1, "to": 1}], dims=(1, 2))
        self.assertIn("condition input 1 must have dimension 1, got 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"conditionSlot": 3, "to": 1}], dims=(1, 1))
        self.assertIn(
            "transitions[0].conditionSlot must reference a connected condition input",
            str(ctx.exception),
        )

    def test_transition_trigger_validation(self):
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"from": [0], "to": 1}])
        self.assertIn(
            "transitions[0] requires command, afterSteps, or conditionSlot",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"command": 0, "to": 1}])
        self.assertIn("transitions[0].command must not be zero", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_state_machine([{"command": 1, "to": 1, "resetSteps": "yes"}])
        self.assertIn("transitions[0].resetSteps must be boolean", str(ctx.exception))

    def test_compute_rejects_non_integer_commands(self):
        processor = make_state_machine([{"command": 1, "to": 1}])
        for bad in (1.5, float("nan")):
            with self.assertRaises(ValueError) as ctx:
                processor.compute([np.array([bad], dtype=np.float32)])
            self.assertIn("command input must be a finite integer", str(ctx.exception))


# ---------------------------------------------------------------------------
# RuleIndexProcessor (proc-rule-index)
# ---------------------------------------------------------------------------

NESTED_RULES = [
    {
        "value": 2,
        "when": {
            "any": [
                {"index": 2, "op": "gt", "value": 0.1},
                {
                    "all": [
                        {"index": 0, "op": "gt", "value": 0.1},
                        {"index": 1, "op": "gt", "value": 0.1},
                    ],
                },
            ],
        },
    },
    {"value": 1, "when": {"index": 0, "op": "gte", "value": 0.1}},
]


def make_rule_index(rules=None, default=0, dim=3):
    processor = _modes.RuleIndexProcessor()
    processor.init(
        {"rules": NESTED_RULES if rules is None else rules, "defaultValue": default},
        [dim],
    )
    return processor


class RuleIndexProcessorTest(unittest.TestCase):
    def test_first_match_wins_and_default_falls_through(self):
        processor = make_rule_index()
        # The output starts at defaultValue before the first compute.
        self.assertEqual(float(processor.output_buffers[0][0]), 0.0)
        for values, expected in [
            ([1, 1, 0], 2),  # nested all branch matches
            ([0, 1, 5], 2),  # any branch matches on index 2
            ([1, 0, 0], 1),  # second rule
            ([0, 0, 0], 0),  # default
        ]:
            processor.compute([np.array(values, dtype=np.float32)])
            self.assertEqual(float(processor.output_buffers[0][0]), float(expected))

    def test_comparison_operators(self):
        cases = [
            ("gt", 2.0, 2.0, False),
            ("gt", 2.0, 2.1, True),
            ("gte", 2.0, 2.0, True),
            ("gte", 2.0, 1.9, False),
            ("lt", 2.0, 2.0, False),
            ("lt", 2.0, 1.9, True),
            ("lte", 2.0, 2.0, True),
            ("lte", 2.0, 2.1, False),
            ("eq", 2.0, 2.0, True),
            ("eq", 2.0, 2.5, False),
            ("ne", 2.0, 2.0, False),
            ("ne", 2.0, 2.5, True),
        ]
        for operator, threshold, sample, expected in cases:
            with self.subTest(operator=operator):
                processor = make_rule_index(
                    rules=[
                        {"value": 1, "when": {"index": 0, "op": operator, "value": threshold}}
                    ],
                    dim=1,
                )
                processor.compute([np.array([sample], dtype=np.float32)])
                self.assertEqual(
                    float(processor.output_buffers[0][0]), 1.0 if expected else 0.0
                )

    def test_init_validates_rules_and_conditions(self):
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(rules=[{"value": 1, "when": {"index": 3, "op": "eq", "value": 0}}])
        self.assertIn(
            "rules[0].when.index must be an integer between 0 and 2", str(ctx.exception)
        )
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(rules=[{"value": 1.5, "when": {"index": 0, "op": "eq", "value": 0}}])
        self.assertIn("rules[0].value must be an integer", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(
                rules=[{"value": 1, "when": {"index": 0, "op": "eq", "value": 0}}],
                default=0.5,
            )
        self.assertIn("defaultValue must be an integer", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(
                rules=[{"value": 1, "when": {"index": 0, "op": "eq", "value": 0, "x": 1}}]
            )
        self.assertIn(
            "rules[0].when leaf must contain exactly index, op, and value",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(rules=[{"value": 1, "when": {"index": 0, "op": "near", "value": 0}}])
        self.assertIn(
            "rules[0].when.op must be one of gt, gte, lt, lte, eq, ne",
            str(ctx.exception),
        )
        with self.assertRaises(ValueError) as ctx:
            make_rule_index(rules=[])
        self.assertIn("rules must be a non-empty list", str(ctx.exception))

    def test_compute_rejects_missing_changed_and_non_finite_inputs(self):
        processor = make_rule_index()
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("[RuleIndexProcessor] Missing input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.zeros(2, dtype=np.float32)])
        self.assertIn("input length changed from 3 to 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([np.array([0, np.nan, 0], dtype=np.float32)])
        self.assertIn("input[1] must be finite", str(ctx.exception))

    def test_stateless_contract_has_no_reset(self):
        self.assertFalse(hasattr(make_rule_index(), "reset"))


# ---------------------------------------------------------------------------
# PolicyInputProcessor (proc-policy-input)
# ---------------------------------------------------------------------------

SINGLE_POLICY_CONFIG = {
    "modeSlot": 0,
    "policies": [
        {
            "key": "walk",
            "mode": 1,
            "outputSlot": 0,
            "historyFrames": 1,
            "fields": [
                {
                    "kind": "mapped",
                    "inputSlot": 1,
                    "defaultValues": [1.0, 2.0, 3.0],
                    "scale": 2.0,
                },
                {"kind": "copy", "inputSlot": 2},
                {"kind": "previous-action", "inputSlot": 3, "length": 3},
            ],
        }
    ],
}


def make_policy_input(config, dims):
    processor = _modes.PolicyInputProcessor()
    processor.init(config, list(dims))
    return processor


class PolicyInputProcessorTest(unittest.TestCase):
    def test_single_policy_state_assembly(self):
        processor = make_policy_input(SINGLE_POLICY_CONFIG, [1, 3, 2, 3])
        mode = scalar(1)
        dof = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        command = np.array([0.5, -0.5], dtype=np.float32)
        action = np.array([9.0, 9.0, 9.0], dtype=np.float32)

        processor.compute([mode, dof, command, action])
        # mapped (source - default) * scale, copy, previous action of this step
        self.assertEqual(
            processor.output_buffers[0].tolist(),
            [8.0, 8.0, 8.0, 0.5, -0.5, 9.0, 9.0, 9.0],
        )

        action[:] = [1.0, 2.0, 3.0]
        processor.compute([mode, dof, command, action])
        self.assertEqual(
            processor.output_buffers[0].tolist(),
            [8.0, 8.0, 8.0, 0.5, -0.5, 1.0, 2.0, 3.0],
        )

        # An inactive mode keeps the last captured previous action.
        mode[0] = 2
        action[:] = [4.0, 5.0, 6.0]
        processor.compute([mode, dof, command, action])
        self.assertEqual(
            processor.output_buffers[0].tolist(),
            [8.0, 8.0, 8.0, 0.5, -0.5, 1.0, 2.0, 3.0],
        )
        # Fractional modes truncate like the TS Math.trunc path.
        mode[0] = 1.5
        action[:] = [4.0, 5.0, 6.0]
        processor.compute([mode, dof, command, action])
        self.assertEqual(
            processor.output_buffers[0].tolist(),
            [8.0, 8.0, 8.0, 0.5, -0.5, 4.0, 5.0, 6.0],
        )
        # Inputs are zero-copy aliases and stay untouched.
        self.assertEqual(dof.tolist(), [5.0, 6.0, 7.0])
        self.assertEqual(command.tolist(), [0.5, -0.5])

    def test_history_stacking_and_mode_change_refill(self):
        config = {
            "modeSlot": 0,
            "policies": [
                {
                    "key": "walk",
                    "mode": 1,
                    "historyFrames": 3,
                    "fields": [{"kind": "copy", "inputSlot": 1}],
                }
            ],
        }
        processor = make_policy_input(config, [1, 2])
        mode = scalar(1)
        source = np.zeros(2, dtype=np.float32)

        def step(state, mode_code=1):
            mode[0] = mode_code
            source[:] = state
            processor.compute([mode, source])
            return processor.output_buffers[0].tolist()

        self.assertEqual(step([1, 2]), [1, 2, 1, 2, 1, 2])  # first sample fills all frames
        self.assertEqual(step([3, 4]), [1, 2, 1, 2, 3, 4])
        self.assertEqual(step([5, 6]), [1, 2, 3, 4, 5, 6])   # oldest frame first
        # A foreign mode freezes the history but the output is still emitted.
        self.assertEqual(step([7, 8], mode_code=2), [1, 2, 3, 4, 5, 6])
        # Returning to the policy mode refills every frame.
        self.assertEqual(step([9, 10]), [9, 10, 9, 10, 9, 10])

        processor.reset()
        self.assertEqual(processor.output_buffers[0].tolist(), [0.0] * 6)
        self.assertEqual(step([11, 12]), [11, 12, 11, 12, 11, 12])

    def test_policy_bank_shares_sources_across_policies(self):
        config = {
            "modeSlot": 0,
            "policies": [
                {
                    "key": "walk",
                    "mode": 1,
                    "outputSlot": 0,
                    "historyFrames": 1,
                    "fields": [{"kind": "copy", "inputSlot": 1}],
                },
                {
                    "key": "run",
                    "mode": 2,
                    "outputSlot": 1,
                    "historyFrames": 2,
                    "referenceSlot": 2,
                    "fields": [{"kind": "copy", "inputSlot": 1, "length": 2}],
                },
            ],
        }
        processor = make_policy_input(config, [1, 3, 2])
        mode = scalar(1)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        reference = np.array([10.0, 20.0], dtype=np.float32)

        processor.compute([mode, state, reference])
        self.assertEqual(processor.output_buffers[0].tolist(), [1.0, 2.0, 3.0])
        # run is inactive: reference plus the zero-initialized history.
        self.assertEqual(
            processor.output_buffers[1].tolist(), [10.0, 20.0, 0.0, 0.0, 0.0, 0.0]
        )

        mode[0] = 2
        state[:] = [4.0, 5.0, 6.0]
        processor.compute([mode, state, reference])
        self.assertEqual(processor.output_buffers[0].tolist(), [4.0, 5.0, 6.0])
        # Mode change primes both history frames with the current state.
        self.assertEqual(
            processor.output_buffers[1].tolist(), [10.0, 20.0, 4.0, 5.0, 4.0, 5.0]
        )

    def test_reset_clears_previous_actions_and_mode_memory(self):
        processor = make_policy_input(SINGLE_POLICY_CONFIG, [1, 3, 2, 3])
        mode = scalar(1)
        dof = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        command = np.array([0.5, -0.5], dtype=np.float32)
        action = np.array([9.0, 9.0, 9.0], dtype=np.float32)

        processor.compute([mode, dof, command, action])
        self.assertEqual(processor.output_buffers[0].tolist()[-3:], [9.0, 9.0, 9.0])

        processor.reset()
        self.assertEqual(processor.output_buffers[0].tolist(), [0.0] * 8)
        # Inactive mode after reset: the previous action falls back to zeros.
        mode[0] = 2
        processor.compute([mode, dof, command, action])
        self.assertEqual(processor.output_buffers[0].tolist()[-3:], [0.0, 0.0, 0.0])

    def test_init_validates_mode_dimension_and_policy_list(self):
        with self.assertRaises(ValueError) as ctx:
            make_policy_input(SINGLE_POLICY_CONFIG, [2, 3, 2, 3])
        self.assertIn("mode input must have dimension 1, got 2", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            make_policy_input({"modeSlot": 0, "policies": []}, [1, 3])
        self.assertIn("policies must contain at least one policy", str(ctx.exception))

    def test_init_validates_field_connections_and_state_dim(self):
        config = {
            "modeSlot": 0,
            "policies": [
                {
                    "key": "walk",
                    "mode": 1,
                    "fields": [{"kind": "copy", "inputSlot": 5}],
                }
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            make_policy_input(config, [1, 3])
        self.assertIn(
            "policies[0].fields[0].inputSlot is not connected", str(ctx.exception)
        )
        config = {
            "modeSlot": 0,
            "policies": [
                {
                    "key": "walk",
                    "mode": 1,
                    "stateDim": 5,
                    "fields": [{"kind": "copy", "inputSlot": 1}],
                }
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            make_policy_input(config, [1, 3])
        self.assertIn(
            "policies[0].stateDim 5 does not match fields total 3", str(ctx.exception)
        )

    def test_indices_must_be_unique_and_within_the_input(self):
        for indices in ([0, 0], [3]):
            with self.subTest(indices=indices):
                config = {
                    "modeSlot": 0,
                    "policies": [
                        {
                            "key": "walk",
                            "mode": 1,
                            "fields": [
                                {"kind": "copy", "inputSlot": 1, "indices": indices}
                            ],
                        }
                    ],
                }
                with self.assertRaises(ValueError) as ctx:
                    make_policy_input(config, [1, 3])
                self.assertIn(
                    "policies[0].fields[0].indices must contain unique valid input indices",
                    str(ctx.exception),
                )

    def test_compute_requires_scalar_mode_input(self):
        processor = make_policy_input(SINGLE_POLICY_CONFIG, [1, 3, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            processor.compute([])
        self.assertIn("[PolicyInputProcessor] Missing mode input", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            processor.compute([
                np.array([np.nan], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ])
        self.assertIn("[PolicyInputProcessor] mode must be an integer", str(ctx.exception))


# ---------------------------------------------------------------------------
# Registry aliases and plan-executor plug-in contract
# ---------------------------------------------------------------------------


class RegistryAliasTest(unittest.TestCase):
    def test_registry_covers_all_mode_node_types_with_history_aliases(self):
        registry = _modes.REGISTRY
        self.assertIs(registry["proc-fsm"], _modes.ModeSwitchProcessor)
        self.assertIs(registry["proc-mode-switch"], _modes.ModeSwitchProcessor)
        self.assertIs(registry["proc-select"], _modes.ModeSelectProcessor)
        self.assertIs(registry["proc-mode-select"], _modes.ModeSelectProcessor)
        self.assertIs(registry["proc-state-machine"], _modes.StateMachineProcessor)
        self.assertIs(registry["proc-rule-index"], _modes.RuleIndexProcessor)
        self.assertIs(registry["proc-policy-input"], _modes.PolicyInputProcessor)

    def test_alias_keys_behave_identically(self):
        # proc-fsm / proc-select are the pre-rename ids of the mode
        # processors; instantiate through the registry and run one step.
        switch = _modes.REGISTRY["proc-fsm"]()
        switch.init(
            {
                "initialMode": 0,
                "transitions": [{"from": "any", "command": 1, "to": 1}],
            },
            [1],
        )
        switch.compute([scalar(1)])
        self.assertEqual(float(switch.output_buffers[0][0]), 1.0)

        select = _modes.REGISTRY["proc-select"]()
        select.init({"passiveMode": 0, "slotModes": [1]}, [1, 2, 2])
        select.compute([
            scalar(1),
            np.array([5.0, 5.0], dtype=np.float32),
            np.array([6.0, 7.0], dtype=np.float32),
        ])
        self.assertEqual(select.output_buffers[0].tolist(), [6.0, 7.0])


class ScriptedSource:
    """Test-only source emitting one scripted scalar per compute step."""

    def __init__(self):
        self.output_buffers = []
        self._script = []

    def init(self, config, input_dims):
        self._script = list(config["values"])
        self.output_buffers = [np.zeros(1, dtype=np.float32)]

    def compute(self, inputs):
        value = self._script.pop(0) if self._script else 0.0
        self.output_buffers[0][0] = value


class ConstantSource:
    """Test-only constant vector source."""

    def __init__(self):
        self.output_buffers = []

    def init(self, config, input_dims):
        self.output_buffers = [np.asarray(config["values"], dtype=np.float32).copy()]

    def compute(self, inputs):
        pass


class CaptureSink:
    """Test-only sink recording a copy of every received buffer."""

    def __init__(self):
        self.output_buffers = []
        self.received = []

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        self.received.append(np.array(inputs[0], dtype=np.float32))


class PlanExecutorIntegrationTest(unittest.TestCase):
    def test_mode_pipeline_runs_inside_the_plan_executor(self):
        registry = dict(_modes.REGISTRY)
        registry["test-source"] = ScriptedSource
        registry["test-constant"] = ConstantSource
        registry["out-scope"] = CaptureSink

        nodes = [
            {"id": "command", "type": "test-source", "data": {"values": [0, 1]}},
            {
                "id": "switch",
                "type": "proc-fsm",
                "data": {
                    "initialMode": 0,
                    "transitions": [{"from": [0], "command": 1, "to": 5}],
                },
            },
            {"id": "hold", "type": "test-constant", "data": {"values": [7.0, 7.0]}},
            {"id": "data", "type": "test-constant", "data": {"values": [9.0, 9.0]}},
            {
                "id": "select",
                "type": "proc-mode-select",
                "data": {"passiveMode": 0, "slotModes": [5]},
            },
            {"id": "sink", "type": "out-scope", "data": {}},
        ]
        edges = [
            {"id": "e1", "source": "command", "target": "switch"},
            {"id": "e2", "source": "switch", "target": "select", "targetPort": "in-0"},
            {"id": "e3", "source": "hold", "target": "select", "targetPort": "in-1"},
            {"id": "e4", "source": "data", "target": "select", "targetPort": "in-2"},
            {"id": "e5", "source": "select", "target": "sink"},
        ]

        executor = _executor.PlanExecutor(registry)
        executor.compile(nodes, edges)
        executor.step()
        executor.step()
        sink = executor.get_processor("sink")
        self.assertEqual(
            [values.tolist() for values in sink.received],
            [[0.0, 0.0], [9.0, 9.0]],  # passive zeros, then the matched data slot
        )

        executor.reset()
        executor.step()
        self.assertEqual(sink.received[-1].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
