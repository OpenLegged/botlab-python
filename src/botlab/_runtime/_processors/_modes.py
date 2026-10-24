"""Mode-class processors for the local plan executor (numpy hot path).

Python/numpy mirrors of the TypeScript mode processors under
src/domain/runtime/engine/processors/ (ModeSwitchProcessor.ts,
ModeSelectProcessor.ts, StateMachineProcessor.ts, RuleIndexProcessor.ts and
PolicyInputProcessor.ts). Transition tables, mode-code coercion, broadcast
and selection rules, history stacking and the error message shapes follow
the TypeScript sources; config keys stay camelCase exactly as the engine
reads them.

Deliberate differences, kept explicit per the executor contract:

- buffers are numpy float32 arrays; scalar math runs in float64 and stores
  back to float32, matching Float32Array read/assign semantics;
- ModeSwitchProcessor and PolicyInputProcessor implement reset() so the
  local runtime clears temporal state without recompiling; the TS versions
  have no reset hook and keep mode/history across engine resets;
- JS Number() coercion is mirrored for numbers, booleans and numeric
  strings; other shapes coerce to NaN and fail the integer checks loudly.
"""


import math

import numpy as np

from ._modes_common import (
    _LIST_SPLIT,
    _dim_at,
    _fail,
    _input_at,
    _integer_value,
    _is_empty,
    _is_integer_number,
    _number,
    _trunc,
)
from ._policy_input import PolicyInputProcessor

# ---------------------------------------------------------------------------
# ModeSwitchProcessor (proc-fsm / proc-mode-switch)
# ---------------------------------------------------------------------------

_MODE_SWITCH = "ModeSwitchProcessor"


def _mode_code(value, fallback, label):
    if _is_empty(value):
        return fallback
    parsed = _number(value)
    if not _is_integer_number(parsed):
        _fail(_MODE_SWITCH, f"{label} must be an integer mode code")
    return int(parsed)


def _switch_command_code(value, label):
    if _is_empty(value):
        _fail(_MODE_SWITCH, f"{label} is required")
    parsed = _number(value)
    if not _is_integer_number(parsed):
        _fail(_MODE_SWITCH, f"{label} must be an integer command code")
    if parsed == 0:
        _fail(_MODE_SWITCH, f"{label} must be non-zero")
    return int(parsed)


def _mode_set(value, label):
    """Parse a from-mode filter; None matches any source mode."""
    if _is_empty(value):
        return None
    if isinstance(value, str):
        if value == "any":
            return None
        values = [part for part in _LIST_SPLIT.split(value) if part]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    else:
        values = [value]
    if not values:
        return None
    return frozenset(
        _mode_code(item, 0, f"{label}[{index}]") for index, item in enumerate(values)
    )


def _parse_switch_transitions(config):
    raw_transitions = config.get("transitions")
    if not isinstance(raw_transitions, (list, tuple)) or not raw_transitions:
        _fail(_MODE_SWITCH, "transitions must be a non-empty array")
    transitions = []
    for index, raw in enumerate(raw_transitions):
        if not isinstance(raw, dict):
            _fail(_MODE_SWITCH, f"transitions[{index}] must be an object")
        transitions.append((
            _switch_command_code(raw.get("command"), f"transitions[{index}].command"),
            _mode_set(raw.get("from"), f"transitions[{index}].from"),
            _mode_code(raw.get("to"), 0, f"transitions[{index}].to"),
        ))
    return transitions


class ModeSwitchProcessor:
    """Persistent controller mode driven by edge-triggered command pulses."""

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._initial_mode = 0
        self._mode = 0
        self._last_command_code = 0.0
        self._transitions = []

    def init(self, config, input_dims):
        command_dim = _dim_at(input_dims, 0)
        if command_dim != 1:
            _fail(
                _MODE_SWITCH,
                f"command input must have dimension 1, got {command_dim}",
            )
        self._initial_mode = _mode_code(config.get("initialMode"), 0, "initialMode")
        self._mode = self._initial_mode
        self._last_command_code = 0.0
        self._transitions = _parse_switch_transitions(config)
        self.output_buffers[0][0] = self._mode

    def compute(self, inputs):
        command_input = _input_at(inputs, 0)
        if command_input is None or len(command_input) != 1:
            _fail(_MODE_SWITCH, "Missing command input")

        command_code = _trunc(float(command_input[0]))
        if command_code != self._last_command_code:
            self._last_command_code = command_code
            if command_code != 0:
                for transition_command, from_modes, to_mode in self._transitions:
                    if transition_command != command_code:
                        continue
                    if from_modes is not None and self._mode not in from_modes:
                        continue
                    self._mode = to_mode
                    break

        self.output_buffers[0][0] = self._mode

    def reset(self):
        """Restore the initial mode and re-arm command edges (config kept)."""
        self._mode = self._initial_mode
        self._last_command_code = 0.0
        self.output_buffers[0][0] = self._mode


# ---------------------------------------------------------------------------
# ModeSelectProcessor (proc-select / proc-mode-select)
# ---------------------------------------------------------------------------

_MODE_SELECT = "ModeSelectProcessor"


def _slot_modes_from_config(config):
    raw_slot_modes = config.get("slotModes")
    if not isinstance(raw_slot_modes, (list, tuple, np.ndarray)):
        _fail(_MODE_SELECT, "slotModes must be a non-empty integer list")
    raw_values = list(raw_slot_modes)
    if not raw_values:
        _fail(_MODE_SELECT, "slotModes must be a non-empty integer list")
    return [
        _integer_value(value, index + 1, f"slotModes[{index}]", _MODE_SELECT)
        for index, value in enumerate(raw_values)
    ]


class ModeSelectProcessor:
    """Broadcast the mode-matched data input, or the hold vector / zeros."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._dim = 0
        self._passive_mode = 0
        self._slot_modes = []

    def init(self, config, input_dims):
        mode_dim = _dim_at(input_dims, 0)
        hold_dim = _dim_at(input_dims, 1)

        if mode_dim != 1:
            _fail(
                _MODE_SELECT,
                f"mode input must have dimension 1, got {mode_dim}",
            )
        if not hold_dim:
            _fail(
                _MODE_SELECT,
                f"hold input must have a positive dimension; got {hold_dim}",
            )

        for slot in range(2, len(input_dims)):
            dim = input_dims[slot]
            if not dim:
                continue
            if dim != hold_dim:
                _fail(
                    _MODE_SELECT,
                    f"data inputs must all have dimension {hold_dim}; "
                    f"slot {slot} has {dim}",
                )

        self._dim = hold_dim
        self._passive_mode = _integer_value(
            config.get("passiveMode"), 0, "passiveMode", _MODE_SELECT
        )
        self._slot_modes = _slot_modes_from_config(config)
        connected_data_slots = max(0, len(input_dims) - 2)
        if connected_data_slots > len(self._slot_modes):
            _fail(
                _MODE_SELECT,
                f"slotModes has {len(self._slot_modes)} entries but "
                f"{connected_data_slots} data inputs are connected",
            )
        self.output_buffers[0] = np.zeros(self._dim, dtype=np.float32)

    def compute(self, inputs):
        mode_input = _input_at(inputs, 0)
        hold = _input_at(inputs, 1)
        if mode_input is None or hold is None:
            _fail(_MODE_SELECT, "Missing mode or hold input")
        if len(hold) != self._dim:
            _fail(_MODE_SELECT, "input dimensions changed after init")

        mode = _trunc(float(mode_input[0]))
        output = self.output_buffers[0]
        if mode == self._passive_mode:
            output[:] = 0.0
            return

        for slot in range(2, min(len(inputs), 2 + len(self._slot_modes))):
            if mode != self._slot_modes[slot - 2]:
                continue
            source = inputs[slot]
            if source is None:
                break
            if len(source) != self._dim:
                _fail(_MODE_SELECT, "input dimensions changed after init")
            output[:] = source
            return

        output[:] = hold


# ---------------------------------------------------------------------------
# StateMachineProcessor (proc-state-machine)
# ---------------------------------------------------------------------------

_STATE_MACHINE = "StateMachineProcessor"


class _StateTransition:
    __slots__ = (
        "from_states",
        "to_state",
        "command_code",
        "after_steps",
        "condition_slot",
        "reset_steps",
    )

    def __init__(
        self, from_states, to_state, command_code, after_steps, condition_slot, reset_steps
    ):
        self.from_states = from_states
        self.to_state = to_state
        self.command_code = command_code
        self.after_steps = after_steps
        self.condition_slot = condition_slot
        self.reset_steps = reset_steps


def _strict_integer(raw, field):
    """StateMachine config integers must be actual numbers, not strings."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
        _fail(_STATE_MACHINE, f"{field} must be an integer")
    value = float(raw)
    if not _is_integer_number(value):
        _fail(_STATE_MACHINE, f"{field} must be an integer")
    return int(value)


def _non_negative_integer(raw, field):
    value = _strict_integer(raw, field)
    if value < 0:
        _fail(_STATE_MACHINE, f"{field} must be non-negative")
    return value


def _state_set(raw, field):
    if isinstance(raw, str) and raw == "any":
        return None
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    if not values:
        _fail(_STATE_MACHINE, f"{field} must not be empty")
    return frozenset(
        _strict_integer(value, f"{field}[{index}]") for index, value in enumerate(values)
    )


def _parse_state_transitions(config, input_dims):
    raw_transitions = config.get("transitions")
    if not isinstance(raw_transitions, (list, tuple)) or not raw_transitions:
        _fail(_STATE_MACHINE, "transitions must be a non-empty list")
    transitions = []
    for index, raw in enumerate(raw_transitions):
        if not isinstance(raw, dict):
            _fail(_STATE_MACHINE, f"transitions[{index}] must be an object")
        label = f"transitions[{index}]"
        raw_from = raw.get("from")
        from_states = _state_set("any" if raw_from is None else raw_from, f"{label}.from")
        to_state = _strict_integer(raw.get("to"), f"{label}.to")

        # TS distinguishes a missing trigger key from an explicit null; only
        # a missing key leaves the trigger unset.
        if "command" not in raw:
            command_code = None
        else:
            command_code = _strict_integer(raw["command"], f"{label}.command")
            if command_code == 0:
                _fail(_STATE_MACHINE, f"{label}.command must not be zero")
        after_steps = (
            None
            if "afterSteps" not in raw
            else _non_negative_integer(raw["afterSteps"], f"{label}.afterSteps")
        )
        condition_slot = (
            None
            if "conditionSlot" not in raw
            else _non_negative_integer(raw["conditionSlot"], f"{label}.conditionSlot")
        )
        if condition_slot is not None:
            if condition_slot == 0 or condition_slot >= len(input_dims):
                _fail(
                    _STATE_MACHINE,
                    f"{label}.conditionSlot must reference a connected condition input",
                )
            condition_dim = input_dims[condition_slot]
            if condition_dim != 1:
                _fail(
                    _STATE_MACHINE,
                    f"condition input {condition_slot} must have dimension 1, "
                    f"got {condition_dim}",
                )
        if command_code is None and after_steps is None and condition_slot is None:
            _fail(_STATE_MACHINE, f"{label} requires command, afterSteps, or conditionSlot")
        if "resetSteps" in raw and not isinstance(raw["resetSteps"], bool):
            _fail(_STATE_MACHINE, f"{label}.resetSteps must be boolean")
        transitions.append(_StateTransition(
            from_states,
            to_state,
            command_code,
            after_steps,
            condition_slot,
            raw.get("resetSteps") is not False,
        ))
    return transitions


class StateMachineProcessor:
    """Finite-state machine with command edges, step counts and conditions."""

    def __init__(self):
        self.output_buffers = [
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]
        self._initial_state = 0
        self._state = 0
        self._elapsed_steps = 0
        self._last_command_code = 0.0
        self._has_computed = False
        self._transitions = []

    def init(self, config, input_dims):
        command_dim = _dim_at(input_dims, 0)
        if command_dim != 1:
            _fail(
                _STATE_MACHINE,
                f"command input must have dimension 1, got {command_dim}",
            )
        self._initial_state = _strict_integer(config.get("initialState"), "initialState")
        self._transitions = _parse_state_transitions(config, input_dims)
        self.reset()

    def _condition_value(self, inputs, slot):
        condition = _input_at(inputs, slot)
        if condition is None or len(condition) != 1:
            _fail(_STATE_MACHINE, f"condition input {slot} must be finite")
        value = float(condition[0])
        if not math.isfinite(value):
            _fail(_STATE_MACHINE, f"condition input {slot} must be finite")
        return value

    def _eligible(self, transition, prospective_steps, inputs):
        if transition.from_states is not None and self._state not in transition.from_states:
            return False
        if transition.after_steps is not None and prospective_steps < transition.after_steps:
            return False
        if transition.condition_slot is not None:
            if self._condition_value(inputs, transition.condition_slot) == 0:
                return False
        return True

    def compute(self, inputs):
        command_input = _input_at(inputs, 0)
        if command_input is None or len(command_input) != 1:
            _fail(_STATE_MACHINE, "Missing scalar command input")
        raw_command = float(command_input[0])
        if not _is_integer_number(raw_command):
            _fail(_STATE_MACHINE, "command input must be a finite integer")

        command_edge = raw_command != self._last_command_code
        self._last_command_code = raw_command
        prospective_steps = (
            self._elapsed_steps + 1 if self._has_computed else self._elapsed_steps
        )

        transition = None
        if command_edge and raw_command != 0:
            for candidate in self._transitions:
                if candidate.command_code == raw_command and self._eligible(
                    candidate, prospective_steps, inputs
                ):
                    transition = candidate
                    break
        if transition is None:
            for candidate in self._transitions:
                if candidate.command_code is None and self._eligible(
                    candidate, prospective_steps, inputs
                ):
                    transition = candidate
                    break

        if transition is not None:
            self._state = transition.to_state
            self._elapsed_steps = 0 if transition.reset_steps else prospective_steps
        else:
            self._elapsed_steps = prospective_steps
        self._has_computed = True
        self.output_buffers[0][0] = self._state
        self.output_buffers[1][0] = self._elapsed_steps

    def reset(self):
        """Return to the initial state with zero elapsed steps (config kept)."""
        self._state = self._initial_state
        self._elapsed_steps = 0
        self._last_command_code = 0.0
        self._has_computed = False
        self.output_buffers[0][0] = self._state
        self.output_buffers[1][0] = 0


# ---------------------------------------------------------------------------
# RuleIndexProcessor (proc-rule-index)
# ---------------------------------------------------------------------------

_RULE_INDEX = "RuleIndexProcessor"
_RULE_OPERATORS = ("gt", "gte", "lt", "lte", "eq", "ne")


def _finite_number(raw, field):
    value = _number(raw)
    if not math.isfinite(value):
        _fail(_RULE_INDEX, f"{field} must be finite")
    return value


def _rule_integer(raw, field):
    value = _number(raw)
    if not _is_integer_number(value):
        _fail(_RULE_INDEX, f"{field} must be an integer")
    return int(value)


def _parse_condition(raw, input_dim, label):
    if not isinstance(raw, dict):
        _fail(_RULE_INDEX, f"{label} must be a condition object")
    has_all = "all" in raw
    has_any = "any" in raw
    if has_all or has_any:
        if has_all == has_any or len(raw) != 1:
            _fail(_RULE_INDEX, f"{label} must contain exactly one all or any condition")
        key = "all" if has_all else "any"
        nested = raw[key]
        if not isinstance(nested, (list, tuple)) or not nested:
            _fail(_RULE_INDEX, f"{label}.{key} must be a non-empty condition list")
        return (
            key,
            tuple(
                _parse_condition(item, input_dim, f"{label}.{key}[{index}]")
                for index, item in enumerate(nested)
            ),
        )

    if len(raw) != 3 or not {"index", "op", "value"} <= set(raw):
        _fail(_RULE_INDEX, f"{label} leaf must contain exactly index, op, and value")

    index = _number(raw["index"])
    if not _is_integer_number(index) or index < 0 or index >= input_dim:
        _fail(
            _RULE_INDEX,
            f"{label}.index must be an integer between 0 and {input_dim - 1}",
        )
    operator = raw["op"]
    if not isinstance(operator, str) or operator not in _RULE_OPERATORS:
        _fail(_RULE_INDEX, f"{label}.op must be one of gt, gte, lt, lte, eq, ne")
    return ("leaf", int(index), operator, _finite_number(raw["value"], f"{label}.value"))


def _condition_matches(condition, values):
    kind = condition[0]
    if kind == "leaf":
        _, index, operator, expected = condition
        value = float(values[index])
        if operator == "gt":
            return value > expected
        if operator == "gte":
            return value >= expected
        if operator == "lt":
            return value < expected
        if operator == "lte":
            return value <= expected
        if operator == "eq":
            return value == expected
        return value != expected
    if kind == "all":
        return all(_condition_matches(nested, values) for nested in condition[1])
    return any(_condition_matches(nested, values) for nested in condition[1])


class RuleIndexProcessor:
    """Emit the value of the first matching numeric rule, else the default."""

    def __init__(self):
        self.output_buffers = [np.zeros(1, dtype=np.float32)]
        self._input_dim = 0
        self._rules = []
        self._default_value = 0

    def init(self, config, input_dims):
        self._input_dim = _dim_at(input_dims, 0)
        if not self._input_dim:
            _fail(_RULE_INDEX, "inputDims[0] is required")
        raw_rules = config.get("rules")
        if not isinstance(raw_rules, (list, tuple)) or not raw_rules:
            _fail(_RULE_INDEX, "rules must be a non-empty list")
        rules = []
        for index, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                _fail(_RULE_INDEX, f"rules[{index}] must be an object")
            if len(raw) != 2 or not {"value", "when"} <= set(raw):
                _fail(_RULE_INDEX, f"rules[{index}] must contain exactly value and when")
            rules.append((
                _rule_integer(raw["value"], f"rules[{index}].value"),
                _parse_condition(raw["when"], self._input_dim, f"rules[{index}].when"),
            ))
        self._rules = rules
        self._default_value = _rule_integer(config.get("defaultValue"), "defaultValue")
        self.output_buffers[0] = np.array([self._default_value], dtype=np.float32)

    def compute(self, inputs):
        source = _input_at(inputs, 0)
        if source is None:
            _fail(_RULE_INDEX, "Missing input")
        if len(source) != self._input_dim:
            _fail(
                _RULE_INDEX,
                f"input length changed from {self._input_dim} to {len(source)}",
            )
        finite = np.isfinite(source)
        if not finite.all():
            bad_index = int(np.flatnonzero(~finite)[0])
            _fail(_RULE_INDEX, f"input[{bad_index}] must be finite")

        for rule_value, condition in self._rules:
            if _condition_matches(condition, source):
                self.output_buffers[0][0] = rule_value
                return
        self.output_buffers[0][0] = self._default_value


# Historical internal plans can still carry the pre-rename type ids, so the
# registry keeps the aliases exactly like SimulationEngine.ts.
REGISTRY = {
    "proc-fsm": ModeSwitchProcessor,
    "proc-mode-switch": ModeSwitchProcessor,
    "proc-select": ModeSelectProcessor,
    "proc-mode-select": ModeSelectProcessor,
    "proc-state-machine": StateMachineProcessor,
    "proc-rule-index": RuleIndexProcessor,
    "proc-policy-input": PolicyInputProcessor,
}
