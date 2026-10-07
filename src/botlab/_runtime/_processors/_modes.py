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
import re

import numpy as np

_LIST_SPLIT = re.compile(r"[\s,]+")


def _fail(label, message):
    raise ValueError(f"[{label}] {message}")


def _is_empty(raw):
    """JS null / undefined / '' — the values TS treats as "not provided"."""
    return raw is None or (isinstance(raw, str) and raw == "")


def _number(raw):
    """Mirror JS Number() for plan config scalars; junk coerces to NaN."""
    if isinstance(raw, (bool, np.bool_)):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return float("nan")
    return float("nan")


def _is_integer_number(value):
    return math.isfinite(value) and value == math.floor(value)


def _trunc(value):
    """Mirror JS Math.trunc: NaN and infinities pass through unchanged."""
    if math.isnan(value) or math.isinf(value):
        return value
    return float(math.trunc(value))


def _integer_value(raw, fallback, field, label):
    if _is_empty(raw):
        return fallback
    value = _number(raw)
    if not _is_integer_number(value):
        _fail(label, f"{field} must be an integer")
    return int(value)


def _required_integer(raw, field, label):
    if _is_empty(raw):
        _fail(label, f"{field} is required")
    return _integer_value(raw, 0, field, label)


def _mode_from_input(buffer, label):
    """Read the mode code from a scalar input (fractions are truncated)."""
    mode = _trunc(float(buffer[0]))
    if not _is_integer_number(mode):
        _fail(label, "mode must be an integer")
    return int(mode)


def _dim_at(input_dims, slot):
    return input_dims[slot] if 0 <= slot < len(input_dims) else 0


def _input_at(inputs, slot):
    return inputs[slot] if 0 <= slot < len(inputs) else None


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


# ---------------------------------------------------------------------------
# PolicyInputProcessor (proc-policy-input)
# ---------------------------------------------------------------------------

_POLICY_INPUT = "PolicyInputProcessor"

_DEFAULT_KEYS = (
    "defaultValues",
    "default_values",
    "defaultPos",
    "default_pos",
    "defaultDofPos",
    "default_dof_pos",
    "defaults",
    "offset",
)


class _PolicyField:
    __slots__ = ("kind", "input_slot", "length", "gather", "defaults", "scale")

    def __init__(self, kind, input_slot, length, gather, defaults, scale):
        self.kind = kind
        self.input_slot = input_slot
        self.length = length
        self.gather = gather
        self.defaults = defaults
        self.scale = scale


class _PolicySpec:
    __slots__ = (
        "key",
        "mode",
        "output_slot",
        "reference_slot",
        "history_frames",
        "state_dim",
        "input_dim",
        "fields",
    )

    def __init__(
        self, key, mode, output_slot, reference_slot, history_frames, state_dim, input_dim, fields
    ):
        self.key = key
        self.mode = mode
        self.output_slot = output_slot
        self.reference_slot = reference_slot
        self.history_frames = history_frames
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.fields = fields


def _number_list(raw, label):
    """Scalar / string / list config value into a finite float list."""
    if _is_empty(raw):
        return None
    if isinstance(raw, (bool, np.bool_)):
        _fail(_POLICY_INPUT, f"{label} must be a number list")
    if isinstance(raw, (int, float, np.integer, np.floating)):
        value = float(raw)
        if not math.isfinite(value):
            _fail(_POLICY_INPUT, f"{label} must be finite")
        return [value]
    if isinstance(raw, str):
        parts = [part for part in _LIST_SPLIT.split(raw) if part]
        if not parts:
            return None
        values = [_number(part) for part in parts]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        values = [_number(item) for item in raw]
    else:
        _fail(_POLICY_INPUT, f"{label} must be a number list")
    if any(not math.isfinite(value) for value in values):
        _fail(_POLICY_INPUT, f"{label} must contain finite numbers")
    return values


def _fixed_list(raw, length, label):
    values = _number_list(raw, label)
    if values is None:
        return None
    if len(values) != length:
        _fail(_POLICY_INPUT, f"{label} must contain {length} values")
    return np.asarray(values, dtype=np.float32)


def _scale_value(raw, length, label):
    values = _number_list(raw, label)
    if values is None:
        return None
    if len(values) == 1:
        return values[0]
    if len(values) != length:
        _fail(_POLICY_INPUT, f"{label} must be scalar or contain {length} values")
    return np.asarray(values, dtype=np.float32)


def _indices_value(raw, input_dim, label):
    values = _number_list(raw, label)
    if values is None:
        return None
    truncated = np.trunc(np.asarray(values, dtype=np.float64))
    if (
        (truncated < 0).any()
        or (truncated >= input_dim).any()
        or np.unique(truncated).size != truncated.size
    ):
        _fail(_POLICY_INPUT, f"{label} must contain unique valid input indices")
    return truncated.astype(np.int64)


def _positive_integer(raw, fallback, label):
    value = _integer_value(raw, fallback, label, _POLICY_INPUT)
    if value <= 0:
        _fail(_POLICY_INPUT, f"{label} must be positive")
    return value


def _first_present(mapping, keys):
    """Mirror the TS `a ?? b ?? ...` config chains (None continues the chain)."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _parse_policy_field(raw, policy_index, field_index, input_dims):
    label = f"policies[{policy_index}].fields[{field_index}]"
    if not isinstance(raw, dict):
        _fail(_POLICY_INPUT, f"{label} must be an object")
    kind = raw.get("kind")
    if kind not in ("mapped", "previous-action"):
        kind = "copy"

    raw_slot = raw.get("inputSlot")
    if raw_slot is None:
        raw_slot = raw.get("slot")
    input_slot = _required_integer(raw_slot, f"{label}.inputSlot", _POLICY_INPUT)
    if input_slot < 0 or not _dim_at(input_dims, input_slot):
        _fail(_POLICY_INPUT, f"{label}.inputSlot is not connected")
    input_dim = input_dims[input_slot]

    indices = _indices_value(
        _first_present(raw, ("indices", "indexMap")), input_dim, f"{label}.indices"
    )
    if indices is not None:
        length = int(indices.size)
        gather = indices
    else:
        length = _positive_integer(raw.get("length"), input_dim, f"{label}.length")
        if length > input_dim:
            _fail(_POLICY_INPUT, f"{label}.length exceeds input dimension")
        gather = np.arange(length, dtype=np.int64)

    defaults = _fixed_list(_first_present(raw, _DEFAULT_KEYS), input_dim, f"{label}.defaults")
    scale = _scale_value(raw.get("scale"), input_dim, f"{label}.scale")
    return _PolicyField(kind, input_slot, length, gather, defaults, scale)


def _parse_policy(raw, index, input_dims):
    label = f"policies[{index}]"
    if not isinstance(raw, dict):
        _fail(_POLICY_INPUT, f"{label} must be an object")
    raw_key = raw.get("key")
    if raw_key is None:
        raw_key = raw.get("label")
    if raw_key is None:
        raw_key = f"policy-{index}"
    key = str(raw_key)
    mode = _required_integer(raw.get("mode"), f"{label}.mode", _POLICY_INPUT)

    raw_output = raw.get("outputSlot")
    if raw_output is None:
        raw_output = raw.get("output")
    if raw_output is None:
        raw_output = index
    output_slot = _integer_value(raw_output, index, f"{label}.outputSlot", _POLICY_INPUT)
    if output_slot < 0:
        _fail(_POLICY_INPUT, f"{label}.outputSlot must be non-negative")

    raw_reference = raw.get("referenceSlot")
    if _is_empty(raw_reference):
        reference_slot = None
    else:
        reference_slot = _required_integer(
            raw_reference, f"{label}.referenceSlot", _POLICY_INPUT
        )
    if reference_slot is not None and (
        reference_slot < 0 or not _dim_at(input_dims, reference_slot)
    ):
        _fail(_POLICY_INPUT, f"{label}.referenceSlot is not connected")
    reference_dim = 0 if reference_slot is None else input_dims[reference_slot]
    history_frames = _positive_integer(
        raw.get("historyFrames"), 1, f"{label}.historyFrames"
    )

    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
        _fail(_POLICY_INPUT, f"{label}.fields must contain at least one field")
    fields = [
        _parse_policy_field(field, index, field_index, input_dims)
        for field_index, field in enumerate(raw_fields)
    ]
    state_dim = sum(field.length for field in fields)
    raw_state_dim = raw.get("stateDim")
    if _is_empty(raw_state_dim):
        configured_state_dim = state_dim
    else:
        configured_state_dim = _positive_integer(
            raw_state_dim, state_dim, f"{label}.stateDim"
        )
    if configured_state_dim != state_dim:
        _fail(
            _POLICY_INPUT,
            f"{label}.stateDim {configured_state_dim} does not match "
            f"fields total {state_dim}",
        )
    return _PolicySpec(
        key,
        mode,
        output_slot,
        reference_slot,
        history_frames,
        state_dim,
        reference_dim + state_dim * history_frames,
        fields,
    )


class PolicyInputProcessor:
    """Assemble per-policy input tensors from shared, mode-gated sources."""

    def __init__(self):
        self.output_buffers = []
        self._mode_slot = 0
        self._policies = []
        self._histories = {}
        self._history_initialized = {}
        self._previous_actions = {}
        self._state_scratch = {}
        self._last_mode = float("nan")

    def init(self, config, input_dims):
        self._mode_slot = _integer_value(
            config.get("modeSlot"), 0, "modeSlot", _POLICY_INPUT
        )
        mode_dim = _dim_at(input_dims, self._mode_slot)
        if mode_dim != 1:
            _fail(_POLICY_INPUT, f"mode input must have dimension 1, got {mode_dim}")

        raw_policies = config.get("policies")
        if not isinstance(raw_policies, (list, tuple)) or not raw_policies:
            _fail(_POLICY_INPUT, "policies must contain at least one policy")
        self._policies = [
            _parse_policy(raw, index, input_dims)
            for index, raw in enumerate(raw_policies)
        ]

        max_output_slot = max((policy.output_slot for policy in self._policies), default=0)
        self.output_buffers = [
            np.zeros(0, dtype=np.float32) for _ in range(max_output_slot + 1)
        ]
        self._histories = {}
        self._history_initialized = {}
        self._previous_actions = {}
        self._state_scratch = {}
        self._last_mode = float("nan")

        for policy in self._policies:
            self.output_buffers[policy.output_slot] = np.zeros(
                policy.input_dim, dtype=np.float32
            )
            if policy.history_frames > 1:
                self._histories[policy.key] = np.zeros(
                    policy.state_dim * policy.history_frames, dtype=np.float32
                )
                self._history_initialized[policy.key] = False
            for field in policy.fields:
                if field.kind == "previous-action" and policy.key not in self._previous_actions:
                    self._previous_actions[policy.key] = np.zeros(
                        field.length, dtype=np.float32
                    )

    def _capture_previous_action(self, policy, inputs):
        for field in policy.fields:
            if field.kind != "previous-action":
                continue
            source = _input_at(inputs, field.input_slot)
            if source is None or len(source) < field.length:
                _fail(
                    _POLICY_INPUT,
                    f'previous action input for "{policy.key}" is missing or too short',
                )
            target = self._previous_actions.get(policy.key)
            if target is None:
                target = np.zeros(field.length, dtype=np.float32)
                self._previous_actions[policy.key] = target
            target[:] = source[: field.length]

    def _build_state(self, policy, inputs):
        state = self._state_scratch.get(policy.key)
        if state is None or len(state) != policy.state_dim:
            state = np.zeros(policy.state_dim, dtype=np.float32)
            self._state_scratch[policy.key] = state
        offset = 0
        for field in policy.fields:
            if field.kind == "previous-action":
                previous = self._previous_actions.get(policy.key)
                if previous is None:
                    previous = np.zeros(field.length, dtype=np.float32)
                state[offset : offset + field.length] = previous
                offset += field.length
                continue

            source = _input_at(inputs, field.input_slot)
            if source is None:
                _fail(
                    _POLICY_INPUT,
                    f'Missing input slot {field.input_slot} for policy "{policy.key}"',
                )
            # Scalar math in float64, stored back to float32 like the TS
            # Float32Array writes; gather keeps mapped defaults/scale aligned
            # with the source indices.
            values = np.asarray(source, dtype=np.float64)[field.gather]
            if field.kind == "mapped" and field.defaults is not None:
                values = values - field.defaults[field.gather]
            if field.scale is not None:
                if isinstance(field.scale, np.ndarray):
                    values = values * field.scale[field.gather]
                else:
                    values = values * field.scale
            state[offset : offset + field.length] = values
            offset += field.length
        return state

    def _write_history(self, policy, active_mode, mode_changed, state, output, output_offset):
        history = self._histories.get(policy.key)
        if history is None:
            _fail(
                _POLICY_INPUT,
                f'history buffer for policy "{policy.key}" is missing',
            )
        if active_mode == policy.mode:
            initialized = self._history_initialized.get(policy.key, False)
            if mode_changed or not initialized:
                history.reshape(policy.history_frames, policy.state_dim)[:] = state
                self._history_initialized[policy.key] = True
            else:
                history[: -policy.state_dim] = history[policy.state_dim :]
                history[-policy.state_dim :] = state
        output[output_offset : output_offset + len(history)] = history

    def compute(self, inputs):
        mode_input = _input_at(inputs, self._mode_slot)
        if mode_input is None or len(mode_input) != 1:
            _fail(_POLICY_INPUT, "Missing mode input")
        active_mode = _mode_from_input(mode_input, _POLICY_INPUT)
        mode_changed = active_mode != self._last_mode
        self._last_mode = active_mode

        for policy in self._policies:
            if policy.mode == active_mode:
                self._capture_previous_action(policy, inputs)

        for policy in self._policies:
            output = self.output_buffers[policy.output_slot]
            output_offset = 0
            if policy.reference_slot is not None:
                reference = _input_at(inputs, policy.reference_slot)
                if reference is None:
                    _fail(
                        _POLICY_INPUT,
                        f'Missing reference input for policy "{policy.key}"',
                    )
                output[: len(reference)] = reference
                output_offset = len(reference)

            state = self._build_state(policy, inputs)
            if policy.history_frames > 1:
                self._write_history(
                    policy, active_mode, mode_changed, state, output, output_offset
                )
            else:
                output[output_offset : output_offset + policy.state_dim] = state

    def reset(self):
        """Clear history, previous actions and mode memory (config kept)."""
        for history in self._histories.values():
            history[:] = 0.0
        self._history_initialized = {key: False for key in self._histories}
        for previous in self._previous_actions.values():
            previous[:] = 0.0
        self._state_scratch = {}
        self._last_mode = float("nan")
        for policy in self._policies:
            self.output_buffers[policy.output_slot][:] = 0.0


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
