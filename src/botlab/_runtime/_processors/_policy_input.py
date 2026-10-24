"""PolicyInputProcessor: policy observation gather, scale and history stacking."""

import math

import numpy as np

from ._modes_common import (
    _LIST_SPLIT,
    _dim_at,
    _fail,
    _input_at,
    _integer_value,
    _is_empty,
    _mode_from_input,
    _number,
    _required_integer,
)

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
