"""Policy banks: exactly one ONNX policy inferred per controller mode."""
from ._declaration import _non_empty_string, _plan_reference
from ._resources import Model, _model_call_data
from ._signal import Signal, _require_signal
from ._vectors import _non_negative_integer, _positive_integer


_POLICY_FIELD_KINDS = ("copy", "mapped", "previous-action")


def _policy_key(value, label):
    return _non_empty_string(value, label)


def _policy_mode(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def policy_inputs(mode, inputs, policies):
    """Build the distinct input tensor for every policy in a policy bank.

    ``inputs`` names shared source signals. Policy fields refer to those names
    with ``source`` instead of exposing engine input-slot numbers.
    """
    mode = _require_signal(mode, "policy_inputs mode")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("policy_inputs inputs must be a non-empty mapping")
    source_names = []
    source_signals = []
    for name, source in inputs.items():
        name = _policy_key(name, "policy_inputs source name")
        if name in source_names:
            raise ValueError(f'policy_inputs source "{name}" is duplicated')
        source_names.append(name)
        source_signals.append(_require_signal(source, f'policy_inputs source "{name}"'))
    source_slots = {name: index + 1 for index, name in enumerate(source_names)}

    items = list(policies)
    if not items:
        raise ValueError("policy_inputs policies must not be empty")
    normalized = []
    keys = []
    for policy_index, policy in enumerate(items):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_inputs policy {policy_index} must be a mapping")
        unknown = set(policy) - {
            "key",
            "mode",
            "reference",
            "history_frames",
            "state_dim",
            "fields",
        }
        if unknown:
            raise ValueError(
                f"policy_inputs policy {policy_index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_inputs policy {policy_index} key")
        if key in keys:
            raise ValueError(f'policy_inputs policy key "{key}" is duplicated')
        keys.append(key)
        fields = policy.get("fields")
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError(f'policy_inputs policy "{key}" fields must not be empty')
        normalized_fields = []
        for field_index, field in enumerate(fields):
            if isinstance(field, str):
                field = {"source": field}
            if not isinstance(field, dict):
                raise TypeError(
                    f'policy_inputs policy "{key}" field {field_index} '
                    "must be a source name or mapping"
                )
            unknown_fields = set(field) - {
                "source",
                "kind",
                "length",
                "indices",
                "default",
                "scale",
            }
            if unknown_fields:
                raise ValueError(
                    f'policy_inputs policy "{key}" field {field_index} has unknown '
                    f"fields {sorted(unknown_fields)}"
                )
            source_name = _policy_key(
                field.get("source"),
                f'policy_inputs policy "{key}" field {field_index} source',
            )
            if source_name not in source_slots:
                raise ValueError(
                    f'policy_inputs policy "{key}" references unknown source '
                    f'"{source_name}"'
                )
            kind = field.get("kind", "copy")
            if kind not in _POLICY_FIELD_KINDS:
                raise ValueError(
                    f'policy_inputs policy "{key}" field {field_index} kind must be '
                    f"one of {_POLICY_FIELD_KINDS}"
                )
            normalized_field = {"kind": kind, "inputSlot": source_slots[source_name]}
            if "length" in field:
                normalized_field["length"] = _positive_integer(
                    field["length"],
                    f'policy_inputs policy "{key}" field {field_index} length',
                )
            if "indices" in field:
                raw_indices = field["indices"]
                reference = _plan_reference(raw_indices)
                if reference is not None:
                    indices = reference
                else:
                    indices = list(raw_indices)
                    if not indices or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in indices
                    ):
                        raise ValueError(
                            f'policy_inputs policy "{key}" field {field_index} indices '
                            "must contain non-negative integers"
                        )
                    if len(indices) != len(set(indices)):
                        raise ValueError(
                            f'policy_inputs policy "{key}" field {field_index} indices '
                            "must be unique"
                        )
                normalized_field["indices"] = indices
            if "default" in field:
                normalized_field["defaultValues"] = field["default"]
            if "scale" in field:
                normalized_field["scale"] = field["scale"]
            normalized_fields.append(normalized_field)

        normalized_policy = {
            "key": key,
            "mode": _policy_mode(
                policy.get("mode"),
                f'policy_inputs policy "{key}" mode',
            ),
            "outputSlot": policy_index,
            "historyFrames": _positive_integer(
                policy.get("history_frames", 1),
                f'policy_inputs policy "{key}" history_frames',
            ),
            "fields": normalized_fields,
        }
        reference = policy.get("reference")
        if reference is not None:
            reference = _policy_key(
                reference,
                f'policy_inputs policy "{key}" reference',
            )
            if reference not in source_slots:
                raise ValueError(
                    f'policy_inputs policy "{key}" references unknown source '
                    f'"{reference}"'
                )
            normalized_policy["referenceSlot"] = source_slots[reference]
        if "state_dim" in policy:
            normalized_policy["stateDim"] = _positive_integer(
                policy["state_dim"],
                f'policy_inputs policy "{key}" state_dim',
            )
        normalized.append(normalized_policy)

    node = Signal(
        "proc-policy-input",
        "policy-inputs",
        {"modeSlot": 0, "inputLabels": ["MODE", *source_names], "policies": normalized},
        (mode, *source_signals),
    )
    return {
        key: node._output(index, f"policy-input-{key}")
        for index, key in enumerate(keys)
    }


def _policy_model_stage(raw, policy_key, stage_index):
    if not isinstance(raw, dict):
        raise TypeError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            "must be a mapping"
        )
    unknown = set(raw) - {
        "model",
        "key",
        "label",
        "output_dim",
        "input_name",
        "output_name",
        "constants",
        "extra_inputs",
    }
    if unknown:
        raise ValueError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            f"has unknown fields {sorted(unknown)}"
        )
    model = raw.get("model")
    if not isinstance(model, Model):
        raise TypeError(
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} '
            "model must be a BotLab Model"
        )
    result = {
        "key": _policy_key(
            raw.get("key", f"stage-{stage_index}"),
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} key',
        ),
        "label": _policy_key(
            raw.get("label", raw.get("key", f"stage-{stage_index}")),
            f'policy_bank policy "{policy_key}" pipeline stage {stage_index} label',
        ),
        "resourceId": model.id,
    }
    result.update(_model_call_data(
        raw,
        f'policy_bank policy "{policy_key}" pipeline stage {stage_index}',
    ))
    return result


_COMBINED_POLICY_INPUT_FIELDS = {
    "reference",
    "history_frames",
    "state_dim",
    "fields",
}
_COMBINED_POLICY_MODEL_FIELDS = {
    "model",
    "pipeline",
    "label",
    "input_name",
    "output_name",
    "constants",
    "extra_inputs",
    "output_dim",
}


def _split_combined_policy_specs(policies):
    input_specs = []
    model_specs = []
    keys = set()
    modes = set()
    allowed = {"key", "mode", *_COMBINED_POLICY_INPUT_FIELDS, *_COMBINED_POLICY_MODEL_FIELDS}
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_bank combined policy {index} must be a mapping")
        unknown = set(policy) - allowed
        if unknown:
            raise ValueError(
                f"policy_bank combined policy {index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_bank combined policy {index} key")
        mode = _policy_mode(policy.get("mode"), f'policy_bank combined policy "{key}" mode')
        if key in keys:
            raise ValueError(f'policy_bank combined policy key "{key}" is duplicated')
        if mode in modes:
            raise ValueError(f"policy_bank combined policy mode {mode} is duplicated")
        keys.add(key)
        modes.add(mode)
        if "fields" not in policy:
            raise ValueError(f'policy_bank combined policy "{key}" requires fields')
        input_specs.append({
            name: policy[name]
            for name in ("key", "mode", *_COMBINED_POLICY_INPUT_FIELDS)
            if name in policy
        })
        model_specs.append({
            name: policy[name]
            for name in ("key", "mode", *_COMBINED_POLICY_MODEL_FIELDS)
            if name in policy
        })
    if not input_specs:
        raise ValueError("policy_bank combined policies must not be empty")
    return input_specs, model_specs


def policy_bank(
    mode,
    inputs=None,
    policies=None,
    *,
    sources=None,
    control_decimation=1,
    call_id=None,
):
    """Run exactly one ONNX policy (or explicit model pipeline) for ``mode``.

    Pass prebuilt ``inputs`` with the low-level two-table API, or pass named
    ``sources`` and combined policy specs containing both ``fields`` and the
    policy ``model``/``pipeline``.
    """
    mode = _require_signal(mode, "policy_bank mode")
    if sources is not None:
        if inputs is not None:
            raise ValueError("policy_bank accepts only one of inputs or sources")
        input_specs, policies = _split_combined_policy_specs(list(policies or ()))
        inputs = policy_inputs(mode, sources, input_specs)
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("policy_bank inputs must be a non-empty mapping")
    state_keys = []
    state_signals = []
    for key, source in inputs.items():
        key = _policy_key(key, "policy_bank input key")
        if key in state_keys:
            raise ValueError(f'policy_bank input key "{key}" is duplicated')
        state_keys.append(key)
        state_signals.append(_require_signal(source, f'policy_bank input "{key}"'))
    state_slots = {key: index + 1 for index, key in enumerate(state_keys)}

    items = list(policies)
    if not items:
        raise ValueError("policy_bank policies must not be empty")
    normalized = []
    resource_ids = []
    policy_keys = []
    for policy_index, policy in enumerate(items):
        if not isinstance(policy, dict):
            raise TypeError(f"policy_bank policy {policy_index} must be a mapping")
        unknown = set(policy) - {
            "key",
            "mode",
            "model",
            "pipeline",
            "label",
            "input_name",
            "output_name",
            "constants",
            "output_dim",
            "extra_inputs",
        }
        if unknown:
            raise ValueError(
                f"policy_bank policy {policy_index} has unknown fields {sorted(unknown)}"
            )
        key = _policy_key(policy.get("key"), f"policy_bank policy {policy_index} key")
        if key in policy_keys:
            raise ValueError(f'policy_bank policy key "{key}" is duplicated')
        policy_keys.append(key)
        if key not in state_slots:
            raise ValueError(f'policy_bank policy "{key}" has no matching input')
        has_model = policy.get("model") is not None
        has_pipeline = policy.get("pipeline") is not None
        if has_model == has_pipeline:
            raise ValueError(
                f'policy_bank policy "{key}" must define exactly one of model or pipeline'
            )
        result = {
            "key": key,
            "mode": _policy_mode(policy.get("mode"), f'policy_bank policy "{key}" mode'),
            "inputSlot": state_slots[key],
            "label": _policy_key(
                policy.get("label", key),
                f'policy_bank policy "{key}" label',
            ),
        }
        if has_model:
            model = policy["model"]
            if not isinstance(model, Model):
                raise TypeError(f'policy_bank policy "{key}" model must be a BotLab Model')
            result["resourceId"] = model.id
            resource_ids.append(model.id)
        else:
            pipeline = list(policy["pipeline"])
            if not pipeline:
                raise ValueError(f'policy_bank policy "{key}" pipeline must not be empty')
            result["pipeline"] = [
                _policy_model_stage(stage, key, stage_index)
                for stage_index, stage in enumerate(pipeline)
            ]
            resource_ids.extend(stage["resourceId"] for stage in result["pipeline"])
        result.update(_model_call_data(policy, f'policy_bank policy "{key}"'))
        normalized.append(result)

    extra_inputs = set(state_keys) - set(policy_keys)
    if extra_inputs:
        raise ValueError(f"policy_bank inputs have no policy: {sorted(extra_inputs)}")
    node_id = resource_ids[0]
    if call_id is not None:
        call_id = _non_empty_string(call_id, "policy_bank call_id")
    return Signal(
        "model-onnx",
        node_id or "policy-bank",
        {
            "resourceId": node_id,
            "controlDecimation": _positive_integer(
                control_decimation,
                "policy_bank control_decimation",
            ),
            "policies": normalized,
        },
        (mode, *state_signals),
        node_id=call_id,
    )
