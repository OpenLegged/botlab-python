"""Numeric rule matching and one-hot encoding."""
import math

from ._signal import Signal, _require_signal
from ._vectors import _non_negative_integer, _positive_integer


_RULE_OPERATORS = ("gt", "gte", "lt", "lte", "eq", "ne")


def _rule_condition(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    branch_keys = [key for key in ("all", "any") if key in value]
    if branch_keys:
        if len(branch_keys) != 1 or set(value) != {branch_keys[0]}:
            raise ValueError(f'{label} must contain exactly one of "all" or "any"')
        key = branch_keys[0]
        conditions = value[key]
        if not isinstance(conditions, (list, tuple)) or not conditions:
            raise ValueError(f'{label}.{key} must be a non-empty list')
        return {
            key: [
                _rule_condition(condition, f"{label}.{key}[{index}]")
                for index, condition in enumerate(conditions)
            ]
        }
    if set(value) != {"index", "op", "value"}:
        raise ValueError(
            f'{label} leaf must contain exactly "index", "op", and "value"'
        )
    index = _non_negative_integer(value["index"], f"{label}.index")
    operator = value["op"]
    if operator not in _RULE_OPERATORS:
        raise ValueError(f"{label}.op must be one of {_RULE_OPERATORS}")
    comparison = value["value"]
    if (
        isinstance(comparison, bool)
        or not isinstance(comparison, (int, float))
        or not math.isfinite(comparison)
    ):
        raise ValueError(f"{label}.value must be a finite number")
    return {"index": index, "op": operator, "value": float(comparison)}


def rule_index(source, rules, *, default=0):
    """Return the value of the first recursively matched numeric rule."""
    source = _require_signal(source, "rule_index source")
    if isinstance(default, bool) or not isinstance(default, int):
        raise ValueError("rule_index default must be an integer")
    items = list(rules)
    if not items:
        raise ValueError("rule_index rules must not be empty")
    normalized = []
    for index, rule in enumerate(items):
        label = f"rule_index rule {index}"
        if not isinstance(rule, dict):
            raise TypeError(f"{label} must be a mapping")
        if set(rule) != {"value", "when"}:
            raise ValueError(f'{label} must contain exactly "value" and "when"')
        output = rule["value"]
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError(f"{label}.value must be an integer")
        normalized.append({
            "value": output,
            "when": _rule_condition(rule["when"], f"{label}.when"),
        })
    return Signal(
        "proc-rule-index",
        "rule-index",
        {"rules": normalized, "defaultValue": default},
        (source,),
    )


def one_hot(source, size):
    """Convert a scalar integer-index signal into a fixed-size one-hot vector."""
    source = _require_signal(source, "one_hot source")
    return Signal(
        "proc-one-hot",
        "one-hot",
        {"size": _positive_integer(size, "one_hot size")},
        (source,),
    )
