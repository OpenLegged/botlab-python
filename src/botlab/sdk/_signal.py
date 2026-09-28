"""The Signal expression core: the node class, arithmetic and logic constructors."""
import math

from ._declaration import _non_empty_string, _plan_reference
from ._vectors import (
    _finite_number,
    _finite_values,
    _negated,
    _non_negative_integer,
    _positive_integer,
    _reciprocal,
)


# ---------------------------------------------------------------------------
# Signals
#
# A controller is an expression: observations flow into a policy, the policy
# action flows into an output stage. Every call below returns a Signal, and
# passing a Signal into another call *is* the connection -- there are no node
# ids, ports, or edges to write down. PolicyController lowers the expression
# into the plan BotLab's TypeScript engine compiles once and runs per frame.
# ---------------------------------------------------------------------------


def _require_signal(value, label):
    if not isinstance(value, Signal):
        raise TypeError(f"{label} must be a BotLab Signal")
    return value


class Signal:
    """One value in a controller expression, produced from zero or more inputs.

    Signals behave like the vectors they carry: ``scale * signal + offset`` is an
    affine map, ``signal[indices]`` reorders, and ``signal.clip(low, high)``
    limits. Passing a Signal into another call is what connects them.
    """

    __slots__ = (
        "_type",
        "_label",
        "_data",
        "_inputs",
        "_node_id",
        "_modify",
        "_node",
        "_output_slot",
    )

    def __init__(self, node_type, label, data, inputs=(), *, node_id=None, modify=None):
        self._type = _non_empty_string(node_type, "Signal type")
        self._label = _non_empty_string(label, "Signal label")
        self._data = data
        self._inputs = tuple(inputs)
        self._node_id = node_id
        self._modify = modify
        self._node = self
        self._output_slot = 0

    def __repr__(self):
        return f"<Signal {self._label} ({self._type})>"

    def __bool__(self):
        raise TypeError(
            "BotLab Signal values change at runtime and cannot be used in Python "
            "if/and/or/not. Use where(condition, when_true, when_false), & / | / ~, "
            "or a custom Controller.step() for arbitrary Python control flow."
        )

    def clip(self, low, high):
        return Signal(
            "proc-clamp",
            f"{self._label}-clip",
            {"min": low, "max": high},
            (self,),
        )

    def __getitem__(self, indices):
        """Gather values by index, the way ``actions[motor_order]`` reads in NumPy."""
        return take(self, indices)

    def at(self, index):
        """Select one vector element as a scalar Signal for comparisons."""
        return take(self, [_non_negative_integer(index, "Signal.at index")])

    def compare(self, operator, other):
        """Build a scalar runtime comparison against a number or scalar Signal."""
        if operator not in ("gt", "gte", "lt", "lte", "eq", "ne"):
            raise ValueError(
                "Signal.compare operator must be one of gt, gte, lt, lte, eq, ne"
            )
        if isinstance(other, Signal):
            return Signal(
                "proc-compare",
                f"{self._label}-{operator}",
                {"operator": operator},
                (self, other),
            )
        return Signal(
            "proc-compare",
            f"{self._label}-{operator}",
            {
                "operator": operator,
                "value": _finite_number(other, "Signal comparison value"),
            },
            (self,),
        )

    def gt(self, other):
        return self.compare("gt", other)

    def gte(self, other):
        return self.compare("gte", other)

    def lt(self, other):
        return self.compare("lt", other)

    def lte(self, other):
        return self.compare("lte", other)

    def eq(self, other):
        return self.compare("eq", other)

    def ne(self, other):
        return self.compare("ne", other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __and__(self, other):
        return all_of(self, other)

    def __or__(self, other):
        return any_of(self, other)

    def __invert__(self):
        return logical_not(self)

    def action(self, *, modify=None):
        """Mark this signal as the controller action.

        The marked value is fed back to ``robot.last_action()`` on the next frame
        and is the one per-frame Python extension point: ``modify(action, frame)``
        runs between the policy-order action and the output stage.
        """
        if modify is not None and not callable(modify):
            raise TypeError("Signal.action modify must be callable or None")
        return Signal("proc-action-hook", "action", {}, (self,), modify=modify)

    def _affine(self, scale, offset):
        return Signal(
            "proc-math",
            f"{self._label}-affine",
            {"operation": "affine", "scale": scale, "offset": offset},
            (self,),
        )

    def _elementwise(self, operation, other):
        """Per-frame arithmetic against another signal.

        A constant operand folds into this signal's own affine node; a second signal
        cannot, because both operands only exist per frame.
        """
        return Signal(
            "proc-vector-math",
            f"{self._label}-{operation}",
            {"operation": operation},
            (self, other),
        )

    def __mul__(self, scale):
        if isinstance(scale, Signal):
            return self._elementwise("multiply", scale)
        return self._affine(scale, 0.0)

    __rmul__ = __mul__

    def __add__(self, offset):
        if isinstance(offset, Signal):
            return self._elementwise("add", offset)
        # `scale * signal + offset` is one affine node, not two: fold the offset
        # into the scale-only node this signal already is.
        if self._type == "proc-math" and self._data["offset"] == 0.0:
            return Signal(
                "proc-math",
                self._label,
                {"operation": "affine", "scale": self._data["scale"], "offset": offset},
                self._inputs,
            )
        return self._affine(1.0, offset)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Signal):
            return self._elementwise("subtract", other)
        return self + _negated(other, "Signal subtraction operand")

    def __rsub__(self, other):
        # `other - signal` is `-1 * signal + other`, still one affine node.
        return self._affine(-1.0, other)

    def __truediv__(self, other):
        if isinstance(other, Signal):
            return self._elementwise("divide", other)
        return self._affine(_reciprocal(other, "Signal division operand"), 0.0)

    def __neg__(self):
        return self._affine(-1.0, 0.0)

    def _output(self, slot, label):
        """Return another output of this node without duplicating the node."""
        slot = _non_negative_integer(slot, "Signal output slot")
        if slot == 0:
            return self
        output = Signal.__new__(Signal)
        output._type = self._type
        output._label = _non_empty_string(label, "Signal output label")
        output._data = self._data
        output._inputs = self._inputs
        output._node_id = self._node_id
        output._modify = self._modify
        output._node = self._node
        output._output_slot = slot
        return output


def concat(signals):
    """Concatenate signals into one vector, in the order given."""
    items = list(signals)
    if not items:
        raise ValueError("concat requires at least one signal")
    for index, signal in enumerate(items):
        _require_signal(signal, f"concat input {index}")
    if len(items) == 1:
        return items[0]
    return Signal("proc-mux", "concat", {"channels": len(items)}, items)


def take(source, indices):
    """Gather a vector signal by explicit non-negative indices."""
    source = _require_signal(source, "take source")
    reference = _plan_reference(indices)
    if reference is not None:
        return Signal(
            "proc-reorder",
            f"{source._label}-reordered",
            {"indices": reference},
            (source,),
        )
    if isinstance(indices, (int, slice, str)) or isinstance(indices, bool):
        raise TypeError(
            "take indices must be a sequence, such as signal[robot.motor_order]; "
            "single elements and slices are not signals"
        )
    selected = list(indices)
    if not selected:
        raise ValueError("take indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected):
        raise ValueError("take indices must contain non-negative integers")
    return Signal(
        "proc-reorder",
        f"{source._label}-reordered",
        {"indices": selected},
        (source,),
    )


def split(source, sizes):
    """Split one vector signal into ordered or named fixed-size outputs."""
    source = _require_signal(source, "split source")
    names = None
    if isinstance(sizes, dict):
        if not sizes:
            raise ValueError("split sizes must not be empty")
        names = [_non_empty_string(name, "split output name") for name in sizes]
        values = list(sizes.values())
    else:
        values = list(sizes)
    if not values:
        raise ValueError("split sizes must not be empty")
    normalized = [
        _positive_integer(size, f"split size {index}")
        for index, size in enumerate(values)
    ]
    node = Signal("proc-demux", f"{source._label}-split", {"sizes": normalized}, (source,))
    outputs = [node._output(index, f"{source._label}-part-{index}") for index in range(len(normalized))]
    if names is not None:
        return dict(zip(names, outputs))
    return tuple(outputs)


def _as_signal(source, label):
    """Accept a Signal, or a list of Signals to concatenate, anywhere one is used."""
    if isinstance(source, Signal):
        return source
    if isinstance(source, (list, tuple)):
        return concat(source)
    raise TypeError(f"{label} must be a Signal or a list of Signals")


def _numeric_signal(value):
    return value if isinstance(value, Signal) else constant(value)


def _logical_signal(operator, conditions):
    items = list(conditions)
    if not items:
        raise ValueError(f"{operator} requires at least one condition")
    for index, condition in enumerate(items):
        _require_signal(condition, f"{operator} condition {index}")
    if len(items) == 1 and operator != "not":
        return items[0]
    return Signal(
        "proc-logic",
        operator,
        {"operator": operator},
        items,
    )


def all_of(*conditions):
    """Return a scalar Signal that is true when every condition is non-zero."""
    return _logical_signal("all", conditions)


def any_of(*conditions):
    """Return a scalar Signal that is true when any condition is non-zero."""
    return _logical_signal("any", conditions)


def logical_not(condition):
    """Invert one scalar runtime condition."""
    return _logical_signal("not", (condition,))


def where(condition, when_true, when_false):
    """Choose between same-sized Signals from a scalar runtime condition.

    Both branches are part of the compiled graph. Use ``policy_bank`` instead when
    selecting between expensive ONNX policies so only the active policy is inferred.
    """
    condition = _require_signal(condition, "where condition")
    when_true = _as_signal(when_true, "where when_true")
    when_false = _as_signal(when_false, "where when_false")
    return Signal(
        "proc-where",
        "where",
        {},
        (condition, when_true, when_false),
    )


def constant(values, *, size=None):
    """A fixed vector, for policy inputs the simulation does not produce."""
    reference = _plan_reference(values)
    if reference is not None:
        if size is None:
            raise ValueError("constant resource values require an explicit size")
        size = _positive_integer(size, "constant size")
        return Signal("obs-constant", "constant", {"dim": size, "values": reference})
    if size is None:
        vector = _finite_values(values, "constant values")
    else:
        size = _positive_integer(size, "constant size")
        if isinstance(values, bool) or not isinstance(values, (int, float)):
            raise ValueError("constant with size requires one finite numeric value")
        value = float(values)
        if not math.isfinite(value):
            raise ValueError("constant value must be finite")
        vector = [value] * size
    if not vector:
        raise ValueError("constant values must not be empty")
    return Signal("obs-constant", "constant", {"dim": len(vector), "values": vector})
