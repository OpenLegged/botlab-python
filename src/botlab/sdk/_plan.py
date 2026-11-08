"""Lowering a Signal expression into the compiled control plan."""
from ._declaration import _plan_value
from ._signal import Signal


def _visit_signals(signal, seen, ordered):
    node = signal._node
    key = id(node)
    if key in seen:
        return
    seen.add(key)
    for source in node._inputs:
        _visit_signals(source, seen, ordered)
    ordered.append(node)


def _unique_node_id(label, taken):
    candidate = label
    suffix = 2
    while candidate in taken:
        candidate = f"{label}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _control_plan(robot_id, sink, pd_signal=None, control_signal=None, metrics=None):
    if (pd_signal is None) == (control_signal is None):
        raise ValueError("control plan requires exactly one output signal")
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict):
        raise TypeError("PolicyController metrics must be a mapping of names to scalar Signals")
    for name, signal in metrics.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("PolicyController metric names must be non-empty strings without surrounding whitespace")
        if not isinstance(signal, Signal):
            raise TypeError(f'PolicyController metric "{name}" must be a scalar Signal')
    ordered = []
    seen = set()
    _visit_signals(sink, seen, ordered)
    for signal in metrics.values():
        _visit_signals(signal, seen, ordered)

    hooks = [signal for signal in ordered if signal._type == "proc-action-hook"]
    if len(hooks) != 1:
        raise ValueError(
            "a control plan needs exactly one Signal.action(), got "
            f"{len(hooks)}"
        )

    taken = set()
    for signal in ordered:
        if signal._node_id is None:
            continue
        if signal._node_id in taken:
            raise ValueError(f'control plan node id "{signal._node_id}" is duplicated')
        taken.add(signal._node_id)

    node_ids = {
        id(signal): signal._node_id or _unique_node_id(signal._label, taken)
        for signal in ordered
    }

    nodes = []
    edges = []
    for signal in ordered:
        node_id = node_ids[id(signal)]
        nodes.append({
            "id": node_id,
            "type": signal._type,
            "data": _plan_value(signal._data, f'control plan node "{node_id}" data'),
        })
        for index, source in enumerate(signal._inputs):
            edges.append({
                "id": f"edge-{len(edges) + 1}",
                "source": node_ids[id(source._node)],
                "target": node_id,
                "sourcePort": f"out-{source._output_slot}",
                "targetPort": f"in-{index}",
            })

    output = {
        "policyActionNodeId": node_ids[id(hooks[0])],
    }
    if pd_signal is not None:
        output["pdNodeId"] = node_ids[id(pd_signal._node)]
    else:
        output.update({
            "kind": "raw",
            "controlNodeId": node_ids[id(control_signal._node)],
            "controlOutputSlot": control_signal._output_slot,
        })
    plan = {
        "version": 1,
        "robot": robot_id,
        "nodes": nodes,
        "edges": edges,
        "output": output,
    }
    if metrics:
        plan["metrics"] = {
            name: {"nodeId": node_ids[id(signal._node)], "outputSlot": signal._output_slot}
            for name, signal in metrics.items()
        }
    return plan, hooks[0]._modify
