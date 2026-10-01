"""Plan graph scheduler — Python mirror of SimulationEngine compile/step.

Consumes the same ``{nodes, edges, output}`` plan contract as the TypeScript
engine (src/domain/runtime/engine/SimulationEngine.ts) with identical
semantics: port-to-slot routing, sink-reachability pruning, topological
execution, zero-copy buffer aliasing, execution-data keys, reset/dispose and
the explicit error shapes. Differences are deliberate and documented:

- synchronous compute only: the browser needs computeAsync for ORT-Web, the
  local onnxruntime call is synchronous;
- processors are duck-typed (output_buffers / init / compute / reset? /
  dispose?) exactly like INodeProcessor;
- stdlib only: buffers are opaque to the scheduler (processors own numpy).
"""

import re

SIGNAL_SEND_NODE_TYPE = "proc-signal-send"
SIGNAL_RECEIVE_NODE_TYPE = "proc-signal-receive"

SINK_NODE_TYPES = frozenset({"out-mujoco", "out-scope", "out-logger"})

_PORT_PATTERN = "^(?P<direction>{direction})-(?P<slot>\\d+)$"


def slot_from_port_id(port, direction):
    if port is None:
        return 0
    match = re.match(_PORT_PATTERN.format(direction=direction), port)
    if not match:
        raise ValueError(f"[executionKeys] Invalid {direction} port \"{port}\"")
    return int(match.group("slot"))


def signal_label_key(value):
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _display_signal_label(node):
    label = node.get("data", {}).get("signalLabel")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return str(node.get("data", {}).get("label", node["id"]))


def build_signal_label_virtual_edges(nodes, source):
    senders = {}
    duplicate_labels = set()

    for node in nodes:
        if node["type"] != SIGNAL_SEND_NODE_TYPE:
            continue
        key = signal_label_key(node.get("data", {}).get("signalLabel"))
        if not key:
            continue
        if key in senders:
            duplicate_labels.add(key)
            continue
        senders[key] = node

    if duplicate_labels:
        raise ValueError(
            f"[SignalLabels] {source} has duplicate signal send labels: "
            + ", ".join(sorted(duplicate_labels))
        )

    virtual_edges = []
    for receiver in nodes:
        if receiver["type"] != SIGNAL_RECEIVE_NODE_TYPE:
            continue
        key = signal_label_key(receiver.get("data", {}).get("signalLabel"))
        if not key:
            continue
        sender = senders.get(key)
        if sender is None or key in duplicate_labels:
            raise ValueError(
                f"[SignalLabels] {source} receiver {receiver['id']} cannot resolve "
                f"label \"{_display_signal_label(receiver)}\""
            )
        virtual_edges.append(
            {
                "id": f"signal:{sender['id']}:{receiver['id']}",
                "source": sender["id"],
                "sourcePort": "out-0",
                "target": receiver["id"],
                "targetPort": "in-0",
            }
        )
    return virtual_edges


def select_runtime_nodes(nodes, edges):
    node_by_id = {node["id"]: node for node in nodes}
    incoming_sources = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        if source not in node_by_id or target not in node_by_id:
            continue
        incoming_sources.setdefault(target, []).append(source)

    active_ids = set()

    def visit(node_id):
        if node_id in active_ids:
            return
        active_ids.add(node_id)
        for source in incoming_sources.get(node_id, []):
            visit(source)

    for node in nodes:
        if node["type"] in SINK_NODE_TYPES:
            visit(node["id"])

    return [node for node in nodes if node["id"] in active_ids]


def topological_sort(nodes, edges):
    node_ids = {node["id"] for node in nodes}
    outgoing = {}
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        if edge["source"] not in node_ids:
            raise ValueError(
                f"[SimulationEngine] Edge {edge['id']} source \"{edge['source']}\" is missing"
            )
        if edge["target"] not in node_ids:
            raise ValueError(
                f"[SimulationEngine] Edge {edge['id']} target \"{edge['target']}\" is missing"
            )
        outgoing.setdefault(edge["source"], []).append(edge["target"])
        indegree[edge["target"]] += 1

    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
    result = []
    head = 0
    while head < len(queue):
        node_id = queue[head]
        head += 1
        result.append(node_id)
        for target in outgoing.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(result) != len(nodes):
        raise ValueError("[SimulationEngine] Cycle detected in plan")
    return result


def _output_execution_key(node_id, slot):
    return node_id if slot == 0 else f"{node_id}_out-{slot}"


class PlanExecutor:
    """Compiles a plan once and executes it across control steps."""

    def __init__(self, registry):
        self._registry = registry
        self._compiled_nodes = []
        self._links = []
        self._step_input_buffers = []
        self._id_to_index = {}
        self._frame = None
        self._observation_bindings = []

    @property
    def observation_bindings(self):
        """Mirror of the engine's observation bindings (fill each step)."""
        return self._observation_bindings

    def compile(self, nodes, edges):
        runtime_edges = edges + build_signal_label_virtual_edges(nodes, "runtime plan")
        active_nodes = select_runtime_nodes(nodes, runtime_edges)
        active_ids = {node["id"] for node in active_nodes}
        active_edges = [
            edge
            for edge in runtime_edges
            if edge["source"] in active_ids and edge["target"] in active_ids
        ]
        execution_order_ids = topological_sort(active_nodes, active_edges)

        self.dispose()

        node_by_id = {node["id"]: node for node in active_nodes}
        incoming_by_target = {}
        for edge in active_edges:
            incoming_by_target.setdefault(edge["target"], []).append(edge)

        for index, node_id in enumerate(execution_order_ids):
            self._id_to_index[node_id] = index

        try:
            for index, node_id in enumerate(execution_order_ids):
                node = node_by_id[node_id]
                processor_class = self._registry.get(node["type"])
                if processor_class is None:
                    raise ValueError(
                        f"[SimulationEngine] No processor registered for node type \"{node['type']}\""
                    )

                incoming_edges = incoming_by_target.get(node_id, [])
                max_target_slot = -1
                for edge in incoming_edges:
                    max_target_slot = max(
                        max_target_slot, slot_from_port_id(edge.get("targetPort"), "in")
                    )
                input_dims = [0] * (max_target_slot + 1)
                input_segments = [None] * (max_target_slot + 1)
                seen_target_slots = set()

                node_links = []
                for edge in incoming_edges:
                    if edge["source"] not in self._id_to_index:
                        raise ValueError(
                            f"[SimulationEngine] Edge source \"{edge['source']}\" "
                            "not found in compiled plan"
                        )
                    source_index = self._id_to_index[edge["source"]]
                    source_slot = slot_from_port_id(edge.get("sourcePort"), "out")
                    target_slot = slot_from_port_id(edge.get("targetPort"), "in")
                    if target_slot in seen_target_slots:
                        target_port = edge.get("targetPort") or "in-0"
                        raise ValueError(
                            f"[SimulationEngine] Multiple edges target "
                            f"{edge['target']}:{target_port}"
                        )
                    seen_target_slots.add(target_slot)

                    source_processor = self._compiled_nodes[source_index]["processor"]
                    source_outputs = source_processor.output_buffers
                    if source_slot >= len(source_outputs) or source_outputs[source_slot] is None:
                        source_port = edge.get("sourcePort") or "out-0"
                        raise ValueError(
                            f"[SimulationEngine] Source {edge['source']}:{source_port} "
                            "has no output buffer"
                        )

                    source_output = source_outputs[source_slot]
                    input_dims[target_slot] = len(source_output)
                    segments = getattr(source_processor, "output_segments", None)
                    if segments is not None and source_slot < len(segments):
                        input_segments[target_slot] = segments[source_slot]
                    else:
                        input_segments[target_slot] = [len(source_output)]
                    node_links.append((source_index, source_slot, target_slot))

                processor = processor_class()
                self._compiled_nodes.append(
                    {
                        "id": node_id,
                        "type": node["type"],
                        "data": node.get("data", {}),
                        "processor": processor,
                    }
                )
                config = dict(node.get("data", {}))
                config["nodeId"] = node_id
                config["nodeType"] = node["type"]
                config["upstreamInputSegments"] = input_segments
                processor.init(config, input_dims)

                if getattr(processor, "is_observation", False):
                    self._observation_bindings.append(
                        {
                            "nodeId": node_id,
                            "type": node["type"],
                            "data": node.get("data", {}),
                            "processor": processor,
                            "qposOffset": 0,
                            "qvelOffset": 0,
                            "namePrefix": "",
                        }
                    )

                self._links.append(node_links)
                self._step_input_buffers.append([None] * (max_target_slot + 1))
        except Exception:
            try:
                self.dispose()
            except Exception as dispose_error:
                raise RuntimeError(
                    "[SimulationEngine] Compilation and partial processor cleanup both failed: "
                    f"{dispose_error}"
                ) from dispose_error
            raise

    def get_processor(self, node_id):
        index = self._id_to_index.get(node_id)
        if index is None or index >= len(self._compiled_nodes):
            return None
        return self._compiled_nodes[index]["processor"]

    def step(self, execution_data=None, frame=None):
        """Run one control step; processors with ``uses_frame`` get the frame.

        The frame mirrors SimulationStepContext.frame ({dt, time, index}) —
        the TS engine only exposes it through computeAsync; here it is an
        explicit channel for the same processors (e.g. proc-clock).
        """
        self._frame = frame
        for index, compiled in enumerate(self._compiled_nodes):
            node_links = self._links[index]
            inputs = self._step_input_buffers[index]

            if node_links:
                for source_index, source_slot, target_slot in node_links:
                    source_buffer = self._compiled_nodes[source_index]["processor"].output_buffers[
                        source_slot
                    ]
                    inputs[target_slot] = source_buffer
                    if execution_data is not None:
                        node_id = compiled["id"]
                        execution_data[f"{node_id}_in-{target_slot}"] = source_buffer
                        if target_slot == 0:
                            execution_data[f"{node_id}_in"] = source_buffer
                self._compute(compiled, inputs)
            else:
                self._compute(compiled, [])

            if execution_data is not None:
                outputs = compiled["processor"].output_buffers
                for slot in range(len(outputs)):
                    execution_data[
                        _output_execution_key(compiled["id"], slot)
                    ] = outputs[slot]
        self._frame = None

    def _compute(self, compiled, inputs):
        processor = compiled["processor"]
        try:
            if self._frame is not None and getattr(processor, "uses_frame", False):
                processor.compute(inputs, self._frame)
            else:
                processor.compute(inputs)
        except Exception as error:
            raise ValueError(
                f"Node {compiled['id']} ({compiled['type']}) compute failed: {error}"
            ) from error

    def reset(self):
        for compiled in self._compiled_nodes:
            reset = getattr(compiled["processor"], "reset", None)
            if reset is not None:
                reset()

    def dispose(self):
        errors = []
        for compiled in self._compiled_nodes:
            dispose = getattr(compiled["processor"], "dispose", None)
            if dispose is not None:
                try:
                    dispose()
                except Exception as error:
                    errors.append(error)

        self._compiled_nodes = []
        self._links = []
        self._step_input_buffers = []
        self._id_to_index = {}
        self._observation_bindings = []

        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise RuntimeError(
                "[SimulationEngine] Multiple processors failed to dispose: "
                + "; ".join(str(error) for error in errors)
            )
