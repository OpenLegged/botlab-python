"""Plan graph scheduler tests (tier A: stdlib only, no runtime extras).

Mirrors src/domain/runtime/engine/SimulationEngine.ts compile/step semantics:
port->slot routing, sink pruning, topological order, buffer aliasing,
execution-data keys, reset/dispose, and the explicit error shapes.
"""

import unittest

from botlab._runtime import _executor  # noqa: E402


class ConstantProcessor:
    def __init__(self):
        self.output_buffers = []
        self.resets = 0

    def init(self, config, input_dims):
        self.output_buffers = [list(config["values"])]
        self.config = config

    def compute(self, inputs):
        pass

    def reset(self):
        self.resets += 1


class AddOneProcessor:
    def __init__(self):
        self.output_buffers = []
        self.saw_inputs = None
        self.config = None

    def init(self, config, input_dims):
        self.config = config
        self.output_buffers = [list(range(input_dims[0]))] if input_dims[0] else [[]]

    def compute(self, inputs):
        self.saw_inputs = list(inputs)
        source = inputs[0]
        self.output_buffers[0] = [value + 1 for value in source]


class FailingProcessor:
    def __init__(self):
        self.output_buffers = [[0.0]]

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        raise ValueError("boom")


class FrameCounterProcessor:
    uses_frame = True

    def __init__(self):
        self.output_buffers = [[0.0]]
        self.frames = []

    def init(self, config, input_dims):
        pass

    def compute(self, inputs, frame=None):
        self.frames.append((frame.time, frame.index))
        self.output_buffers[0][0] = float(frame.index)


class SinkProcessor:
    def __init__(self):
        self.output_buffers = []
        self.received = None

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        self.received = [list(values) for values in inputs]


REGISTRY = {
    "obs-constant": ConstantProcessor,
    "proc-add-one": AddOneProcessor,
    "proc-fail": FailingProcessor,
    "proc-frame": FrameCounterProcessor,
    "out-scope": SinkProcessor,
}


def node(node_id, node_type, **data):
    return {"id": node_id, "type": node_type, "data": data}


def edge(edge_id, source, target, source_port=None, target_port=None):
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "sourcePort": source_port,
        "targetPort": target_port,
    }


class PortSlotTest(unittest.TestCase):
    def test_missing_port_is_slot_zero(self):
        self.assertEqual(_executor.slot_from_port_id(None, "in"), 0)

    def test_numbered_port_parses(self):
        self.assertEqual(_executor.slot_from_port_id("in-2", "in"), 2)
        self.assertEqual(_executor.slot_from_port_id("out-1", "out"), 1)

    def test_wrong_direction_or_shape_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _executor.slot_from_port_id("out-1", "in")
        self.assertIn("Invalid in port", str(ctx.exception))


class SignalLabelEdgesTest(unittest.TestCase):
    def test_send_receive_label_pairs_into_virtual_edge(self):
        senders = [node("s1", "proc-signal-send", signalLabel="Imu")]
        receivers = [node("r1", "proc-signal-receive", signalLabel="imu")]
        edges = _executor.build_signal_label_virtual_edges(senders + receivers, "runtime plan")
        self.assertEqual(
            edges,
            [
                {
                    "id": "signal:s1:r1",
                    "source": "s1",
                    "sourcePort": "out-0",
                    "target": "r1",
                    "targetPort": "in-0",
                }
            ],
        )

    def test_duplicate_send_labels_raise_in_strict_mode(self):
        nodes = [
            node("s1", "proc-signal-send", signalLabel="imu"),
            node("s2", "proc-signal-send", signalLabel="Imu "),
            node("r1", "proc-signal-receive", signalLabel="imu"),
        ]
        with self.assertRaises(ValueError) as ctx:
            _executor.build_signal_label_virtual_edges(nodes, "runtime plan")
        self.assertIn("duplicate signal send labels", str(ctx.exception))

    def test_unresolvable_receiver_raises_in_strict_mode(self):
        nodes = [node("r1", "proc-signal-receive", signalLabel="gone")]
        with self.assertRaises(ValueError) as ctx:
            _executor.build_signal_label_virtual_edges(nodes, "runtime plan")
        self.assertIn("cannot resolve label", str(ctx.exception))


class RuntimeNodeSelectionTest(unittest.TestCase):
    def test_nodes_not_reaching_a_sink_are_pruned(self):
        nodes = [
            node("keep", "obs-constant", values=[1.0]),
            node("drop", "obs-constant", values=[2.0]),
            node("sink", "out-scope"),
        ]
        edges = [edge("e1", "keep", "sink")]
        active = _executor.select_runtime_nodes(nodes, edges)
        self.assertEqual([n["id"] for n in active], ["keep", "sink"])

    def test_unknown_edge_endpoints_are_ignored(self):
        nodes = [node("sink", "out-scope")]
        edges = [edge("e1", "ghost", "sink")]
        active = _executor.select_runtime_nodes(nodes, edges)
        self.assertEqual([n["id"] for n in active], ["sink"])


class TopologicalSortTest(unittest.TestCase):
    def test_dependencies_execute_before_dependents(self):
        nodes = [node("a", "obs-constant"), node("b", "proc-add-one"), node("sink", "out-scope")]
        edges = [edge("e1", "a", "b"), edge("e2", "b", "sink")]
        order = _executor.topological_sort(nodes, edges)
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("b"), order.index("sink"))

    def test_cycle_raises(self):
        nodes = [node("a", "proc-add-one"), node("b", "proc-add-one")]
        edges = [edge("e1", "a", "b"), edge("e2", "b", "a")]
        with self.assertRaises(ValueError) as ctx:
            _executor.topological_sort(nodes, edges)
        self.assertIn("Cycle detected", str(ctx.exception))


class CompileStepTest(unittest.TestCase):
    def compile_plan(self, nodes, edges, **options):
        executor = _executor.PlanExecutor(REGISTRY)
        executor.compile(nodes, edges, **options)
        return executor

    def test_unknown_node_type_raises_at_compile(self):
        with self.assertRaises(ValueError) as ctx:
            self.compile_plan(
                [node("x", "proc-mystery"), node("sink", "out-scope")],
                [edge("e1", "x", "sink")],
            )
        self.assertIn('No processor registered for node type "proc-mystery"', str(ctx.exception))

    def test_step_executes_in_order_and_routes_buffers(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[1.0, 2.0]),
                node("add", "proc-add-one"),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "add"), edge("e2", "add", "sink")],
        )
        executor.step()
        add = executor.get_processor("add")
        self.assertEqual(add.output_buffers[0], [2.0, 3.0])

    def test_inputs_alias_source_buffers(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[1.0]),
                node("add", "proc-add-one"),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "add"), edge("e2", "add", "sink")],
        )
        executor.step()
        add = executor.get_processor("add")
        constant = executor.get_processor("c")
        self.assertIs(add.saw_inputs[0], constant.output_buffers[0])

    def test_execution_data_keys_match_engine_contract(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[1.0]),
                node("add", "proc-add-one"),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "add"), edge("e2", "add", "sink")],
        )
        data = {}
        executor.step(data)
        self.assertIn("c", data)  # slot-0 output key is the bare node id
        self.assertIn("add_in-0", data)
        self.assertIn("add_in", data)  # primary input key
        self.assertIn("add", data)

    def test_init_receives_node_metadata_and_segments(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[3.0, 4.0]),
                node("add", "proc-add-one"),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "add"), edge("e2", "add", "sink")],
        )
        add = executor.get_processor("add")
        self.assertEqual(add.config["nodeId"], "add")
        self.assertEqual(add.config["nodeType"], "proc-add-one")
        self.assertEqual(add.config["upstreamInputSegments"], [[2]])

    def test_multiple_edges_to_same_target_slot_raise(self):
        with self.assertRaises(ValueError) as ctx:
            self.compile_plan(
                [
                    node("a", "obs-constant", values=[1.0]),
                    node("b", "obs-constant", values=[2.0]),
                    node("add", "proc-add-one"),
                    node("sink", "out-scope"),
                ],
                [
                    edge("e1", "a", "add"),
                    edge("e2", "b", "add", None, "in-0"),
                    edge("e3", "add", "sink"),
                ],
            )
        self.assertIn("Multiple edges target add:in-0", str(ctx.exception))

    def test_compute_errors_are_wrapped_with_node_identity(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[1.0]),
                node("bad", "proc-fail"),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "bad"), edge("e2", "bad", "sink")],
        )
        with self.assertRaises(ValueError) as ctx:
            executor.step()
        self.assertIn("Node bad (proc-fail) compute failed", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_reset_calls_every_processor(self):
        executor = self.compile_plan(
            [
                node("c", "obs-constant", values=[1.0]),
                node("sink", "out-scope"),
            ],
            [edge("e1", "c", "sink")],
        )
        executor.reset()
        self.assertEqual(executor.get_processor("c").resets, 1)

    def test_dispose_clears_state(self):
        executor = self.compile_plan(
            [node("c", "obs-constant", values=[1.0]), node("sink", "out-scope")],
            [edge("e1", "c", "sink")],
        )
        executor.dispose()
        self.assertIsNone(executor.get_processor("c"))

    def test_frame_reaches_uses_frame_processors(self):
        class Frame:
            def __init__(self, index):
                self.dt = 0.02
                self.time = index * 0.02
                self.index = index

        executor = self.compile_plan(
            [node("f", "proc-frame"), node("sink", "out-scope")],
            [edge("e1", "f", "sink")],
        )
        executor.step(frame=Frame(0))
        executor.step(frame=Frame(1))
        counter = executor.get_processor("f")
        self.assertEqual(counter.frames, [(0.0, 0), (0.02, 1)])
        self.assertEqual(counter.output_buffers[0], [1.0])


if __name__ == "__main__":
    unittest.main()
