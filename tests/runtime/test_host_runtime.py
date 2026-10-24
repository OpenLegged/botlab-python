"""Local runtime session + botlab_host three-state tests (tier A: stdlib only)."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from botlab import _host as botlab_host
from botlab._runtime import _host_runtime  # noqa: E402
from botlab._runtime import run as runtime_run  # noqa: E402


class ConstantSource:
    def __init__(self):
        self.output_buffers = []

    def init(self, config, input_dims):
        self.output_buffers = [list(config["values"])]

    def compute(self, inputs):
        pass


class Sink:
    def __init__(self):
        self.output_buffers = []
        self.received = []

    def init(self, config, input_dims):
        pass

    def compute(self, inputs):
        self.received.append([list(values) for values in inputs])


FAKE_REGISTRY = {"obs-constant": ConstantSource, "out-scope": Sink}


class SpyPhysics:
    """Records the control-loop call sequence; no MuJoCo involved."""

    def __init__(self, stop_after=3):
        self.calls = []
        self.applied = []
        self.observations = {"robot-1": [1.0, 2.0, 3.0]}
        self.stop_after = stop_after
        self.control_steps = 0

    def fill_observations(self, executor, last_action=None, parameters=None):
        self.calls.append("fill")

    def step_substeps(self, steps, ctrl=None, substep_pd=None, observer=None):
        self.calls.append(("step", steps, ctrl, substep_pd))
        self.control_steps += 1

    def after_control_step(self):
        self.calls.append("after")

    def qpos(self):
        return [0.0, 0.0, 1.0]

    def qvel(self):
        return [0.0]

    def should_stop(self):
        return self.control_steps >= self.stop_after

    def reset(self):
        self.calls.append("reset")

    def robot_observation(self, robot_id):
        self.calls.append(("observe", robot_id))
        return self.observations[robot_id]

    def apply_action(self, robot_id, action):
        self.calls.append(("apply", robot_id, list(action)))
        self.applied.append(list(action))

    def request_reset(self, reason, spec=None):
        self.calls.append(("request_reset", reason))

    def read_mujoco(self, robot_id, query, target=None, start=None, length=None):
        self.calls.append(("read_mujoco", robot_id, query, target, start, length))
        return [4.0]

    def sensor_observation(self, sensor_id, channel=None):
        raise AssertionError("not exercised here")

    def load_motion(self, motion_id):
        self.calls.append(("load_motion", motion_id))
        return {"marker": motion_id}

    def update_heightfield(self, terrain_id, heights):
        raise AssertionError("not exercised here")


class FakeDisplay:
    def __init__(self):
        self.frames = []
        self.metrics = []

    def broadcast(self, time_seconds, qpos, qvel):
        self.frames.append((time_seconds, list(qpos), list(qvel)))

    def update_metrics(self, metrics):
        self.metrics.append(metrics)

    def update_diagnostics(self, diagnostics):
        pass


def simple_plan(output=None):
    return {
        "nodes": [
            {"id": "c", "type": "obs-constant", "data": {"values": [1.0, 2.0]}},
            {"id": "sink", "type": "out-scope", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "c", "target": "sink",
             "sourcePort": "out-0", "targetPort": "in-0"},
        ],
        "output": output or {
            "policyActionNodeId": "c",
            "kind": "raw",
            "controlNodeId": "c",
            "controlOutputSlot": 0,
        },
    }


class HostThreeStateTest(unittest.TestCase):
    def tearDown(self):
        botlab_host.set_runtime_session(None)

    def test_offline_state_keeps_explicit_error(self):
        with self.assertRaises(botlab_host.BotLabError) as ctx:
            botlab_host.observe("robot-1")
        self.assertIn("offline packager never steps", str(ctx.exception))

    def test_offline_substep_observer_keeps_explicit_error(self):
        with self.assertRaises(botlab_host.BotLabError) as ctx:
            botlab_host.registerSubstepObserver(lambda frame: None)
        self.assertIn("on_substep", str(ctx.exception))

    def test_session_state_dispatches_runtime_calls(self):
        session = _host_runtime.LocalRuntimeSession(SpyPhysics())
        self.addCleanup(session.close)
        botlab_host.set_runtime_session(session)
        self.assertEqual(botlab_host.observe("robot-1"), [1.0, 2.0, 3.0])
        botlab_host.applyAction("robot-1", [9.0])
        self.assertEqual(session._physics.applied, [[9.0]])
        self.assertEqual(
            botlab_host.readMujoco("robot-1", "qpos", None, 0, 1), [4.0]
        )

    def test_offline_registration_behavior_is_unchanged(self):
        plan = simple_plan()
        botlab_host.registerPolicyController(plan)
        self.assertEqual(botlab_host.ACTIVE.policy_controller_plan["nodes"], plan["nodes"])
        botlab_host.registerPolicyController(plan, modifier=lambda action: action)
        self.assertTrue(botlab_host.ACTIVE.plan_has_modifier)


class SessionPlanLoopTest(unittest.TestCase):
    def tearDown(self):
        botlab_host.set_runtime_session(None)

    def make_session(self, **kwargs):
        physics = SpyPhysics(**kwargs.pop("physics_kwargs", {}))
        session = _host_runtime.LocalRuntimeSession(
            physics, registry=FAKE_REGISTRY,
            control_hz=kwargs.pop("control_hz", 50),
            physics_hz=kwargs.pop("physics_hz", 200),
            **kwargs
        )
        self.addCleanup(session.close)
        return session, physics

    def test_plan_loop_fill_step_payload_substeps_broadcast(self):
        display = FakeDisplay()
        session, physics = self.make_session(display=display, physics_kwargs={"stop_after": 3})
        session.registerPolicyController(simple_plan())
        steps = session.run()
        self.assertEqual(steps, 3)
        # per control step: fill -> executor step -> step(4 substeps, raw ctrl) -> after
        first_cycle = physics.calls[physics.calls.index("fill"):physics.calls.index("after") + 1]
        self.assertEqual(
            first_cycle,
            ["fill", ("step", 4, [1.0, 2.0], None), "after"],
        )
        self.assertEqual(len(display.frames), 3)
        self.assertEqual(display.frames[0], (0.0, [0.0, 0.0, 1.0], [0.0]))

    def test_position_pd_output_routes_substep_view(self):
        class PdStub:
            def __init__(self):
                self.output_buffers = [[0.0]]

            def init(self, config, input_dims):
                pass

            def compute(self, inputs):
                pass

            def get_substep_pd(self):
                return {
                    "target": [0.5], "kps": [40.0], "kds": [4.0],
                    "effort_min": [-10.0], "effort_max": [10.0],
                }

        physics = SpyPhysics(stop_after=1)
        session = _host_runtime.LocalRuntimeSession(
            physics,
            registry={**FAKE_REGISTRY, "proc-pd": PdStub},
        )
        self.addCleanup(session.close)
        session.registerPolicyController({
            "nodes": [
                {"id": "c", "type": "obs-constant", "data": {"values": [1.0, 2.0]}},
                {"id": "pd", "type": "proc-pd", "data": {}},
                {"id": "sink", "type": "out-scope", "data": {}},
            ],
            "edges": [
                {"id": "e1", "source": "c", "target": "pd"},
                {"id": "e2", "source": "pd", "target": "sink"},
            ],
            "output": {"policyActionNodeId": "c", "pdNodeId": "pd"},
        })
        session.run()
        pd_calls = [c for c in physics.calls if isinstance(c, tuple) and c[0] == "step"]
        self.assertEqual(len(pd_calls), 1)
        _label, _steps, ctrl, substep_pd = pd_calls[0]
        self.assertIsNone(ctrl)
        self.assertEqual(substep_pd["target"], [0.5])

    def test_missing_policy_output_is_explicit(self):
        class NoOutputSource(ConstantSource):
            def init(self, config, input_dims):
                ConstantSource.init(self, config, input_dims)

        session, _physics = self.make_session()
        plan = simple_plan(output={"policyActionNodeId": "ghost"})
        session.registerPolicyController(plan)
        with self.assertRaises(RuntimeError) as ctx:
            session.run()
        self.assertIn("policy action output is unavailable", str(ctx.exception))

    def test_modifier_without_hook_node_is_explicit(self):
        session, _physics = self.make_session()
        with self.assertRaises(RuntimeError) as ctx:
            session.registerPolicyController(simple_plan(), modifier=lambda action: action)
        self.assertIn("no proc-action-hook node", str(ctx.exception))

    def test_run_without_controller_is_explicit(self):
        session, _physics = self.make_session()
        with self.assertRaises(RuntimeError) as ctx:
            session.run()
        self.assertIn("no controller", str(ctx.exception))

    def test_bad_frequency_ratio_is_explicit(self):
        session, _physics = self.make_session(control_hz=30, physics_hz=200)
        session.registerPolicyController(simple_plan())
        with self.assertRaises(RuntimeError) as ctx:
            session.run()
        self.assertIn("multiple of control_hz", str(ctx.exception))

    def test_motion_data_is_injected_before_compile(self):
        session, physics = self.make_session()
        plan = {
            "nodes": [
                {
                    "id": "m",
                    "type": "proc-motion-sample",
                    "data": {"motionSource": "walk.npz"},
                },
                {"id": "sink", "type": "out-scope", "data": {}},
            ],
            "edges": [
                {"id": "e1", "source": "m", "target": "sink",
                 "sourcePort": "out-0", "targetPort": "in-0"},
            ],
        }
        # The fake registry skips real motion validation; assert the injection
        # channel (config["motionData"]) that the real processor consumes.
        captured = {}

        class MotionProbe:
            def __init__(self):
                self.output_buffers = [[0.0]]

            def init(self, config, input_dims):
                captured.update(config)

            def compute(self, inputs):
                pass

        session._registry = {**FAKE_REGISTRY, "proc-motion-sample": MotionProbe}
        session.registerPolicyController(plan)
        self.assertIn(("load_motion", "walk.npz"), physics.calls)
        self.assertEqual(captured["motionData"], {"marker": "walk.npz"})
        # The plan dict itself stays untouched apart from the injected copy.
        self.assertNotIn("motionData", plan["nodes"][0]["data"])


class SessionCustomLoopTest(unittest.TestCase):
    def tearDown(self):
        botlab_host.set_runtime_session(None)

    def test_custom_controller_observe_apply_substeps_each_step(self):
        physics = SpyPhysics(stop_after=2)
        session = _host_runtime.LocalRuntimeSession(physics, registry=FAKE_REGISTRY)
        self.addCleanup(session.close)
        seen = []

        def step_callback(dt, time, index):
            observation = session.observe("robot-1")
            seen.append((observation, dt, time, index))
            session.applyAction("robot-1", [float(index)])

        session.registerStep(step_callback, ["robot-1"])
        steps = session.run()
        self.assertEqual(steps, 2)
        self.assertEqual(seen[0], ([1.0, 2.0, 3.0], 0.02, 0.0, 0))
        self.assertEqual(physics.applied, [[0.0], [1.0]])
        self.assertEqual(
            physics.calls[:5],
            ["reset", ("observe", "robot-1"), ("apply", "robot-1", [0.0]), ("step", 4, None, None), "after"],
        )


class SessionDelegationTest(unittest.TestCase):
    def tearDown(self):
        botlab_host.set_runtime_session(None)

    def test_request_reset_and_metrics_route_through_session(self):
        physics = SpyPhysics()
        display = FakeDisplay()
        session = _host_runtime.LocalRuntimeSession(physics, display=display, registry=FAKE_REGISTRY)
        self.addCleanup(session.close)
        botlab_host.set_runtime_session(session)
        botlab_host.requestReset("test reset")
        botlab_host.updateMetrics({"speed": 1.5})
        self.assertIn(("request_reset", "test reset"), physics.calls)
        self.assertEqual(display.metrics, [{"speed": 1.5}])


class RunnerConfigurationTest(unittest.TestCase):
    def test_display_port_accepts_only_explicit_tcp_port_range(self):
        self.assertIsNone(runtime_run._display_port({}))
        self.assertIsNone(runtime_run._display_port({"BOTLAB_DISPLAY_PORT": "  "}))
        self.assertEqual(
            runtime_run._display_port({"BOTLAB_DISPLAY_PORT": " 8765 "}), 8765
        )
        for value in ("0", "65536", "1.5", "abc", "-1", "+5"):
            with self.subTest(value=value), self.assertRaises(RuntimeError) as ctx:
                runtime_run._display_port({"BOTLAB_DISPLAY_PORT": value})
            self.assertIn("integer from 1 to 65535", str(ctx.exception))

    def test_runner_preserves_primary_error_and_releases_session_and_display(self):
        manifest = {
            "name": "broken",
            "control_hz": 50,
            "physics_hz": 200,
            "resources": [{
                "id": "robot", "kind": "robot", "source": "robot.xml", "config": {},
            }],
        }
        state = mock.Mock()
        state.manifest.return_value = manifest
        physics = SimpleNamespace(
            model=object(),
            _mujoco=object(),
            data=SimpleNamespace(
                qpos=SimpleNamespace(shape=(1,)),
                qvel=SimpleNamespace(shape=(1,)),
            ),
        )
        session = mock.Mock()
        session.close.side_effect = RuntimeError("close failed")
        display = mock.Mock(port=8765)
        original_directory = os.getcwd()
        primary_error = RuntimeError("main failed")

        with (
            mock.patch.object(runtime_run, "_parse_args", return_value=("/project", None)),
            mock.patch.object(runtime_run, "_require_runtime"),
            mock.patch.object(runtime_run, "_declare_project", return_value=state),
            mock.patch.object(runtime_run, "_build_physics", return_value=physics),
            mock.patch.object(runtime_run.runpy, "run_path", side_effect=primary_error),
            mock.patch(
                "botlab._runtime._host_runtime.LocalRuntimeSession",
                return_value=session,
            ),
            mock.patch("botlab._runtime._host_runtime.PooledModelRuntime"),
            mock.patch("botlab._runtime._display.DisplayServer", return_value=display),
            mock.patch("botlab._runtime.mujoco_scene.MujocoSceneProvider"),
            mock.patch.dict(os.environ, {"BOTLAB_DISPLAY": "1"}, clear=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "main failed") as raised:
                runtime_run.main(["botlab.run", "/project"])

        self.assertIs(raised.exception, primary_error)
        if hasattr(primary_error, "__notes__"):
            self.assertIn("cleanup failed: close failed", primary_error.__notes__[0])
        self.assertIsNone(botlab_host._runtime_session)
        session.close.assert_called_once_with()
        display.stop.assert_called_once_with()
        self.assertEqual(os.getcwd(), original_directory)


class PooledModelRuntimePathTest(unittest.TestCase):
    def tearDown(self):
        botlab_host.project_state().reset()

    def test_custom_predict_resolves_declared_source_with_runner_resolver(self):
        class Input:
            name = "input"
            shape = [1, 2]

        class Session:
            def get_inputs(self):
                return [Input()]

            def run(self, _outputs, values):
                self.values = values
                return [[3.0]]

        class Pool:
            def __init__(self):
                self.path = None
                self.session = Session()

            def get(self, resource_id, path):
                self.path = (resource_id, path)
                return self.session

        botlab_host.project_state().reset("/project")
        botlab_host.declareResource(
            "model", "actor", "models/actor.onnx", None, None, {}
        )
        pool = Pool()
        runtime = _host_runtime.PooledModelRuntime(
            pool=pool, path_resolver=lambda source: "/assets/" + source
        )
        result = asyncio.run(runtime.predict("actor", [1.0, 2.0]))
        self.assertEqual(pool.path, ("actor", "/assets/models/actor.onnx"))
        self.assertEqual(result, [3.0])

    def test_predict_model_without_model_runtime_is_explicit(self):
        session = _host_runtime.LocalRuntimeSession(SpyPhysics(), registry=FAKE_REGISTRY)
        self.addCleanup(session.close)
        with self.assertRaises(RuntimeError) as ctx:
            session._loop.run_until_complete(session.predictModel("actor.onnx", [0.0]))
        self.assertIn("without one", str(ctx.exception))

    def test_async_custom_controller_step_drives_through_session_loop(self):
        physics = SpyPhysics(stop_after=2)
        session = _host_runtime.LocalRuntimeSession(physics, registry=FAKE_REGISTRY)
        self.addCleanup(session.close)

        async def step_callback(dt, time, index):
            await asyncio.sleep(0)
            observation = await session_loop_probe(session)
            session.applyAction("robot-1", [float(observation[0]) + index])

        async def session_loop_probe(current_session):
            return current_session.observe("robot-1")

        session.registerStep(step_callback, ["robot-1"])
        self.assertEqual(session.run(), 2)
        self.assertEqual(physics.applied, [[1.0], [2.0]])

    def test_async_action_hook_modifier_runs_inside_plan_loop(self):
        from botlab._runtime._processors._output import ActionHookProcessor

        physics = SpyPhysics(stop_after=1)
        session = _host_runtime.LocalRuntimeSession(
            physics,
            registry={**FAKE_REGISTRY, "proc-action-hook": ActionHookProcessor},
        )
        self.addCleanup(session.close)
        plan = {
            "nodes": [
                {"id": "c", "type": "obs-constant", "data": {"values": [3.0]}},
                {"id": "hook", "type": "proc-action-hook", "data": {}},
                {"id": "sink", "type": "out-scope", "data": {}},
            ],
            "edges": [
                {"id": "e1", "source": "c", "target": "hook"},
                {"id": "e2", "source": "hook", "target": "sink"},
            ],
            "output": {
                "policyActionNodeId": "hook",
                "kind": "raw",
                "controlNodeId": "hook",
                "controlOutputSlot": 0,
            },
        }

        async def modifier(action, frame):
            await asyncio.sleep(0)
            return [value * 2 for value in action]

        session.registerPolicyController(plan, modifier=modifier)
        session.run()
        sink = session._executor.get_processor("sink")
        self.assertEqual(sink.received, [[[6.0]]])


class GroupSpyPhysics(SpyPhysics):
    """SpyPhysics + 多机器人 scope/layout 访问器（raw ctrl 长度可配）。"""

    def __init__(self, layouts, ctrl_length=4, **kwargs):
        super().__init__(**kwargs)
        self._layouts = layouts
        self.data = SimpleNamespace(ctrl=[0.0] * ctrl_length)

    def robot_layout(self, robot_id):
        return self._layouts[robot_id]

    def robot_scope(self, robot_id):
        layout = self._layouts[robot_id]
        return {
            "qpos_start": layout["qpos_start"],
            "qvel_start": layout["qvel_start"],
            "actuator_start": layout["actuator_start"],
            "name_prefix": layout.get("name_prefix", ""),
        }

    def compose_payloads(self, actions):
        from botlab._runtime._physics_ctrl import compose_scoped_actions

        return compose_scoped_actions(self._layouts, actions, self.data.ctrl)


GROUP_LAYOUTS = {
    "first": {
        "dimension": 2, "qpos_offset": 7, "qvel_offset": 6,
        "actuator_start": 0, "qpos_start": 0, "qvel_start": 0, "name_prefix": "",
    },
    "second": {
        "dimension": 2, "qpos_offset": 22, "qvel_offset": 19,
        "actuator_start": 2, "qpos_start": 15, "qvel_start": 13, "name_prefix": "r1_",
    },
}


class ObsStub:
    """is_observation 标记的观测桩：记录 set_data 收到的值。"""

    is_observation = True

    def __init__(self):
        self.output_buffers = [[0.0, 0.0]]
        self.seen = []

    def init(self, config, input_dims):
        pass

    def set_data(self, values):
        self.seen.append(list(values))

    def compute(self, inputs):
        pass


def _group_plan(robot_id, values, kind="raw"):
    output = {"policyActionNodeId": "c"}
    if kind == "raw":
        output.update({"kind": "raw", "controlNodeId": "c"})
    return {
        "robot": robot_id,
        "nodes": [
            {"id": "c", "type": "obs-constant", "data": {"values": values}},
            {"id": "sink", "type": "out-scope", "data": {}},
        ],
        "edges": [{"id": "e", "source": "c", "target": "sink"}],
        "output": output,
    }


def _declare_two_robots():
    import tempfile

    directory = tempfile.mkdtemp(prefix="botlab-group-test-")
    botlab_host.project_state().reset(directory)
    for rid in ("first", "second"):
        botlab_host.declareResource(
            "robot", rid, f"{rid}.xml", None, None,
            {"actionDimension": 2, "jointQposOffset": 7, "jointQvelOffset": 6},
        )


class ImuSensorsPlanLoadTest(unittest.TestCase):
    """imuSensors is resolved once at plan load (mujocoPolicyController.parseNode)."""

    def captured(self, robot, flag="absent"):
        seen = {}

        class Probe(ConstantSource):
            def init(self, config, input_dims):
                seen.update(config)
                ConstantSource.init(self, config, input_dims)

        data = {"values": [1.0]} if flag == "absent" else {"values": [1.0], "imuSensors": flag}
        plan = {
            "nodes": [{"id": "c", "type": "obs-constant", "data": data},
                      {"id": "sink", "type": "out-scope", "data": {}}],
            "edges": [{"id": "e", "source": "c", "target": "sink"}],
        }
        if robot is not None:
            plan["robot"] = robot
        session = _host_runtime.LocalRuntimeSession(
            SpyPhysics(), registry={**FAKE_REGISTRY, "obs-constant": Probe}
        )
        self.addCleanup(session.close)
        session.registerPolicyController(plan)
        return seen.get("imuSensors")

    def test_missing_field_follows_legacy_g1_rule(self):
        self.assertIs(self.captured("g1"), True)
        self.assertIs(self.captured("robot"), False)
        self.assertIs(self.captured(None), False)

    def test_explicit_field_wins_both_ways(self):
        self.assertIs(self.captured("g1", False), False)
        self.assertIs(self.captured("robot", True), True)

    def test_non_boolean_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "imuSensors must be a boolean"):
            self.captured("robot", "yes")


class SessionGroupLoopTest(unittest.TestCase):
    """Native controller group：逐机器人 plan + 组合 payload（对照 TS scoped actions）。"""

    def tearDown(self):
        botlab_host.set_runtime_session(None)
        botlab_host.project_state().reset()

    def make_session(self, **kwargs):
        physics = GroupSpyPhysics(GROUP_LAYOUTS, **kwargs.pop("physics_kwargs", {}))
        session = _host_runtime.LocalRuntimeSession(
            physics, registry={**FAKE_REGISTRY, "obs-stub": ObsStub},
            control_hz=kwargs.pop("control_hz", 50),
            physics_hz=kwargs.pop("physics_hz", 200),
            **kwargs,
        )
        self.addCleanup(session.close)
        return session, physics

    def test_group_runs_each_robot_plan_and_composes_ctrl(self):
        _declare_two_robots()
        session, physics = self.make_session(physics_kwargs={"stop_after": 2})
        session.registerPolicyController({
            "version": 1,
            "controllers": [
                _group_plan("first", [1.0, 2.0]),
                _group_plan("second", [9.0, 8.0]),
            ],
        })
        steps = session.run()
        self.assertEqual(steps, 2)
        step_calls = [c for c in physics.calls if isinstance(c, tuple) and c[0] == "step"]
        self.assertEqual(step_calls[0], ("step", 4, [1.0, 2.0, 9.0, 8.0], None))
        # 每个控制步：两个机器人各 fill 一次（顺序 = 声明顺序），然后一次组合 payload
        first_cycle = physics.calls[physics.calls.index("fill"):physics.calls.index("after") + 1]
        self.assertEqual(
            first_cycle,
            ["fill", "fill", ("step", 4, [1.0, 2.0, 9.0, 8.0], None), "after"],
        )

    def test_group_scopes_observation_bindings(self):
        _declare_two_robots()
        session, _physics = self.make_session(physics_kwargs={"stop_after": 1})
        plans = []
        for rid in ("first", "second"):
            plan = _group_plan(rid, [1.0, 2.0])
            plan["nodes"].insert(
                0, {"id": "obs", "type": "obs-stub", "data": {"dim": 2}}
            )
            plan["edges"] = [
                {"id": "e1", "source": "c", "target": "obs"},
                {"id": "e2", "source": "obs", "target": "sink"},
            ]
            plans.append(plan)
        session.registerPolicyController({"version": 1, "controllers": plans})
        second_executor = session._groups[1]["executor"]
        binding = second_executor.observation_bindings[0]
        self.assertEqual(binding["qposOffset"], 15)
        self.assertEqual(binding["qvelOffset"], 13)
        self.assertEqual(binding["namePrefix"], "r1_")

    def test_group_unresolvable_robot_reference_is_explicit(self):
        _declare_two_robots()
        session, _physics = self.make_session()
        plans = []
        for rid in ("first", "second"):
            plan = _group_plan(rid, [1.0, 2.0])
            plan["nodes"].insert(
                0, {"id": "obs", "type": "obs-stub", "data": {"dim": 2, "robot": "missing"}}
            )
            plan["edges"] = [
                {"id": "e1", "source": "c", "target": "obs"},
                {"id": "e2", "source": "obs", "target": "sink"},
            ]
            plans.append(plan)
        with self.assertRaisesRegex(
            RuntimeError, 'references robot "missing", which has no scene offsets'
        ):
            session.registerPolicyController({"version": 1, "controllers": plans})

    def test_group_robot_reference_selects_its_scope(self):
        _declare_two_robots()
        session, _physics = self.make_session()
        plans = []
        for rid, ref in (("first", None), ("second", "first")):
            plan = _group_plan(rid, [1.0, 2.0])
            data = {"dim": 2} if ref is None else {"dim": 2, "robot": ref}
            plan["nodes"].insert(0, {"id": "obs", "type": "obs-stub", "data": data})
            plan["edges"] = [
                {"id": "e1", "source": "c", "target": "obs"},
                {"id": "e2", "source": "obs", "target": "sink"},
            ]
            plans.append(plan)
        session.registerPolicyController({"version": 1, "controllers": plans})
        binding = session._groups[1]["executor"].observation_bindings[0]
        self.assertEqual((binding["qposOffset"], binding["qvelOffset"]), (0, 0))

    def test_group_requires_full_coverage(self):
        _declare_two_robots()
        session, _physics = self.make_session()
        with self.assertRaisesRegex(RuntimeError, "cover every declared robot"):
            session.registerPolicyController({
                "version": 1,
                "controllers": [_group_plan("first", [1.0, 2.0])],
            })

    def test_group_rejects_duplicate_robots(self):
        _declare_two_robots()
        session, _physics = self.make_session()
        with self.assertRaisesRegex(RuntimeError, "distinct robots"):
            session.registerPolicyController({
                "version": 1,
                "controllers": [
                    _group_plan("first", [1.0, 2.0]),
                    _group_plan("first", [3.0, 4.0]),
                ],
            })

    def test_group_merges_pd_with_absolute_indices(self):
        _declare_two_robots()

        class PdStub:
            def __init__(self):
                self.output_buffers = [[0.0, 0.0]]

            def init(self, config, input_dims):
                pass

            def compute(self, inputs):
                pass

            def get_substep_pd(self):
                return {
                    "target": [0.5, 0.6], "kps": [40.0, 40.0], "kds": [4.0, 4.0],
                    "effort_min": [-10.0, -10.0], "effort_max": [10.0, 10.0],
                }

        physics = GroupSpyPhysics(GROUP_LAYOUTS, stop_after=1)
        session = _host_runtime.LocalRuntimeSession(
            physics, registry={**FAKE_REGISTRY, "proc-pd": PdStub},
        )
        self.addCleanup(session.close)
        plans = []
        for rid in ("first", "second"):
            plan = _group_plan(rid, [1.0, 2.0], kind="pd")
            plan["nodes"].insert(1, {"id": "pd", "type": "proc-pd", "data": {}})
            plan["edges"] = [
                {"id": "e1", "source": "c", "target": "pd"},
                {"id": "e2", "source": "pd", "target": "sink"},
            ]
            plan["output"] = {"policyActionNodeId": "c", "pdNodeId": "pd"}
            plans.append(plan)
        session.registerPolicyController({"version": 1, "controllers": plans})
        session.run()
        step_calls = [c for c in physics.calls if isinstance(c, tuple) and c[0] == "step"]
        _label, _steps, ctrl, substep_pd = step_calls[0]
        self.assertIsNone(ctrl)
        self.assertEqual(list(substep_pd["target"]), [0.5, 0.6, 0.5, 0.6])
        # first: qpos 偏移 7 / ctrl 0；second: scope 平移后 22 / 2
        self.assertEqual(list(substep_pd["qpos_indices"]), [7, 8, 22, 23])
        self.assertEqual(list(substep_pd["qvel_indices"]), [6, 7, 19, 20])
        self.assertEqual(list(substep_pd["ctrl_indices"]), [0, 1, 2, 3])

    def test_group_actuator_overlap_is_explicit(self):
        _declare_two_robots()
        overlapping = {
            "first": dict(GROUP_LAYOUTS["first"]),
            "second": {**GROUP_LAYOUTS["second"], "actuator_start": 1},
        }
        physics = GroupSpyPhysics(overlapping, stop_after=1)
        session = _host_runtime.LocalRuntimeSession(physics, registry=FAKE_REGISTRY)
        self.addCleanup(session.close)
        session.registerPolicyController({
            "version": 1,
            "controllers": [
                _group_plan("first", [1.0, 2.0]),
                _group_plan("second", [9.0, 8.0]),
            ],
        })
        with self.assertRaisesRegex(RuntimeError, "actuator scopes overlap"):
            session.run()


if __name__ == "__main__":
    unittest.main()
