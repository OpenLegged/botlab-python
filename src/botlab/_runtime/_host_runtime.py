"""Local runtime session: the third botlab_host state.

Offline host: declarations + compiled plan only. Browser: the TypeScript
bridge replaces the host module outright. Here: a session owns the plan
executor, a physics adapter (native MuJoCo), optional ONNX models and an
optional display server, and drives the control loop with the same semantics
as the browser stack — observations fill at the control-step boundary, the
graph executes once, the output reaches physics as ctrl or a substep PD view,
then physics substeps run for the rest of the control period.
"""

import asyncio
import inspect
import os

from ..sdk._runtime_types import Frame


class PooledModelRuntime:
    """predictModel backing for custom controllers: shared ONNX session pool.

    Paths resolve through botlab_host.resourceSource (the declaration state
    knows every resource source); shapes go through the same dynamic-dim
    inference the model node uses.
    """

    def __init__(self, pool=None, path_resolver=None):
        if pool is None:
            from ._processors._model import shared_session_pool

            pool = shared_session_pool()
        self._pool = pool
        self._path_resolver = path_resolver

    async def predict(self, resource_id, values):
        import botlab_host
        import numpy as np

        from ._processors._model import input_tensor_dims_from_metadata

        source = botlab_host.resourceSource(resource_id)
        path = (
            self._path_resolver(source)
            if self._path_resolver is not None
            else source
        )
        session = self._pool.get(resource_id, path)
        array = np.asarray(values, dtype=np.float32)
        inputs = session.get_inputs()
        primary = inputs[0]
        metadata = {"is_tensor": True, "shape": list(primary.shape)}
        dims = input_tensor_dims_from_metadata(
            metadata, f'input "{primary.name}"', int(array.size)
        )
        outputs = session.run(None, {primary.name: array.reshape(dims)})
        return np.asarray(outputs[0], dtype=np.float32).ravel().tolist()


class LocalRuntimeSession:
    """Physics-agnostic driver; adapters provide MuJoCo, models and display.

    The SDK's controller callbacks and Model.predict are async (the browser
    bridge returns promises); this session owns one event loop and drives
    them to completion per control step.
    """

    def __init__(self, physics, display=None, models=None, registry=None,
                 control_hz=50, physics_hz=200, path_resolver=None):
        self._physics = physics
        self._display = display
        self._models = models
        self._registry = registry
        self._path_resolver = path_resolver
        self._executor = None
        self._groups = []
        self._plan_nodes = None
        self._plan_output = {}
        self._modifier = None
        self._reset_callback = None
        self._step_callback = None
        self._step_robot_ids = None
        self._current_frame = None
        self._recorder = None
        self._substep_observer = None
        self._custom_step_active = False
        self._custom_step_actions = set()
        self._control_hz = control_hz
        self._physics_hz = physics_hz
        self._loop = asyncio.new_event_loop()
        self._closed = False

    def _call_maybe_async(self, result):
        if inspect.isawaitable(result):
            return self._loop.run_until_complete(result)
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        if self._executor is not None:
            try:
                self._executor.dispose()
            except Exception as error:
                errors.append(error)
        for entry in self._groups:
            try:
                entry["executor"].dispose()
            except Exception as error:
                errors.append(error)
        self._groups = []
        try:
            from ._observations import dispose_camera_renderers

            dispose_camera_renderers()
        except Exception as error:
            errors.append(error)
        try:
            self._loop.close()
        except Exception as error:
            errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RuntimeError(
                "[botlab] local runtime cleanup failed: "
                + "; ".join(str(error) for error in errors)
            )

    # -- registration surface (botlab_host delegates here) --------------------

    def registerPolicyController(self, plan, modifier=None):
        from ._executor import PlanExecutor

        registry = self._registry
        if registry is None:
            from ._processors import build_registry

            registry = build_registry()

        controllers = plan.get("controllers") if isinstance(plan, dict) else None
        if controllers is not None:
            self._register_controller_group(controllers, registry)
            return

        executor = PlanExecutor(registry)
        nodes = self._resolve_plan_references(plan["nodes"])
        edges = self._resolve_plan_references(plan["edges"])
        nodes = self._inject_motion_data(nodes)
        self._resolve_model_paths(nodes)
        executor.compile(nodes, edges)
        self._executor = executor
        self._plan_nodes = nodes
        self._plan_output = plan.get("output") or {}
        self._modifier = modifier
        if modifier is not None:
            self._attach_action_hook(modifier)

    def _compile_sub_plan(self, registry, sub_plan):
        from ._executor import PlanExecutor

        executor = PlanExecutor(registry)
        nodes = self._resolve_plan_references(sub_plan["nodes"])
        edges = self._resolve_plan_references(sub_plan["edges"])
        nodes = self._inject_motion_data(nodes)
        self._resolve_model_paths(nodes)
        executor.compile(nodes, edges)
        return executor, nodes

    def _register_controller_group(self, controllers, registry):
        """Native controller group: one compiled plan per declared robot.

        Mirrors the browser bridge (botlabBridge.ts): each sub-plan is
        compiled independently and drives its own robot scope; observation
        bindings get the robot's composed-model scope so reads and writes
        land in the right slices.
        """
        import botlab_host

        if not isinstance(controllers, list) or not controllers:
            raise RuntimeError(
                "[botlab] PolicyController group plan.controllers must be a non-empty list"
            )
        declared = [
            item["id"] for item in botlab_host.project_state().manifest()["resources"]
            if item["kind"] == "robot"
        ]
        plan_robot_ids = []
        for index, sub_plan in enumerate(controllers):
            robot_id = sub_plan.get("robot") if isinstance(sub_plan, dict) else None
            if not isinstance(robot_id, str) or not robot_id:
                raise RuntimeError(
                    f"[botlab] group controller {index} is missing its robot id"
                )
            plan_robot_ids.append(robot_id)
        if len(plan_robot_ids) != len(set(plan_robot_ids)):
            raise RuntimeError(
                "[botlab] Simulation.run controllers must target distinct robots"
            )
        missing = [robot_id for robot_id in declared if robot_id not in plan_robot_ids]
        unknown = [robot_id for robot_id in plan_robot_ids if robot_id not in declared]
        if missing or unknown:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise RuntimeError(
                "[botlab] PolicyController plans must cover every declared robot "
                f"exactly once ({'; '.join(details)})"
            )

        groups = []
        try:
            for sub_plan, robot_id in zip(controllers, plan_robot_ids, strict=True):
                executor, nodes = self._compile_sub_plan(registry, sub_plan)
                scope = self._physics.robot_scope(robot_id)
                for binding in executor.observation_bindings:
                    binding["qposOffset"] = scope["qpos_start"]
                    binding["qvelOffset"] = scope["qvel_start"]
                    binding["namePrefix"] = scope["name_prefix"]
                groups.append({
                    "robot_id": robot_id,
                    "executor": executor,
                    "nodes": nodes,
                    "output": sub_plan.get("output") or {},
                })
        except Exception:
            for entry in groups:
                entry["executor"].dispose()
            raise
        self._groups = groups

    def _inject_motion_data(self, nodes):
        """Motion bytes load host-side (browser: asset registry); inject locally.

        Returns a new node list — the caller's plan stays untouched, matching
        the offline host's deep-copy behavior.
        """
        motion_types = ("proc-motion-reference", "proc-motion-sample")
        injected = []
        for node in nodes:
            data = node.get("data") or {}
            if (
                node.get("type") in motion_types
                and "motionData" not in data
                and "motion_data" not in data
            ):
                source = data.get("motionSource") or data.get("motionPath")
                if source:
                    enriched = dict(data)
                    enriched["motionData"] = self._physics.load_motion(str(source))
                    injected.append(dict(node, data=enriched))
                    continue
            injected.append(node)
        return injected

    def _resolve_plan_references(self, value):
        """Mirror mujocoPolicyController.resolvePlanValue: swap ``$botlab``
        plan references for resource config values at registration time."""
        import botlab_host

        if isinstance(value, list):
            return [self._resolve_plan_references(item) for item in value]
        if not isinstance(value, dict):
            return value

        reference_kind = value.get("$botlab")
        if reference_kind == "resource-config":
            resource_id = value["resource"]
            field = value["field"]
            config = botlab_host.project_state().resource(resource_id)["config"]
            if field not in config:
                raise RuntimeError(
                    f'[botlab] Resource "{resource_id}" has no config field "{field}"'
                )
            return config[field]
        if reference_kind == "optional-resource-config":
            resource_id = value["resource"]
            config = botlab_host.project_state().resource(resource_id)["config"]
            return config.get(value["field"])
        if reference_kind == "default-joint-positions":
            return botlab_host.defaultJointPositions(value["resource"])
        if reference_kind is not None:
            raise RuntimeError(
                f"[botlab] plan has unknown BotLab reference \"{reference_kind}\""
            )
        return {key: self._resolve_plan_references(item) for key, item in value.items()}

    def _resolve_model_paths(self, nodes):
        """ONNX sources resolve through the runner's path resolver (shared
        assets root) before the model pool loads them."""
        if self._path_resolver is None:
            return
        import botlab_host

        for node in nodes:
            if node.get("type") != "model-onnx":
                continue
            data = node.get("data") or {}
            path = data.get("modelPath") or data.get("model_path")
            if isinstance(path, str) and os.path.isabs(path):
                continue
            resource_id = data.get("resourceId") or data.get("resource_id") or node.get("id")
            try:
                source = botlab_host.resourceSource(resource_id)
            except Exception:
                source = path if isinstance(path, str) else None
            if isinstance(source, str) and source:
                enriched = dict(data)
                enriched["modelPath"] = self._path_resolver(source)
                node["data"] = enriched

    def _attach_action_hook(self, modifier):
        hook_nodes = [node for node in self._plan_nodes if node["type"] == "proc-action-hook"]
        if not hook_nodes:
            raise RuntimeError(
                "plan carries a Signal.action(modify=...) but has no proc-action-hook node"
            )
        for node in hook_nodes:
            processor = self._executor.get_processor(node["id"])
            if processor is None:
                raise RuntimeError(f"action hook node {node['id']} was not compiled")
            # ActionHookProcessor calls run(action) with one argument; the SDK
            # modifier signature is modify(action, frame) and may be async, so
            # close over the frame the loop publishes before each step and
            # drive awaitables on the session loop.
            def run(action, _modifier=modifier):
                result = _modifier(list(action), self._current_frame)
                if inspect.isawaitable(result):
                    return self._loop.run_until_complete(result)
                return result

            processor.run = run

    def registerSubstepObserver(self, callback):
        if self._substep_observer is not None:
            raise RuntimeError("[botlab] on_substep accepts exactly one observer")
        def drive(frame):
            result = callback(frame)
            if inspect.isawaitable(result):
                return self._loop.run_until_complete(result)
            return result
        self._substep_observer = drive

    def registerReset(self, callback):
        self._reset_callback = callback

    def registerStep(self, callback, robot_ids=None):
        import botlab_host

        declared = [
            item["id"] for item in botlab_host.project_state().manifest()["resources"]
            if item["kind"] == "robot"
        ]
        ids = declared if robot_ids is None and len(declared) == 1 else list(robot_ids or [])
        if not ids:
            raise RuntimeError(
                "[botlab] Custom controller robot IDs must be a non-empty list"
            )
        if len(ids) != len(set(ids)):
            raise RuntimeError(
                "[botlab] Simulation.run controllers must target distinct robots"
            )
        missing = [robot_id for robot_id in declared if robot_id not in ids]
        unknown = (
            [robot_id for robot_id in ids if robot_id not in declared]
            if declared else []
        )
        if missing or unknown:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise RuntimeError(
                "[botlab] Custom controllers must cover every declared robot exactly "
                f"once ({'; '.join(details)})"
            )
        self._step_callback = callback
        self._step_robot_ids = ids

    # -- runtime host surface (custom controllers) ------------------------------

    def observe(self, robot_id):
        return self._physics.robot_observation(robot_id)

    def observeSensor(self, sensor_id, channel=None):
        return self._physics.sensor_observation(sensor_id, channel)

    def applyAction(self, robot_id, action):
        if self._custom_step_active:
            if robot_id not in self._step_robot_ids:
                raise RuntimeError(
                    f'[botlab] No custom controller is registered for robot "{robot_id}"'
                )
            if robot_id in self._custom_step_actions:
                raise RuntimeError(
                    f'[botlab] Controller.step must produce exactly one action for robot "{robot_id}"'
                )
        self._physics.apply_action(robot_id, action)
        if self._custom_step_active:
            self._custom_step_actions.add(robot_id)

    async def predictModel(self, resource_id, inputs):
        if self._models is None:
            raise RuntimeError(
                f"predictModel({resource_id}) needs a model runtime; "
                "the local session was started without one"
            )
        return await self._models.predict(resource_id, inputs)

    def readMujoco(self, robot_id, query, target=None, start=None, length=None):
        return self._physics.read_mujoco(robot_id, query, target, start, length)

    def loadMotion(self, motion_id):
        return self._physics.load_motion(motion_id)

    def updateHeightfield(self, terrain_id, heights):
        return self._physics.update_heightfield(terrain_id, heights)

    def updateMetrics(self, metrics):
        if self._recorder is not None:
            frame = self._current_frame
            self._recorder.record_metrics(frame.time if frame else 0.0, metrics)
        if self._display is not None:
            self._display.update_metrics(metrics)

    def updateDiagnostics(self, diagnostics):
        if self._display is not None:
            self._display.update_diagnostics(diagnostics)

    def requestReset(self, reason, spec=None):
        result = self._physics.request_reset(reason, spec)
        if self._recorder is not None:
            frame = self._current_frame
            self._recorder.record_reset(frame.time if frame else 0.0, reason)
        return result

    def set_recorder(self, recorder):
        self._recorder = recorder

    # -- control loop -------------------------------------------------------------

    def run(self, max_control_steps=None):
        if self._groups:
            return self._run_group(max_control_steps)
        if self._executor is not None:
            return self._run_plan(max_control_steps)
        if self._step_callback is not None:
            return self._run_custom(max_control_steps)
        raise RuntimeError(
            "local runtime has no controller; call Simulation.run() before running"
        )

    @property
    def decimation(self):
        if self._control_hz <= 0 or self._physics_hz <= 0:
            raise RuntimeError("control_hz and physics_hz must be positive")
        if self._physics_hz % self._control_hz != 0:
            raise RuntimeError(
                f"physics_hz {self._physics_hz} must be a multiple of control_hz {self._control_hz}"
            )
        return self._physics_hz // self._control_hz

    def _policy_action(self):
        node_id = self._plan_output.get("policyActionNodeId")
        processor = self._executor.get_processor(node_id) if node_id else None
        buffer = processor.output_buffers[0] if processor else None
        if buffer is None:
            raise RuntimeError(
                "[botlab] PolicyController policy action output is unavailable"
            )
        return buffer

    def _group_control_payload(self):
        """Read each robot's plan output and compose one physics payload.

        Mirrors composePythonRobotActions (src/domain/python/pythonScopedActions.ts):
        the scope math and overlap checks live in MujocoPhysics.compose_payloads.
        """
        import numpy as np

        physics = self._physics
        actions = {}
        for entry in self._groups:
            robot_id = entry["robot_id"]
            layout = physics.robot_layout(robot_id)
            dimension = layout["dimension"]
            output = entry["output"]
            executor = entry["executor"]
            if output.get("kind") == "raw":
                node_id = output.get("controlNodeId")
                processor = executor.get_processor(node_id) if node_id else None
                slot = int(output.get("controlOutputSlot") or 0)
                buffer = (
                    processor.output_buffers[slot]
                    if processor and len(processor.output_buffers) > slot
                    else None
                )
                if buffer is None:
                    raise RuntimeError(
                        "[botlab] PolicyController raw control output is unavailable "
                        f'for robot "{robot_id}"'
                    )
                values = np.asarray(buffer, dtype=np.float64).ravel()
                if values.shape[0] != dimension:
                    raise RuntimeError(
                        f"[botlab] {robot_id} ctrl action must contain {dimension} values"
                    )
                actions[robot_id] = ("ctrl", values)
            else:
                node_id = output.get("pdNodeId")
                processor = executor.get_processor(node_id) if node_id else None
                view = processor.get_substep_pd() if processor is not None else None
                if view is None:
                    raise RuntimeError(
                        "[botlab] PolicyController PD output is unavailable "
                        f'for robot "{robot_id}"'
                    )
                actions[robot_id] = ("pd", view)
        ctrl, substep_pd = physics.compose_payloads(actions)
        return (ctrl.tolist() if ctrl is not None else None), substep_pd

    def _control_payload(self):
        output = self._plan_output
        if output.get("kind") == "raw":
            node_id = output.get("controlNodeId")
            processor = self._executor.get_processor(node_id) if node_id else None
            slot = int(output.get("controlOutputSlot") or 0)
            buffer = (
                processor.output_buffers[slot] if processor and len(processor.output_buffers) > slot else None
            )
            if buffer is None:
                raise RuntimeError(
                    "[botlab] PolicyController raw control output is unavailable"
                )
            return buffer, None
        node_id = output.get("pdNodeId")
        processor = self._executor.get_processor(node_id) if node_id else None
        view = processor.get_substep_pd() if processor is not None else None
        if view is None:
            raise RuntimeError("[botlab] PolicyController PD output is unavailable")
        return None, view

    def _fill_parameters(self, name):
        import botlab_host

        return botlab_host.parameter(name)

    def _run_plan(self, max_control_steps):
        import numpy as np

        from ._observations import clear_pose_velocity_states

        dt = 1.0 / self._control_hz
        if self._reset_callback is not None:
            self._call_maybe_async(self._reset_callback())
        clear_pose_velocity_states()
        self._executor.reset()
        self._physics.reset()
        last_action = None

        step_index = 0
        while max_control_steps is None or step_index < max_control_steps:
            frame = Frame(dt, step_index * dt, step_index)
            self._current_frame = frame
            self._physics.fill_observations(
                self._executor,
                last_action=last_action,
                parameters=self._fill_parameters,
            )
            self._executor.step(frame=frame)
            action = self._policy_action()
            last_action = np.array(action, dtype=np.float32)
            ctrl, substep_pd = self._control_payload()
            self._physics.step_substeps(
                self.decimation, ctrl=ctrl, substep_pd=substep_pd,
                observer=self._substep_observer,
            )
            self._physics.after_control_step()
            if self._display is not None:
                self._display.broadcast(
                    frame.time,
                    self._physics.qpos(),
                    self._physics.qvel(),
                )
            if self._recorder is not None:
                self._recorder.record_step(self._physics)
            step_index += 1
            if self._physics.should_stop():
                break
        return step_index

    def _run_group(self, max_control_steps):
        """Drive a native controller group: one compiled plan per robot."""
        import numpy as np

        from ._observations import clear_pose_velocity_states

        dt = 1.0 / self._control_hz
        if self._reset_callback is not None:
            self._call_maybe_async(self._reset_callback())
        clear_pose_velocity_states()
        for entry in self._groups:
            entry["executor"].reset()
        self._physics.reset()
        last_actions = {entry["robot_id"]: None for entry in self._groups}

        step_index = 0
        while max_control_steps is None or step_index < max_control_steps:
            frame = Frame(dt, step_index * dt, step_index)
            self._current_frame = frame
            for entry in self._groups:
                self._physics.fill_observations(
                    entry["executor"],
                    last_action=last_actions[entry["robot_id"]],
                    parameters=self._fill_parameters,
                )
            for entry in self._groups:
                entry["executor"].step(frame=frame)
            for entry in self._groups:
                node_id = entry["output"].get("policyActionNodeId")
                processor = (
                    entry["executor"].get_processor(node_id) if node_id else None
                )
                buffer = processor.output_buffers[0] if processor else None
                if buffer is None:
                    raise RuntimeError(
                        "[botlab] PolicyController policy action output is unavailable "
                        f'for robot "{entry["robot_id"]}"'
                    )
                last_actions[entry["robot_id"]] = np.array(buffer, dtype=np.float32)
            ctrl, substep_pd = self._group_control_payload()
            self._physics.step_substeps(
                self.decimation, ctrl=ctrl, substep_pd=substep_pd,
                observer=self._substep_observer,
            )
            self._physics.after_control_step()
            if self._display is not None:
                self._display.broadcast(
                    frame.time,
                    self._physics.qpos(),
                    self._physics.qvel(),
                )
            if self._recorder is not None:
                self._recorder.record_step(self._physics)
            step_index += 1
            if self._physics.should_stop():
                break
        return step_index

    def _run_custom(self, max_control_steps):
        dt = 1.0 / self._control_hz
        if self._reset_callback is not None:
            self._call_maybe_async(self._reset_callback())
        self._physics.reset()

        step_index = 0
        while max_control_steps is None or step_index < max_control_steps:
            frame = Frame(dt, step_index * dt, step_index)
            self._current_frame = frame
            self._custom_step_actions.clear()
            self._custom_step_active = True
            try:
                self._call_maybe_async(self._step_callback(dt, frame.time, frame.index))
            finally:
                self._custom_step_active = False
            missing = [
                robot_id for robot_id in self._step_robot_ids
                if robot_id not in self._custom_step_actions
            ]
            if missing:
                raise RuntimeError(
                    "[botlab] Controller.step must produce exactly one action for: "
                    + ", ".join(missing)
                )
            self._physics.step_substeps(self.decimation, observer=self._substep_observer)
            self._physics.after_control_step()
            if self._display is not None:
                self._display.broadcast(
                    frame.time,
                    self._physics.qpos(),
                    self._physics.qvel(),
                )
            if self._recorder is not None:
                self._recorder.record_step(self._physics)
            step_index += 1
            if self._physics.should_stop():
                break
        return step_index
