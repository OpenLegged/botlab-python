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
        self._plan_robot_id = None
        self._modifier = None
        self._reset_callback = None
        self._step_callback = None
        self._step_robot_ids = None
        self._current_frame = None
        self._in_plan_step = False
        self._pending_plan_reset = None
        self._plan_step_offset = 0
        self._completed_steps = 0
        self.completed = False
        self._active_step_limit = None
        self._run_step_limit = None
        self._run_revision = 0
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
            if any(node.get("data", {}).get("resetOnChange") is not None
                   for entry in controllers for node in entry.get("nodes", [])):
                raise ValueError("reset_on_change requires a single controller")
            if any("endAfterSteps" in node.get("data", {})
                   for entry in controllers for node in entry.get("nodes", [])):
                raise ValueError("end_after_steps requires a single controller")
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
        self._plan_robot_id = plan.get("robot")
        self._modifier = modifier
        reset_specs = [node.get("data", {}).get("resetOnChange") for node in nodes
                       if node.get("data", {}).get("resetOnChange") is not None]
        if reset_specs:
            if modifier is not None or len(reset_specs) != 1:
                raise ValueError("reset_on_change requires one reset rule and no Python modifier")
            modifier = self._motion_reset_modifier(reset_specs[0])
        end_specs = [node["data"]["endAfterSteps"] for node in nodes
                     if "endAfterSteps" in node.get("data", {})]
        self._run_step_limit = None
        if end_specs:
            import botlab_host

            from ..sdk._run_end import parse_end_after_steps

            if self._modifier is not None or len(end_specs) != 1:
                raise ValueError("end_after_steps requires one rule and no Python modifier")
            self._run_step_limit = parse_end_after_steps(
                end_specs[0], botlab_host.project_state().manifest()["parameters"],
            )
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

    def _motion_reset_modifier(self, spec):
        import botlab_host

        manifest = botlab_host.project_state().manifest()
        parameter = spec.get("parameter")
        options = manifest["parameters"].get(parameter, {}).get("options")
        mappings = spec.get("motions")
        resources = {entry["id"]: entry["kind"] for entry in manifest["resources"]}
        if not options or not isinstance(mappings, list) or len(mappings) != len(options):
            raise ValueError("reset_on_change must map every option value to a Motion")
        motion_ids = {}
        for entry in mappings:
            value, motion_id = entry.get("value"), entry.get("motionId")
            if value not in {option["value"] for option in options} or value in motion_ids or resources.get(motion_id) != "motion":
                raise ValueError("reset_on_change must map every option value to a Motion")
            motion_ids[value] = motion_id
        if spec.get("robotId") != self._plan_robot_id:
            raise ValueError("reset_on_change must reset its controller's robot")
        active, active_revision = None, -1

        def modify(action, frame):
            nonlocal active, active_revision
            selected = botlab_host.parameter(parameter)
            if selected not in motion_ids:
                raise ValueError("reset_on_change received an unknown option value")
            if selected != active or active_revision != self._run_revision:
                active, active_revision = selected, self._run_revision
                self.requestReset("Parameter changed", {
                    "motionId": motion_ids[selected], "robotId": spec["robotId"],
                    "frame": 0, "rootBody": 0,
                })
            return action

        return modify

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
        if self._in_plan_step:
            self._pending_plan_reset = (reason, spec)
            return None
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
        # The PD processor emits robot-local indices, just like a controller
        # group. Resolve all three absolute index arrays before worker stepping.
        return self._physics.compose_payloads({self._plan_robot_id: ("pd", view)})

    def _fill_parameters(self, name):
        import botlab_host

        return botlab_host.parameter(name)

    def _run_plan(self, max_control_steps):
        self.reset_for_run()
        last_action = None

        step_index = 0
        while max_control_steps is None or step_index < max_control_steps:
            frame, last_action, _ctrl, _substep_pd = self.plan_step(
                step_index, last_action
            )
            self._broadcast_display(frame)
            step_index += 1
            if self.completed or self._physics.should_stop():
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
        self.reset_for_run()

        step_index = 0
        while max_control_steps is None or step_index < max_control_steps:
            frame = self.custom_step(step_index)
            self._broadcast_display(frame)
            step_index += 1
            if self._physics.should_stop():
                break
        return step_index

    # -- shared rollout primitives -------------------------------------------
    # ``botlab-run``'s loops and ``botlab.open_headless`` call the same
    # primitives below, so a phase added here cannot silently skip the
    # headless verification path (or vice versa).

    def reset_for_run(self):
        """The reset preamble every rollout performs before its first step."""
        from ._observations import clear_pose_velocity_states

        self._pending_plan_reset = None
        self._plan_step_offset = 0
        self._completed_steps = 0
        self.completed = False
        self._active_step_limit = None
        self._run_revision += 1
        if self._reset_callback is not None:
            self._call_maybe_async(self._reset_callback())
        clear_pose_velocity_states()
        if self._executor is not None:
            self._executor.reset()
        self._physics.reset()

    def plan_step(self, step_index, last_action):
        """Advance one compiled-plan control step.

        ``last_action`` is the previous step's returned action (``None``
        before the first). Returns ``(frame, action, ctrl, substep_pd)``
        where ``action`` is a float32 copy safe to retain and feed back.
        """
        import numpy as np

        if self.completed:
            raise RuntimeError("run is completed; reset before stepping again")
        dt = 1.0 / self._control_hz
        # A hook may request a scene reset after inference. Discard that plan's
        # output and rebuild the first action from reset state before physics.
        for attempt in range(2):
            if self._active_step_limit is None and self._run_step_limit is not None:
                self._active_step_limit = self._run_step_limit(self._fill_parameters)
            index = step_index - self._plan_step_offset
            frame = Frame(dt, index * dt, index)
            self._current_frame = frame
            self._physics.fill_observations(
                self._executor, last_action=last_action, parameters=self._fill_parameters,
            )
            self._in_plan_step = True
            try:
                self._executor.step(frame=frame)
            finally:
                self._in_plan_step = False
            if self._pending_plan_reset is None:
                break
            reason, spec = self._pending_plan_reset
            self._pending_plan_reset = None
            if attempt:
                raise RuntimeError("[botlab] controller requested reset again on its first reset frame")
            start_time = self.requestReset(reason, spec)
            from ._observations import clear_pose_velocity_states

            clear_pose_velocity_states()
            self._executor.reset()
            self._plan_step_offset = step_index - round(float(start_time or 0) / dt)
            self._completed_steps = 0
            self._active_step_limit = None
            last_action = None
        action = np.array(self._policy_action(), dtype=np.float32)
        ctrl, substep_pd = self._control_payload()
        self._physics.step_substeps(
            self.decimation, ctrl=ctrl, substep_pd=substep_pd,
            observer=self._substep_observer,
        )
        self._physics.after_control_step()
        if self._recorder is not None:
            self._recorder.record_step(self._physics)
        self._completed_steps += 1
        self.completed = (self._active_step_limit is not None
                          and self._completed_steps >= self._active_step_limit)
        return frame, action, ctrl, substep_pd

    def custom_step(self, step_index):
        """Advance one custom-controller control step (no plan payload)."""
        dt = 1.0 / self._control_hz
        frame = Frame(dt, step_index * dt, step_index)
        self._current_frame = frame
        self._custom_step_actions.clear()
        self._custom_step_active = True
        try:
            self._call_maybe_async(self._step_callback(dt, frame.time, frame.index))
        finally:
            self._custom_step_active = False
        missing = [
            robot_id for robot_id in (self._step_robot_ids or [])
            if robot_id not in self._custom_step_actions
        ]
        if missing:
            raise RuntimeError(
                "[botlab] Controller.step must produce exactly one action for: "
                + ", ".join(missing)
            )
        self._physics.step_substeps(self.decimation, observer=self._substep_observer)
        self._physics.after_control_step()
        if self._recorder is not None:
            self._recorder.record_step(self._physics)
        return frame

    def _broadcast_display(self, frame):
        if self._display is not None:
            self._display.broadcast(
                frame.time,
                self._physics.qpos(),
                self._physics.qvel(),
            )
