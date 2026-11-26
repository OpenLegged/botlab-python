"""Simulation entry point with resource, parameter and output handles."""
import inspect
import json
import math
import re
from collections.abc import Mapping

import botlab_host

from ._compat import create_proxy
from ._controllers import Controller, PolicyController, PositionController
from ._declaration import _non_empty_string
from ._resources import Model, Motion, Scene, Sensor
from ._robot import Robot
from ._runtime_types import Frame
from ._vectors import require_vector


def _native_modifier_proxy(modify_action):
    """Wrap one ``Signal.action(modify=...)`` callable for the native host.

    Returns ``None`` when the controller declares no modify hook. Under the
    static host the Python callable itself marks the plan as needing the
    Python runtime; under Pyodide it is wrapped with the snapshot/Frame
    adapter and exposed through ``create_proxy``.
    """
    if modify_action is None:
        return None
    if create_proxy is None:
        return modify_action

    def snapshot_action(action):
        # Engine buffers cross as typed views under Pyodide; reading the
        # shared memory and materializing the list in C beats iterating
        # the proxy element by element. The native host already sends
        # plain sequences.
        to_py = getattr(action, "to_py", None)
        return list(action if to_py is None else to_py())

    async def modify_action_proxy(action, dt, time, index):
        result = modify_action(
            snapshot_action(action),
            Frame(dt, time, index),
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    return create_proxy(modify_action_proxy)


class Resources:
    """Handles for resources the project already declared, looked up by id."""

    def robot(self, resource_id):
        botlab_host.requireResource(str(resource_id), "robot")
        return Robot._from_id(str(resource_id))

    def model(self, resource_id):
        botlab_host.requireResource(str(resource_id), "model")
        return Model._from_id(str(resource_id))

    def motion(self, resource_id):
        botlab_host.requireResource(str(resource_id), "motion")
        return Motion._from_id(str(resource_id))

    def sensor(self, resource_id):
        botlab_host.requireResource(str(resource_id), "sensor")
        return Sensor._from_id(str(resource_id))

    def scene(self, resource_id):
        botlab_host.requireResource(str(resource_id), "scene")
        return Scene._from_id(str(resource_id))


class Parameters:
    def get(self, name, default=None):
        value = botlab_host.parameter(str(name))
        return default if value is None else value


class OutputChannel:
    def __init__(self, name):
        self._name = name

    def update(self, value):
        getattr(botlab_host, self._name)(value)


class OverlayChannel(OutputChannel):
    """``sim.overlay``: game-style scene annotations for the viewer.

    The per-item rules (kinds, fields, ranges, colors) live in the
    TypeScript contract -- ``parseSceneOverlay`` is the single source of
    truth and this side does not duplicate them. Python only insists the
    payload itself is a mapping before handing it to the host.
    """

    def update(self, value):
        if not isinstance(value, Mapping):
            raise TypeError("overlay.update() expects a mapping of {id: item}")
        OutputChannel.update(self, value)


class ControlsHudChannel(OutputChannel):
    """``sim.controls_hud``: bottom-strip key hints for custom controllers.

    The per-item rules (fields, localized labels, limits) live in the
    TypeScript contract -- ``parseControlsHudDeclaration`` is the single
    source of truth and this side does not duplicate them. Python only
    insists the payload itself is a list before handing it to the host.
    """

    def update(self, value):
        if not isinstance(value, list):
            raise TypeError("controls_hud.update() expects a list of items")
        OutputChannel.update(self, value)


class Simulation:
    @classmethod
    def from_project(cls):
        return cls()

    def __init__(self):
        self.resources = Resources()
        self.parameters = Parameters()
        self.metrics = OutputChannel("updateMetrics")
        self.diagnostics = OutputChannel("updateDiagnostics")
        self.overlay = OverlayChannel("updateOverlay")
        self.controls_hud = ControlsHudChannel("updateControlsHud")

    def model(self, path):
        """The ONNX model this project declared at `path`.

        The path is the resource ID, so the reference here reads as the file it loads.
        """
        return self.resources.model(path)

    def write_result(self, path, value):
        """Store JSON run output under ``results/`` in the current project.

        The host owns the project and control-step lifetime. It rejects writes
        outside a live step, stale runs and overwriting ordinary project files.
        The envelope identifies generated output without reserving every JSON
        file under results/ as a runtime result.
        """
        if not isinstance(path, str):
            raise TypeError("write_result path must be a string")
        if re.fullmatch(
            r"results/(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.json",
            path,
        ) is None:
            raise ValueError("write_result path must be a safe results/<name>.json path")
        content = json.dumps(
            {"format": "botlab.runtime-result", "version": 1, "data": value},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
        write = getattr(botlab_host, "writeResult", None)
        if write is None:
            raise RuntimeError(
                "Simulation.write_result requires a runtime with project result storage"
            )
        write(path, content)

    def request_reset(self, reason="Python controller requested reset"):
        botlab_host.requestReset(str(reason))

    def reset_to(
        self,
        motion,
        *,
        frame=0,
        robot=None,
        root_body=0,
        joint_indices=None,
        reason="Python controller requested motion reset",
    ):
        """Reset the scene to one reference motion frame.

        The frame's root pose, root velocity and joint state become the robot's
        initial state, and the simulation clock starts at ``frame / fps`` so
        ``Motion.sample(clock())`` resumes from that frame instead of frame 0.
        Motion joint channels map onto the robot's native joint order, one to one
        by default; pass ``joint_indices`` (the motion index of every robot joint,
        in robot order) when the export uses another order. ``root_body`` selects
        the motion body channel that carries the root pose (0 by default).
        ``robot`` is required when the project declares more than one robot.
        """
        if not isinstance(motion, Motion):
            raise TypeError("reset_to motion must be a BotLab Motion")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("reset_to frame must be a non-negative integer")
        if isinstance(root_body, bool) or not isinstance(root_body, int) or root_body < 0:
            raise ValueError("reset_to root_body must be a non-negative integer")
        spec = {"motionId": motion.id, "frame": frame, "rootBody": root_body}
        if robot is not None:
            if not isinstance(robot, Robot):
                raise TypeError("reset_to robot must be a BotLab Robot")
            spec["robotId"] = robot.id
        if joint_indices is not None:
            indices = list(joint_indices)
            if not indices or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in indices
            ):
                raise ValueError(
                    "reset_to joint_indices must be non-empty non-negative integers"
                )
            spec["jointIndices"] = indices
        botlab_host.requestReset(str(reason), spec)

    def place_body(
        self,
        body,
        *,
        position,
        quaternion=None,
        linear_velocity=(0, 0, 0),
        angular_velocity=(0, 0, 0),
    ):
        """Set one free-joint scene body's state before the next physics substep.

        ``body`` is a scene-global MuJoCo body name (e.g. a ball), not a robot base:
        robots move with ``robot.teleport()``. ``position`` is world-frame xyz;
        ``quaternion`` is a world-frame wxyz pose (normalized here; omitted keeps
        the body's current orientation); ``linear_velocity`` is world-frame m/s and
        ``angular_velocity`` world-frame rad/s (the runtime stores it body-local,
        as MuJoCo free-joint qvel requires).

        Only valid inside a control step. The write lands before the first physics
        substep of the next tick, followed by ``mj_forward``, so the controller's
        next observation sees it. Calling it again for the same body in the same
        control step replaces the earlier write: the last call wins.
        """
        name = _non_empty_string(body, "place_body body")
        position_values = require_vector(position, 3, "place_body position")
        quaternion_values = None
        if quaternion is not None:
            quaternion_values = require_vector(quaternion, 4, "place_body quaternion")
            norm = math.hypot(*quaternion_values)
            if not math.isfinite(norm):
                raise ValueError("place_body quaternion norm must be finite")
            if norm == 0.0:
                raise ValueError("place_body quaternion must not be all zeros")
            quaternion_values = [value / norm for value in quaternion_values]
        linear_values = require_vector(
            linear_velocity, 3, "place_body linear_velocity"
        )
        angular_values = require_vector(
            angular_velocity, 3, "place_body angular_velocity"
        )
        botlab_host.placeBody(
            name,
            position_values,
            quaternion_values,
            linear_values,
            angular_values,
        )

    def on_substep(self, callback):
        """Register a read-only observer called after every physics substep.

        The callback receives ``Frame(dt, time, index)`` at physics rate and
        may read live state through ``robot.read(...)``. It is an evaluation
        channel, not a control channel: it cannot write state, and raising
        inside it aborts the run with the error pointing at the hook.

        Local botlab-python runtime only — the browser editor and published
        packages have no per-substep Python path, so declaring a substep
        observer there fails at declaration time instead of being silently
        dropped.
        """
        if not callable(callback):
            raise TypeError("on_substep callback must be callable")
        register = getattr(botlab_host, "registerSubstepObserver", None)
        if register is None:
            raise RuntimeError(
                "[botlab] on_substep is only supported by the local "
                "botlab-python runtime (botlab run); the browser editor and "
                "published packages do not run Python per physics substep"
            )
        register(callback)

    def run(self, controller):
        is_group = isinstance(controller, (list, tuple))
        controllers = list(controller) if is_group else [controller]
        if not controllers or any(
            not isinstance(item, Controller) for item in controllers
        ):
            raise TypeError(
                "Simulation.run(controller) requires a BotLab Controller or "
                "a non-empty list/tuple of Controllers"
            )

        robot_ids = [item.robot.id for item in controllers]
        if len(robot_ids) != len(set(robot_ids)):
            raise ValueError(
                "Simulation.run controllers must target distinct robots"
            )

        native_controllers = [
            isinstance(item, PolicyController) or type(item) is PositionController
            for item in controllers
        ]
        if any(native_controllers) and not all(native_controllers):
            raise TypeError(
                "Simulation.run cannot mix native and custom Controllers"
            )
        if all(native_controllers):
            if is_group:
                # 每个控制器各带一个可选 modify：第二参数是与 controllers 对齐
                # 的列表（无 modify 的项为 None）。全部为 None 时保持旧的
                # 单参数调用形式。
                modifier_proxies = [
                    _native_modifier_proxy(item.modify_action)
                    for item in controllers
                ]
                envelope = {
                    "version": 1,
                    "controllers": [item._plan_data for item in controllers],
                }
                try:
                    if any(proxy is not None for proxy in modifier_proxies):
                        botlab_host.registerPolicyController(
                            envelope,
                            modifier_proxies,
                        )
                    else:
                        botlab_host.registerPolicyController(envelope)
                except Exception:
                    for proxy in modifier_proxies:
                        if proxy is not None and create_proxy is not None:
                            proxy.destroy()
                    raise
                botlab_host.log(
                    "info",
                    "Python",
                    "sim.run(native controllers) started",
                )
                return

            controller = controllers[0]
            modifier_proxy = _native_modifier_proxy(controller.modify_action)
            modifier_is_proxy = (
                modifier_proxy is not None and create_proxy is not None
            )
            try:
                if modifier_proxy is None:
                    botlab_host.registerPolicyController(controller._plan_data)
                else:
                    botlab_host.registerPolicyController(
                        controller._plan_data,
                        modifier_proxy,
                    )
            except Exception:
                if modifier_is_proxy:
                    modifier_proxy.destroy()
                raise
            botlab_host.log(
                "info",
                "Python",
                f"sim.run({type(controller).__name__}) started",
            )
            return

        async def reset_controller():
            for item in controllers:
                result = item.reset()
                if inspect.isawaitable(result):
                    await result

        async def step_controller(dt, time, index):
            frame = Frame(dt, time, index)
            for item in controllers:
                observation = item.robot.observe()
                action = item.step(observation, frame)
                if inspect.isawaitable(action):
                    action = await action
                item.robot.apply_action(action)

        if create_proxy is not None:
            reset_proxy = create_proxy(reset_controller)
            step_proxy = create_proxy(step_controller)
        else:
            reset_proxy = reset_controller
            step_proxy = step_controller
        try:
            botlab_host.registerReset(reset_proxy)
            botlab_host.registerStep(step_proxy, robot_ids)
        except Exception:
            if create_proxy is not None:
                reset_proxy.destroy()
                step_proxy.destroy()
            raise
        label = "controllers" if is_group else "controller"
        botlab_host.log("info", "Python", f"sim.run({label}) started")
