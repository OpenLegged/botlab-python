"""Simulation entry point with resource, parameter and output handles."""
import inspect

import botlab_host

from ._compat import create_proxy
from ._controllers import Controller, PolicyController, PositionController
from ._resources import Model, Motion, Scene, Sensor
from ._robot import Robot
from ._runtime_types import Frame


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


class Simulation:
    @classmethod
    def from_project(cls):
        return cls()

    def __init__(self):
        self.resources = Resources()
        self.parameters = Parameters()
        self.metrics = OutputChannel("updateMetrics")
        self.diagnostics = OutputChannel("updateDiagnostics")

    def model(self, path):
        """The ONNX model this project declared at `path`.

        The path is the resource ID, so the reference here reads as the file it loads.
        """
        return self.resources.model(path)

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
                if any(item.modify_action is not None for item in controllers):
                    raise TypeError(
                        "Simulation.run native controller groups do not support "
                        "Signal.action(modify=...)"
                    )
                botlab_host.registerPolicyController({
                    "version": 1,
                    "controllers": [item._plan_data for item in controllers],
                })
                botlab_host.log(
                    "info",
                    "Python",
                    "sim.run(native controllers) started",
                )
                return

            controller = controllers[0]
            modifier_proxy = None
            modifier_is_proxy = False
            if controller.modify_action is not None:
                def snapshot_action(action):
                    # Engine buffers cross as typed views under Pyodide; reading the
                    # shared memory and materializing the list in C beats iterating
                    # the proxy element by element. The native host already sends
                    # plain sequences.
                    to_py = getattr(action, "to_py", None)
                    return list(action if to_py is None else to_py())

                async def modify_action(action, dt, time, index):
                    result = controller.modify_action(
                        snapshot_action(action),
                        Frame(dt, time, index),
                    )
                    if inspect.isawaitable(result):
                        result = await result
                    return result

                if create_proxy is None:
                    # Static host: the callable itself marks the plan as
                    # needing the Python runtime instead of a JS callback.
                    modifier_proxy = controller.modify_action
                else:
                    modifier_proxy = create_proxy(modify_action)
                    modifier_is_proxy = True
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
