"""Controller base class, native PolicyController and fixed PositionController."""
from ._declaration import (
    _optional_resource_config,
    _plan_value,
    _policy_pd_vector,
)
from ._plan import _control_plan
from ._robot import Robot
from ._signal import Signal, _as_signal, constant
from ._vectors import _expand_vector, require_vector


class Controller:
    def __init__(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError("Controller robot must be a BotLab Robot")
        self.robot = robot

    def reset(self):
        pass

    def step(self, observation, frame):
        raise NotImplementedError("Controller.step(observation, frame) must return an action")


class PolicyController(Controller):
    """Run a complete signal expression in the engine hot path.

    Exactly one output is explicit: ``target`` enables BotLab position PD, while
    ``control`` writes native MuJoCo actuator controls directly. Both paths keep
    exactly one upstream ``Signal.action()`` as the previous-action/hook boundary.
    """

    def __init__(
        self,
        robot,
        *,
        target=None,
        control=None,
        kp=None,
        kd=None,
        effort_limit=None,
        reset_on_change=None,
        end_after_steps=None,
    ):
        if not isinstance(robot, Robot):
            raise TypeError("PolicyController robot must be a BotLab Robot")
        if (target is None) == (control is None):
            raise ValueError("PolicyController requires exactly one of target or control")
        super().__init__(robot)

        if control is not None:
            if any(value is not None for value in (kp, kd, effort_limit)):
                raise ValueError(
                    "PolicyController raw control does not accept kp, kd, or effort_limit"
                )
            control = _as_signal(control, "PolicyController control")
            sink = Signal("out-mujoco", "controller", {}, (control,))
            self._plan_data, self.modify_action = _control_plan(
                robot.id,
                sink,
                control_signal=control,
            )
        else:
            target = _as_signal(target, "PolicyController target")
            dynamic_gains = {}
            gain_inputs = []
            for name, value in (("kps", kp), ("kds", kd)):
                if isinstance(value, Signal):
                    dynamic_gains[f"{name}InputSlot"] = 3 + len(gain_inputs)
                    gain_inputs.append(value)
            explicit_pd_values = (kp, kd, effort_limit)
            dimension = (
                robot.action_dimension()
                if any(value is not None for value in explicit_pd_values)
                else None
            )
            pd = Signal(
                "proc-pd",
                "position-pd",
                {
                    "pdType": "position",
                    "kps": (
                        robot.kp
                        if kp is None or isinstance(kp, Signal)
                        else _policy_pd_vector(kp, dimension, "PolicyController kp")
                    ),
                    "kds": (
                        robot.kd
                        if kd is None or isinstance(kd, Signal)
                        else _policy_pd_vector(kd, dimension, "PolicyController kd")
                    ),
                    "effortLimit": (
                        robot.effort_limit
                        if effort_limit is None
                        else _policy_pd_vector(
                            effort_limit,
                            dimension,
                            "PolicyController effort_limit",
                        )
                    ),
                    "ctrlIndices": _optional_resource_config(
                        robot.id,
                        "actuatorJointOrder",
                    ),
                    **dynamic_gains,
                },
                (
                    target,
                    robot.joint_position(order="native", _label="pd-joint-position"),
                    robot.joint_velocity(order="native", _label="pd-joint-velocity"),
                    *gain_inputs,
                ),
            )
            sink = Signal("out-mujoco", "controller", {}, (pd,))
            self._plan_data, self.modify_action = _control_plan(robot.id, sink, pd)

        if reset_on_change is not None:
            from ._resources import Motion
            if self.modify_action is not None:
                raise ValueError("reset_on_change cannot be combined with an action modifier")
            if not isinstance(reset_on_change, dict) or set(reset_on_change) != {"parameter", "motions"}:
                raise ValueError("reset_on_change requires parameter and motions")
            parameter = reset_on_change["parameter"]
            choices = reset_on_change["motions"]
            if not isinstance(parameter, str) or not parameter.strip():
                raise ValueError("reset_on_change parameter must be non-empty")
            if not isinstance(choices, dict) or not choices:
                raise ValueError("reset_on_change motions must map option values to Motion resources")
            mappings = []
            for value, motion in choices.items():
                if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 9007199254740991:
                    raise ValueError("reset_on_change values must be safe integers")
                if not isinstance(motion, Motion):
                    raise TypeError("reset_on_change choices must be Motion resources")
                mappings.append({"value": value, "motionId": motion.id})
            hook = next(node for node in self._plan_data["nodes"] if node["type"] == "proc-action-hook")
            hook["data"]["resetOnChange"] = {
                "parameter": parameter, "motions": mappings, "robotId": robot.id,
            }

        if end_after_steps is not None:
            from ._run_end import encode_end_after_steps
            if self.modify_action is not None:
                raise ValueError("end_after_steps cannot be combined with an action modifier")
            hook = next(node for node in self._plan_data["nodes"] if node["type"] == "proc-action-hook")
            hook["data"]["endAfterSteps"] = encode_end_after_steps(end_after_steps)

    @property
    def plan(self):
        return _plan_value(self._plan_data, "PolicyController plan")


class PositionController(Controller):
    """Holds a fixed joint-position target through a Python-side PD command."""

    def __init__(self, robot, *, kp, kd, effort_limit, target=None):
        super().__init__(robot)
        dimension = robot.action_dimension()
        self.target = require_vector(
            robot.default_joint_positions() if target is None else target,
            dimension,
            "PositionController target",
        )
        self.kp = _expand_vector(kp, dimension, "PositionController kp")
        self.kd = _expand_vector(kd, dimension, "PositionController kd")
        self.effort_limit = _expand_vector(
            effort_limit,
            dimension,
            "PositionController effort_limit",
        )
        if any(value <= 0 for value in self.effort_limit):
            raise ValueError("PositionController effort_limit values must be positive")
        native = PolicyController(
            robot,
            target=constant(self.target).action(),
            kp=self.kp,
            kd=self.kd,
            effort_limit=self.effort_limit,
        )
        self._plan_data = native._plan_data
        self.modify_action = None

    def step(self, observation, frame):
        return self.robot.position_pd(
            self.target,
            self.kp,
            self.kd,
            self.effort_limit,
        )
