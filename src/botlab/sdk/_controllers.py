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
                        if kp is None
                        else _policy_pd_vector(kp, dimension, "PolicyController kp")
                    ),
                    "kds": (
                        robot.kd
                        if kd is None
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
                },
                (
                    target,
                    robot.joint_position(order="native", _label="pd-joint-position"),
                    robot.joint_velocity(order="native", _label="pd-joint-velocity"),
                ),
            )
            sink = Signal("out-mujoco", "controller", {}, (pd,))
            self._plan_data, self.modify_action = _control_plan(robot.id, sink, pd)

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
