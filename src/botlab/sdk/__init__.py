"""BotLab Python SDK: declare a robot project and compile its controller.

Everything public is re-exported here; import names from ``botlab`` only.
"""

from ._compat import create_proxy
from ._controllers import Controller, PolicyController, PositionController
from ._declaration import default_joint_positions, project, resource_config
from ._history import history, resize
from ._math import (
    absolute,
    clock,
    cos,
    cross,
    derivative,
    dot,
    exp,
    log,
    low_pass,
    moving_average,
    noise,
    norm,
    normalize,
    reduce_max,
    reduce_min,
    reduce_sum,
    sin,
    sqrt,
)
from ._modes import (
    delay,
    mode_select,
    mode_switch,
    one_hot_index,
    phase_clock,
    state_machine,
)
from ._motion import motion_reference
from ._policy import policy_bank, policy_inputs
from ._resources import Model, Motion, Scene, Sensor
from ._robot import Robot, waypoint_tracking
from ._rules import one_hot, rule_index
from ._runtime_types import Frame, HistoryBuffer
from ._signal import (
    Signal,
    all_of,
    any_of,
    concat,
    constant,
    logical_not,
    split,
    take,
    where,
)
from ._simulation import OutputChannel, Parameters, Resources, Simulation
from ._spatial import (
    inverse_rotate,
    projected_gravity,
    quaternion_heading,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_yaw,
    relative_position,
    relative_quaternion,
    rotate,
    rotation_6d,
    rotation_matrix,
    tangent_normal,
)
from ._vectors import affine, clip, reorder, require_vector

__version__ = "0.2.0"

__all__ = [
    "Controller",
    "Frame",
    "HistoryBuffer",
    "Model",
    "Motion",
    "OutputChannel",
    "Parameters",
    "PolicyController",
    "PositionController",
    "Resources",
    "Robot",
    "Scene",
    "Sensor",
    "Signal",
    "Simulation",
    "absolute",
    "affine",
    "all_of",
    "any_of",
    "clip",
    "clock",
    "concat",
    "constant",
    "cos",
    "cross",
    "default_joint_positions",
    "delay",
    "derivative",
    "dot",
    "exp",
    "history",
    "inverse_rotate",
    "log",
    "logical_not",
    "low_pass",
    "mode_select",
    "mode_switch",
    "motion_reference",
    "moving_average",
    "noise",
    "norm",
    "normalize",
    "one_hot",
    "one_hot_index",
    "phase_clock",
    "policy_bank",
    "policy_inputs",
    "project",
    "projected_gravity",
    "quaternion_heading",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_slerp",
    "quaternion_yaw",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "relative_position",
    "relative_quaternion",
    "reorder",
    "require_vector",
    "resize",
    "resource_config",
    "rotate",
    "rotation_6d",
    "rotation_matrix",
    "rule_index",
    "sin",
    "split",
    "sqrt",
    "state_machine",
    "take",
    "tangent_normal",
    "waypoint_tracking",
    "where",
]
