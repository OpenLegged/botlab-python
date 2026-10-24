"""Observation assembly for the native MuJoCo runtime — TypeScript mirror.

This module ports the "data extraction" half of the browser observation
pipeline to native ``mujoco`` + ``numpy``, one source file per counterpart:

- ``ObservationProcessor`` mirrors
  ``src/domain/runtime/engine/processors/ObservationProcessor.ts`` (passive
  buffer holder: ``init``/``set_data``/``set_data_all``, ``compute`` no-op)
  and ``REGISTRY`` maps every ``obs-*`` node type of SimulationEngine.ts's
  ``ProcessorRegistry`` to it.
- ``fill_observation`` mirrors the per-control-step binding loop of
  ``mujocoPolicyController.ts`` (L450-510): camera bindings get whole frame
  lists, height scans are validated against their declared dim, obs-commands
  receive the parameter injection, everything else flows through
  ``get_mujoco_observation`` into ``processor.set_data``.
- ``get_mujoco_observation`` mirrors ``mujocoObservations.ts``
  (qpos/qvel/dof slices, imu, projected gravity, base height and velocities,
  constants, commands, actions, mujoco queries) with the slice/transform
  chain of ``mujocoObservationTransforms.ts``.
- ``read_mujoco_access_query`` mirrors ``mujocoAccessRegistry.ts``; body
  velocities are derived exactly like ``mujocoWorkerSnapshots.ts``.
- ``height_scan`` mirrors ``heightScanObservation.ts`` +
  ``mujocoWorkerHeightScan.ts``.
- ``camera_frames`` mirrors the frame contract of ``cameraObservation.ts``
  and the pose algebra of ``cameraObservationPose.ts`` (Three.js camera
  conventions mapped back into MuJoCo world coordinates), rendering through
  ``mujoco.Renderer``. ``read_camera_config`` mirrors
  ``cameraObservationConfig.ts`` (including the sensor-resource-config merge
  that ``mujocoPolicyController.ts`` ``parseNode`` performs for obs-camera
  nodes) and the depth post-processing chain of
  ``cameraObservationRuntime.ts`` + ``cameraDepthCorruption.ts``: render the
  full pinhole sensor, crop the observation window, scale by ``depthScale``
  with a [0, 1] clamp, blur with the separable gaussian, then apply the
  seeded depth corruption. Deliberate differences are documented on
  ``camera_frames``.

Deliberate differences, all documented in place:

- The local runtime has no attached keyboard/gamepad/touch devices, so
  ``obs-commands`` uses an explicit neutral input snapshot: axis modes keep
  parameter/project defaults, key-gated mappings are inactive and button
  modes emit the neutral command code 0.
  ``poseVelocityCommand`` is ported in full: in the browser the vector
  entering the transform has already passed through the interactive
  overrides, and with no user input active those overrides leave the
  parameter/project values untouched — so the local baseline (the
  parameter-injected vector) is exactly the browser's no-input behavior.
- Contact timing / per-body contact force query fields are maintained by the
  physics layer (aggregation is task T6); the query registry keeps the full
  field-name contract and reads those fields from an optional snapshot
  mapping, failing explicitly when the physics layer has not published it.
- Native MjData exposes 2-D arrays (``xquat`` (nbody, 4), ``cvel`` (nbody, 6),
  ...); every slice flattens the view first so index arithmetic matches the
  flat TS arrays. Outputs are ``np.float32`` copies, never aliases into
  MjData.

Requires the runtime extras (``numpy``, ``mujoco``).
"""

import math

import numpy as np

try:
    import mujoco
except ImportError as _error:  # pragma: no cover - environment guard
    raise ImportError(
        "[_observations] the native observation layer requires the mujoco "
        "runtime extra (pip install botlab[runtime])"
    ) from _error

from ._camera import camera_frames, dispose_camera_renderers, reset_camera_observation_state
from ._camera_config import (
    CAMERA_OBSERVATION_NODE_TYPE,
    camera_frame_size,
    camera_output_mode,
    read_camera_config,
)
from ._commands import _command_vector, _write_command_parameters, clear_pose_velocity_states
from ._kinematics import (
    _base_angular_velocity,
    _base_height,
    _base_linear_velocity,
    _projected_gravity_quaternion,
    _root_qpos_offset,
    _root_quaternion,
    _root_qvel_offset,
    projected_gravity_from_quaternion,
)
from ._queries import read_mujoco_access_query
from ._values import (
    _error,
    _finite_copy,
    _first_present,
    _flat,
    _float32_slice,
    _is_integer,
    _number,
    _positive_integer,
    configured_dim,
    constant_vector,
    parse_number_list,
    transform_vector,
    vector_from,
)

HEIGHT_SCAN_NODE_TYPE = "obs-height-scan"


# heightScanObservation.ts DEFAULT_HEIGHT_SCAN (config keys keep camelCase).
DEFAULT_HEIGHT_SCAN = {"frame": "yaw", "lift": 1.0, "maxDistance": 10.0}

__all__ = [
    "REGISTRY",
    "camera_frames",
    "clear_pose_velocity_states",
    "dispose_camera_renderers",
    "fill_observation",
    "height_scan",
    "read_camera_config",
    "read_mujoco_access_query",
]


# ---------------------------------------------------------------------------
# ObservationProcessor.ts mirror
# ---------------------------------------------------------------------------


class ObservationProcessor:
    """Passive source node: data is pushed in via ``set_data``/``set_data_all``.

    ``is_observation`` mirrors the TS engine's `instanceof
    ObservationProcessor` test: the executor collects these nodes into
    observation bindings the physics layer fills at each control step.
    """

    is_observation = True

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._camera_node_id = None

    def init(self, config, input_dims=None):
        config = config or {}
        self._camera_node_id = None
        output_dims = config.get("outputDims")
        if isinstance(output_dims, (list, tuple, np.ndarray)):
            buffers = []
            for index, dim in enumerate(output_dims):
                dim = _number(dim)
                if not _is_integer(dim) or dim <= 0:
                    raise _error(
                        "ObservationProcessor",
                        f"outputDims[{index}] must be a positive integer",
                    )
                buffers.append(np.zeros(int(dim), dtype=np.float32))
            self.output_buffers = buffers
            return

        if config.get("nodeType") == CAMERA_OBSERVATION_NODE_TYPE:
            width, height = camera_frame_size(config)
            mode = camera_output_mode(config)
            # The executor's reset() re-anchors the camera noise stream per
            # run (TS resetOutput clears the depth frame indexes).
            self._camera_node_id = config.get("nodeId")
            include_age_steps = config.get("includeAgeSteps") is True
            if include_age_steps and mode != "depth":
                raise _error(
                    "ObservationProcessor",
                    "includeAgeSteps requires a depth camera observation",
                )
            if mode == "rgb":
                self.output_buffers = [np.zeros(width * height * 3, dtype=np.float32)]
            elif mode == "depth":
                self.output_buffers = [np.zeros(width * height, dtype=np.float32)]
                if include_age_steps:
                    self.output_buffers.append(np.zeros(1, dtype=np.float32))
            else:
                self.output_buffers = [
                    np.zeros(width * height * 3, dtype=np.float32),
                    np.zeros(width * height, dtype=np.float32),
                ]
            return

        dim = config.get("dim")
        if dim is not None:
            value = _number(dim)
            if not math.isfinite(value) or value <= 0:
                raise _error(
                    "ObservationProcessor",
                    "dim must be a positive number when provided",
                )
            self.output_buffers[0] = np.zeros(int(math.floor(value)), dtype=np.float32)

    def set_data(self, data):
        self._set_output_data(0, data)

    def reset(self):
        """Controller reset hook: re-anchor the camera noise stream."""
        if self._camera_node_id:
            reset_camera_observation_state(self._camera_node_id)

    def set_data_all(self, outputs):
        self.output_buffers = [None] * len(outputs)
        for slot, data in enumerate(outputs):
            self._set_output_data(slot, data)

    def compute(self, inputs=None):
        """Data is updated via set_data externally (no-op, like the TS class)."""

    def _set_output_data(self, slot, data):
        array = np.asarray(data)
        if array.dtype == np.float32:
            # Keep the exact buffer object for the common 1-D case so
            # downstream zero-copy aliasing matches the TS reference swap.
            self.output_buffers[slot] = array if array.ndim == 1 else array.reshape(-1)
            return
        values = array.astype(np.float64).reshape(-1)
        if (
            slot >= len(self.output_buffers)
            or self.output_buffers[slot] is None
            or len(self.output_buffers[slot]) != len(values)
        ):
            while len(self.output_buffers) <= slot:
                self.output_buffers.append(None)
            self.output_buffers[slot] = np.zeros(len(values), dtype=np.float32)
        self.output_buffers[slot][:] = values


# Every obs-* node type of SimulationEngine.ts's ProcessorRegistry (the
# registry text is the source of truth; obs-camera and obs-height-scan go
# through the same passive holder and are fed by fill_observation).
REGISTRY = {
    node_type: ObservationProcessor
    for node_type in (
        "obs-qpos",
        "obs-qvel",
        "obs-imu",
        "obs-constant",
        "obs-base-height",
        "obs-base-lin-vel",
        "obs-base-ang-vel",
        "obs-projected-gravity",
        "obs-commands",
        "obs-dof-pos",
        "obs-dof-vel",
        "obs-actions",
        CAMERA_OBSERVATION_NODE_TYPE,
        "obs-mujoco-query",
        HEIGHT_SCAN_NODE_TYPE,
    )
}


def _dim_bounded_end(node_data, start, source, label):
    """Narrow an offset-based slice to the observation's declared dim."""
    raw = _first_present(node_data or {}, ("dim",))
    if raw is None or raw == "":
        return None
    dim = _number(raw)
    if not math.isfinite(dim) or dim <= 0:
        raise _error(
            "mujocoObservations",
            f"observation dimension must be positive, got {raw}",
        )
    end = start + int(math.floor(dim))
    if source is not None:
        length = len(_flat(source, label))
        if end > length:
            raise _error(
                "mujocoObservations",
                f"{label} slice {start}:{end} exceeds length {length}",
            )
    return end


def _raw_state_end(node_data, start, source, label):
    indices = parse_number_list((node_data or {}).get("indices"), "indices")
    if not indices:
        return _dim_bounded_end(node_data, start, source, label)
    if any(value < 0 or not _is_integer(value) for value in indices):
        raise _error("mujocoObservations", "indices must be non-negative integers")
    end = start + int(max(indices)) + 1
    if source is not None:
        length = len(_flat(source, label))
        if end > length:
            raise _error(
                "mujocoObservations",
                f"{label} slice {start}:{end} exceeds length {length}",
            )
    return end


def _imu_vector(context):
    quat = _root_quaternion(context["model"], context["data"], context.get("nodeData"))
    ang_vel = _base_angular_velocity(
        context["model"], context["data"], context.get("nodeData")
    )
    return np.concatenate([quat, ang_vel]).astype(np.float32)


def _actions_vector(context):
    node_data = context.get("nodeData") or {}
    model = context["model"]
    data = context["data"]
    raw = context.get("lastAction")
    raw_length = 0
    if raw is not None:
        try:
            raw_length = len(raw)
        except TypeError:
            raw_length = 0
    if raw_length > 0:
        fallback_dim = raw_length
    else:
        ctrl_length = 0
        try:
            ctrl_length = len(data.ctrl)
        except AttributeError:
            ctrl_length = 0
        fallback_dim = max(int(model.nu) or 0, ctrl_length or 0, 1)
    return transform_vector(
        vector_from(raw, configured_dim(node_data, fallback_dim), "obs-actions"),
        node_data,
    )


# ---------------------------------------------------------------------------
# Observation readers (mujocoObservations.ts OBSERVATION_READERS)
# ---------------------------------------------------------------------------


def _read_qpos(context):
    data, node_data = context["data"], context.get("nodeData")
    qpos_offset = context.get("qposOffset", 0)
    return transform_vector(
        _float32_slice(
            data.qpos,
            "data.qpos",
            qpos_offset,
            _raw_state_end(node_data, qpos_offset, data.qpos, "data.qpos"),
        ),
        node_data,
    )


def _read_qvel(context):
    data, node_data = context["data"], context.get("nodeData")
    qvel_offset = context.get("qvelOffset", 0)
    return transform_vector(
        _float32_slice(
            data.qvel,
            "data.qvel",
            qvel_offset,
            _raw_state_end(node_data, qvel_offset, data.qvel, "data.qvel"),
        ),
        node_data,
    )


def _read_dof_pos(context):
    model, data, node_data = context["model"], context["data"], context.get("nodeData")
    start = context.get("qposOffset", 0) + _root_qpos_offset(model, data)
    return transform_vector(
        _float32_slice(
            data.qpos,
            "data.qpos",
            start,
            _dim_bounded_end(node_data, start, data.qpos, "data.qpos"),
        ),
        node_data,
    )


def _read_dof_vel(context):
    model, data, node_data = context["model"], context["data"], context.get("nodeData")
    start = context.get("qvelOffset", 0) + _root_qvel_offset(model, data)
    return transform_vector(
        _float32_slice(
            data.qvel,
            "data.qvel",
            start,
            _dim_bounded_end(node_data, start, data.qvel, "data.qvel"),
        ),
        node_data,
    )


_OBSERVATION_READERS = {
    "obs-qpos": _read_qpos,
    "obs-qvel": _read_qvel,
    "obs-dof-pos": _read_dof_pos,
    "obs-dof-vel": _read_dof_vel,
    "obs-base-height": lambda context: transform_vector(
        _base_height(context["model"], context["data"]), context.get("nodeData")
    ),
    "obs-base-lin-vel": lambda context: transform_vector(
        _base_linear_velocity(context["model"], context["data"], context.get("nodeData")),
        context.get("nodeData"),
    ),
    "obs-base-ang-vel": lambda context: transform_vector(
        _base_angular_velocity(context["model"], context["data"], context.get("nodeData")),
        context.get("nodeData"),
    ),
    "obs-projected-gravity": lambda context: transform_vector(
        projected_gravity_from_quaternion(
            _projected_gravity_quaternion(
                context["model"],
                context["data"],
                context.get("nodeData"),
                context.get("namePrefix", ""),
            )
        ),
        context.get("nodeData"),
    ),
    "obs-imu": lambda context: transform_vector(
        _imu_vector(context), context.get("nodeData")
    ),
    "obs-constant": lambda context: constant_vector(context.get("nodeData")),
    "obs-commands": lambda context: _command_vector(
        context.get("nodeData"),
        context.get("nodeId"),
        {"model": context["model"], "data": context["data"]},
    ),
    "obs-actions": _actions_vector,
    "obs-mujoco-query": lambda context: transform_vector(
        read_mujoco_access_query(
            {
                "model": context["model"],
                "data": context["data"],
                "snapshot": context.get("snapshot"),
                **({"namePrefix": context["namePrefix"]} if context.get("namePrefix") else {}),
            },
            context.get("nodeData") or {},
        ),
        context.get("nodeData"),
    ),
}


def get_mujoco_observation(node_type, context):
    """``getMujocoObservation``: extract one observation vector.

    ``context``: ``{"nodeId"?, "model", "data", "nodeData"?, "lastAction"?,
    "qposOffset"?, "qvelOffset"?, "namePrefix"?, "snapshot"?}``.
    """
    if node_type == CAMERA_OBSERVATION_NODE_TYPE:
        raise _error(
            "mujocoObservations",
            "obs-camera requires renderer-backed camera observation",
        )
    reader = _OBSERVATION_READERS.get(node_type)
    if reader is None:
        raise _error(
            "mujocoObservations", f'unsupported observation type "{node_type}"'
        )
    return reader(context)


# ---------------------------------------------------------------------------
# heightScanObservation.ts + mujocoWorkerHeightScan.ts mirror
# ---------------------------------------------------------------------------


def _resolve_height_scan_body(model, spec, node_id):
    raw = spec.get("body")
    body = str(raw if raw is not None else "").strip()
    if not body:
        raise _error(
            "heightScan", f'node "{node_id}" needs the body it scans around'
        )
    if body.isascii() and body.isdigit():
        index = int(body)
        if 0 <= index < int(model.nbody):
            return index
        raise _error(
            "heightScan", f'node "{node_id}" body {index} is outside the model'
        )
    for index in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        if name == body:
            return index
    raise _error("heightScan", f'node "{node_id}" body "{body}" was not found')


def height_scan(model, data, spec):
    """Terrain height under each sample point, in world coordinates.

    Mirrors ``computeHeightScan``: each sample casts a vertical ray from
    ``lift`` above the anchor body and reports ``origin_z - hit_distance``;
    a miss (or a hit beyond ``maxDistance``) reads as the full ``maxDistance``
    drop so the observation stays a fixed-width vector with a documented
    floor. ``mj_multiRay`` cannot batch this contract (the C API takes one
    shared origin while every sample point has its own origin; slanted rays
    would silently change hit semantics on non-flat terrain), so the port
    casts one ``mj_ray`` per point exactly like the TS worker, with the point
    rotation and clamping vectorized.
    """
    node_id = str(spec.get("nodeId") if spec.get("nodeId") is not None else "")
    points_raw = spec.get("points")
    if not isinstance(points_raw, (list, tuple, np.ndarray)) or len(points_raw) == 0:
        raise _error(
            "heightScan", f'node "{node_id}" needs a non-empty (x, y) point list'
        )
    points = np.asarray(points_raw, dtype=np.float64).reshape(-1)
    if len(points) % 2 != 0:
        raise _error(
            "heightScan", f'node "{node_id}" needs a non-empty (x, y) point list'
        )
    bad = np.flatnonzero(~np.isfinite(points))
    if len(bad):
        raise _error(
            "heightScan", f'node "{node_id}" point {int(bad[0]) // 2} is not finite'
        )

    lift = _number(DEFAULT_HEIGHT_SCAN["lift"] if spec.get("lift") is None else spec["lift"])
    if not math.isfinite(lift) or lift <= 0:
        raise _error("heightScan", f'node "{node_id}" lift must be a positive number')
    max_distance = _number(
        DEFAULT_HEIGHT_SCAN["maxDistance"]
        if spec.get("maxDistance") is None
        else spec["maxDistance"]
    )
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise _error(
            "heightScan", f'node "{node_id}" maxDistance must be a positive number'
        )

    body_id = _resolve_height_scan_body(model, spec, node_id)
    xmat = _flat(data.xmat, "data.xmat")
    yaw = (
        0.0
        if spec.get("frame") == "world"
        else math.atan2(xmat[body_id * 9 + 3], xmat[body_id * 9 + 0])
    )
    cos, sin = math.cos(yaw), math.sin(yaw)
    origin = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    origin[2] += lift

    offsets = points.reshape(-1, 2)
    world_x = origin[0] + offsets[:, 0] * cos - offsets[:, 1] * sin
    world_y = origin[1] + offsets[:, 0] * sin + offsets[:, 1] * cos

    groups_raw = spec.get("geomGroups")
    if groups_raw is None:
        groups = np.ones(6, dtype=np.uint8)
    else:
        if not isinstance(groups_raw, (list, tuple, np.ndarray)) or len(groups_raw) != 6:
            raise _error("heightScan", f'node "{node_id}" geomGroups must contain 6 flags')
        groups = np.asarray(groups_raw, dtype=np.uint8)

    direction = np.array([0.0, 0.0, -1.0])
    distances = np.empty(len(offsets), dtype=np.float64)
    geom_id = np.zeros(1, dtype=np.int32)
    for index in range(len(offsets)):
        point = np.array([world_x[index], world_y[index], origin[2]])
        distances[index] = mujoco.mj_ray(
            model, data, point, direction, groups, True, body_id, geom_id
        )

    hits = np.where((distances >= 0) & (distances <= max_distance), distances, max_distance)
    return (origin[2] - hits).astype(np.float32)


# ---------------------------------------------------------------------------
# mujocoPolicyController.ts step-loop mirror
# ---------------------------------------------------------------------------


def fill_observation(binding, context):
    """Push one observation binding's data into its processor.

    ``binding``: ``{"nodeId", "type", "data", "processor", "qposOffset",
    "qvelOffset", "namePrefix"}`` (same shape as the engine's observation
    bindings). ``context``: ``{"model", "data", "last_action", "parameters",
    "height_scan", "camera_frames"}`` — ``height_scan``/``camera_frames`` are
    callables taking the node data spec (bind ``model``/``data`` with
    ``functools.partial``), invoked only for the matching node types. The
    last-action roll (controller writes its policy action back after the
    engine step) stays with the runtime loop; this layer only reads it.
    """
    node_id = binding["nodeId"]
    node_type = binding["type"]
    processor = binding["processor"]

    if node_type == CAMERA_OBSERVATION_NODE_TYPE:
        camera_frames = context.get("camera_frames")
        if not callable(camera_frames):
            raise _error(
                "botlab",
                f'Camera node "{node_id}" requires a camera_frames callable in '
                "the observation context",
            )
        spec = dict(binding.get("data") or {})
        # The depth noise stream keys on the node id, like the TS renderer's
        # per-nodeId depth frame index.
        spec.setdefault("nodeId", node_id)
        processor.set_data_all(camera_frames(spec))
        return

    if node_type == HEIGHT_SCAN_NODE_TYPE:
        height_scan = context.get("height_scan")
        if not callable(height_scan):
            raise _error(
                "botlab",
                f'Height scan node "{node_id}" requires a height_scan callable '
                "in the observation context",
            )
        spec = dict(binding.get("data") or {})
        spec.setdefault("nodeId", node_id)
        heights = height_scan(spec)
        expected = _positive_integer(
            (binding.get("data") or {}).get("dim"), f"{node_id}.dim"
        )
        if len(heights) != expected:
            raise _error(
                "botlab",
                f'Height scan "{node_id}" returned {len(heights)} samples, '
                f"expected {expected}",
            )
        processor.set_data(_finite_copy(heights, f"{node_id}.heights"))
        return

    if node_type == "obs-commands":
        _write_command_parameters(binding["data"], context)

    processor.set_data(
        get_mujoco_observation(
            node_type,
            {
                "nodeId": node_id,
                "model": context["model"],
                "data": context["data"],
                "nodeData": binding.get("data"),
                "lastAction": context.get("last_action"),
                "qposOffset": binding.get("qposOffset", 0),
                "qvelOffset": binding.get("qvelOffset", 0),
                "snapshot": context.get("snapshot"),
                **(
                    {"namePrefix": binding["namePrefix"]}
                    if binding.get("namePrefix")
                    else {}
                ),
            },
        )
    )
