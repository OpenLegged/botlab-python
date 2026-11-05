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
import re

import numpy as np

try:
    import mujoco
except ImportError as _error:  # pragma: no cover - environment guard
    raise ImportError(
        "[_observations] the native observation layer requires the mujoco "
        "runtime extra (pip install botlab[runtime])"
    ) from _error

CAMERA_OBSERVATION_NODE_TYPE = "obs-camera"
HEIGHT_SCAN_NODE_TYPE = "obs-height-scan"

FREE_JOINT_QPOS_DIM = 7
FREE_JOINT_QVEL_DIM = 6
DEFAULT_COMMAND_DIM = 5

# heightScanObservation.ts DEFAULT_HEIGHT_SCAN (config keys keep camelCase).
DEFAULT_HEIGHT_SCAN = {"frame": "yaw", "lift": 1.0, "maxDistance": 10.0}

# cameraObservation.ts DEFAULT_CAMERA_OBSERVATION (full field set; config keys
# keep camelCase).
DEFAULT_CAMERA_OBSERVATION = {
    "outputMode": "rgbd",
    "targetKind": "body",
    "rayAlignment": "body",
    "cameraOffsetFrame": "ray",
    "target": "1",
    "depthBackend": "render",
    "width": 64,
    "height": 64,
    "fov": 60,
    "near": 0.02,
    "far": 10,
    "depthScale": 10,
    "depthGaussianKernelSize": 0,
    "depthGaussianSigma": 1,
    "depthNoiseStd": 0,
    "depthNoiseDistanceStd": 0,
    "depthDropoutProbability": 0,
    "depthDropoutPatchCount": 0,
    "depthDropoutPatchRadius": 0,
    "depthNoiseSeed": 1,
    "realtimeReadbackBackoffMs": 1000,
    "realtimeReadbackIntervalMs": 100,
    "realtimeReadbackSlowThresholdMs": 50,
    "strictReadbackIntervalMs": 0,
    "readbackMode": "async",
    "depthReadbackFormat": "packed-rgba8",
    "readbackPipelineDepth": 1,
    "raycastGeomGroups": [],
    "excludeGeoms": [],
    "excludeBodies": [],
    "cameraOffset": [0, 0, 0],
    "cameraRotationMode": "euler",
    "cameraEulerDeg": [0, -90, 0],
}

# cameraObservation.ts MAX_CAMERA_READBACK_PIPELINE_DEPTH.
MAX_CAMERA_READBACK_PIPELINE_DEPTH = 4

# Contact-derived snapshot fields published by the physics layer (T6): the
# registry keeps the field-name contract, reads pull them from the optional
# ``snapshot`` mapping of the access context.
SNAPSHOT_FIELDS = frozenset({
    "contact_force_body",
    "contact_force_geom",
    "contact_body",
    "contact_geom",
    "body_current_air_time",
    "body_current_contact_time",
    "body_last_air_time",
    "body_last_contact_time",
    "body_first_contact",
    "geom_current_air_time",
    "geom_current_contact_time",
    "geom_last_air_time",
    "geom_last_contact_time",
    "geom_first_contact",
})

# Derived per-body velocity fields (mujocoWorkerSnapshots.ts snapshotBodyVelocities).
_DERIVED_FIELDS = frozenset({
    "body_linvel_world",
    "body_linvel_local",
    "body_angvel_world",
    "body_angvel_local",
})


def _error(prefix, message):
    return ValueError(f"[{prefix}] {message}")


def _number(raw):
    """JS ``Number()`` coercion: null -> 0, bools -> 0/1, numeric strings parse."""
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return float("nan")
    return float("nan")


def _is_integer(value):
    value = float(value)
    return math.isfinite(value) and value.is_integer()


def _flat(source, label):
    """``requireArray`` + flatten: MjData gives 2-D views, TS reads flat ones."""
    if source is None:
        raise _error("mujocoObservations", f"{label} is missing or not array-like")
    try:
        array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError):
        raise _error("mujocoObservations", f"{label} is missing or not array-like") from None
    if array.ndim == 0:
        raise _error("mujocoObservations", f"{label} is missing or not array-like")
    return array.reshape(-1)


def _float32_slice(source, label, start=0, end=None):
    """``float32Slice``: bounds-checked finite-checked float32 copy."""
    values = _flat(source, label)
    length = len(values)
    if start < 0:
        raise _error("mujocoObservations", f"{label} slice start {start} is negative")
    if end is not None and end < start:
        raise _error("mujocoObservations", f"{label} slice end {end} is before start {start}")
    if start > length or (end is not None and end > length):
        shown = length if end is None else end
        raise _error(
            "mujocoObservations",
            f"{label} slice {start}:{shown} exceeds length {length}",
        )
    safe_end = length if end is None else end
    output = values[start:safe_end]
    bad = np.flatnonzero(~np.isfinite(output))
    if len(bad):
        raise _error("mujocoObservations", f"{label}[{int(bad[0])}] must be finite")
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# mujocoObservationTransforms.ts mirror
# ---------------------------------------------------------------------------


def parse_number_list(raw, label):
    if raw is None:
        return None

    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        value = _number(raw)
        if not math.isfinite(value):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return [value]

    if isinstance(raw, str):
        parts = [part for part in raw.replace(",", " ").split() if part]
        if not parts:
            return None
        values = [_number(part) for part in parts]
        if not all(math.isfinite(value) for value in values):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return values

    if isinstance(raw, (list, tuple, np.ndarray)):
        values = [_number(value) for value in raw]
        if not all(math.isfinite(value) for value in values):
            raise _error("mujocoObservations", f"{label} must contain finite numbers")
        return values

    raise _error(
        "mujocoObservations",
        f"{label} must be a number, list string, or numeric array",
    )


def _first_present(config, keys):
    """JS ``??`` chain: only ``None`` (and missing keys) fall through."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def configured_dim(config, fallback):
    raw = _first_present(config or {}, ("dim", "size", "length"))
    if raw is None or raw == "":
        if fallback > 0:
            return int(math.floor(fallback))
        raise _error("mujocoObservations", "observation dimension could not be inferred")
    dim = _number(raw)
    if math.isfinite(dim) and dim > 0:
        return int(math.floor(dim))
    raise _error(
        "mujocoObservations",
        f"observation dimension must be positive, got {raw}",
    )


def vector_from(raw, dim, label, fallback=0.0):
    if not _is_integer(dim) or dim <= 0:
        raise _error("mujocoObservations", f"{label} dimension must be a positive integer")
    dim = int(dim)
    output = np.full(dim, fallback, dtype=np.float32)
    values = parse_number_list(raw, label)
    if not values:
        return output
    count = min(len(values), dim)
    output[:count] = values[:count]
    return output


def constant_vector(config):
    config = config or {}
    values = parse_number_list(_first_present(config, ("values", "value")), "obs-constant values")
    if not values:
        raise _error("mujocoObservations", "obs-constant requires a finite values array")
    dim = configured_dim(config, len(values))
    if len(values) != dim:
        raise _error(
            "mujocoObservations",
            f"obs-constant values length {len(values)} does not match dim {dim}",
        )
    return np.asarray(values, dtype=np.float32)


def _index_map(config):
    config = config or {}
    values = parse_number_list(
        _first_present(config, ("indices", "joint2motorIdx", "joint2motor_idx")), "indices"
    )
    if not values:
        return None
    indices = [int(math.trunc(value)) for value in values]
    if any(value < 0 or not _is_integer(value) for value in values):
        raise _error("mujocoObservations", "indices must be non-negative integers")
    return indices


def _default_value_source(config):
    return _first_present(
        config,
        (
            "defaultAngles",
            "default_angles",
            "defaultPos",
            "default_pos",
            "defaultDofPos",
            "default_dof_pos",
            "reference",
            "offset",
        ),
    )


def _mapped_value(values, output_index, source_index, has_indices, fallback):
    if values is None:
        return fallback
    if len(values) == 1:
        return values[0]
    index = source_index if has_indices else output_index
    if index < 0 or index >= len(values):
        raise _error(
            "mujocoObservations",
            f"transform value index {index} exceeds length {len(values)}",
        )
    return values[index]


def transform_vector(values, config):
    """``transformVector``: index gather, default subtraction, scalar/vector scale."""
    config = config or {}
    values = np.asarray(values, dtype=np.float32)
    defaults = parse_number_list(_default_value_source(config), "default position")
    indices = _index_map(config)
    parsed_scale = parse_number_list(_first_present(config, ("scale", "scales")), "scale")
    scalar_scale = parsed_scale[0] if parsed_scale and len(parsed_scale) == 1 else math.nan
    scale_values = parsed_scale if parsed_scale and len(parsed_scale) > 1 else None

    if indices is not None:
        for index in indices:
            if index < 0 or index >= len(values):
                raise _error(
                    "mujocoObservations",
                    f"source index {index} exceeds observation length {len(values)}",
                )
        source_indices = np.asarray(indices, dtype=np.int64)
    else:
        source_indices = np.arange(len(values), dtype=np.int64)

    output = values[source_indices].astype(np.float32)
    output_length = len(output)
    if defaults is not None:
        output -= np.asarray(
            [
                _mapped_value(defaults, i, int(source_indices[i]), indices is not None, 0.0)
                for i in range(output_length)
            ],
            dtype=np.float32,
        )
    if scale_values is not None:
        output *= np.asarray(
            [
                _mapped_value(scale_values, i, int(source_indices[i]), indices is not None, 1.0)
                for i in range(output_length)
            ],
            dtype=np.float32,
        )
    elif math.isfinite(scalar_scale):
        output *= np.float32(scalar_scale)
    return output


# ---------------------------------------------------------------------------
# cameraObservation.ts + cameraObservationConfig.ts mirror
# ---------------------------------------------------------------------------


def _camera_node_data(data):
    """Obs-camera node data with its sensor resource config merged under it.

    ``mujocoPolicyController.ts`` ``parseNode`` builds the obs-camera node
    data as ``{...resource(sensorId).config, ...data}``, so the browser sees
    the preset's camera fields; the local executor hands the raw node data
    around, so the same merge happens here (node-level keys win).
    """
    data = dict(data or {})
    resource_id = data.get("resourceId")
    if resource_id is None or resource_id == "":
        resource_id = data.get("resource_id")
    if resource_id is None or resource_id == "":
        return data
    try:
        import botlab_host

        item = botlab_host.project_state().resource(str(resource_id), "sensor")
    except Exception as error:
        raise _error(
            "botlab",
            f'camera resource "{resource_id}" could not be resolved ({error})',
        ) from None
    merged = dict(item["config"])
    merged.update(data)
    return merged


def camera_output_mode(data=None):
    value = _camera_node_data(data).get("outputMode")
    if value in ("rgb", "depth", "rgbd"):
        return value
    return DEFAULT_CAMERA_OBSERVATION["outputMode"]


def camera_output_labels(data=None):
    mode = camera_output_mode(data)
    if mode == "rgb":
        return ["RGB"]
    if mode == "depth":
        return ["DEPTH"]
    return ["RGB", "DEPTH"]


def camera_frame_size(data=None):
    """Cropped output size (``cameraFrameSize`` reads the rendered size)."""

    def positive_size(raw, default):
        value = _number(raw)
        return int(math.floor(value)) if math.isfinite(value) and value > 0 else default

    source = _camera_node_data(data)
    width = positive_size(
        source.get("width", DEFAULT_CAMERA_OBSERVATION["width"]),
        DEFAULT_CAMERA_OBSERVATION["width"],
    )
    height = positive_size(
        source.get("height", DEFAULT_CAMERA_OBSERVATION["height"]),
        DEFAULT_CAMERA_OBSERVATION["height"],
    )
    return width, height


def _field_value(source, field):
    """JS ``source.field ?? DEFAULT[field]``: only None falls through."""
    raw = source.get(field)
    return DEFAULT_CAMERA_OBSERVATION[field] if raw is None else raw


def _positive_integer_field(source, field):
    value = _number(_field_value(source, field))
    if not _is_integer(value) or value <= 0:
        raise _error("CameraObservation", f"{field} must be a positive integer")
    return int(value)


def _positive_number_field(source, field):
    value = _number(_field_value(source, field))
    if not math.isfinite(value) or value <= 0:
        raise _error("CameraObservation", f"{field} must be a finite number > 0")
    return value


def _non_negative_number_field(source, field):
    value = _number(_field_value(source, field))
    if not math.isfinite(value) or value < 0:
        raise _error("CameraObservation", f"{field} must be a finite number >= 0")
    return value


def _non_negative_integer_field(source, field):
    value = _number(_field_value(source, field))
    if not _is_integer(value) or value < 0:
        raise _error("CameraObservation", f"{field} must be a non-negative integer")
    return int(value)


def _probability_field(source, field):
    value = _number(_field_value(source, field))
    if not math.isfinite(value) or value < 0 or value > 1:
        raise _error("CameraObservation", f"{field} must be a finite number between 0 and 1")
    return value


def _enum_field(source, field, allowed):
    value = _field_value(source, field)
    if value in allowed:
        return value
    raise _error(
        "CameraObservation", f"{field} must be {', '.join(str(item) for item in allowed)}"
    )


def _depth_backend_field(source):
    value = _field_value(source, "depthBackend")
    if value in ("raycaster", "webgpu-raycaster"):
        # Legacy projects may still pin the removed raycaster backends.
        return "render"
    if value == "render":
        return value
    raise _error("CameraObservation", "depthBackend must be render")


def _readback_pipeline_depth_field(source):
    depth = _positive_integer_field(source, "readbackPipelineDepth")
    if depth > MAX_CAMERA_READBACK_PIPELINE_DEPTH:
        raise _error(
            "CameraObservation",
            f"readbackPipelineDepth must be <= {MAX_CAMERA_READBACK_PIPELINE_DEPTH}",
        )
    return depth


def _gaussian_kernel_size_field(source):
    kernel_size = _non_negative_integer_field(source, "depthGaussianKernelSize")
    if kernel_size > 1 and (kernel_size < 3 or kernel_size % 2 == 0):
        raise _error(
            "CameraObservation", "depthGaussianKernelSize must be 0/1 or an odd integer >= 3"
        )
    return kernel_size


def _vector_field(source, field, length):
    values = parse_number_list(_field_value(source, field), field)
    if not values or len(values) != length or not all(
        math.isfinite(value) for value in values
    ):
        raise _error(
            "CameraObservation",
            f"{field} must contain exactly {length} finite numbers",
        )
    return values


def _optional_vector6_field(source, field):
    raw = source.get(field)
    if raw is None or raw == "":
        return None
    return _vector_field(source, field, 6)


def _optional_integer_list_field(source, field):
    raw = source.get(field)
    if raw is None or raw == "":
        return []
    values = parse_number_list(raw, field)
    if any(value < 0 or not _is_integer(value) for value in values):
        raise _error("CameraObservation", f"{field} must contain non-negative integers")
    return [int(value) for value in values]


def _optional_name_list_field(source, field):
    raw = source.get(field)
    if raw is None or raw == "":
        return []
    if not isinstance(raw, (list, tuple)) or any(not isinstance(item, str) for item in raw):
        raise _error("CameraObservation", f"{field} must be a list of names")
    names = [item.strip() for item in raw]
    if any(not name for name in names):
        raise _error("CameraObservation", f"{field} must not contain empty names")
    return names


def assert_crop_window(config):
    """``assertCropWindow``: the crop window must fit the full sensor."""
    if config["cropX"] + config["width"] > config["fullWidth"]:
        raise _error(
            "CameraObservation",
            f'cropX ({config["cropX"]}) + width ({config["width"]}) '
            f'exceeds fullWidth ({config["fullWidth"]})',
        )
    if config["cropY"] + config["height"] > config["fullHeight"]:
        raise _error(
            "CameraObservation",
            f'cropY ({config["cropY"]}) + height ({config["height"]}) '
            f'exceeds fullHeight ({config["fullHeight"]})',
        )


def read_camera_config(node_data):
    """``readCameraConfig``: parse + validate one obs-camera node's config.

    ``fullWidth``/``fullHeight`` are the pinhole sensor resolution the
    renderer produces (``fov`` is its vertical FoV); ``width``/``height`` are
    the cropped observation output. Returns a plain dict with the
    ``cameraObservationConfig.ts`` ``CameraConfig`` field names.
    """
    source = _camera_node_data(node_data)

    raw_target = source.get("target")
    target = str(
        DEFAULT_CAMERA_OBSERVATION["target"] if raw_target is None else raw_target
    ).strip()
    if not target:
        raise _error("CameraObservation", "target is required")

    width = _positive_integer_field(source, "width")
    height = _positive_integer_field(source, "height")
    raw_full_width = source.get("fullWidth")
    full_width = width if raw_full_width is None else _positive_integer_field(
        {"fullWidth": raw_full_width}, "fullWidth"
    )
    raw_full_height = source.get("fullHeight")
    full_height = height if raw_full_height is None else _positive_integer_field(
        {"fullHeight": raw_full_height}, "fullHeight"
    )
    raw_crop_x = source.get("cropX")
    crop_x = 0 if raw_crop_x is None else _non_negative_integer_field(
        {"cropX": raw_crop_x}, "cropX"
    )
    raw_crop_y = source.get("cropY")
    crop_y = 0 if raw_crop_y is None else _non_negative_integer_field(
        {"cropY": raw_crop_y}, "cropY"
    )

    near = _positive_number_field(source, "near")
    far = _positive_number_field(source, "far")
    if far <= near:
        raise _error("CameraObservation", "far must be greater than near")

    raw_xyaxes = source.get("cameraXyaxes")
    rotation_mode = "xyaxes" if not (raw_xyaxes is None or raw_xyaxes == "") else "euler"

    horizontal_fov = None
    if source.get("horizontalFov") is not None:
        horizontal_fov = _positive_number_field(source, "horizontalFov")
        if horizontal_fov >= 180:
            raise _error("CameraObservation", "horizontalFov must be less than 180 degrees")

    config = {
        "outputMode": _enum_field(source, "outputMode", ("rgb", "depth", "rgbd")),
        "targetKind": _enum_field(source, "targetKind", ("body", "site")),
        "rayAlignment": _enum_field(source, "rayAlignment", ("body", "yaw")),
        "cameraOffsetFrame": _enum_field(source, "cameraOffsetFrame", ("ray", "body")),
        "depthBackend": _depth_backend_field(source),
        "target": target,
        "width": width,
        "height": height,
        "fullWidth": full_width,
        "fullHeight": full_height,
        "cropX": crop_x,
        "cropY": crop_y,
        "fov": _positive_number_field(source, "fov"),
        "horizontalFov": horizontal_fov,
        "near": near,
        "far": far,
        "depthScale": _positive_number_field(source, "depthScale"),
        "depthGaussianKernelSize": _gaussian_kernel_size_field(source),
        "depthGaussianSigma": _positive_number_field(source, "depthGaussianSigma"),
        "depthNoiseStd": _non_negative_number_field(source, "depthNoiseStd"),
        "depthNoiseDistanceStd": _non_negative_number_field(
            source, "depthNoiseDistanceStd"
        ),
        "depthDropoutProbability": _probability_field(source, "depthDropoutProbability"),
        "depthDropoutPatchCount": _non_negative_integer_field(
            source, "depthDropoutPatchCount"
        ),
        "depthDropoutPatchRadius": _non_negative_integer_field(
            source, "depthDropoutPatchRadius"
        ),
        "depthNoiseSeed": _non_negative_integer_field(source, "depthNoiseSeed"),
        "realtimeReadbackBackoffMs": _non_negative_number_field(
            source, "realtimeReadbackBackoffMs"
        ),
        "realtimeReadbackIntervalMs": _non_negative_number_field(
            source, "realtimeReadbackIntervalMs"
        ),
        "realtimeReadbackSlowThresholdMs": _non_negative_number_field(
            source, "realtimeReadbackSlowThresholdMs"
        ),
        "strictReadbackIntervalMs": _non_negative_number_field(
            source, "strictReadbackIntervalMs"
        ),
        "readbackMode": _enum_field(source, "readbackMode", ("async", "sync")),
        "depthReadbackFormat": _enum_field(
            source, "depthReadbackFormat", ("packed-rgba8", "linear-float32")
        ),
        "readbackPipelineDepth": _readback_pipeline_depth_field(source),
        "raycastGeomGroups": _optional_integer_list_field(source, "raycastGeomGroups"),
        "excludeGeoms": _optional_name_list_field(source, "excludeGeoms"),
        "excludeBodies": _optional_name_list_field(source, "excludeBodies"),
        "cameraOffset": _vector_field(source, "cameraOffset", 3),
        "cameraRotationMode": (
            rotation_mode
            if source.get("cameraRotationMode") is None
            else _enum_field(source, "cameraRotationMode", ("euler", "xyaxes"))
        ),
        "cameraEulerDeg": _vector_field(source, "cameraEulerDeg", 3),
        "cameraXyaxes": _optional_vector6_field(source, "cameraXyaxes"),
    }
    assert_crop_window(config)
    return config


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


# ---------------------------------------------------------------------------
# mujocoObservations.ts mirror — kinematics helpers
# ---------------------------------------------------------------------------


def _has_root_free_joint(model, data):
    if int(model.njnt) < 1:
        return False
    qpos = _flat(data.qpos, "data.qpos")
    qvel = _flat(data.qvel, "data.qvel")
    return (
        int(model.jnt_type[0]) == 0
        and int(model.jnt_qposadr[0]) == 0
        and int(model.jnt_dofadr[0]) == 0
        and len(qpos) >= FREE_JOINT_QPOS_DIM
        and len(qvel) >= FREE_JOINT_QVEL_DIM
    )


def _root_qpos_offset(model, data):
    return FREE_JOINT_QPOS_DIM if _has_root_free_joint(model, data) else 0


def _root_qvel_offset(model, data):
    return FREE_JOINT_QVEL_DIM if _has_root_free_joint(model, data) else 0


def _base_body_index(model):
    return 1 if int(model.nbody) > 1 else 0


def _body_index_by_target(model, raw_target, name_prefix=""):
    target = str(raw_target if raw_target is not None else "").strip()
    if not target:
        raise _error("mujocoObservations", "projected gravity body target is required")
    nbody = int(model.nbody)
    if target.isascii() and target.isdigit():
        index = int(target)
        if 0 <= index < nbody:
            return index
        raise _error(
            "mujocoObservations",
            f"body index {index} is out of bounds (nbody={nbody})",
        )

    # Scoped name first (composed scenes rename extra-robot bodies), unprefixed second.
    candidates = [f"{name_prefix}{target}", target] if name_prefix else [target]
    found = {}
    for index in range(nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        if name in candidates and name not in found:
            found[name] = index
    for candidate in candidates:
        if candidate in found:
            return found[candidate]
    raise _error("mujocoObservations", f'body "{target}" was not found')


def _sensor_slice_by_name(model, data, names, length, label):
    sensor_count = int(model.nsensor)
    if sensor_count <= 0:
        return None
    wanted = set(names)
    for sensor in range(sensor_count):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor) or ""
        if name not in wanted:
            continue
        start = int(model.sensor_adr[sensor])
        dim = int(model.sensor_dim[sensor])
        if dim < length:
            raise _error(
                "mujocoObservations",
                f'{label} sensor "{name}" dim {dim} is smaller than {length}',
            )
        sensordata = _flat(data.sensordata, "data.sensordata")
        if len(sensordata) < start + length:
            raise _error(
                "mujocoObservations",
                f"{label} requires G1 sensordata[{start}:{start + length}], "
                f"got length {len(sensordata)}",
            )
        return _float32_slice(sensordata, label, start, start + length)
    return None


def _is_g1_config(config):
    config = config or {}
    return (
        config.get("robot") == "g1"
        or config.get("resolvedRobot") == "g1"
        or config.get("robotProfileApplied") == "g1"
    )


def _root_kinematic_quaternion(model, data):
    if _has_root_free_joint(model, data):
        return _float32_slice(data.qpos, "data.qpos", 3, 7)
    body_index = _base_body_index(model)
    address = body_index * 4
    return _float32_slice(data.xquat, "data.xquat", address, address + 4)


def _root_quaternion(model, data, config):
    if _is_g1_config(config):
        sensor_quat = _sensor_slice_by_name(
            model, data, ["imu_quat", "secondary_imu_quat"], 4, "data.sensordata imu_quat"
        )
        if sensor_quat is not None:
            return sensor_quat
    return _root_kinematic_quaternion(model, data)


def _body_quaternion(model, data, target, name_prefix=""):
    body_index = _body_index_by_target(model, target, name_prefix)
    address = body_index * 4
    return _float32_slice(data.xquat, "data.xquat", address, address + 4)


def _projected_gravity_quaternion(model, data, config, name_prefix=""):
    config = config or {}
    if config.get("targetKind") == "body" or config.get("gravityTargetKind") == "body":
        target = _first_present(config, ("target", "body", "bodyName"))
        return _body_quaternion(model, data, target, name_prefix)
    return _root_quaternion(model, data, config)


def _inverse_rotate_by_quaternion(vector, quat):
    """Apply R(q)^T to a 3-vector (quat wxyz), exactly like the TS helper."""
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if len(v) < 3:
        raise _error("mujocoObservations", f"vector[{2}] is out of bounds (length {len(v)})")
    if len(q) < 4:
        raise _error("mujocoObservations", f"quat[{3}] is out of bounds (length {len(q)})")
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.asarray(
        [
            (1 - 2 * (yy + zz)) * v[0] + 2 * (xy + wz) * v[1] + 2 * (xz - wy) * v[2],
            2 * (xy - wz) * v[0] + (1 - 2 * (xx + zz)) * v[1] + 2 * (yz + wx) * v[2],
            2 * (xz + wy) * v[0] + 2 * (yz - wx) * v[1] + (1 - 2 * (xx + yy)) * v[2],
        ],
        dtype=np.float32,
    )


def projected_gravity_from_quaternion(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if len(quat) < 4:
        raise _error("mujocoObservations", f"quat[{3}] is out of bounds (length {len(quat)})")
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return np.asarray(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def _velocity_source(config):
    config = config or {}
    return str(
        _first_present(config, ("velocitySource", "baseVelocitySource")) or "auto"
    ).strip().lower()


def _should_read_g1_velocity_sensor(config):
    source = _velocity_source(config)
    return source not in ("root", "qvel", "freejoint") and _is_g1_config(config)


def _base_linear_velocity(model, data, config):
    if _velocity_source(config) == "sensor" and _is_g1_config(config):
        sensor_velocity = _sensor_slice_by_name(
            model,
            data,
            ["imu_lin_vel", "base_lin_vel", "secondary_imu_lin_vel"],
            3,
            "data.sensordata imu_lin_vel",
        )
        if sensor_velocity is not None:
            return sensor_velocity
    if _has_root_free_joint(model, data):
        return _inverse_rotate_by_quaternion(
            _float32_slice(data.qvel, "data.qvel", 0, 3),
            _root_kinematic_quaternion(model, data),
        )
    body_index = _base_body_index(model)
    address = body_index * 6
    return _float32_slice(data.cvel, "data.cvel", address + 3, address + 6)


def _base_angular_velocity(model, data, config):
    if _should_read_g1_velocity_sensor(config):
        sensor_gyro = _sensor_slice_by_name(
            model,
            data,
            ["imu_gyro", "imu_ang_vel", "secondary_imu_gyro"],
            3,
            "data.sensordata imu_gyro",
        )
        if sensor_gyro is not None:
            return sensor_gyro
    if _has_root_free_joint(model, data):
        # Match the browser contract: marked plans use MuJoCo's body-local
        # rotational qvel; previously published unmarked plans keep their path.
        if (config or {}).get("rootAngularVelocityFrame") == "body":
            return _float32_slice(data.qvel, "data.qvel", 3, 6)
        if "rootAngularVelocityFrame" in (config or {}):
            raise _error("mujocoObservations", "rootAngularVelocityFrame must be body")
        return _inverse_rotate_by_quaternion(
            _float32_slice(data.qvel, "data.qvel", 3, 6),
            _root_kinematic_quaternion(model, data),
        )
    body_index = _base_body_index(model)
    address = body_index * 6
    return _float32_slice(data.cvel, "data.cvel", address, address + 3)


def _base_height(model, data):
    if _has_root_free_joint(model, data):
        qpos = _flat(data.qpos, "data.qpos")
        return np.asarray([qpos[2]], dtype=np.float32)
    body_index = _base_body_index(model)
    xpos = _flat(data.xpos, "data.xpos")
    return np.asarray([xpos[body_index * 3 + 2]], dtype=np.float32)


def _dim_bounded_end(node_data, start, source, label):
    """Narrow an offset-based slice to the observation's declared dim."""
    raw = _first_present(node_data or {}, ("dim", "size", "length"))
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
    if raw is None:
        raw = _first_present(node_data, ("values", "action", "actions"))
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
# mujocoObservationCommands.ts mirror — local (non-interactive) subset
# ---------------------------------------------------------------------------

_INTERACTIVE_COMMAND_MODES = frozenset(
    {
        "keyboard",
        "gamepad",
        "keyboard-gamepad",
        "keyboard-buttons",
        "gamepad-buttons",
        "keyboard-gamepad-buttons",
    }
)

# mujocoObservationCommands.ts pose-velocity constants (config keys keep
# camelCase).
DEFAULT_POSE_VELOCITY_LOOKAHEAD = 2.0
DEFAULT_POSE_VELOCITY_STIFFNESS = 2.0
DEFAULT_HEADING_STIFFNESS = 2.0
DEFAULT_TARGET_DISTANCE_THRESHOLD = 0.4
FORWARD_ACTIVE_EPSILON = 1e-4
YAW_ACTIVE_EPSILON = 1e-4

# keyboardCommands.ts commandAxisConfig defaults.
_DEFAULT_COMMAND_AXIS_MIN = (-1.0, -1.0, -1.0)
_DEFAULT_COMMAND_AXIS_MAX = (1.0, 1.0, 1.0)
_DEFAULT_COMMAND_AXIS_INDICES = (0, 1, 2)


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def _wrap_pi(value):
    wrapped = value
    while wrapped > math.pi:
        wrapped -= 2 * math.pi
    while wrapped < -math.pi:
        wrapped += 2 * math.pi
    return wrapped


def _finite_number_field(raw, field, fallback):
    """``finiteNumber``: missing/empty keeps the fallback, non-finite errors."""
    if raw is None or raw == "":
        return fallback
    value = _number(raw)
    if math.isfinite(value):
        return value
    raise _error("mujocoObservations", f"{field} must be finite")


def _positive_finite_number(raw, field, fallback):
    value = _finite_number_field(raw, field, fallback)
    if value > 0:
        return value
    raise _error("mujocoObservations", f"{field} must be > 0")


def _non_negative_finite_number(raw, field, fallback):
    value = _finite_number_field(raw, field, fallback)
    if value >= 0:
        return value
    raise _error("mujocoObservations", f"{field} must be >= 0")


# ---------------------------------------------------------------------------
# keyboardCommands.ts commandAxisConfig mirror (axis indices/bounds)
# ---------------------------------------------------------------------------


def _axis_number_list(raw, field):
    """``finiteNumberList`` input decoding: string / list / numeric array."""
    if isinstance(raw, str):
        return [_number(part) for part in raw.replace(",", " ").split() if part]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [_number(item) for item in raw]
    raise _error("keyboardCommands", f"{field} must be a number list")


def _command_axis_indices(config, dim):
    raw = config.get("commandIndices")
    if raw is None or raw == "":
        values = list(_DEFAULT_COMMAND_AXIS_INDICES)
    else:
        values = _axis_number_list(raw, "commandIndices")
        if len(values) != 3:
            raise _error(
                "keyboardCommands", "commandIndices must contain exactly 3 numbers"
            )
    if not all(math.isfinite(value) for value in values):
        raise _error("keyboardCommands", "commandIndices contains a non-finite number")
    if any(not _is_integer(value) for value in values):
        raise _error("keyboardCommands", "commandIndices must contain integers")

    seen = set()
    for index in values:
        if index < 0 or index >= dim:
            raise _error(
                "keyboardCommands",
                f"command index {index:g} is outside dim {dim}",
            )
        if index in seen:
            raise _error("keyboardCommands", f"duplicate command index {index:g}")
        seen.add(index)
    # Validated integers, cast for use as array/list indices.
    return [int(value) for value in values]


def _command_axis_bounds(raw, field, dim, indices, fallback):
    if raw is None or raw == "":
        values = list(fallback)
    else:
        values = _axis_number_list(raw, field)
    if len(values) != 3 and len(values) != dim:
        raise _error("keyboardCommands", f"{field} must contain 3 or {dim} numbers")
    if not all(math.isfinite(value) for value in values):
        raise _error("keyboardCommands", f"{field} contains a non-finite number")
    return values if len(values) == 3 else [values[index] for index in indices]


def command_axis_config(config, dim):
    """``commandAxisConfig``: the 3 command axes' indices and bounds."""
    config = config or {}
    indices = _command_axis_indices(config, dim)
    minimum = _command_axis_bounds(
        config.get("cmdMin"), "cmdMin", dim, indices, _DEFAULT_COMMAND_AXIS_MIN
    )
    maximum = _command_axis_bounds(
        config.get("cmdMax"), "cmdMax", dim, indices, _DEFAULT_COMMAND_AXIS_MAX
    )
    for i in range(3):
        if minimum[i] > maximum[i]:
            raise _error("keyboardCommands", f"cmdMin[{i}] must be <= cmdMax[{i}]")
    return {"indices": indices, "min": minimum, "max": maximum}


def _parse_root_x_range(raw, label):
    values = parse_number_list(raw, label)
    if not values or len(values) != 2:
        raise _error("mujocoObservations", f"{label} must contain exactly 2 numbers")
    if values[0] > values[1]:
        raise _error("mujocoObservations", f"{label}[0] must be <= {label}[1]")
    return (values[0], values[1])


def _parse_root_x_ranges(raw, label):
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple, np.ndarray)) and (
        not raw or not isinstance(raw[0], (list, tuple, np.ndarray))
    ):
        flat = parse_number_list(raw, label) or []
        if len(flat) == 2:
            return [(flat[0], flat[1])]
        if len(flat) % 2 != 0:
            raise _error(
                "mujocoObservations", f"{label} flat list must contain range pairs"
            )
        return [
            _parse_root_x_range([flat[i * 2], flat[i * 2 + 1]], f"{label}[{i}]")
            for i in range(len(flat) // 2)
        ]
    if isinstance(raw, (str, bool, int, float, np.integer, np.floating)):
        flat = parse_number_list(raw, label) or []
        if len(flat) == 2:
            return [(flat[0], flat[1])]
        if len(flat) % 2 != 0:
            raise _error(
                "mujocoObservations", f"{label} flat list must contain range pairs"
            )
        return [
            _parse_root_x_range([flat[i * 2], flat[i * 2 + 1]], f"{label}[{i}]")
            for i in range(len(flat) // 2)
        ]
    if not isinstance(raw, (list, tuple)):
        raise _error("mujocoObservations", f"{label} must be a range or range list")
    return [_parse_root_x_range(item, f"{label}[{i}]") for i, item in enumerate(raw)]


def _yaw_from_quaternion(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _root_pose_2d(context):
    """``rootPose2d``: the root's world (x, y, yaw) pose."""
    model, data = context["model"], context["data"]
    if _has_root_free_joint(model, data):
        qpos = _flat(data.qpos, "data.qpos")
        return (
            float(qpos[0]),
            float(qpos[1]),
            _yaw_from_quaternion(qpos[3], qpos[4], qpos[5], qpos[6]),
        )
    index = _base_body_index(model)
    xpos = _flat(data.xpos, "data.xpos")
    xquat = _flat(data.xquat, "data.xquat")
    pos_start = index * 3
    quat_start = index * 4
    return (
        float(xpos[pos_start]),
        float(xpos[pos_start + 1]),
        _yaw_from_quaternion(
            xquat[quat_start],
            xquat[quat_start + 1],
            xquat[quat_start + 2],
            xquat[quat_start + 3],
        ),
    )


def _root_pose_x(context):
    """x coordinate of rootPose2d (mujocoObservationCommands.ts)."""
    return _root_pose_2d(context)[0]


def _write_command_mapping(output, mapping, label):
    values = mapping.get("values")
    if values is not None:
        output[:] = vector_from(values, len(output), f"{label}.values")
        return
    index = _number(mapping.get("index"))
    if not _is_integer(index) or index < 0 or index >= len(output):
        raise _error(
            "mujocoObservations",
            f"{label}.index must be an integer in [0, {len(output) - 1}]",
        )
    raw_value = mapping.get("value")
    value = 1.0 if raw_value is None or raw_value == "" else _number(raw_value)
    if not math.isfinite(value):
        raise _error("mujocoObservations", f"{label}.value must be finite")
    output.fill(0.0)
    output[int(index)] = value


def _command_key_list(raw, label):
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [part for part in re.split(r"[\s,]+", raw) if part]
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    raise _error(
        "mujocoObservations", f"{label} must be a string or string array"
    )


def _local_command_mapping_active(mapping, label):
    all_keys = _command_key_list(
        _first_present(mapping, ("keys", "allKeys")), f"{label}.keys"
    )
    any_keys = _command_key_list(
        _first_present(mapping, ("key", "anyKey", "anyKeys")),
        f"{label}.anyKeys",
    )
    pressed = _LOCAL_COMMAND_INPUT_STATE["pressed_keys"]
    if all_keys and not all(key in pressed for key in all_keys):
        return False
    if any_keys and not any(key in pressed for key in any_keys):
        return False
    # 手柄按钮触发器（gamepad.yaml 的 mappings）：本地输入状态没有手柄，
    # 未按下的按钮映射不激活——与浏览器侧 commandMappingActive 同一 gate。
    button = mapping.get("button")
    if isinstance(button, str) and button:
        if button not in _LOCAL_COMMAND_INPUT_STATE["pressed_buttons"]:
            return False
    return True


def _mapped_command_vector(config, output, context, node_id=None):
    # CLI/AI 命令注入优先于实体按键/手柄映射（浏览器侧 runtimeIoCommands 的
    # 本地镜像）：注入向量整段覆盖，清回静息码由注入方负责。
    if node_id is not None:
        injected = _LOCAL_COMMAND_INPUT_STATE["injected_commands"].get(node_id)
        if injected is not None:
            output[:] = vector_from(injected, len(output), "injected command values")
            return output
    raw_mappings = _first_present(config, ("commandMappings", "commandMap", "mappings"))
    if raw_mappings is None or raw_mappings == "":
        return output
    if not isinstance(raw_mappings, (list, tuple)):
        raise _error("mujocoObservations", "commandMappings must be an array")

    for i, mapping in enumerate(raw_mappings):
        label = f"commandMappings[{i}]"
        if not isinstance(mapping, dict):
            raise _error("mujocoObservations", f"{label} must be an object")
        if not _local_command_mapping_active(mapping, label):
            continue
        ranges = [
            *_parse_root_x_ranges(
                _first_present(mapping, ("rootXRange", "rootRange")), f"{label}.rootXRange"
            ),
            *_parse_root_x_ranges(
                _first_present(mapping, ("rootXRanges", "rootRanges")), f"{label}.rootXRanges"
            ),
        ]
        if ranges:
            x = _root_pose_x(context)
            if not any(minimum <= x <= maximum for minimum, maximum in ranges):
                continue
        _write_command_mapping(output, mapping, label)
        return output
    return output


def _object_command_vector(raw, dim):
    def part(*keys):
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key]
        return 0
    return vector_from(
        [
            part("grasp"),
            part("linX", "vx", "x"),
            part("linY", "vy", "y"),
            part("yaw", "wz"),
            part("pitch"),
        ],
        dim,
        "obs-commands",
    )


_pose_velocity_state_by_node = {}

# Native headless runs have an input device state just like the browser host,
# but no keyboard/gamepad is attached to it. Keeping this as explicit state is
# important: an inactive key gate means "try the next mapping / keep the
# declared default", not "ignore a feature the runtime failed to implement".
# ``injected_commands`` carries the discrete command injection (node id ->
# values): the headless counterpart of the browser's CLI/AI command channel.
_LOCAL_COMMAND_INPUT_STATE = {
    "pressed_keys": frozenset(),
    "pressed_buttons": frozenset(),
    "axis_active": False,
    "injected_commands": {},
}


def set_injected_command(node_id, values):
    """Inject one mapped commands node's value from the next control step on.

    Mirrors the browser's ``runtimeIoCommands``: the injected vector covers the
    whole node output (overriding held key mappings) until
    :func:`clear_injected_commands` or the matching ``set_injected_command``
    call replaces it. ``HeadlessRun.set_command`` range-checks the values
    against the node's mapping codes before calling this.
    """
    _LOCAL_COMMAND_INPUT_STATE["injected_commands"][str(node_id)] = tuple(
        float(value) for value in values
    )


def clear_injected_commands(node_id=None):
    """Drop one node's injection, or all of them (``reset`` semantics: the
    commands fall back to their resting codes)."""
    if node_id is None:
        _LOCAL_COMMAND_INPUT_STATE["injected_commands"].clear()
    else:
        _LOCAL_COMMAND_INPUT_STATE["injected_commands"].pop(str(node_id), None)


def clear_pose_velocity_states(node_id=None):
    """``resetPoseVelocityCommandState``: drop one node's state or all of it.

    The controller reset/dispose loop calls this per obs-commands node so a
    replayed run re-anchors centerY/targetX from the reset pose instead of
    the pre-reset trajectory.
    """
    if node_id is None:
        _pose_velocity_state_by_node.clear()
        return
    _pose_velocity_state_by_node.pop(node_id, None)


def _pose_velocity_command_vector(config, node_id, output, context):
    """``poseVelocityCommandVector`` (mujocoObservationCommands.ts).

    Waypoint-tracking rewrite of the command vector against the root pose:
    while a forward command is active the robot steers toward a lookahead
    target on the line it was walking (centerY anchors the lateral target,
    targetX only ever advances). The browser version receives the vector
    after the interactive overrides; the local baseline is the
    parameter-injected vector, which is exactly the browser's no-input
    behavior. This is per-control-step scalar math, so it follows the TS
    line by line instead of vectorizing.
    """
    config = config or {}
    if config.get("poseVelocityCommand") not in (True, "true", 1, "1"):
        return output
    if context is None:
        raise _error(
            "mujocoObservations",
            "poseVelocityCommand requires MuJoCo model/data context",
        )
    if len(output) < 3:
        raise _error(
            "mujocoObservations", "poseVelocityCommand requires command dim >= 3"
        )
    if not node_id:
        raise _error(
            "mujocoObservations", "poseVelocityCommand obs-commands requires nodeId"
        )

    axes = command_axis_config(config, len(output))
    indices, minimum, maximum = axes["indices"], axes["min"], axes["max"]
    raw_vx = float(output[indices[0]])
    raw_yaw = float(output[indices[2]])
    yaw_min_forward_velocity = _non_negative_finite_number(
        config.get("yawMinForwardVelocity"), "yawMinForwardVelocity", 0
    )
    # Yaw-only commands still get a minimum forward velocity so the robot
    # keeps walking through turns.
    yaw_forward_assist_active = (
        yaw_min_forward_velocity > 0
        and raw_vx <= FORWARD_ACTIVE_EPSILON
        and abs(raw_yaw) > YAW_ACTIVE_EPSILON
    )
    if yaw_forward_assist_active:
        output[indices[0]] = _clamp(yaw_min_forward_velocity, minimum[0], maximum[0])
        return output

    commanded_vx = float(output[indices[0]])
    forward_active = commanded_vx > FORWARD_ACTIVE_EPSILON
    pose_x, pose_y, pose_yaw = _root_pose_2d(context)

    state = _pose_velocity_state_by_node.get(node_id)
    if state is None or not forward_active or not state["wasForwardActive"]:
        state = {
            "centerY": pose_y,
            "targetX": pose_x
            + _positive_finite_number(
                config.get("targetLookahead"),
                "targetLookahead",
                DEFAULT_POSE_VELOCITY_LOOKAHEAD,
            ),
            "wasForwardActive": forward_active,
        }
        _pose_velocity_state_by_node[node_id] = state

    if not forward_active:
        state["wasForwardActive"] = False
        return output

    lookahead = _positive_finite_number(
        config.get("targetLookahead"),
        "targetLookahead",
        DEFAULT_POSE_VELOCITY_LOOKAHEAD,
    )
    velocity_stiffness = _positive_finite_number(
        config.get("velocityControlStiffness"),
        "velocityControlStiffness",
        DEFAULT_POSE_VELOCITY_STIFFNESS,
    )
    heading_stiffness = _positive_finite_number(
        config.get("headingControlStiffness"),
        "headingControlStiffness",
        DEFAULT_HEADING_STIFFNESS,
    )
    target_distance_threshold = _non_negative_finite_number(
        config.get("targetDisThreshold"),
        "targetDisThreshold",
        DEFAULT_TARGET_DISTANCE_THRESHOLD,
    )

    state["wasForwardActive"] = True
    state["targetX"] = max(state["targetX"], pose_x + lookahead)

    dx = state["targetX"] - pose_x
    dy = state["centerY"] - pose_y
    target_distance = math.hypot(dx, dy)
    if target_distance <= target_distance_threshold:
        output[indices[0]] = 0
        output[indices[1]] = 0
        output[indices[2]] = raw_yaw
        return output

    cos_yaw = math.cos(pose_yaw)
    sin_yaw = math.sin(pose_yaw)
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    target_heading = math.atan2(dy, dx)
    heading_command = _wrap_pi(target_heading - pose_yaw) * heading_stiffness

    output[indices[0]] = _clamp(
        body_x * velocity_stiffness, minimum[0], min(maximum[0], commanded_vx)
    )
    output[indices[1]] = _clamp(body_y * velocity_stiffness, minimum[1], maximum[1])
    output[indices[2]] = _clamp(heading_command + raw_yaw, minimum[2], maximum[2])
    return output


def _command_vector(config, node_id, context):
    config = config or {}
    command_mode = config.get("commandMode")
    if command_mode in (
        "keyboard-buttons",
        "gamepad-buttons",
        "keyboard-gamepad-buttons",
    ):
        return np.zeros(1, dtype=np.float32)
    if command_mode in _INTERACTIVE_COMMAND_MODES:
        # Explicit native-host input state is neutral. The declared project
        # command therefore stays unchanged, matching browser no-input state.
        assert not _LOCAL_COMMAND_INPUT_STATE["pressed_keys"]
        assert not _LOCAL_COMMAND_INPUT_STATE["pressed_buttons"]
        assert not _LOCAL_COMMAND_INPUT_STATE["axis_active"]

    dim = configured_dim(config, DEFAULT_COMMAND_DIM)
    raw = _first_present(config, ("values", "commands", "command", "cmd", "cmd_init"))
    if isinstance(raw, dict):
        output = _object_command_vector(raw, dim)
    else:
        output = vector_from(raw if raw is not None else [], dim, "obs-commands")

    if config.get("commandEncoding") in ("one-hot", "mapped") or config.get("encoding") in (
        "one-hot",
        "mapped",
    ):
        return _mapped_command_vector(config, output, context, node_id)
    return _pose_velocity_command_vector(config, node_id, output, context)


# ---------------------------------------------------------------------------
# mujocoAccessRegistry.ts mirror
# ---------------------------------------------------------------------------

_OBJ_TYPES = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "joint": mujoco.mjtObj.mjOBJ_JOINT,
    "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
}

_RAW_QUERIES = {
    "qpos": "qpos",
    "qvel": "qvel",
    "ctrl": "ctrl",
    "sensordata": "sensordata",
    "body.xpos.all": "xpos",
    "body.xquat.all": "xquat",
    "body.xmat.all": "xmat",
    "body.cvel.all": "cvel",
    "geom.xpos.all": "geom_xpos",
    "geom.xmat.all": "geom_xmat",
    "site.xpos.all": "site_xpos",
    "site.xmat.all": "site_xmat",
    "actuator.force.all": "actuator_force",
    "actuator.length.all": "actuator_length",
    "actuator.velocity.all": "actuator_velocity",
    "qfrc.applied": "qfrc_applied",
    "qfrc.bias": "qfrc_bias",
    "qfrc.passive": "qfrc_passive",
    "qfrc.actuator": "qfrc_actuator",
    "qfrc.constraint": "qfrc_constraint",
}

# query id -> (entity kind, data field, row size, count attribute)
_ROW_QUERIES = {
    "body.xpos": ("body", "xpos", 3, "nbody"),
    "body.xquat": ("body", "xquat", 4, "nbody"),
    "body.xmat": ("body", "xmat", 9, "nbody"),
    "body.cvel": ("body", "cvel", 6, "nbody"),
    "body.linvel.world": ("body", "body_linvel_world", 3, "nbody"),
    "body.linvel.local": ("body", "body_linvel_local", 3, "nbody"),
    "body.angvel.world": ("body", "body_angvel_world", 3, "nbody"),
    "body.angvel.local": ("body", "body_angvel_local", 3, "nbody"),
    "geom.xpos": ("geom", "geom_xpos", 3, "ngeom"),
    "geom.xmat": ("geom", "geom_xmat", 9, "ngeom"),
    "site.xpos": ("site", "site_xpos", 3, "nsite"),
    "site.xmat": ("site", "site_xmat", 9, "nsite"),
    "body.contactforce": ("body", "contact_force_body", 3, "nbody"),
    "geom.contactforce": ("geom", "contact_force_geom", 3, "ngeom"),
    "body.contact": ("body", "contact_body", 1, "nbody"),
    "geom.contact": ("geom", "contact_geom", 1, "ngeom"),
    "body.air_time": ("body", "body_current_air_time", 1, "nbody"),
    "body.contact_time": ("body", "body_current_contact_time", 1, "nbody"),
    "body.last_air_time": ("body", "body_last_air_time", 1, "nbody"),
    "body.last_contact_time": ("body", "body_last_contact_time", 1, "nbody"),
    "body.first_contact": ("body", "body_first_contact", 1, "nbody"),
    "geom.air_time": ("geom", "geom_current_air_time", 1, "ngeom"),
    "geom.contact_time": ("geom", "geom_current_contact_time", 1, "ngeom"),
    "geom.last_air_time": ("geom", "geom_last_air_time", 1, "ngeom"),
    "geom.last_contact_time": ("geom", "geom_last_contact_time", 1, "ngeom"),
    "geom.first_contact": ("geom", "geom_first_contact", 1, "ngeom"),
}

_JOINT_QUERIES = {
    "joint.qpos": "qpos",
    "joint.qvel": "qvel",
    "joint.qfrc_applied": "qfrc_applied",
    "joint.qfrc_bias": "qfrc_bias",
    "joint.qfrc_passive": "qfrc_passive",
    "joint.qfrc_actuator": "qfrc_actuator",
    "joint.qfrc_constraint": "qfrc_constraint",
}

_SENSOR_QUERIES = {"sensor.data"}

_SCALAR_ENTITY_QUERIES = {
    "actuator.ctrl": "ctrl",
    "actuator.force": "actuator_force",
    "actuator.length": "actuator_length",
    "actuator.velocity": "actuator_velocity",
}


def _supported_query_ids():
    return sorted(
        list(_RAW_QUERIES)
        + list(_ROW_QUERIES)
        + list(_JOINT_QUERIES)
        + list(_SENSOR_QUERIES)
        + list(_SCALAR_ENTITY_QUERIES)
    )


def _unsupported_query_error(query):
    return _error(
        "mujocoAccessRegistry",
        f'unsupported MuJoCo query "{query}"; '
        f"expected one of {', '.join(_supported_query_ids())}",
    )


def _resolve_entity_index(model, kind, count, raw_target, name_prefix=""):
    target = str(raw_target if raw_target is not None else "").strip()
    if not target:
        raise _error("mujocoAccessRegistry", f"{kind} target is required")
    if target.isascii() and target.isdigit():
        index = int(target)
        if 0 <= index < count:
            return index
        raise _error(
            "mujocoAccessRegistry",
            f"{kind} index {index} is out of bounds (count {count})",
        )

    obj_type = _OBJ_TYPES[kind]
    # The robot's own namespace wins: composed scenes rename every entity of
    # an extra robot to `<prefix><name>`; an unprefixed name remains an
    # explicit scene-global escape.
    candidates = [f"{name_prefix}{target}", target] if name_prefix else [target]
    found = {}
    for index in range(count):
        name = mujoco.mj_id2name(model, obj_type, index) or ""
        if name in candidates and name not in found:
            found[name] = index
    for candidate in candidates:
        if candidate in found:
            return found[candidate]
    suffix = f' (robot namespace "{name_prefix}")' if name_prefix else ""
    raise _error("mujocoAccessRegistry", f'{kind} "{target}" was not found{suffix}')


def _copy_array_slice(source, label, start=0, length=None):
    values = _flat(source, label)
    if not _is_integer(start) or start < 0:
        raise _error(
            "mujocoAccessRegistry", f"{label} slice start must be a non-negative integer"
        )
    if length is None:
        length = len(values) - start
    if not _is_integer(length) or length < 0:
        raise _error(
            "mujocoAccessRegistry", f"{label} slice length must be a non-negative integer"
        )
    start, length = int(start), int(length)
    if start + length > len(values):
        raise _error(
            "mujocoAccessRegistry",
            f"{label} slice {start}:{start + length} exceeds length {len(values)}",
        )
    output = values[start : start + length]
    bad = np.flatnonzero(~np.isfinite(output))
    if len(bad):
        raise _error("mujocoAccessRegistry", f"{label}[{int(bad[0])}] must be finite")
    return output.astype(np.float32)


def _next_address(addresses, current, total):
    following = [int(a) for a in addresses if int(a) > current]
    return min(following) if following else total


def _data_field(model, data, field):
    if field in _DERIVED_FIELDS:
        return _derived_body_velocities(model, data)[field]
    try:
        return getattr(data, field)
    except AttributeError:
        raise _error(
            "mujocoAccessRegistry", f"snapshot field {field} is unavailable"
        ) from None


def _derived_body_velocities(model, data):
    """Vectorized ``snapshotBodyVelocities`` (mujocoWorkerSnapshots.ts).

    cvel is the subtree-COM-frame 6-D velocity [rot(3); tran(3)]; the body
    origin velocity adds omega x (xpos - subtree_com of the root body), and
    local variants map the world vectors back through xmat's transpose.
    """
    nbody = int(model.nbody)
    cvel = np.asarray(data.cvel, dtype=np.float64).reshape(nbody, 6)
    xpos = np.asarray(data.xpos, dtype=np.float64).reshape(nbody, 3)
    xmat = np.asarray(data.xmat, dtype=np.float64).reshape(nbody, 3, 3)
    subtree_com = np.asarray(data.subtree_com, dtype=np.float64).reshape(-1, 3)
    roots = np.asarray(model.body_rootid, dtype=np.int64)

    angular = cvel[:, 0:3]
    r = xpos - subtree_com[roots]
    linear = cvel[:, 3:6] + np.cross(angular, r)
    # xmat maps body-local vectors to world vectors; its transpose maps back.
    local_angular = np.matmul(np.transpose(xmat, (0, 2, 1)), angular[:, :, None])[:, :, 0]
    local_linear = np.matmul(np.transpose(xmat, (0, 2, 1)), linear[:, :, None])[:, :, 0]
    return {
        "body_angvel_world": angular,
        "body_linvel_world": linear,
        "body_angvel_local": local_angular,
        "body_linvel_local": local_linear,
    }


def _snapshot_field(context, field):
    snapshot = context.get("snapshot")
    if snapshot is None or field not in snapshot or snapshot[field] is None:
        raise _error(
            "mujocoAccessRegistry",
            f"snapshot field {field} is unavailable (contact aggregation is "
            "maintained by the physics layer and published as a snapshot)",
        )
    return snapshot[field]


def _apply_user_slice(values, config):
    raw_start = config.get("start")
    start = _number(raw_start) if raw_start is not None else 0.0
    raw_length = config.get("length")
    if raw_length is None:
        return _copy_array_slice(values, "MuJoCo query output", start)
    return _copy_array_slice(
        values, "MuJoCo query output", start, _number(raw_length)
    )


def read_mujoco_access_query(context, config):
    """``readMujocoAccessQuery`` against native MjModel/MjData.

    ``context``: ``{"model", "data", "namePrefix"?, "snapshot"?}`` — ``snapshot``
    is the physics layer's mapping of SNAPSHOT_FIELDS to arrays (contact
    forces/timing; T6 publishes it).
    """
    model = context["model"]
    data = context["data"]
    name_prefix = context.get("namePrefix") or ""
    config = config or {}

    query = str(config.get("query") if config.get("query") is not None else "").strip()
    if query in _RAW_QUERIES:
        output = _copy_array_slice(_data_field(model, data, _RAW_QUERIES[query]), query)
    elif query in _ROW_QUERIES:
        kind, field, row_size, count_attr = _ROW_QUERIES[query]
        count = int(getattr(model, count_attr))
        index = _resolve_entity_index(model, kind, count, config.get("target"), name_prefix)
        if field in SNAPSHOT_FIELDS:
            source = _snapshot_field(context, field)
        else:
            source = _data_field(model, data, field)
        output = _copy_array_slice(source, query, index * row_size, row_size)
    elif query in _JOINT_QUERIES:
        joint = _resolve_entity_index(
            model, "joint", int(model.njnt), config.get("target"), name_prefix
        )
        field = _JOINT_QUERIES[query]
        if field == "qpos":
            start = int(model.jnt_qposadr[joint])
            end = _next_address(model.jnt_qposadr, start, int(model.nq))
            output = _copy_array_slice(data.qpos, query, start, end - start)
        else:
            start = int(model.jnt_dofadr[joint])
            end = _next_address(model.jnt_dofadr, start, int(model.nv))
            output = _copy_array_slice(_data_field(model, data, field), query, start, end - start)
    elif query in _SENSOR_QUERIES:
        sensor = _resolve_entity_index(
            model, "sensor", int(model.nsensor), config.get("target"), name_prefix
        )
        start = int(model.sensor_adr[sensor])
        dim = int(model.sensor_dim[sensor])
        output = _copy_array_slice(data.sensordata, query, start, dim)
    elif query in _SCALAR_ENTITY_QUERIES:
        field = _SCALAR_ENTITY_QUERIES[query]
        count = int(model.nu)
        index = _resolve_entity_index(
            model, "actuator", count, config.get("target"), name_prefix
        )
        output = _copy_array_slice(_data_field(model, data, field), query, index, 1)
    else:
        raise _unsupported_query_error(query)

    return _apply_user_slice(output, config)


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
# cameraObservation.ts + cameraObservationPose.ts mirror
# ---------------------------------------------------------------------------

_RENDERERS = {}


def _three_to_mujoco(vector):
    """(x, y, z)_three -> (x, -z, y)_mujoco (three.js y-up vs MuJoCo z-up)."""
    return np.asarray([vector[0], -vector[2], vector[1]], dtype=np.float64)


def _mujoco_to_three(vector):
    """(x, y, z)_mujoco -> (x, z, -y)_three (inverse of ``_three_to_mujoco``)."""
    return np.asarray([vector[0], vector[2], -vector[1]], dtype=np.float64)


def _quat_to_matrix(quat):
    """wxyz quaternion -> row-major 3x3 matrix (mujoco convention)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=np.float64).reshape(-1))
    return rot.reshape(3, 3)


def _yaw_matrix(quat):
    """Yaw-only rotation about the world up axis, like TS 'YXZ euler .y'."""
    qw, qx, qy, qz = (float(value) for value in quat[:4])
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rotation_x(degrees):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)


def _rotation_y(degrees):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)


def _rotation_z(degrees):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)


def _camera_local_rotation_matrix(spec):
    """Local camera rotation in three.js axes: XYZ euler or xyaxes basis."""
    if spec.get("cameraRotationMode") == "xyaxes":
        raw = spec.get("cameraXyaxes")
        if raw is None or raw == "":
            raise _error(
                "CameraObservation",
                "cameraXyaxes is required when cameraRotationMode is xyaxes",
            )
        values = parse_number_list(raw, "cameraXyaxes")
        if not values or len(values) != 6:
            raise _error(
                "CameraObservation",
                "cameraXyaxes must contain 6 numbers (two 3-D axes)",
            )
        x = _mujoco_to_three(values[0:3])
        y = _mujoco_to_three(values[3:6])
        if np.dot(x, x) == 0 or np.dot(y, y) == 0:
            raise _error("CameraObservation", "cameraXyaxes axes must be non-zero")
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        if np.dot(z, z) == 0:
            raise _error("CameraObservation", "cameraXyaxes axes must not be parallel")
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        # Column-basis rotation matrix in three.js axes.
        return np.column_stack([x, y, z])

    euler = spec.get("cameraEulerDeg")
    values = parse_number_list(
        DEFAULT_CAMERA_OBSERVATION["cameraEulerDeg"] if euler is None else euler,
        "cameraEulerDeg",
    )
    if not values or len(values) != 3:
        raise _error("CameraObservation", "cameraEulerDeg must contain 3 numbers")
    ex, ey, ez = values
    # three.js Euler order 'XYZ' composes R = Rx * Ry * Rz.
    return _rotation_x(ex) @ _rotation_y(ey) @ _rotation_z(ez)


def _camera_target(model, data, spec):
    target_kind = "site" if spec.get("targetKind") == "site" else "body"
    target = str(
        spec.get("target") if spec.get("target") is not None else ""
    ).strip() or str(DEFAULT_CAMERA_OBSERVATION["target"])
    count = int(model.nsite if target_kind == "site" else model.nbody)
    if count <= 0:
        raise _error("CameraObservation", f"model has no {target_kind} targets")
    if target.isascii() and target.isdigit():
        index = int(target)
        if not 0 <= index < count:
            raise _error(
                "CameraObservation",
                f"{target_kind} index {index} is out of bounds for count {count}",
            )
    else:
        obj_type = _OBJ_TYPES[target_kind]
        index = -1
        for candidate in range(count):
            if (mujoco.mj_id2name(model, obj_type, candidate) or "") == target:
                index = candidate
                break
        if index < 0:
            sample = [
                mujoco.mj_id2name(model, obj_type, i) or f"#{i}"
                for i in range(min(count, 8))
            ]
            raise _error(
                "CameraObservation",
                f'{target_kind} target "{target}" not found. Available: '
                + ", ".join(sample),
            )

    if target_kind == "body":
        position = np.asarray(data.xpos[index], dtype=np.float64).copy()
        quat = np.asarray(data.xquat[index], dtype=np.float64).copy()
    else:
        position = np.asarray(data.site_xpos[index], dtype=np.float64).copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[index], dtype=np.float64).reshape(-1))
    return position, quat


def _camera_pose(model, data, spec):
    """Camera pos/forward/up in MuJoCo world coordinates.

    Mirrors ``CameraObservationPoseResolver.update``: frame quaternion from
    the target (body or yaw-only), offset in the ray or body frame, local
    rotation from euler/xyaxes in three.js camera axes (-Z forward, +Y up),
    conjugated back into MuJoCo z-up coordinates.
    """
    position, quat = _camera_target(model, data, spec)

    if spec.get("rayAlignment") == "yaw":
        frame_matrix = _yaw_matrix(quat)
    else:
        frame_matrix = _quat_to_matrix(quat)
    target_matrix = _quat_to_matrix(quat)
    offset_frame = target_matrix if spec.get("cameraOffsetFrame") == "body" else frame_matrix

    offset_raw = spec.get("cameraOffset")
    offset_values = parse_number_list(
        DEFAULT_CAMERA_OBSERVATION["cameraOffset"] if offset_raw is None else offset_raw,
        "cameraOffset",
    )
    if not offset_values or len(offset_values) != 3:
        raise _error("CameraObservation", "cameraOffset must contain 3 numbers")
    offset = np.asarray(offset_values, dtype=np.float64)

    local_rotation = _camera_local_rotation_matrix(spec)
    # three.js cameras look along local -Z with +Y up; map both into MuJoCo
    # coordinates and rotate them into the world by the frame quaternion.
    forward = frame_matrix @ _three_to_mujoco(local_rotation @ np.array([0.0, 0.0, -1.0]))
    up = frame_matrix @ _three_to_mujoco(local_rotation @ np.array([0.0, 1.0, 0.0]))
    camera_position = position + offset_frame @ offset
    return camera_position, forward, up


def _create_renderer(model, width, height):
    """Renderer factory hook (tests substitute a fake here)."""
    return mujoco.Renderer(model, height, width)


def _renderer_for(model, width, height):
    key = (id(model), width, height)
    renderer = _RENDERERS.get(key)
    if renderer is None:
        try:
            renderer = _create_renderer(model, width, height)
        except Exception as error:
            raise RuntimeError(
                "[CameraObservation] MuJoCo GL renderer is unavailable on this "
                f"machine ({error}). Camera observations need an offscreen "
                "OpenGL context: set MUJOCO_GL=egl (or osmesa on headless "
                "machines) and retry; the local runtime fails loudly instead "
                "of returning empty frames."
            ) from error
        _RENDERERS[key] = renderer
    return renderer


# Per-node depth frame index (TS depthFrameIndexes): the noise stream is
# decorrelated per rendered frame and re-anchored on reset.
_camera_frame_counters = {}


def reset_camera_observation_state(node_id=None):
    """``resetOutput`` counterpart: drop one node's depth frame index or all."""
    if node_id is None:
        _camera_frame_counters.clear()
        return
    _camera_frame_counters.pop(node_id, None)


def dispose_camera_renderers():
    """Close cached renderers (owned by this module's cache only)."""
    errors = []
    while _RENDERERS:
        _, renderer = _RENDERERS.popitem()
        try:
            renderer.close()
        except Exception as error:  # pragma: no cover - defensive
            errors.append(error)
    reset_camera_observation_state()
    if errors:
        raise RuntimeError(
            "[CameraObservation] renderer cleanup failed: "
            + "; ".join(str(error) for error in errors)
        )


# MjvOption cache keyed by the selected geom-group tuple (render_scene passes
# raycastGeomGroups as the scene's geom-group visibility mask).
_SCENE_OPTIONS = {}


def _camera_scene_option(geom_groups):
    key = tuple(geom_groups)
    option = _SCENE_OPTIONS.get(key)
    if option is None:
        option = mujoco.MjvOption()
        if geom_groups:
            selected = set(geom_groups)
            # Groups beyond MuJoCo's 6 slots never enable, exactly like the
            # TS includes() filter that can never match them either.
            option.geomgroup[:] = [
                1 if group in selected else 0 for group in range(6)
            ]
        _SCENE_OPTIONS[key] = option
    return option


def _camera_render(renderer, model, data, config):
    """Render the full sensor, with an optional independent horizontal FoV.

    A zero frustum_width preserves MuJoCo's viewport-derived horizontal view.
    Otherwise MuJoCo uses this field as the frustum *halfwidth*. Cropping is
    applied to the full frame later, matching the browser's setViewOffset.
    """
    position, forward, up = _camera_pose(model, data, config)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # This mujoco build has no MjvCamera.fovy; the free-camera vertical fov
    # lives on the model's visual globals, so borrow it for the update and
    # restore it immediately (the frustum is already baked into scene.camera).
    previous_fovy = float(model.vis.global_.fovy)
    model.vis.global_.fovy = config["fov"]
    try:
        renderer.update_scene(
            data,
            camera=camera,
            scene_option=_camera_scene_option(config["raycastGeomGroups"]),
        )
    finally:
        model.vis.global_.fovy = previous_fovy

    half_height = config["near"] * math.tan(math.radians(config["fov"]) / 2)
    horizontal_fov = config["horizontalFov"]
    half_width = (0.0 if horizontal_fov is None else
                  config["near"] * math.tan(math.radians(horizontal_fov) / 2))
    for side in (0, 1):
        view = renderer.scene.camera[side]
        view.pos[:] = position
        view.forward[:] = forward
        view.up[:] = up
        view.frustum_near = config["near"]
        view.frustum_far = config["far"]
        view.frustum_bottom = -half_height
        view.frustum_top = half_height
        view.frustum_center = 0.0
        # 0 lets mjr_render derive the halfwidth from the viewport aspect
        # (the full-sensor aspect); a nonzero value pins it instead.
        view.frustum_width = half_width
        view.orthographic = 0


def _render_depth_meters(renderer, model, config):
    """Full-sensor depth in meters (view-Z, like the TS linear readback).

    ``mujoco.Renderer`` undoes the GL projection using
    ``model.vis.map.znear/zfar * stat.extent`` while the scene camera renders
    with the config's near/far, so the model maps are borrowed for the read
    to keep the linearization exact (restored immediately).
    """
    extent = float(model.stat.extent)
    previous = (float(model.vis.map.znear), float(model.vis.map.zfar))
    model.vis.map.znear = config["near"] / extent
    model.vis.map.zfar = config["far"] / extent
    renderer.enable_depth_rendering()
    try:
        depth = np.asarray(renderer.render(), dtype=np.float32)
    finally:
        model.vis.map.znear, model.vis.map.zfar = previous
        renderer.disable_depth_rendering()
    if depth.shape != (config["fullHeight"], config["fullWidth"]):
        raise _error(
            "CameraObservation",
            f"renderer produced depth {depth.shape[1]}x{depth.shape[0]}, "
            f'expected {config["fullWidth"]}x{config["fullHeight"]}',
        )
    return depth


# ---------------------------------------------------------------------------
# cameraObservationRuntime.ts + cameraDepthCorruption.ts depth post-processing
# ---------------------------------------------------------------------------

_GAUSSIAN_KERNELS = {}


def _gaussian_kernel(kernel_size, sigma):
    """1-D gaussian weights exp(-k^2 / 2 sigma^2) and their sum."""
    key = (kernel_size, sigma)
    cached = _GAUSSIAN_KERNELS.get(key)
    if cached is None:
        offsets = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float64)
        weights = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
        cached = (weights, float(weights.sum()))
        _GAUSSIAN_KERNELS[key] = cached
    return cached


def _apply_gaussian_blur(frame, kernel_size, sigma):
    """``applyGaussianBlur``: separable 'same' convolution, edge replication.

    Horizontal pass into a scratch buffer, then vertical pass back, each
    normalized by the kernel weight sum — the exact TS pass order (scale and
    clip already happened; the caller applies them first).
    """
    weights, weight_sum = _gaussian_kernel(kernel_size, sigma)
    radius = kernel_size // 2
    height, width = frame.shape

    padded = np.pad(frame.astype(np.float64), ((0, 0), (radius, radius)), mode="edge")
    accumulator = np.zeros((height, width), dtype=np.float64)
    for k in range(kernel_size):
        accumulator += weights[k] * padded[:, k:k + width]
    scratch = (accumulator / weight_sum).astype(np.float32)

    padded = np.pad(scratch.astype(np.float64), ((radius, radius), (0, 0)), mode="edge")
    accumulator = np.zeros((height, width), dtype=np.float64)
    for k in range(kernel_size):
        accumulator += weights[k] * padded[k:k + height, :]
    frame[:] = (accumulator / weight_sum).astype(np.float32)


def _fnv1a_hash(text):
    """``hashString``: FNV-1a over UTF-16 code units."""
    value = 2166136261
    for char in str(text):
        value = ((value ^ ord(char)) * 16777619) & 0xFFFFFFFF
    return value


def _imul32(a, b):
    """``Math.imul``: 32-bit wrap-around multiply."""
    return (int(a) * int(b)) & 0xFFFFFFFF


def _mulberry32_uniforms(state, count):
    """The first ``count`` uniforms of ``seededRandom``'s mulberry32 stream.

    mulberry32 advances its state linearly (``state += 0x6d2b79f5``), so the
    whole draw sequence is the arithmetic progression of states scrambled in
    parallel — the same values the TS closure yields, consumed in order.
    """
    steps = np.arange(1, count + 1, dtype=np.uint64)
    value = (np.uint64(state & 0xFFFFFFFF) + steps * np.uint64(0x6D2B79F5)).astype(
        np.uint32
    )
    value = (value ^ (value >> np.uint32(15))) * (value | np.uint32(1))
    value = value ^ (value + (value ^ (value >> np.uint32(7))) * (value | np.uint32(61)))
    return (value ^ (value >> np.uint32(14))).astype(np.float64) / 4294967296.0


def _camera_depth_corruption_enabled(config):
    """``cameraDepthCorruptionEnabled``."""
    return (
        config["depthNoiseStd"] > 0
        or config["depthNoiseDistanceStd"] > 0
        or config["depthDropoutProbability"] > 0
        or config["depthDropoutPatchCount"] > 0
    )


def _corrupt_camera_depth(config, node_id, frame_index, frame):
    """``corruptCameraDepth`` on the cropped, scaled, blurred depth frame.

    The uniform stream is seeded ``seed ^ FNV1a(nodeId) ^ imul(frameIndex+1,
    0x9e3779b1)`` and consumed exactly like the TS closure: gaussian noise
    (two uniforms per active pixel, row-major), pixel dropout (one per pixel),
    then patch dropout (two per patch). Transcendental results may differ
    from the JS engine in the last ulp; the stream and ordering do not.
    """
    width, height = config["width"], config["height"]
    total = width * height
    flat = frame.reshape(-1)
    state = (
        int(config["depthNoiseSeed"])
        ^ _fnv1a_hash(node_id)
        ^ _imul32(frame_index + 1, 0x9E3779B1)
    ) & 0xFFFFFFFF
    miss_depth = min(max(config["far"] / config["depthScale"], 0.0), 1.0)
    offset = 0

    noise_std = config["depthNoiseStd"]
    distance_std = config["depthNoiseDistanceStd"]
    if noise_std > 0 or distance_std > 0:
        std = noise_std + flat.astype(np.float64) * distance_std
        active = np.flatnonzero(std > 0)
        if len(active):
            uniforms = _mulberry32_uniforms(state, offset + 2 * len(active))[offset:]
            offset += 2 * len(active)
            u1 = np.maximum(uniforms[0::2], 2.0**-52)
            u2 = uniforms[1::2]
            gaussian = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            flat[active] = np.clip(
                flat[active].astype(np.float64) + gaussian * std[active], 0.0, 1.0
            ).astype(np.float32)

    probability = config["depthDropoutProbability"]
    if probability > 0:
        uniforms = _mulberry32_uniforms(state, offset + total)[offset:]
        flat[uniforms < probability] = np.float32(miss_depth)

    patch_count = config["depthDropoutPatchCount"]
    radius = config["depthDropoutPatchRadius"]
    if patch_count > 0:
        uniforms = _mulberry32_uniforms(state, offset + 2 * patch_count)[offset:]
        radius_sq = radius * radius
        for patch in range(patch_count):
            center_x = int(math.floor(uniforms[2 * patch] * width))
            center_y = int(math.floor(uniforms[2 * patch + 1] * height))
            for y in range(max(center_y - radius, 0), min(center_y + radius, height - 1) + 1):
                for x in range(max(center_x - radius, 0), min(center_x + radius, width - 1) + 1):
                    dx = x - center_x
                    dy = y - center_y
                    if dx * dx + dy * dy <= radius_sq:
                        frame[y, x] = np.float32(miss_depth)


def camera_frames(model, data, spec):
    """Render one camera observation into flat float32 frame list.

    Mirrors the browser pipeline end to end: the sensor's full pinhole
    resolution (``fullWidth`` x ``fullHeight``, vertical ``fov``) is rendered
    first, the ``cropX``/``cropY``/``width``/``height`` window is cropped out
    (TS does the same through ``setViewOffset`` while rendering at the crop
    size), and the depth channel then runs the TS post-processing chain in
    order: ``clamp01(meters / depthScale)`` -> separable gaussian blur ->
    seeded corruption (noise / pixel dropout / patch dropout). RGB is the
    cropped full-resolution render scaled to [0, 1]. Channel order follows
    ``cameraOutputLabels``: ``rgb`` -> [H*W*3], ``depth`` -> [H*W] (+ a zero
    age channel when ``includeAgeSteps`` — native rendering is synchronous so
    the frame age is always 0 steps), ``rgbd`` -> [rgb, depth].

    The noise stream keys on the node id (``fill_observation`` injects it) and
    a per-node frame counter that ``ObservationProcessor.reset`` /
    ``reset_camera_observation_state`` re-anchor, so replays with the same
    seed reproduce the same frames.

    Deliberate differences from the TS readback path: depth comes from the
    native float32 depth buffer instead of the packed-rgba8 24-bit encoding
    (no readback quantization), and the async readback cadence fields
    (``readbackMode``/``*ReadbackMs``/``readbackPipelineDepth``) are parsed
    and validated but inert — rendering is synchronous. ``excludeGeoms`` /
    ``excludeBodies`` fail explicitly instead of filtering the scene.
    """
    spec = dict(spec or {})
    config = read_camera_config(spec)
    node_id = str(spec.get("nodeId") if spec.get("nodeId") is not None else "")
    mode = config["outputMode"]
    width, height = config["width"], config["height"]
    crop_x, crop_y = config["cropX"], config["cropY"]
    if config["excludeGeoms"] or config["excludeBodies"]:
        raise _error(
            "CameraObservation",
            "excludeGeoms/excludeBodies are not ported to the native camera "
            "render; move the geometry out of the sensor frustum or its geom "
            "group instead",
        )

    renderer = _renderer_for(model, config["fullWidth"], config["fullHeight"])
    _camera_render(renderer, model, data, config)

    frames = []
    if mode in ("rgb", "rgbd"):
        rgb_full = np.asarray(renderer.render(), dtype=np.uint8)
        if rgb_full.shape != (config["fullHeight"], config["fullWidth"], 3):
            raise _error(
                "CameraObservation",
                f"renderer produced RGB {rgb_full.shape[1]}x{rgb_full.shape[0]}, "
                f'expected {config["fullWidth"]}x{config["fullHeight"]}',
            )
        rgb = rgb_full[crop_y : crop_y + height, crop_x : crop_x + width, :]
        frames.append(
            (rgb.astype(np.float64) * (1.0 / 255.0)).astype(np.float32).reshape(-1)
        )
    if mode in ("depth", "rgbd"):
        frame_index = _camera_frame_counters.get(node_id, 0)
        _camera_frame_counters[node_id] = frame_index + 1
        depth_full = _render_depth_meters(renderer, model, config)
        depth = depth_full[crop_y : crop_y + height, crop_x : crop_x + width]
        depth = np.clip(
            depth.astype(np.float64) * (1.0 / config["depthScale"]), 0.0, 1.0
        ).astype(np.float32)
        if config["depthGaussianKernelSize"] > 1:
            _apply_gaussian_blur(
                depth, config["depthGaussianKernelSize"], config["depthGaussianSigma"]
            )
        if _camera_depth_corruption_enabled(config):
            _corrupt_camera_depth(config, node_id, frame_index, depth)
        frames.append(depth.reshape(-1))
    if spec.get("includeAgeSteps") is True:
        frames.append(np.zeros(1, dtype=np.float32))
    return frames


# ---------------------------------------------------------------------------
# mujocoPolicyController.ts step-loop mirror
# ---------------------------------------------------------------------------


def _finite_copy(values, label):
    values = _flat(values, label)
    bad = np.flatnonzero(~np.isfinite(values))
    if len(bad):
        raise _error("botlab", f"{label}[{int(bad[0])}] must be finite")
    return values.astype(np.float32)


def _positive_integer(value, label):
    number = _number(value)
    if not _is_integer(number) or number <= 0:
        raise _error("botlab", f"{label} must be a positive integer")
    return int(number)


def _write_command_parameters(node_data, context):
    raw_names = node_data.get("parameterNames")
    if not isinstance(raw_names, (list, tuple)) or any(
        not isinstance(item, str) for item in raw_names
    ):
        raise _error("botlab", "obs-commands parameterNames must be a string list")
    names = list(raw_names)
    # 常量轴（空串占位）：值取 cmdMin 的恒等对，不查参数。
    cmd_min = None
    if any(name == "" for name in names):
        raw_min = node_data.get("cmdMin")
        if not isinstance(raw_min, (list, tuple)) or len(raw_min) != len(names):
            raise _error(
                "botlab",
                "obs-commands constant axes require a cmdMin vector matching dim",
            )
        cmd_min = [float(item) for item in raw_min]
    parameters = context.get("parameters")
    if not callable(parameters):
        raise _error(
            "botlab",
            "obs-commands requires a parameters callable (name -> float) in "
            "the observation context",
        )
    values = node_data.get("values")
    if not (
        isinstance(values, np.ndarray)
        and values.dtype == np.float32
        and len(values) == len(names)
    ):
        values = np.zeros(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        values[index] = cmd_min[index] if name == "" else parameters(name)
    node_data["values"] = values


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
