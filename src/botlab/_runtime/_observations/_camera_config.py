"""Camera observation defaults and obs-camera config parsing (cameraObservationConfig.ts mirror)."""

import math

from ._values import _error, _is_integer, _number, parse_number_list

CAMERA_OBSERVATION_NODE_TYPE = "obs-camera"


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
