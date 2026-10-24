"""Camera pose, rendering and depth post-processing (cameraObservation*.ts mirror)."""

import math

import mujoco
import numpy as np

from ._camera_config import DEFAULT_CAMERA_OBSERVATION, read_camera_config
from ._queries import _OBJ_TYPES
from ._values import _error, parse_number_list

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
    """Update the renderer scene for one parsed camera config.

    The frustum mirrors ``CameraObservationPoseResolver.update``:
    ``fov`` is the vertical FoV of the FULL sensor and the horizontal extent
    follows the full-sensor aspect (``camera.aspect = fullWidth/fullHeight``
    in TS). MuJoCo's ``setView`` derives the frustum halfwidth from the
    viewport aspect when ``frustum_width`` is 0 — the renderer viewport IS
    fullWidth x fullHeight, so leaving it 0 reproduces the TS projection; the
    crop window is taken from the rendered full frame afterwards (TS applies
    it via ``camera.setViewOffset`` before rendering at width x height).
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
        view.frustum_width = 0.0
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
