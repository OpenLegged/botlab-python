"""Motion asset loaders for the local Python runtime.

Semantic mirror of src/domain/controls/motionReference.ts, which stays the
single source of truth: ``decode_bin`` follows ``readBinArray`` byte for byte,
``load_motion_folder`` follows the inline folder assembly (seven required
.bin files, dtype/shape consistency, finite-data checks), and
``load_motion_npz`` follows ``loadNpzMotion`` including the per-slot name
fallback chains — numpy parses the zip members and their .npy payloads
natively. The browser lease cache is deliberately not ported: it is a
page-lifetime concern owned by the TS host.

Output dicts follow the ``loadPythonMotion`` contract in
src/domain/python/botlabBridge.ts: snake_case keys with flat float32 numpy
buffers plus optional ``joint_names``/``body_names`` lists when the motion
format declares them (NPZ arrays or a .bin folder's optional metadata.json).
"""

import io
import json
import struct
import zipfile
from collections import namedtuple

import numpy as np

MOTION_REQUIRED_FILES = (
    "body_pos_w.bin",
    "body_quat_w.bin",
    "body_lin_vel_w.bin",
    "body_ang_vel_w.bin",
    "joint_pos.bin",
    "joint_vel.bin",
    "fps.bin",
)
MOTION_METADATA_FILE = "metadata.json"

# (file name, output key, finite-check label) in the TS copy/finite-check order.
_FOLDER_FLOAT_FILES = (
    ("body_pos_w.bin", "body_position", "body_pos_w"),
    ("body_quat_w.bin", "body_quaternion", "body_quat_w"),
    ("joint_pos.bin", "joint_position", "joint_pos"),
    ("joint_vel.bin", "joint_velocity", "joint_vel"),
    ("body_lin_vel_w.bin", "body_linear_velocity", "body_lin_vel_w"),
    ("body_ang_vel_w.bin", "body_angular_velocity", "body_ang_vel_w"),
)

_BIN_MAGIC = b"NPZ\x00"

#: Decoded .bin array: dtype 'f' or 'l', the shape tuple, and a flat
#: read-only numpy view over the payload bytes.
BinArray = namedtuple("BinArray", ("dtype", "shape", "data"))

#: One parsed NPZ member: float32 flat buffer (numeric) or name list (unicode).
_NpyValue = namedtuple("_NpyValue", ("shape", "data"))


def _product(values):
    total = 1
    for value in values:
        total *= int(value)
    return total


def _shape_text(shape):
    return "x".join(str(int(dim)) for dim in shape)


def _assert_shape(actual, expected, name):
    if len(actual) != len(expected) or any(a != e for a, e in zip(actual, expected)):
        raise ValueError(
            f"[motionReference] {name} shape {_shape_text(actual)} "
            f"does not match {_shape_text(expected)}"
        )


def _assert_finite(values, name):
    finite = np.isfinite(values)
    if not finite.all():
        index = int(np.flatnonzero(~finite)[0])
        raise ValueError(f"[motionReference] {name} contains a non-finite value at {index}")


def decode_bin(buffer, source):
    """Decode one little-endian .bin motion array, mirroring readBinArray.

    The layout is: 'NPZ\\0' magic, uint32 ndims (1-4), uint32 shape dims
    (each positive), uint32 dtype size, one dtype char ('f' float32 or
    'l' int64) in a 4-byte slot, then the row-major payload. Trailing bytes
    after the payload are tolerated, exactly like the TS reader.
    """
    data = bytes(buffer)
    if len(data) < 20:
        raise ValueError(f"[motionReference] {source} is too small to contain a motion array")
    if data[:4] != _BIN_MAGIC:
        raise ValueError(f"[motionReference] {source} has invalid NPZ magic")
    ndims = struct.unpack_from("<I", data, 4)[0]
    if ndims < 1 or ndims > 4:
        raise ValueError(f"[motionReference] {source} has invalid dimension count {ndims}")
    if 8 + ndims * 4 + 8 > len(data):
        raise ValueError(f"[motionReference] {source} data payload is truncated")
    shape = []
    for index in range(ndims):
        dim = struct.unpack_from("<I", data, 8 + index * 4)[0]
        if dim <= 0:
            raise ValueError(f"[motionReference] {source} has invalid shape dimension {dim}")
        shape.append(dim)
    dtype_offset = 8 + ndims * 4
    dtype_size = struct.unpack_from("<I", data, dtype_offset)[0]
    dtype = chr(data[dtype_offset + 4])
    data_offset = dtype_offset + 8
    total = _product(shape)
    if data_offset + total * dtype_size > len(data):
        raise ValueError(f"[motionReference] {source} data payload is truncated")
    if dtype == "f" and dtype_size == 4:
        values = np.frombuffer(data, dtype="<f4", count=total, offset=data_offset)
        return BinArray("f", tuple(shape), values)
    if dtype == "l" and dtype_size == 8:
        values = np.frombuffer(data, dtype="<i8", count=total, offset=data_offset)
        return BinArray("l", tuple(shape), values)
    raise ValueError(f"[motionReference] {source} has unsupported dtype {dtype}{dtype_size}")


def load_motion_folder(files, source="inline:motion"):
    """Assemble the seven required .bin files into the motion data dict.

    Mirrors the inline branch of loadMotionData in motionReference.ts: the
    six channel buffers must be float ('f') .bin arrays with consistent
    frame/body/dof shapes, fps must be a single int64 ('l') value, and every
    buffer value must be finite. Optional metadata.json supplies explicit
    joint_names/body_names without changing legacy seven-file loading.
    """
    arrays = {}
    for file_name in MOTION_REQUIRED_FILES:
        buffer = files.get(file_name)
        if buffer is None:
            raise ValueError(f"[motionReference] {source} is missing required file {file_name}")
        arrays[file_name] = decode_bin(buffer, f"{source}:{file_name}")

    if (
        any(arrays[file_name].dtype != "f" for file_name, _, _ in _FOLDER_FLOAT_FILES)
        or arrays["fps.bin"].dtype != "l"
    ):
        raise ValueError("[motionReference] motion folder has missing or invalid required files")

    joint_pos = arrays["joint_pos.bin"]
    body_pos = arrays["body_pos_w.bin"]
    if len(joint_pos.shape) < 2 or len(body_pos.shape) < 2:
        raise ValueError("[motionReference] motion folder has invalid joint or body dimensions")
    frame_count = int(joint_pos.shape[0])
    dof_count = int(joint_pos.shape[1])
    body_count = int(body_pos.shape[1])

    _assert_shape(arrays["joint_vel.bin"].shape, (frame_count, dof_count), "joint_vel")
    _assert_shape(arrays["body_pos_w.bin"].shape, (frame_count, body_count, 3), "body_pos_w")
    _assert_shape(arrays["body_quat_w.bin"].shape, (frame_count, body_count, 4), "body_quat_w")
    _assert_shape(arrays["body_lin_vel_w.bin"].shape, (frame_count, body_count, 3), "body_lin_vel_w")
    _assert_shape(arrays["body_ang_vel_w.bin"].shape, (frame_count, body_count, 3), "body_ang_vel_w")
    _assert_shape(arrays["fps.bin"].shape, (1,), "fps")

    fps_value = float(arrays["fps.bin"].data[0])
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(f"[motionReference] fps must be positive, got {fps_value}")

    buffers = {
        output_key: arrays[file_name].data.astype(np.float32, copy=True).ravel()
        for file_name, output_key, _ in _FOLDER_FLOAT_FILES
    }
    for _, output_key, label in _FOLDER_FLOAT_FILES:
        _assert_finite(buffers[output_key], label)

    result = {
        "frame_count": frame_count,
        "body_count": body_count,
        "dof_count": dof_count,
        "fps": fps_value,
        **buffers,
    }
    if MOTION_METADATA_FILE in files:
        metadata = json.loads(bytes(files[MOTION_METADATA_FILE]).decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("[motionReference] metadata.json must be an object")
        for key, count in (("body_names", body_count), ("joint_names", dof_count)):
            if key not in metadata:
                continue
            names = metadata[key]
            if (
                not isinstance(names, list)
                or len(names) != count
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    f"[motionReference] metadata {key} must contain {count} unique non-empty names"
                )
            result[key] = names
    return result


def _read_npy(payload, source):
    """Parse and validate one .npy payload, mirroring readNpy constraints."""
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, EOFError, struct.error) as error:
        raise ValueError(f"[motionReference] {source} is not an NPY array") from error
    shape = tuple(int(dim) for dim in array.shape)
    dtype = array.dtype
    if dtype.byteorder == ">":
        raise ValueError(f"[motionReference] {source} must use little-endian NPY data")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"[motionReference] {source} must use a C-order NPY dtype")
    if dtype.kind in "fiu" and dtype.itemsize in (4, 8):
        wide = array.astype(np.float64)
        finite = np.isfinite(wide)
        if not finite.all():
            index = int(np.flatnonzero(~finite)[0])
            raise ValueError(f"[motionReference] {source} contains a non-finite value at {index}")
        values = array.astype(np.float32)
        representable = np.isfinite(values)
        if not representable.all():
            index = int(np.flatnonzero(~representable)[0])
            raise ValueError(
                f"[motionReference] {source} value at {index} cannot be represented as float32"
            )
        return _NpyValue(shape, values.ravel())
    if dtype.kind == "U" and len(shape) == 1:
        names = [str(value) for value in array]
        if "" in names:
            raise ValueError(f"[motionReference] {source} contains an empty name")
        return _NpyValue(shape, names)
    raise ValueError(f"[motionReference] {source} has unsupported NPY dtype {dtype.str}")


def _npz_slot(arrays, names, label):
    found = [name for name in names if name in arrays]
    if len(found) != 1:
        raise ValueError(
            f"[motionReference] NPZ must contain exactly one {label} array "
            f"({' or '.join(names)})"
        )
    return arrays[found[0]]


def _npy_floats(value, name):
    if not isinstance(value.data, np.ndarray):
        raise ValueError(f"[motionReference] {name} must be numeric")
    return value.data


def _npy_names(value, expected, name):
    if len(value.shape) != 1 or value.shape[0] != expected or isinstance(value.data, np.ndarray):
        raise ValueError(f"[motionReference] {name} must contain {expected} names")
    if len(set(value.data)) != len(value.data):
        raise ValueError(f"[motionReference] {name} must not contain duplicate names")
    return list(value.data)


def load_motion_npz(buffer, source="inline:motion.npz"):
    """Load an NPZ motion archive into the motion data dict.

    Mirrors loadNpzMotion in motionReference.ts: every .npy member is keyed
    by its base name, each slot accepts exactly one of its primary or legacy
    fallback names, shapes must stay consistent, fps must be a positive
    scalar, and optional 'dof_names'/'body_names' unicode arrays provide the
    joint/body name lists.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(bytes(buffer)))
    except zipfile.BadZipFile as error:
        raise ValueError(f"[motionReference] {source} is not a valid NPZ archive") from error
    arrays = {}
    with archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            key = member[:-4].split("/")[-1]
            if key in arrays:
                raise ValueError(f"[motionReference] {source} has duplicate NPZ key {key}")
            arrays[key] = _read_npy(archive.read(member), f"{source}:{key}")

    joint_pos = _npz_slot(arrays, ("joint_pos", "dof_positions"), "joint position")
    joint_vel = _npz_slot(arrays, ("joint_vel", "dof_velocities"), "joint velocity")
    body_pos = _npz_slot(arrays, ("body_pos_w", "body_positions"), "body position")
    body_quat = _npz_slot(arrays, ("body_quat_w", "body_rotations"), "body quaternion")
    body_lin_vel = _npz_slot(arrays, ("body_lin_vel_w", "body_linear_velocities"), "body linear velocity")
    body_ang_vel = _npz_slot(arrays, ("body_ang_vel_w", "body_angular_velocities"), "body angular velocity")
    fps = _npz_slot(arrays, ("fps",), "fps")

    if len(joint_pos.shape) < 2 or len(body_pos.shape) < 2:
        raise ValueError(f"[motionReference] {source} has invalid motion dimensions")
    frame_count = int(joint_pos.shape[0])
    dof_count = int(joint_pos.shape[1])
    body_count = int(body_pos.shape[1])

    _assert_shape(joint_pos.shape, (frame_count, dof_count), "joint position")
    _assert_shape(joint_vel.shape, (frame_count, dof_count), "joint velocity")
    _assert_shape(body_pos.shape, (frame_count, body_count, 3), "body position")
    _assert_shape(body_quat.shape, (frame_count, body_count, 4), "body quaternion")
    _assert_shape(body_lin_vel.shape, (frame_count, body_count, 3), "body linear velocity")
    _assert_shape(body_ang_vel.shape, (frame_count, body_count, 3), "body angular velocity")
    if len(fps.shape) > 1 or _product(fps.shape) != 1:
        raise ValueError("[motionReference] fps must be a scalar")
    fps_value = float(_npy_floats(fps, "fps")[0])
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(f"[motionReference] fps must be positive, got {fps_value}")

    motion = {
        "frame_count": frame_count,
        "body_count": body_count,
        "dof_count": dof_count,
        "fps": fps_value,
    }
    if "dof_names" in arrays:
        motion["joint_names"] = _npy_names(arrays["dof_names"], dof_count, "dof_names")
    if "body_names" in arrays:
        motion["body_names"] = _npy_names(arrays["body_names"], body_count, "body_names")
    motion.update({
        "body_position": _npy_floats(body_pos, "body position"),
        "body_quaternion": _npy_floats(body_quat, "body quaternion"),
        "joint_position": _npy_floats(joint_pos, "joint position"),
        "joint_velocity": _npy_floats(joint_vel, "joint velocity"),
        "body_linear_velocity": _npy_floats(body_lin_vel, "body linear velocity"),
        "body_angular_velocity": _npy_floats(body_ang_vel, "body angular velocity"),
    })
    return motion
