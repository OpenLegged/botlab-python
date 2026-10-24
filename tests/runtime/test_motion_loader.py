"""Motion loader tests: .bin decoding, folder assembly, and the NPZ variant.

Byte fixtures mirror the motionBin builder from
src/domain/runtime/__tests__/processorGoldenExport.test.ts so the Python
loaders are checked against the same bytes the TS golden exporter writes;
the output dict follows the loadPythonMotion contract in
src/domain/python/botlabBridge.ts (snake_case keys, flat float32 buffers,
optional joint_names/body_names lists).
"""

import io
import json
import math
import struct
import unittest

import numpy as np  # noqa: E402

from botlab._runtime import _motion  # noqa: E402


def motion_bin(shape, dtype, values=()):
    """Python twin of the TS motionBin builder (processorGoldenExport.test.ts).

    'NPZ\\0' magic, uint32 ndims, uint32 shape dims, uint32 element size,
    dtype char in a 4-byte slot, then the little-endian payload; missing
    values default to 0 for 'f' and 2 for 'l', like the TS builder.
    """
    element_size = 4 if dtype == "f" else 8
    dtype_offset = 8 + len(shape) * 4
    data_offset = dtype_offset + 8
    count = 1
    for dim in shape:
        count *= dim
    parts = [b"NPZ\x00", struct.pack("<I", len(shape))]
    parts.extend(struct.pack("<I", dim) for dim in shape)
    parts.append(struct.pack("<I", element_size))
    parts.append(dtype.encode("ascii") + b"\x00\x00\x00")
    for index in range(count):
        default = 0 if dtype == "f" else 2
        value = values[index] if index < len(values) else default
        if dtype == "f":
            parts.append(struct.pack("<f", value))
        else:
            parts.append(struct.pack("<q", value))
    assert data_offset + count * element_size == len(b"".join(parts))
    return b"".join(parts)


# 29-DOF / 1-body / 5-frame motion satisfying writeMotionReference's contract
# (processorGoldenExport.test.ts lines 84-143).
FRAMES = 5
DOF = 29


def _body_pos_value(index):
    component = index % 3
    frame = index // 3
    return frame * 0.1 if component == 0 else (0.2 if component == 1 else 0.3)


def _body_quat_value(index):
    component = index % 4
    frame = index // 4
    if component == 0:
        return math.cos(frame * math.pi / 8)
    return math.sin(frame * math.pi / 8) if component == 3 else 0.0


def _body_lin_vel_value(index):
    component = index % 3
    frame = index // 3
    return frame * 0.05 if component == 0 else (0.1 if component == 1 else 0.2)


def _body_ang_vel_value(index):
    component = index % 3
    frame = index // 3
    if component == 0:
        return 0.01
    return -0.02 if component == 1 else frame * 0.03


def _joint_pos_value(index):
    return (index // DOF) + (index % DOF) * 0.01


def _joint_vel_value(index):
    return ((index // DOF) + 1) * 0.1 + (index % DOF) * 0.001


def golden_motion_files():
    """The 29-DOF / 5-frame / 1-body golden motion folder (all 7 files)."""
    return {
        "body_pos_w.bin": motion_bin(
            (FRAMES, 1, 3), "f", [_body_pos_value(i) for i in range(FRAMES * 3)]),
        "body_quat_w.bin": motion_bin(
            (FRAMES, 1, 4), "f", [_body_quat_value(i) for i in range(FRAMES * 4)]),
        "body_lin_vel_w.bin": motion_bin(
            (FRAMES, 1, 3), "f", [_body_lin_vel_value(i) for i in range(FRAMES * 3)]),
        "body_ang_vel_w.bin": motion_bin(
            (FRAMES, 1, 3), "f", [_body_ang_vel_value(i) for i in range(FRAMES * 3)]),
        "joint_pos.bin": motion_bin(
            (FRAMES, DOF), "f", [_joint_pos_value(i) for i in range(FRAMES * DOF)]),
        "joint_vel.bin": motion_bin(
            (FRAMES, DOF), "f", [_joint_vel_value(i) for i in range(FRAMES * DOF)]),
        "fps.bin": motion_bin((1,), "l", [50]),
    }


def golden_expected_buffers():
    return {
        "body_position": np.float32([_body_pos_value(i) for i in range(FRAMES * 3)]),
        "body_quaternion": np.float32([_body_quat_value(i) for i in range(FRAMES * 4)]),
        "body_linear_velocity": np.float32([_body_lin_vel_value(i) for i in range(FRAMES * 3)]),
        "body_angular_velocity": np.float32([_body_ang_vel_value(i) for i in range(FRAMES * 3)]),
        "joint_position": np.float32([_joint_pos_value(i) for i in range(FRAMES * DOF)]),
        "joint_velocity": np.float32([_joint_vel_value(i) for i in range(FRAMES * DOF)]),
    }


BASE_KEYS = {
    "frame_count",
    "body_count",
    "dof_count",
    "fps",
    "body_position",
    "body_quaternion",
    "joint_position",
    "joint_velocity",
    "body_linear_velocity",
    "body_angular_velocity",
}


class DecodeBinTest(unittest.TestCase):
    def test_decodes_float_array(self):
        payload = motion_bin((2, 3), "f", [1.5, -2.25, 3.0, 4.0, 5.0, 6.5])
        result = _motion.decode_bin(payload, "motion/body_pos.bin")
        self.assertEqual(result.dtype, "f")
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            result.data, np.float32([1.5, -2.25, 3.0, 4.0, 5.0, 6.5])
        )

    def test_decodes_int64_array(self):
        payload = motion_bin((1,), "l", [50])
        result = _motion.decode_bin(payload, "motion/fps.bin")
        self.assertEqual(result.dtype, "l")
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.data.dtype, np.dtype("<i8"))
        np.testing.assert_array_equal(result.data, np.array([50], dtype=np.int64))

    def test_pads_missing_values_like_ts_builder(self):
        payload = motion_bin((3,), "f", [1.0])
        result = _motion.decode_bin(payload, "motion/joint.bin")
        np.testing.assert_array_equal(result.data, np.float32([1.0, 0.0, 0.0]))

    def test_accepts_trailing_bytes(self):
        payload = motion_bin((2,), "f", [1.0, 2.0]) + b"\x00\x00\x00"
        result = _motion.decode_bin(payload, "motion/joint.bin")
        np.testing.assert_array_equal(result.data, np.float32([1.0, 2.0]))

    def test_rejects_small_buffer(self):
        with self.assertRaises(ValueError) as ctx:
            _motion.decode_bin(b"NPZ\x00" + b"\x00" * 8, "motion/x.bin")
        self.assertIn("is too small to contain a motion array", str(ctx.exception))

    def test_rejects_bad_magic(self):
        payload = bytearray(motion_bin((1,), "f", [1.0]))
        payload[0] = 0x4D
        with self.assertRaises(ValueError) as ctx:
            _motion.decode_bin(bytes(payload), "motion/x.bin")
        self.assertIn("has invalid NPZ magic", str(ctx.exception))

    def test_rejects_dimension_count_out_of_range(self):
        for ndims in (0, 5):
            payload = bytearray(motion_bin((2,), "f", [1.0, 2.0]))
            payload[4:8] = struct.pack("<I", ndims)
            with self.assertRaises(ValueError) as ctx:
                _motion.decode_bin(bytes(payload), "motion/x.bin")
            self.assertIn(f"has invalid dimension count {ndims}", str(ctx.exception))

    def test_rejects_zero_shape_dimension(self):
        payload = bytearray(motion_bin((2,), "f", [1.0, 2.0]))
        payload[8:12] = struct.pack("<I", 0)
        with self.assertRaises(ValueError) as ctx:
            _motion.decode_bin(bytes(payload), "motion/x.bin")
        self.assertIn("has invalid shape dimension 0", str(ctx.exception))

    def test_rejects_unsupported_dtype(self):
        payload = bytearray(motion_bin((1,), "f", [1.0]))
        payload[8 + 1 * 4 + 4] = ord("d")  # dtype char slot, size stays 4
        with self.assertRaises(ValueError) as ctx:
            _motion.decode_bin(bytes(payload), "motion/x.bin")
        self.assertIn("has unsupported dtype d4", str(ctx.exception))

    def test_rejects_truncated_payload(self):
        payload = motion_bin((3,), "f", [1.0, 2.0, 3.0])[:-4]
        with self.assertRaises(ValueError) as ctx:
            _motion.decode_bin(payload, "motion/x.bin")
        self.assertIn("data payload is truncated", str(ctx.exception))


class LoadMotionFolderTest(unittest.TestCase):
    def test_optional_folder_names_metadata(self):
        files = golden_motion_files()
        files["metadata.json"] = json.dumps({
            "body_names": ["pelvis"],
            "joint_names": [f"joint_{i}" for i in range(DOF)],
        }).encode()
        motion = _motion.load_motion_folder(files)
        self.assertEqual(motion["body_names"], ["pelvis"])
        self.assertEqual(len(motion["joint_names"]), DOF)
        self.assertEqual(motion["joint_position"].shape, (FRAMES * DOF,))

    def test_rejects_malformed_folder_names(self):
        files = golden_motion_files()
        files["metadata.json"] = b'{"body_names": ["pelvis", "foot"]}'
        with self.assertRaisesRegex(ValueError, "1 unique non-empty names"):
            _motion.load_motion_folder(files)

    def test_golden_motion_folder(self):
        motion = _motion.load_motion_folder(golden_motion_files(), "inline:golden/motion")
        self.assertEqual(set(motion), BASE_KEYS)
        self.assertEqual(motion["frame_count"], 5)
        self.assertEqual(motion["body_count"], 1)
        self.assertEqual(motion["dof_count"], 29)
        self.assertEqual(motion["fps"], 50.0)
        for key in ("frame_count", "body_count", "dof_count"):
            self.assertIsInstance(motion[key], int)
        expected = golden_expected_buffers()
        for key, want in expected.items():
            self.assertEqual(motion[key].dtype, np.dtype(np.float32))
            self.assertEqual(motion[key].ndim, 1)
            np.testing.assert_array_equal(motion[key], want)

    def test_missing_required_file(self):
        files = golden_motion_files()
        del files["joint_vel.bin"]
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn(
            "inline:golden/motion is missing required file joint_vel.bin",
            str(ctx.exception),
        )

    def test_wrong_fps_dtype_rejected(self):
        files = golden_motion_files()
        files["fps.bin"] = motion_bin((1,), "f", [50.0])
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn(
            "motion folder has missing or invalid required files", str(ctx.exception)
        )

    def test_shape_mismatch_rejected(self):
        files = golden_motion_files()
        values = [_joint_vel_value(i) for i in range(FRAMES * (DOF - 1))]
        files["joint_vel.bin"] = motion_bin((FRAMES, DOF - 1), "f", values)
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn(
            "[motionReference] joint_vel shape 5x28 does not match 5x29",
            str(ctx.exception),
        )

    def test_one_dimensional_joint_positions_rejected(self):
        files = golden_motion_files()
        files["joint_pos.bin"] = motion_bin((FRAMES,), "f", [0.0] * FRAMES)
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn(
            "motion folder has invalid joint or body dimensions", str(ctx.exception)
        )

    def test_non_finite_value_rejected(self):
        files = golden_motion_files()
        values = [_joint_pos_value(i) for i in range(FRAMES * DOF)]
        values[3] = float("nan")
        files["joint_pos.bin"] = motion_bin((FRAMES, DOF), "f", values)
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn(
            "[motionReference] joint_pos contains a non-finite value at 3",
            str(ctx.exception),
        )

    def test_non_positive_fps_rejected(self):
        files = golden_motion_files()
        files["fps.bin"] = motion_bin((1,), "l", [0])
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_folder(files, "inline:golden/motion")
        self.assertIn("fps must be positive, got 0", str(ctx.exception))


NPZ_FRAMES = 3
NPZ_BODIES = 2
NPZ_DOF = 2


def _npz_fixture(primary):
    """A small 3-frame / 2-body / 2-dof motion saved as a real .npz file."""
    joint_pos = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    joint_vel = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    body_pos = np.arange(1.0, 1.0 + NPZ_FRAMES * NPZ_BODIES * 3).reshape(
        NPZ_FRAMES, NPZ_BODIES, 3)
    body_quat = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0]), NPZ_FRAMES * NPZ_BODIES
    ).reshape(NPZ_FRAMES, NPZ_BODIES, 4)
    body_lin = np.arange(1.0, 1.0 + 18.0).reshape(NPZ_FRAMES, NPZ_BODIES, 3)
    body_ang = np.arange(10.0, 10.0 + 18.0).reshape(NPZ_FRAMES, NPZ_BODIES, 3)
    if primary:
        mapping = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_lin,
            "body_ang_vel_w": body_ang,
            "fps": np.float64(2.0),
        }
    else:
        mapping = {
            "dof_positions": joint_pos,
            "dof_velocities": joint_vel,
            "body_positions": body_pos,
            "body_rotations": body_quat,
            "body_linear_velocities": body_lin,
            "body_angular_velocities": body_ang,
            "fps": np.float64(2.0),
        }
    buffer = io.BytesIO()
    np.savez(buffer, **mapping)
    return buffer.getvalue()


def _npz_expected_buffers():
    return {
        "joint_position": np.float32([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        "joint_velocity": np.float32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "body_position": np.arange(1.0, 19.0, dtype=np.float32),
        "body_quaternion": np.float32([1.0, 0.0, 0.0, 0.0] * (NPZ_FRAMES * NPZ_BODIES)),
        "body_linear_velocity": np.arange(1.0, 19.0, dtype=np.float32),
        "body_angular_velocity": np.arange(10.0, 28.0, dtype=np.float32),
    }


def _assert_motions_equal(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key in ("frame_count", "body_count", "dof_count", "fps"):
        test.assertEqual(actual[key], expected[key], key)
    for key in ("joint_position", "joint_velocity", "body_position",
                "body_quaternion", "body_linear_velocity", "body_angular_velocity"):
        np.testing.assert_array_equal(actual[key], expected[key], key)


class LoadMotionNpzTest(unittest.TestCase):
    def test_primary_member_names(self):
        motion = _motion.load_motion_npz(_npz_fixture(primary=True), "npz://motion")
        self.assertEqual(set(motion), BASE_KEYS)
        self.assertEqual(motion["frame_count"], NPZ_FRAMES)
        self.assertEqual(motion["body_count"], NPZ_BODIES)
        self.assertEqual(motion["dof_count"], NPZ_DOF)
        self.assertEqual(motion["fps"], 2.0)
        for key, want in _npz_expected_buffers().items():
            self.assertEqual(motion[key].dtype, np.dtype(np.float32))
            self.assertEqual(motion[key].ndim, 1)
            np.testing.assert_array_equal(motion[key], want, key)

    def test_fallback_member_names_equivalent(self):
        primary = _motion.load_motion_npz(_npz_fixture(primary=True), "npz://motion")
        fallback = _motion.load_motion_npz(_npz_fixture(primary=False), "npz://motion")
        _assert_motions_equal(self, fallback, primary)

    def test_optional_names_lists(self):
        buffer = io.BytesIO()
        np.savez(
            buffer,
            joint_pos=np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]]),
            joint_vel=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            body_pos_w=np.arange(18.0).reshape(3, 2, 3),
            body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], 6).reshape(3, 2, 4),
            body_lin_vel_w=np.arange(18.0).reshape(3, 2, 3),
            body_ang_vel_w=np.arange(18.0).reshape(3, 2, 3),
            fps=np.float64(2.0),
            dof_names=np.array(["hip", "knee"]),
            body_names=np.array(["base", "foot"]),
        )
        motion = _motion.load_motion_npz(buffer.getvalue(), "npz://motion")
        self.assertEqual(set(motion), BASE_KEYS | {"joint_names", "body_names"})
        self.assertEqual(motion["joint_names"], ["hip", "knee"])
        self.assertEqual(motion["body_names"], ["base", "foot"])

    def test_duplicate_names_rejected(self):
        buffer = io.BytesIO()
        np.savez(
            buffer,
            joint_pos=np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]]),
            joint_vel=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            body_pos_w=np.arange(18.0).reshape(3, 2, 3),
            body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], 6).reshape(3, 2, 4),
            body_lin_vel_w=np.arange(18.0).reshape(3, 2, 3),
            body_ang_vel_w=np.arange(18.0).reshape(3, 2, 3),
            fps=np.float64(2.0),
            dof_names=np.array(["hip", "hip"]),
        )
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_npz(buffer.getvalue(), "npz://motion")
        self.assertIn(
            "dof_names must not contain duplicate names", str(ctx.exception)
        )

    def test_exactly_one_member_name_required(self):
        base = {
            "joint_pos": np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]]),
            "joint_vel": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "body_pos_w": np.arange(18.0).reshape(3, 2, 3),
            "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], 6).reshape(3, 2, 4),
            "body_lin_vel_w": np.arange(18.0).reshape(3, 2, 3),
            "body_ang_vel_w": np.arange(18.0).reshape(3, 2, 3),
            "fps": np.float64(2.0),
        }
        variants = {
            "duplicate": {**base, "body_positions": np.arange(18.0).reshape(3, 2, 3)},
            "missing": {key: value for key, value in base.items() if key != "body_pos_w"},
        }
        for label, mapping in variants.items():
            with self.subTest(variant=label):
                buffer = io.BytesIO()
                np.savez(buffer, **mapping)
                with self.assertRaises(ValueError) as ctx:
                    _motion.load_motion_npz(buffer.getvalue(), "npz://motion")
                self.assertIn(
                    "exactly one body position array (body_pos_w or body_positions)",
                    str(ctx.exception),
                )

    def test_non_finite_member_rejected(self):
        buffer = io.BytesIO()
        np.savez(
            buffer,
            joint_pos=np.array([[0.0, float("nan")], [20.0, 30.0], [40.0, 50.0]]),
            joint_vel=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            body_pos_w=np.arange(18.0).reshape(3, 2, 3),
            body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], 6).reshape(3, 2, 4),
            body_lin_vel_w=np.arange(18.0).reshape(3, 2, 3),
            body_ang_vel_w=np.arange(18.0).reshape(3, 2, 3),
            fps=np.float64(2.0),
        )
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_npz(buffer.getvalue(), "npz://motion")
        self.assertIn(
            "npz://motion:joint_pos contains a non-finite value at 1",
            str(ctx.exception),
        )

    def test_non_scalar_fps_rejected(self):
        buffer = io.BytesIO()
        np.savez(
            buffer,
            joint_pos=np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]]),
            joint_vel=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            body_pos_w=np.arange(18.0).reshape(3, 2, 3),
            body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], 6).reshape(3, 2, 4),
            body_lin_vel_w=np.arange(18.0).reshape(3, 2, 3),
            body_ang_vel_w=np.arange(18.0).reshape(3, 2, 3),
            fps=np.array([[2.0]]),
        )
        with self.assertRaises(ValueError) as ctx:
            _motion.load_motion_npz(buffer.getvalue(), "npz://motion")
        self.assertIn("fps must be a scalar", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
