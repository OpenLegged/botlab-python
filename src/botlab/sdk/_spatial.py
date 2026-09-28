"""Quaternion and spatial math, as plain values or as Signal expressions."""
import math

from ._signal import Signal, _numeric_signal
from ._vectors import require_vector


def quaternion_yaw(quaternion):
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    return math.atan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy * qy + qz * qz),
    )


def inverse_rotate(vector, quaternion):
    if isinstance(vector, Signal) or isinstance(quaternion, Signal):
        return _spatial("quat_rotate_inverse", quaternion, vector)
    vx, vy, vz = require_vector(vector, 3, "vector")
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        (1 - 2 * (yy + zz)) * vx + 2 * (xy + wz) * vy + 2 * (xz - wy) * vz,
        2 * (xy - wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz + wx) * vz,
        2 * (xz + wy) * vx + 2 * (yz - wx) * vy + (1 - 2 * (xx + yy)) * vz,
    ]


def projected_gravity(quaternion):
    if isinstance(quaternion, Signal):
        return inverse_rotate([0, 0, -1], quaternion)
    qw, qx, qy, qz = require_vector(quaternion, 4, "quaternion")
    return [
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ]


def _spatial(operation, *values):
    return Signal("proc-spatial-math", operation, {"operation": operation},
                  tuple(_numeric_signal(value) for value in values))


def rotate(vector, quaternion):
    """Rotate xyz vectors by wxyz quaternions; singleton batches broadcast."""
    return _spatial("quat_rotate", quaternion, vector)


def quaternion_multiply(left, right):
    return _spatial("quat_multiply", left, right)


def quaternion_inverse(quaternion):
    return _spatial("quat_inverse", quaternion)


def quaternion_heading(quaternion):
    """Keep only the quaternion's yaw rotation."""
    return _spatial("quat_heading", quaternion)


def quaternion_slerp(start, end, amount):
    return _spatial("quat_slerp", start, end, amount)


def rotation_matrix(quaternion):
    """Return row-major 3x3 rotation matrices."""
    return _spatial("quat_to_matrix", quaternion)


def rotation_6d(quaternion):
    """First two matrix columns, interleaved by row: r00,r01,r10,r11,r20,r21."""
    return _spatial("quat_to_rotation6d", quaternion)


def tangent_normal(quaternion):
    """Rotated x axis followed by rotated z axis (the AMP tangent/normal layout)."""
    return _spatial("quat_to_tangent_normal", quaternion)


def relative_position(position, origin, orientation):
    """Express world positions in the reference frame (origin, orientation)."""
    return _spatial("relative_position", position, origin, orientation)


def relative_quaternion(orientation, reference):
    return _spatial("relative_quaternion", orientation, reference)
