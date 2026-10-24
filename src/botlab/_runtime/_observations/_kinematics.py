"""Root/body frame and velocity helpers (mujocoObservations.ts kinematics mirror)."""


import mujoco
import numpy as np

from ._values import _error, _first_present, _flat, _float32_slice

FREE_JOINT_QPOS_DIM = 7
FREE_JOINT_QVEL_DIM = 6


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


def _uses_imu_sensors(config):
    """Node data ``imuSensors: true`` (written at plan load) opts into the model's IMU sensors."""
    return (config or {}).get("imuSensors") is True


def _root_kinematic_quaternion(model, data):
    if _has_root_free_joint(model, data):
        return _float32_slice(data.qpos, "data.qpos", 3, 7)
    body_index = _base_body_index(model)
    address = body_index * 4
    return _float32_slice(data.xquat, "data.xquat", address, address + 4)


def _root_quaternion(model, data, config):
    if _uses_imu_sensors(config):
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
    return source not in ("root", "qvel", "freejoint") and _uses_imu_sensors(config)


def _base_linear_velocity(model, data, config):
    if _velocity_source(config) == "sensor" and _uses_imu_sensors(config):
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
        # Preserve unmarked stored plans; new SDK plans explicitly describe
        # MuJoCo's already body-local free-joint angular velocity.
        frame = (config or {}).get("rootAngularVelocityFrame")
        if frame == "body":
            return _float32_slice(data.qvel, "data.qvel", 3, 6)
        if frame is not None:
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
