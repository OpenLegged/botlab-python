"""mujoco access queries (mujocoAccessRegistry.ts mirror)."""


import mujoco
import numpy as np

from ._values import _error, _flat, _is_integer, _number

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
