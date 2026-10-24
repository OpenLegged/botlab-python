"""Control-vector layout, scoped action composition and per-substep PD."""

import numpy as np


def ctrl_offset(total_length, ctrl_length, free_joint_prefix, field_name):
    if total_length == ctrl_length + free_joint_prefix:
        return free_joint_prefix
    if total_length == ctrl_length:
        return 0
    raise ValueError(
        f"[mujocoWorker] {field_name} length {total_length} is incompatible "
        f"with ctrl length {ctrl_length}"
    )


def mapped_index(indices, index, source_length, field_name):
    source_index = indices[index] if indices is not None else index
    if (
        not isinstance(source_index, (int, np.integer))
        or source_index < 0
        or source_index >= source_length
    ):
        raise ValueError(
            f"[mujocoWorker] substep PD {field_name} index {source_index} is "
            f"outside source length {source_length}"
        )
    return int(source_index)


def compose_scoped_actions(layouts, actions, current_ctrl):
    """Compose per-robot actions into one atomic (ctrl, substep_pd) payload.

    ``layouts``: ``{robot_id: {"dimension", "qpos_offset", "qvel_offset",
    "actuator_start"}}``（``MujocoPhysics.robot_layout``）；``actions``:
    ``{robot_id: ("ctrl", values) | ("pd", view)}``，PD view 的索引是机器人
    局部值，这里统一平移到组合模型的绝对下标。对照
    src/domain/python/pythonScopedActions.ts 的 composePythonRobotActions。
    """
    full_ctrl = np.asarray(current_ctrl, dtype=np.float64).copy()
    owners = {}
    pd_parts = []
    has_raw = False
    for robot_id, (kind, payload) in actions.items():
        layout = layouts[robot_id]
        dimension = layout["dimension"]
        actuator_start = layout["actuator_start"]
        for index in range(actuator_start, actuator_start + dimension):
            owner = owners.get(index)
            if owner is not None:
                raise RuntimeError(
                    f"[botlab] Robot actuator scopes overlap at ctrl index "
                    f"{index}: {owner}, {robot_id}"
                )
            owners[index] = robot_id
        if kind == "ctrl":
            vector = np.asarray(payload, dtype=np.float64).ravel()
            if vector.shape[0] != dimension:
                raise ValueError(
                    f"[botlab] {robot_id} ctrl action must contain {dimension} values"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"[botlab] {robot_id} ctrl action must be finite")
            full_ctrl[actuator_start : actuator_start + dimension] = vector
            has_raw = True
        else:
            pd_parts.append((robot_id, layout, payload))

    ctrl = full_ctrl if has_raw else None
    if not pd_parts:
        return ctrl, None

    merged = {name: [] for name in (
        "target", "kps", "kds", "effort_min", "effort_max",
        "qpos_indices", "qvel_indices", "ctrl_indices",
    )}
    for robot_id, layout, view in pd_parts:
        dimension = layout["dimension"]
        identity = np.arange(dimension, dtype=np.int64)
        for key, base in (
            ("qpos_indices", layout["qpos_offset"]),
            ("qvel_indices", layout["qvel_offset"]),
            ("ctrl_indices", layout["actuator_start"]),
        ):
            local = view.get(key)
            if local is None:
                local = identity
            else:
                local = np.asarray(local, dtype=np.int64).ravel()
                if local.shape[0] != dimension:
                    raise ValueError(
                        f"[botlab] {robot_id} substep PD {key} must contain "
                        f"{dimension} values"
                    )
                if np.any(local < 0) or np.any(local >= dimension):
                    raise ValueError(
                        f"[botlab] {robot_id} substep PD {key} index is outside "
                        f"robot action dimension {dimension}"
                    )
            merged[key].append(base + local)
        for name in ("target", "kps", "kds", "effort_min", "effort_max"):
            merged[name].append(np.asarray(view[name], dtype=np.float64).ravel())

    payload = {
        name: np.concatenate(merged[name])
        for name in ("target", "kps", "kds", "effort_min", "effort_max")
    }
    payload["qpos_indices"] = np.concatenate(merged["qpos_indices"]).astype(np.int64)
    payload["qvel_indices"] = np.concatenate(merged["qvel_indices"]).astype(np.int64)
    payload["ctrl_indices"] = np.concatenate(merged["ctrl_indices"]).astype(np.int64)
    return ctrl, payload


class PreparedSubstepPd:
    def __init__(self, pd, qpos, qvel, ctrl):
        target = np.asarray(pd["target"], dtype=np.float64)
        action_count = target.shape[0]
        if action_count <= 0:
            raise ValueError("[mujocoWorker] substep PD buffers must not be empty")

        kps = np.asarray(pd["kps"], dtype=np.float64)
        kds = np.asarray(pd["kds"], dtype=np.float64)
        effort_min = np.asarray(pd["effort_min"], dtype=np.float64)
        effort_max = np.asarray(pd["effort_max"], dtype=np.float64)
        for name, buffer in (
            ("kps", kps), ("kds", kds), ("effortMin", effort_min), ("effortMax", effort_max)
        ):
            if buffer.shape[0] != action_count:
                raise ValueError(
                    "[mujocoWorker] substep PD buffers must all match target length "
                    f"{action_count}"
                )

        qpos_indices = pd.get("qpos_indices")
        qvel_indices = pd.get("qvel_indices")
        ctrl_indices = pd.get("ctrl_indices")
        if qpos_indices is not None:
            qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        if qvel_indices is not None:
            qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        if ctrl_indices is not None:
            ctrl_indices = np.asarray(ctrl_indices, dtype=np.int64)
        for name, buffer in (
            ("qposIndices", qpos_indices), ("qvelIndices", qvel_indices),
            ("ctrlIndices", ctrl_indices),
        ):
            if buffer is not None and len(buffer) != action_count:
                raise ValueError(
                    "[mujocoWorker] substep PD buffers must all match target length "
                    f"{action_count}"
                )

        explicit_index_count = sum(
            buffer is not None for buffer in (qpos_indices, qvel_indices, ctrl_indices)
        )
        if explicit_index_count not in (0, 3):
            raise ValueError(
                "[mujocoWorker] substep PD absolute mapping requires qposIndices, "
                "qvelIndices, and ctrlIndices"
            )
        uses_absolute_indices = explicit_index_count == 3
        if not uses_absolute_indices and action_count != len(ctrl):
            raise ValueError(
                f"[mujocoWorker] legacy substep PD buffers must match ctrl length {len(ctrl)}"
            )

        qpos_offset = (
            0 if uses_absolute_indices
            else ctrl_offset(len(qpos), len(ctrl), 7, "qpos")
        )
        qvel_offset = (
            0 if uses_absolute_indices
            else ctrl_offset(len(qvel), len(ctrl), 6, "qvel")
        )
        qpos_source_length = len(qpos) - qpos_offset
        qvel_source_length = len(qvel) - qvel_offset
        for i in range(action_count):
            mapped_index(qpos_indices, i, qpos_source_length, "qpos")
            mapped_index(qvel_indices, i, qvel_source_length, "qvel")
            mapped_index(ctrl_indices, i, len(ctrl), "ctrl")

        self.target = target
        self.kps = kps
        self.kds = kds
        self.effort_min = effort_min
        self.effort_max = effort_max
        self.qpos_indices = qpos_indices
        self.qvel_indices = qvel_indices
        self.ctrl_indices = ctrl_indices
        self.qpos_offset = qpos_offset
        self.qvel_offset = qvel_offset

    def apply(self, qpos, qvel, ctrl):
        qpos_index = self.qpos_indices
        qvel_index = self.qvel_indices
        ctrl_index = self.ctrl_indices
        indices = np.arange(self.target.shape[0]) if qpos_index is None else qpos_index
        vel_indices = indices if qvel_index is None else qvel_index
        out_indices = indices if ctrl_index is None else ctrl_index
        tau = (self.target - qpos[self.qpos_offset + indices]) * self.kps - qvel[
            self.qvel_offset + vel_indices
        ] * self.kds
        if not np.all(np.isfinite(tau)):
            bad = int(np.flatnonzero(~np.isfinite(tau))[0])
            raise ValueError(f"[mujocoWorker] substep PD tau[{bad}] is not finite")
        np.clip(tau, self.effort_min, self.effort_max, out=tau)
        ctrl[out_indices] = tau
