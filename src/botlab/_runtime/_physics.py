"""Native MuJoCo physics adapter for the local runtime session.

Mirrors the browser worker's step semantics (src/platform/workers/mujocoWorkerStep.ts):
ctrl applies once per control step, substep PD torque is recomputed from fresh
qpos/qvel on EVERY physics substep, applied forces are zeroed per substep.
Legacy PD mapping (no explicit indices) skips the floating-base prefix —
qpos by 7, qvel by 6 — exactly like ctrlOffset() in the worker.

Observation filling and contact timing arrive with the observation assembly
integration; everything here is the physics half only.
"""

import os
import re

import numpy as np

from ..sdk._runtime_types import Frame
from ._physics_ctrl import PreparedSubstepPd, compose_scoped_actions


class MujocoPhysics:
    """Session-facing adapter over a native mujoco.MjModel / MjData pair."""

    def __init__(
        self,
        model,
        data,
        mujoco_module,
        *,
        resources=None,
        path_resolver=None,
        initial_qpos=None,
        initial_qvel=None,
        robot_states=None,
    ):
        self.model = model
        self.data = data
        self._mujoco = mujoco_module
        self._resources = {
            str(resource["id"]): resource for resource in (resources or [])
        }
        self._path_resolver = path_resolver
        robots = [
            resource for resource in self._resources.values()
            if resource.get("kind") == "robot"
        ]
        self._robots = {str(resource["id"]): resource for resource in robots}
        self._robot_ids = [str(resource["id"]) for resource in robots]
        self._robot = robots[0] if robots else None
        if self._robot_ids:
            from ._compose import resolve_robot_scopes

            self._scopes = resolve_robot_scopes(
                mujoco_module, model, self._robot_ids
            )
        else:
            self._scopes = {}
        if robot_states is not None:
            # Composed scene: reset to model defaults, place each robot's
            # initial state at its scope, then snapshot the composed state so
            # reset() replays it deterministically.
            from ._compose import apply_composed_initial_state

            self._mujoco.mj_resetData(model, data)
            apply_composed_initial_state(mujoco_module, model, data, robot_states)
            self._mujoco.mj_forward(model, data)
            self._initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
            self._initial_qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        else:
            self._initial_qpos = self._state_vector(
                initial_qpos, data.qpos, "initialQpos"
            )
            self._initial_qvel = self._state_vector(
                initial_qvel, data.qvel, "initialQvel"
            )
        self._pending = {}
        self._physics_step_index = 0
        self._sensor_sequences = {}
        if robot_states is None:
            self._apply_initial_state()

    @staticmethod
    def _state_vector(values, current, label):
        source = current if values is None else values
        vector = np.asarray(source, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"[botlab.run] robot preset {label} must be an array")
        if len(vector) != len(current):
            raise ValueError(
                f"[botlab.run] robot preset {label} length {len(vector)} "
                f"does not match model state length {len(current)}"
            )
        if not np.all(np.isfinite(vector)):
            bad = int(np.flatnonzero(~np.isfinite(vector))[0])
            raise ValueError(
                f"[botlab.run] robot preset {label}[{bad}] must be finite"
            )
        return vector.copy()

    def _apply_initial_state(self, qpos=None, qvel=None):
        self.data.qpos[:] = self._initial_qpos if qpos is None else qpos
        self.data.qvel[:] = self._initial_qvel if qvel is None else qvel
        self._mujoco.mj_forward(self.model, self.data)

    def _resource(self, resource_id, kind):
        resource = self._resources.get(str(resource_id))
        if resource is None:
            raise ValueError(f'[botlab] unknown resource "{resource_id}"')
        if resource.get("kind") != kind:
            raise ValueError(
                f'[botlab] resource "{resource_id}" is {resource.get("kind")}, '
                f"expected {kind}"
            )
        return resource

    def _robot_resource(self, robot_id):
        if not self._robots:
            raise RuntimeError(
                "[botlab] custom-controller access needs a declared robot resource"
            )
        resource = self._robots.get(str(robot_id))
        if resource is None:
            raise ValueError(
                f'[botlab] unknown robot "{robot_id}" '
                f"(declared: {', '.join(self._robot_ids)})"
            )
        return resource

    @staticmethod
    def _non_negative_integer(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"[botlab] {label} must be a non-negative integer")
        return int(value)

    @staticmethod
    def _positive_integer(value, label):
        number = MujocoPhysics._non_negative_integer(value, label)
        if number == 0:
            raise ValueError(f"[botlab] {label} must be a positive integer")
        return number

    def _robot_layout(self, robot_id):
        """Scope-aware layout: config offsets are robot-local; the composed
        model shifts them by the robot's scope start."""
        resource = self._robot_resource(robot_id)
        config = resource.get("config") or {}
        dimension = self._positive_integer(
            config.get("actionDimension"), f"{robot_id}.actionDimension"
        )
        qpos_offset = self._non_negative_integer(
            config.get("jointQposOffset"), f"{robot_id}.jointQposOffset"
        )
        qvel_offset = self._non_negative_integer(
            config.get("jointQvelOffset"), f"{robot_id}.jointQvelOffset"
        )
        scope = self._scopes.get(str(robot_id))
        if scope is None:
            raise ValueError(f'[botlab] MuJoCo scope for robot "{robot_id}" is unavailable')
        abs_qpos = scope["qpos_start"] + qpos_offset
        abs_qvel = scope["qvel_start"] + qvel_offset
        if abs_qpos + dimension > len(self.data.qpos):
            raise ValueError(
                f"[botlab] {robot_id} joint qpos range is outside MuJoCo qpos"
            )
        if abs_qvel + dimension > len(self.data.qvel):
            raise ValueError(
                f"[botlab] {robot_id} joint qvel range is outside MuJoCo qvel"
            )
        actuator_start = scope["actuator_start"]
        if actuator_start + dimension > len(self.data.ctrl):
            raise ValueError(
                f"[botlab] {robot_id} action dimension {dimension} exceeds "
                f"MuJoCo ctrl length {len(self.data.ctrl)}"
            )
        return resource, dimension, abs_qpos, abs_qvel, scope

    def _actuator_joint_order(self, resource, dimension):
        """Robot-local actuator ctrl indices (actuatorJointOrder preset)."""
        raw = (resource.get("config") or {}).get("actuatorJointOrder")
        if raw is None:
            return np.arange(dimension, dtype=np.int64)
        if (
            not isinstance(raw, list)
            or len(raw) != dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
                or value >= dimension
                for value in raw
            )
        ):
            raise ValueError(
                f"[botlab] {resource['id']}.actuatorJointOrder must contain "
                f"{dimension} robot-local ctrl indices"
            )
        if len(set(raw)) != dimension:
            raise ValueError(
                f"[botlab] {resource['id']}.actuatorJointOrder must not contain duplicates"
            )
        return np.asarray(raw, dtype=np.int64)

    @classmethod
    def from_xml_string(cls, xml, mujoco_module, **kwargs):
        model = mujoco_module.MjModel.from_xml_string(xml)
        data = mujoco_module.MjData(model)
        mujoco_module.mj_forward(model, data)
        return cls(model, data, mujoco_module, **kwargs)

    @classmethod
    def from_xml_path(cls, path, mujoco_module, **kwargs):
        model = mujoco_module.MjModel.from_xml_path(path)
        data = mujoco_module.MjData(model)
        mujoco_module.mj_forward(model, data)
        return cls(model, data, mujoco_module, **kwargs)

    # -- session interface --------------------------------------------------

    def qpos(self):
        return self.data.qpos

    def qvel(self):
        return self.data.qvel

    def step_substeps(self, steps, ctrl=None, substep_pd=None, observer=None):
        data = self.data
        if (ctrl is not None or substep_pd is not None) and self._pending:
            raise RuntimeError(
                "[botlab] explicit plan control conflicts with a custom-controller action"
            )
        if ctrl is None and substep_pd is None and self._pending:
            ctrl, substep_pd = self.compose_payloads(self._pending)
        if ctrl is not None:
            if len(ctrl) != len(data.ctrl):
                raise ValueError(
                    f"[mujocoWorker] ctrl length {len(ctrl)} does not match MuJoCo "
                    f"ctrl length {len(data.ctrl)}"
                )
            data.ctrl[:] = ctrl
        prepared = (
            PreparedSubstepPd(substep_pd, data.qpos, data.qvel, data.ctrl)
            if substep_pd is not None else None
        )
        timestep = float(self.model.opt.timestep)
        for _ in range(steps):
            if prepared is not None:
                prepared.apply(data.qpos, data.qvel, data.ctrl)
            data.qfrc_applied[:] = 0
            self._mujoco.mj_step(self.model, data)
            self._physics_step_index += 1
            if observer is not None:
                try:
                    observer(Frame(timestep, float(data.time), self._physics_step_index))
                except Exception as error:
                    raise RuntimeError(
                        "[botlab] on_substep observer failed at physics step "
                        f"{self._physics_step_index}: {error}"
                    ) from error
        self._pending = {}

    def compose_payloads(self, actions):
        """Compose pending per-robot actions (see compose_scoped_actions)."""
        return compose_scoped_actions(
            {robot_id: self.robot_layout(robot_id) for robot_id in actions},
            actions,
            self.data.ctrl,
        )

    def reset(self):
        self._mujoco.mj_resetData(self.model, self.data)
        self._pending = {}
        self._physics_step_index = 0
        self._apply_initial_state()

    def should_stop(self):
        return False

    def after_control_step(self):
        return None

    def fill_observations(self, executor, last_action=None, parameters=None):
        from ._observations import camera_frames, fill_observation, height_scan

        context = {
            "model": self.model,
            "data": self.data,
            "last_action": last_action,
            "parameters": parameters,
            "height_scan": lambda spec: height_scan(self.model, self.data, spec),
            "camera_frames": lambda spec: camera_frames(self.model, self.data, spec),
        }
        for binding in executor.observation_bindings:
            fill_observation(binding, context)

    def robot_observation(self, robot_id):
        _resource, dimension, qpos_offset, qvel_offset, scope = self._robot_layout(robot_id)
        actuator_start = scope["actuator_start"]
        base = scope["qpos_start"]
        base_vel = scope["qvel_start"]
        actuator_force = np.asarray(self.data.actuator_force, dtype=np.float64)
        return {
            "qpos": np.asarray(self.data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float64).tolist(),
            "joint_position": np.asarray(
                self.data.qpos[qpos_offset : qpos_offset + dimension],
                dtype=np.float64,
            ).tolist(),
            "joint_velocity": np.asarray(
                self.data.qvel[qvel_offset : qvel_offset + dimension],
                dtype=np.float64,
            ).tolist(),
            "joint_torque": actuator_force[actuator_start : actuator_start + dimension].tolist(),
            "base_position": np.asarray(
                self.data.qpos[base : base + 3], dtype=np.float64
            ).tolist(),
            "base_quaternion": np.asarray(
                self.data.qpos[base + 3 : base + 7], dtype=np.float64
            ).tolist(),
            "base_velocity": np.asarray(
                self.data.qvel[base_vel : base_vel + 6], dtype=np.float64
            ).tolist(),
        }

    def robot_scope(self, robot_id):
        """The robot's composed-model scope (qpos/qvel/ctrl start + prefix)."""
        self._robot_resource(robot_id)
        scope = self._scopes.get(str(robot_id))
        if scope is None:
            raise ValueError(f'[botlab] MuJoCo scope for robot "{robot_id}" is unavailable')
        return scope

    def base_pose(self, robot_id):
        """(position, quaternion) of the robot's free root, from its scope."""
        scope = self.robot_scope(robot_id)
        start = scope["qpos_start"]
        qpos = self.data.qpos
        return (
            [float(qpos[start]), float(qpos[start + 1]), float(qpos[start + 2])],
            [
                float(qpos[start + 3]), float(qpos[start + 4]),
                float(qpos[start + 5]), float(qpos[start + 6]),
            ],
        )

    def robot_layout(self, robot_id):
        """Scope-resolved control layout for host-side payload composition."""
        resource, dimension, abs_qpos, abs_qvel, scope = self._robot_layout(robot_id)
        config = resource.get("config") or {}
        return {
            "dimension": dimension,
            "qpos_offset": abs_qpos,
            "qvel_offset": abs_qvel,
            "actuator_start": scope["actuator_start"],
            "qpos_start": scope["qpos_start"],
            "qvel_start": scope["qvel_start"],
            "name_prefix": scope["name_prefix"],
            "ctrl_order": config.get("actuatorJointOrder"),
        }

    def sensor_observation(self, sensor_id, channel=None):
        del channel  # Sensor.observe selects a channel from the returned snapshot.
        from ._observations import camera_frames, read_camera_config

        resource = self._resource(sensor_id, "sensor")
        spec = dict(resource.get("config") or {})
        # The resource config is already resolved by this adapter. Do not send
        # resourceId back through the process-global offline host lookup.
        spec["nodeId"] = f"sensor:{sensor_id}"
        config = read_camera_config(spec)
        frames = camera_frames(self.model, self.data, spec)
        sequence = self._sensor_sequences.get(str(sensor_id), 0) + 1
        self._sensor_sequences[str(sensor_id)] = sequence
        width, height = config["width"], config["height"]
        names = ("rgb",) if config["outputMode"] == "rgb" else (
            ("depth",) if config["outputMode"] == "depth" else ("rgb", "depth")
        )
        snapshot = {}
        for name, values in zip(names, frames):
            snapshot[name] = {
                "data": np.asarray(values, dtype=np.float32).tolist(),
                "shape": [height, width, 3] if name == "rgb" else [height, width],
                "sequence": sequence,
                "age_steps": 0,
            }
        return snapshot

    def apply_action(self, robot_id, action):
        resource, dimension, qpos_offset, qvel_offset, scope = self._robot_layout(robot_id)
        if str(robot_id) in self._pending:
            raise RuntimeError(
                f'[botlab] Controller.step must produce exactly one action for robot "{robot_id}"'
            )
        if isinstance(action, dict) and action.get("kind") == "position-pd":
            values = {}
            for source, target in (
                ("target", "target"),
                ("kp", "kps"),
                ("kd", "kds"),
                ("effort_limit", "effort"),
            ):
                vector = np.asarray(action.get(source), dtype=np.float64)
                if vector.ndim != 1 or len(vector) != dimension:
                    raise ValueError(
                        f"[botlab] position-pd.{source} must contain {dimension} values"
                    )
                if not np.all(np.isfinite(vector)):
                    bad = int(np.flatnonzero(~np.isfinite(vector))[0])
                    raise ValueError(
                        f"[botlab] position-pd.{source}[{bad}] must be finite"
                    )
                values[target] = vector
            if np.any(values["effort"] <= 0):
                raise ValueError(
                    "[botlab] position-pd.effort_limit values must be positive"
                )
            indices = np.arange(dimension, dtype=np.int64)
            # 机器人局部索引；compose_payloads 在 step 时统一平移到组合模型的
            # 绝对下标（scope + preset offset）。
            self._pending[str(robot_id)] = ("pd", {
                "target": values["target"],
                "kps": values["kps"],
                "kds": values["kds"],
                "effort_min": -values["effort"],
                "effort_max": values["effort"],
                "qpos_indices": indices,
                "qvel_indices": indices,
                "ctrl_indices": self._actuator_joint_order(resource, dimension),
            })
            return

        vector = np.asarray(action, dtype=np.float64)
        if vector.ndim != 1 or len(vector) != dimension:
            raise ValueError(
                f"[botlab] ctrl action must contain {dimension} values"
            )
        if not np.all(np.isfinite(vector)):
            bad = int(np.flatnonzero(~np.isfinite(vector))[0])
            raise ValueError(f"[botlab] ctrl action[{bad}] must be finite")
        self._pending[str(robot_id)] = ("ctrl", vector)

    def request_reset(self, reason, spec=None):
        del reason
        if spec is None:
            self.reset()
            return 0.0
        if not isinstance(spec, dict):
            raise ValueError("[botlab] reset_to() expects a motion reset specification")
        motion_id = spec.get("motionId")
        if not isinstance(motion_id, str) or not motion_id:
            raise ValueError("[botlab] reset_to() requires a Motion resource")
        robot_id = spec.get("robotId")
        if robot_id is None:
            robot_id = self._robot["id"] if self._robot is not None else None
        resource, dimension, _abs_qpos, _abs_qvel, scope = self._robot_layout(robot_id)
        local_config = resource.get("config") or {}
        local_offsets = (
            local_config.get("jointQposOffset"),
            local_config.get("jointQvelOffset"),
        )
        if local_offsets not in ((7, 6), (0, 0)):
            raise ValueError(
                f"[botlab] reset_to() requires {robot_id}.jointQposOffset/"
                "jointQvelOffset of 7/6 (floating root) or 0/0 (fixed base)"
            )
        frame = self._non_negative_integer(spec.get("frame"), "reset_to() frame")
        root_body = self._non_negative_integer(
            spec.get("rootBody"), "reset_to() root_body"
        )
        motion = self.load_motion(motion_id)
        if frame >= motion["frame_count"]:
            raise ValueError(
                f"[motionReference] initial-state frame {frame} is outside the motion's "
                f'0..{motion["frame_count"] - 1}'
            )
        if root_body >= motion["body_count"]:
            raise ValueError(
                f"[motionReference] initial-state anchor link {root_body} is outside "
                f'the motion\'s 0..{motion["body_count"] - 1} bodies'
            )
        joint_indices = spec.get("jointIndices")
        if joint_indices is None:
            if motion["dof_count"] != dimension:
                raise ValueError(
                    f"[botlab] reset_to() motion has {motion['dof_count']} dofs but "
                    f'robot "{robot_id}" has {dimension} joints; pass joint_indices to map them'
                )
            joint_indices = list(range(dimension))
        if (
            not isinstance(joint_indices, list)
            or len(joint_indices) != dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
                or value >= motion["dof_count"]
                for value in joint_indices
            )
        ):
            raise ValueError(
                f"[botlab] reset_to() joint_indices must map all {dimension} robot joints"
            )
        qpos, qvel = self._motion_initial_state(
            motion, frame, root_body, joint_indices, local_offsets[0] == 7
        )
        if len(self._robots) > 1:
            # 组合模型：把该机器人的局部向量拼接进它的 scope 切片，其余机器人
            # 保持初始快照。
            full_qpos = self._initial_qpos.copy()
            full_qvel = self._initial_qvel.copy()
            qpos_start = scope["qpos_start"]
            qvel_start = scope["qvel_start"]
            if qpos_start + len(qpos) > len(full_qpos):
                raise ValueError(
                    f"[botlab] reset_to() produced {len(qpos)} qpos values outside "
                    f'robot "{robot_id}" scope in the composed model'
                )
            if qvel_start + len(qvel) > len(full_qvel):
                raise ValueError(
                    f"[botlab] reset_to() produced {len(qvel)} qvel values outside "
                    f'robot "{robot_id}" scope in the composed model'
                )
            full_qpos[qpos_start : qpos_start + len(qpos)] = qpos
            full_qvel[qvel_start : qvel_start + len(qvel)] = qvel
            qpos, qvel = full_qpos, full_qvel
        if len(qpos) != len(self.data.qpos) or len(qvel) != len(self.data.qvel):
            raise ValueError(
                f"[botlab] reset_to() produced {len(qpos)}/{len(qvel)} state values but "
                f"the local model expects {len(self.data.qpos)}/{len(self.data.qvel)}"
            )
        self._mujoco.mj_resetData(self.model, self.data)
        self._pending = {}
        self._apply_initial_state(qpos, qvel)
        return frame / motion["fps"]

    @staticmethod
    def _motion_initial_state(motion, frame, body, joint_indices, include_root):
        dofs = motion["dof_count"]
        bodies = motion["body_count"]
        joint_base = frame * dofs
        joint_qpos = [float(motion["joint_position"][joint_base + index]) for index in joint_indices]
        joint_qvel = [float(motion["joint_velocity"][joint_base + index]) for index in joint_indices]
        if not include_root:
            return joint_qpos, joint_qvel
        body3 = (frame * bodies + body) * 3
        body4 = (frame * bodies + body) * 4
        position = np.asarray(motion["body_position"][body3 : body3 + 3], dtype=np.float64)
        quaternion = np.asarray(
            motion["body_quaternion"][body4 : body4 + 4], dtype=np.float64
        )
        linear = np.asarray(
            motion["body_linear_velocity"][body3 : body3 + 3], dtype=np.float64
        )
        angular_world = np.asarray(
            motion["body_angular_velocity"][body3 : body3 + 3], dtype=np.float64
        )
        norm = float(np.dot(quaternion, quaternion))
        inverse = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=np.float64,
        ) / max(norm, 1e-9)
        w, x, y, z = inverse
        vx, vy, vz = angular_world
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        angular_body = np.array(
            [
                vx + w * tx + y * tz - z * ty,
                vy + w * ty + z * tx - x * tz,
                vz + w * tz + x * ty - y * tx,
            ]
        )
        return (
            [*position.tolist(), *quaternion.tolist(), *joint_qpos],
            [*linear.tolist(), *angular_body.tolist(), *joint_qvel],
        )

    def read_mujoco(self, robot_id, query, target=None, start=None, length=None):
        from ._observations import read_mujoco_access_query

        self._robot_resource(robot_id)
        scope = self._scopes.get(str(robot_id)) or {}
        context = {"model": self.model, "data": self.data}
        if scope.get("name_prefix"):
            context["namePrefix"] = scope["name_prefix"]
        config = {"query": query}
        if target is not None:
            config["target"] = target
        if start is not None:
            config["start"] = start
        if length is not None:
            config["length"] = length
        values = read_mujoco_access_query(context, config)
        return np.asarray(values, dtype=np.float64).tolist()

    def load_motion(self, motion_id):
        from ._motion import (
            MOTION_METADATA_FILE,
            MOTION_REQUIRED_FILES,
            load_motion_folder,
            load_motion_npz,
        )

        resource = self._resources.get(str(motion_id))
        if resource is not None:
            if resource.get("kind") != "motion":
                raise ValueError(
                    f'[botlab] resource "{motion_id}" is {resource.get("kind")}, expected motion'
                )
            source = resource["source"]
        else:
            source = str(motion_id)
        if source.startswith("registry://"):
            raise ValueError(
                "[botlab] registry motion resources require the browser asset registry; "
                "use a local NPZ file or motion folder"
            )
        path = self._path_resolver(source) if self._path_resolver is not None else source
        if os.path.isdir(path):
            files = {}
            for name in MOTION_REQUIRED_FILES:
                file_path = os.path.join(path, name)
                if not os.path.isfile(file_path):
                    raise ValueError(
                        f"[motionReference] {path} is missing required file {name}"
                    )
                with open(file_path, "rb") as handle:
                    files[name] = handle.read()
            metadata_path = os.path.join(path, MOTION_METADATA_FILE)
            if os.path.isfile(metadata_path):
                with open(metadata_path, "rb") as handle:
                    files[MOTION_METADATA_FILE] = handle.read()
            return load_motion_folder(files, path)
        if not os.path.isfile(path):
            raise ValueError(f'[botlab] motion source "{source}" was not found')
        if not path.lower().endswith(".npz"):
            raise ValueError(
                f'[botlab] motion source "{source}" must be an NPZ file or motion folder'
            )
        with open(path, "rb") as handle:
            return load_motion_npz(handle.read(), path)

    def update_heightfield(self, terrain_id, heights):
        resource = self._resource(terrain_id, "scene")
        config = resource.get("config") or {}
        if config.get("terrainKind") != "heightfield":
            raise ValueError(f'[botlab] Scene "{terrain_id}" is not a dynamic heightfield')
        grid = config.get("heightfieldGrid", [33, 33])
        if (
            not isinstance(grid, list)
            or len(grid) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid)
        ):
            raise ValueError(
                f"[botlab] {terrain_id}.heightfieldGrid must contain two positive integers"
            )
        rows, columns = grid
        array = np.asarray(heights, dtype=np.float32)
        if array.ndim == 2:
            if array.shape != (rows, columns):
                raise ValueError(
                    f"[botlab] heightfield values shape {array.shape} does not match "
                    f"{rows}x{columns}"
                )
            values = array.reshape(-1)
        elif array.ndim == 1 and len(array) == rows * columns:
            values = array
        else:
            raise ValueError(
                f"[botlab] heightfield values must contain {rows * columns} values"
            )
        invalid = np.flatnonzero(~np.isfinite(values) | (values < 0) | (values > 1))
        if len(invalid):
            raise ValueError(
                f"[mujocoWorker] heightfield values[{int(invalid[0])}] must be finite "
                "and between 0 and 1"
            )
        clean = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{terrain_id}_heightfield")
        name = clean if re.match(r"^[A-Za-z_]", clean) else f"terrain_{clean}"
        heightfield_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_HFIELD, name
        )
        if heightfield_id < 0:
            raise ValueError(f'[mujocoWorker] heightfield name "{name}" was not found')
        model_rows = int(self.model.hfield_nrow[heightfield_id])
        model_columns = int(self.model.hfield_ncol[heightfield_id])
        if (rows, columns) != (model_rows, model_columns):
            raise ValueError(
                f"[mujocoWorker] heightfield {heightfield_id} dimensions {rows}x{columns} "
                f"do not match model {model_rows}x{model_columns}"
            )
        start = int(self.model.hfield_adr[heightfield_id])
        self.model.hfield_data[start : start + len(values)] = values
        self._mujoco.mj_forward(self.model, self.data)
        return None
