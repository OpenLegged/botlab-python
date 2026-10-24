"""PDProcessor (proc-pd) and its controller helpers."""

import math

import numpy as np

from ._output_common import _first_bad_index, _first_configured, _is_blank, _is_integer, _number

# ---------------------------------------------------------------------------
# PDProcessor (proc-pd) and its controller helpers
# ---------------------------------------------------------------------------


def _fixed_number_list(raw, name, length, context):
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(f"[{context}] {name} must be an array with {length} numbers")
    values = [_number(value) for value in raw]
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError(
            f"[{context}] {name} must contain exactly {length} finite numbers"
        )
    return np.array(values, dtype=np.float32)


def _scalar_or_fixed_number_list(raw, name, length, context):
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(
        raw, bool
    ):
        if not math.isfinite(float(raw)):
            raise ValueError(f"[{context}] {name} must be finite")
        return np.full(length, float(raw), dtype=np.float32)
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(
            f"[{context}] {name} must be a finite number or "
            f"an array with {length} numbers"
        )
    values = [_number(value) for value in raw]
    if len(values) == 1 and length != 1:
        if not math.isfinite(values[0]):
            raise ValueError(f"[{context}] {name}[0] must be finite")
        return np.full(length, values[0], dtype=np.float32)
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError(
            f"[{context}] {name} must contain exactly {length} finite numbers"
        )
    return np.array(values, dtype=np.float32)


def _first_negative(values):
    negative = values < 0
    return _first_bad_index(negative) if negative.any() else None


def _fixed_non_negative_number_list(raw, name, length, context):
    values = _fixed_number_list(raw, name, length, context)
    index = _first_negative(values)
    if index is not None:
        raise ValueError(f"[{context}] {name}[{index}] must be >= 0")
    return values


def _fixed_non_negative_limit_list(raw, name, length, context):
    values = _scalar_or_fixed_number_list(raw, name, length, context)
    index = _first_negative(values)
    if index is not None:
        raise ValueError(f"[{context}] {name}[{index}] must be >= 0")
    return values


def _symmetric_clip_bounds(raw, name, length, context):
    limit = _fixed_non_negative_limit_list(raw, name, length, context)
    return -limit, limit


def _explicit_clip_bounds(raw_min, raw_max, min_name, max_name, length, context):
    if raw_min is None or raw_max is None:
        raise ValueError(
            f"[{context}] {min_name} and {max_name} must be set together"
        )
    mins = _scalar_or_fixed_number_list(raw_min, min_name, length, context)
    maxs = _scalar_or_fixed_number_list(raw_max, max_name, length, context)
    exceeded = mins > maxs
    if exceeded.any():
        index = _first_bad_index(exceeded)
        raise ValueError(
            f"[{context}] {min_name}[{index}] must be <= {max_name}[{index}]"
        )
    return mins, maxs


class _PDController:
    """Position PD gains and torque computation (engine/controllers/PDController.ts)."""

    def __init__(self):
        self._context = "PDController"
        self._dof_count = 0
        self._kps = np.zeros(0, dtype=np.float32)
        self._kds = np.zeros(0, dtype=np.float32)
        self._effort_min = np.zeros(0, dtype=np.float32)
        self._effort_max = np.zeros(0, dtype=np.float32)
        self._kps64 = np.zeros(0, dtype=np.float64)
        self._kds64 = np.zeros(0, dtype=np.float64)
        self._effort_min64 = np.zeros(0, dtype=np.float64)
        self._effort_max64 = np.zeros(0, dtype=np.float64)

    @property
    def kps(self):
        return self._kps

    @property
    def kds(self):
        return self._kds

    @property
    def effort_min(self):
        return self._effort_min

    @property
    def effort_max(self):
        return self._effort_max

    def init(self, config, dof_count, context):
        if not isinstance(dof_count, int) or dof_count <= 0:
            raise ValueError(f"[{context}] dofCount must be a positive integer")
        self._context = context
        self._dof_count = dof_count
        self._kps = _fixed_non_negative_number_list(config.get("kps"), "kps", dof_count, context)
        self._kds = _fixed_non_negative_number_list(config.get("kds"), "kds", dof_count, context)

        effort_min_raw = config.get("effortLimitMin")
        if effort_min_raw is None:
            effort_min_raw = config.get("effort_limit_min")
        effort_max_raw = config.get("effortLimitMax")
        if effort_max_raw is None:
            effort_max_raw = config.get("effort_limit_max")
        if effort_min_raw is not None or effort_max_raw is not None:
            self._effort_min, self._effort_max = _explicit_clip_bounds(
                effort_min_raw,
                effort_max_raw,
                "effortLimitMin",
                "effortLimitMax",
                dof_count,
                context,
            )
        else:
            effort_limit_raw = config.get("effortLimit")
            if effort_limit_raw is None:
                effort_limit_raw = config.get("effort_limit")
            self._effort_min, self._effort_max = _symmetric_clip_bounds(
                effort_limit_raw, "effortLimit", dof_count, context
            )
        self._kps64 = self._kps.astype(np.float64)
        self._kds64 = self._kds.astype(np.float64)
        self._effort_min64 = self._effort_min.astype(np.float64)
        self._effort_max64 = self._effort_max.astype(np.float64)

    def compute(self, target, qpos, qvel, output):
        if (
            len(target) != self._dof_count
            or len(qpos) != self._dof_count
            or len(qvel) != self._dof_count
            or len(output) != self._dof_count
        ):
            raise ValueError(
                f"[{self._context}] target, qpos, qvel, and output must all "
                f"have {self._dof_count} values"
            )
        tau = (
            (np.asarray(target, dtype=np.float64) - np.asarray(qpos, dtype=np.float64))
            * self._kps64
            - np.asarray(qvel, dtype=np.float64) * self._kds64
        )
        finite = np.isfinite(tau)
        if not finite.all():
            raise ValueError(
                f"[{self._context}] computed tau[{_first_bad_index(~finite)}] "
                "is not finite"
            )
        np.clip(tau, self._effort_min64, self._effort_max64, out=tau)
        output[:] = tau

    def update_gains(self, kps=None, kds=None):
        for name, values in (("kps", kps), ("kds", kds)):
            if values is None:
                continue
            if len(values) != self._dof_count:
                raise ValueError(
                    f"[{self._context}] {name} input must contain exactly "
                    f"{self._dof_count} values"
                )
            for index, value in enumerate(values):
                if not math.isfinite(float(value)) or value < 0:
                    raise ValueError(
                        f"[{self._context}] {name} input[{index}] must be finite and >= 0"
                    )
        if kps is not None:
            self._kps[:] = kps
            self._kps64[:] = self._kps
        if kds is not None:
            self._kds[:] = kds
            self._kds64[:] = self._kds


def _resolve_pd_type(config):
    raw = config.get("pdType")
    if raw == "position":
        return "position"
    if raw == "velocity":
        return "velocity"
    if raw == "effort":
        return "effort"
    if raw is None or raw == "":
        return "position"
    raise ValueError(f'[PDProcessor] unsupported pdType "{str(raw)}"')


def _optional_non_negative_integer(raw, name):
    if _is_blank(raw):
        return 0
    value = _number(raw)
    if not _is_integer(value) or value < 0:
        raise ValueError(f"[PDProcessor] {name} must be a non-negative integer")
    return int(value)


def _optional_index_list(raw, name, length):
    if _is_blank(raw):
        return None
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(
            f"[PDProcessor] {name} must be an array with "
            f"{length} non-negative integers"
        )
    values = np.array([_number(value) for value in raw], dtype=np.float64)
    if len(values) != length:
        raise ValueError(f"[PDProcessor] {name} must contain exactly {length} values")
    bad = ~(np.isfinite(values) & (values == np.floor(values)) & (values >= 0))
    if bad.any():
        raise ValueError(
            f"[PDProcessor] {name}[{_first_bad_index(bad)}] "
            "must be a non-negative integer"
        )
    return values.astype(np.int32)


def _gain_input_slot(config, name, input_dims, dof):
    if name not in config:
        return None
    raw = config[name]
    if (
        not isinstance(raw, (int, float)) or isinstance(raw, bool)
        or not math.isfinite(raw) or raw != math.floor(raw) or raw < 3
    ):
        raise ValueError(f"[PDProcessor] {name} must be an input slot >= 3")
    slot = int(raw)
    if slot >= len(input_dims) or input_dims[slot] != dof:
        raise ValueError(
            f"[PDProcessor] {name} input must contain exactly {dof} values"
        )
    return slot


class PDProcessor:
    """Position / velocity / effort PD output with an optional substep view."""

    def __init__(self):
        self.output_buffers = [np.zeros(0, dtype=np.float32)]
        self._pd_type = "position"
        self._dof_count = 0
        self._controller = _PDController()
        self._target_dof_pos = np.zeros(0, dtype=np.float32)
        self._target_blend_start = np.zeros(0, dtype=np.float32)
        self._target_blend_steps = 0
        self._target_blend_step = 0
        self._qpos_indices = None
        self._qvel_indices = None
        self._ctrl_indices = None
        self._kps_input_slot = None
        self._kds_input_slot = None
        self._vel_kps64 = np.zeros(0, dtype=np.float64)
        self._effort_min64 = np.zeros(0, dtype=np.float64)
        self._effort_max64 = np.zeros(0, dtype=np.float64)

    def init(self, config, input_dims):
        self._pd_type = _resolve_pd_type(config)
        if self._pd_type == "velocity":
            self._init_velocity(config, input_dims)
            return
        if self._pd_type == "effort":
            self._init_effort(config, input_dims)
            return
        self._init_position(config, input_dims)

    def _init_position(self, config, input_dims):
        target_dim = input_dims[0] if len(input_dims) > 0 else 0
        qpos_dim = input_dims[1] if len(input_dims) > 1 else 0
        qvel_dim = input_dims[2] if len(input_dims) > 2 else 0
        if not target_dim or not qpos_dim or not qvel_dim:
            raise ValueError(
                "[PDProcessor] position type requires target, qpos, and qvel inputs"
            )
        if target_dim != qpos_dim or qpos_dim != qvel_dim:
            raise ValueError(
                "[PDProcessor] input dimensions must match; got "
                f"target={target_dim}, qpos={qpos_dim}, qvel={qvel_dim}"
            )
        self._dof_count = target_dim
        self._controller.init(config, target_dim, "PDProcessor")
        self._kps_input_slot = _gain_input_slot(config, "kpsInputSlot", input_dims, target_dim)
        self._kds_input_slot = _gain_input_slot(config, "kdsInputSlot", input_dims, target_dim)
        if self._kps_input_slot is not None and self._kps_input_slot == self._kds_input_slot:
            raise ValueError("[PDProcessor] kpsInputSlot and kdsInputSlot must be distinct")
        self._target_dof_pos = np.zeros(target_dim, dtype=np.float32)
        self._target_blend_start = np.zeros(target_dim, dtype=np.float32)
        self._target_blend_steps = _optional_non_negative_integer(
            config.get("targetBlendSteps"), "targetBlendSteps"
        )
        self._target_blend_step = 0
        self._qpos_indices = _optional_index_list(
            _first_configured(config.get("qposIndices"), config.get("qpos_indices")),
            "qposIndices",
            target_dim,
        )
        self._qvel_indices = _optional_index_list(
            _first_configured(
                config.get("qvelIndices"), config.get("qvel_indices"), self._qpos_indices
            ),
            "qvelIndices",
            target_dim,
        )
        self._ctrl_indices = _optional_index_list(
            _first_configured(config.get("ctrlIndices"), config.get("ctrl_indices")),
            "ctrlIndices",
            target_dim,
        )
        self.output_buffers = [np.zeros(target_dim, dtype=np.float32)]

    def _init_velocity(self, config, input_dims):
        vel_dim = input_dims[0] if len(input_dims) > 0 else 0
        qvel_dim = input_dims[1] if len(input_dims) > 1 else 0
        if not vel_dim or not qvel_dim:
            raise ValueError(
                "[PDProcessor] velocity type requires vel_target and qvel inputs"
            )
        if vel_dim != qvel_dim:
            raise ValueError(
                "[PDProcessor] vel_target and qvel dimensions must match; "
                f"got {vel_dim} vs {qvel_dim}"
            )
        self._dof_count = vel_dim
        vel_kps = _fixed_non_negative_number_list(
            config.get("kps"), "kps", vel_dim, "PDProcessor"
        )
        self._vel_kps64 = vel_kps.astype(np.float64)
        effort_limit_raw = config.get("effortLimit")
        if effort_limit_raw is None:
            effort_limit_raw = config.get("effort_limit")
        effort_min, effort_max = _symmetric_clip_bounds(
            effort_limit_raw, "effortLimit", vel_dim, "PDProcessor"
        )
        self._effort_min64 = effort_min.astype(np.float64)
        self._effort_max64 = effort_max.astype(np.float64)
        self.output_buffers = [np.zeros(vel_dim, dtype=np.float32)]

    def _init_effort(self, config, input_dims):
        tau_dim = input_dims[0] if len(input_dims) > 0 else 0
        if not tau_dim:
            raise ValueError("[PDProcessor] effort type requires target_tau input")
        self._dof_count = tau_dim
        effort_limit_raw = config.get("effortLimit")
        if effort_limit_raw is None:
            effort_limit_raw = config.get("effort_limit")
        effort_min, effort_max = _symmetric_clip_bounds(
            effort_limit_raw, "effortLimit", tau_dim, "PDProcessor"
        )
        self._effort_min64 = effort_min.astype(np.float64)
        self._effort_max64 = effort_max.astype(np.float64)
        self.output_buffers = [np.zeros(tau_dim, dtype=np.float32)]

    def compute(self, inputs):
        if self._pd_type == "velocity":
            self._compute_velocity(inputs)
            return
        if self._pd_type == "effort":
            self._compute_effort(inputs)
            return
        self._compute_position(inputs)

    def reset(self):
        self.output_buffers[0].fill(0)
        self._target_dof_pos.fill(0)
        self._target_blend_start.fill(0)
        self._target_blend_step = 0

    def _compute_position(self, inputs):
        target = inputs[0] if len(inputs) > 0 else None
        qpos = inputs[1] if len(inputs) > 1 else None
        qvel = inputs[2] if len(inputs) > 2 else None
        if target is None or qpos is None or qvel is None:
            raise ValueError(
                "[PDProcessor] Missing required inputs for position PD"
            )
        kps = self._gain_input(inputs, self._kps_input_slot, "kps")
        kds = self._gain_input(inputs, self._kds_input_slot, "kds")
        if kps is not None or kds is not None:
            self._controller.update_gains(kps, kds)
        effective_target = self._position_target(target, qpos)
        self._controller.compute(
            effective_target, qpos, qvel, self.output_buffers[0]
        )

    def _gain_input(self, inputs, slot, name):
        if slot is None:
            return None
        if slot >= len(inputs) or inputs[slot] is None:
            raise ValueError(f"[PDProcessor] Missing {name} input at slot {slot}")
        return inputs[slot]

    def _position_target(self, target, qpos):
        if self._target_blend_steps <= 0:
            self._target_dof_pos[:] = target
            return self._target_dof_pos

        if self._target_blend_step == 0:
            self._target_blend_start[:] = qpos

        alpha = (
            1.0
            if self._target_blend_steps <= 1
            else min(1.0, self._target_blend_step / (self._target_blend_steps - 1))
        )
        if alpha >= 1:
            self._target_dof_pos[:] = target
        else:
            start = self._target_blend_start.astype(np.float64)
            goal = np.asarray(target, dtype=np.float64)
            self._target_dof_pos[:] = start + alpha * (goal - start)
        self._target_blend_step += 1
        return self._target_dof_pos

    def _compute_velocity(self, inputs):
        vel_target = inputs[0] if len(inputs) > 0 else None
        qvel = inputs[1] if len(inputs) > 1 else None
        if vel_target is None or qvel is None:
            raise ValueError(
                "[PDProcessor] Missing vel_target or qvel input for velocity PD"
            )
        tau = self._vel_kps64 * (
            np.asarray(vel_target, dtype=np.float64)
            - np.asarray(qvel, dtype=np.float64)
        )
        finite = np.isfinite(tau)
        if not finite.all():
            raise ValueError(
                f"[PDProcessor] computed tau[{_first_bad_index(~finite)}] is not finite"
            )
        np.clip(tau, self._effort_min64, self._effort_max64, out=tau)
        self.output_buffers[0][:] = tau

    def _compute_effort(self, inputs):
        target_tau = inputs[0] if len(inputs) > 0 else None
        if target_tau is None:
            raise ValueError(
                "[PDProcessor] Missing target_tau input for effort control"
            )
        values = np.asarray(target_tau, dtype=np.float64)
        self.output_buffers[0][:] = np.clip(
            values, self._effort_min64, self._effort_max64
        )

    def get_substep_pd(self):
        """Live substep PD view for the physics layer (TS ``getSubstepPd``).

        Returns ``None`` unless this is a position PD; the arrays are views
        of the live state, so the caller re-reads them per physics substep.
        Optional index keys are present only when configured.
        """
        if self._pd_type != "position":
            return None
        view = {
            "target": self._target_dof_pos,
            "kps": self._controller.kps,
            "kds": self._controller.kds,
            "effort_min": self._controller.effort_min,
            "effort_max": self._controller.effort_max,
        }
        if self._qpos_indices is not None:
            view["qpos_indices"] = self._qpos_indices
        if self._qvel_indices is not None:
            view["qvel_indices"] = self._qvel_indices
        if self._ctrl_indices is not None:
            view["ctrl_indices"] = self._ctrl_indices
        return view
