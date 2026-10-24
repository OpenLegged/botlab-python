"""Declaration and packaging constants shared by the host, project loader and release."""

DEFAULT_CONTROL_HZ = 50
DEFAULT_PHYSICS_HZ = 200

RESOURCE_KINDS = frozenset({"robot", "model", "scene", "sensor", "motion", "mesh"})
MODEL_PROVIDERS = frozenset({"auto", "webgpu", "webnn", "wasm"})

PRESET_FILE_SUFFIX = ".preset.yaml"

# Node types the release validator accepts (SimulationEngine ProcessorRegistry).
SUPPORTED_NODE_TYPES = frozenset({
    "obs-qpos", "obs-qvel", "obs-imu", "obs-constant", "obs-base-height",
    "obs-base-lin-vel", "obs-base-ang-vel", "obs-projected-gravity", "obs-commands",
    "obs-dof-pos", "obs-dof-vel", "obs-actions", "obs-camera", "obs-mujoco-query",
    "obs-height-scan",
    "proc-math", "proc-history", "proc-mux", "proc-demux", "proc-clamp",
    "proc-reorder", "proc-resize", "proc-reroute", "proc-pd",
    "proc-motion-reference", "proc-motion-sample", "proc-policy-input",
    "proc-onehot-index",
    "proc-action-hook", "proc-delay", "proc-rule-index", "proc-one-hot",
    "proc-compare", "proc-logic", "proc-where", "proc-state-machine",
    "proc-phase-clock", "proc-clock", "proc-spatial-math", "proc-signal-math",
    "proc-signal-noise", "proc-signal-filter", "proc-vector-math",
    "proc-fsm", "proc-mode-switch", "proc-select",
    "proc-mode-select", "proc-signal-send", "proc-signal-receive",
    "model-onnx", "out-action", "out-mujoco", "out-scope", "out-logger",
    "scene-robot", "scene-terrain",
})

MOTION_REQUIRED_FILES = (
    "body_pos_w.bin",
    "body_quat_w.bin",
    "body_lin_vel_w.bin",
    "body_ang_vel_w.bin",
    "joint_pos.bin",
    "joint_vel.bin",
    "fps.bin",
)
