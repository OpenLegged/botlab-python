"""Dependency probing and selection for the botlab local runtime.

This module must stay importable without any runtime extras installed: the
decision functions are pure and the import probe runs the actual imports.
Failure is explicit — missing dependencies raise with the install command,
CUDA falls back to CPU only with a visible warning (a documented contract,
never a silent default).
"""

import importlib

RUNTIME_REQUIREMENTS = ("numpy", "mujoco", "onnxruntime", "websockets")

_INSTALL_COMMAND = 'pip install "botlab[runtime]"'


class RuntimeDependenciesError(RuntimeError):
    """Local runtime extras are missing; the offline host cannot run physics."""


def describe_missing(missing):
    if not missing:
        return None
    names = ", ".join(missing)
    return (
        f"local simulation needs {names}, which are not installed; "
        f"install the runtime extras with: {_INSTALL_COMMAND}"
    )


def select_execution_providers(available, requested="auto"):
    """Pick onnxruntime execution providers, preferring CUDA when present.

    Returns (providers, warning); warning is None unless the user explicitly
    asked for CUDA and it is unavailable, in which case the fallback to CPU
    is reported so it can be surfaced, not swallowed.
    """
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    else:
        providers.append("CPUExecutionProvider")

    warning = None
    if requested == "cuda" and "CUDAExecutionProvider" not in providers:
        warning = (
            "CUDAExecutionProvider requested but not available in this "
            "onnxruntime build; falling back to CPUExecutionProvider. Install "
            "onnxruntime-gpu with a matching CUDA/cuDNN for GPU inference."
        )
    return providers, warning


class _ProbeResult:
    def __init__(self, available, missing):
        self.available = available
        self.missing = missing


def probe_runtime_deps():
    available = []
    missing = []
    for name in RUNTIME_REQUIREMENTS:
        try:
            importlib.import_module(name)
            available.append(name)
        except ImportError:
            missing.append(name)
    return _ProbeResult(available, missing)


def require_runtime_deps(available, missing):
    report = describe_missing(missing)
    if report is not None:
        raise RuntimeDependenciesError(report)
    return available
