"""ONNX session pool and execution-provider request mapping."""

import os

from .. import _deps

_PROVIDER_CHOICES = ("auto", "cpu", "cuda", "wasm", "webgpu", "webnn")

_NATIVE_PROVIDER_MAPPING = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    # Public Model providers describe browser execution. Preserve their intent
    # without expanding that API with native-only names: WASM is CPU, WebGPU is
    # the native CUDA path, and WebNN has no native ORT equivalent.
    "wasm": "cpu",
    "webgpu": "cuda",
    "webnn": "auto",
}

# onnxruntime names ONNX element types 'float'/'double'; the TypeScript engine
# calls the same metadata 'float32'/'float64'.
_ONNX_ELEMENT_TYPE_NAMES = {"float": "float32", "double": "float64"}


def _normalized_provider(raw):
    if raw is None or raw == "":
        return "auto"
    if not isinstance(raw, str) or raw.strip().lower() not in _PROVIDER_CHOICES:
        raise ValueError(
            "[ModelProcessor] executionProvider must be one of "
            f"{list(_PROVIDER_CHOICES)} when provided"
        )
    return raw.strip().lower()


def _native_provider_request(raw):
    requested = _normalized_provider(raw)
    warning = None
    if requested == "webnn":
        warning = (
            'browser model provider "webnn" has no native onnxruntime '
            "equivalent; using automatic CUDA/CPU selection"
        )
    return _NATIVE_PROVIDER_MAPPING[requested], warning


def _io_metadata(args, kind, resource_id):
    """Normalized ``{name, is_tensor, type, shape}`` entries for one session side."""
    if not args:
        raise ValueError(
            f'[ModelProcessor] ONNX model resource "{resource_id}" has no {kind} names'
        )
    entries = []
    for arg in args:
        type_name = str(getattr(arg, "type", "") or "")
        is_tensor = type_name.startswith("tensor(") and type_name.endswith(")")
        element_type = type_name[7:-1] if is_tensor else type_name
        entries.append(
            {
                "name": str(arg.name),
                "is_tensor": is_tensor,
                "type": _ONNX_ELEMENT_TYPE_NAMES.get(element_type, element_type),
                "shape": list(arg.shape) if getattr(arg, "shape", None) is not None else None,
            }
        )
    return entries


def _default_session_factory(model_bytes, providers):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(
        model_bytes, sess_options=options, providers=providers
    )


class OnnxSessionPool:
    """One onnxruntime session per model resource id, loaded from file bytes.

    The browser preloads project model resources into a registry keyed by
    resource id; this pool is the local equivalent. The first ``get`` for an id
    reads the model file (explicit failure when it is missing or will not load
    as ONNX) and creates a session with ``_deps.select_execution_providers``
    — a requested-but-missing CUDA provider prints the fallback warning, it is
    never silent. Later ``get`` calls return the same session whatever path is
    passed: the id is the identity, the path only feeds the initial load.
    """

    def __init__(self, session_factory=None, available_providers=None):
        self._sessions = {}
        self._session_factory = session_factory or _default_session_factory
        self._available_providers = available_providers

    def get(self, resource_id, model_path, requested="auto"):
        session = self._sessions.get(resource_id)
        if session is not None:
            return session
        requested, compatibility_warning = _native_provider_request(requested)
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(
                f'[ModelProcessor] ONNX model resource "{resource_id}" has no model path'
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'[ModelProcessor] ONNX model resource "{resource_id}" model file '
                f'not found: "{model_path}"'
            )
        with open(model_path, "rb") as handle:
            model_bytes = handle.read()
        available = self._available_execution_providers()
        if requested == "cpu":
            available = [
                provider for provider in available
                if provider != "CUDAExecutionProvider"
            ]
        providers, selection_warning = _deps.select_execution_providers(
            available, requested
        )
        for warning in (compatibility_warning, selection_warning):
            if warning:
                print(f'[ModelProcessor] ONNX model resource "{resource_id}": {warning}')
        try:
            session = self._session_factory(model_bytes, providers)
        except Exception as error:
            raise RuntimeError(
                f'[ModelProcessor] ONNX model resource "{resource_id}" failed to '
                f'load from "{model_path}": {error}'
            ) from error
        self._sessions[resource_id] = session
        return session

    def _available_execution_providers(self):
        if self._available_providers is not None:
            return list(self._available_providers)
        import onnxruntime

        return onnxruntime.get_available_providers()

    def clear(self):
        """Drop every cached session; later ``get`` calls reload from disk.

        ``botlab-run`` loads one project per process so the cache never
        goes stale, but repeatable open/close cycles (``botlab.open_headless``)
        re-fetch the same resource ids against different files — a run that
        closes must not leave its sessions behind for the next one.
        """
        self._sessions = {}


_SHARED_SESSION_POOL = None


def shared_session_pool():
    """The process-wide pool model nodes reuse, like the browser's registry."""
    global _SHARED_SESSION_POOL
    if _SHARED_SESSION_POOL is None:
        _SHARED_SESSION_POOL = OnnxSessionPool()
    return _SHARED_SESSION_POOL
