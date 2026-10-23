"""Execution plugins. A factory returns an object implementing describe/load/infer/close.

load(path) returns (handle, inputs, outputs); infer(handle, named_tensors)
returns named tensors. Only explicit --backend module:factory imports plugin code.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from ._onnx import require_embedded_tensors
from .protocol import HilError


class OnnxRuntimeBackend:
    def __init__(self):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise HilError("Install numpy and onnxruntime on the target to use this backend") from exc
        self.np, self.ort = np, ort

    def describe(self):
        return {"backend": "onnxruntime", "provider": "CPUExecutionProvider", "dtypes": ["float32"]}

    def load(self, path: Path):
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        content = path.read_bytes()
        require_embedded_tensors(content)
        session = self.ort.InferenceSession(content, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        def ports(items):
            result = []
            for item in items:
                if item.type != "tensor(float)":
                    raise HilError(f"Unsupported port {item.name}: {item.type}; HIL v1 requires float32")
                result.append({"name": item.name, "dtype": "float32", "shape": list(item.shape)})
            return result
        return session, ports(session.get_inputs()), ports(session.get_outputs())

    def infer(self, handle, inputs):
        feeds = {name: self.np.asarray(v["data"], dtype=self.np.float32).reshape(v["shape"])
                 for name, v in inputs.items()}
        values = handle.run(None, feeds)
        return {meta.name: {"dtype": "float32", "shape": list(value.shape), "data": value.ravel().tolist()}
                for meta, value in zip(handle.get_outputs(), values, strict=True)}

    def close(self, handle):
        pass  # Removing the final handle releases the ORT session.


def create_backend(name: str):
    if name == "onnxruntime":
        return OnnxRuntimeBackend()
    if ":" not in name:
        raise HilError("Backend must be onnxruntime or module:factory")
    module, factory = name.split(":", 1)
    backend = getattr(importlib.import_module(module), factory)()
    for method in ("describe", "load", "infer", "close"):
        if not callable(getattr(backend, method, None)):
            raise HilError(f"Backend is missing {method}()")
    return backend
