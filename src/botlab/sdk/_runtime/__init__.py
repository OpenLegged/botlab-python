"""Private local-runtime implementation for botlab (native CPython only).

Everything here is reachable only when the `runtime` extras (numpy, mujoco,
onnxruntime) are installed; the offline host keeps raising explicit errors
when they are not. No names from this package are re-exported by `botlab`.
"""
