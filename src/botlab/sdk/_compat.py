"""Pyodide/CPython compatibility: JS proxies only exist in the browser runtime."""


try:
    from pyodide.ffi import create_proxy
except ImportError:
    # The static CPython host (botlab CLI) never wires JS callbacks; it only
    # collects declarations and the compiled plan, so proxies stay unused there.
    create_proxy = None
