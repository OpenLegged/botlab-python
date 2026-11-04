"""Killable inference process; target management stays alive after backend faults."""
from __future__ import annotations

import math
import multiprocessing
import threading
from pathlib import Path

from .backends import create_backend
from .protocol import HilError


def _serve(connection, backend_name):
    backend = None
    handles = {}
    try:
        backend = create_backend(backend_name)
        connection.send({"ok": True, "value": backend.describe()})
        while True:
            op, args = connection.recv()
            try:
                if op == "load":
                    handle, inputs, outputs = backend.load(Path(args[0]))
                    identity = max(handles, default=0) + 1
                    handles[identity] = handle
                    value = identity, inputs, outputs
                elif op == "infer":
                    value = backend.infer(handles[args[0]], args[1])
                elif op == "close":
                    handle = handles.pop(args[0], None)
                    if handle is not None:
                        backend.close(handle)
                    value = None
                else:
                    raise HilError("Unknown inference worker operation")
                connection.send({"ok": True, "value": value})
            except Exception as exc:
                connection.send({"ok": False, "error": str(exc) or type(exc).__name__})
    except (EOFError, OSError):
        pass
    except Exception as exc:
        try:
            connection.send({"ok": False, "error": str(exc) or type(exc).__name__})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class WorkerBackend:
    """Backend-compatible proxy used by the CLI, with opaque generation handles."""

    def __init__(self, name, *, inference_timeout=3.0, load_timeout=30.0):
        for value in (inference_timeout, load_timeout):
            if not math.isfinite(value) or value <= 0:
                raise HilError("Worker timeouts must be finite and positive")
        self.name = name
        self.inference_timeout = inference_timeout
        self.load_timeout = load_timeout
        self.lock = threading.Lock()
        self.process = self.connection = None
        self.generation = 0
        self.metadata = None
        with self.lock:
            self._start()

    def _read(self, connection, timeout):
        if not connection.poll(timeout):
            raise HilError("Inference worker timed out; its process was terminated")
        value = connection.recv()
        if not value.get("ok"):
            raise HilError(value.get("error") or "Inference worker failed")
        return value["value"]

    def _start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_serve, args=(child, self.name), daemon=True)
        self.process, self.connection = process, parent
        self.generation += 1
        try:
            process.start()
            child.close()
            self.metadata = self._read(parent, self.load_timeout)
        except Exception:
            child.close()
            self.reset()
            raise

    def _call(self, op, args, timeout, generation=None):
        with self.lock:
            if generation is not None and generation != self.generation:
                raise HilError("Inference worker session has been released")
            if self.process is None:
                if generation is not None:
                    raise HilError("Inference worker session has been released")
                self._start()
            connection = self.connection
            current = self.generation
            try:
                connection.send((op, args))
                result = self._read(connection, timeout)
                if current != self.generation:
                    raise HilError("Inference worker session has been released")
                return result
            except (OSError, EOFError, TypeError, HilError) as exc:
                self.reset()
                raise HilError(str(exc) or "Inference worker disconnected") from exc

    def describe(self):
        return dict(self.metadata)

    def load(self, path):
        identity, inputs, outputs = self._call("load", [str(path)], self.load_timeout)
        return (self.generation, identity), inputs, outputs

    def infer(self, handle, inputs):
        generation, identity = handle
        return self._call("infer", [identity, inputs], self.inference_timeout, generation)

    def close(self, handle):
        generation, identity = handle
        if generation == self.generation and self.process is not None:
            self._call("close", [identity], 1.0, generation)

    def reset(self):
        # Deliberately does not acquire the request lock: Stop can interrupt a
        # hung native call and unblock an in-flight request in another thread.
        process, connection = self.process, self.connection
        self.process = self.connection = None
        self.generation += 1
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.3)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.3)
        if connection is not None:
            connection.close()
