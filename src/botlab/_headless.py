"""Public headless rollouts: step a compiled project and record every step.

``python -m botlab.run`` drives a project start to finish; verification
scripts need the same loop under their own control — step by step, with the
policy action, the control payload, and the live physics state in hand, plus
parameter changes between steps (the headless counterpart of pressing a key).
:func:`open_headless` assembles the exact runtime that entry uses — same
declaration reset, same physics wiring, same session and model pool — and
hands it back unrun:

.. code-block:: python

    with open_headless(project_dir) as run:
        run.set_parameter("clip_command", 2)
        run.reset()
        for step in run.steps(320):
            error = metric(run.physics.data, step.frame.index)

:func:`load_motion` reads either motion layout (.npz archive or .bin folder)
for reference data; the runtime accepts both, so verification no longer has
to care which one the project ships.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._project import evict_project_modules


class StepRecord:
    """What one control step produced, snapshotted after physics advanced.

    ``action`` is the policy's raw action (the ``Signal.action()`` output),
    ``control`` the raw ctrl payload when the plan writes one directly;
    ``substep_pd`` is the plan's live substep PD view, already consumed by
    the step that produced it. Custom-controller steps carry ``None`` for
    all three — their actions go through ``Robot.apply_action`` instead.
    """

    def __init__(self, frame, action, control, substep_pd):
        self.frame = frame
        self.action = action
        self.control = control
        self.substep_pd = substep_pd


class HeadlessRun:
    """A live native runtime for one project, driven one control step at a time."""

    def __init__(self, directory, manifest, physics, session,
                 original_directory=None, original_path=None):
        self._directory = directory
        self._manifest = manifest
        self.physics = physics
        self.session = session
        self._step_index = 0
        self._last_action = None
        self._closed = False
        self._reset_done = False
        # The assembly chdir'd into the project; close() returns the caller to
        # where open_headless was called from, like the runner's finally block.
        self._original_directory = original_directory or os.getcwd()
        self._original_path = list(sys.path if original_path is None else original_path)

    # -- configuration -----------------------------------------------------

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def control_hz(self) -> int:
        return self.session._control_hz

    @property
    def physics_hz(self) -> int:
        return self.session._physics_hz

    def set_parameter(self, name, value) -> None:
        """Override a declared parameter from the next control step on."""
        import botlab_host

        botlab_host.project_state().set_parameter(name, value)

    # -- driving ------------------------------------------------------------

    def reset(self) -> None:
        """Reset like the runner does before its first step.

        Called automatically before the first :meth:`step` when skipped;
        call it directly to re-run a rollout (parameters overrides survive).
        """
        self.session.reset_for_run()
        self._step_index = 0
        self._last_action = None
        self._reset_done = True

    def step(self) -> StepRecord:
        """Advance exactly one control step and return what it produced."""
        if self._closed:
            raise RuntimeError("headless run is closed")
        if not self._reset_done:
            self.reset()
        session = self.session
        if session._executor is not None:
            import numpy as np

            frame, action, ctrl, substep_pd = session.plan_step(
                self._step_index, self._last_action
            )
            self._last_action = action
            control = None if ctrl is None else np.array(ctrl, dtype=np.float32)
            self._step_index += 1
            return StepRecord(frame, action, control, substep_pd)

        if session._step_callback is None:
            raise RuntimeError(
                "headless run has no controller; the project's main.py never called "
                "Simulation.run()"
            )
        frame = session.custom_step(self._step_index)
        self._step_index += 1
        return StepRecord(frame, None, None, None)

    def steps(self, count, on_step=None) -> list[StepRecord]:
        """Run up to ``count`` control steps, stopping early if physics asks."""
        records = []
        for _ in range(count):
            record = self.step()
            records.append(record)
            if on_step is not None:
                on_step(record)
            if self.physics.should_stop():
                break
        return records

    # -- assets ---------------------------------------------------------------

    def load_motion(self, source) -> dict:
        """Read a motion asset resolving relative sources against the project
        directory — works regardless of the caller's CWD."""
        return load_motion(source, project=self._directory)

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        import botlab_host

        botlab_host.set_runtime_session(None)
        try:
            self.session.close()
        finally:
            # Drop this run's cached ONNX sessions: the pool is keyed by
            # resource id (not file contents), so leaving them behind would
            # make the next open_headless silently re-run this project's
            # models — even for the same resource id re-exported on disk.
            try:
                from botlab._runtime._processors._model import shared_session_pool

                shared_session_pool().clear()
            except Exception:
                pass  # teardown must never fail on pool cleanup
            # Evict the project's modules so the next open_headless (another
            # project, or this one again) re-imports controller.py and its
            # declarations run against the fresh session.
            evict_project_modules(self._directory)
            os.chdir(self._original_directory)
            sys.path[:] = self._original_path

    def __enter__(self) -> HeadlessRun:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_headless(directory) -> HeadlessRun:
    """Assemble the ``botlab-run`` runtime for a project without running it.

    Requires the runtime extras (numpy, mujoco, onnxruntime); the error names
    what is missing. The returned run owns the process-global declaration and
    runtime state — close it (or use it as a context manager) before opening
    another.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"project directory not found: {directory}")

    import botlab_host

    from botlab._runtime._host_runtime import LocalRuntimeSession, PooledModelRuntime
    from botlab._runtime.run import (
        _build_physics,
        _declare_project,
        _require_runtime,
        _resolve_source,
    )

    _require_runtime()
    original_directory = os.getcwd()
    original_path = list(sys.path)
    session = None
    try:
        # Evict cached project modules first: a previous load_project or run in
        # this process leaves "controller" imported, and re-importing the cache
        # would declare nothing against the fresh session.
        evict_project_modules(directory)
        state = _declare_project(str(directory))
        manifest = state.manifest()

        def resolve(item):
            return _resolve_source(str(directory), item)

        physics = _build_physics(state, str(directory), resolve)
        session = LocalRuntimeSession(
            physics,
            models=PooledModelRuntime(path_resolver=resolve),
            control_hz=manifest["control_hz"],
            physics_hz=manifest["physics_hz"],
            path_resolver=resolve,
        )
        botlab_host.set_runtime_session(session)

        import runpy

        runpy.run_path(str(directory / "main.py"), run_name="__main__")
        return HeadlessRun(
            directory, manifest, physics, session,
            original_directory=original_directory, original_path=original_path,
        )
    except BaseException:
        if session is not None:
            botlab_host.set_runtime_session(None)
            try:
                session.close()
            except Exception:
                pass
        os.chdir(original_directory)
        sys.path[:] = original_path
        raise


def load_motion(source, project=None) -> dict:
    """Read a motion asset in either layout: a ``.npz`` archive or a ``.bin``
    folder. Returns the motion data dict the runtime's ``Motion.load()``
    produces (frame/body/dof counts, fps, flat float32 channel buffers, and
    ``joint_names``/``body_names`` when the format declares them).

    Relative sources resolve against ``project`` (a project directory)
    first, then the CWD; pass ``project`` when calling outside a live run —
    :meth:`HeadlessRun.load_motion` forwards it for you.
    """
    from botlab._runtime._motion import (
        MOTION_METADATA_FILE,
        MOTION_REQUIRED_FILES,
        load_motion_folder,
        load_motion_npz,
    )

    from ._host import BotLabError

    raw = Path(source)
    if raw.is_absolute():
        candidates = [raw]
    elif project is not None:
        candidates = [Path(project) / raw, raw]
    else:
        candidates = [raw]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise BotLabError(
            f'[botlab] motion source "{source}" was not found (tried: {tried})'
        )
    path = existing[0]
    if path.is_dir():
        missing = [name for name in MOTION_REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise BotLabError(
                f"[botlab] motion folder {path} is missing required file(s): "
                f"{', '.join(missing)}. Motions ship either as a single .npz "
                "archive (source ending in .npz) or as a folder with the "
                f"required .bin files ({', '.join(MOTION_REQUIRED_FILES)})."
            )
        files = {name: (path / name).read_bytes() for name in MOTION_REQUIRED_FILES}
        metadata_path = path / MOTION_METADATA_FILE
        if metadata_path.is_file():
            files[MOTION_METADATA_FILE] = metadata_path.read_bytes()
        return load_motion_folder(files, str(path))
    if path.suffix.lower() == ".npz":
        return load_motion_npz(path.read_bytes(), str(path))
    raise BotLabError(
        f'[botlab] motion source "{source}" must be an NPZ file or motion folder'
    )
