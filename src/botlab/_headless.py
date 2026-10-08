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
        run.close()

:func:`load_motion` reads either motion layout (.npz archive or .bin folder)
for reference data; the runtime accepts both, so verification no longer has
to care which one the project ships.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .sdk._runtime_types import Frame


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

    def __init__(self, directory, manifest, physics, session, path_resolver,
                 original_directory=None, original_path=None):
        self._directory = directory
        self._manifest = manifest
        self.physics = physics
        self.session = session
        self._path_resolver = path_resolver
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
        session = self.session
        if session._reset_callback is not None:
            session._call_maybe_async(session._reset_callback())
        if session._executor is not None:
            from botlab.sdk._runtime._observations import clear_pose_velocity_states

            clear_pose_velocity_states()
            session._executor.reset()
        self.physics.reset()
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
        dt = 1.0 / self.control_hz
        frame = Frame(dt, self._step_index * dt, self._step_index)
        session._current_frame = frame
        if session._executor is not None:
            import numpy as np

            self.physics.fill_observations(
                session._executor,
                last_action=self._last_action,
                parameters=session._fill_parameters,
            )
            session._executor.step(frame=frame)
            action = np.array(session._policy_action(), dtype=np.float32)
            self._last_action = action
            ctrl, substep_pd = session._control_payload()
            control = None if ctrl is None else np.array(ctrl, dtype=np.float32)
            self.physics.step_substeps(session.decimation, ctrl=ctrl, substep_pd=substep_pd)
            self.physics.after_control_step()
            self._step_index += 1
            return StepRecord(frame, action, control, substep_pd)

        if session._step_callback is None:
            raise RuntimeError(
                "headless run has no controller; the project's main.py never called "
                "Simulation.run()"
            )
        session._custom_step_actions.clear()
        session._custom_step_active = True
        try:
            session._call_maybe_async(session._step_callback(dt, frame.time, frame.index))
        finally:
            session._custom_step_active = False
        missing = [
            robot_id for robot_id in (session._step_robot_ids or [])
            if robot_id not in session._custom_step_actions
        ]
        if missing:
            raise RuntimeError(
                "[botlab] Controller.step must produce exactly one action for: "
                + ", ".join(missing)
            )
        self.physics.step_substeps(session.decimation)
        self.physics.after_control_step()
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
            # Evict the project's modules so the next open_headless (another
            # project, or this one again) re-imports controller.py and its
            # declarations run against the fresh session — the same eviction
            # the offline loader performs before importing.
            for name in ("controller", "main"):
                sys.modules.pop(name, None)
            for entry in self._directory.iterdir():
                if entry.is_file() and entry.suffix == ".py":
                    sys.modules.pop(entry.stem, None)
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

    from botlab.sdk._runtime._host_runtime import LocalRuntimeSession, PooledModelRuntime
    from botlab.sdk._runtime.run import (
        _build_physics,
        _declare_project,
        _primary_robot,
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
        for name in ("controller", "main"):
            sys.modules.pop(name, None)
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix == ".py":
                sys.modules.pop(entry.stem, None)
        state = _declare_project(str(directory))
        _project, robot, source = _primary_robot(state)
        manifest = state.manifest()
        resolve = lambda item: _resolve_source(str(directory), item)
        physics = _build_physics(
            source, robot["config"], resources=manifest["resources"], path_resolver=resolve
        )
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
            directory, manifest, physics, session, resolve,
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


def load_motion(source) -> dict:
    """Read a motion asset in either layout: a ``.npz`` archive or a ``.bin``
    folder. Returns the motion data dict the runtime's ``Motion.load()``
    produces (frame/body/dof counts, fps, flat float32 channel buffers, and
    ``joint_names``/``body_names`` when the format declares them).
    """
    from botlab.sdk._runtime._motion import (
        MOTION_REQUIRED_FILES,
        load_motion_folder,
        load_motion_npz,
    )

    path = Path(source)
    if path.suffix.lower() == ".npz":
        return load_motion_npz(path.read_bytes(), str(path))
    files = {name: (path / name).read_bytes() for name in MOTION_REQUIRED_FILES}
    return load_motion_folder(files, str(path))
