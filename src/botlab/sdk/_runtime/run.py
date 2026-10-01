"""Local project runner: `python -m botlab.run <project-dir>`.

Wires the third host state before the project's own ``main.py`` executes:
compile declarations, build native MuJoCo physics from the primary robot
resource, install the local runtime session (display on when
``BOTLAB_DISPLAY=1``), then run ``main.py`` — its ``Simulation.run(...)``
registers the controller into the session — and finally drive the control
loop. ``BOTLAB_STEPS`` caps control steps (headless runs and tests);
without it the loop runs until the physics adapter stops it.
"""

import os
import runpy
import sys


def _require_runtime():
    from ._deps import probe_runtime_deps, require_runtime_deps

    probe = probe_runtime_deps()
    require_runtime_deps(probe.available, probe.missing)


def _project_directory(argv):
    if len(argv) != 2:
        print("usage: python -m botlab.run <project directory>", file=sys.stderr)
        raise SystemExit(2)
    directory = os.path.abspath(argv[1])
    if not os.path.isdir(directory):
        print(f"[botlab.run] project directory not found: {directory}", file=sys.stderr)
        raise SystemExit(2)
    return directory


def _declare_project(directory):
    """Import controller.py only: declarations land in botlab_host state.

    main.py runs later, with the session installed, so its Simulation.run
    registers into the local runtime; importing controller once here means
    main.py's own import finds it cached and no resource is declared twice.
    """
    import importlib

    import botlab_host

    sys.path.insert(0, directory)
    os.chdir(directory)
    botlab_host.project_state().reset(directory)
    importlib.import_module("controller")
    return botlab_host.project_state()


def _resolve_source(directory, source):
    """Template-relative first, then the shared assets root (public/…).

    BotLab templates reference shared model trees (public/models/…) with
    paths that only resolve from the assets root; local projects keep
    everything inside their directory.
    """
    candidates = [
        source if os.path.isabs(source) else os.path.join(directory, source),
    ]
    if not os.path.isabs(source):
        assets_root = os.path.dirname(os.path.dirname(directory))
        candidates.append(os.path.join(assets_root, source))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        f"[botlab.run] robot model source \"{source}\" was not found under "
        f"{directory} or the shared assets root"
    )


def _primary_robot(state):
    resources = [item for item in state.manifest()["resources"] if item["kind"] == "robot"]
    if not resources:
        raise RuntimeError(
            "[botlab.run] the project declares no robot resource; nothing to simulate"
        )
    if len(resources) > 1:
        raise RuntimeError(
            "[botlab.run] multi-robot projects are not supported by the local "
            "runtime yet"
        )
    robot = resources[0]
    source = _resolve_source(state.directory or os.getcwd(), robot["source"])
    return state, robot, source


def _load_model_with_transform(source, transform, mujoco_module):
    """Compile the entry XML with ``transform`` applied to every XML in its
    include chain — the browser does the same per VFS file (transforms.all).

    XMLs land in a temp directory with the include layout preserved; every
    other file of the model tree is symlinked so meshes still resolve.
    """
    import os
    import re
    import shutil
    import tempfile

    model_dir = os.path.dirname(os.path.abspath(source))
    tmp_dir = tempfile.mkdtemp(prefix="botlab-model-")
    try:
        written = set()

        def process(xml_path, rel_name):
            if rel_name in written:
                return
            written.add(rel_name)
            with open(xml_path, "r", encoding="utf-8") as handle:
                text = handle.read()
            text = transform(text)
            target = os.path.join(tmp_dir, rel_name)
            os.makedirs(os.path.dirname(target) or tmp_dir, exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)
            for include in re.finditer(r"<include\s+file=\"([^\"]+)\"", text):
                include_rel = os.path.normpath(include.group(1)).replace(os.sep, "/")
                include_abs = os.path.normpath(os.path.join(model_dir, include_rel))
                if os.path.isfile(include_abs):
                    process(include_abs, include_rel)

        process(os.path.abspath(source), os.path.basename(source))

        for root, _dirs, files in os.walk(model_dir):
            for name in files:
                absolute = os.path.join(root, name)
                rel = os.path.relpath(absolute, model_dir).replace(os.sep, "/")
                if rel in written:
                    continue
                destination = os.path.join(tmp_dir, rel)
                os.makedirs(os.path.dirname(destination) or tmp_dir, exist_ok=True)
                if not os.path.lexists(destination):
                    os.symlink(absolute, destination)

        return mujoco_module.MjModel.from_xml_path(
            os.path.join(tmp_dir, os.path.basename(source))
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_physics(source, robot_config, resources=None, path_resolver=None):
    import mujoco

    from ._physics import MujocoPhysics, inject_auto_actuators

    auto_actuators = robot_config.get("autoActuators")
    model = _load_model_with_transform(
        source,
        lambda xml: inject_auto_actuators(xml, auto_actuators, label=source),
        mujoco,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoPhysics(
        model,
        data,
        mujoco,
        resources=resources,
        path_resolver=path_resolver,
        initial_qpos=robot_config.get("initialQpos"),
        initial_qvel=robot_config.get("initialQvel"),
    )


def _display_port(environ):
    raw = environ.get("BOTLAB_DISPLAY_PORT")
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip()
    if not text.isascii() or not text.isdecimal():
        raise RuntimeError(
            "[botlab.run] BOTLAB_DISPLAY_PORT must be an integer from 1 to 65535"
        )
    port = int(text)
    if port < 1 or port > 65535:
        raise RuntimeError(
            "[botlab.run] BOTLAB_DISPLAY_PORT must be an integer from 1 to 65535"
        )
    return port


def main(argv=None):
    args = list(sys.argv if argv is None else argv)
    directory = _project_directory(args)
    _require_runtime()
    original_directory = os.getcwd()
    original_path = list(sys.path)
    display = None
    session = None
    botlab_host = None
    primary_error = None
    try:
        state = _declare_project(directory)
        _project, robot, source = _primary_robot(state)
        manifest = state.manifest()
        resolve = lambda resource_source: _resolve_source(directory, resource_source)
        physics = _build_physics(
            source,
            robot["config"],
            resources=manifest["resources"],
            path_resolver=resolve,
        )

        if os.environ.get("BOTLAB_DISPLAY", "").strip() not in ("", "0"):
            from ._display import DisplayServer

            display = DisplayServer(
                metadata={
                    "project": manifest["name"],
                    "controlHz": manifest["control_hz"],
                    "physicsHz": manifest["physics_hz"],
                    "qposDim": int(physics.data.qpos.shape[0]),
                    "qvelDim": int(physics.data.qvel.shape[0]),
                },
                port=_display_port(os.environ),
            )
            display.start()
            print(
                f"[botlab.run] display ready: {display.url}",
                file=sys.stderr,
            )

        import botlab_host as active_host

        from ._host_runtime import LocalRuntimeSession, PooledModelRuntime

        botlab_host = active_host
        session = LocalRuntimeSession(
            physics,
            display=display,
            models=PooledModelRuntime(path_resolver=resolve),
            control_hz=manifest["control_hz"],
            physics_hz=manifest["physics_hz"],
            path_resolver=resolve,
        )
        botlab_host.set_runtime_session(session)

        runpy.run_path(os.path.join(directory, "main.py"), run_name="__main__")

        steps_env = os.environ.get("BOTLAB_STEPS", "").strip()
        max_steps = int(steps_env) if steps_env else None
        completed = session.run(max_steps)
        print(f"[botlab.run] completed {completed} control steps", file=sys.stderr)
        return 0
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        if botlab_host is not None:
            botlab_host.set_runtime_session(None)
        try:
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    cleanup_errors.append(error)
            if display is not None:
                try:
                    display.stop()
                except Exception as error:
                    cleanup_errors.append(error)
        finally:
            os.chdir(original_directory)
            sys.path[:] = original_path
        if cleanup_errors:
            message = "[botlab.run] cleanup failed: " + "; ".join(
                str(error) for error in cleanup_errors
            )
            if primary_error is not None:
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(message)
                else:  # Python 3.9/3.10: preserve the primary traceback.
                    print(message, file=sys.stderr)
            elif len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            else:
                raise RuntimeError(message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[botlab.run] interrupted", file=sys.stderr)
        raise SystemExit(130) from None
