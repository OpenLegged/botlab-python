"""Internal native runner behind ``botlab run`` and ``botlab run --headless``.

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


def _parse_args(argv):
    if len(argv) < 2:
        print("usage: botlab run <project directory> [--headless] [--report PATH]",
              file=sys.stderr)
        raise SystemExit(2)
    positional = []
    report = None
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--report":
            if index + 1 >= len(argv):
                print("[botlab.run] --report needs a path", file=sys.stderr)
                raise SystemExit(2)
            report = argv[index + 1]
            index += 2
            continue
        if arg == "--headless":
            index += 1
            continue
        positional.append(arg)
        index += 1
    if len(positional) != 1:
        print("usage: botlab run <project directory> [--headless] [--report PATH]",
              file=sys.stderr)
        raise SystemExit(2)
    directory = os.path.abspath(positional[0])
    if not os.path.isdir(directory):
        print(f"[botlab.run] project directory not found: {directory}", file=sys.stderr)
        raise SystemExit(2)
    return directory, report


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


def _robots(state):
    resources = [item for item in state.manifest()["resources"] if item["kind"] == "robot"]
    if not resources:
        raise RuntimeError(
            "[botlab.run] the project declares no robot resource; nothing to simulate"
        )
    return resources


def _robot_format(path):
    lower = str(path).lower()
    if lower.endswith(".mjb"):
        return "mjb"
    if lower.endswith(".urdf"):
        return "urdf"
    return "xml"


def _load_model_with_transform(source, transform, mujoco_module, entry_transform=None):
    """Stage only the model dependency closure, preserving sibling directories."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from .._mjcf import mjcf_dependencies

    source = Path(os.path.abspath(source))
    documents, assets = mjcf_dependencies(
        source, transform=transform, entry_transform=entry_transform,
    )
    common = Path(os.path.commonpath([str(path.parent) for path in set(documents) | assets]))
    with tempfile.TemporaryDirectory(prefix="botlab-model-") as temporary:
        root = Path(temporary)
        for path, document in documents.items():
            target = root / path.relative_to(common)
            target.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(document).write(target, encoding="utf-8")
        for path in assets - set(documents):
            target = root / path.relative_to(common)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        return mujoco_module.MjModel.from_xml_path(str(root / source.relative_to(common)))


def _build_physics(state, directory, path_resolver):
    import mujoco

    from ._compose import compose_entry_xml, load_extra_robot
    from ._physics import MujocoPhysics, inject_auto_actuators

    robots = _robots(state)
    primary = robots[0]
    primary_config = primary.get("config") or {}
    source = _resolve_source(directory, primary["source"])

    extras = []
    if len(robots) > 1:
        if _robot_format(source) != "xml":
            raise RuntimeError(
                f'[botlab.run] robot "{primary["id"]}" is not MJCF; '
                "composed projects require MJCF for every robot"
            )
        for index, robot in enumerate(robots[1:], start=1):
            extra_source = _resolve_source(directory, robot["source"])
            if _robot_format(extra_source) != "xml":
                raise RuntimeError(
                    f'[botlab.run] extra robot "{robot["id"]}" uses '
                    f'{_robot_format(extra_source).upper()}; composed robots require MJCF'
                )
            extras.append(
                load_extra_robot(
                    extra_source,
                    f"r{index}_",
                    (robot.get("config") or {}).get("autoActuators"),
                    label=robot["id"],
                )
            )

    auto_actuators = primary_config.get("autoActuators")

    def entry_transform(xml):
        return compose_entry_xml(xml, extras)

    model = _load_model_with_transform(
        source,
        lambda xml: inject_auto_actuators(xml, auto_actuators, label=source),
        mujoco,
        entry_transform=entry_transform if extras else None,
    )
    # 物理步长以 project(physics_hz=…) 为准（浏览器 worker 同此契约）；XML 里
    # 显式声明了不同 timestep 是配置冲突，明确报错而不是跑错速度。
    physics_hz = state.manifest()["physics_hz"]
    target_timestep = 1.0 / physics_hz
    current_timestep = float(model.opt.timestep)
    if abs(current_timestep - target_timestep) > 1e-12:
        if abs(current_timestep - 0.002) > 1e-12:
            raise RuntimeError(
                f"[botlab.run] the model declares timestep={current_timestep} but "
                f"project physics_hz={physics_hz} implies {target_timestep}; "
                "align the MJCF <option> or the project declaration"
            )
        model.opt.timestep = target_timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manifest_resources = state.manifest()["resources"]
    if not extras:
        return MujocoPhysics(
            model,
            data,
            mujoco,
            resources=manifest_resources,
            path_resolver=path_resolver,
            initial_qpos=primary_config.get("initialQpos"),
            initial_qvel=primary_config.get("initialQvel"),
        )
    robot_states = [
        {
            **(robot.get("config") or {}),
            "robotId": robot["id"],
            "namespace": "" if index == 0 else f"r{index}_",
        }
        for index, robot in enumerate(robots)
    ]
    return MujocoPhysics(
        model,
        data,
        mujoco,
        resources=manifest_resources,
        path_resolver=path_resolver,
        robot_states=robot_states,
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
    directory, report_path = _parse_args(args)
    if report_path is None:
        report_path = os.environ.get("BOTLAB_REPORT") or None
    _require_runtime()
    original_directory = os.getcwd()
    original_path = list(sys.path)
    display = None
    session = None
    botlab_host = None
    primary_error = None
    recorder = None
    try:
        state = _declare_project(directory)
        manifest = state.manifest()
        def resolve(resource_source):
            return _resolve_source(directory, resource_source)
        physics = _build_physics(state, directory, resolve)

        if os.environ.get("BOTLAB_DISPLAY", "").strip() not in ("", "0"):
            from ._display import DisplayServer
            from .mujoco_scene import MujocoSceneProvider

            display = DisplayServer(
                metadata={
                    "project": manifest["name"],
                    "controlHz": manifest["control_hz"],
                    "physicsHz": manifest["physics_hz"],
                    "qposDim": int(physics.data.qpos.shape[0]),
                    "qvelDim": int(physics.data.qvel.shape[0]),
                },
                port=_display_port(os.environ),
                scene_provider=MujocoSceneProvider(
                    physics.model, physics.data, physics._mujoco, manifest["name"]
                ),
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
        if report_path is not None:
            from ._report import RunRecorder

            recorder = RunRecorder(
                report_path,
                manifest["name"],
                manifest["control_hz"],
                manifest["physics_hz"],
                physics._robot_ids,
            )
            session.set_recorder(recorder)

        runpy.run_path(os.path.join(directory, "main.py"), run_name="__main__")

        steps_env = os.environ.get("BOTLAB_STEPS", "").strip()
        max_steps = int(steps_env) if steps_env else None
        completed = session.run(max_steps)
        if recorder is not None:
            print(f"[botlab.run] run report: {recorder.write()}", file=sys.stderr)
        print(f"[botlab.run] completed {completed} control steps", file=sys.stderr)
        return 0
    except BaseException as error:
        primary_error = error
        if recorder is not None:
            # 中断/失败也落盘：已完成的部分正是排障要看的。
            try:
                recorder.write()
            except Exception as report_error:
                cleanup_note = f"[botlab.run] writing the run report failed: {report_error}"
                if hasattr(error, "add_note"):
                    error.add_note(cleanup_note)
                else:
                    print(cleanup_note, file=sys.stderr)
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
