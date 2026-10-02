# botlab-python (Python SDK + CLI)

BotLab's Python SDK and command-line interface. Declare robot projects with
`project()`, `Robot`, `Model`, `Sensor`, `Motion`, and `Scene` in
`controller.py`, express control logic as composable signals, and run them
natively on your machine with one command — real MuJoCo physics and ONNX
Runtime inference in local CPython, with the browser as a read-only viewer.

```bash
pip install "botlab-python[runtime]"
cd my-robot-project
botlab run
```

`botlab run` (the default) is the native flow: the CLI compiles the project,
executes it in a CPython subprocess (native `mujoco` + `onnxruntime` + NumPy),
and opens the browser at `embed.html?ws=…` — a viewer over the loopback
display stream. The browser renders; it never computes. `Ctrl+C` stops the
run. The startup banner prints the viewer, editor, and stream URLs; use
`--no-browser` to skip auto-opening.

`botlab run --browser` is the live editor debug loop instead: it serves the
complete BotLab editor — not a stripped-down release page — with the project
tree, Monaco editor, Console, Watch, and Metrics panels, and the browser
itself executes the project (MuJoCo WASM + ONNX Runtime Web):

1. Import `controller.py` and collect the project declarations, including
   resources, parameters, and runtime frequencies.
2. Execute `main.py` and compile the `PolicyController` signal expressions
   into a control plan.
3. Assemble an integrity-checked release in memory and serve it to the
   browser without writing an archive to disk.
4. Start a local static server and open the browser. The Vite-style startup
   banner prints clickable `Local` and `Network` links; the server binds to
   `0.0.0.0` by default so devices on the same network can connect.

The editor opens at `/`. Its localized `index.html` includes the
`window.BOTLAB_LOCAL_PROJECT` flag, so it loads the current directory from
`/local-project.json` without adding parameters to the URL. `/embed.html`
remains available for previewing the published form.

In the debug loop, edit the source and refresh the page — the editor rereads
the directory and the debug release is rebuilt without restarting the command.
In the native flow, edits take effect by restarting the run. Use `--verbose`
to show resources, plan size, and package size. Use `botlab build` when you
need a compressed `release.zip` for distribution or upload.

The CLI is project-agnostic: it works with any BotLab project that follows the
two-file `main.py` + `controller.py` contract and contains no robot- or
policy-specific behavior.

## Project structure

```text
my-robot-project/
├── main.py                 # Fixed four-line entry point
├── controller.py           # Project declarations and control expressions
├── robot.preset.yaml       # Joint order, PD gains, initial state, and calibration
└── assets/                 # Project-relative MJCF, ONNX, meshes, and other assets
```

`main.py` must use the standard four-line entry point required by compiled
controller releases:

```python
from botlab import Simulation
from controller import create_controller

sim = Simulation.from_project()
sim.run(create_controller(sim))
```

Write observations, models, and outputs in `controller.py` with the same
general-purpose components used by the browser editor. The same source runs in
both environments:

```python
from botlab import Model, PolicyController, Robot, concat, project

project("My Robot", control_hz=50, physics_hz=200)
robot = Robot("assets/robot/scene.xml", preset="robot.preset.yaml")
actor = Model("assets/models/policy.onnx")


def create_controller(sim):
    observation = concat([
        robot.projected_gravity(),
        robot.joint_position(default=robot.default_angles),
    ])
    action = actor(observation).action()
    target = robot.default_angles + robot.action_scale * action[robot.motor_order]
    return PolicyController(robot, control=target)
```

See the BotLab repository's `docs/api/` directory for the complete API
contract, including signal expressions, `policy_bank`, state machines, and
resource preset fields.

## Commands

| Command | Description |
| --- | --- |
| `botlab run [DIR]` | Compile and run natively (local MuJoCo + ONNX Runtime), opening the browser as a read-only `?ws=` viewer. `--browser` switches to the editor debug loop where the browser executes the project. Neither writes a zip unless `--out` is provided. Supports `--verbose`, `--no-browser`, `--host`, and `--port`. |
| `botlab build [DIR]` | Export `release.zip` to the project directory or a path selected with `--out`. The output uses the same format as BotLab's hosted `/d/<id>/` demos. `--site <dir>` additionally assembles a self-contained static site (see below). |
| `botlab validate [DIR]` | Collect declarations, compile the plan, and build the release in memory without starting a server. |
| `botlab-run [DIR]` | Native headless run without the viewer server: the same runner `botlab run` drives, started directly. Requires the `runtime` extra. |
| `botlab ui sync --botlab <path>` | Synchronize the editor and embed runtime from a BotLab build, excluding built-in templates and shared robot models. |
| `botlab ui status` | Show the status of the bundled UI snapshot. |
| `botlab docs` | Print where the bundled docs (component reference, decision table, error guide) live in this install. |

## Standalone site (`botlab build --site <dir>`)

`botlab build --site docs/demo` writes a directory you can push to any static
host — GitHub Pages included — next to the `release.zip`:

```text
docs/demo/
├── index.html        ← the embed runtime entry, renamed
├── assets/, wasm/    ← the runtime closure scanned from the bundled UI snapshot
├── project/          ← the whole project directory (robot MJCF, meshes, …)
└── release.zip       ← fetched by the page itself
```

Every runtime URL is page-relative, so the site works from a sub-path such as
`user.github.io/<repo>/` and needs no special response headers (the embed
runtime is single-threaded). Visitors can drag the robot, drive it with the
keyboard, and switch policies without installing anything.

The project must be self-contained — every resource under the project
directory. A package that still needs a live-Python (`modify=…`) controller is
rejected with a hint, matching what the embed runtime itself would do.

## Native local runtime (`botlab run`, `botlab-run`)

`botlab run --browser` executes the simulation in the browser (MuJoCo WASM +
ONNX Runtime Web). The default and `botlab-run` are the other way around:
everything runs natively in a CPython process — native `mujoco` physics,
`onnxruntime` inference, NumPy signal math. `botlab run` additionally serves
the viewer page and opens it for you; `botlab-run` is the bare headless
runner (CI, smoke tests, no UI snapshot needed).

```bash
pip install "botlab-python[runtime]"
BOTLAB_STEPS=500 botlab-run my-robot-project
```

- `BOTLAB_STEPS=<n>` caps the run at `n` control steps and prints the
  completed count to stderr; without it the loop runs until interrupted.
- Robot/scene sources resolve against the project directory first, then the
  shared assets root (`../..`), so BotLab templates that reference
  `models/...` trees run as-is.
- Projects using built-in resource presets (e.g. `g1-hiking-robot`) work: the
  registry is vendored as `botlab/_presets.py` and resolved offline.
- Headless depth/RGB cameras need a working EGL driver: start with
  `MUJOCO_GL=egl`. GPU inference needs `onnxruntime-gpu` matching your local
  CUDA/cuDNN; CPU works out of the box.
- Without the `runtime` extra the CLI still compiles and packages projects;
  `botlab run` (native default) and `botlab-run` fail up front listing the
  missing packages and the install command — `botlab run --browser` keeps
  working, since the browser brings its own runtimes.
- `botlab run` compiles the project before starting the native runner, so it
  accepts the same controller set as `botlab build` (compiled
  `PolicyController` plans and `Signal.action(modify=...)` hooks). Projects
  built around a custom `Controller.step()` class run via `botlab-run`, which
  installs the live session before `main.py` executes.

`BOTLAB_DISPLAY=1` (optionally `BOTLAB_DISPLAY_PORT=<port>`) opens a
loopback-only WebSocket that streams one metadata frame plus one
little-endian float32 frame (`[time, *qpos, *qvel]`) per control step. The
runner prints a token URL like `ws://127.0.0.1:<port>/display/<token>`; the
browser embed viewer accepts it as the `?ws=` parameter — which is exactly
what `botlab run` opens for you. The socket validates browser origins, limits
frame sizes, and rejects every client write — it is a local monitoring
surface, not a remote-control interface.

## Documentation

- [Component reference](https://github.com/OpenLegged/botlab-python/blob/main/docs/component-reference.md)
  — all public SDK symbols and their contracts.
- [Decision table](https://github.com/OpenLegged/botlab-python/blob/main/docs/decision-table.md)
  — requirement-to-component mapping, code ownership rules, and common
  mistakes.
- [Error guide](https://github.com/OpenLegged/botlab-python/blob/main/docs/error-guide.md)
  — error messages, causes, and fixes.
- The same three documents ship inside the wheel as `botlab/docs/`, so local AI
  coding agents can read them straight after `pip install`; `botlab docs`
  prints where they are.
- The BotLab repository's `docs/api/` and `docs/examples/` directories contain
  the complete API semantics and tutorials.

`botlab run` binds to `0.0.0.0` on a random available port by default. The
server sends `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`
headers because MuJoCo and ONNX Runtime Web require `SharedArrayBuffer`. A
proxy or reverse proxy must preserve these headers. Successful static asset
requests are intentionally omitted from the access log; HTTP errors remain
visible.

## UI snapshot (`botlab/ui/`)

The bundled browser runtime comes from BotLab's `dist/` build output and is
synchronized from a BotLab checkout with `botlab ui sync`. The snapshot keeps
the editor and embed runtime closure but excludes `/d` demo applications,
`templates/` built-in projects, and `models/` shared robot models.

`botlab run` serves the current project's MJCF, ONNX, meshes, and other assets
directly under `/project/`. Analytics scripts are removed from the localized
`index.html`. The generated snapshot is ignored by Git and included in the
wheel with Hatch's `force-include` setting; `botlab ui status` reports whether
it is present in the active install.

## Runtime model

The browser embed runtime reads `./release.zip`, which contains `release.json`
(the manifest, compiled controller plan, and SHA-256 integrity manifest), the
project sources under `project/`, and binary assets under `assets/`.

A release with a compiled controller does not need Pyodide in the browser. The
TypeScript simulation engine—MuJoCo worker plus ONNX Runtime Web—executes the
plan on every control frame without crossing into Python.

Current constraints:

- `main.py` must use the standard four-line entry point.
- `Signal.action(modify=...)` callbacks are supported by `botlab run` and
  offline packages built with `botlab build`. These projects carry source plus
  the manifest and use in-browser Pyodide instead of a compiled controller.
  This adds Python startup cost and one small Python boundary crossing per
  control step. Custom `Controller.step()` callbacks use the same path.
- Legacy projects with a non-empty `botlab.yaml` are not supported by the CLI;
  Python declarations are the only CLI entry point.
- Project assets must use project-relative paths and must exist in the project
  directory. (`botlab-run` additionally resolves the shared assets root, but
  `botlab build`/`validate` still require self-contained projects.)

## Development

```bash
uv venv
uv pip install -e '.[dev]'
pytest
```

`[dev]` includes the `runtime` extras (mujoco, onnxruntime, numpy, websockets)
so the native `botlab-run` tests run; a bare
`uv pip install -e .` skips them and those tests skip too.

Editable installs do not include the UI snapshot: `src/botlab/ui/` is
Git-ignored build output, so `botlab run` and `botlab build` fail with
"UI snapshot is missing index.html" until the snapshot is generated — build
the frontend in a BotLab checkout (`npm install && npm run build`), then run
`botlab ui sync --botlab <BotLab checkout>`. A plain
`pip install botlab-python` from PyPI ships the snapshot in the wheel and
needs no extra step.

`src/botlab/sdk/` is a verbatim mirror of the `botlab` package in the BotLab
repository (`src/domain/python/botlab/`): its flat modules are the browser
SDK and its `_runtime/` subpackage is the native runner behind `botlab-run`.
The generated built-in preset registry (`botlab_presets.py` upstream) is
mirrored as `src/botlab/_presets.py`. Re-sync everything after BotLab-side
changes with `python scripts/sync_sdk.py --botlab <BotLab checkout>`.
`_host.py` is the CPython counterpart of the browser's JavaScript bridge; the
native runtime installs its session into it via `set_runtime_session` before
a project's `main.py` runs.

## Candidate wheel acceptance

Before uploading a release, verify the **exact wheel to be uploaded**, including
its real synchronized UI snapshot. From this checkout, with `uv` on PATH:

```bash
python scripts/verify_wheel.py dist/pypi-0.3.0/botlab_python-0.3.0-py3-none-any.whl \
  --out tmp/wheel-acceptance-0.3.0
```

Use a new output directory for each run. The command installs that wheel with
`[runtime]` into a fresh isolated venv, checks the installed docs, UI and both
CLI entrypoints, then runs `docs`, `ui status`, `validate`, a bounded real native
MuJoCo simulation, `build`, native `run --no-browser --port 0`, and `build --site`.
It uses a generated self-contained `main.py` / `controller.py` project with local
MJCF; no BotLab checkout, existing venv, fixed machine path, or fixed port is
required. It does not rebuild, modify, or upload the input wheel.

`report.json` records the input SHA-256, each command's exit code and the site
directory; command output is saved as separate logs. Browser acceptance is a
separate required step: serve `site/` using an ordinary static server, open it
at a nested URL without isolation headers, and verify the robot runs and can
be dragged. The script deliberately records `browser_verified: false`.

CI additionally runs this command with `--source-wheel` against a non-editable
wheel built from the public checkout. This checks docs, CLI, compilation and
native MuJoCo without UI or site assembly; **it is not the candidate release
gate**. `BOTLAB_ALLOW_MISSING_UI=1` remains the explicit contributor/CI build
override; it never permits missing bundled docs. The sdist includes docs for
source installs but has no bundled UI snapshot. Release uploads remain wheel
only; this verification command does not upload an sdist or any other artifact.

## License

Proprietary — same as the main BotLab repository.
