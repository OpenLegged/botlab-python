"""Install and exercise an immutable candidate wheel in a new isolated environment.

Requires uv on PATH. This gate produces a real site for separate browser acceptance;
it neither builds the input wheel nor uploads anything. --source-wheel is the CI
contract check for a source checkout without the private BotLab UI build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DOCS = ("component-reference.md", "decision-table.md", "error-guide.md")
MAIN = """from botlab import Simulation
from controller import create_controller

sim = Simulation.from_project()
sim.run(create_controller(sim))
"""
CONTROLLER = '''from botlab import PolicyController, Robot, constant, project

project("Wheel Acceptance", control_hz=50, physics_hz=200)
robot = Robot("robot.xml", config={
    "actionDimension": 1,
    "jointQposOffset": 7,
    "jointQvelOffset": 6,
    "defaultJointPositions": [0.0],
    "policyJointOrder": [0],
    "motorJointOrder": [0],
    "actionScale": 1.0,
})

def create_controller(sim):
    return PolicyController(robot, control=constant([0.0]).action())
'''
MJCF = '''<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="3 3 0.1"/>
    <light pos="0 0 4"/>
    <body name="base" pos="0 0 1.5">
      <freejoint/>
      <geom type="sphere" size="0.15" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.7" size="0.06" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-10 10"/></actuator>
</mujoco>
'''
CHECK_INSTALL = '''
import importlib.metadata
import json
import sys
from pathlib import Path
import botlab
import mujoco
import numpy
import onnxruntime
import websockets

root = Path(botlab.__file__).resolve().parent
assert root.is_relative_to(Path(sys.prefix).resolve()), root
dist = importlib.metadata.distribution("botlab-python")
direct = json.loads(dist.read_text("direct_url.json"))
assert not direct.get("dir_info", {}).get("editable"), direct
assert direct["url"] == Path(sys.argv[1]).as_uri(), direct
assert "archive_info" in direct, direct
scripts = {item.name: item.value for item in dist.entry_points if item.group == "console_scripts"}
assert scripts == {"botlab": "botlab._cli:main"}, scripts
for name in ("component-reference.md", "decision-table.md", "error-guide.md"):
    assert (root / "docs" / name).is_file(), name
assert not any(root.rglob("*.wasm*")), "core wheel contains a WASM runtime"
if sys.argv[2] == "release":
    for name in ("index.html", "embed.html"):
        assert (root / "ui" / name).is_file(), name
else:
    assert not (root / "ui").exists(), "source-wheel check expects no UI"
print(json.dumps({"package": str(root), "version": dist.version,
                  "mujoco": mujoco.__version__, "direct_url": direct}, indent=2))
'''


def verify_archive(wheel: Path, *, source_wheel: bool) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {f"botlab/docs/{name}" for name in DOCS}
        required.update({"botlab/_cli.py", "botlab/_runtime/run.py"})
        if any(".wasm" in name for name in names):
            raise ValueError("core wheel must not contain WASM runtimes (including compressed copies)")
        if source_wheel:
            if any(name.startswith("botlab/ui/") for name in names):
                raise ValueError("--source-wheel expects a CI wheel without the UI snapshot")
        else:
            required.update({"botlab/ui/index.html", "botlab/ui/embed.html"})
        missing = required - names
        if missing:
            raise ValueError(f"wheel is missing required files: {', '.join(sorted(missing))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new directory for evidence and venv")
    parser.add_argument("--source-wheel", action="store_true", help="CI contract only: no UI/site")
    parser.add_argument("--browser-runtime", type=Path,
                        help="separate version-matched runtime archive for offline site acceptance")
    args = parser.parse_args(argv)
    wheel, out = args.wheel.resolve(), args.out.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error("wheel must be an existing .whl file")
    if out.exists():
        parser.error("--out must be a new directory; existing environments are never modified")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv must be installed and available on PATH")
    verify_archive(wheel, source_wheel=args.source_wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    runtime = args.browser_runtime.resolve() if args.browser_runtime else None
    if runtime is not None and not runtime.is_file():
        parser.error("--browser-runtime must be an existing archive")
    runtime_digest = hashlib.sha256(runtime.read_bytes()).hexdigest() if runtime else None
    out.mkdir(parents=True)
    (out / "tmp").mkdir()
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not key.startswith("BOTLAB_")
    }
    env.update({"PYTHONNOUSERSITE": "1", "TMPDIR": str(out / "tmp"),
                "BOTLAB_RUNTIME_CACHE_DIR": str(out / "browser-cache")})
    commands = []
    report = {"wheel": str(wheel), "sha256": digest, "mode": "source" if args.source_wheel
              else "release", "commands": commands, "browser_verified": False}
    if runtime is not None:
        report["browser_runtime"] = {"archive": str(runtime), "sha256": runtime_digest}

    def run(label: str, command: list[str], *, extra_env: dict[str, str] | None = None) -> str:
        print(f"[{label}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=out, env={**env, **(extra_env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=600)
        (out / f"{label}.log").write_text(result.stdout, encoding="utf-8")
        commands.append({"label": label, "argv": command, "exit_code": result.returncode})
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {out}")
        return result.stdout

    try:
        venv = out / "venv"
        run("venv", [uv, "venv", "--python", sys.executable, str(venv)])
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        cli = bin_dir / ("botlab.exe" if os.name == "nt" else "botlab")
        run("install", [uv, "pip", "install", "--python", str(python), f"{wheel}[runtime]"])
        run("installed", [str(python), "-I", "-c", CHECK_INSTALL, str(wheel), report["mode"]])
        run("version", [str(cli), "--version"])
        docs_output = run("docs", [str(cli), "docs"])
        if not all(name in docs_output for name in DOCS):
            raise RuntimeError("botlab docs did not list every bundled reference")
        ui_output = run("ui-status", [str(cli), "ui", "status"])
        if not args.source_wheel and ui_output.count("present: True") != 2:
            raise RuntimeError("botlab ui status did not report both real entrypoints")
        project = out / "project"
        project.mkdir()
        for name, content in (("main.py", MAIN), ("controller.py", CONTROLLER),
                              ("robot.xml", MJCF)):
            (project / name).write_text(content, encoding="utf-8")
        run("validate", [str(cli), "validate", str(project)])
        steps_env = {"BOTLAB_STEPS": "30"}
        native_output = run("native", [str(cli), "run", str(project), "--headless"],
                            extra_env=steps_env)
        if "completed 30 control steps" not in native_output:
            raise RuntimeError("native MuJoCo run did not report all requested steps")
        run("build", [str(cli), "build", str(project), "--out", str(out / "release.zip")])
        if (out / "browser-cache").exists():
            raise RuntimeError("compilation or native execution acquired a browser runtime")
        if not args.source_wheel:
            run("native-viewer", [str(cli), "run", str(project), "--no-browser",
                                  "--host", "127.0.0.1", "--port", "0"], extra_env=steps_env)
            if (out / "browser-cache").exists():
                raise RuntimeError("the native viewer acquired a browser runtime")
            if runtime is not None:
                run("runtime-import", [str(cli), "ui", "fetch", "--archive", str(runtime)])
            site = out / "site"
            run("site", [str(cli), "build", str(project), "--site", str(site)])
            if not (site / "index.html").is_file() or not any(site.rglob("*.wasm")):
                raise RuntimeError("standalone site is missing runtime files")
            report["site"] = str(site)
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != digest:
            raise RuntimeError("input wheel changed during acceptance")
        if runtime is not None and hashlib.sha256(runtime.read_bytes()).hexdigest() != runtime_digest:
            raise RuntimeError("input browser runtime changed during acceptance")
        report["exit_code"] = 0
        print("PASS: " + ("source wheel contract (no UI/site; not a release gate)"
                          if args.source_wheel else "candidate wheel; browser acceptance still required"))
        return 0
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report["exit_code"] = 1
        report["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
