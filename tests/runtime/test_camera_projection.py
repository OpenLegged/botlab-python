"""Shared pinhole golden rays and real rendered depth, without policy weights."""
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from botlab._runtime import _observations as obs

GOLDEN = json.loads((Path(__file__).parents[1] / "fixtures/camera-projection.json").read_text())


@pytest.mark.parametrize("value", [0, -1, 180, 181, float("nan"), float("inf"), "bad"])
def test_invalid_horizontal_fov(value):
    with pytest.raises(ValueError, match="horizontalFov"):
        obs.read_camera_config({"horizontalFov": value})


def test_optional_horizontal_fov():
    assert obs.read_camera_config({})["horizontalFov"] is None
    assert obs.read_camera_config({"horizontalFov": None})["horizontalFov"] is None


def camera_scene(fixture):
    # A tilted plane has an analytic depth at every ray. This also verifies
    # MuJoCo actually consumes frustum_width, rather than only storing it.
    angle = 0.3
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom type="plane" size="10 10 .1" '
        f'quat="{math.cos(angle / 2)} 0 {math.sin(angle / 2)} 0"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = obs.read_camera_config({**fixture["config"], "target": "0",
        "cameraOffset": [0, 0, 2], "cameraOffsetFrame": "body",
        "cameraXyaxes": [1, 0, 0, 0, 1, 0], "near": .05, "far": 100})
    return model, data, config


@pytest.mark.parametrize("fixture", GOLDEN["cases"], ids=lambda item: item["name"])
def test_native_pixel_rays_without_gl(fixture):
    model, data, config = camera_scene(fixture)
    # Scene-camera fields are MuJoCo float32 storage, but constructing the
    # scene and writing its projection does not need an OpenGL context.
    renderer = SimpleNamespace(
        scene=mujoco.MjvScene(model, maxgeom=10),
        update_scene=lambda *args, **kwargs: None,
    )
    obs._camera_render(renderer, model, data, config)
    view = renderer.scene.camera[0]
    half_width = view.frustum_width or (view.frustum_top - view.frustum_bottom) / 2 * config["fullWidth"] / config["fullHeight"]
    rays = []
    for y in range(config["height"]):
        for x in range(config["width"]):
            rays.append([
                (2 * (x + config["cropX"] + .5) / config["fullWidth"] - 1) * half_width / view.frustum_near,
                (2 * (y + config["cropY"] + .5) / config["fullHeight"] - 1) * view.frustum_top / view.frustum_near,
                1,
            ])
    np.testing.assert_allclose(rays, fixture["rays"], atol=1e-7, rtol=0)


@pytest.mark.parametrize("fixture", GOLDEN["cases"], ids=lambda item: item["name"])
def test_native_rendered_depth(fixture):
    # A missing/broken GL context can abort native code, bypassing try/except
    # and poisoning subsequent Renderer attempts. Keep GL outside pytest.
    result = subprocess.run(
        [sys.executable, __file__, fixture["name"]],
        capture_output=True, text=True, timeout=30,
    )
    ready = "BOTLAB_RENDERER_READY" in result.stdout
    if not ready and (result.returncode == 77 or result.returncode < 0):
        pytest.skip(
            f"MuJoCo GL renderer unavailable (probe exit {result.returncode}); "
            f"set MUJOCO_GL=egl. {result.stdout[-1000:]} {result.stderr[-1000:]}"
        )
    assert result.returncode == 0, result.stdout + result.stderr


def render_depth_case(fixture):
    model, data, config = camera_scene(fixture)
    try:
        renderer = mujoco.Renderer(model, width=config["fullWidth"], height=config["fullHeight"])
    except Exception as error:
        print(f"MuJoCo GL renderer unavailable: {error}", flush=True)
        raise SystemExit(77) from error
    print("BOTLAB_RENDERER_READY", flush=True)
    with renderer:
        obs._camera_render(renderer, model, data, config)
        renderer.enable_depth_rendering()
        depth = obs._render_depth_meters(renderer, model, config)
        crop = depth[config["cropY"]:config["cropY"] + config["height"], config["cropX"]:config["cropX"] + config["width"]]
        expected = 2 / (1 - math.tan(0.3) * np.asarray(fixture["rays"])[:, 0])
        # macOS GL lacks ARB_clip_control; depth quantization at 2 m is ~60 um.
        np.testing.assert_allclose(crop.reshape(-1), expected, atol=1e-4, rtol=0)


if __name__ == "__main__":
    # Do not leave large core dumps when a headless GL driver aborts.
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    render_depth_case(next(case for case in GOLDEN["cases"] if case["name"] == sys.argv[1]))
