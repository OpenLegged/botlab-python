"""Browser acceptance: native `botlab run` renders through the ?ws= viewer.

The simulation executes in the CLI's CPython subprocess; the browser page is
a read-only display client. Requires a botlab checkout with a fresh
`npm run build`, the UI snapshot synced, and the runtime extras installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.e2e

BOTLAB_CHECKOUT = "/home/xiangyk/Project/BotLab"
MICRODUCK_TEMPLATE = f"{BOTLAB_CHECKOUT}/public/templates/microduck"


@pytest.fixture(scope="module")
def native_run():
    output: list[str] = []

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "botlab._cli",
            "run",
            MICRODUCK_TEMPLATE,
            "--port",
            "0",
            "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None

    def pump():
        for line in process.stdout:
            output.append(line.rstrip("\n"))

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()

    viewer_url = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for line in output:
            match = re.search(r"Viewer:\s*(\S+embed\.html\?ws=\S+)", line)
            if match:
                viewer_url = match.group(1)
                break
        if viewer_url or process.poll() is not None:
            break
        time.sleep(0.2)

    if not viewer_url:
        process.kill()
        pytest.fail(f"native run did not report a viewer URL; output:\n{chr(10).join(output)}")
    yield viewer_url
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def test_native_run_renders_in_browser_viewer(native_run):
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(error.message))

        page.goto(native_run)
        page.wait_for_selector("canvas", timeout=120_000)
        page.wait_for_timeout(5_000)  # display stream drives the render
        page.screenshot(path="tmp/e2e-native-viewer.png")

        assert errors == []
        browser.close()
