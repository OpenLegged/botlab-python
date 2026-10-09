"""Browser acceptance: native `botlab run` renders through the ?ws= viewer.

The simulation executes in the CLI's CPython subprocess; the browser page is
a read-only display client. Requires a botlab checkout with a fresh
`npm run build`, the UI snapshot synced, and the runtime extras installed.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

BOTLAB_CHECKOUT = Path(os.environ.get("BOTLAB_CHECKOUT", Path(__file__).resolve().parents[3] / "BotLab"))
MICRODUCK_TEMPLATE = BOTLAB_CHECKOUT / "public/templates/microduck"


@pytest.fixture(scope="module")
def native_run():
    if not MICRODUCK_TEMPLATE.is_dir():
        pytest.skip("set BOTLAB_CHECKOUT to a BotLab checkout with the MicroDuck template")
    output: list[str] = []

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "botlab._cli",
            "run",
            str(MICRODUCK_TEMPLATE),
            "--native",
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
        process.wait(timeout=10)
        pytest.fail(f"native run did not report a viewer URL; output:\n{chr(10).join(output)}")
    try:
        yield viewer_url
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        thread.join(timeout=1)


def test_native_run_renders_in_browser_viewer(native_run):
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(error.message))
        requests: list[str] = []
        page.on("request", lambda request: requests.append(request.url))
        # Rendering a native run must work even when WASM execution is disabled.
        page.add_init_script("""
            WebAssembly.instantiate = WebAssembly.compile = () => {
                throw new Error('native display must not execute WebAssembly');
            };
        """)

        page.goto(native_run)
        page.wait_for_selector("canvas", timeout=120_000)
        page.wait_for_function("""() => Number(document.querySelector(
            '[data-testid="remote-display-status"]'
        )?.dataset.appliedSequence) > 1""", timeout=120_000)
        page.get_by_role("button", name="Follow: on").click()
        assert page.get_by_role("button", name="Follow: off").get_attribute("aria-pressed") == "false"
        page.get_by_role("button", name="Center view").click()
        page.mouse.move(600, 350)
        page.mouse.down()
        page.mouse.move(700, 400, steps=10)
        page.mouse.up()
        page.mouse.wheel(0, 100)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.get_by_role("button", name="Center view").is_visible()
        page.screenshot(path="tmp/e2e-native-viewer.png")

        assert errors == []
        assert not any(re.search(r"\.wasm|pyodide|ort-wasm|release\.zip", url) for url in requests)
        browser.close()
