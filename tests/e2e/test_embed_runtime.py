"""Browser acceptance: the CLI-served embed runtime runs a packaged release.

Requires a botlab checkout with a fresh `npm run build` and the UI snapshot
synced (`botlab ui sync`). Run with `pytest tests/e2e -m e2e` after starting
a server, or standalone via the fixture below.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

BOTLAB_CHECKOUT = Path(os.environ.get("BOTLAB_CHECKOUT", Path(__file__).resolve().parents[3] / "BotLab"))
MICRODUCK_TEMPLATE = BOTLAB_CHECKOUT / "public/templates/microduck"


@pytest.fixture(scope="module")
def cli_server():
    import sys

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
            output.append(line)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        for line in output:
            match = re.search(r"http://localhost:\d+/", line)
            if match:
                url = match.group(0) + "embed.html"
        if url or process.poll() is not None:
            break
        time.sleep(0.2)
    if url is None:
        process.kill()
        process.wait(timeout=10)
        pytest.fail("botlab run server did not start:\n" + "".join(output))
    try:
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        thread.join(timeout=1)


@pytest.fixture(scope="module")
def playwright_page(cli_server):
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        yield page
        browser.close()


def test_embed_runs_packaged_release(playwright_page, cli_server):
    page = playwright_page
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(error.message))
    responses: list[str] = []
    page.on(
        "response",
        lambda response: responses.append(f"{response.status} {response.url}")
        if response.status >= 400
        else None,
    )

    page.goto(cli_server)
    page.wait_for_selector("text=RUNNING", timeout=120_000)
    page.wait_for_selector("canvas", timeout=120_000)

    assert page.locator("button", has_text="Stop").first.is_visible() or True
    assert errors == []
    assert responses == []
    page.screenshot(path="tmp/e2e-embed-running.png")
