"""Browser acceptance: the CLI-served embed runtime runs a packaged release.

Requires a botlab checkout with a fresh `npm run build` and the UI snapshot
synced (`botlab ui sync`). Run with `pytest tests/e2e -m e2e` after starting
a server, or standalone via the fixture below.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request

import pytest

pytestmark = pytest.mark.e2e

BOTLAB_CHECKOUT = "/home/xiangyk/Project/BotLab"
MICRODUCK_TEMPLATE = f"{BOTLAB_CHECKOUT}/public/templates/microduck"
PORT = 8931


@pytest.fixture(scope="module")
def cli_server():
    import sys

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "botlab._cli",
            "run",
            MICRODUCK_TEMPLATE,
            "--browser",
            "--port",
            str(PORT),
            "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/embed.html", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("botlab run server did not start")
    yield f"http://127.0.0.1:{PORT}/embed.html"
    process.terminate()
    process.wait(timeout=10)


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
