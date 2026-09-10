"""Local UI server: serve the embedded BotLab runtime plus one project release.

The embed runtime (``embed.html``) is a static bundle with relative asset
paths; it fetches ``./release.zip`` from its own directory. This server hosts
the UI snapshot, the built ``release.zip``, and the project directory so
project-relative sources in the manifest resolve exactly like a workspace
project. Cross-origin isolation headers (COOP/COEP/CORP) are mandatory:
the MuJoCo and ONNX runtimes need ``SharedArrayBuffer``.
"""

from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path

CROSS_ORIGIN_ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".gz": "application/gzip",
    ".zip": "application/zip",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".xml": "application/xml",
    ".onnx": "application/octet-stream",
    ".stl": "application/octet-stream",
    ".glb": "model/binary",
    ".bin": "application/octet-stream",
}


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def ui_snapshot_dir() -> Path:
    return Path(__file__).parent / "ui"


class BotlabUIRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with isolation headers and no directory listings."""

    def end_headers(self):
        for name, value in CROSS_ORIGIN_ISOLATION_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        # The embed bundle lives at the root of the UI snapshot; release.zip
        # and project sources are injected as extra root files by the server.
        from urllib.parse import unquote, urlparse

        url_path = unquote(urlparse(path).path)
        while url_path.startswith("/"):
            url_path = url_path[1:]
        if url_path == "release.zip":
            return str(self.server.botlab_release_zip)  # type: ignore[attr-defined]
        if url_path.startswith("project/"):
            relative = url_path[len("project/"):]
            root = Path(self.server.botlab_project_dir)  # type: ignore[attr-defined]
            candidate = (root / relative).resolve()
            if candidate == root or root in candidate.parents:
                return str(candidate)
            self.send_error(403, "Invalid project asset path")
            return ""
        return str(self.server.botlab_ui_dir / url_path)  # type: ignore[attr-defined]

    def list_directory(self, path: str):
        self.send_error(403, "Directory listings are disabled")
        return None

    def log_message(self, format: str, *args):  # noqa: A002 - stdlib signature
        import sys

        print(f"[botlab] {self.address_string()} {format % args}", file=sys.stderr)


def serve_project(
    project_dir: Path,
    release_zip: Path,
    host: str,
    port: int,
    *,
    open_browser: bool = True,
    ui_dir: Path | None = None,
) -> str:
    ui_dir = ui_dir or ui_snapshot_dir()
    if not (ui_dir / "embed.html").is_file():
        raise FileNotFoundError(
            f"BotLab UI snapshot is missing embed.html under {ui_dir}. "
            "Run `botlab ui sync --botlab <BotLab checkout>` first."
        )

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    handler = BotlabUIRequestHandler
    handler.directory = str(ui_dir)
    with Server((host, port), handler) as httpd:
        httpd.botlab_ui_dir = ui_dir  # type: ignore[attr-defined]
        httpd.botlab_release_zip = release_zip  # type: ignore[attr-defined]
        httpd.botlab_project_dir = project_dir.resolve()  # type: ignore[attr-defined]
        bound_host, bound_port = httpd.socket.getsockname()
        url = f"http://{bound_host if bound_host != '0.0.0.0' else 'localhost'}:{bound_port}/embed.html"
        print(f"[botlab] serving {project_dir.name} at {url}")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[botlab] stopped")
    return url
