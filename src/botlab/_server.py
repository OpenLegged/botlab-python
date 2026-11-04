"""Local UI server: serve one project with a chosen UI directory.

The default package UI is the pure display viewer. Browser execution passes
the separately cached editor runtime as ``ui_dir``; its localized
``index.html`` loads the project through ``local-project.json``
(current source files, read from disk on every request) and applies like any
other workspace project. Project-relative assets in the manifest resolve
through ``/project/``, which this server maps onto the directory on disk —
edits land on reload, no zip round trip. ``embed.html`` keeps running
packaged releases for previewing the published form.

Cross-origin isolation headers (COOP/COEP/CORP) are mandatory: the MuJoCo
and ONNX runtimes need ``SharedArrayBuffer``.

The release endpoint stays for the embed surface: either a finished
``release.zip`` on disk (``botlab build`` output, passed as ``release_zip``)
or a callable producing the package bytes on demand (``botlab run`` debug
mode, passed as ``release_provider``); the debug provider rebuilds from
source when project files changed, so a browser reload picks edits up
without restarting the server.
"""

from __future__ import annotations

import http.server
import json
import secrets
import socketserver
import webbrowser
from collections.abc import Callable
from pathlib import Path

from ._host import BotLabError
from ._project import is_project_text_file, walk_project_files
from .hil.protocol import error_details

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


def local_project_payload(project_dir: Path) -> dict:
    """Describe the local project: text sources plus every file asset.

    Read on every request so the editor sees source edits after a page
    reload, mirroring the release provider's rebuild-on-change contract.
    Files map 1:1 onto ``/project/`` URLs: the editor's asset inventory uses
    base ``project``, so every path must be relative to the project root.
    Source classification mirrors the editor's project file tree and the
    release packer: whatever ``is_project_text_file`` accepts (main.py /
    controller.py, presets, nested Python packages, ``config/*.yaml``) is a
    text source; every other file (MJCF, ONNX, meshes) is a binary asset the
    tree shows at its real relative path. Local build noise (dot files,
    ``release.zip``, caches, symlinks) stays out of the payload entirely.
    """
    files = {}
    assets = []
    for posix, entry in walk_project_files(project_dir):
        if posix != "botlab.yaml" and is_project_text_file(posix):
            try:
                files[posix] = entry.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise BotLabError(
                    f'Project file "{posix}" is not UTF-8 text: {error}'
                ) from error
        else:
            assets.append(posix)
    return {
        "format": "botlab-local-project",
        "version": 1,
        "name": project_dir.name,
        "root": str(project_dir),
        "files": files,
        "assets": assets,
    }


class BotlabUIRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static site handler with isolation headers and no directory listings."""

    def end_headers(self):
        for name, value in CROSS_ORIGIN_ISOLATION_HEADERS.items():
            self.send_header(name, value)
        # A debug loop must never serve a stale release or stale sources.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        server = self.server  # type: ignore[attr-defined]
        if getattr(server, "botlab_hil", None) is not None:
            if not self._hil_host_allowed():
                self.send_error(403, "Invalid Host")
                return
            path = _request_path(self.path)
            if path.startswith("hil/"):
                self._serve_hil("GET", path)
                return
            if path in ("", "index.html"):
                self._serve_hil_index()
                return
        provider = getattr(server, "botlab_release_provider", None)
        request_path = _request_path(self.path)
        if provider is not None and request_path == "release.zip":
            self._serve_release_bytes(provider)
            return
        if request_path == "local-project.json":
            self._serve_local_project()
            return
        super().do_GET()

    def _hil_host_allowed(self) -> bool:
        return self.headers.get("Host") in self.server.botlab_hil_hosts

    def _hil_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_hil_index(self) -> None:
        server = self.server
        bootstrap = json.dumps({
            "token": server.botlab_hil_token,
            "selectedTarget": server.botlab_hil.handle("GET", "/hil/state")["selectedTarget"],
        }).replace("<", "\\u003c")
        source = (server.botlab_ui_dir / "index.html").read_text(encoding="utf-8")
        injection = f"<script>window.BOTLAB_HIL={bootstrap};</script>"
        source = source.replace("</head>", injection + "</head>") if "</head>" in source else injection + source
        data = source.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_hil(self, method: str, path: str) -> None:
        server = self.server
        token = self.headers.get("X-Botlab-Hil-Token", "")
        origin = self.headers.get("Origin")
        if (not self._hil_host_allowed()
                or not secrets.compare_digest(token, server.botlab_hil_token)
                or origin is not None and origin not in server.botlab_hil_origins):
            self.close_connection = True
            self._hil_json(403, {"error": "HIL requests require the local session token and origin."})
            return
        try:
            body = None
            if method == "POST":
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked requests are not supported")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 48 * 1024 * 1024:
                    raise ValueError("Invalid HIL request size")
                self.connection.settimeout(15)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
            result = server.botlab_hil.handle(method, "/" + path, body)
            self._hil_json(200, result)
        except Exception as error:
            self.close_connection = True
            self._hil_json(400, {"error": str(error), **error_details(error)})

    def do_POST(self):
        path = _request_path(self.path)
        if getattr(self.server, "botlab_hil", None) is None or not path.startswith("hil/"):
            self.send_error(404, "Not found")
            return
        self._serve_hil("POST", path)

    def _serve_local_project(self) -> None:
        server = self.server  # type: ignore[attr-defined]
        project_dir = Path(server.botlab_project_dir)  # type: ignore[attr-defined]
        try:
            payload = local_project_payload(project_dir)
        except Exception as error:  # noqa: BLE001 - surface any read failure to the browser
            self.send_error(500, f"botlab local project read failed: {error}")
            return
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_release_bytes(self, provider: Callable[[], bytes]):
        try:
            data = provider()
        except Exception as error:  # noqa: BLE001 - report any rebuild failure to the browser
            self.send_error(500, f"botlab debug rebuild failed: {error}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def translate_path(self, path: str) -> str:
        # The UI snapshot is the site root; the project directory is grafted
        # onto it at project/ so manifest sources resolve like a workspace
        # project, and release.zip is injected as an extra root file.
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
            if ((candidate == root or root in candidate.parents)
                    and not any(part.startswith(".") for part in Path(relative).parts)):
                return str(candidate)
            self.send_error(403, "Invalid project asset path")
            return ""
        root = Path(self.server.botlab_ui_dir).resolve()
        candidate = (root / url_path).resolve()
        if candidate == root or root in candidate.parents:
            return str(candidate)
        self.send_error(403, "Invalid UI asset path")
        return ""

    def list_directory(self, path: str):
        self.send_error(403, "Directory listings are disabled")
        return None

    def log_message(self, format: str, *args):  # noqa: A002 - stdlib signature
        import sys

        print(f"[botlab] {self.address_string()} {format % args}", file=sys.stderr)

    def log_request(self, code="-", size="-"):  # noqa: A002 - stdlib signature
        # A page load fetches a hundred JS chunks and meshes; logging each
        # one is noise. Errors still surface through send_error (log_error
        # routes to log_message) so broken assets stay visible.
        pass


def _request_path(path: str) -> str:
    from urllib.parse import unquote, urlparse

    url_path = unquote(urlparse(path).path)
    return url_path.lstrip("/")


def _run_server(host: str, port: int, handler: type) -> socketserver.BaseServer:
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server((host, port), handler)


def _terminal_link(url: str) -> str:
    # OSC 8 hyperlink: clickable in supporting terminals, plain text elsewhere.
    return f"\033]8;;{url}\007{url}\033]8;;\007"


def _lan_ip() -> str | None:
    """The host's LAN address, or None when there is no private network.

    A short-lived UDP connect() to a public address picks the outbound
    interface without sending any packet; 8.8.8.8 is the conventional probe.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


def print_startup_banner(project_name: str, base_url: str, entry: str, host: str = "") -> None:
    """Vite-style banner: Local link always, Network link when reachable."""
    lines = [
        f"[botlab] {project_name} ready",
        f"  ➜  Local:   {_terminal_link(f'{base_url}{entry}')}",
    ]
    if host in ("0.0.0.0", "::"):
        lan = _lan_ip()
        if lan:
            port = base_url.rsplit(":", 1)[1]
            lines.append(f"  ➜  Network: {_terminal_link(f'http://{lan}:{port}{entry}')}")
    # flush: stdout is block-buffered when redirected, and the banner must be
    # visible immediately (Vite-style) even under `botlab run | tee`.
    for line in lines:
        print(line, flush=True)


def start_project_server(
    project_dir: Path,
    release_zip: Path | None,
    host: str,
    port: int,
    *,
    ui_dir: Path | None = None,
    release_provider: Callable[[], bytes] | None = None,
    hil=None,
):
    """Build (but do not serve) the UI server; returns (httpd, base_url)."""
    if (release_zip is None) == (release_provider is None):
        raise ValueError("start_project_server requires exactly one of release_zip or release_provider")
    ui_dir = ui_dir or ui_snapshot_dir()
    if not (ui_dir / "index.html").is_file():
        raise FileNotFoundError(
            f"BotLab UI is missing index.html under {ui_dir}. "
            "Run `botlab ui sync --botlab <BotLab checkout>` first."
        )

    if hil is not None and host not in ("127.0.0.1", "localhost"):
        raise BotLabError("HIL requires --host 127.0.0.1 or localhost.")
    handler = BotlabUIRequestHandler
    handler.directory = str(ui_dir)
    httpd = _run_server(host, port, handler)
    httpd.botlab_ui_dir = ui_dir  # type: ignore[attr-defined]
    httpd.botlab_release_zip = release_zip  # type: ignore[attr-defined]
    httpd.botlab_release_provider = release_provider  # type: ignore[attr-defined]
    httpd.botlab_project_dir = project_dir.resolve()  # type: ignore[attr-defined]
    bound_host, bound_port = httpd.socket.getsockname()
    if hil is not None:
        httpd.botlab_hil = hil
        httpd.botlab_hil_token = secrets.token_urlsafe(32)
        httpd.botlab_hil_hosts = {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}
        httpd.botlab_hil_origins = {f"http://{value}" for value in httpd.botlab_hil_hosts}
    displayed_host = "localhost" if bound_host == "0.0.0.0" else bound_host
    return httpd, f"http://{displayed_host}:{bound_port}"


def serve_project(
    project_dir: Path,
    release_zip: Path | None,
    host: str,
    port: int,
    *,
    open_browser: bool = True,
    ui_dir: Path | None = None,
    release_provider: Callable[[], bytes] | None = None,
    entry: str = "/",
    hil=None,
) -> str:
    httpd, base_url = start_project_server(
        project_dir,
        release_zip,
        host,
        port,
        ui_dir=ui_dir,
        release_provider=release_provider,
        hil=hil,
    )
    editor_url = f"{base_url}{entry}"
    print_startup_banner(project_dir.name, base_url, entry, host=host)
    with httpd:
        if open_browser:
            webbrowser.open(editor_url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[botlab] stopped")
        finally:
            if hil is not None:
                hil.close()
    return editor_url
