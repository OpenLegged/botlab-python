"""Public host context for explicitly selected browser project adapters.

An adapter is a callable ``run(context) -> int``. It serves project sources and
hardware bridges using ``context.ui_dir`` (the SDK-managed browser runtime).
Physics remains in the browser. Importing an adapter executes local Python code,
as importing a project's controller does; only select adapters you trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BrowserRunContext:
    project_dir: Path
    ui_dir: Path
    host: str
    port: int
    open_browser: bool
    inference_endpoint: str | None = None
