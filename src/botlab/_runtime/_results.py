"""Persist generated result documents without overwriting project inputs."""

import json
import os
import re
import tempfile

_RESULT_PATH = re.compile(
    r"results/(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)*[A-Za-z0-9_][A-Za-z0-9_.-]*\.json"
)


def is_runtime_result_path(path):
    return isinstance(path, str) and _RESULT_PATH.fullmatch(path) is not None


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def is_runtime_result_file(path, content):
    """Mirror BotLab runtimeResultFiles.ts when classifying existing files.

    Unmarked or invalid JSON remains input. Writing new results still uses
    the stricter finite-value validation in _result_document.
    """
    if not is_runtime_result_path(path):
        return False
    try:
        document = json.loads(content, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        return False
    return (isinstance(document, dict)
            and document.get("format") == "botlab.runtime-result"
            and type(document.get("version")) in (int, float)
            and document["version"] == 1
            and "data" in document)


def _result_document(content):
    if not isinstance(content, str):
        raise ValueError("[botlab] Runtime result content must be JSON text")
    document = json.loads(content)
    if (not isinstance(document, dict)
            or document.get("format") != "botlab.runtime-result"
            or type(document.get("version")) is not int
            or document["version"] != 1
            or "data" not in document):
        raise ValueError("[botlab] Invalid runtime result document")
    # json.loads accepts both NaN and overflowing exponents; neither is a
    # JSON value that the browser host can store.
    json.dumps(document, allow_nan=False)
    return document


def write_result_file(directory, path, content):
    if not is_runtime_result_path(path):
        raise ValueError("[botlab] Runtime result path must be results/<name>.json")
    _result_document(content)
    directory = directory.resolve()
    target = directory / path
    # Reject symlinks in every component, including the destination file.
    # A results folder must never redirect output to another project or tree.
    candidate = directory
    for part in path.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("[botlab] Runtime result path cannot contain symlinks")
    if not target.resolve().is_relative_to(directory):
        raise ValueError("[botlab] Runtime result path must stay inside the project")
    if target.exists():
        try:
            _result_document(target.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError, OSError) as error:
            raise ValueError(f"[botlab] Cannot overwrite project input: {path}") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write completely before replacing the previous result; an interrupted
    # write leaves the prior JSON intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=".result-", delete=False,
        ) as output:
            temporary = output.name
            output.write(content)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
