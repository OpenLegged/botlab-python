"""Durable original request identities. No credential is ever written here."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .discovery import local_endpoint, private_directory, private_file, project_key
from .protocol import ControlError, identifier, parse_json


class Receipts:
    def __init__(self, project: str | Path):
        self.project = str(Path(project).resolve())
        self.project_key = project_key(project)
        self.root = Path(os.environ.get("BOTLAB_CONTROL_RECEIPTS_DIR", Path.home() / ".local/state/botlab/control/receipts"))
        self.directory = self.root / self.project_key

    def path(self, request_id: str) -> Path:
        identifier(request_id, "request_id")
        return self.directory / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")

    def load(self, request_id: str) -> dict | None:
        path = self.path(request_id)
        if not path.exists():
            return None
        try:
            private_directory(self.root)
            private_directory(self.directory)
            private_file(path)
            receipt = parse_json(path.read_text())
            if (not isinstance(receipt, dict) or receipt.get("schema") != 1
                    or receipt.get("project_key") != self.project_key
                    or receipt.get("project") != self.project):
                raise ControlError("invalid_receipt", "Request receipt has invalid project identity.")
            envelope = receipt.get("envelope", {})
            if envelope.get("request_id") != request_id:
                raise ControlError("invalid_receipt", "Request receipt has invalid request identity.")
            for key in ("instance_id", "session_id", "episode_id", "capabilities_revision", "capability"):
                identifier(envelope.get(key), key)
            if not isinstance(envelope.get("args"), dict) or not isinstance(receipt.get("intent"), dict):
                raise ControlError("invalid_receipt", "Request receipt is incomplete.")
            local_endpoint(receipt.get("endpoint"))
            return receipt
        except OSError as error:
            raise ControlError("receipt_unavailable", "Cannot read the original request receipt.") from error

    def save(self, envelope: dict, endpoint: str, intent: dict, *,
             control_generation: int | str | None = None) -> tuple[dict, bool]:
        """Publish completely written data with a no-overwrite atomic hard link."""
        request_id = envelope["request_id"]
        receipt = {"schema": 1, "project": self.project, "project_key": self.project_key,
                   "endpoint": local_endpoint(endpoint), "intent": intent, "envelope": envelope}
        if control_generation is not None:
            receipt["control_generation"] = control_generation
        temporary = None
        try:
            private_directory(self.root, create=True)
            private_directory(self.directory, create=True)
            fd, temporary = tempfile.mkstemp(prefix=".request-", dir=self.directory)
            with os.fdopen(fd, "w") as handle:
                json.dump(receipt, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, self.path(request_id))
            except FileExistsError:
                existing = self.load(request_id)
                if existing is None:
                    raise ControlError("receipt_unavailable", "Request receipt changed during publication.") from None
                return existing, False
            if os.name == "posix":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return receipt, True
        except OSError as error:
            raise ControlError("receipt_unavailable", "Cannot persist request identity; no command was sent.") from error
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
