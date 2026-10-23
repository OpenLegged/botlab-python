"""Robot-independent clients for the BotLab Control API."""

from .client import ControlClient
from .protocol import ControlError

__all__ = ["ControlClient", "ControlError"]
