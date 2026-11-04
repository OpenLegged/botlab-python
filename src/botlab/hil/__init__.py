"""Generic hardware-in-the-loop connection and inference services."""
from .host import HilHost
from .protocol import HilError

__all__ = ["HilError", "HilHost"]
