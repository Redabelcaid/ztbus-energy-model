"""ZTBus energy-model identification package.

Public API is re-exported from this module sparingly; prefer fully-qualified
imports inside the codebase to keep coupling explicit.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ztbus-energy-model")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
