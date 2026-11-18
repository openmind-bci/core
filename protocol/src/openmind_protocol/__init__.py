"""OpenMind BCI command protocol: topics, schemas and validated messages."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openmind-protocol")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
