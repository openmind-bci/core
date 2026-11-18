"""Local data locations (outside the repositories).

Datasets, caches, profiles and model weights live in ``$OPENMIND_DATA_DIR``
(default ``~/openmind-data``), never in Git.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides the data directory.
DATA_DIR_ENV = "OPENMIND_DATA_DIR"


def data_dir(create: bool = False) -> Path:
    """Return the OpenMind data directory, optionally creating it."""
    value = os.environ.get(DATA_DIR_ENV, "").strip()
    path = Path(value).expanduser() if value else Path.home() / "openmind-data"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
