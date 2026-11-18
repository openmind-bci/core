"""Access to the JSON Schemas of the protocol and validation against them."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from openmind_protocol.topics import TOPICS, Kind


def schema_dir() -> Path:
    """Directory holding ``v1/*.schema.json``.

    Installed packages ship the schemas inside ``openmind_protocol/schemas``; in a
    source checkout they are read from ``protocol/schemas``.
    """
    packaged = Path(str(resources.files("openmind_protocol"))) / "schemas"
    if (packaged / "v1").is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "schemas"


@cache
def load_schema(kind: Kind) -> dict[str, Any]:
    """Return the JSON Schema of a message kind (``"command"``, ``"status"``...)."""
    if kind not in TOPICS:
        raise KeyError(f"unknown message kind {kind!r}; known: {sorted(TOPICS)}")
    path = schema_dir() / "v1" / f"{kind}.schema.json"
    schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return schema


@cache
def _validator(kind: Kind) -> Draft202012Validator:
    return Draft202012Validator(load_schema(kind))


def schema_errors(kind: Kind, document: Any) -> list[str]:
    """Return every schema violation of ``document`` (empty when valid).

    Each error is prefixed with the JSON path of the offending field
    (``confidence: 1.5 is greater than the maximum of 1``).
    """
    errors = sorted(_validator(kind).iter_errors(document), key=lambda e: list(e.absolute_path))
    return [
        f"{'.'.join(str(p) for p in error.absolute_path) or '(root)'}: {error.message}"
        for error in errors
    ]
