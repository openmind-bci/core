"""Tests for channel label normalization and the package's private imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from openmind_core.channels import normalize_label

SRC = Path(__file__).resolve().parents[1] / "src" / "openmind_core"


@pytest.mark.parametrize("label", ["FC3", "fc3", "Fc3.", "EEG FC3", " FC3 ", "EEG Fc3."])
def test_spellings_of_a_label_share_a_key(label: str) -> None:
    assert normalize_label(label) == normalize_label("FC3") == "fc3"


def test_different_labels_keep_different_keys() -> None:
    assert normalize_label("C3") != normalize_label("CP3")
    assert normalize_label("Cz") == normalize_label("CZ")


def _subpackage(module: str) -> str:
    """``openmind_core.online.decoder`` -> ``online``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def test_no_private_name_is_imported_across_subpackages() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        here = _subpackage(
            ".".join(("openmind_core", *path.relative_to(SRC).with_suffix("").parts))
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = (node.module or "") if isinstance(node, ast.ImportFrom) else ""
            if module.startswith("openmind_core") and _subpackage(module) != here:
                offenders += [
                    f"{path.relative_to(SRC)}: {node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_") and not alias.name.startswith("__")
                ]
    assert offenders == []
