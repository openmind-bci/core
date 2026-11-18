"""Smoke tests for the core package skeleton."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import openmind_core
from openmind_core.cli import app

runner = CliRunner()


def test_version_is_exposed() -> None:
    assert openmind_core.__version__


def test_cli_version_shows_protocol() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert openmind_core.__version__ in result.output
    assert "protocol" in result.output


def test_cli_info_uses_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert str(tmp_path) in result.output
