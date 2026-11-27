"""Tests of ``openmind-core models list|show|delete`` and ``delete_model``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import CLASSES, synthetic_epochs
from openmind_core.cli import app
from openmind_core.datasets import MONTAGE_A_MI
from openmind_core.models import make_baseline
from openmind_core.models.registry import delete_model, list_models, models_dir, save_model


@pytest.fixture(scope="module")
def fitted():  # type: ignore[no-untyped-def]
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=15)
    return make_baseline("ts-lr").fit(epochs.X, epochs.y)


@pytest.fixture
def registry(fitted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    for name, alignment in (("ts-lr-a", "riemann"), ("ts-lr-b", None)):
        save_model(
            fitted,
            name,
            MONTAGE_A_MI,
            training={"dataset": "synthetic"},
            alignment=alignment,
            srate=125.0,
            n_times=250,
            architecture="ts-lr",
        )
    return models_dir()


def test_list_shows_every_model(registry: Path) -> None:
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split()[:4] == ["name", "architecture", "classes", "alignment"]
    assert lines[1].split()[:4] == ["ts-lr-a", "ts-lr", str(len(CLASSES)), "riemann"]
    assert lines[2].split()[:4] == ["ts-lr-b", "ts-lr", str(len(CLASSES)), "-"]
    assert all("KB" in line or " B" in line for line in lines[1:])


def test_list_without_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "no models" in result.output


def test_show_prints_the_card(registry: Path) -> None:
    result = CliRunner().invoke(app, ["models", "show", "ts-lr-a"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "ts-lr-a" in out and "sklearn" in out and "riemann" in out
    assert ",".join(sorted(CLASSES)) in out
    assert ",".join(MONTAGE_A_MI) in out
    assert "125" in out and "250" in out
    assert '"dataset": "synthetic"' in out
    assert "size" in out


def test_delete_asks_then_removes(registry: Path) -> None:
    runner = CliRunner()
    declined = runner.invoke(app, ["models", "delete", "ts-lr-a"], input="n\n")
    assert declined.exit_code != 0
    assert (registry / "ts-lr-a").exists()
    result = runner.invoke(app, ["models", "delete", "ts-lr-a", "--yes"])
    assert result.exit_code == 0, result.output
    assert "deleted model ts-lr-a" in result.output
    assert not (registry / "ts-lr-a").exists()
    assert [c.name for c in list_models()] == ["ts-lr-b"]


@pytest.mark.parametrize(
    "args",
    [
        ["models", "show", "missing"],
        ["models", "delete", "missing", "--yes"],
        ["models", "delete", "../outside", "--yes"],
        ["models", "show", "Bad Name"],
    ],
)
def test_errors_are_reported(registry: Path, args: list[str]) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "Traceback" not in result.output


def test_delete_model_refuses_names_outside_the_registry(registry: Path) -> None:
    outside = registry.parent / "keep-me"
    outside.mkdir()
    with pytest.raises(ValueError, match="invalid model name"):
        delete_model("../keep-me")
    assert outside.exists()
    with pytest.raises(FileNotFoundError):
        delete_model("missing")
    assert delete_model("ts-lr-b") == 2  # card.json and model.joblib
