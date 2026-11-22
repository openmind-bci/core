"""Tests of ``openmind-core train`` (the dataset loader is replaced by synthetic epochs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import synthetic_epochs
from openmind_core import training
from openmind_core.cli import app
from openmind_core.datasets import MONTAGE_A_MI, MONTAGE_C
from openmind_core.models.registry import load_card, load_model

runner = CliRunner()


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def load_epochs(
        dataset: str, subjects: list[int], classes: list[str], channels: tuple[str, ...]
    ) -> Any:
        calls.append({"dataset": dataset, "subjects": subjects, "classes": classes})
        return synthetic_epochs(subjects=(1,), sessions=("0", "1"), per_class=10)

    monkeypatch.setattr(training, "load_epochs", load_epochs)
    return calls


def test_parse_subjects() -> None:
    assert training.parse_subjects("1-3,5, 8-9,2") == [1, 2, 3, 5, 8, 9]
    for bad in ("", "a", "3-1", "1-"):
        with pytest.raises(ValueError, match="subject"):
            training.parse_subjects(bad)


def test_resolve_channels() -> None:
    assert training.resolve_channels("c") == (MONTAGE_C, "c")
    assert training.resolve_channels("A") == (MONTAGE_A_MI, "a")
    assert training.resolve_channels("C3, Cz,C4") == (("C3", "Cz", "C4"), "3ch")
    with pytest.raises(ValueError, match="duplicate"):
        training.resolve_channels("C3,C3")
    assert len(MONTAGE_C) == 8


def test_default_name() -> None:
    assert training.default_name("ts-lr", "bnci2014_001", "c", "none") == "ts-lr-bnci2014_001-c"
    assert training.default_name("mdm", "physionetmi", "3ch", "riemann") == (
        "mdm-physionetmi-3ch-riemann"
    )


def test_train_command_registers_a_model(
    fake_loader: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    result = runner.invoke(
        app,
        ["train", "--dataset", "bnci2014_001", "--subjects", "1-2", "--alignment", "euclid",
         "--channels", "A"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert fake_loader == [
        {"dataset": "bnci2014_001", "subjects": [1, 2], "classes": ["left", "right", "forward"]}
    ]
    name = "ts-lr-bnci2014_001-a-euclid"
    assert name in result.output
    model, card = load_model(name)
    assert card.alignment == "euclid"
    assert card.format == "sklearn"
    assert card.channels == MONTAGE_A_MI
    assert card.n_times == 250
    assert card.training["subjects"] == [1, 2]
    assert card.training["n_epochs"] == 60
    assert model.predict_proba(np.random.default_rng(0).normal(size=(1, 7, 250))).shape == (1, 3)
    assert load_card(name).training["classes"] == ["left", "right", "forward"]

    again = runner.invoke(
        app, ["train", "--dataset", "bnci2014_001", "--subjects", "1", "--name", name]
    )
    assert again.exit_code == 1
    assert "already exists" in again.output
    ok = runner.invoke(
        app,
        ["train", "--dataset", "bnci2014_001", "--subjects", "1", "--name", name, "--overwrite"],
    )
    assert ok.exit_code == 0, ok.output
    assert load_card(name).alignment is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--subjects", "x"], "invalid subject list"),
        (["--subjects", "1", "--alignment", "magic"], "unknown alignment"),
        (["--subjects", "1", "--model", "svm"], "unknown model"),
        (["--subjects", "1", "--name", "Bad Name"], "invalid model name"),
    ],
)
def test_train_command_errors(
    fake_loader: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
    message: str,
) -> None:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["train", "--dataset", "bnci2014_001", *args])
    assert result.exit_code == 1
    assert message in result.output
    assert not fake_loader


def test_train_reports_a_missing_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def load_epochs(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("MOABB is not installed")

    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(training, "load_epochs", load_epochs)
    result = runner.invoke(app, ["train", "--dataset", "bnci2014_001", "--subjects", "1"])
    assert result.exit_code == 1
    assert "research extra" in result.output
