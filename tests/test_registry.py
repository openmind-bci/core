"""Tests of the model registry with classical models (no torch needed)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import CLASSES, synthetic_epochs
from openmind_core.datasets import MONTAGE_A_MI
from openmind_core.models import make_baseline
from openmind_core.models.registry import (
    ModelCard,
    list_models,
    load_card,
    load_model,
    save_model,
)


@pytest.fixture(scope="module")
def fitted() -> tuple:  # type: ignore[type-arg]
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=15)
    return make_baseline("ts-lr").fit(epochs.X, epochs.y), epochs.X


def test_sklearn_round_trip(fitted: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    model, X = fitted
    path = save_model(
        model,
        "ts-lr-test",
        MONTAGE_A_MI,
        training={"dataset": "synthetic"},
        root=tmp_path,
        alignment="riemann",
        filters={"low_hz": 8.0, "high_hz": 30.0},
        srate=125.0,
        n_times=250,
        architecture="ts-lr",
    )
    assert sorted(p.name for p in path.iterdir()) == ["card.json", "model.joblib"]
    loaded, card = load_model("ts-lr-test", root=tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert card.format == "sklearn"
    assert card.architecture == "ts-lr"
    assert card.alignment == "riemann"
    assert card.filters == {"low_hz": 8.0, "high_hz": 30.0}
    assert card.channels == MONTAGE_A_MI
    assert (card.srate, card.n_times) == (125.0, 250)
    assert card.classes == tuple(sorted(CLASSES))
    assert card.scale == (1.0,) * len(MONTAGE_A_MI)
    assert [c.name for c in list_models(tmp_path)] == ["ts-lr-test"]


def test_baselines_need_srate_and_n_times(fitted: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    model, _ = fitted
    with pytest.raises(ValueError, match="srate and n_times"):
        save_model(model, "m", MONTAGE_A_MI, root=tmp_path)
    with pytest.raises(ValueError, match="not trained"):
        save_model(make_baseline("mdm"), "m", MONTAGE_A_MI, root=tmp_path, srate=1, n_times=1)
    with pytest.raises(ValueError, match="invalid model name"):
        save_model(model, "A B", MONTAGE_A_MI, root=tmp_path, srate=125, n_times=250)
    with pytest.raises(ValueError, match="unknown alignment"):
        save_model(model, "m", MONTAGE_A_MI, root=tmp_path, srate=125, n_times=250, alignment="x")


def test_overwrite_is_refused_unless_asked(fitted: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    model, _ = fitted
    kwargs = {"root": tmp_path, "srate": 125.0, "n_times": 250}
    save_model(model, "m1", MONTAGE_A_MI, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        save_model(model, "m1", MONTAGE_A_MI, **kwargs)  # type: ignore[arg-type]
    save_model(model, "m1", MONTAGE_A_MI, overwrite=True, **kwargs)  # type: ignore[arg-type]


def test_old_cards_stay_loadable(tmp_path: Path) -> None:
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "card.json").write_text(
        json.dumps(
            {
                "name": "old",
                "architecture": "eegnet",
                "channels": ["C3", "C4"],
                "srate": 250.0,
                "n_times": 500,
                "classes": ["left", "right"],
                "scale": [1.0, 1.0],
                "params": {},
                "training": {},
                "openmind_core": "0.0.1",
                "created": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    card = load_card("old", root=tmp_path)
    assert isinstance(card, ModelCard)
    assert (card.format, card.alignment, card.filters) == ("torch", None, {})


def test_unknown_format_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "odd"
    folder.mkdir()
    card = ModelCard("odd", "x", ("C3",), 250.0, 500, ("a", "b"), (1.0,), format="onnx")
    from dataclasses import asdict

    (folder / "card.json").write_text(json.dumps(asdict(card)), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown model format"):
        load_model("odd", root=tmp_path)


def test_registry_does_not_import_the_deep_module() -> None:
    code = (
        "import sys, openmind_core.models.registry;"
        "assert 'openmind_core.models.deep' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
