"""Tests for the deep decoders and the model registry (need the deep extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("braindecode")

from helpers import CLASSES, synthetic_epochs
from openmind_core.datasets import MONTAGE_A_MI
from openmind_core.evaluation import within_session
from openmind_core.models import DEEP_MODELS, make_model, model_names
from openmind_core.models.deep import DeepClassifier, build_network, resolve_device
from openmind_core.models.registry import (
    list_models,
    load_card,
    load_model,
    save_model,
)

FAST = {"max_epochs": 60, "patience": 60, "device": "cpu", "batch_size": 32, "lr": 3e-3}


@pytest.fixture(scope="module")
def split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=40)
    even = np.arange(len(epochs)) % 2 == 0
    return epochs.X[even], epochs.y[even], epochs.X[~even], epochs.y[~even]


@pytest.fixture(scope="module")
def trained(split: tuple[np.ndarray, ...]) -> DeepClassifier:
    x_train, y_train, _, _ = split
    return make_model("shallow", srate=125.0, seed=1, **FAST).fit(x_train, y_train)


@pytest.mark.parametrize("architecture", DEEP_MODELS)
def test_networks_learn_separable_classes(architecture: str, split: tuple[np.ndarray, ...]) -> None:
    x_train, y_train, x_test, y_test = split
    model = make_model(architecture, srate=125.0, seed=0, **FAST).fit(x_train, y_train)
    assert (model.predict(x_test) == y_test).mean() > 0.8


def test_probabilities(trained: DeepClassifier, split: tuple[np.ndarray, ...]) -> None:
    proba = trained.predict_proba(split[2])
    assert proba.shape == (len(split[2]), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert tuple(trained.classes_) == tuple(sorted(CLASSES))
    assert trained.predict(split[2]).tolist() == trained.classes_[proba.argmax(1)].tolist()


def test_training_is_reproducible(split: tuple[np.ndarray, ...]) -> None:
    x_train, y_train, x_test, _ = split
    settings = FAST | {"max_epochs": 5}
    first = make_model("eegnet", srate=125.0, seed=7, **settings).fit(x_train, y_train)
    second = make_model("eegnet", srate=125.0, seed=7, **settings).fit(x_train, y_train)
    np.testing.assert_allclose(first.predict_proba(x_test), second.predict_proba(x_test))


def test_early_stopping_keeps_history(split: tuple[np.ndarray, ...]) -> None:
    x_train, y_train, _, _ = split
    model = make_model("eegnet", srate=125.0, **(FAST | {"max_epochs": 50, "patience": 3})).fit(
        x_train, y_train
    )
    assert 4 <= len(model.history_) <= 50
    assert {"train_loss", "valid_loss", "valid_accuracy"} <= set(model.history_[0])


def test_training_without_validation(split: tuple[np.ndarray, ...]) -> None:
    x_train, y_train, _, _ = split
    model = make_model(
        "eegnet", srate=125.0, **(FAST | {"max_epochs": 3, "valid_fraction": 0.0})
    ).fit(x_train, y_train)
    assert len(model.history_) == 3
    assert "valid_loss" not in model.history_[0]


def test_inputs_are_scaled_per_channel(
    trained: DeepClassifier, split: tuple[np.ndarray, ...]
) -> None:
    assert trained.scale_.shape == (len(MONTAGE_A_MI),)
    np.testing.assert_allclose(trained.scale_, split[0].std(axis=(0, 2)), rtol=1e-4)


def test_rejects_bad_inputs(trained: DeepClassifier) -> None:
    with pytest.raises(ValueError, match="n_epochs, n_channels, n_times"):
        trained.predict(np.zeros((4, 7)))
    with pytest.raises(ValueError, match="expected 7 channels"):
        trained.predict(np.zeros((4, 5, 250)))
    with pytest.raises(ValueError, match="two classes"):
        DeepClassifier().fit(np.zeros((4, 7, 250)), ["left"] * 4)


def test_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown architecture"):
        build_network("transformer", 7, 2, 250, 125.0)
    with pytest.raises(KeyError, match="unknown model"):
        make_model("transformer")
    with pytest.raises(ValueError, match="takes no parameters"):
        make_model("mdm", max_epochs=3)
    assert set(DEEP_MODELS) <= set(model_names())


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_works_with_evaluation_protocols() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=15)
    scores = within_session(
        epochs, lambda: make_model("shallow", srate=125.0, **FAST), "shallow", n_splits=3
    )
    assert scores[0].accuracy > 0.7


def test_registry_round_trip(
    trained: DeepClassifier, split: tuple[np.ndarray, ...], tmp_path: Path
) -> None:
    path = save_model(
        trained, "shallow-test", MONTAGE_A_MI, training={"dataset": "synthetic"}, root=tmp_path
    )
    assert sorted(p.name for p in path.iterdir()) == ["card.json", "weights.pt"]
    model, card = load_model("shallow-test", root=tmp_path)
    np.testing.assert_allclose(model.predict_proba(split[2]), trained.predict_proba(split[2]))
    assert card.architecture == "shallow"
    assert card.channels == MONTAGE_A_MI
    assert card.srate == 125.0
    assert card.n_times == 250
    assert card.classes == tuple(sorted(CLASSES))
    assert card.training == {"dataset": "synthetic"}
    assert card.params["max_epochs"] == 60
    assert card.created.endswith("+00:00")
    assert [c.name for c in list_models(tmp_path)] == ["shallow-test"]


def test_registry_refuses_overwrite_and_bad_names(trained: DeepClassifier, tmp_path: Path) -> None:
    save_model(trained, "m1", MONTAGE_A_MI, root=tmp_path)
    with pytest.raises(FileExistsError):
        save_model(trained, "m1", MONTAGE_A_MI, root=tmp_path)
    save_model(trained, "m1", MONTAGE_A_MI, root=tmp_path, overwrite=True)
    with pytest.raises(ValueError, match="invalid model name"):
        save_model(trained, "../escape", MONTAGE_A_MI, root=tmp_path)
    with pytest.raises(ValueError, match="channels given"):
        save_model(trained, "m2", ("C3", "C4"), root=tmp_path)
    with pytest.raises(ValueError, match="not trained"):
        save_model(DeepClassifier(), "m3", MONTAGE_A_MI, root=tmp_path)


def test_registry_missing_model(tmp_path: Path) -> None:
    assert list_models(tmp_path / "nowhere") == []
    with pytest.raises(FileNotFoundError, match="no model"):
        load_card("absent", root=tmp_path)


def test_registry_uses_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from openmind_core.models.registry import models_dir

    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    assert models_dir() == tmp_path / "models"
