"""Tests for the fine-tuning strategies (tiny networks, a few epochs; need the deep extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("braindecode")

import torch

from helpers import synthetic_epochs
from openmind_core.calibration import (
    STRATEGIES,
    FineTuneSettings,
    calibration_curve,
    finetune_strategy,
)
from openmind_core.datasets import EpochSet
from openmind_core.models import make_model
from openmind_core.models.deep import DeepClassifier
from openmind_core.models.registry import save_model

TINY = {"max_epochs": 2, "patience": 2, "device": "cpu", "batch_size": 16}

#: Children of the EEGNet each strategy is allowed to change.
CHANGED = {
    "head": {"final_layer"},
    "final": {"conv_separable_depth", "conv_separable_point", "bnorm_2", "final_layer"},
    "full": {
        "conv_temporal",
        "bnorm_temporal",
        "conv_spatial",
        "bnorm_1",
        "conv_separable_depth",
        "conv_separable_point",
        "bnorm_2",
        "final_layer",
    },
    "adapter": set(),
}


@pytest.fixture(scope="module")
def data() -> EpochSet:
    return synthetic_epochs(subjects=(1,), sessions=("0",), per_class=12, separation=4.0)


@pytest.fixture(scope="module")
def pretrained(data: EpochSet) -> DeepClassifier:
    return make_model("eegnet", srate=125.0, seed=3, **TINY).fit(data.X, data.y)


def changed_children(before: DeepClassifier, after: DeepClassifier) -> set[str]:
    a, b = before.module_.state_dict(), after.module_.state_dict()
    return {key.split(".")[0] for key in a if not torch.equal(a[key], b[key])}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_strategy_changes_only_its_parameters(
    strategy: str, pretrained: DeepClassifier, data: EpochSet
) -> None:
    snapshot = {k: v.clone() for k, v in pretrained.module_.state_dict().items()}
    tuned = pretrained.fine_tune(data.X[:15], data.y[:15], strategy, epochs=2, lr=1e-2)
    assert changed_children(pretrained, tuned) == CHANGED[strategy]
    # the source model is left untouched, and every parameter is trainable again
    for key, value in pretrained.module_.state_dict().items():
        assert torch.equal(value, snapshot[key])
    assert all(p.requires_grad for p in tuned.module_.parameters())
    if strategy == "adapter":
        assert float(tuned.adapter_.delta.detach().abs().sum()) > 0
    else:
        assert getattr(tuned, "adapter_", None) is None


def test_frozen_layers_keep_their_batch_norm_statistics(
    pretrained: DeepClassifier, data: EpochSet
) -> None:
    tuned = pretrained.fine_tune(data.X[:15], data.y[:15], "head", epochs=2, lr=1e-2)
    before, after = pretrained.module_.state_dict(), tuned.module_.state_dict()
    for key in before:
        if "running_" in key or "num_batches" in key:
            assert torch.equal(before[key], after[key]), key


def test_the_adapter_starts_as_the_identity(pretrained: DeepClassifier, data: EpochSet) -> None:
    tuned = pretrained.fine_tune(data.X[:15], data.y[:15], "adapter", epochs=0, lr=1e-2)
    np.testing.assert_allclose(
        tuned.predict_proba(data.X), pretrained.predict_proba(data.X), atol=1e-6
    )


def test_adapter_only_trains_the_adapter_and_is_used_to_predict(
    pretrained: DeepClassifier, data: EpochSet
) -> None:
    tuned = pretrained.fine_tune(data.X[:15], data.y[:15], "adapter", epochs=3, lr=5e-2)
    assert not np.allclose(tuned.predict_proba(data.X), pretrained.predict_proba(data.X))


def test_no_trial_returns_an_unchanged_copy(pretrained: DeepClassifier, data: EpochSet) -> None:
    tuned = pretrained.fine_tune(data.X[:0], data.y[:0], "full")
    assert tuned is not pretrained
    assert changed_children(pretrained, tuned) == set()
    np.testing.assert_array_equal(tuned.predict(data.X), pretrained.predict(data.X))


def test_fine_tuning_is_deterministic(pretrained: DeepClassifier, data: EpochSet) -> None:
    first = pretrained.fine_tune(data.X[:30], data.y[:30], "final", epochs=3, lr=1e-2)
    second = pretrained.fine_tune(data.X[:30], data.y[:30], "final", epochs=3, lr=1e-2)
    np.testing.assert_array_equal(first.predict_proba(data.X), second.predict_proba(data.X))


def test_early_stopping_needs_enough_trials(pretrained: DeepClassifier, data: EpochSet) -> None:
    few = pretrained.fine_tune(data.X[:15], data.y[:15], "head", epochs=4, patience=1, lr=1e-2)
    assert len(few.history_) == 4  # no validation set: fixed number of epochs
    assert "valid_loss" not in few.history_[0]
    many = pretrained.fine_tune(
        data.X[:30], data.y[:30], "head", epochs=4, patience=1, lr=1e-2, min_validation_trials=30
    )
    assert "valid_loss" in many.history_[0]


def test_sample_weights_change_the_result(pretrained: DeepClassifier, data: EpochSet) -> None:
    kwargs = {"strategy": "head", "epochs": 3, "lr": 1e-2}
    plain = pretrained.fine_tune(data.X[:15], data.y[:15], **kwargs)
    weights = np.linspace(0.1, 5.0, 15)
    weighted = pretrained.fine_tune(data.X[:15], data.y[:15], sample_weight=weights, **kwargs)
    assert not np.allclose(plain.predict_proba(data.X), weighted.predict_proba(data.X))
    with pytest.raises(ValueError, match="sample_weight"):
        pretrained.fine_tune(data.X[:15], data.y[:15], sample_weight=[1.0])


def test_fine_tune_validates_its_arguments(pretrained: DeepClassifier, data: EpochSet) -> None:
    with pytest.raises(ValueError, match="strategy"):
        pretrained.fine_tune(data.X[:15], data.y[:15], "everything")
    with pytest.raises(ValueError, match="classes"):
        pretrained.fine_tune(data.X[:15], np.array(["up"] * 15))
    with pytest.raises(ValueError, match="not trained"):
        DeepClassifier().fine_tune(data.X[:15], data.y[:15])


def test_registry_refuses_a_model_with_an_adapter(
    pretrained: DeepClassifier, data: EpochSet, tmp_path: Path
) -> None:
    tuned = pretrained.fine_tune(data.X[:15], data.y[:15], "adapter", epochs=1)
    with pytest.raises(ValueError, match="adapter"):
        save_model(tuned, "adapted", data.channels, root=tmp_path)


def test_settings_resolve_the_learning_rate_per_strategy() -> None:
    settings = FineTuneSettings.from_mapping({"epochs": 7, "lr": {"head": 0.5}})
    assert settings.learning_rate("head") == 0.5
    assert settings.learning_rate("full") < settings.learning_rate("head")  # default
    assert FineTuneSettings(lr=0.1).learning_rate("full") == 0.1
    assert settings.kwargs("head")["epochs"] == 7
    assert finetune_strategy("aligned+finetune-adapter") == "adapter"
    assert finetune_strategy("aligned") is None


@pytest.fixture(scope="module")
def three_subjects() -> EpochSet:
    epochs = synthetic_epochs(subjects=(1, 2, 3), sessions=("0",), per_class=14, separation=4.0)
    rng = np.random.default_rng(0)
    order = np.arange(len(epochs))
    for subject in epochs.subjects:
        rows = np.flatnonzero(epochs.subject == subject)
        order[rows] = rng.permutation(rows)
    return EpochSet(
        epochs.X[order],
        epochs.y[order],
        epochs.subject[order],
        epochs.session[order],
        epochs.channels,
        epochs.srate,
        epochs.dataset,
    )


def test_curve_runs_deep_finetune_conditions_over_seeds(three_subjects: EpochSet) -> None:
    conditions = ["aligned", "aligned+finetune-head", "aligned+finetune-adapter"]
    kwargs = {
        "models": {"eegnet": lambda: make_model("eegnet", srate=125.0, **TINY)},
        "conditions": conditions,
        "n_values": (0, 3),
        "subjects": [1],
        "min_test_per_class": 3,
        "unlabeled_start": 6,
        "finetune": {"epochs": 2, "min_validation_trials": 1000},
    }
    scores = calibration_curve(three_subjects, seeds=[0, 1], **kwargs)  # type: ignore[arg-type]
    by_key = {(s.condition, s.n): s for s in scores}
    assert set(by_key) == {(c, n) for c in conditions for n in (0, 3)}
    assert all(s.n_seeds == 2 for s in scores)
    # n = 0 is the pre-trained model, whatever the strategy
    for condition in conditions[1:]:
        assert by_key[(condition, 0)].accuracy == by_key[("aligned", 0)].accuracy
    # one seed: no spread
    single = calibration_curve(three_subjects, seeds=[0], **kwargs)  # type: ignore[arg-type]
    assert all(s.n_seeds == 1 and s.accuracy_seed_std == 0.0 for s in single)


def test_finetune_condition_needs_a_deep_model(three_subjects: EpochSet) -> None:
    with pytest.raises(ValueError, match="fine_tune"):
        calibration_curve(
            three_subjects,
            ["mdm"],
            conditions=["aligned+finetune-head"],
            n_values=(0, 3),
            subjects=[1],
            min_test_per_class=3,
            unlabeled_start=6,
        )
