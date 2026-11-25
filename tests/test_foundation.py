"""Tests of the LaBraM foundation classifier (random weights; no download in CI)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("braindecode")

import torch

from helpers import synthetic_epochs
from openmind_core.datasets import EpochSet
from openmind_core.models import FOUNDATION_MODELS, make_model, model_names
from openmind_core.models.foundation import FoundationClassifier
from openmind_core.models.registry import save_model

TINY = {
    "pretrained": False,
    "encoder_params": {"num_layers": 2},
    "device": "cpu",
    "max_epochs": 2,
    "patience": 2,
    "batch_size": 16,
}
NET_PREFIXES = {
    "head": {"final_layer"},
    "final": {"blocks", "norm", "fc_norm", "final_layer"},
    "adapter": set(),
}


@pytest.fixture(scope="module")
def data() -> EpochSet:
    return synthetic_epochs(subjects=(1,), sessions=("0",), per_class=12, separation=4.0)


@pytest.fixture(scope="module")
def source(data: EpochSet) -> FoundationClassifier:
    model = make_model("labram", srate=125.0, seed=1, ch_names=data.channels, **TINY)
    assert isinstance(model, FoundationClassifier)
    return model.fit(data.X, data.y)


def changed(before: FoundationClassifier, after: FoundationClassifier) -> set[str]:
    a, b = before.module_.state_dict(), after.module_.state_dict()
    return {key.split(".")[1] for key in a if not torch.equal(a[key], b[key])}


def test_epochs_are_resampled_to_the_model_rate(
    source: FoundationClassifier, data: EpochSet
) -> None:
    assert data.X.shape[2] == 250  # 2 s at 125 Hz
    assert source.n_times_ == 400  # 2 s at 200 Hz: two one-second patches
    normalized = source._normalize(data.X)
    assert normalized.shape == (len(data.X), len(data.channels), 400)
    np.testing.assert_allclose(normalized.std(axis=(0, 2)), source.target_std, rtol=0.1)
    assert source.predict_proba(data.X).shape == (len(data.X), 3)


def test_the_headset_rate_is_resampled_by_a_rational_factor() -> None:
    model = FoundationClassifier(srate=250.0)
    model.scale_ = np.ones(2, dtype=np.float32)
    assert model._normalize(np.zeros((1, 2, 500), dtype=np.float32)).shape == (1, 2, 400)


def test_fit_with_a_frozen_encoder_trains_only_the_head(data: EpochSet) -> None:
    model = make_model(
        "labram", srate=125.0, seed=1, ch_names=data.channels, fit_strategy="head", **TINY
    )
    torch.manual_seed(1)
    start = model._build_encoder(len(data.channels), 3, 400).state_dict()
    model.fit(data.X, data.y)
    after = model.module_.net.state_dict()
    moved = {k.split(".")[0] for k, v in after.items() if not torch.equal(v, start[k])}
    # same seed as fit: the encoder is initialized identically, only the head moved
    assert "final_layer" in moved
    assert not moved - {"final_layer"}


@pytest.mark.parametrize("strategy", ["head", "final", "adapter", "full"])
def test_fine_tuning_strategies_change_the_expected_parameters(
    strategy: str, source: FoundationClassifier, data: EpochSet
) -> None:
    before = {k: v.clone() for k, v in source.module_.state_dict().items()}
    tuned = source.fine_tune(data.X[:15], data.y[:15], strategy, epochs=2, lr=1e-2)
    if strategy == "full":
        assert {"patch_embed", "blocks", "final_layer"} <= changed(source, tuned)
    else:
        assert not changed(source, tuned) - NET_PREFIXES[strategy]
        assert ("final_layer" in changed(source, tuned)) == (strategy != "adapter")
    # "final" leaves the first block alone, the source model is untouched
    if strategy == "final":
        key = "net.blocks.0."
        assert all(
            torch.equal(v, tuned.module_.state_dict()[k]) for k, v in before.items() if key in k
        )
    for k, v in source.module_.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.requires_grad for p in tuned.module_.parameters())
    assert (getattr(tuned, "adapter_", None) is not None) == (strategy == "adapter")


def test_no_trial_returns_an_unchanged_copy(source: FoundationClassifier, data: EpochSet) -> None:
    tuned = source.fine_tune(data.X[:0], data.y[:0])
    np.testing.assert_allclose(tuned.predict_proba(data.X), source.predict_proba(data.X))
    assert tuned is not source


def test_make_model_imports_the_foundation_module_lazily() -> None:
    assert "labram" in FOUNDATION_MODELS and "labram" in model_names()
    for name in [m for m in sys.modules if m == "openmind_core.models.foundation"]:
        del sys.modules[name]
    importlib.reload(importlib.import_module("openmind_core.models"))
    assert "openmind_core.models.foundation" not in sys.modules
    model = importlib.import_module("openmind_core.models").make_model("labram", ch_names=["C3"])
    assert type(model).__name__ == "FoundationClassifier"


def test_it_is_an_estimator_with_a_seed(data: EpochSet) -> None:
    from sklearn.base import clone

    model = make_model("labram", seed=4, ch_names=list(data.channels), **TINY)
    assert clone(model).get_params(deep=False)["seed"] == 4
    assert "architecture" not in model.get_params(deep=False)


def test_channel_names_are_checked(data: EpochSet) -> None:
    with pytest.raises(ValueError, match="needs ch_names"):
        make_model("labram", **TINY).fit(data.X, data.y)
    with pytest.raises(ValueError, match="channel names given"):
        make_model("labram", ch_names=["C3"], **TINY).fit(data.X, data.y)
    names = ["XX", *data.channels[1:]]
    with pytest.raises(ValueError, match="unknown to LaBraM"):
        make_model("labram", ch_names=names, **TINY).fit(data.X, data.y)
    with pytest.raises(ValueError, match="fit_strategy"):
        make_model("labram", ch_names=data.channels, fit_strategy="adapter", **TINY).fit(
            data.X, data.y
        )


def test_loading_failures_are_explained(data: EpochSet, monkeypatch: pytest.MonkeyPatch) -> None:
    from braindecode.models import Labram

    def offline(*args: object, **kwargs: object) -> None:
        raise OSError("no network")

    monkeypatch.setattr(Labram, "from_pretrained", offline)
    model = make_model("labram", ch_names=data.channels, device="cpu", srate=125.0)
    with pytest.raises(RuntimeError, match="Hugging Face"):
        model.fit(data.X, data.y)

    def missing(*args: object, **kwargs: object) -> None:
        raise ImportError("hub")

    monkeypatch.setattr(Labram, "from_pretrained", missing)
    with pytest.raises(ImportError, match="foundation extra"):
        model.fit(data.X, data.y)


def test_registry_refuses_foundation_models(
    source: FoundationClassifier, data: EpochSet, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="research only"):
        save_model(source, "labram-test", data.channels, root=tmp_path)


@pytest.mark.slow
def test_real_pretrained_weights_load_with_montage_c_names() -> None:
    names = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "CPz"]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 8, 500)).astype(np.float32)
    y = np.array(["left", "right", "forward"] * 8)
    model = FoundationClassifier(
        ch_names=names, srate=250.0, fit_strategy="head", max_epochs=1, device="cpu"
    ).fit(x, y)
    assert model.predict_proba(x).shape == (24, 3)
    # the transformer carries pre-trained weights, not a random initialization
    fresh = FoundationClassifier(ch_names=names, pretrained=False)._build_encoder(8, 3, 400)
    pre = model.module_.net.state_dict()["blocks.0.attn.qkv.weight"]
    assert not torch.equal(pre, fresh.state_dict()["blocks.0.attn.qkv.weight"])
