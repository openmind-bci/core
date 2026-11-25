"""Local registry of trained models, deep and classical.

Each model lives in ``$OPENMIND_DATA_DIR/models/<name>/``, never in Git:

- ``weights.pt`` (deep models, ``format="torch"``): the network's ``state_dict``
  only (tensors, no pickled code), read back with
  ``torch.load(weights_only=True)``;
- ``model.joblib`` (baselines, ``format="sklearn"``): the fitted scikit-learn
  pipeline. **Joblib files are pickles: loading one executes code.** Only load
  models you trained yourself or trust (the local registry); never a file
  downloaded from an unknown source;
- ``card.json``: everything needed to rebuild and use the model (architecture,
  channels, sampling rate, epoch length, classes, input scale, alignment and
  filters of the training data) and where it comes from (training data,
  metrics, OpenMind version, date).

Torch is imported only when a deep model is saved or loaded, so the registry
works without the ``deep`` extra for classical models.

A model trained on public data can be shared as these two files, for instance
on Hugging Face with a model card generated from ``card.json``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from openmind_core import __version__
from openmind_core.paths import data_dir

WEIGHTS: Final = "weights.pt"
SKLEARN_MODEL: Final = "model.joblib"
FORMATS: Final[tuple[str, ...]] = ("torch", "sklearn")
CARD: Final = "card.json"
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ModelCard:
    """Metadata of a registered model.

    Attributes:
        name: Registry name (lower case, digits, ``.``, ``_``, ``-``).
        architecture: Network architecture (see ``models.deep.ARCHITECTURES``).
        channels: Input channel labels, in order.
        srate: Sampling rate of the inputs, in Hz.
        n_times: Samples per input epoch.
        classes: Output labels, in the order of the network outputs.
        scale: Per-channel divisor applied to µV inputs.
        params: Training hyper-parameters.
        training: Free description of the training data and results.
        openmind_core: Version of OpenMind Core that trained the model.
        created: ISO 8601 creation time (UTC).
        format: ``"torch"`` (``weights.pt``) or ``"sklearn"`` (``model.joblib``).
        alignment: Method the training epochs were aligned with (``None``,
            ``"euclid"`` or ``"riemann"``); the online decoder applies the same.
        filters: Band-pass and notch of the training epochs (``FilterConfig``
            fields); empty when unknown.
    """

    name: str
    architecture: str
    channels: tuple[str, ...]
    srate: float
    n_times: int
    classes: tuple[str, ...]
    scale: tuple[float, ...]
    params: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    openmind_core: str = __version__
    created: str = ""
    format: str = "torch"
    alignment: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)


def models_dir() -> Path:
    """Default registry location: ``$OPENMIND_DATA_DIR/models``."""
    return data_dir() / "models"


def validate_name(name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is a valid registry name."""
    if not _NAME.match(name):
        raise ValueError(f"invalid model name {name!r}: use a-z, 0-9, '.', '_' or '-'")


def _is_deep(model: Any) -> bool:
    return type(model).__module__ == "openmind_core.models.deep"


def save_model(
    model: Any,
    name: str,
    channels: tuple[str, ...] | list[str],
    training: dict[str, Any] | None = None,
    root: Path | None = None,
    overwrite: bool = False,
    *,
    alignment: str | None = None,
    filters: dict[str, Any] | None = None,
    srate: float | None = None,
    n_times: int | None = None,
    architecture: str | None = None,
) -> Path:
    """Register a trained model and return its directory.

    Args:
        model: A trained ``DeepClassifier`` or a fitted baseline pipeline
            (anything with ``predict_proba`` and ``classes_``).
        name: Registry name.
        channels: Input channel labels, in order.
        training: Free description of the training data and results.
        root: Registry location (default :func:`models_dir`).
        overwrite: Replace an existing model.
        alignment: Alignment applied to the training epochs.
        filters: Filter settings of the training epochs.
        srate: Sampling rate of the inputs; required for baselines (deep
            models carry it).
        n_times: Samples per epoch; required for baselines.
        architecture: Name stored in the card of a baseline (default
            ``"sklearn"``); deep models use their own.

    Raises:
        ValueError: If the name is invalid, the model is not trained, the
            channels do not match its input or a baseline lacks ``srate`` or
            ``n_times``.
        FileExistsError: If the model exists and ``overwrite`` is false.
    """
    validate_name(name)
    if alignment is not None and alignment not in ("euclid", "riemann"):
        raise ValueError(f"unknown alignment {alignment!r}")
    if getattr(model, "research_only", False):
        raise ValueError(
            f"{type(model).__name__} models are research only for now and cannot be registered"
        )
    deep = _is_deep(model)
    if deep:
        if not hasattr(model, "module_"):
            raise ValueError("the model is not trained")
        if getattr(model, "adapter_", None) is not None:
            raise ValueError("a model fine-tuned with an adapter cannot be registered yet")
        if len(channels) != len(model.scale_):
            raise ValueError(f"{len(channels)} channels given, the model takes {len(model.scale_)}")
        card_srate, card_n_times = float(model.srate), int(model.n_times_)
        card_architecture, scale = model.architecture, tuple(float(s) for s in model.scale_)
        params = model.get_params()
    else:
        if not (hasattr(model, "predict_proba") and hasattr(model, "classes_")):
            raise ValueError("the model is not trained (no predict_proba or classes_)")
        if srate is None or n_times is None:
            raise ValueError("srate and n_times are required to register a baseline")
        card_srate, card_n_times = float(srate), int(n_times)
        card_architecture, scale = architecture or "sklearn", (1.0,) * len(channels)
        params = {}
    target = (root or models_dir()) / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"model {name!r} already exists in {target.parent}")
    target.mkdir(parents=True, exist_ok=True)
    card = ModelCard(
        name=name,
        architecture=card_architecture,
        channels=tuple(channels),
        srate=card_srate,
        n_times=card_n_times,
        classes=tuple(str(c) for c in model.classes_),
        scale=scale,
        params=params,
        training=training or {},
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        format="torch" if deep else "sklearn",
        alignment=alignment,
        filters=dict(filters or {}),
    )
    if deep:
        import torch

        torch.save(model.module_.state_dict(), target / WEIGHTS)
    else:
        import joblib

        joblib.dump(model, target / SKLEARN_MODEL)
    (target / CARD).write_text(json.dumps(asdict(card), indent=2) + "\n")
    return target


def load_card(name: str, root: Path | None = None) -> ModelCard:
    """Read the card of a registered model.

    Raises:
        FileNotFoundError: If the model is not registered.
    """
    path = (root or models_dir()) / name / CARD
    if not path.exists():
        raise FileNotFoundError(f"no model {name!r} in {path.parent.parent}")
    raw = json.loads(path.read_text())
    for key in ("channels", "classes", "scale"):
        raw[key] = tuple(raw[key])
    return ModelCard(**raw)


def load_model(name: str, root: Path | None = None) -> tuple[Any, ModelCard]:
    """Rebuild a registered model, ready to ``predict_proba``.

    Warning:
        A ``sklearn`` model is a joblib (pickle) file: loading it executes code.
        Only load models you trained or trust.

    Raises:
        FileNotFoundError: If the model is not registered.
        ValueError: If the card declares an unknown format.
    """
    card = load_card(name, root)
    folder = (root or models_dir()) / name
    if card.format == "sklearn":
        import joblib

        return joblib.load(folder / SKLEARN_MODEL), card
    if card.format != "torch":
        raise ValueError(f"unknown model format {card.format!r}; available: {list(FORMATS)}")
    import torch

    from openmind_core.models.deep import DeepClassifier, build_network

    model = DeepClassifier(**card.params)
    model.classes_ = np.array(card.classes)
    model.scale_ = np.array(card.scale, dtype=np.float32)
    model.n_times_ = card.n_times
    module = build_network(
        card.architecture, len(card.channels), len(card.classes), card.n_times, card.srate
    )
    state = torch.load(folder / WEIGHTS, weights_only=True)
    module.load_state_dict(state)
    model.module_ = module.eval()
    return model, card


def list_models(root: Path | None = None) -> list[ModelCard]:
    """Cards of every registered model, sorted by name."""
    base = root or models_dir()
    if not base.exists():
        return []
    return [load_card(path.parent.name, base) for path in sorted(base.glob(f"*/{CARD}"))]
