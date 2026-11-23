"""Local registry of trained deep models.

Each model lives in ``$OPENMIND_DATA_DIR/models/<name>/``, never in Git:

- ``weights.pt``: the network's ``state_dict`` only (tensors, no pickled code),
  read back with ``torch.load(weights_only=True)``;
- ``card.json``: everything needed to rebuild and use the model (architecture,
  channels, sampling rate, epoch length, classes, input scale) and where it
  comes from (training data, metrics, OpenMind version, date).

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
import torch

from openmind_core import __version__
from openmind_core.models.deep import DeepClassifier, build_network
from openmind_core.paths import data_dir

WEIGHTS: Final = "weights.pt"
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


def models_dir() -> Path:
    """Default registry location: ``$OPENMIND_DATA_DIR/models``."""
    return data_dir() / "models"


def save_model(
    model: DeepClassifier,
    name: str,
    channels: tuple[str, ...] | list[str],
    training: dict[str, Any] | None = None,
    root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Register a trained model and return its directory.

    Raises:
        ValueError: If the name is invalid, the model is not trained or the
            channels do not match its input.
        FileExistsError: If the model exists and ``overwrite`` is false.
    """
    if not _NAME.match(name):
        raise ValueError(f"invalid model name {name!r}: use a-z, 0-9, '.', '_' or '-'")
    if not hasattr(model, "module_"):
        raise ValueError("the model is not trained")
    if getattr(model, "adapter_", None) is not None:
        raise ValueError("a model fine-tuned with an adapter cannot be registered yet")
    if len(channels) != len(model.scale_):
        raise ValueError(f"{len(channels)} channels given, the model takes {len(model.scale_)}")
    target = (root or models_dir()) / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"model {name!r} already exists in {target.parent}")
    target.mkdir(parents=True, exist_ok=True)
    params = model.get_params()
    card = ModelCard(
        name=name,
        architecture=model.architecture,
        channels=tuple(channels),
        srate=float(model.srate),
        n_times=int(model.n_times_),
        classes=tuple(str(c) for c in model.classes_),
        scale=tuple(float(s) for s in model.scale_),
        params=params,
        training=training or {},
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    torch.save(model.module_.state_dict(), target / WEIGHTS)
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


def load_model(name: str, root: Path | None = None) -> tuple[DeepClassifier, ModelCard]:
    """Rebuild a registered model, ready to predict."""
    card = load_card(name, root)
    model = DeepClassifier(**card.params)
    model.classes_ = np.array(card.classes)
    model.scale_ = np.array(card.scale, dtype=np.float32)
    model.n_times_ = card.n_times
    module = build_network(
        card.architecture, len(card.channels), len(card.classes), card.n_times, card.srate
    )
    state = torch.load((root or models_dir()) / name / WEIGHTS, weights_only=True)
    module.load_state_dict(state)
    model.module_ = module.eval()
    return model, card


def list_models(root: Path | None = None) -> list[ModelCard]:
    """Cards of every registered model, sorted by name."""
    base = root or models_dir()
    if not base.exists():
        return []
    return [load_card(path.parent.name, base) for path in sorted(base.glob(f"*/{CARD}"))]
