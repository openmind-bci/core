"""Train a model on a public dataset and register it (``openmind-core train``).

The training data go through exactly the preprocessing of the online decoder
(causal filter, 2 s epochs, amplitude rejection, optional per-session alignment),
and the registry card records what was applied, so that
:class:`~openmind_core.online.OnlineDecoder` can repeat it on a live stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openmind_core.alignment import align_epochs
from openmind_core.datasets import MONTAGES, load_epochs
from openmind_core.models import make_model
from openmind_core.models.registry import models_dir, save_model, validate_name

#: Accepted values of ``--alignment``.
ALIGNMENTS = ("none", "euclid", "riemann")


def parse_subjects(text: str) -> list[int]:
    """Parse ``"1-3,5,8-9"`` into ``[1, 2, 3, 5, 8, 9]`` (sorted, without duplicates).

    Raises:
        ValueError: On an empty or malformed list, or a reversed range.
    """
    subjects: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                first, last = (int(x) for x in part.split("-", 1))
                if first > last:
                    raise ValueError
                subjects.update(range(first, last + 1))
            else:
                subjects.add(int(part))
        except ValueError:
            raise ValueError(f"invalid subject list {text!r}: use e.g. 1-9 or 1,3,5-7") from None
    if not subjects:
        raise ValueError("no subjects given")
    return sorted(subjects)


def resolve_channels(text: str) -> tuple[tuple[str, ...], str]:
    """Channels from a montage letter (``A``, ``C``) or a comma-separated list.

    Returns:
        ``(channels, label)`` where ``label`` is the montage letter (lower case)
        or ``"<n>ch"``.
    """
    key = text.strip().upper()
    if key in MONTAGES:
        return MONTAGES[key], key.lower()
    channels = tuple(c.strip() for c in text.split(",") if c.strip())
    if not channels:
        raise ValueError("no channels given")
    if len(set(channels)) != len(channels):
        raise ValueError(f"duplicate channels in {text!r}")
    return channels, f"{len(channels)}ch"


def default_name(model: str, dataset: str, montage: str, alignment: str) -> str:
    """Registry name ``<model>-<dataset>-<montage>[-<alignment>]``."""
    name = f"{model}-{dataset}-{montage}"
    return name if alignment == "none" else f"{name}-{alignment}"


def train_and_register(
    dataset: str,
    subjects: Sequence[int],
    model_name: str,
    classes: Sequence[str],
    channels: Sequence[str],
    alignment: str,
    name: str,
    overwrite: bool = False,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Load epochs, align, fit on all of them and register the model.

    Returns:
        The model directory and the training metadata stored in its card.

    Raises:
        ValueError: On an unknown alignment or a dataset without epochs.
        ImportError: Without the ``research`` extra (datasets) or, for deep
            models, the ``deep`` extra.
        FileExistsError: If the name is taken and ``overwrite`` is false.
    """
    if alignment not in ALIGNMENTS:
        raise ValueError(f"unknown alignment {alignment!r}; available: {list(ALIGNMENTS)}")
    validate_name(name)
    base = root or models_dir()
    if (base / name).exists() and not overwrite:
        raise FileExistsError(f"model {name!r} already exists in {base}")
    make_model(model_name)  # fail fast on an unknown model or a missing extra
    epochs = load_epochs(dataset, list(subjects), list(classes), channels=tuple(channels))
    if len(epochs) == 0:
        raise ValueError("no epochs left after loading and rejection")
    if alignment != "none":
        epochs = align_epochs(epochs, method=alignment, by="session")
    model = make_model(model_name, srate=epochs.srate)
    model.fit(epochs.X, epochs.y)
    filters = dict(epochs.info.get("filters", {}))
    training: dict[str, Any] = {
        "dataset": dataset,
        "subjects": list(subjects),
        "classes": list(classes),
        "n_epochs": len(epochs),
        "counts": epochs.counts(),
        "alignment": alignment,
        "preprocessing": filters,
    }
    path = save_model(
        model,
        name,
        epochs.channels,
        training=training,
        root=root,
        overwrite=overwrite,
        alignment=None if alignment == "none" else alignment,
        filters=filters,
        srate=epochs.srate,
        n_times=int(epochs.X.shape[2]),
        architecture=model_name,
    )
    return path, training
