"""Decoders: classical baselines, deep networks and the local model registry.

Deep models (:mod:`openmind_core.models.deep`, :mod:`openmind_core.models.registry`)
need the ``deep`` extra and are imported on demand.
"""

from __future__ import annotations

from typing import Any, Final

from openmind_core.models.baselines import BASELINES, csp_lda, make_baseline, mdm, ts_lr

#: Deep architectures, available with the ``deep`` extra.
DEEP_MODELS: Final[tuple[str, ...]] = ("eegnet", "shallow")


def model_names() -> tuple[str, ...]:
    """Every model :func:`make_model` can build."""
    return (*sorted(BASELINES), *DEEP_MODELS)


def make_model(name: str, srate: float = 250.0, seed: int = 0, **params: Any) -> Any:
    """Build a fresh, unfitted classifier, classical or deep, by name.

    Args:
        name: A baseline (``csp-lda``, ``mdm``, ``ts-lr``) or a deep
            architecture (``eegnet``, ``shallow``).
        srate: Sampling rate of the epochs (deep models only).
        seed: Random seed (deep models only).
        **params: Training settings of deep models (see ``DeepClassifier``).

    Raises:
        KeyError: If the name is unknown.
        ImportError: For a deep model without the ``deep`` extra.
    """
    if name in BASELINES:
        if params:
            raise ValueError(f"baseline {name!r} takes no parameters, got {sorted(params)}")
        return make_baseline(name)
    if name in DEEP_MODELS:
        try:
            from openmind_core.models.deep import DeepClassifier
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("deep models need the deep extra (torch, braindecode)") from error
        return DeepClassifier(architecture=name, srate=srate, seed=seed, **params)
    raise KeyError(f"unknown model {name!r}; available: {list(model_names())}")


__all__ = [
    "BASELINES",
    "DEEP_MODELS",
    "csp_lda",
    "make_baseline",
    "make_model",
    "mdm",
    "model_names",
    "ts_lr",
]
