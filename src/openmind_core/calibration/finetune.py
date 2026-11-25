"""Fine-tuning strategies for a pre-trained deep decoder (settings, no torch needed).

A decoder pre-trained on other users is adapted to a new user with a handful of
calibration trials. The strategies differ in *what* is allowed to change:

=============  ==============================================================
``head``       the output layer only (a few hundred weights)
``final``      the last block and the output layer
``adapter``    a light per-user 1x1 spatial layer inserted before the network
               (identity at the start); the network itself is frozen
``full``       every weight, with a low learning rate
=============  ==============================================================

The training itself is :meth:`openmind_core.models.deep.DeepClassifier.fine_tune`;
this module holds the strategy names and the settings that the calibration
curve passes to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

#: Supported strategies, from the least to the most flexible.
STRATEGIES: Final[tuple[str, ...]] = ("head", "final", "adapter", "full")

#: Default learning rate of each strategy (the whole network moves with ``full``).
DEFAULT_LR: Final[dict[str, float]] = {
    "head": 2e-3,
    "final": 1e-3,
    "adapter": 2e-3,
    "full": 3e-4,
}


@dataclass(frozen=True)
class FineTuneSettings:
    """Training settings of the fine-tuning step.

    Attributes:
        epochs: Maximum number of passes over the calibration trials.
        lr: Learning rate, one value for every strategy or one per strategy.
        patience: Epochs without validation improvement before stopping.
        valid_fraction: Share of the calibration trials held out for early
            stopping, when there are at least ``min_validation_trials``.
        min_validation_trials: Below this many calibration trials nothing is
            held out and training runs for exactly ``epochs``.
    """

    epochs: int = 100
    lr: float | Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_LR))
    patience: int = 20
    valid_fraction: float = 0.2
    min_validation_trials: int = 30

    def learning_rate(self, strategy: str) -> float:
        """Learning rate for ``strategy``."""
        if isinstance(self.lr, Mapping):
            return float(self.lr.get(strategy, DEFAULT_LR[strategy]))
        return float(self.lr)

    def kwargs(self, strategy: str) -> dict[str, Any]:
        """Arguments of ``DeepClassifier.fine_tune`` for ``strategy``."""
        return {
            "strategy": strategy,
            "epochs": self.epochs,
            "lr": self.learning_rate(strategy),
            "patience": self.patience,
            "valid_fraction": self.valid_fraction,
            "min_validation_trials": self.min_validation_trials,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> FineTuneSettings:
        """Settings from a configuration mapping (missing keys keep their default)."""
        return cls(**dict(raw or {}))
