"""Staged model update during calibration: a personal classifier blended in.

The generic model (trained on public data) is the only one available at the
start of a session. As the user's labelled epochs come in (every completed
trial, see :class:`~openmind_core.calibration.CalibrationRecorder`), a small
**personal classifier** (tangent space + logistic regression, trained on the
user's aligned epochs only) is fitted after each block, or every
``update_every`` trials, once each class has at least ``min_per_class`` epochs.

The online probabilities are a convex blend::

    p = (1 - w) * p_generic + w * p_personal,   w = min(1, n_min / target_per_class)

where ``n_min`` is the size of the rarest class when the personal model was last
fitted. This is deliberately simple and safe: no source data is needed online,
``w`` is 0 until the personal model exists and reaches 1 only with
``target_per_class`` epochs of every class, and a poor personal model trained on
few epochs weighs little. Fine-tuning a deep model is a later task.

``assistance = 1 - w`` is the share of the decision that still comes from the
generic model; it decreases as the user calibrates (``status.assistance``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import numpy.typing as npt

from openmind_core.alignment import Aligner
from openmind_core.calibration.session import CalibrationRecorder
from openmind_core.io import Trial
from openmind_core.models.baselines import ts_lr
from openmind_core.models.registry import ModelCard
from openmind_core.online.evidence import Decision
from openmind_core.profiles import ProfileStore
from openmind_core.scoring import BlockScore, ProgressTracker, ScoreConfig

if TYPE_CHECKING:
    from openmind_core.online.adaptation import PseudoLabelTuner

FloatArray = npt.NDArray[np.float64]

#: File of the personal classifier in the user's ``weights`` directory.
PERSONAL_MODEL = "personal.joblib"


@dataclass(frozen=True)
class PersonalState:
    """Snapshot of the fitted part of a :class:`PersonalModel` (for rollback)."""

    personal: Any
    weight: float
    updates: int


class PersonalModel:
    """Generic model + personal classifier, usable as ``OnlineDecoder.model``.

    Args:
        generic: Generic model with ``predict_proba`` (columns in the card's
            class order, on aligned windows if the card says so).
        card: Its registry card.
        recorder: Source of the user's labelled epochs.
        tracker: Progress tracker fed with the trials (default: one over the
            card's classes with blocks closed by :meth:`end_block`).
        min_per_class: Epochs of every class needed to fit the personal model.
        target_per_class: Epochs per class at which the personal model has full
            weight.
        update_every: Also refit every this many completed trials (``None``:
            only at block ends).
        reference: Reference covariance the windows are aligned with (needed
            when the card has an alignment); can be set later with
            :meth:`set_reference`.
        c: Regularization of the personal logistic regression.
    """

    def __init__(
        self,
        generic: Any,
        card: ModelCard,
        recorder: CalibrationRecorder,
        tracker: ProgressTracker | None = None,
        min_per_class: int = 5,
        target_per_class: int = 20,
        update_every: int | None = None,
        reference: FloatArray | None = None,
        c: float = 1.0,
    ) -> None:
        if min_per_class < 2 or target_per_class < min_per_class:
            raise ValueError("need 2 <= min_per_class <= target_per_class")
        if update_every is not None and update_every < 1:
            raise ValueError("update_every must be >= 1")
        self.generic = generic
        self.card = card
        self.classes_ = np.array(card.classes)
        self.recorder = recorder
        self.tracker = tracker or ProgressTracker(card.classes, ScoreConfig(block_size=None))
        self.min_per_class = min_per_class
        self.target_per_class = target_per_class
        self.update_every = update_every
        self.c = c
        self.weight = 0.0
        self.updates = 0
        self._personal: Any = None
        self._aligner: Aligner | None = None
        self._since_update = 0
        #: Optional source of pseudo-labelled windows (see ``online.adaptation``).
        self.pseudo: PseudoLabelTuner | None = None
        self.set_reference(reference)

    # -- state -------------------------------------------------------------------

    @property
    def assistance(self) -> float:
        """Share of the decision still owed to the generic model, in ``[0, 1]``."""
        return 1.0 - self.weight

    @property
    def fitted(self) -> bool:
        """Whether a personal classifier exists."""
        return self._personal is not None

    def set_reference(self, covariance: FloatArray | None) -> None:
        """Reference covariance of the session (``decoder.reference_covariance``)."""
        self._aligner = (
            None
            if covariance is None or self.card.alignment is None
            else Aligner.from_reference(covariance, self.card.alignment)
        )
        self._reference_set = covariance is not None

    def get_state(self) -> PersonalState:
        """Snapshot of the personal classifier (refits build new objects, never mutate)."""
        return PersonalState(self._personal, self.weight, self.updates)

    def set_state(self, state: PersonalState) -> None:
        """Restore a snapshot taken with :meth:`get_state`."""
        self._personal, self.weight, self.updates = state.personal, state.weight, state.updates

    # -- prediction ----------------------------------------------------------------

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Blended class probabilities of aligned windows ``(n, n_channels, n_times)``."""
        generic = np.asarray(self.generic.predict_proba(X), dtype=np.float64)
        if self._personal is None or self.weight <= 0.0:
            return generic
        personal = np.zeros_like(generic)
        order = {str(c): i for i, c in enumerate(self._personal.classes_)}
        proba = self._personal.predict_proba(np.asarray(X))
        for j, name in enumerate(self.card.classes):
            personal[:, j] = proba[:, order[name]]
        return np.asarray((1.0 - self.weight) * generic + self.weight * personal)

    # -- session events ---------------------------------------------------------------

    def trial_done(self, trial: Trial, decisions: Sequence[Decision]) -> BlockScore | None:
        """A trial completed: score it with the decisions taken meanwhile.

        Returns the block score if the tracker closes a block by size; refits
        every ``update_every`` trials.
        """
        score = self.tracker.add_trial(trial, decisions)
        self._since_update += 1
        if self.update_every is not None and self._since_update >= self.update_every:
            self.update()
        return score

    def end_block(self) -> BlockScore | None:
        """``block_end`` marker: close the score block and refit the personal model."""
        score = self.tracker.close_block() if self.tracker.pending_trials else None
        self.update()
        return score

    def update(self) -> bool:
        """Refit the personal classifier on every recorded epoch if the classes allow.

        Returns whether a model was fitted. Epochs of labels unknown to the
        generic model are ignored.
        """
        self._since_update = 0
        labels = np.array(self.recorder.labels())
        if len(labels) == 0:
            return False
        counts = [int(np.sum(labels == c)) for c in self.card.classes]
        if min(counts) < self.min_per_class:
            return False
        X = self.recorder.epochs()
        keep = np.isin(labels, self.card.classes)
        X, y = X[keep], labels[keep]
        if self.card.alignment is not None:
            if self._aligner is None:
                return False  # the windows' reference is not known yet
            X = np.asarray(self._aligner.transform(X), dtype=np.float64)
        if self.pseudo is not None:
            extra_x, extra_y = self.pseudo.training_set(
                {c: int(np.sum(y == c)) for c in self.card.classes}
            )
            if len(extra_y) > 0:
                if self._aligner is not None:
                    extra_x = np.asarray(self._aligner.transform(extra_x), dtype=np.float64)
                X = np.concatenate([X, extra_x])
                y = np.concatenate([y, extra_y])
        model = ts_lr(self.c)
        model.fit(X, y)
        self._personal = model
        self.weight = min(1.0, min(counts) / self.target_per_class)
        self.updates += 1
        return True

    # -- persistence ---------------------------------------------------------------------

    def to_record(self, user: str, ts: float | None = None) -> dict[str, Any]:
        """Score record of the tracker, plus ``assistance`` and ``personal_weight``."""
        record = self.tracker.to_record(user, ts)
        record["assistance"] = self.assistance
        record["personal_weight"] = self.weight
        return record

    def save(self, store: ProfileStore, user: str) -> None:
        """Write the personal classifier and the score into the profile of ``user``.

        The classifier goes to ``weights/personal.joblib`` (a joblib pickle:
        load only your own files) and one record is appended to the scores.
        """
        if self._personal is not None:
            directory = store.weights_dir(user, create=True)
            joblib.dump(
                {
                    "model": self._personal,
                    "classes": list(self.card.classes),
                    "weight": self.weight,
                    "alignment": self.card.alignment,
                    "epochs": self.recorder.counts_per_class(),
                },
                directory / PERSONAL_MODEL,
            )
        store.append_score(user, self.to_record(user))
