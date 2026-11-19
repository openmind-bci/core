"""Evaluation protocols for decoders, on preprocessed :class:`EpochSet` objects.

The protocols follow the MOABB definitions, but run on OpenMind's own epochs:
MOABB's evaluation classes re-run their own (zero-phase) preprocessing, which
would no longer match what the online decoder sees.

=================  ===========================================================
``within-session``  Stratified k-fold inside each session of each subject. The
                    optimistic reference: same subject, same day, same cap.
``cross-session``   Per subject, train on the other sessions and test on one.
                    Measures day-to-day drift; needs two sessions or more.
``cross-subject``   Leave one subject out: train on every other subject, test
                    on all epochs of the held-out one. This is the situation of
                    a new user before any calibration.
=================  ===========================================================

A random split alone is never enough: always report a cross-session or
cross-subject score next to a within-session one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from openmind_core.datasets.epochs import EpochSet

logger = logging.getLogger(__name__)

WITHIN_SESSION: Final = "within-session"
CROSS_SESSION: Final = "cross-session"
CROSS_SUBJECT: Final = "cross-subject"

#: Supported protocols, from the most to the least optimistic.
PROTOCOLS: Final[tuple[str, ...]] = (WITHIN_SESSION, CROSS_SESSION, CROSS_SUBJECT)

#: A function returning a fresh, unfitted scikit-learn classifier.
ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class Score:
    """Result of one evaluation unit.

    Attributes:
        pipeline: Name of the decoder.
        protocol: One of :data:`PROTOCOLS`.
        subject: Tested subject.
        session: Tested session (``"all"`` for cross-subject).
        n_train: Number of training epochs (per fold for within-session).
        n_test: Number of tested epochs.
        accuracy: Fraction of correct predictions.
        balanced_accuracy: Mean recall over classes (robust to imbalance).
        kappa: Cohen's kappa, 0 at chance level (BCI Competition IV metric).
    """

    pipeline: str
    protocol: str
    subject: int
    session: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    kappa: float


@dataclass(frozen=True)
class Summary:
    """Scores of one pipeline under one protocol, averaged over subjects.

    Sessions are averaged within each subject first, so every subject weighs
    the same.
    """

    pipeline: str
    protocol: str
    n_subjects: int
    accuracy_mean: float
    accuracy_std: float
    kappa_mean: float
    chance: float


def within_session(
    epochs: EpochSet,
    factory: ModelFactory,
    pipeline: str = "model",
    n_splits: int = 5,
    seed: int = 0,
) -> list[Score]:
    """Stratified k-fold inside each (subject, session).

    Predictions of the ``n_splits`` folds are pooled, giving one score per
    session. Sessions with fewer than ``n_splits`` epochs in some class are
    skipped with a warning.
    """
    scores: list[Score] = []
    for subject in epochs.subjects:
        for session in _sessions(epochs.select(epochs.subject == subject)):
            part = epochs.select((epochs.subject == subject) & (epochs.session == session))
            if min(part.counts().values()) < n_splits or len(part.classes) < 2:
                logger.warning(
                    "skip subject %s session %s: too few epochs per class", subject, session
                )
                continue
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            predicted = np.empty(len(part), dtype=part.y.dtype)
            n_train = 0
            for train, test in folds.split(part.X, part.y):
                model = factory()
                model.fit(part.X[train], part.y[train])
                predicted[test] = model.predict(part.X[test])
                n_train = len(train)
            scores.append(
                _score(pipeline, WITHIN_SESSION, subject, session, n_train, part.y, predicted)
            )
    return scores


def cross_session(epochs: EpochSet, factory: ModelFactory, pipeline: str = "model") -> list[Score]:
    """Per subject, train on all other sessions and test on each session in turn.

    Subjects with a single session are skipped.
    """
    scores: list[Score] = []
    for subject in epochs.subjects:
        own = epochs.select(epochs.subject == subject)
        sessions = _sessions(own)
        if len(sessions) < 2:
            logger.warning("skip subject %s: a single session", subject)
            continue
        for session in sessions:
            test = own.session == session
            model = factory()
            model.fit(own.X[~test], own.y[~test])
            predicted = model.predict(own.X[test])
            scores.append(
                _score(
                    pipeline,
                    CROSS_SESSION,
                    subject,
                    session,
                    int((~test).sum()),
                    own.y[test],
                    predicted,
                )
            )
    return scores


def cross_subject(epochs: EpochSet, factory: ModelFactory, pipeline: str = "model") -> list[Score]:
    """Leave one subject out. Needs at least two subjects."""
    if len(epochs.subjects) < 2:
        logger.warning("skip cross-subject evaluation: a single subject")
        return []
    scores: list[Score] = []
    for subject in epochs.subjects:
        test = epochs.subject == subject
        model = factory()
        model.fit(epochs.X[~test], epochs.y[~test])
        predicted = model.predict(epochs.X[test])
        scores.append(
            _score(
                pipeline,
                CROSS_SUBJECT,
                subject,
                "all",
                int((~test).sum()),
                epochs.y[test],
                predicted,
            )
        )
    return scores


def evaluate(
    epochs: EpochSet,
    pipelines: Mapping[str, ModelFactory],
    protocols: Sequence[str] = PROTOCOLS,
    n_splits: int = 5,
    seed: int = 0,
) -> list[Score]:
    """Run every pipeline under every protocol.

    Raises:
        ValueError: If a protocol is unknown.
    """
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"unknown protocols {unknown}; supported: {list(PROTOCOLS)}")
    scores: list[Score] = []
    for name, factory in pipelines.items():
        for protocol in protocols:
            logger.info("evaluate %s, %s", name, protocol)
            if protocol == WITHIN_SESSION:
                scores += within_session(epochs, factory, name, n_splits=n_splits, seed=seed)
            elif protocol == CROSS_SESSION:
                scores += cross_session(epochs, factory, name)
            else:
                scores += cross_subject(epochs, factory, name)
    return scores


def summarize(scores: Iterable[Score], n_classes: int) -> list[Summary]:
    """Average scores per pipeline and protocol, each subject weighing the same.

    Args:
        scores: Output of :func:`evaluate`.
        n_classes: Number of classes, for the chance level ``1 / n_classes``.
    """
    groups: dict[tuple[str, str], dict[int, list[Score]]] = defaultdict(lambda: defaultdict(list))
    for score in scores:
        groups[(score.pipeline, score.protocol)][score.subject].append(score)
    summaries: list[Summary] = []
    for (pipeline, protocol), by_subject in groups.items():
        accuracy = np.array([np.mean([s.accuracy for s in v]) for v in by_subject.values()])
        kappa = np.array([np.mean([s.kappa for s in v]) for v in by_subject.values()])
        summaries.append(
            Summary(
                pipeline=pipeline,
                protocol=protocol,
                n_subjects=len(by_subject),
                accuracy_mean=float(accuracy.mean()),
                accuracy_std=float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                kappa_mean=float(kappa.mean()),
                chance=1.0 / n_classes,
            )
        )
    order = {protocol: i for i, protocol in enumerate(PROTOCOLS)}
    return sorted(summaries, key=lambda s: (order.get(s.protocol, len(order)), s.pipeline))


def _sessions(epochs: EpochSet) -> list[str]:
    return sorted(set(epochs.session.tolist()))


def _score(
    pipeline: str,
    protocol: str,
    subject: int,
    session: str,
    n_train: int,
    y_true: npt.NDArray[np.str_],
    y_pred: npt.NDArray[Any],
) -> Score:
    return Score(
        pipeline=pipeline,
        protocol=protocol,
        subject=int(subject),
        session=str(session),
        n_train=int(n_train),
        n_test=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
    )
