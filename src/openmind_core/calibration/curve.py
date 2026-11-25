"""Calibration curve: accuracy as a function of the number of calibration trials.

Protocol (one new user, one session at a time, leave-one-subject-out):

1. **Split.** The calibration set is the first ``n`` trials *per class* of the
   session, in time order. The test set is every trial after the last
   calibration trial and after the first ``unlabeled_start`` trials of the
   session. An ``n`` that would leave fewer than ``min_test_per_class`` test
   trials in some class is skipped. The *common* test set starts after the
   largest feasible ``n``: it is the same for every ``n``, which makes the
   curves comparable (a curve measured on shrinking test sets is not).
2. **Source data.** Every trial of every other subject, all sessions, each
   session aligned on its own reference when the condition uses alignment.
3. **Conditions** (:data:`CONDITIONS`): ``generic``, ``aligned``,
   ``aligned+tuned``, ``scratch``.

For ``n = 0`` the aligned conditions have no calibration trials to build a
reference from. They use the first ``unlabeled_start`` trials of the session,
without their labels: an unsupervised, *transductive* start (the reference sees
signals that are part of the session, though never its labels). Those trials are
excluded from every test set so that all conditions are tested on the same data.

Nothing here reads the labels of the target session except to pick the
calibration trials (the first ``n`` per class) and to score predictions: the
aligners only see signals.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, cohen_kappa_score

from openmind_core.alignment import Aligner, align_epochs
from openmind_core.datasets.epochs import EpochSet
from openmind_core.models import make_model

logger = logging.getLogger(__name__)

IntArray = npt.NDArray[np.intp]
ModelFactory = Callable[[], Any]

GENERIC: Final = "generic"
ALIGNED: Final = "aligned"
TUNED: Final = "aligned+tuned"
SCRATCH: Final = "scratch"

#: Supported conditions, in reporting order.
CONDITIONS: Final[tuple[str, ...]] = (GENERIC, ALIGNED, TUNED, SCRATCH)

#: Calibration trials per class of the curve.
DEFAULT_N_PER_CLASS: Final[tuple[int, ...]] = (0, 5, 10, 20, 40)

#: Smallest number of calibration trials per class that ``scratch`` is trained on.
MIN_SCRATCH_PER_CLASS: Final = 5


@dataclass(frozen=True)
class Split:
    """Calibration and test trials of one session, as positions in that session.

    Attributes:
        calibration: ``n`` -> positions of the first ``n`` trials per class.
        reference: ``n`` -> positions the aligned conditions build their
            reference from (the calibration trials; for ``n = 0`` the first
            ``unlabeled_start`` trials of the session).
        test: ``n`` -> positions after the calibration and the unlabeled start.
        common_test: Positions after the largest feasible ``n``.
        skipped: Requested ``n`` the session cannot provide.
    """

    calibration: dict[int, IntArray]
    reference: dict[int, IntArray]
    test: dict[int, IntArray]
    common_test: IntArray
    skipped: tuple[int, ...] = ()

    @property
    def n_values(self) -> tuple[int, ...]:
        """Feasible ``n``, ascending."""
        return tuple(sorted(self.test))


@dataclass(frozen=True)
class CurveScore:
    """Result of one (model, condition, n, subject, session).

    ``accuracy``, ``kappa`` and ``n_test`` are on the common test set (the main
    result); the ``*_own`` fields are on the test set specific to ``n``.
    """

    dataset: str
    model: str
    condition: str
    n: int
    subject: int
    session: str
    n_calibration: int
    n_test: int
    accuracy: float
    kappa: float
    n_test_own: int
    accuracy_own: float
    kappa_own: float


@dataclass(frozen=True)
class CurveSummary:
    """Scores of one (model, condition, n) over subjects (sessions averaged first)."""

    model: str
    condition: str
    n: int
    n_subjects: int
    accuracy_mean: float
    accuracy_std: float
    kappa_mean: float
    accuracy_own_mean: float
    n_test_mean: float = field(default=float("nan"))


def split_calibration(
    epochs: EpochSet,
    n_per_class: Sequence[int] = DEFAULT_N_PER_CLASS,
    min_test_per_class: int = 10,
    unlabeled_start: int = 20,
) -> Split:
    """Split the trials of **one session**, in time order, for every ``n``.

    Args:
        epochs: Trials of a single session, in time order.
        n_per_class: Calibration trials per class to try.
        min_test_per_class: Smallest number of test trials of each class.
        unlabeled_start: Trials at the start of the session reserved for the
            unsupervised reference of ``n = 0``; never tested.

    Raises:
        ValueError: If an ``n`` is negative or ``unlabeled_start`` < 1.
    """
    if unlabeled_start < 1:
        raise ValueError(f"unlabeled_start must be >= 1, got {unlabeled_start}")
    if any(n < 0 for n in n_per_class):
        raise ValueError(f"n_per_class must be >= 0, got {list(n_per_class)}")
    y = epochs.y
    total = len(y)
    positions = {label: np.flatnonzero(y == label) for label in epochs.classes}
    calibration: dict[int, IntArray] = {}
    reference: dict[int, IntArray] = {}
    starts: dict[int, int] = {}
    skipped: list[int] = []
    for n in sorted(set(n_per_class)):
        if any(len(p) < n for p in positions.values()):
            skipped.append(n)
            continue
        cal = (
            np.sort(np.concatenate([p[:n] for p in positions.values()]))
            if n > 0
            else np.empty(0, dtype=np.intp)
        )
        start = max(int(cal[-1]) + 1 if n > 0 else 0, unlabeled_start)
        rest = y[start:]
        if any(int((rest == label).sum()) < min_test_per_class for label in positions):
            skipped.append(n)
            continue
        calibration[n] = cal.astype(np.intp)
        reference[n] = cal.astype(np.intp) if n > 0 else np.arange(unlabeled_start, dtype=np.intp)
        starts[n] = start
    common = max(starts.values()) if starts else total
    return Split(
        calibration=calibration,
        reference=reference,
        test={n: np.arange(start, total, dtype=np.intp) for n, start in starts.items()},
        common_test=np.arange(common, total, dtype=np.intp),
        skipped=tuple(skipped),
    )


def align_with_reference(
    reference: npt.ArrayLike, X: npt.ArrayLike, method: str = "riemann"
) -> npt.NDArray[np.float32]:
    """Whiten ``X`` by the reference covariance of the ``reference`` epochs.

    Takes signals only: the labels of the reference epochs cannot influence the
    result.
    """
    return Aligner(method).fit(reference).transform(X)


def fit_weighted(
    model: Any, X: npt.NDArray[Any], y: npt.NDArray[Any], weights: npt.ArrayLike
) -> Any:
    """Fit ``model`` giving each epoch a weight.

    Scikit-learn and pyriemann steps that accept ``sample_weight`` receive it
    (``ts-lr``: tangent space, scaler and logistic regression; ``mdm``: weighted
    class means). When the final estimator cannot use weights (the LDA of
    ``csp-lda``, deep models) the epochs are duplicated instead, each repeated
    ``round(weight)`` times (at least once).
    """
    weight = np.asarray(weights, dtype=np.float64)
    steps = getattr(model, "steps", None)
    if steps and _accepts_weights(steps[-1][1]):
        params = {f"{name}__sample_weight": weight for name, est in steps if _accepts_weights(est)}
        return model.fit(X, y, **params)
    if not steps and _accepts_weights(model):
        return model.fit(X, y, sample_weight=weight)
    repeats = np.maximum(1, np.rint(weight)).astype(int)
    return model.fit(np.repeat(X, repeats, axis=0), np.repeat(y, repeats, axis=0))


def run_condition(
    condition: str,
    factory: ModelFactory,
    target: EpochSet,
    split: Split,
    source: EpochSet | None = None,
    source_model: Any | None = None,
    method: str = "riemann",
    target_weight: float = 10.0,
    min_scratch_per_class: int = MIN_SCRATCH_PER_CLASS,
) -> dict[int, npt.NDArray[Any]]:
    """Predict the test trials of one target session under one condition.

    Args:
        condition: One of :data:`CONDITIONS`.
        factory: Builds a fresh, unfitted model.
        target: Trials of the target session, in time order, unaligned.
        split: :func:`split_calibration` of ``target``.
        source: Training trials of the other subjects, **already aligned** for
            ``aligned`` and ``aligned+tuned`` (see :func:`calibration_curve`).
            Unused by ``scratch``.
        source_model: A model already fitted on ``source``, to avoid refitting it
            for every target session (``generic``, ``aligned``, ``aligned+tuned``
            at ``n = 0``).
        method: Alignment method of the target session.
        target_weight: Weight of the calibration trials in ``aligned+tuned``.
        min_scratch_per_class: Smallest ``n`` for ``scratch``.

    Returns:
        For every ``n`` the condition applies to, the predictions on the test
        trials of that ``n`` (``split.test[n]``).

    Raises:
        ValueError: For an unknown condition, or a missing ``source``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; available: {list(CONDITIONS)}")
    if condition != SCRATCH and source is None and source_model is None:
        raise ValueError(f"condition {condition!r} needs source data or a fitted source model")
    X, y = target.X, target.y
    fitted = source_model
    predictions: dict[int, npt.NDArray[Any]] = {}
    for n in split.n_values:
        test = split.test[n]
        if condition == SCRATCH:
            if n < min_scratch_per_class:
                continue
            cal = split.calibration[n]
            model = factory().fit(X[cal], y[cal])
            predictions[n] = model.predict(X[test])
            continue
        if condition == GENERIC:
            if fitted is None:
                assert source is not None
                fitted = factory().fit(source.X, source.y)
            predictions[n] = fitted.predict(X[test])
            continue
        aligner = Aligner(method).fit(X[split.reference[n]])
        if condition == TUNED and n > 0:
            assert source is not None
            cal = split.calibration[n]
            train_x = np.concatenate([source.X, aligner.transform(X[cal])])
            train_y = np.concatenate([source.y, y[cal]])
            weights = np.concatenate([np.ones(len(source)), np.full(len(cal), target_weight)])
            model = fit_weighted(factory(), train_x, train_y, weights)
        else:
            if fitted is None:
                assert source is not None
                fitted = factory().fit(source.X, source.y)
            model = fitted
        predictions[n] = model.predict(aligner.transform(X[test]))
    return predictions


def calibration_curve(
    epochs: EpochSet,
    models: Sequence[str] | Mapping[str, ModelFactory],
    conditions: Sequence[str] = CONDITIONS,
    n_values: Sequence[int] = DEFAULT_N_PER_CLASS,
    alignment: Mapping[str, Any] | None = None,
    target_weight: float = 10.0,
    seed: int = 0,
    subjects: Sequence[int] | None = None,
    aligned_epochs: EpochSet | None = None,
    min_test_per_class: int = 10,
    unlabeled_start: int = 20,
) -> list[CurveScore]:
    """Run the calibration experiment: every target subject, session, model, condition, n.

    Args:
        epochs: Unaligned epochs of every subject and session, in time order
            within each session.
        models: Model names (built with :func:`make_model`, seeded) or factories
            by name.
        conditions: Conditions to run.
        n_values: Calibration trials per class.
        alignment: ``{"method": "riemann" | "euclid"}``; the source sessions are
            each aligned on all their own epochs.
        target_weight: Weight of the calibration trials in ``aligned+tuned``.
        seed: Seed of the models built from names.
        subjects: Target subjects (default: all). The source is always every
            other subject of ``epochs``.
        aligned_epochs: ``epochs`` already aligned per session, to avoid
            recomputing it when this function is called subject by subject.
        min_test_per_class, unlabeled_start: See :func:`split_calibration`.

    Raises:
        ValueError: For an unknown condition.
    """
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; available: {list(CONDITIONS)}")
    method = str((alignment or {}).get("method", "riemann"))
    factories: dict[str, ModelFactory] = (
        dict(models)
        if isinstance(models, Mapping)
        else {
            name: functools.partial(make_model, name, srate=epochs.srate, seed=seed)
            for name in models
        }
    )
    needs_aligned = bool({ALIGNED, TUNED} & set(conditions))
    if needs_aligned and aligned_epochs is None:
        aligned_epochs = align_epochs(epochs, method=method, by="session")
    scores: list[CurveScore] = []
    for subject in subjects if subjects is not None else epochs.subjects:
        is_target = epochs.subject == subject
        source = epochs.select(~is_target)
        source_aligned = aligned_epochs.select(~is_target) if aligned_epochs is not None else None
        if len(source) == 0:
            raise ValueError("need at least two subjects: the source is the other subjects")
        fitted: dict[tuple[str, bool], Any] = {}
        for session in sorted(set(epochs.session[is_target].tolist())):
            target = epochs.select(is_target & (epochs.session == session))
            split = split_calibration(target, n_values, min_test_per_class, unlabeled_start)
            if not split.n_values:
                logger.warning("skip subject %s session %s: too few trials", subject, session)
                continue
            if split.skipped:
                logger.info(
                    "subject %s session %s: n = %s not feasible", subject, session, split.skipped
                )
            for name, factory in factories.items():
                for condition in conditions:
                    uses_alignment = condition in (ALIGNED, TUNED)
                    data = source_aligned if uses_alignment else source
                    key = (name, uses_alignment)
                    if condition in (GENERIC, ALIGNED, TUNED) and key not in fitted:
                        assert data is not None
                        fitted[key] = factory().fit(data.X, data.y)
                    predictions = run_condition(
                        condition,
                        factory,
                        target,
                        split,
                        source=data,
                        source_model=fitted.get(key),
                        method=method,
                        target_weight=target_weight,
                    )
                    scores += _scores(
                        epochs.dataset,
                        name,
                        condition,
                        subject,
                        session,
                        target,
                        split,
                        predictions,
                    )
    return scores


def summarize_curve(scores: Sequence[CurveScore]) -> list[CurveSummary]:
    """Mean and standard deviation over subjects per (model, condition, n).

    Sessions are averaged within each subject first, so every subject weighs the
    same. Accuracy is on the common test set; ``accuracy_own_mean`` on the test
    set of each ``n``.
    """
    groups: dict[tuple[str, str, int], dict[int, list[CurveScore]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in scores:
        groups[(score.model, score.condition, score.n)][score.subject].append(score)
    summaries = []
    for (model, condition, n), by_subject in groups.items():
        accuracy = np.array([np.mean([s.accuracy for s in v]) for v in by_subject.values()])
        summaries.append(
            CurveSummary(
                model=model,
                condition=condition,
                n=n,
                n_subjects=len(by_subject),
                accuracy_mean=float(accuracy.mean()),
                accuracy_std=float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                kappa_mean=_nanmean([_nanmean([s.kappa for s in v]) for v in by_subject.values()]),
                accuracy_own_mean=float(
                    np.mean([np.mean([s.accuracy_own for s in v]) for v in by_subject.values()])
                ),
                n_test_mean=float(
                    np.mean([np.mean([s.n_test for s in v]) for v in by_subject.values()])
                ),
            )
        )
    order = {name: i for i, name in enumerate(CONDITIONS)}
    return sorted(summaries, key=lambda s: (s.model, order.get(s.condition, len(order)), s.n))


def _scores(
    dataset: str,
    model: str,
    condition: str,
    subject: int,
    session: str,
    target: EpochSet,
    split: Split,
    predictions: Mapping[int, npt.NDArray[Any]],
) -> list[CurveScore]:
    scores = []
    common_start = int(split.common_test[0]) if len(split.common_test) else len(target)
    for n, predicted in predictions.items():
        test = split.test[n]
        truth = target.y[test]
        in_common = test >= common_start
        scores.append(
            CurveScore(
                dataset=dataset,
                model=model,
                condition=condition,
                n=n,
                subject=int(subject),
                session=session,
                n_calibration=len(split.calibration[n]),
                n_test=int(in_common.sum()),
                accuracy=float(accuracy_score(truth[in_common], predicted[in_common])),
                kappa=float(cohen_kappa_score(truth[in_common], predicted[in_common])),
                n_test_own=len(test),
                accuracy_own=float(accuracy_score(truth, predicted)),
                kappa_own=float(cohen_kappa_score(truth, predicted)),
            )
        )
    return scores


def _accepts_weights(estimator: Any) -> bool:
    fit = getattr(estimator, "fit", None)
    return fit is not None and "sample_weight" in inspect.signature(fit).parameters


def _nanmean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")
