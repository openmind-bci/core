"""Similarity-based cold start: train on the source subjects closest to the new user.

A new user has no labelled trial yet. Their signals do say something about them
though: the covariance of a few unlabelled windows (the start of the session, or
a rest period) is a *signature* of the head, the headset fit and the background
rhythms. The source subjects whose signature is nearest to it, in the
affine-invariant Riemannian distance, are the best guess of "people like this
user"; a model trained on them alone is compared with the model trained on
everyone (see ``coldstart-knn`` in :mod:`openmind_core.calibration.curve`).

Nothing here reads a label.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

try:  # pyriemann >= 0.12
    from pyriemann.geometry.distance import distance_riemann
except ImportError:  # pragma: no cover - older pyriemann
    from pyriemann.utils.distance import distance_riemann

from openmind_core.alignment import reference_covariance
from openmind_core.datasets.epochs import EpochSet

FloatArray = npt.NDArray[np.float64]


def rest_signature(X: npt.ArrayLike) -> FloatArray:
    """Riemannian mean covariance of unlabelled windows ``(n, n_channels, n_times)``.

    The windows can be rest periods or the first trials of a session; their
    labels, if any, are not an input.
    """
    return reference_covariance(X, method="riemann")


def subject_signatures(epochs: EpochSet, n_first: int | None = None) -> dict[int, FloatArray]:
    """Signature of every subject of ``epochs``.

    Args:
        epochs: Unaligned epochs, in time order within each session.
        n_first: Use only the first ``n_first`` epochs of each session (the
            same recipe as the new user's unlabelled start); all of them if
            ``None``.
    """
    signatures: dict[int, FloatArray] = {}
    for subject in epochs.subjects:
        rows = []
        for session in sorted(set(epochs.session[epochs.subject == subject].tolist())):
            positions = np.flatnonzero((epochs.subject == subject) & (epochs.session == session))
            rows.append(positions if n_first is None else positions[:n_first])
        signatures[int(subject)] = rest_signature(epochs.X[np.concatenate(rows)])
    return signatures


def nearest_subjects(
    target_signature: npt.ArrayLike,
    source_signatures: Mapping[int, npt.ArrayLike],
    k: int = 3,
) -> list[tuple[int, float]]:
    """The ``k`` source subjects closest to the target, nearest first.

    Args:
        target_signature: Covariance of the new user (see :func:`rest_signature`).
        source_signatures: Covariance of each source subject.
        k: Number of subjects to keep (all of them if larger).

    Returns:
        ``(subject, distance)`` pairs, ascending distance (ties by subject).

    Raises:
        ValueError: If ``k`` < 1 or there is no source subject.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not source_signatures:
        raise ValueError("need at least one source subject")
    target = np.asarray(target_signature, dtype=np.float64)
    distances = [
        (int(subject), float(distance_riemann(target, np.asarray(cov, dtype=np.float64))))
        for subject, cov in source_signatures.items()
    ]
    return sorted(distances, key=lambda item: (item[1], item[0]))[:k]
