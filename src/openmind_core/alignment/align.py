"""Euclidean alignment and Riemannian re-centering of EEG epochs (see the package docstring)."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

try:  # pyriemann >= 0.12
    from pyriemann.geometry.mean import mean_riemann
except ImportError:  # pragma: no cover - older pyriemann
    from pyriemann.utils.mean import mean_riemann

from openmind_core.datasets.epochs import EpochSet

FloatArray = npt.NDArray[np.float64]

#: Supported reference means.
METHODS: Final[tuple[str, ...]] = ("euclid", "riemann")

#: Relative regularization of covariances (fraction of their mean eigenvalue).
_SHRINK: Final = 1e-6


def covariances(X: npt.ArrayLike) -> FloatArray:
    """Sample covariance of each epoch ``(n_epochs, n_channels, n_times)``, regularized."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")
    centred = data - data.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centred, centred) / max(data.shape[2] - 1, 1)
    n_channels = covs.shape[1]
    scale = np.trace(covs, axis1=1, axis2=2) / n_channels
    return np.asarray(covs + _SHRINK * scale[:, None, None] * np.eye(n_channels)[None])


def reference_covariance(X: npt.ArrayLike, method: str = "riemann") -> FloatArray:
    """Mean covariance of a set of epochs, Euclidean or Riemannian.

    Raises:
        ValueError: If the method is unknown or there are no epochs.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; available: {list(METHODS)}")
    covs = covariances(X)
    if len(covs) == 0:
        raise ValueError("need at least one epoch to compute a reference")
    if method == "euclid" or len(covs) == 1:
        return np.asarray(covs.mean(axis=0))
    return np.asarray(mean_riemann(covs))


def inverse_sqrt(matrix: FloatArray) -> FloatArray:
    """Inverse square root of a symmetric positive definite matrix."""
    values, vectors = np.linalg.eigh(matrix)
    if float(np.min(values)) <= 0:
        raise ValueError("the reference covariance is not positive definite")
    return np.asarray((vectors / np.sqrt(values)) @ vectors.T)


class Aligner:
    """Align the epochs of one session with its own reference covariance.

    Args:
        method: ``"euclid"`` or ``"riemann"``.

    Attributes (after ``fit``):
        reference_: Reference covariance ``(n_channels, n_channels)``.
        whitening_: Its inverse square root, applied to the signals.
    """

    def __init__(self, method: str = "riemann") -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; available: {list(METHODS)}")
        self.method = method

    def fit(self, X: npt.ArrayLike) -> Aligner:
        """Estimate the reference from epochs of the session (no labels needed)."""
        self.reference_ = reference_covariance(X, self.method)
        self.whitening_ = inverse_sqrt(self.reference_)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Whiten epochs ``(n, n_channels, n_times)`` or one window ``(n_channels, n_times)``."""
        if not hasattr(self, "whitening_"):
            raise ValueError("the aligner is not fitted")
        data = np.asarray(X, dtype=np.float64)
        if data.shape[-2] != self.whitening_.shape[0]:
            raise ValueError(f"expected {self.whitening_.shape[0]} channels, got {data.shape[-2]}")
        return np.asarray(np.einsum("cd,...dt->...ct", self.whitening_, data), dtype=np.float32)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float32]:
        return self.fit(X).transform(X)


def align_epochs(
    epochs: EpochSet,
    method: str = "riemann",
    by: str = "session",
    reference_epochs: int | None = None,
) -> EpochSet:
    """Align every session (or subject) of an epoch set with its own reference.

    Args:
        epochs: Epochs of one or more subjects and sessions, in time order
            within each session.
        method: ``"euclid"`` or ``"riemann"``.
        by: ``"session"`` (each recording session separately, the realistic
            setting) or ``"subject"`` (all sessions of a subject together).
        reference_epochs: Number of first epochs of each group used for the
            reference (the first minutes of use), or ``None`` for all of them
            (an upper bound: the reference then sees the test epochs, without
            their labels).

    Raises:
        ValueError: If ``by`` is unknown or ``reference_epochs`` < 1.
    """
    if by not in ("session", "subject"):
        raise ValueError(f"by must be 'session' or 'subject', got {by!r}")
    if reference_epochs is not None and reference_epochs < 1:
        raise ValueError(f"reference_epochs must be >= 1, got {reference_epochs}")
    aligned = np.empty_like(epochs.X, dtype=np.float32)
    keys = (
        list(zip(epochs.subject.tolist(), epochs.session.tolist(), strict=True))
        if by == "session"
        else epochs.subject.tolist()
    )
    for key in dict.fromkeys(keys):
        rows = np.flatnonzero([k == key for k in keys])
        reference = rows if reference_epochs is None else rows[:reference_epochs]
        aligner = Aligner(method).fit(epochs.X[reference])
        aligned[rows] = aligner.transform(epochs.X[rows])
    info = dict(epochs.info) | {
        "alignment": {"method": method, "by": by, "reference_epochs": reference_epochs}
    }
    return EpochSet(
        X=aligned,
        y=epochs.y,
        subject=epochs.subject,
        session=epochs.session,
        channels=epochs.channels,
        srate=epochs.srate,
        dataset=epochs.dataset,
        info=info,
    )
