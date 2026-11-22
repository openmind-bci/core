"""Tests for Euclidean alignment and Riemannian re-centering."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import synthetic_epochs
from openmind_core.alignment import Aligner, align_epochs, covariances, reference_covariance
from openmind_core.alignment.align import mean_riemann
from openmind_core.datasets import EpochSet
from openmind_core.evaluation import cross_subject
from openmind_core.models import make_baseline

IDENTITY = np.eye(7)


def distorted(epochs: EpochSet, subject: int, seed: int = 0) -> EpochSet:
    """Apply a strong, subject-specific spatial distortion (another head, another cap)."""
    rng = np.random.default_rng(seed)
    mixing = np.diag(rng.uniform(0.3, 3.0, 7)) @ (np.eye(7) + 0.3 * rng.normal(size=(7, 7)))
    X = epochs.X.copy()
    rows = epochs.subject == subject
    X[rows] = np.einsum("cd,ndt->nct", mixing, X[rows]).astype(np.float32)
    return EpochSet(X, epochs.y, epochs.subject, epochs.session, epochs.channels, epochs.srate, "d")


def test_euclidean_alignment_centres_the_arithmetic_mean() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",))
    aligned = Aligner("euclid").fit_transform(epochs.X)
    np.testing.assert_allclose(covariances(aligned).mean(axis=0), IDENTITY, atol=1e-3)


def test_riemannian_recentering_centres_the_riemannian_mean() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",))
    aligned = Aligner("riemann").fit_transform(epochs.X)
    np.testing.assert_allclose(mean_riemann(covariances(aligned)), IDENTITY, atol=1e-3)


def test_reference_from_the_first_epochs_only() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=20)
    aligner = Aligner("euclid").fit(epochs.X[:10])
    first = covariances(aligner.transform(epochs.X[:10])).mean(axis=0)
    np.testing.assert_allclose(first, IDENTITY, atol=1e-3)
    np.testing.assert_allclose(aligner.reference_, reference_covariance(epochs.X[:10], "euclid"))


def test_align_epochs_by_session_and_by_subject() -> None:
    epochs = distorted(synthetic_epochs(subjects=(1, 2), sessions=("0", "1")), subject=2)
    by_session = align_epochs(epochs, "riemann", by="session")
    for subject in (1, 2):
        for session in ("0", "1"):
            rows = (by_session.subject == subject) & (by_session.session == session)
            np.testing.assert_allclose(
                mean_riemann(covariances(by_session.X[rows])), IDENTITY, atol=1e-3
            )
    by_subject = align_epochs(epochs, "euclid", by="subject")
    for subject in (1, 2):
        rows = by_subject.subject == subject
        np.testing.assert_allclose(
            covariances(by_subject.X[rows]).mean(axis=0), IDENTITY, atol=1e-3
        )
    assert by_session.info["alignment"] == {
        "method": "riemann",
        "by": "session",
        "reference_epochs": None,
    }
    assert by_session.X.dtype == np.float32
    assert by_session.y.tolist() == epochs.y.tolist()


def test_align_epochs_with_a_few_reference_epochs() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=20)
    aligned = align_epochs(epochs, "euclid", reference_epochs=12)
    expected = Aligner("euclid").fit(epochs.X[:12]).transform(epochs.X)
    np.testing.assert_allclose(aligned.X, expected)


@pytest.mark.parametrize("method", ["euclid", "riemann"])
def test_alignment_helps_a_model_trained_on_another_subject(method: str) -> None:
    epochs = distorted(synthetic_epochs(subjects=(1, 2), sessions=("0",), separation=1.5), 2)
    raw = cross_subject(epochs, lambda: make_baseline("mdm"), "mdm")
    aligned = cross_subject(align_epochs(epochs, method), lambda: make_baseline("mdm"), "mdm")
    before = np.mean([s.accuracy for s in raw])
    after = np.mean([s.accuracy for s in aligned])
    assert after > before + 0.1, (before, after)
    assert after > 0.8


def test_transform_accepts_a_single_window() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",))
    aligner = Aligner().fit(epochs.X)
    window = aligner.transform(epochs.X[0])
    assert window.shape == epochs.X[0].shape
    np.testing.assert_allclose(window, aligner.transform(epochs.X[:1])[0])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        Aligner("mean")
    with pytest.raises(ValueError, match="not fitted"):
        Aligner().transform(np.zeros((2, 7, 10)))
    with pytest.raises(ValueError, match="at least one epoch"):
        reference_covariance(np.zeros((0, 7, 10)))
    with pytest.raises(ValueError, match="n_epochs, n_channels, n_times"):
        covariances(np.zeros((7, 10)))
    aligner = Aligner().fit(synthetic_epochs(subjects=(1,), sessions=("0",)).X)
    with pytest.raises(ValueError, match="expected 7 channels"):
        aligner.transform(np.zeros((2, 5, 10)))
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",))
    with pytest.raises(ValueError, match="by must be"):
        align_epochs(epochs, by="day")
    with pytest.raises(ValueError, match="reference_epochs"):
        align_epochs(epochs, reference_epochs=0)


def test_from_reference_matches_a_fitted_aligner() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",))
    fitted = Aligner("riemann").fit(epochs.X)
    restored = Aligner.from_reference(fitted.reference_, "riemann")
    np.testing.assert_allclose(restored.transform(epochs.X), fitted.transform(epochs.X))
    with pytest.raises(ValueError, match="square"):
        Aligner.from_reference(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="positive definite"):
        Aligner.from_reference(-np.eye(3))
    with pytest.raises(ValueError, match="unknown method"):
        Aligner.from_reference(np.eye(3), "mean")
