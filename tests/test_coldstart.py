"""Tests for the similarity-based cold start (synthetic subjects, no MOABB)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import synthetic_epochs
from openmind_core.alignment import align_epochs
from openmind_core.calibration import (
    ALIGNED,
    COLDSTART,
    calibration_curve,
    nearest_subjects,
    rest_signature,
    run_condition,
    split_calibration,
    subject_signatures,
)
from openmind_core.datasets import EpochSet
from openmind_core.models import make_baseline


def scaled_noise(rng: np.random.Generator, gains: np.ndarray, n: int = 12) -> np.ndarray:
    return rng.normal(size=(n, len(gains), 200)) * gains[None, :, None]


def test_signature_is_a_covariance_of_the_windows() -> None:
    rng = np.random.default_rng(0)
    sig = rest_signature(scaled_noise(rng, np.array([1.0, 2.0, 3.0, 1.0])))
    assert sig.shape == (4, 4)
    np.testing.assert_allclose(sig, sig.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(sig) > 0)
    assert np.diag(sig)[2] > np.diag(sig)[0]


def test_nearest_subjects_are_found_by_covariance() -> None:
    rng = np.random.default_rng(1)
    base = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    gains = {  # subjects 1-3 look like the target, 4-5 do not
        1: base * [1.0, 1.1, 1.0, 0.9, 1.0],
        2: base * [1.2, 1.0, 1.0, 1.0, 1.1],
        3: base * [1.0, 1.0, 0.8, 1.0, 1.0],
        4: base * [5.0, 0.2, 1.0, 4.0, 1.0],
        5: base * [0.2, 6.0, 0.3, 1.0, 5.0],
    }
    signatures = {s: rest_signature(scaled_noise(rng, g)) for s, g in gains.items()}
    target = rest_signature(scaled_noise(rng, base))
    nearest = nearest_subjects(target, signatures, k=3)
    assert {s for s, _ in nearest} == {1, 2, 3}
    distances = [d for _, d in nearest]
    assert distances == sorted(distances)
    assert nearest_subjects(target, signatures, k=1)[0][0] in {1, 2, 3}
    assert len(nearest_subjects(target, signatures, k=10)) == 5  # k larger than the pool
    assert nearest_subjects(signatures[4], signatures, k=1) == [(4, pytest.approx(0.0, abs=1e-9))]


def test_nearest_subjects_validates_its_arguments() -> None:
    sig = np.eye(3)
    with pytest.raises(ValueError, match="k must"):
        nearest_subjects(sig, {1: sig}, k=0)
    with pytest.raises(ValueError, match="source subject"):
        nearest_subjects(sig, {}, k=1)


def test_subject_signatures_use_the_start_of_each_session() -> None:
    epochs = synthetic_epochs(subjects=(1, 2), per_class=6)
    signatures = subject_signatures(epochs, n_first=4)
    assert set(signatures) == {1, 2}
    rows = np.concatenate(
        [
            np.flatnonzero((epochs.subject == 1) & (epochs.session == s))[:4]
            for s in sorted(set(epochs.session.tolist()))
        ]
    )
    np.testing.assert_allclose(signatures[1], rest_signature(epochs.X[rows]))


@pytest.fixture(scope="module")
def four_subjects() -> EpochSet:
    epochs = synthetic_epochs(subjects=(1, 2, 3, 4), sessions=("0",), per_class=20, separation=4.0)
    rng = np.random.default_rng(0)
    order = np.arange(len(epochs))
    for subject in epochs.subjects:
        rows = np.flatnonzero(epochs.subject == subject)
        order[rows] = rng.permutation(rows)
    return EpochSet(
        epochs.X[order],
        epochs.y[order],
        epochs.subject[order],
        epochs.session[order],
        epochs.channels,
        epochs.srate,
        epochs.dataset,
    )


def test_coldstart_condition_scores_n_zero_only(four_subjects: EpochSet) -> None:
    scores = calibration_curve(
        four_subjects,
        ["mdm", "ts-lr"],
        conditions=[ALIGNED, COLDSTART],
        n_values=(0, 5),
        subjects=[1],
        min_test_per_class=3,
        unlabeled_start=6,
        coldstart_k=2,
    )
    keys = {(s.model, s.condition, s.n) for s in scores}
    assert {m for m, c, _ in keys if c == COLDSTART} == {"mdm", "ts-lr"}
    assert {n for _, c, n in keys if c == COLDSTART} == {0}
    assert all(0.0 <= s.accuracy <= 1.0 for s in scores)


def test_coldstart_with_every_subject_is_the_aligned_model(four_subjects: EpochSet) -> None:
    kwargs = {
        "n_values": (0,),
        "subjects": [1],
        "min_test_per_class": 3,
        "unlabeled_start": 6,
    }
    scores = calibration_curve(
        four_subjects, ["mdm"], conditions=[ALIGNED, COLDSTART], coldstart_k=3, **kwargs
    )
    accuracy = {s.condition: s.accuracy for s in scores}
    assert accuracy[COLDSTART] == accuracy[ALIGNED]  # k = every source subject


def test_coldstart_prediction_does_not_read_the_target_labels(four_subjects: EpochSet) -> None:
    target = four_subjects.select(four_subjects.subject == 1)
    source = align_epochs(four_subjects.select(four_subjects.subject != 1))
    split = split_calibration(target, (0,), min_test_per_class=3, unlabeled_start=6)
    model = make_baseline("mdm").fit(source.X, source.y)
    honest = run_condition(
        COLDSTART, lambda: make_baseline("mdm"), target, split, source_model=model
    )
    scrambled = EpochSet(
        target.X,
        np.random.default_rng(0).permutation(target.y),
        target.subject,
        target.session,
        target.channels,
        target.srate,
        target.dataset,
    )
    other = run_condition(
        COLDSTART, lambda: make_baseline("mdm"), scrambled, split, source_model=model
    )
    np.testing.assert_array_equal(honest[0], other[0])
    assert set(honest) == {0}
