"""Tests for the classical baselines and the evaluation protocols."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from helpers import CLASSES, synthetic_epochs
from openmind_core.evaluation import (
    CROSS_SESSION,
    CROSS_SUBJECT,
    PROTOCOLS,
    WITHIN_SESSION,
    cross_session,
    cross_subject,
    drop_small_subjects,
    evaluate,
    summarize,
    within_session,
)
from openmind_core.models import BASELINES, make_baseline


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_learn_separable_classes(name: str) -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=40)
    even = np.arange(len(epochs)) % 2 == 0
    train, test = epochs.select(even), epochs.select(~even)
    model = make_baseline(name).fit(train.X, train.y)
    assert set(model.predict(test.X)) <= set(CLASSES)
    assert (model.predict(test.X) == test.y).mean() > 0.9


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_are_at_chance_on_noise(name: str) -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), separation=0.0, per_class=30)
    scores = within_session(epochs, lambda: make_baseline(name), name)
    assert scores[0].accuracy < 0.6


def test_baselines_reject_flat_input() -> None:
    with pytest.raises(ValueError, match="n_epochs, n_channels, n_times"):
        make_baseline("mdm").fit(np.zeros((10, 7)), np.array(["left"] * 5 + ["right"] * 5))


def test_unknown_baseline() -> None:
    with pytest.raises(KeyError, match="available"):
        make_baseline("svm")


def test_make_baseline_returns_fresh_models() -> None:
    assert make_baseline("ts-lr") is not make_baseline("ts-lr")


def test_within_session_scores_each_session() -> None:
    scores = within_session(synthetic_epochs(), lambda: make_baseline("mdm"), "mdm", n_splits=4)
    assert [(s.subject, s.session) for s in scores] == [
        (1, "0train"),
        (1, "1test"),
        (2, "0train"),
        (2, "1test"),
    ]
    for score in scores:
        assert score.protocol == WITHIN_SESSION
        assert score.n_test == 60
        assert score.n_train == 45  # 3/4 of the session
        assert score.accuracy > 0.9
        assert -1 <= score.kappa <= 1


def test_within_session_is_reproducible() -> None:
    epochs = synthetic_epochs(separation=0.5)
    first = within_session(epochs, lambda: make_baseline("ts-lr"), seed=3)
    second = within_session(epochs, lambda: make_baseline("ts-lr"), seed=3)
    assert first == second


def test_within_session_skips_tiny_sessions(caplog: pytest.LogCaptureFixture) -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=3)
    with caplog.at_level(logging.WARNING):
        assert within_session(epochs, lambda: make_baseline("mdm"), n_splits=5) == []
    assert "too few epochs" in caplog.text


def test_cross_session_trains_on_the_other_session() -> None:
    scores = cross_session(synthetic_epochs(), lambda: make_baseline("mdm"), "mdm")
    assert [(s.subject, s.session) for s in scores] == [
        (1, "0train"),
        (1, "1test"),
        (2, "0train"),
        (2, "1test"),
    ]
    assert all(s.protocol == CROSS_SESSION and s.n_train == 60 and s.n_test == 60 for s in scores)


def test_cross_session_skips_single_session_subjects() -> None:
    epochs = synthetic_epochs(sessions=("0",))
    assert cross_session(epochs, lambda: make_baseline("mdm")) == []


def test_cross_subject_leaves_one_subject_out() -> None:
    epochs = synthetic_epochs(subjects=(1, 2, 3), sessions=("0",))
    scores = cross_subject(epochs, lambda: make_baseline("ts-lr"), "ts-lr")
    assert [s.subject for s in scores] == [1, 2, 3]
    assert all(s.protocol == CROSS_SUBJECT and s.session == "all" for s in scores)
    assert all(s.n_train == 120 and s.n_test == 60 for s in scores)


def test_cross_subject_needs_two_subjects() -> None:
    epochs = synthetic_epochs(subjects=(1,))
    assert cross_subject(epochs, lambda: make_baseline("mdm")) == []


def test_evaluate_runs_every_pipeline_and_protocol() -> None:
    pipelines = {name: (lambda name=name: make_baseline(name)) for name in BASELINES}
    scores = evaluate(synthetic_epochs(), pipelines, n_splits=4)
    assert {(s.pipeline, s.protocol) for s in scores} == {
        (name, protocol) for name in BASELINES for protocol in PROTOCOLS
    }


def test_evaluate_rejects_unknown_protocol() -> None:
    with pytest.raises(ValueError, match="unknown protocols"):
        evaluate(synthetic_epochs(), {"mdm": lambda: make_baseline("mdm")}, ["random-split"])


def test_summarize_weighs_subjects_equally() -> None:
    scores = evaluate(
        synthetic_epochs(subjects=(1, 2, 3)), {"mdm": lambda: make_baseline("mdm")}, n_splits=4
    )
    summaries = summarize(scores, n_classes=3)
    assert [s.protocol for s in summaries] == list(PROTOCOLS)
    for summary in summaries:
        assert summary.n_subjects == 3
        assert summary.chance == pytest.approx(1 / 3)
        own = [s for s in scores if s.protocol == summary.protocol]
        per_subject = [np.mean([s.accuracy for s in own if s.subject == k]) for k in (1, 2, 3)]
        assert summary.accuracy_mean == pytest.approx(np.mean(per_subject))
        assert summary.accuracy_std == pytest.approx(np.std(per_subject, ddof=1))


def test_drop_small_subjects() -> None:
    epochs = synthetic_epochs(subjects=(1, 2, 3), sessions=("0",), per_class=12)
    # Subject 2 keeps only 4 "left" epochs; subject 3 loses "forward" entirely.
    keep = ~(
        ((epochs.subject == 2) & (epochs.y == "left") & (np.arange(len(epochs)) % 12 >= 4))
        | ((epochs.subject == 3) & (epochs.y == "forward"))
    )
    remaining, dropped = drop_small_subjects(epochs.select(keep), min_per_class=10)
    assert dropped == [2, 3]
    assert remaining.subjects == (1,)
    assert len(remaining) == 36


def test_scores_report_recall_per_class() -> None:
    epochs = synthetic_epochs(subjects=(1, 2), sessions=("0",))
    scores = cross_subject(epochs, lambda: make_baseline("ts-lr"), "ts-lr")
    for score in scores:
        assert set(score.recall) == set(CLASSES)
        assert all(0 <= value <= 1 for value in score.recall.values())
        # Balanced accuracy is the mean recall over classes.
        assert np.mean(list(score.recall.values())) == pytest.approx(score.balanced_accuracy)
    (summary,) = summarize(scores, n_classes=3)
    for label in CLASSES:
        assert summary.recall_mean[label] == pytest.approx(
            np.mean([s.recall[label] for s in scores])
        )


@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
def test_summarize_ignores_undefined_kappa() -> None:
    epochs = synthetic_epochs(subjects=(1, 2, 3), sessions=("0",))
    lone = (epochs.subject != 3) | (epochs.y == "left")  # subject 3: a single class
    scores = cross_subject(epochs.select(lone), lambda: make_baseline("ts-lr"), "ts-lr")
    assert np.isnan(scores[2].kappa)
    (summary,) = summarize(scores, n_classes=3)
    assert np.isfinite(summary.kappa_mean)


@pytest.mark.slow
def test_baselines_beat_chance_on_bnci2014_001() -> None:
    pytest.importorskip("moabb")
    from openmind_core.datasets import load_epochs

    epochs = load_epochs("bnci2014_001", subjects=[1], classes=("left", "right"))
    scores = within_session(epochs, lambda: make_baseline("ts-lr"))
    assert np.mean([s.accuracy for s in scores]) > 0.7
