"""Tests for the calibration curve protocol (synthetic epochs, no MOABB)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import synthetic_epochs
from openmind_core.calibration import (
    ALIGNED,
    CONDITIONS,
    GENERIC,
    SCRATCH,
    TUNED,
    calibration_curve,
    fit_weighted,
    run_condition,
    split_calibration,
    summarize_curve,
)
from openmind_core.datasets import EpochSet
from openmind_core.models import make_baseline

N_VALUES = (0, 5, 8)


def shuffled_session(per_class: int = 30, seed: int = 0) -> EpochSet:
    """One session whose trials come in random class order, like a real run."""
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=per_class, seed=seed)
    order = np.random.default_rng(seed).permutation(len(epochs))
    return EpochSet(
        epochs.X[order],
        epochs.y[order],
        epochs.subject[order],
        epochs.session[order],
        epochs.channels,
        epochs.srate,
        epochs.dataset,
    )


def shuffled(epochs: EpochSet, seed: int = 0) -> EpochSet:
    """Interleave the classes inside each (subject, session), keeping them in blocks."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(epochs))
    for subject in epochs.subjects:
        for session in sorted(set(epochs.session.tolist())):
            rows = np.flatnonzero((epochs.subject == subject) & (epochs.session == session))
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


def test_split_takes_the_first_trials_of_each_class_in_time_order() -> None:
    session = shuffled_session()
    split = split_calibration(session, (0, 5, 10), min_test_per_class=3, unlabeled_start=10)
    for n in (5, 10):
        cal = split.calibration[n]
        assert np.all(np.diff(cal) > 0)
        counts = {label: int((session.y[cal] == label).sum()) for label in session.classes}
        assert set(counts.values()) == {n}
        for label in session.classes:  # the first n of the class, not any n
            assert list(cal[session.y[cal] == label]) == list(
                np.flatnonzero(session.y == label)[:n]
            )
        assert split.test[n][0] > cal[-1]
    assert len(split.calibration[0]) == 0


def test_split_test_set_follows_calibration_and_unlabeled_start() -> None:
    session = shuffled_session()
    split = split_calibration(session, (0, 5), min_test_per_class=3, unlabeled_start=20)
    assert split.test[0][0] == 20
    assert np.array_equal(split.reference[0], np.arange(20))
    assert split.test[5][0] >= 20  # the unlabeled start is never tested
    assert np.array_equal(split.reference[5], split.calibration[5])
    assert all(np.array_equal(t, np.arange(t[0], len(session))) for t in split.test.values())


def test_split_common_test_set_starts_after_the_largest_n() -> None:
    session = shuffled_session()
    split = split_calibration(session, (0, 5, 10), min_test_per_class=3, unlabeled_start=5)
    last = split.calibration[10][-1]
    assert split.common_test[0] == last + 1
    assert all(set(split.common_test) <= set(test) for test in split.test.values())


def test_split_keeps_enough_test_trials_per_class_and_skips_the_rest() -> None:
    session = shuffled_session(per_class=20)
    split = split_calibration(session, (0, 5, 10, 15, 40), min_test_per_class=8, unlabeled_start=5)
    assert 40 in split.skipped
    assert 15 in split.skipped  # 15 calibration + 8 test do not fit in 20 per class
    for n in split.n_values:
        remaining = session.y[split.test[n]]
        assert all((remaining == label).sum() >= 8 for label in session.classes)
    assert len(split.common_test) > 0
    assert all((session.y[split.common_test] == c).sum() >= 8 for c in session.classes)


def test_split_validates_its_arguments() -> None:
    session = shuffled_session()
    with pytest.raises(ValueError, match="unlabeled_start"):
        split_calibration(session, (0,), unlabeled_start=0)
    with pytest.raises(ValueError, match="n_per_class"):
        split_calibration(session, (-1,))


@pytest.fixture(scope="module")
def three_subjects() -> EpochSet:
    return shuffled(synthetic_epochs(subjects=(1, 2, 3), per_class=20, separation=4.0))


def curve(epochs: EpochSet, **kwargs: object) -> list:  # type: ignore[type-arg]
    settings: dict[str, object] = {
        "models": ["mdm", "ts-lr"],
        "n_values": N_VALUES,
        "subjects": [1],
        "min_test_per_class": 3,
        "unlabeled_start": 6,
    }
    return calibration_curve(epochs, **(settings | kwargs))  # type: ignore[arg-type]


def test_every_condition_scores_every_feasible_n(three_subjects: EpochSet) -> None:
    scores = curve(three_subjects)
    by_key = {(s.model, s.condition, s.n, s.session) for s in scores}
    for model in ("mdm", "ts-lr"):
        for session in ("0train", "1test"):
            for condition in (GENERIC, ALIGNED, TUNED):
                for n in N_VALUES:
                    assert (model, condition, n, session) in by_key
            for n in (5, 8):
                assert (model, SCRATCH, n, session) in by_key
    assert all(0.0 <= s.accuracy <= 1.0 for s in scores)
    assert {s.n_test for s in scores if s.n == 0} == {s.n_test for s in scores if s.n == 8}


def test_scratch_is_skipped_below_five_trials_per_class(three_subjects: EpochSet) -> None:
    scores = curve(three_subjects, n_values=(0, 5, 3), conditions=[SCRATCH], models=["mdm"])
    assert {s.n for s in scores} == {5}


def test_generic_does_not_depend_on_n(three_subjects: EpochSet) -> None:
    scores = [s for s in curve(three_subjects, models=["mdm"]) if s.condition == GENERIC]
    by_n = {n: [s.accuracy_own for s in scores if s.n == n] for n in N_VALUES}
    assert all(len(v) == 2 for v in by_n.values())
    # one model, same trials: the common-test accuracy cannot depend on n
    common = {n: [s.accuracy for s in scores if s.n == n] for n in N_VALUES}
    assert common[0] == common[5] == common[8]


def test_aligned_condition_uses_no_labels_of_the_target(three_subjects: EpochSet) -> None:
    target = three_subjects.select(
        (three_subjects.subject == 1) & (three_subjects.session == "0train")
    )
    source = three_subjects.select(three_subjects.subject != 1)
    split = split_calibration(target, N_VALUES, min_test_per_class=4, unlabeled_start=6)
    wrong = target.y.copy()
    wrong[np.concatenate(list(split.calibration.values()))] = "left"  # destroy cal. labels
    corrupted = EpochSet(
        target.X, wrong, target.subject, target.session, target.channels, target.srate, "d"
    )
    from openmind_core.alignment import align_epochs

    aligned = align_epochs(source, "riemann", "session")
    model = make_baseline("ts-lr").fit(aligned.X, aligned.y)
    clean = run_condition(ALIGNED, lambda: make_baseline("ts-lr"), target, split, aligned, model)
    dirty = run_condition(ALIGNED, lambda: make_baseline("ts-lr"), corrupted, split, aligned, model)
    for n in clean:
        assert np.array_equal(clean[n], dirty[n])


def test_tuned_equals_aligned_without_calibration(three_subjects: EpochSet) -> None:
    scores = curve(three_subjects)
    for model in ("mdm", "ts-lr"):
        aligned = {
            s.session: s.accuracy
            for s in scores
            if (s.model, s.condition, s.n) == (model, ALIGNED, 0)
        }
        tuned = {
            s.session: s.accuracy
            for s in scores
            if (s.model, s.condition, s.n) == (model, TUNED, 0)
        }
        assert aligned == tuned


def test_tuning_helps_when_the_target_is_distorted() -> None:
    rng = np.random.default_rng(3)
    epochs = shuffled(synthetic_epochs(subjects=(1, 2, 3), sessions=("0",), per_class=30, seed=3))
    mixing = np.diag(rng.uniform(0.3, 3.0, 7)) @ (np.eye(7) + 0.5 * rng.normal(size=(7, 7)))
    X = epochs.X.copy()
    rows = epochs.subject == 1
    X[rows] = np.einsum("cd,ndt->nct", mixing, X[rows]).astype(np.float32)
    distorted = EpochSet(X, epochs.y, epochs.subject, epochs.session, epochs.channels, 125.0, "d")
    scores = curve(distorted, n_values=(0, 10), models=["ts-lr"], conditions=[GENERIC, TUNED])
    accuracy = {(s.condition, s.n): s.accuracy for s in scores}
    assert accuracy[(TUNED, 10)] > accuracy[(GENERIC, 10)]


def test_fit_weighted_uses_weights_or_duplicates() -> None:
    epochs = synthetic_epochs(subjects=(1,), sessions=("0",), per_class=8)
    weights = np.ones(len(epochs))
    weights[:3] = 4
    for name in ("ts-lr", "mdm", "csp-lda"):
        model = fit_weighted(make_baseline(name), epochs.X, epochs.y, weights)
        assert model.predict(epochs.X[:5]).shape == (5,)


def test_summary_averages_sessions_within_subject(three_subjects: EpochSet) -> None:
    scores = curve(three_subjects, models=["mdm"], subjects=[1, 2])
    summaries = summarize_curve(scores)
    assert {s.condition for s in summaries} == set(CONDITIONS)
    row = next(s for s in summaries if (s.condition, s.n) == (ALIGNED, 5))
    assert row.n_subjects == 2
    per_subject = [
        np.mean([s.accuracy for s in scores if (s.condition, s.n, s.subject) == (ALIGNED, 5, sub)])
        for sub in (1, 2)
    ]
    assert row.accuracy_mean == pytest.approx(np.mean(per_subject))
    assert row.accuracy_std == pytest.approx(np.std(per_subject, ddof=1))


def test_unknown_condition_is_rejected(three_subjects: EpochSet) -> None:
    with pytest.raises(ValueError, match="conditions"):
        curve(three_subjects, conditions=["magic"])


def test_research_script_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The script runs on a patched loader, writes its report and CSV, and resumes."""
    pytest.importorskip("yaml")
    script = (
        Path(__file__).resolve().parents[1] / "research" / "experiments" / "calibration_curve.py"
    )
    spec = importlib.util.spec_from_file_location("calibration_curve_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    epochs = shuffled(synthetic_epochs(subjects=(1, 2, 3), per_class=20, separation=4.0))
    monkeypatch.setattr(module, "cached_epochs", lambda experiment: epochs)
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path / "data"))
    output = tmp_path / "report.md"
    config = tmp_path / "config.yaml"
    config.write_text(
        "name: smoke\ntitle: Smoke\ntask: TEST\n"
        f"output: {output}\n"
        "seed: 1\nmin_epochs_per_class: 5\nmodels: [mdm]\n"
        "n_per_class: [0, 5]\nmin_test_per_class: 3\nunlabeled_start: 6\n"
        "experiments:\n  - name: synthetic\n    dataset: bnci2014_001\n"
        "    subjects: [1, 2, 3]\n    classes: [left, right, forward]\n",
        encoding="utf-8",
    )
    module.main(["--config", str(config), "--no-tracking"])
    report = output.read_text(encoding="utf-8")
    assert "### `mdm`" in report
    assert "`aligned+tuned`" in report
    csv_file = tmp_path / "data" / "results" / "smoke.csv"
    assert (
        csv_file.read_text(encoding="utf-8").splitlines()[0].startswith("experiment,dataset,model")
    )
    assert len(csv_file.read_text(encoding="utf-8").splitlines()) > 10
