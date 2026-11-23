"""Tests of the progress score."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from openmind_core.io.markers import Trial
from openmind_core.online.evidence import Decision
from openmind_core.profiles import ProfileStore
from openmind_core.scoring import (
    ProgressTracker,
    ScoreConfig,
    expected_calibration_error,
    score_trial,
)
from openmind_protocol import Score, decode, encode

CLASSES = ("left", "right", "rest")


def _trial(n: int, label: str) -> Trial:
    start = n * 10.0
    return Trial(n, label, cue_time=start + 2.0, start=start, end=start + 8.0)


def _decisions(trial: Trial, command: str, confidence: float = 0.9) -> list[Decision]:
    """Rest before the cue, then ``command`` for the whole trial."""
    return [
        Decision("rest", 0.5, trial.start),
        Decision(command, confidence, trial.cue_time + 0.2),  # type: ignore[arg-type]
        Decision("rest", 0.5, trial.end - 0.5),
    ]


def _tracker(**kw: object) -> ProgressTracker:
    return ProgressTracker(CLASSES, ScoreConfig(block_size=6, **kw))  # type: ignore[arg-type]


def _feed(tracker: ProgressTracker, first: int, predict: object) -> None:
    for i in range(first, first + 6):
        trial = _trial(i, CLASSES[i % 3])
        tracker.add_trial(trial, _decisions(trial, predict(trial)))  # type: ignore[operator]


def test_perfect_decisions_ready_after_two_blocks() -> None:
    tracker = _tracker()
    _feed(tracker, 0, lambda t: t.label)
    assert tracker.stable_blocks == 1 and not tracker.ready
    _feed(tracker, 6, lambda t: t.label)
    assert tracker.ready and tracker.stable_blocks == 2
    last = tracker.blocks[-1]
    assert last.overall == 1.0 and last.false_start_rate == 0.0
    assert all(v == 1.0 for v in last.per_command.values())


def test_random_decisions_not_ready() -> None:
    rng = random.Random(1)
    tracker = _tracker()
    for first in (0, 6, 12, 18):
        _feed(tracker, first, lambda _t: rng.choice(("left", "right", "forward")))
    assert not tracker.ready


def test_failed_block_resets_stable_blocks() -> None:
    tracker = _tracker()
    _feed(tracker, 0, lambda t: t.label)
    _feed(tracker, 6, lambda _t: "forward")
    assert tracker.stable_blocks == 0
    _feed(tracker, 12, lambda t: t.label)
    assert tracker.stable_blocks == 1 and not tracker.ready


def test_false_start_on_rest_cue_and_before_cue() -> None:
    rest = _trial(0, "rest")
    outcome = score_trial(rest, _decisions(rest, "left"))
    assert not outcome.correct and outcome.false_starts == 1 and outcome.rest_periods == 2
    clean = score_trial(rest, _decisions(rest, "rest"))
    assert clean.correct and clean.false_starts == 0

    left = _trial(1, "left")
    early = [Decision("right", 0.9, left.start + 0.2), Decision("left", 0.9, left.cue_time + 0.2)]
    outcome = score_trial(left, early)
    assert outcome.correct and outcome.false_starts == 1 and outcome.rest_periods == 1
    brief = [Decision("right", 0.9, left.start + 0.2), Decision("rest", 0.9, left.start + 0.4)]
    assert score_trial(left, brief).false_starts == 0


def test_majority_is_time_weighted() -> None:
    trial = _trial(0, "left")
    t = trial.cue_time
    decisions = [
        Decision("right", 0.9, t + 0.5),
        Decision("left", 0.6, t + 1.0),  # held 3 s of the window
    ]
    outcome = score_trial(trial, decisions)
    assert outcome.correct and outcome.confidence == pytest.approx(0.6)
    assert not score_trial(_trial(0, "right"), [Decision("rest", 1.0, 0.0)]).correct


def test_ece_known_values() -> None:
    assert expected_calibration_error([1.0] * 4, [True] * 4) == pytest.approx(0.0)
    calibrated = [0.8] * 10
    assert expected_calibration_error(calibrated, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    assert expected_calibration_error([0.95] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        expected_calibration_error([], [])


def test_message_validates_and_round_trips() -> None:
    tracker = _tracker()
    empty = tracker.to_message("P01")
    assert not empty.ready and empty.per_command == {}
    _feed(tracker, 0, lambda t: t.label)
    _feed(tracker, 6, lambda t: t.label)
    message = tracker.to_message("P01", ts=1.0)
    decoded = decode(encode(message), Score)
    assert decoded.ready and decoded.stable_blocks == 2
    assert set(decoded.per_command) == set(CLASSES)


def test_record_round_trip_in_profile_store(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profiles")
    store.create("P01")
    tracker = _tracker()
    _feed(tracker, 0, lambda t: t.label)
    store.append_score("P01", tracker.to_record("P01", ts=5.0))
    (record,) = store.scores("P01")
    assert record["ready"] is False and record["blocks"] == 1 and record["ts"] == 5.0
    assert record["per_command"]["left"] == 1.0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ScoreConfig(window_start=4.0, window_end=1.0)
    with pytest.raises(ValueError):
        ProgressTracker(())
    with pytest.raises(ValueError):
        _tracker().close_block()
