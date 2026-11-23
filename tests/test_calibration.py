"""Tests of the live calibration epochs, the staged model update and the resume mode."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from openmind_core.calibration import (
    PERSONAL_MODEL,
    CalibrationRecorder,
    EpochBuffer,
    PersonalModel,
    ResumeProtocol,
)
from openmind_core.datasets import MONTAGE_A_MI
from openmind_core.io import Trial
from openmind_core.models import make_baseline
from openmind_core.models.registry import ModelCard
from openmind_core.online import Decision
from openmind_core.preprocessing import FilterConfig, epochs_at, filter_continuous
from openmind_core.profiles import ProfileStore

FS = 250.0
T0 = 1000.0
CLASSES = ("forward", "left", "right")
FILTERS = FilterConfig()
ATTENUATED = {"left": ("C4", "CP4"), "right": ("FC3", "C3", "CP3"), "forward": ("Cz",)}
N_CH = len(MONTAGE_A_MI)


def card(alignment: str | None = None) -> ModelCard:
    return ModelCard(
        name="synthetic",
        architecture="ts-lr",
        channels=MONTAGE_A_MI,
        srate=FS,
        n_times=500,
        classes=CLASSES,
        scale=(1.0,) * N_CH,
        format="sklearn",
        alignment=alignment,
        filters=asdict(FILTERS),
    )


def segment(
    label: str | None, seconds: float, rng: np.random.Generator, t0: float, swap: bool = False
) -> np.ndarray:
    """10 Hz rhythm, attenuated on the electrodes of ``label`` (none for ``None``)."""
    t = t0 + np.arange(round(seconds * FS)) / FS
    amplitude = np.full(N_CH, 20.0)
    if label is not None:
        shown = {"left": "right", "right": "forward", "forward": "left"}[label] if swap else label
        for channel in ATTENUATED[shown]:
            amplitude[MONTAGE_A_MI.index(channel)] = 3.0
    phase = np.linspace(0, 3, N_CH)
    rhythm = amplitude * np.sin(2 * np.pi * 10 * t[:, None] + phase)
    return np.asarray(rhythm + rng.normal(0, 5, (len(t), N_CH)) + 40.0)


def sandbox(kind: str, trial: int | None = None, label: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"v": 1, "type": kind, "source": "sandbox", "scene": "bird"}
    if trial is not None:
        payload["trial"] = trial
    if label is not None:
        payload["label"] = label
    return payload


def make_session(
    labels: list[str], block_size: int, seed: int = 0
) -> tuple[np.ndarray, list[tuple[float, dict[str, Any]]], list[float]]:
    """Signal (2 s fixation + 6 s of the class per trial) and its sandbox markers."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    markers: list[tuple[float, dict[str, Any]]] = [(T0, sandbox("event", label="session_start"))]
    cues: list[float] = []
    t = 0.0
    for n, label in enumerate(labels, start=1):
        if (n - 1) % block_size == 0:
            block = (n - 1) // block_size + 1
            markers.append((T0 + t, sandbox("event", label=f"block_{block}_start")))
        parts += [segment(None, 2.0, rng, t), segment(label, 6.0, rng, t + 2.0)]
        markers += [
            (T0 + t, sandbox("trial_start", n)),
            (T0 + t + 2, sandbox("cue", n, label)),
            (T0 + t + 8, sandbox("trial_end", n)),
        ]
        cues.append(T0 + t + 2)
        t += 8.0
        if n % block_size == 0:
            markers.append((T0 + t, sandbox("event", label=f"block_{n // block_size}_end")))
    return np.vstack(parts), markers, cues


def feed(recorder: CalibrationRecorder, data: np.ndarray, seed: int = 0, start: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pos = 0
    while pos < len(data):
        size = int(rng.integers(1, 400))
        times = T0 + (start + pos + np.arange(min(size, len(data) - pos))) / FS
        recorder.push(data[pos : pos + size], times)
        pos += size


# -- 1. live epochs -------------------------------------------------------------------


def test_live_epochs_equal_offline_cutting() -> None:
    labels = ["left", "right", "forward", "left"]
    data, markers, cues = make_session(labels, block_size=2)
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI[::-1], FS, keep_s=60)
    feed(recorder, data[:, ::-1])  # stream channels in reverse order
    recorder.add_markers(markers)
    recorder.finish()

    filtered = filter_continuous(data, FS, FILTERS)
    onsets = np.round((np.array(cues) - T0) * FS)
    expected, kept = epochs_at(filtered, onsets, FS)
    assert kept.all()
    assert recorder.labels() == labels
    assert recorder.epochs().shape == expected.shape
    np.testing.assert_allclose(recorder.epochs(), expected, atol=1e-9)
    assert recorder.counts_per_class() == {"left": 2, "right": 1, "forward": 1}
    assert [t.block for t in recorder.trials] == [1, 1, 2, 2]
    assert recorder.block_ends == 2
    assert recorder.missed == 0


def test_buffer_epoch_at_matches_epochs_at() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 10, (3000, 3))
    buffer = EpochBuffer(("a", "b", "c"), FS, FILTERS, keep_s=60)
    buffer.push(data, T0 + np.arange(3000) / FS)
    epoch = buffer.epoch_at(T0 + 4.0)
    expected, _ = epochs_at(filter_continuous(data, FS, FILTERS), [4.0 * FS], FS)
    assert epoch is not None
    np.testing.assert_allclose(epoch, expected[0])
    assert buffer.epoch_at(T0 + 100.0) is None
    assert buffer.status(T0 + 100.0) == "pending"
    with pytest.raises(ValueError, match="expected shape"):
        buffer.push(np.zeros((5, 2)))


def test_interrupted_dropped_and_missed_counted() -> None:
    data, markers, _ = make_session(["left", "right", "left"], block_size=3)
    # Trial 2 is interrupted: its trial_end is removed, trial 3 starts after it.
    markers = [m for m in markers if not (m[1]["type"] == "trial_end" and m[1]["trial"] == 2)]
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI, FS, keep_s=12)
    feed(recorder, data)  # only the last 12 s remain: trial 1's epoch is gone
    recorder.add_markers(markers)
    recorder.finish()
    assert recorder.interrupted == 1
    assert recorder.labels() == ["left"]  # trial 3 (the last one)
    assert [t.trial for t in recorder.trials] == [3]
    assert recorder.missed == 1  # trial 1


def test_markers_before_data_wait_for_the_samples() -> None:
    data, markers, _ = make_session(["left", "right"], block_size=2)
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI, FS, keep_s=60)
    recorder.add_markers(markers)  # nothing received yet
    assert recorder.trials == []
    feed(recorder, data)
    assert recorder.labels() == ["left", "right"]
    assert recorder.missed == 0


def test_recorder_requires_the_card_channels() -> None:
    with pytest.raises(ValueError, match="missing from the stream"):
        CalibrationRecorder(card(), MONTAGE_A_MI[:-1], FS)


# -- 2. staged update -------------------------------------------------------------------


def poor_generic() -> object:
    """Generic model trained on another 'population': classes are mapped to other patterns."""
    rng = np.random.default_rng(5)
    X, y = [], []
    for label in CLASSES:
        for _ in range(30):
            sig = filter_continuous(segment(label, 4.0, rng, 0.0, swap=True), FS, FILTERS)
            X.append(sig[250:750].T)
            y.append(label)
    return make_baseline("ts-lr").fit(np.array(X), np.array(y))


def accuracy(model: Any, seed: int) -> float:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in CLASSES:
        for _ in range(20):
            sig = filter_continuous(segment(label, 4.0, rng, 0.0), FS, FILTERS)
            X.append(sig[250:750].T)
            y.append(label)
    proba = model.predict_proba(np.array(X))
    return float(np.mean(np.array(CLASSES)[proba.argmax(1)] == np.array(y)))


def test_personal_model_blending_and_assistance() -> None:
    generic = poor_generic()
    labels = ["forward", "left", "right"] * 20
    data, markers, _ = make_session(labels, block_size=15, seed=3)
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI, FS, keep_s=300)
    personal = PersonalModel(generic, card(), recorder, min_per_class=5, target_per_class=20)
    base = accuracy(personal, seed=9)
    assert base < 0.5  # the generic model is wrong for this user
    assert personal.assistance == 1.0 and not personal.fitted

    seen: list[tuple[int, float, float]] = []
    trial_markers = [m for m in markers if m[1]["type"] in ("trial_start", "cue", "trial_end")]
    ends = [m for m in markers if str(m[1].get("label", "")).endswith("_end")]
    for block in range(4):
        lo, hi = block * 15, (block + 1) * 15
        feed(recorder, data[lo * 2000 : hi * 2000], start=lo * 2000)
        recorder.add_markers(trial_markers[lo * 3 : hi * 3])
        # no decision at all: every trial is scored as a miss/rest
        for trial in recorder.collector.trials[lo:hi]:
            personal.trial_done(trial, [])
        recorder.add_markers([e for e in ends if e[1]["label"] == f"block_{block + 1}_end"])
        assert recorder.block_ends == block + 1
        score = personal.end_block()
        assert score is not None and score.n_trials == 15
        seen.append((len(recorder.trials), personal.weight, personal.assistance))

    # 5 per class after the first block (w = 0.25), 10, 15, 20 per class (w = 1).
    assert [round(w, 2) for _, w, _ in seen] == [0.25, 0.5, 0.75, 1.0]
    assert [round(a, 2) for _, _, a in seen] == [0.75, 0.5, 0.25, 0.0]
    assert personal.updates == 4
    final = accuracy(personal, seed=9)
    assert final > 0.9 > base
    assert len(personal.tracker.blocks) == 4


def test_personal_model_waits_for_min_per_class_and_update_every() -> None:
    labels = ["left", "right"] * 3 + ["forward"] * 6 + ["left", "right"] * 3
    data, markers, _ = make_session(labels, block_size=100, seed=4)
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI, FS, keep_s=300)
    strict = PersonalModel(poor_generic(), card(), recorder, min_per_class=6)
    personal = PersonalModel(poor_generic(), card(), recorder, min_per_class=3, update_every=4)
    per_trial = [m for m in markers if m[1]["type"] in ("trial_start", "cue", "trial_end")]
    updates, weights = [], []
    for n in range(len(labels)):  # trial by trial, as in a live session
        feed(recorder, data[n * 2000 : (n + 1) * 2000], start=n * 2000)
        recorder.add_markers(per_trial[3 * n : 3 * n + 3])
        personal.trial_done(recorder.collector.trials[n], [])
        updates.append(personal.updates)
        weights.append(personal.weight)
        assert strict.update() == (n == len(labels) - 1)  # 6 of every class only at the end
    assert updates[:11] == [0] * 11  # 4th and 8th trials: no forward epoch yet
    assert updates[11] == 1 and weights[11] == pytest.approx(0.15)  # 3 of each class
    assert strict.fitted and strict.weight == pytest.approx(0.3)  # 6 of 20


def test_alignment_needs_a_reference() -> None:
    labels = ["forward", "left", "right"] * 6
    data, markers, _ = make_session(labels, block_size=18, seed=6)
    recorder = CalibrationRecorder(card("euclid"), MONTAGE_A_MI, FS, keep_s=300)
    feed(recorder, data)
    recorder.add_markers(markers)
    personal = PersonalModel(poor_generic(), card("euclid"), recorder)
    assert personal.update() is False
    personal.set_reference(np.eye(N_CH) * 100.0)
    assert personal.update() is True and personal.weight == pytest.approx(0.3)


def test_save_to_profile_store(tmp_path: Path) -> None:
    labels = ["forward", "left", "right"] * 7
    data, markers, _ = make_session(labels, block_size=21, seed=2)
    recorder = CalibrationRecorder(card(), MONTAGE_A_MI, FS, keep_s=300)
    feed(recorder, data)
    recorder.add_markers(markers)
    personal = PersonalModel(poor_generic(), card(), recorder)
    for trial in recorder.collector.trials:
        personal.trial_done(trial, [Decision("rest", 0.5, trial.cue_time + 1.0)])
    personal.end_block()
    store = ProfileStore(tmp_path)
    store.create("P01")
    personal.save(store, "P01")
    saved = joblib.load(store.weights_dir("P01") / PERSONAL_MODEL)
    assert saved["classes"] == list(CLASSES) and saved["weight"] == personal.weight
    assert saved["epochs"] == {"forward": 7, "left": 7, "right": 7}
    (record,) = store.scores("P01")
    assert record["assistance"] == pytest.approx(1 - 7 / 20)
    assert record["blocks"] == 1


# -- 3. resume mode -----------------------------------------------------------------------


def rest_signal(seconds: float, mixing: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(0, 10, (round(seconds * FS), N_CH)) @ mixing.T + 30.0)


def resume_markers(labels: list[str], rest_s: float) -> list[tuple[float, dict[str, Any]]]:
    markers = [
        (T0, sandbox("event", label="session_start")),
        (T0 + rest_s, sandbox("event", label="block_1_start")),
    ]
    t = rest_s + 1
    for n, label in enumerate(labels, start=1):
        markers += [
            (T0 + t, sandbox("trial_start", n)),
            (T0 + t + 2, sandbox("cue", n, label)),
            (T0 + t + 8, sandbox("trial_end", n)),
        ]
        t += 9
    return markers


def make_resume(**kwargs: Any) -> ResumeProtocol:
    return ResumeProtocol(
        MONTAGE_A_MI, FS, ("left", "right"), alignment="euclid", filters=FILTERS, **kwargs
    )


def test_resume_reference_and_control_cues() -> None:
    rng = np.random.default_rng(1)
    mixing = np.eye(N_CH) + 0.4 * rng.normal(size=(N_CH, N_CH))
    protocol = make_resume(rest_s=45.0, control_cues=5)
    data = rest_signal(60.0, mixing, seed=2)
    pos = 0
    while pos < len(data):  # blocks of 100 samples, markers arrive before the data
        if pos == 0:
            protocol.add_markers(resume_markers(["left", "right", "left", "right", "left"], 50.0))
        protocol.push(
            data[pos : pos + 100], T0 + (pos + np.arange(len(data[pos : pos + 100]))) / FS
        )
        pos += 100
    assert protocol.phase == "control"
    reference = protocol.reference
    assert reference is not None
    # Covariance of the band-passed rest signal, estimated on the same stretch.
    filtered = filter_continuous(data[: round(50 * FS)], FS, FILTERS)[round(FS) :]
    truth = np.cov(filtered.T)
    relative = np.linalg.norm(reference - truth) / np.linalg.norm(truth)
    assert relative < 0.15
    # Same shape as the mixing's covariance (up to the band-pass attenuation).
    expected = mixing @ mixing.T
    cosine = np.sum(reference * expected) / np.linalg.norm(reference) / np.linalg.norm(expected)
    assert cosine > 0.95

    # The control cues: 4 right out of 5 decisions (the last left cue is wrong).
    cues = [
        (T0 + 51 + 9 * n + 2, label) for n, label in enumerate(["left", "right"] * 2 + ["left"])
    ]
    decisions = [
        Decision("left" if label == "left" else "right", 0.9, t + 1.0) for t, label in cues[:-1]
    ] + [Decision("right", 0.9, cues[-1][0] + 1.0)]
    protocol.add_decisions(decisions, until=T0 + 200)
    assert protocol.result is not None
    result = protocol.result
    assert result.passed and result.accuracy == pytest.approx((2 / 3 + 1) / 2)
    assert protocol.phase == "done"
    np.testing.assert_allclose(result.reference, reference)
    assert result.rest_windows > 40


def test_resume_fails_with_wrong_decisions_and_short_rest() -> None:
    mixing = np.eye(N_CH)
    protocol = make_resume(rest_s=45.0, control_cues=2)
    protocol.push(rest_signal(60.0, mixing, 3), T0 + np.arange(round(60 * FS)) / FS)
    protocol.add_markers(resume_markers(["left", "right"], 50.0))
    assert protocol.result is None  # decisions not received yet
    protocol.add_decisions([Decision("right", 0.9, T0 + 51 + 2 + 1.0)])  # trial 1 wants left
    assert protocol.result is None
    protocol.add_decisions([Decision("left", 0.9, T0 + 60 + 2 + 1.0)], until=T0 + 70)  # trial 2
    result = protocol.result
    assert result is not None and not result.passed and result.accuracy == 0.0

    short = make_resume(rest_s=45.0)
    short.push(rest_signal(31.0, mixing, 4), T0 + np.arange(round(31 * FS)) / FS)
    with pytest.raises(ValueError, match="clean rest"):
        short.add_markers(resume_markers(["left"], 30.0))
    with pytest.raises(ValueError, match="rest_s"):
        make_resume(rest_s=2.0)


def test_trial_dataclass_used_by_tracker() -> None:
    # Guards the contract between the markers and the tracker used above.
    assert Trial(1, "left", 2.0, 0.0, 8.0).label == "left"
