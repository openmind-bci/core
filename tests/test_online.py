"""Tests of the online decoder on synthetic continuous EEG."""

from __future__ import annotations

import numpy as np
import pytest

from openmind_core.alignment import align_epochs, reference_covariance
from openmind_core.datasets import MONTAGE_A_MI, EpochSet
from openmind_core.models import make_baseline
from openmind_core.models.registry import ModelCard
from openmind_core.online import OnlineConfig, OnlineDecoder
from openmind_core.preprocessing import FilterConfig, epochs_at, filter_continuous, window_ends
from openmind_core.profiles import Reference

FS = 250.0
N_TIMES = 500
CLASSES = ("forward", "left", "right")  # sorted, like the model's classes_
# Stream with extra channels, other order and other spellings of the labels.
STREAM = ("Fp1", "CZ", "C3", "FC4", "EEG C4", "CP3", "FC3.", "CP4", "O1")
FILTERS = FilterConfig()  # 8-30 Hz and a 50 Hz notch
ATTENUATED = {"left": ("C4", "CP4"), "right": ("FC3", "C3", "CP3"), "forward": ("Cz",)}


def segment(label: str, seconds: float, rng: np.random.Generator, t0: float) -> np.ndarray:
    """10 Hz rhythm on every electrode, attenuated on those of the imagined class."""
    t = t0 + np.arange(round(seconds * FS)) / FS
    amplitude = np.full(len(MONTAGE_A_MI), 20.0)
    for channel in ATTENUATED[label]:
        amplitude[MONTAGE_A_MI.index(channel)] = 3.0
    phase = np.linspace(0, 3, len(MONTAGE_A_MI))
    rhythm = amplitude * np.sin(2 * np.pi * 10 * t[:, None] + phase)
    return np.asarray(rhythm + rng.normal(0, 5, (len(t), len(MONTAGE_A_MI))) + 40.0)


def recording(
    labels: list[str], seconds: float, seed: int
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Concatenate segments; return the data (n, 7) and ``(label, start, stop)`` samples."""
    rng = np.random.default_rng(seed)
    parts, spans, start = [], [], 0
    for label in labels:
        part = segment(label, seconds, rng, start / FS)
        parts.append(part)
        spans.append((label, start, start + len(part)))
        start += len(part)
    return np.vstack(parts), spans


def as_stream(data: np.ndarray, seed: int = 0) -> np.ndarray:
    """Place the 7 montage channels at their position in ``STREAM`` (the rest is noise)."""
    out = np.random.default_rng(seed).normal(0, 5, (len(data), len(STREAM)))
    for column, label in zip(data.T, MONTAGE_A_MI, strict=True):
        key = label.casefold()
        index = next(
            i for i, s in enumerate(STREAM) if s.strip(".").removeprefix("EEG ").casefold() == key
        )
        out[:, index] = column
    return out


def train_epochs(alignment: str | None) -> EpochSet:
    data, spans = recording(list(CLASSES) * 8, 5.0, seed=1)
    filtered = filter_continuous(data, FS, FILTERS)
    X, y = [], []
    for label, start, _ in spans:
        for offset in (0.5, 1.5, 2.5):
            epochs, _ = epochs_at(filtered, [start], FS, offset_s=offset)
            X.append(epochs[0])
            y.append(label)
    n = len(X)
    epochs = EpochSet(
        X=np.array(X, dtype=np.float32),
        y=np.array(y),
        subject=np.ones(n, dtype=np.int64),
        session=np.array(["0"] * n),
        channels=MONTAGE_A_MI,
        srate=FS,
        dataset="synthetic",
    )
    return align_epochs(epochs, alignment) if alignment else epochs


def card_for(alignment: str | None) -> ModelCard:
    return ModelCard(
        name="synthetic",
        architecture="ts-lr",
        channels=MONTAGE_A_MI,
        srate=FS,
        n_times=N_TIMES,
        classes=CLASSES,
        scale=(1.0,) * len(MONTAGE_A_MI),
        format="sklearn",
        alignment=alignment,
    )


@pytest.fixture(scope="module")
def plain() -> tuple[object, ModelCard]:
    epochs = train_epochs(None)
    return make_baseline("ts-lr").fit(epochs.X, epochs.y), card_for(None)


@pytest.fixture(scope="module")
def aligned() -> tuple[object, ModelCard]:
    epochs = train_epochs("riemann")
    return make_baseline("ts-lr").fit(epochs.X, epochs.y), card_for("riemann")


def push_in_blocks(
    decoder: OnlineDecoder, stream: np.ndarray, seed: int = 0, times: np.ndarray | None = None
) -> tuple[list, list[float]]:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    predictions: list = []  # type: ignore[type-arg]
    actions: list[float] = []
    pos = 0
    while pos < len(stream):
        size = int(rng.integers(1, 400))
        step = decoder.push(
            stream[pos : pos + size], None if times is None else times[pos : pos + size]
        )
        predictions += step.predictions
        actions += step.actions
        pos += size
    return predictions, actions


def test_same_probabilities_as_offline(plain: tuple, aligned: tuple) -> None:  # type: ignore[type-arg]
    data, _ = recording(["left", "right", "forward", "left"], 6.0, seed=7)
    stream = as_stream(data)
    for model, card in (plain, aligned):
        reference = None
        if card.alignment:
            # A reference given up front: the whole path (filter, windows, alignment) is exercised.
            filtered = filter_continuous(data, FS, FILTERS)
            ends = np.arange(N_TIMES, len(data) + 1, 62)  # arbitrary, only for the reference
            epochs, _ = epochs_at(filtered, ends - N_TIMES, FS, offset_s=0.0)
            reference = Reference(reference_covariance(epochs, card.alignment), MONTAGE_A_MI)
        decoder = OnlineDecoder(model, card, STREAM, FS, reference=reference)
        times = 1000.0 + np.arange(len(data)) / FS
        predictions, _ = push_in_blocks(decoder, stream, times=times)
        assert decoder.state == "running"

        filtered = filter_continuous(data, FS, FILTERS)
        ends = window_ends(len(data), FS)
        epochs, kept = epochs_at(filtered, ends - N_TIMES, FS, offset_s=0.0)
        assert kept.all()
        assert len(predictions) == len(ends)
        if card.alignment:
            from openmind_core.alignment import Aligner

            epochs = Aligner.from_reference(reference.covariance, card.alignment).transform(epochs)  # type: ignore[union-attr]
        expected = model.predict_proba(epochs)  # type: ignore[attr-defined]
        online = np.array(
            [[p.probabilities[c] for c in CLASSES] for p in predictions if p.ok]  # type: ignore[index]
        )
        mask = np.array([p.ok for p in predictions])
        assert mask.mean() > 0.9
        np.testing.assert_allclose(online, expected[mask], atol=1e-6)
        np.testing.assert_allclose(
            [p.timestamp for p in predictions], 1000.0 + (ends - 1) / FS, atol=1e-9
        )


def test_decodes_the_imagined_class(plain: tuple) -> None:  # type: ignore[type-arg]
    labels = ["left", "forward", "right", "right", "left", "forward"]
    data, spans = recording(labels, 6.0, seed=11)
    decoder = OnlineDecoder(*plain, STREAM, FS)
    predictions, _ = push_in_blocks(decoder, as_stream(data))
    hits, total = 0, 0
    for p in predictions:
        end = round(p.timestamp * FS) + 1
        truth = next((lab for lab, a, b in spans if a + 1 * FS <= end - N_TIMES and end <= b), None)
        if truth is None or not p.ok:
            continue
        total += 1
        hits += max(p.probabilities, key=p.probabilities.get) == truth  # type: ignore[arg-type, union-attr]
    assert total > 40
    assert hits / total > 0.85, (hits, total)


def test_invalid_setups(plain: tuple) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match=r"needs channels \['CP4'\] missing"):
        OnlineDecoder(*plain, ("Fp1", "C3", "Cz", "C4", "FC3", "FC4", "CP3"), FS)
    with pytest.raises(ValueError, match=r"expects 250\.0 Hz"):
        OnlineDecoder(*plain, STREAM, 500.0)
    with pytest.raises(ValueError, match="takes 500"):
        OnlineDecoder(*plain, STREAM, FS, OnlineConfig(window_s=1.0))
    decoder = OnlineDecoder(*plain, STREAM, FS)
    with pytest.raises(ValueError, match="expected shape"):
        decoder.push(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="timestamps"):
        decoder.push(np.zeros((10, len(STREAM))), np.zeros(4))


def test_reference_must_match_the_card(aligned: tuple) -> None:  # type: ignore[type-arg]
    bad = Reference(np.eye(3), ("C3", "Cz", "C4"))
    with pytest.raises(ValueError, match="reference channels"):
        OnlineDecoder(*aligned, STREAM, FS, reference=bad)


def test_warmup_then_running(aligned: tuple) -> None:  # type: ignore[type-arg]
    data, _ = recording(["left", "right", "forward"] * 3, 6.0, seed=3)
    decoder = OnlineDecoder(*aligned, STREAM, FS, OnlineConfig(warmup_s=10.0))
    assert decoder.state == "calibrating"
    assert decoder.reference_covariance is None
    predictions, _ = push_in_blocks(decoder, as_stream(data))
    reasons = [p.reason for p in predictions]
    first_ok = next(i for i, p in enumerate(predictions) if p.ok)
    assert set(reasons[:first_ok]) == {"warmup"}
    # 10 s of reference: a 2 s window plus 32 steps of 0.25 s reach 2500 samples.
    assert first_ok == 32
    assert all(p.probabilities is None for p in predictions[:first_ok])
    assert all(p.ok for p in predictions[first_ok:])
    assert decoder.state == "running"
    reference = decoder.reference_covariance
    assert reference is not None
    assert reference.shape == (7, 7)


def covered_samples(predictions: list) -> int:  # type: ignore[type-arg]
    """Distinct samples covered by the windows ending at these predictions."""
    covered: set[int] = set()
    for p in predictions:
        end = round(p.timestamp * FS)
        covered.update(range(end - N_TIMES + 1, end + 1))
    return len(covered)


def test_warmup_counts_distinct_samples_around_rejected_windows(aligned: tuple) -> None:  # type: ignore[type-arg]
    data, _ = recording(["left", "right", "forward"] * 3, 6.0, seed=3)
    t = np.arange(round(1.0 * FS)) / FS
    data[round(5 * FS) : round(6 * FS), 2] += 1500 * np.sin(2 * np.pi * 10 * t) * np.hanning(len(t))
    decoder = OnlineDecoder(*aligned, STREAM, FS, OnlineConfig(warmup_s=10.0))
    predictions, _ = push_in_blocks(decoder, as_stream(data))
    first_ok = next(i for i, p in enumerate(predictions) if p.ok)
    assert any(p.reason == "rejected" for p in predictions[:first_ok])
    used = [p for p in predictions[:first_ok] if p.reason == "warmup"]
    # The reference rests on at least 10 s of distinct signal, and ends as soon as it does.
    assert covered_samples([*used, predictions[first_ok]]) >= 10.0 * FS
    assert covered_samples(used) < 10.0 * FS


def test_given_reference_starts_running(aligned: tuple) -> None:  # type: ignore[type-arg]
    reference = Reference(np.eye(7) * 25.0, MONTAGE_A_MI)
    decoder = OnlineDecoder(*aligned, STREAM, FS, reference=reference)
    assert decoder.state == "running"
    np.testing.assert_allclose(decoder.reference_covariance, np.eye(7) * 25.0)
    data, _ = recording(["left"], 4.0, seed=4)
    predictions, _ = push_in_blocks(decoder, as_stream(data))
    assert predictions[0].ok


def test_unaligned_model_ignores_warmup(plain: tuple) -> None:  # type: ignore[type-arg]
    decoder = OnlineDecoder(*plain, STREAM, FS)
    assert decoder.state == "running"
    assert decoder.reference_covariance is None


def test_large_artifacts_are_rejected(plain: tuple) -> None:  # type: ignore[type-arg]
    data, _ = recording(["left"] * 3, 6.0, seed=5)
    t = np.arange(round(1.0 * FS)) / FS
    data[round(8 * FS) : round(9 * FS), 2] += 1500 * np.sin(2 * np.pi * 10 * t) * np.hanning(len(t))
    decoder = OnlineDecoder(*plain, STREAM, FS)
    predictions, _ = push_in_blocks(decoder, as_stream(data))
    rejected = [p for p in predictions if p.reason == "rejected"]
    assert len(rejected) >= 4
    assert all(not p.ok and p.probabilities is None for p in rejected)
    # Every window overlapping the artifact (8-9 s) is rejected, the others are decoded.
    for p in predictions:
        end = (round(p.timestamp * FS) + 1) / FS
        overlaps = end > 8.0 + 0.1 and end - 2.0 < 9.0 - 0.1
        if overlaps:
            assert p.reason == "rejected", end
        elif end > 11.5 or end < 8.0:
            assert p.ok, (end, p.reason)


def test_jaw_double_clench_gives_an_action_and_pauses_decoding(plain: tuple) -> None:  # type: ignore[type-arg]
    from scipy import signal

    data, _ = recording(["left"] * 3, 6.0, seed=6)
    rng = np.random.default_rng(0)
    sos = signal.butter(4, 30.0, btype="highpass", fs=FS, output="sos")
    for at in (12.0, 12.6):  # 0.3 s clenches, 0.3 s apart
        n = round(0.3 * FS)
        noise = signal.sosfilt(sos, rng.standard_normal((n, 7)), axis=0)
        noise /= noise.std(axis=0)
        data[round(at * FS) : round(at * FS) + n] += (
            120 * noise * signal.windows.tukey(n, 0.3)[:, None]
        )
    times = 500.0 + np.arange(len(data)) / FS
    decoder = OnlineDecoder(*plain, STREAM, FS)
    predictions, actions = push_in_blocks(decoder, as_stream(data), times=times)
    assert len(actions) == 1
    assert 500.0 + 12.9 < actions[0] < 500.0 + 13.3
    muscle = [p for p in predictions if p.reason == "muscle"]
    assert muscle
    assert all(500.0 + 12.0 <= p.timestamp <= 500.0 + 13.2 for p in muscle)
    assert all(p.probabilities is None and not p.ok for p in muscle)
    assert any(p.ok for p in predictions if p.timestamp < 500.0 + 11.9)


def test_config_filters_default_to_the_card(plain: tuple) -> None:  # type: ignore[type-arg]
    model, card = plain
    custom = ModelCard(**{**card.__dict__, "filters": {"low_hz": 4.0, "high_hz": 40.0}})
    decoder = OnlineDecoder(model, custom, STREAM, FS)
    assert decoder._filter.config == FilterConfig(low_hz=4.0, high_hz=40.0)
    assert OnlineDecoder(model, card, STREAM, FS)._filter.config == FilterConfig()
