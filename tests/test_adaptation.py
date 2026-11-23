"""Tests of online adaptation: re-centering, pseudo-labels and rollback."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from openmind_core.alignment import align_epochs
from openmind_core.calibration import PersonalModel
from openmind_core.datasets import MONTAGE_A_MI, EpochSet
from openmind_core.models import make_baseline
from openmind_core.models.registry import ModelCard
from openmind_core.online import OnlineConfig, OnlineDecoder
from openmind_core.online.adaptation import (
    AdaptationGuard,
    PseudoLabelTuner,
    RecenteringAdapter,
    geodesic_step,
    riemann_distance,
)
from openmind_core.preprocessing import FilterConfig, epochs_at, filter_continuous
from openmind_core.profiles import Reference

FS = 250.0
N_CH = len(MONTAGE_A_MI)
CLASSES = ("forward", "left", "right")
FILTERS = FilterConfig()
ATTENUATED = {"left": ("C4", "CP4"), "right": ("FC3", "C3", "CP3"), "forward": ("Cz",)}
SEGMENT_S = 5.0


def make_card() -> ModelCard:
    return ModelCard(
        name="synthetic",
        architecture="ts-lr",
        channels=MONTAGE_A_MI,
        srate=FS,
        n_times=500,
        classes=CLASSES,
        scale=(1.0,) * N_CH,
        format="sklearn",
        alignment="riemann",
        filters=asdict(FILTERS),
    )


def user_stream(
    labels: list[str], seed: int, final_gain: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Cued segments; per-channel gains drift slowly from 1 to ``final_gain``."""
    rng = np.random.default_rng(seed)
    n_seg = round(SEGMENT_S * FS)
    parts, spans = [], []
    phase = np.linspace(0, 3, N_CH)
    for k, label in enumerate(labels):
        t = (k * n_seg + np.arange(n_seg)) / FS
        amplitude = np.full(N_CH, 20.0)
        for channel in ATTENUATED[label]:
            amplitude[MONTAGE_A_MI.index(channel)] = 3.0
        rhythm = amplitude * np.sin(2 * np.pi * 10 * t[:, None] + phase)
        parts.append(rhythm + rng.normal(0, 5, (n_seg, N_CH)))
        spans.append((label, k * n_seg, (k + 1) * n_seg))
    data = np.vstack(parts)
    if final_gain is not None:
        progress = np.linspace(0, 1, len(data))[:, None]
        data = data * final_gain[None, :] ** progress
    return data, spans


@pytest.fixture(scope="module")
def generic() -> object:
    data, spans = user_stream(list(CLASSES) * 8, seed=1)
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
    aligned = align_epochs(epochs, "riemann")
    return make_baseline("ts-lr").fit(aligned.X, aligned.y)


# Slow drift: right-hand electrodes gain amplitude, left-hand ones lose it.
DRIFT = np.array(
    [
        0.35 if c in ("FC3", "C3", "CP3") else 2.8 if c in ("C4", "CP4") else 1.0
        for c in MONTAGE_A_MI
    ]
)


def run_session(
    model: object,
    adapter: RecenteringAdapter | None,
    seed: int = 5,
    warmup_s: float = 10.0,
) -> tuple[float, OnlineDecoder]:
    labels = list(CLASSES) * 12
    data, spans = user_stream(labels, seed=seed, final_gain=DRIFT)
    decoder = OnlineDecoder(model, make_card(), MONTAGE_A_MI, FS, OnlineConfig(warmup_s=warmup_s))
    if adapter is not None:
        adapter.attach(decoder)
    times = np.arange(len(data)) / FS
    predictions = []
    for pos in range(0, len(data), 250):
        predictions += decoder.push(data[pos : pos + 250], times[pos : pos + 250]).predictions
    hits = total = 0
    for p in predictions:
        if not p.ok or p.probabilities is None or p.timestamp < 60.0:
            continue
        start = p.timestamp - 2.0 + 1 / FS
        k = int(p.timestamp // SEGMENT_S)
        if int(start // SEGMENT_S) != k:
            continue  # window straddling two cues
        total += 1
        hits += max(p.probabilities, key=lambda c: p.probabilities[c]) == spans[k][0]
    return hits / total, decoder


def test_recentering_follows_slow_drift(generic: object) -> None:
    plain, _ = run_session(generic, None)
    adapter = RecenteringAdapter(rate=0.02, max_step=0.05, max_drift=20.0, min_windows=8)
    adapted, decoder = run_session(generic, adapter)
    assert adapter.updates > 100
    assert adapter.drift > 0.5
    assert adapted >= plain
    assert adapted > 0.8
    assert decoder.reference_covariance is not None


def warmed_decoder(model: object) -> tuple[OnlineDecoder, np.ndarray]:
    """Decoder past its warm-up, and the rest of a drifting recording to push."""
    data, _ = user_stream(list(CLASSES) * 6, seed=3, final_gain=DRIFT)
    decoder = OnlineDecoder(model, make_card(), MONTAGE_A_MI, FS, OnlineConfig(warmup_s=10.0))
    return decoder, data


def start(
    model: object, adapter: RecenteringAdapter | None = None
) -> tuple[OnlineDecoder, np.ndarray]:
    decoder, data = warmed_decoder(model)
    if adapter is not None:
        adapter.attach(decoder)
    cut = round(15.0 * FS)
    decoder.push(data[:cut])
    assert decoder.state == "running"
    return decoder, data[cut:]


def random_windows(n: int, scale: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, (n, N_CH, 500))


def test_decoder_set_reference(generic: object) -> None:
    decoder, _ = start(generic)
    before = decoder.reference_covariance
    assert before is not None
    decoder.set_reference(before * 4.0)
    np.testing.assert_allclose(decoder.reference_covariance, before * 4.0)
    with pytest.raises(ValueError, match="covariance"):
        decoder.set_reference(np.eye(3))
    plain = OnlineDecoder(
        generic, ModelCard(**{**asdict(make_card()), "alignment": None}), MONTAGE_A_MI, FS
    )
    with pytest.raises(ValueError, match="alignment"):
        plain.set_reference(np.eye(N_CH))
    with pytest.raises(ValueError, match="aligned"):
        RecenteringAdapter().attach(plain)


def test_geodesic_and_distance() -> None:
    a = np.diag([1.0, 2.0, 4.0])
    b = np.diag([4.0, 2.0, 1.0])
    assert riemann_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    mid = geodesic_step(a, b, 0.5)
    assert riemann_distance(a, mid) == pytest.approx(riemann_distance(a, b) / 2)
    np.testing.assert_allclose(geodesic_step(a, b, 1.0), b, atol=1e-9)


def test_half_life_sets_rate() -> None:
    adapter = RecenteringAdapter(half_life_s=60.0, step_s=0.25)
    assert (1 - adapter.rate) ** (60.0 / 0.25) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="not both"):
        RecenteringAdapter(rate=0.1, half_life_s=60.0)


@pytest.mark.parametrize("method", ["riemann", "euclid"])
def test_step_cap(method: str) -> None:
    adapter = RecenteringAdapter(method, rate=1.0, min_windows=0, max_step=0.05, max_drift=100.0)
    reference = np.eye(N_CH) * 100.0  # windows are 25 times weaker than the reference
    adapter.observe(random_windows(1, 1.0)[0], reference)
    last = adapter.reference
    for window in random_windows(20, 1.0, seed=1):
        adapter.observe(window)
        assert last is not None and adapter.reference is not None
        assert riemann_distance(last, adapter.reference) <= 0.05 + 1e-9
        last = adapter.reference
    assert adapter.updates == 21
    assert adapter.drift == pytest.approx(21 * 0.05, rel=1e-3)


def test_drift_cap_freezes() -> None:
    adapter = RecenteringAdapter("riemann", rate=1.0, min_windows=0, max_step=0.1, max_drift=0.3)
    adapter.observe(random_windows(1, 1.0)[0], np.eye(N_CH) * 100.0)
    for window in random_windows(10, 1.0, seed=2):
        adapter.observe(window)
    assert adapter.frozen
    assert adapter.updates == 3
    assert adapter.drift <= 0.3
    frozen_at = adapter.reference
    assert not adapter.observe(random_windows(1, 1.0)[0])
    np.testing.assert_array_equal(adapter.reference, frozen_at)
    assert AdaptationGuard(_dummy_decoder(), adapter).status_message() == (
        "adaptation frozen (drift limit)"
    )


def _dummy_decoder() -> OnlineDecoder:
    return OnlineDecoder(
        object(), make_card(), MONTAGE_A_MI, FS, reference=Reference(np.eye(N_CH), MONTAGE_A_MI)
    )


def test_min_windows_delay() -> None:
    adapter = RecenteringAdapter(rate=1.0, min_windows=5, max_drift=100.0)
    windows = random_windows(8, 1.0)
    moved = [adapter.observe(w, np.eye(N_CH) * 50.0) for w in windows]
    assert moved == [False] * 5 + [True] * 3


def test_initial_model_and_reference_untouched(generic: object) -> None:
    adapter = RecenteringAdapter(rate=0.02, min_windows=5, max_drift=20.0)
    decoder, rest = start(generic, adapter)
    guard = AdaptationGuard(decoder, adapter)
    probe = random_windows(4, 1.0, seed=9)
    before = np.asarray(generic.predict_proba(probe))  # type: ignore[attr-defined]
    initial = guard.initial_reference
    assert initial is not None
    snapshot = initial.copy()
    guard.begin_block()
    decoder.push(rest)
    assert adapter.updates > 0 and adapter.drift > 0.1
    np.testing.assert_array_equal(initial, snapshot)
    assert not initial.flags.writeable
    np.testing.assert_array_equal(generic.predict_proba(probe), before)  # type: ignore[attr-defined]
    guard.reset()
    np.testing.assert_allclose(decoder.reference_covariance, snapshot)
    assert adapter.drift == pytest.approx(0.0, abs=1e-9)


def test_unsupervised_running_score(generic: object) -> None:
    decoder, rest = start(generic)
    guard = AdaptationGuard(decoder)
    guard.begin_block()
    assert guard.running_score() is None
    decoder.push(rest[: round(20 * FS)])
    score = guard.running_score()
    assert score is not None and 1 / 3 <= score <= 1.0
    assert guard.end_block() is False
    assert guard.end_block() is False  # nothing decoded since: no score, no change


class StubRecorder:
    """Labelled epochs of a calibration, as ``PersonalModel`` reads them."""

    def __init__(self, per_class: int = 6) -> None:
        self._y = [c for c in CLASSES for _ in range(per_class)]
        self._x = random_windows(len(self._y), 5.0, seed=4)

    def labels(self) -> list[str]:
        return list(self._y)

    def epochs(self) -> np.ndarray:
        return self._x


def make_personal(generic: object, decoder: OnlineDecoder) -> PersonalModel:
    personal = PersonalModel(
        generic,
        make_card(),
        StubRecorder(),  # type: ignore[arg-type]
        reference=decoder.reference_covariance,
    )
    assert personal.update()
    return personal


def test_rollback_restores_reference_and_model(generic: object) -> None:
    adapter = RecenteringAdapter(rate=0.05, min_windows=2, max_drift=20.0)
    tuner = PseudoLabelTuner(threshold=0.5, enabled=True)
    decoder, rest = start(generic, adapter)
    tuner.attach(decoder)
    personal = make_personal(generic, decoder)
    personal.pseudo = tuner
    guard = AdaptationGuard(decoder, adapter, personal, tuner, max_drop=0.10)

    guard.begin_block()
    safe_reference = decoder.reference_covariance
    safe_state = personal.get_state()
    assert safe_reference is not None
    assert not guard.end_block(0.90)

    guard.begin_block()
    decoder.push(rest[: round(30 * FS)])  # adaptation moves the reference, fills the tuner
    assert adapter.updates > 0 and len(tuner) > 0
    personal.update()  # refit: a new personal model
    assert personal.get_state().personal is not safe_state.personal
    assert guard.end_block(0.85) is False  # within max_drop: kept
    assert not guard.rollbacks

    guard.begin_block()
    decoder.push(rest[round(30 * FS) : round(50 * FS)])
    assert guard.end_block(0.60) is True  # 0.30 under the best recent block
    assert len(guard.rollbacks) == 1
    assert guard.status_message() == "adaptation rolled back"
    np.testing.assert_allclose(decoder.reference_covariance, safe_reference)
    np.testing.assert_allclose(adapter.reference, safe_reference)
    assert personal.get_state().personal is safe_state.personal
    assert personal.weight == safe_state.weight
    assert len(tuner) == 0
    assert adapter.paused and tuner.paused

    # The pause lasts one block, then the message goes away.
    guard.begin_block()
    assert not guard.end_block(0.90)
    assert not adapter.paused and not tuner.paused
    assert guard.status_message() is None


def test_no_rollback_when_score_holds_or_improves(generic: object) -> None:
    decoder, _ = start(generic)
    guard = AdaptationGuard(decoder, max_drop=0.10)
    for score in (0.5, 0.6, 0.55, 0.7):
        guard.begin_block()
        assert guard.end_block(score) is False
    assert guard.rollbacks == []
    assert guard.status_message() is None


def test_pseudo_labels_off_by_default() -> None:
    tuner = PseudoLabelTuner()
    assert not tuner.enabled
    proba = {"forward": 0.01, "left": 0.98, "right": 0.01}
    assert not any(tuner.observe(w, proba) for w in random_windows(5, 1.0))
    assert len(tuner) == 0
    x, y = tuner.training_set({"left": 10})
    assert len(x) == 0 and len(y) == 0


def test_pseudo_labels_threshold_and_cap() -> None:
    tuner = PseudoLabelTuner(threshold=0.9, enabled=True, smoothing=1.0, max_share=0.3)
    confident = {"forward": 0.02, "left": 0.96, "right": 0.02}
    unsure = {"forward": 0.3, "left": 0.4, "right": 0.3}
    windows = random_windows(40, 1.0)
    assert not tuner.observe(windows[0], unsure)
    assert not tuner.observe(windows[1], None)
    assert all(tuner.observe(w, confident) for w in windows[2:])
    assert len(tuner) == 38
    # 10 true epochs per class: at most 30 % of the final set, i.e. 10 * 0.3/0.7 = 4.
    x, y = tuner.training_set({"forward": 10, "left": 10, "right": 10})
    assert x.shape == (4, N_CH, 500)
    assert set(y) == {"left"}
    total = 30 + len(y)
    assert len(y) / total <= 0.3 + 1e-9
    # Smoothing: one confident window after unsure ones does not pass the threshold.
    slow = PseudoLabelTuner(threshold=0.9, enabled=True, smoothing=0.3)
    slow.observe(windows[0], unsure)
    assert not slow.observe(windows[1], confident)


def test_personal_model_trains_with_capped_pseudo_labels(
    generic: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openmind_core.calibration.adaptive as adaptive

    sizes: list[tuple[int, dict[str, int]]] = []
    real = adaptive.ts_lr

    def spy(c: float) -> object:
        model = real(c)
        fit = model.fit

        def recording_fit(X: np.ndarray, y: np.ndarray) -> object:
            sizes.append((len(y), {c: int(np.sum(y == c)) for c in CLASSES}))
            return fit(X, y)

        model.fit = recording_fit  # type: ignore[method-assign]
        return model

    monkeypatch.setattr(adaptive, "ts_lr", spy)
    decoder, _ = start(generic)
    personal = make_personal(generic, decoder)
    assert sizes[-1][0] == 18  # true labels only
    tuner = PseudoLabelTuner(threshold=0.5, enabled=True, smoothing=1.0)
    for window in random_windows(50, 5.0, seed=6):
        tuner.observe(window, {"forward": 0.1, "left": 0.1, "right": 0.8})
    personal.pseudo = tuner
    assert personal.update()
    total, per_class = sizes[-1]
    assert per_class["right"] == 6 + 2  # 6 * 0.3 / 0.7 = 2.57 -> 2
    assert per_class["left"] == per_class["forward"] == 6
    assert (total - 18) / total <= 0.3
    tuner.enabled = False
    assert personal.update()
    assert sizes[-1][0] == 18
