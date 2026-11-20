"""Tests for the jaw clench detector (a muscle signal, not a brain signal)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from openmind_core.preprocessing import JawClenchDetector, JawConfig, detect_double_clenches

FS = 250.0
N_CHANNELS = 8
STRONG_UV = 120.0  # about 26 dB above the background, like a deliberate clench
WEAK_UV = 40.0  # about 16 dB: below the 20 dB peak rule, like a swallow


def background(seconds: float, seed: int = 0, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Independent 1/f backgrounds of 15 µV RMS."""
    rng = np.random.default_rng(seed)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    x = signal.lfilter(b, a, rng.standard_normal((int(seconds * FS), n_channels)), axis=0)
    return np.asarray(x / x.std(axis=0) * 15.0)


def add_burst(
    x: np.ndarray,
    at_s: float,
    amplitude: float = STRONG_UV,
    length_s: float = 0.3,
    channels: slice | list[int] = slice(None),
    seed: int = 1,
) -> None:
    """Muscle burst: noise above 30 Hz with a tapered envelope."""
    rng = np.random.default_rng(seed + int(at_s * 1000))
    start, n = int(at_s * FS), int(length_s * FS)
    sos = signal.butter(4, 30.0, btype="highpass", fs=FS, output="sos")
    noise = signal.sosfilt(sos, rng.standard_normal((n, x.shape[1])), axis=0)
    noise /= noise.std(axis=0)
    burst = amplitude * noise * signal.windows.tukey(n, alpha=0.3)[:, None]
    x[start : start + n, channels] += burst[:, channels]


def add_double(x: np.ndarray, at_s: float, gap_s: float = 0.6, **kwargs: object) -> None:
    """Clench 0.3 s, release 0.3 s, clench again: the documented gesture."""
    add_burst(x, at_s, **kwargs)  # type: ignore[arg-type]
    add_burst(x, at_s + gap_s, **kwargs)  # type: ignore[arg-type]


def test_double_clench_is_detected() -> None:
    x = background(20)
    add_double(x, 10.0)
    events = detect_double_clenches(x, FS)
    assert len(events) == 1
    # Reported once the second burst (10.6-10.9 s) has ended.
    assert 10.9 < events[0] < 11.2


def test_single_clenches_are_not_double() -> None:
    x = background(30)
    for at in (5.0, 9.0, 13.0, 17.0, 21.0):
        add_burst(x, at)
    detector = JawClenchDetector(FS, N_CHANNELS)
    assert detector.update(x) == []
    assert len(detector.clenches) == 5
    assert all(
        abs(t - at) < 0.1 for t, at in zip(detector.clenches, (5, 9, 13, 17, 21), strict=True)
    )


def test_triple_clench_gives_one_event() -> None:
    x = background(20)
    for at in (10.0, 10.6, 11.2):
        add_burst(x, at)
    assert len(detect_double_clenches(x, FS)) == 1


def test_two_double_clenches_after_refractory() -> None:
    x = background(25)
    add_double(x, 8.0)
    add_double(x, 14.0)
    assert len(detect_double_clenches(x, FS)) == 2


@pytest.mark.parametrize("seed", range(5))
def test_background_alone_gives_no_detection(seed: int) -> None:
    detector = JawClenchDetector(FS, N_CHANNELS)
    assert detector.update(background(60, seed)) == []
    assert detector.clenches == []


def test_weak_bursts_are_ignored() -> None:
    x = background(20)
    add_double(x, 10.0, amplitude=WEAK_UV)
    detector = JawClenchDetector(FS, N_CHANNELS)
    assert detector.update(x) == []
    assert detector.clenches == []


def test_long_bursts_are_ignored_but_pause_decoding() -> None:
    x = background(20)
    add_burst(x, 10.0, length_s=2.0)  # a sustained contraction, not a clench
    add_burst(x, 12.5, length_s=2.0)
    detector = JawClenchDetector(FS, N_CHANNELS)
    assert detector.update(x[: int(11 * FS)]) == []
    assert detector.active  # muscle activity in progress
    assert detector.update(x[int(11 * FS) :]) == []
    assert detector.clenches == []
    assert not detector.active


def test_one_electrode_cannot_trigger() -> None:
    x = background(20)
    add_double(x, 10.0, amplitude=4 * STRONG_UV, channels=[2])  # a bad contact on C3
    assert detect_double_clenches(x, FS) == []


def test_online_blocks_give_the_same_events() -> None:
    x = background(40)
    add_double(x, 10.0)
    add_burst(x, 20.0)
    add_double(x, 30.0)
    offline = JawClenchDetector(FS, N_CHANNELS)
    expected = offline.update(x)
    rng = np.random.default_rng(3)
    online = JawClenchDetector(FS, N_CHANNELS)
    events: list[float] = []
    start = 0
    while start < len(x):
        size = int(rng.integers(1, 120))
        events += online.update(x[start : start + size])
        start += size
    assert len(expected) == 2
    assert events == pytest.approx(expected)
    assert online.clenches == pytest.approx(offline.clenches)


def test_no_detection_before_a_background_is_known() -> None:
    x = background(5)
    add_double(x, 0.2)  # within the first second
    detector = JawClenchDetector(FS, N_CHANNELS)
    assert detector.update(x) == []
    assert detector.ready


def test_dc_offsets_are_ignored() -> None:
    x = background(20) + np.linspace(-5000, 5000, N_CHANNELS)[None, :]
    add_double(x, 10.0)
    assert len(detect_double_clenches(x, FS)) == 1


def test_lower_sampling_rates_use_a_narrower_band() -> None:
    srate = 160.0
    x = signal.resample_poly(background(20), 16, 25, axis=0)
    rng = np.random.default_rng(0)
    for at in (10.0, 10.6):
        start, n = int(at * srate), int(0.3 * srate)
        sos = signal.butter(4, 30.0, btype="highpass", fs=srate, output="sos")
        noise = signal.sosfilt(sos, rng.standard_normal((n, N_CHANNELS)), axis=0)
        x[start : start + n] += STRONG_UV * noise / noise.std(axis=0)
    assert len(detect_double_clenches(x, srate)) == 1


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="no muscle band"):
        JawClenchDetector(60.0, N_CHANNELS)
    detector = JawClenchDetector(FS, N_CHANNELS)
    with pytest.raises(ValueError, match="expected shape"):
        detector.update(np.zeros((10, 3)))
    assert detector.update(np.zeros((0, N_CHANNELS))) == []


def test_config_changes_the_rules() -> None:
    x = background(20)
    add_double(x, 10.0, amplitude=WEAK_UV)
    loose = JawConfig(min_peak_db=12.0)
    assert len(detect_double_clenches(x, FS, loose)) == 1


def test_with_the_synthetic_source_of_plug() -> None:
    synthetic = pytest.importorskip("openmind_plug.sources.synthetic")
    source = synthetic.SyntheticSource(montage="C", srate=FS, seed=4)
    if not hasattr(source, "clench"):
        pytest.skip("this version of openmind-plug has no jaw clench")
    detector = JawClenchDetector(FS, source.info.n_channels)
    events = detector.update(source.generate(int(5 * FS)))
    times = []
    for _ in range(4):
        source.clench(2)
        times.append(source.n_produced / FS)
        events += detector.update(source.generate(int(4 * FS)))
    source.clench(1)  # a single clench is not an action
    events += detector.update(source.generate(int(4 * FS)))
    assert len(events) == len(times)
    for event, at in zip(events, times, strict=True):
        assert at < event < at + 2.0
