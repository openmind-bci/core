"""Tests for the double-blink detector (an eye signal, not a brain signal)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from openmind_core.preprocessing import BlinkConfig, DoubleBlinkDetector, detect_double_blinks

FS = 250.0


def pink_noise(seconds: float, rms: float = 15.0, seed: int = 0) -> np.ndarray:
    """1/f background similar to EEG on Fp1."""
    rng = np.random.default_rng(seed)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    x = signal.lfilter(b, a, rng.standard_normal(int(seconds * FS)))
    return np.asarray(x / x.std() * rms)


def add_blink(x: np.ndarray, at_s: float, amplitude: float = 120.0, length_s: float = 0.3) -> None:
    start, n = int(at_s * FS), int(length_s * FS)
    x[start : start + n] += amplitude * np.hanning(n)


def test_double_blink_is_detected() -> None:
    x = pink_noise(20)
    add_blink(x, 10.0)
    add_blink(x, 10.5)
    events = detect_double_blinks(x, FS)
    assert len(events) == 1
    assert 10.5 < events[0] < 10.9  # at the peak of the second blink


def test_single_blinks_are_not_double() -> None:
    x = pink_noise(30)
    for at in (5.0, 9.0, 13.0, 17.0, 21.0):  # natural blinks, seconds apart
        add_blink(x, at)
    detector = DoubleBlinkDetector(FS)
    assert detector.update(x) == []
    assert len(detector.blinks) == 5


def test_triple_blink_gives_one_event() -> None:
    x = pink_noise(20)
    for at in (10.0, 10.5, 11.0):
        add_blink(x, at)
    assert len(detect_double_blinks(x, FS)) == 1


def test_two_double_blinks_after_refractory() -> None:
    x = pink_noise(30)
    for at in (8.0, 8.5, 15.0, 15.5):
        add_blink(x, at)
    assert len(detect_double_blinks(x, FS)) == 2


@pytest.mark.parametrize("seed", range(5))
def test_background_alone_gives_no_detection(seed: int) -> None:
    assert detect_double_blinks(pink_noise(60, seed=seed), FS) == []


def test_small_blinks_below_threshold_are_ignored() -> None:
    x = pink_noise(20)
    add_blink(x, 10.0, amplitude=30)
    add_blink(x, 10.5, amplitude=30)
    assert detect_double_blinks(x, FS) == []


def test_online_blocks_give_the_same_events() -> None:
    x = pink_noise(30, seed=3)
    for at in (6.0, 6.5, 20.0, 20.5):
        add_blink(x, at)
    offline = detect_double_blinks(x, FS)
    detector = DoubleBlinkDetector(FS)
    online: list[float] = []
    for start in range(0, len(x), 25):  # 100 ms blocks
        online += detector.update(x[start : start + 25])
    assert online == pytest.approx(offline)
    assert len(online) == 2


def test_dc_offset_and_drift_are_ignored() -> None:
    x = pink_noise(20) + 30_000 + np.linspace(0, 400, int(20 * FS))
    add_blink(x, 10.0)
    add_blink(x, 10.5)
    assert len(detect_double_blinks(x, FS)) == 1


def test_threshold_adapts_to_noisy_background() -> None:
    detector = DoubleBlinkDetector(FS, BlinkConfig(min_amplitude_uv=20))
    detector.update(pink_noise(10, rms=40))
    assert detector.threshold_uv > 100


def test_with_the_synthetic_source_of_plug() -> None:
    synthetic = pytest.importorskip("openmind_plug.sources.synthetic")
    source = synthetic.SyntheticSource(montage="A", seed=0)
    fp1 = source.info.labels.index("Fp1")
    chunks = [source.generate(int(10 * FS))]
    source.blink()
    chunks.append(source.generate(int(0.5 * FS)))
    source.blink()
    chunks.append(source.generate(int(5 * FS)))
    for _ in range(3):  # then single blinks, 3 s apart
        source.blink()
        chunks.append(source.generate(int(3 * FS)))
    signal_fp1 = np.vstack(chunks)[:, fp1]
    events = detect_double_blinks(signal_fp1, FS)
    assert len(events) == 1
    assert 10.4 < events[0] < 10.9
