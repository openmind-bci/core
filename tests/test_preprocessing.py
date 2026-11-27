"""Tests for filters, windows and artifact rejection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from openmind_core.preprocessing import (
    EPOCH_OFFSET_S,
    FilterConfig,
    SlidingWindow,
    StreamFilter,
    amplitude_mask,
    epochs_at,
    filter_continuous,
    window_ends,
)

FS = 250.0


def sines(freqs: list[float], seconds: float = 20.0, amplitude: float = 10.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.column_stack([amplitude * np.sin(2 * np.pi * f * t) for f in freqs])


def power_at(x: np.ndarray, freq: float) -> np.ndarray:
    f, p = welch(x[int(5 * FS) :], fs=FS, nperseg=1000, axis=0)
    return p[np.argmin(np.abs(f - freq))]


# --- Filters --------------------------------------------------------------------


def test_bandpass_keeps_mu_beta_and_removes_the_rest() -> None:
    x = sines([2.0, 10.0, 20.0, 40.0])
    y = filter_continuous(x, FS, FilterConfig(notch_hz=None))
    ratios = [
        power_at(y[:, [i]], f)[0] / power_at(x[:, [i]], f)[0]
        for i, f in enumerate([2.0, 10.0, 20.0, 40.0])
    ]
    assert ratios[0] < 0.01  # 2 Hz drift removed
    assert ratios[1] > 0.7 and ratios[2] > 0.7  # mu and beta kept
    assert ratios[3] < 0.05  # 40 Hz removed


@pytest.mark.parametrize("line", [50.0, 60.0])
def test_notch_removes_power_line(line: float) -> None:
    x = sines([line])
    without = filter_continuous(x, FS, FilterConfig(low_hz=8, high_hz=100, notch_hz=None))
    with_notch = filter_continuous(x, FS, FilterConfig(low_hz=8, high_hz=100, notch_hz=line))
    assert power_at(with_notch, line)[0] < 0.01 * power_at(without, line)[0]


def test_block_by_block_equals_whole_recording() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20, (3000, 7)) + 5000  # large DC offset, like real amplifiers
    whole = filter_continuous(x, FS)
    online = StreamFilter(FS, 7)
    blocks = [
        online(x[i : i + n])
        for i, n in zip([0, 1, 64, 600, 2000], [1, 63, 536, 1400, 1000], strict=True)
    ]
    np.testing.assert_allclose(np.vstack(blocks), whole, atol=1e-9)


def test_dc_offset_does_not_ring() -> None:
    x = np.full((1000, 2), 20_000.0)
    assert np.abs(filter_continuous(x, FS)).max() < 1e-6


def test_reset_restarts_the_state() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, (500, 2))
    f = StreamFilter(FS, 2)
    first = f(x)
    f.reset()
    np.testing.assert_allclose(f(x), first)


def test_filter_validation() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        FilterConfig(low_hz=8, high_hz=200).design(FS)
    with pytest.raises(ValueError, match="shape"):
        StreamFilter(FS, 3)(np.zeros((10, 2)))
    assert StreamFilter(FS, 2)(np.zeros((0, 2))).shape == (0, 2)
    # A notch above Nyquist is skipped instead of failing.
    assert FilterConfig(notch_hz=60.0).design(100.0).shape[0] == 4


# --- Windows --------------------------------------------------------------------


def test_epochs_at_cuts_after_the_cue() -> None:
    data = np.arange(2000, dtype=float)[:, None] * np.array([1.0, -1.0])
    epochs, kept = epochs_at(data, [100, 1600, -500], FS)
    assert kept.tolist() == [True, False, False]  # 1600 + 125 + 500 > 2000
    assert epochs.shape == (1, 2, 500)
    start = 100 + int(EPOCH_OFFSET_S * FS)
    np.testing.assert_array_equal(epochs[0, 0], np.arange(start, start + 500))
    np.testing.assert_array_equal(epochs[0, 1], -np.arange(start, start + 500))


def test_sliding_window_matches_offline_epochs() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1500, 3))
    window = SlidingWindow(3, FS)
    produced = []
    for start in range(0, 1500, 37):  # irregular block sizes
        produced += window.push(x[start : start + 37])
    assert window.n_times == 500 and window.step == 0.25 * FS
    # Windows end every 62.5 samples on average (63 and 62 alternately) once 500
    # samples are buffered, like the offline ``window_ends``.
    ends = window_ends(1500, FS)
    assert ends.tolist()[:4] == [500, 563, 625, 688]
    assert len(produced) == len(ends) == 1 + (1500 - 500) * 2 // 125
    for window_data, end in zip(produced, ends, strict=True):
        np.testing.assert_array_equal(window_data, x[end - 500 : end].T)


def test_sliding_window_keeps_exact_steps_over_a_long_stream() -> None:
    minutes = 10
    n = round(minutes * 60 * FS)
    window = SlidingWindow(1, FS)
    count = 0
    for start in range(0, n, 997):
        count += len(window.push(np.zeros((min(997, n - start), 1))))
    # One window every 0.25 s after the first 2 s: no 0.8 % drift (62-sample steps).
    assert count == 1 + (minutes * 60 - 2) * 4


# --- Artifacts ------------------------------------------------------------------


def test_amplitude_mask() -> None:
    epochs = np.zeros((4, 2, 100))
    epochs[1, 0, 10] = 200.0  # too large
    epochs[2, 1, :] = np.linspace(-70, 70, 100)  # 140 µV peak to peak: kept
    epochs[3, 0, 0] = np.nan
    assert amplitude_mask(epochs).tolist() == [True, False, True, False]
    assert amplitude_mask(epochs, max_ptp_uv=100).tolist() == [True, False, False, False]
    assert amplitude_mask(np.zeros((0, 2, 100))).shape == (0,)
