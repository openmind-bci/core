"""Tests of the online band power estimator (no optional extras)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from openmind_core.preprocessing import (
    BANDS,
    BandPowerConfig,
    BandPowerEstimator,
    band_power,
    channel_ok,
)
from openmind_protocol import BandPower, decode, encode

SRATE = 250.0
LABELS = ["C3", "Cz", "C4"]
ALPHA, MU, BETA = 0, 1, 2


def _noise(
    n: int, n_channels: int = 3, seed: int = 0, sigma: float = 2.0
) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).normal(0.0, sigma, (n, n_channels))


def _sine(n: int, freq: float, amp: float = 10.0) -> npt.NDArray[np.float64]:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / SRATE)


def test_mu_sine_dominates_beta() -> None:
    window = np.tile(_sine(500, 10.0), (3, 1)) + _noise(500).T * 0.1
    power = band_power(window, SRATE)
    assert power.shape == (3, 3)
    assert (power[:, MU] > 20 * power[:, BETA]).all()
    assert (power[:, ALPHA] > 20 * power[:, BETA]).all()


def test_beta_sine_dominates_mu() -> None:
    window = np.tile(_sine(500, 20.0), (3, 1)) + _noise(500).T * 0.1
    power = band_power(window, SRATE)
    assert (power[:, BETA] > 20 * power[:, MU]).all()


def test_white_noise_psd_matches_variance() -> None:
    sigma = 5.0
    window = _noise(5000, 1, seed=3, sigma=sigma).T
    power = band_power(window, SRATE)
    expected = 2 * sigma**2 / SRATE
    assert power[0] == pytest.approx([expected] * 3, rel=0.2)


def test_bands_match_protocol_example() -> None:
    assert BANDS == {"alpha": (8.0, 12.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def test_channel_ok_rules() -> None:
    window = _noise(500).T.copy()
    window[0] = 1.0  # flat
    window[1, 10] = np.nan
    saturated = _noise(500, 1, seed=5).T[0]
    saturated[100] = 5000.0
    clipped = np.clip(_noise(500, 1, seed=6, sigma=50.0).T[0], -20, 20)
    full = np.vstack([window, saturated, clipped, _noise(500, 1, seed=7).T])
    assert full.shape[0] == 6
    assert channel_ok(full).tolist() == [False, False, True, False, False, True]


def _stream(n: int = 2500) -> npt.NDArray[np.float64]:
    return _noise(n) + _sine(n, 10.0)[:, None]


def test_block_by_block_equals_one_push() -> None:
    data = _stream()
    stamps = 100.0 + np.arange(len(data)) / SRATE
    whole = BandPowerEstimator(LABELS, SRATE).push(data, stamps)
    estimator = BandPowerEstimator(LABELS, SRATE)
    split: list = []
    for start in range(0, len(data), 37):
        split += estimator.push(data[start : start + 37], stamps[start : start + 37])
    assert len(whole) == len(split) > 0
    for a, b in zip(whole, split, strict=True):
        assert a.timestamp == b.timestamp
        np.testing.assert_array_equal(a.power, b.power)
        np.testing.assert_array_equal(a.ok, b.ok)


def test_frame_rate_and_timestamps() -> None:
    data = _stream(250 * 12)
    stamps = np.arange(len(data)) / SRATE
    frames = BandPowerEstimator(LABELS, SRATE).push(data, stamps)
    # 12 s minus the first 2 s window, one frame per 0.1 s
    assert len(frames) == pytest.approx(100, abs=1)
    assert frames[0].timestamp == pytest.approx(2.0 - 1 / SRATE)
    steps = np.diff([f.timestamp for f in frames])
    assert steps == pytest.approx(0.1, abs=1 / SRATE)
    assert BandPowerEstimator(LABELS, SRATE).push(data)[0].timestamp is None


def test_bad_channels_flagged_and_none_in_message() -> None:
    data = _stream()
    data[:, 0] = 3.0  # flat
    data[2400, 1] = np.nan
    frame = BandPowerEstimator(LABELS, SRATE).push(data)[-1]
    assert frame.ok.tolist() == [False, False, True]
    assert np.isnan(frame.power[:2]).all()
    message = frame.to_message("P01", SRATE)
    assert [c.ok for c in message.channels] == [False, False, True]
    assert message.channel("C3").mu is None
    assert message.channel("Cz").beta is None
    assert message.channel("C4").mu is not None


def test_saturated_channel_not_ok() -> None:
    data = _stream()
    data[:, 2] = np.clip(data[:, 2] * 200, -3000, 3000)
    assert not BandPowerEstimator(LABELS, SRATE).push(data)[-1].ok[2]


def test_missing_expected_label_appended() -> None:
    frame = BandPowerEstimator(LABELS, SRATE).push(_stream())[-1]
    message = frame.to_message("P01", SRATE, expected_labels=["Fp1", "C3", "C4"])
    assert [c.label for c in message.channels] == [*LABELS, "Fp1"]
    fp1 = message.channel("Fp1")
    assert not fp1.ok
    assert fp1.alpha is fp1.mu is fp1.beta is fp1.rel_mu is fp1.rel_beta is None


def test_baseline_start_stop_and_erd() -> None:
    rest = _noise(250 * 8, seed=1) + _sine(250 * 8, 10.0, 10.0)[:, None]
    task = _noise(250 * 8, seed=2) + _sine(250 * 8, 10.0, 10.0 / np.sqrt(2))[:, None]
    estimator = BandPowerEstimator(LABELS, SRATE)
    assert estimator.baseline is None
    assert estimator.push(rest[:600])[-1].relative is None
    estimator.start_baseline()
    estimator.push(rest[600:])
    baseline = estimator.stop_baseline()
    assert baseline.shape == (3, 3)
    np.testing.assert_array_equal(estimator.baseline, baseline)
    frame = estimator.push(task)[-1]
    assert frame.relative is not None
    assert frame.relative[:, MU] == pytest.approx([-0.5] * 3, abs=0.1)
    message = frame.to_message("P01", SRATE)
    assert message.channel("C3").rel_mu == pytest.approx(-0.5, abs=0.1)


def test_baseline_errors_and_invalid_values() -> None:
    estimator = BandPowerEstimator(LABELS, SRATE)
    with pytest.raises(RuntimeError):
        estimator.stop_baseline()
    estimator.start_baseline()
    with pytest.raises(RuntimeError):
        estimator.stop_baseline()
    with pytest.raises(ValueError, match="shape"):
        estimator.set_baseline(np.ones((2, 3)))
    baseline = np.ones((3, 3))
    baseline[0] = 0.0
    baseline[1] = np.nan
    estimator.set_baseline(baseline)
    frame = estimator.push(_stream())[-1]
    assert frame.relative is not None
    assert np.isnan(frame.relative[:2]).all()
    assert np.isfinite(frame.relative[2]).all()
    message = frame.to_message(None, SRATE)
    assert message.channel("C3").rel_mu is None
    assert message.channel("C3").ok


def test_smoothing_reduces_variability() -> None:
    data = _noise(250 * 20, 1, seed=4, sigma=5.0)

    def spread(smoothing: float) -> float:
        config = BandPowerConfig(smoothing=smoothing)
        frames = BandPowerEstimator(["C3"], SRATE, config).push(data)
        return float(np.std([f.power[0, MU] for f in frames]))

    assert spread(0.2) < spread(1.0)


def test_smoothing_one_is_raw_and_ema_resets_after_bad_channel() -> None:
    data = _noise(250 * 6, 1, seed=8, sigma=5.0)
    raw = BandPowerEstimator(["C3"], SRATE, BandPowerConfig(smoothing=1.0)).push(data)
    first = BandPowerEstimator(["C3"], SRATE, BandPowerConfig(smoothing=0.5)).push(data)
    assert first[0].power[0, MU] == raw[0].power[0, MU]
    np.testing.assert_allclose(first[1].power[0], 0.5 * raw[1].power[0] + 0.5 * raw[0].power[0])
    # a channel that was bad starts again from its new value
    bad = data.copy()
    bad[400:1000] = 0.0
    frames = BandPowerEstimator(["C3"], SRATE, BandPowerConfig(smoothing=0.5)).push(bad)
    flags = [f.ok[0] for f in frames]
    index = flags.index(True, flags.index(False))
    reference = BandPowerEstimator(["C3"], SRATE, BandPowerConfig(smoothing=1.0)).push(bad)
    np.testing.assert_array_equal(frames[index].power, reference[index].power)


def test_message_round_trip() -> None:
    estimator = BandPowerEstimator(LABELS, SRATE)
    estimator.set_baseline(np.full((3, 3), 20.0))
    stamps = 1_763_478_000.0 + np.arange(2500) / SRATE
    frame = estimator.push(_stream(), stamps)[-1]
    message = frame.to_message("P03", SRATE, expected_labels=["Fp1"])
    assert message.ts == frame.timestamp
    decoded = decode(encode(message), BandPower)
    assert decoded == message
    assert decoded.bands["mu"] == (8.0, 13.0)
