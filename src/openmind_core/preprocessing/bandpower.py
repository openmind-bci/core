"""Online band power per electrode, for the 3D brain view (sensor level).

For each electrode, the power spectral density (Welch, 1 s segments with 50 %
overlap, Hann window, linear detrending) of the latest 2 s window is averaged
over the bins of each band, in µV²/Hz. A new frame is produced every 100 ms
(about 10 Hz). The input is raw EEG in µV: the DC offset and the power line
(50/60 Hz) lie outside the bands, so no filtering is needed.

The power is smoothed with an exponential moving average and compared with a
resting **baseline** (median of the frames recorded while the user rests):
``relative = (power - baseline) / baseline``, negative when the rhythm
desynchronizes. A channel that is absent, flat, saturated or non-finite is
flagged ``ok = False`` and carries no power.

This is a **sensor-level** measure (what each electrode records), not a source
localization: the 3D brain only colors the cortex under each electrode.

The estimator uses the same :class:`SlidingWindow` as the decoder, so frames are
identical whether the signal is pushed at once or block by block.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal

from openmind_core.preprocessing.windows import SlidingWindow
from openmind_protocol import BandPower, ChannelPower
from openmind_protocol.messages import now

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Frequency bands in Hz (the example of the ``bandpower`` message).
BANDS: Final[dict[str, tuple[float, float]]] = {
    "alpha": (8.0, 12.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
#: A channel whose standard deviation is below this value (µV) is flat.
FLAT_UV: Final = 0.5
#: A channel whose peak absolute value exceeds this value (µV) is saturated.
SATURATION_UV: Final = 2000.0
#: A channel is also saturated when more than this share of its samples equals
#: its maximum or its minimum (clipping).
CLIP_SHARE: Final = 0.05


def band_power(
    window: npt.ArrayLike,
    srate: float,
    bands: dict[str, tuple[float, float]] = BANDS,
) -> FloatArray:
    """Mean power spectral density of each band, in µV²/Hz.

    Args:
        window: Raw signal ``(n_channels, n_times)`` in µV.
        srate: Sampling rate in Hz.
        bands: ``{name: (low_hz, high_hz)}``.

    Returns:
        Array ``(n_channels, n_bands)``, bands in the order of ``bands``. A band
        without any frequency bin gets NaN.
    """
    data = np.asarray(window, dtype=np.float64)
    nperseg = min(data.shape[1], round(srate))
    freqs, psd = signal.welch(
        data, fs=srate, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear", axis=-1
    )
    out = np.full((data.shape[0], len(bands)), np.nan)
    for index, (low, high) in enumerate(bands.values()):
        inside = (freqs >= low) & (freqs <= high)
        if inside.any():
            out[:, index] = psd[:, inside].mean(axis=1)
    return out


def channel_ok(window: npt.ArrayLike) -> BoolArray:
    """Tell which channels of ``window`` ``(n_channels, n_times)`` are usable.

    A channel is not ok when it has a non-finite value, is flat (standard
    deviation below :data:`FLAT_UV`) or saturated (peak absolute value above
    :data:`SATURATION_UV`, or more than :data:`CLIP_SHARE` of its samples equal
    to its maximum or minimum).
    """
    data = np.asarray(window, dtype=np.float64)
    finite = np.isfinite(data).all(axis=1)
    ok = finite.copy()
    for index in np.flatnonzero(finite):
        row = data[index]
        flat = row.std() < FLAT_UV
        peak = np.abs(row).max() > SATURATION_UV
        clipped = max(np.mean(row == row.max()), np.mean(row == row.min())) > CLIP_SHARE
        ok[index] = not (flat or peak or clipped)
    return ok


@dataclass
class BandPowerConfig:
    """Settings of :class:`BandPowerEstimator`.

    Attributes:
        window_s: Window length (the decoder window).
        step_s: Step between frames (0.1 s is about 10 Hz).
        smoothing: Weight of the new frame in the exponential moving average
            (1 means no smoothing).
        bands: ``{name: (low_hz, high_hz)}``.
    """

    window_s: float = 2.0
    step_s: float = 0.1
    smoothing: float = 0.5
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(BANDS))


@dataclass(frozen=True)
class BandPowerFrame:
    """Band power of every channel at one instant.

    Attributes:
        timestamp: Time of the last sample of the window, if timestamps were given.
        labels: Channel labels.
        bands: ``{name: (low_hz, high_hz)}``.
        power: ``(n_channels, n_bands)`` in µV²/Hz, NaN where the channel is not ok.
        relative: ``(n_channels, n_bands)`` change from the baseline, or ``None``
            without baseline (NaN where the baseline is unusable).
        ok: Per-channel quality flag.
    """

    timestamp: float | None
    labels: tuple[str, ...]
    bands: dict[str, tuple[float, float]]
    power: FloatArray
    relative: FloatArray | None
    ok: BoolArray

    def to_message(
        self,
        user: str | None,
        srate: float,
        expected_labels: Sequence[str] | None = None,
    ) -> BandPower:
        """Build the ``bandpower`` message.

        Args:
            user: Pseudonym of the user.
            srate: Sampling rate of the stream in Hz.
            expected_labels: Electrodes the viewer expects; those absent from
                the stream (e.g. ``Fp1``) are appended with ``ok = False`` and
                no values.
        """
        names = list(self.bands)

        def value(array: FloatArray | None, row: int, name: str) -> float | None:
            if array is None or name not in names or not self.ok[row]:
                return None
            number = float(array[row, names.index(name)])
            return number if np.isfinite(number) else None

        channels = [
            ChannelPower(
                label=label,
                alpha=value(self.power, row, "alpha"),
                mu=value(self.power, row, "mu"),
                beta=value(self.power, row, "beta"),
                rel_mu=value(self.relative, row, "mu"),
                rel_beta=value(self.relative, row, "beta"),
                ok=bool(self.ok[row]),
            )
            for row, label in enumerate(self.labels)
        ]
        present = set(self.labels)
        channels.extend(
            ChannelPower(
                label=label,
                alpha=None,
                mu=None,
                beta=None,
                rel_mu=None,
                rel_beta=None,
                ok=False,
            )
            for label in (expected_labels or ())
            if label not in present
        )
        return BandPower(
            ts=now() if self.timestamp is None else self.timestamp,
            user=user,
            srate_hz=srate,
            bands={name: (low, high) for name, (low, high) in self.bands.items()},
            channels=channels,
        )


class BandPowerEstimator:
    """Compute :class:`BandPowerFrame` objects from EEG blocks.

    Args:
        labels: Channel labels, in the column order of the blocks.
        srate: Sampling rate in Hz.
        config: Settings.
    """

    def __init__(
        self,
        labels: Sequence[str],
        srate: float,
        config: BandPowerConfig | None = None,
    ) -> None:
        self.labels = tuple(labels)
        self.srate = srate
        self.config = config or BandPowerConfig()
        n_bands = len(self.config.bands)
        self._window = SlidingWindow(
            len(self.labels), srate, self.config.window_s, self.config.step_s
        )
        self._smoothed: FloatArray = np.full((len(self.labels), n_bands), np.nan)
        self._baseline: FloatArray | None = None
        self._recording: list[FloatArray] | None = None
        self._n_samples = 0
        self._n_windows = 0

    @property
    def baseline(self) -> FloatArray | None:
        """Resting baseline ``(n_channels, n_bands)``, or ``None``."""
        return self._baseline

    def set_baseline(self, power: npt.ArrayLike) -> None:
        """Set the baseline ``(n_channels, n_bands)`` (e.g. loaded from a profile)."""
        array = np.array(power, dtype=np.float64)
        expected = (len(self.labels), len(self.config.bands))
        if array.shape != expected:
            raise ValueError(f"baseline must have shape {expected}, got {array.shape}")
        self._baseline = array

    def start_baseline(self) -> None:
        """Start recording frames; :meth:`stop_baseline` turns them into the baseline."""
        self._recording = []

    def stop_baseline(self) -> FloatArray:
        """Stop recording and set the baseline to the median power of the frames.

        Returns:
            The baseline ``(n_channels, n_bands)``; NaN for a channel never ok.

        Raises:
            RuntimeError: If recording was not started or no frame was produced.
        """
        if self._recording is None:
            raise RuntimeError("start_baseline() was not called")
        frames, self._recording = self._recording, None
        if not frames:
            raise RuntimeError("no frame was produced during the baseline")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baseline = np.nanmedian(np.stack(frames), axis=0)
        self._baseline = baseline
        return baseline

    def push(
        self,
        block: npt.ArrayLike,
        timestamps: npt.ArrayLike | None = None,
    ) -> list[BandPowerFrame]:
        """Add samples ``(n_samples, n_channels)``; return the frames completed.

        Args:
            block: Raw EEG in µV.
            timestamps: Optional time of each sample; a frame gets the time of
                the last sample of its window.
        """
        data = np.asarray(block, dtype=np.float64)
        times = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        before = self._n_samples
        self._n_samples += data.shape[0]
        frames: list[BandPowerFrame] = []
        for window in self._window.push(data):
            self._n_windows += 1
            end = self._window.n_times + (self._n_windows - 1) * self._window.step
            stamp = None if times is None else float(times[end - 1 - before])
            frames.append(self._frame(window, stamp))
        return frames

    def _frame(self, window: FloatArray, timestamp: float | None) -> BandPowerFrame:
        ok = channel_ok(window)
        power = np.full(self._smoothed.shape, np.nan)
        if ok.any():
            power[ok] = band_power(window[ok], self.srate, self.config.bands)
        weight = self.config.smoothing
        previous = self._smoothed
        fresh = ok[:, None] & np.isnan(previous)
        blended = weight * power + (1.0 - weight) * previous
        smoothed = np.where(ok[:, None], np.where(fresh, power, blended), np.nan)
        self._smoothed = smoothed
        if self._recording is not None:
            self._recording.append(smoothed.copy())
        return BandPowerFrame(
            timestamp=timestamp,
            labels=self.labels,
            bands=dict(self.config.bands),
            power=smoothed.copy(),
            relative=self._relative(smoothed),
            ok=ok,
        )

    def _relative(self, power: FloatArray) -> FloatArray | None:
        if self._baseline is None:
            return None
        base = self._baseline
        usable = np.isfinite(base) & (base > 0)
        safe = np.where(usable, base, 1.0)
        return np.where(usable, (power - safe) / safe, np.nan)
