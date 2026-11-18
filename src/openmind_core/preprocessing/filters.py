"""Causal filters shared by offline and online processing.

The same :class:`StreamFilter` processes a whole recording offline and successive
blocks online: its state carries over between calls, so filtering a run in one
call or block by block gives the same samples. Zero-phase filtering
(``filtfilt``) is deliberately not used, because it cannot run in real time and
would make offline training data differ from what the decoder sees live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FilterConfig:
    """Band-pass and notch settings.

    Args:
        low_hz: Lower edge of the band-pass (8 Hz: mu band).
        high_hz: Upper edge of the band-pass (30 Hz: end of beta).
        order: Butterworth order of the band-pass.
        notch_hz: Power-line frequency to remove (50 Hz in Europe, 60 Hz in the
            Americas), or ``None`` to skip the notch.
        notch_quality: Quality factor of the notch (higher = narrower).
    """

    low_hz: float = 8.0
    high_hz: float = 30.0
    order: int = 4
    notch_hz: float | None = 50.0
    notch_quality: float = 30.0

    def design(self, srate: float) -> FloatArray:
        """Second-order sections of the full filter at ``srate``."""
        nyquist = srate / 2
        if not 0 < self.low_hz < self.high_hz < nyquist:
            raise ValueError(
                f"need 0 < low ({self.low_hz}) < high ({self.high_hz}) < Nyquist ({nyquist})"
            )
        sections = [
            signal.butter(
                self.order, (self.low_hz, self.high_hz), btype="bandpass", fs=srate, output="sos"
            )
        ]
        if self.notch_hz is not None and self.notch_hz < nyquist:
            b, a = signal.iirnotch(self.notch_hz, self.notch_quality, fs=srate)
            sections.append(signal.tf2sos(b, a))
        return np.vstack(sections)


class StreamFilter:
    """Stateful causal filter for ``(n_samples, n_channels)`` blocks.

    Args:
        srate: Sampling rate in Hz.
        n_channels: Number of channels.
        config: Filter settings.
    """

    def __init__(self, srate: float, n_channels: int, config: FilterConfig | None = None) -> None:
        self.srate = srate
        self.n_channels = n_channels
        self.config = config or FilterConfig()
        self._sos = self.config.design(srate)
        self.reset()

    def reset(self) -> None:
        """Forget the filter state (start of a new recording)."""
        self._zi = np.zeros((self._sos.shape[0], 2, self.n_channels))
        self._primed = False

    def __call__(self, block: npt.ArrayLike) -> FloatArray:
        """Filter the next block of samples, continuing from the previous call."""
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.n_channels:
            raise ValueError(f"expected shape (n, {self.n_channels}), got {data.shape}")
        if data.shape[0] == 0:
            return data.copy()
        if not self._primed:
            # Start from the steady state of the first sample to avoid a large
            # transient from DC offsets.
            self._zi = signal.sosfilt_zi(self._sos)[:, :, None] * data[0][None, None, :]
            self._primed = True
        out, self._zi = signal.sosfilt(self._sos, data, axis=0, zi=self._zi)
        return np.asarray(out, dtype=np.float64)


def filter_continuous(
    data: npt.ArrayLike, srate: float, config: FilterConfig | None = None
) -> FloatArray:
    """Filter a whole continuous recording ``(n_samples, n_channels)`` causally."""
    array = np.asarray(data, dtype=np.float64)
    return StreamFilter(srate, array.shape[1], config)(array)
