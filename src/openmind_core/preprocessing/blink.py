"""Double-blink detection on a frontal electrode (Fp1).

A blink is an **eye** signal (electro-oculogram picked up by Fp1), not a brain
signal: the ``action`` command it triggers makes OpenMind a hybrid EEG + EOG
interface, and is documented as such.

Method, identical offline and online (the detector keeps its state between
blocks):

1. band-pass 0.1-8 Hz (causal): removes drifts and fast EEG, keeps the slow
   blink wave (a narrower band such as 0.5-10 Hz flattens the 0.3 s blink peak
   and misses blinks of 100 µV);
2. a blink is a positive peak above an adaptive threshold: the larger of
   ``min_amplitude_uv`` and ``k`` times a robust estimate of the background
   (median absolute deviation of the last ``background_s`` seconds, refreshed
   every quarter second of signal, so offline and block-by-block processing give
   the same result);
3. a **double blink** is two blinks whose peaks are ``min_gap_s`` to
   ``max_gap_s`` apart; after a detection the detector ignores blinks for
   ``refractory_s`` so a triple blink gives one event.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass(frozen=True)
class BlinkConfig:
    """Thresholds of :class:`DoubleBlinkDetector`."""

    min_amplitude_uv: float = 60.0
    k: float = 5.0
    min_gap_s: float = 0.15
    max_gap_s: float = 0.8
    refractory_s: float = 1.0
    min_blink_s: float = 0.1
    background_s: float = 10.0


class DoubleBlinkDetector:
    """Detect blinks and double blinks in a stream of Fp1 samples.

    Args:
        srate: Sampling rate in Hz.
        config: Thresholds.
    """

    def __init__(self, srate: float, config: BlinkConfig | None = None) -> None:
        self.srate = srate
        self.config = config or BlinkConfig()
        self._sos = signal.butter(2, (0.1, 8.0), btype="bandpass", fs=srate, output="sos")
        self._zi: npt.NDArray[np.float64] | None = None
        background = round(self.config.background_s * srate)
        self._history: deque[float] = deque(maxlen=background)
        self._mad = 0.0
        self._refresh_every = max(round(srate / 4), 1)
        self._index = 0  # index of the next sample
        self._above_since: int | None = None
        self._peak = (0, -np.inf)  # (index, value) of the current blink
        self._last_blink: int | None = None
        self._refractory_until = -1
        self.blinks: list[float] = []  # times (s) of every blink seen

    @property
    def threshold_uv(self) -> float:
        """Current detection threshold in µV."""
        return max(self.config.min_amplitude_uv, self.config.k * 1.4826 * self._mad)

    def update(self, fp1: npt.ArrayLike) -> list[float]:
        """Process the next Fp1 samples (µV); return times (s) of double blinks.

        Times are counted from the first sample given to the detector, at the
        second blink of each pair.
        """
        data = np.asarray(fp1, dtype=np.float64).ravel()
        if data.size == 0:
            return []
        if self._zi is None:
            self._zi = signal.sosfilt_zi(self._sos) * data[0]
        filtered, self._zi = signal.sosfilt(self._sos, data, zi=self._zi)

        detections: list[float] = []
        min_len = round(self.config.min_blink_s * self.srate / 2)
        for value in filtered:
            threshold = self.threshold_uv
            if value > threshold:
                if self._above_since is None:
                    self._above_since = self._index
                    self._peak = (self._index, value)
                elif value > self._peak[1]:
                    self._peak = (self._index, value)
            elif self._above_since is not None:
                if self._index - self._above_since >= min_len:
                    detections += self._on_blink(self._peak[0])
                self._above_since = None
            else:
                self._history.append(float(value))
            self._index += 1
            if self._index % self._refresh_every == 0 and len(self._history) >= self.srate:
                history = np.fromiter(self._history, dtype=np.float64)
                self._mad = float(np.median(np.abs(history - np.median(history))))
        return detections

    def _on_blink(self, peak: int) -> list[float]:
        cfg = self.config
        self.blinks.append(peak / self.srate)
        if peak < self._refractory_until:
            return []
        previous, self._last_blink = self._last_blink, peak
        if previous is None:
            return []
        gap = (peak - previous) / self.srate
        if cfg.min_gap_s <= gap <= cfg.max_gap_s:
            self._refractory_until = peak + round(cfg.refractory_s * self.srate)
            self._last_blink = None
            return [peak / self.srate]
        return []


def detect_double_blinks(
    fp1: npt.ArrayLike, srate: float, config: BlinkConfig | None = None
) -> list[float]:
    """Times (s) of double blinks in a whole Fp1 recording (µV)."""
    return DoubleBlinkDetector(srate, config).update(fp1)
