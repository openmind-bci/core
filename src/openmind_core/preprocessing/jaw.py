"""Double jaw clench detection on the motor electrodes (the ``action`` gesture).

A jaw clench is **muscle** activity (electromyogram of the masseter and
temporalis muscles) that reaches every scalp electrode, not a brain signal: the
``action`` command it triggers makes OpenMind a hybrid EEG + EMG interface, and
is documented as such. The method and its thresholds come from the study in
``docs/action-gesture.md`` (public recordings, held-out validation).

Method, identical offline and online (the detector keeps its state between
blocks):

1. causal band-pass 30-110 Hz (the upper edge is lowered at low sampling
   rates) with notches at the power-line frequency and its harmonic: above the
   8-30 Hz band used for decoding;
2. power over 100 ms on each electrode, in dB above that electrode's own
   background (median of the previous 10 s, refreshed every quarter second);
3. score: the **median over the electrodes**, so a clench must raise the power
   on most of them at once (a bad contact on one electrode cannot trigger it);
4. a **clench** is a burst whose score exceeds ``onset_db``, peaks above
   ``min_peak_db`` and lasts at most ``max_burst_s`` (deliberate clenches are
   strong and short; swallowing is weaker and longer);
5. a **double clench** is two clench onsets ``min_gap_s`` to ``max_gap_s``
   apart; the detector then ignores clenches for ``refractory_s``.

**The gesture**: clench for about 0.2-0.4 s, release for at least 0.3 s, clench
again. A shorter release merges the two bursts into one (the 100 ms envelope and
the 150 ms merge gap smooth over it): shorter settings would split them, but
also split other artifacts into pairs (2.2 instead of 0.6 false double clenches
per hour on held-out public recordings).

A clench is only known when its burst ends, so a double clench is reported
about 0.15 s after the end of the second burst. While a burst is in progress,
:attr:`JawClenchDetector.active` is true: muscle activity also contaminates the
beta band, and the decoder should pause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class JawConfig:
    """Settings of :class:`JawClenchDetector` (defaults validated in ``docs/action-gesture.md``)."""

    low_hz: float = 30.0
    high_hz: float = 110.0
    line_hz: float = 50.0
    power_window_s: float = 0.1
    background_s: float = 10.0
    refresh_s: float = 0.25
    onset_db: float = 10.0
    min_peak_db: float = 20.0
    min_burst_s: float = 0.1
    max_burst_s: float = 1.0
    merge_gap_s: float = 0.15
    min_gap_s: float = 0.25
    max_gap_s: float = 1.2
    refractory_s: float = 1.0


class JawClenchDetector:
    """Detect jaw clenches and double clenches in multichannel EEG blocks.

    Args:
        srate: Sampling rate in Hz (at least 100 Hz).
        n_channels: Number of electrodes; the motor electrodes of montage C
            (``FC3, FC4, C3, Cz, C4, CP3, CP4, CPz``) in the reference setting.
        config: Thresholds.

    Raises:
        ValueError: If the sampling rate leaves no room for the muscle band.
    """

    def __init__(self, srate: float, n_channels: int, config: JawConfig | None = None) -> None:
        self.srate = srate
        self.n_channels = n_channels
        self.config = cfg = config or JawConfig()
        high = min(cfg.high_hz, 0.45 * srate)
        if not cfg.low_hz < high:
            raise ValueError(
                f"sampling rate {srate} Hz leaves no muscle band above {cfg.low_hz} Hz"
            )
        sections = [signal.butter(4, (cfg.low_hz, high), btype="bandpass", fs=srate, output="sos")]
        for line in (cfg.line_hz, 2 * cfg.line_hz):
            if line < 0.5 * srate:
                b, a = signal.iirnotch(line, 30.0, fs=srate)
                sections.append(signal.tf2sos(b, a))
        self._sos = np.vstack(sections)
        self._band_zi: FloatArray | None = None
        self._window = max(round(cfg.power_window_s * srate), 1)
        self._power_zi = np.zeros((self._window - 1, n_channels))
        self._step = max(round(cfg.refresh_s * srate), 1)
        self._history = np.zeros((max(round(cfg.background_s * srate), 1), n_channels))
        self._history_pos = 0
        self._history_count = 0
        self._background: FloatArray | None = None
        self._index = 0  # index of the next sample
        self._burst_start: int | None = None
        self._burst_peak = -math.inf
        self._last_above = -1
        self._last_active: int | None = None  # last sample with a burst in progress
        self._previous: int | None = None  # onset of an unpaired clench
        self._refractory_until = -1
        self.clenches: list[float] = []  # onset times (s) of every clench seen

    @property
    def active(self) -> bool:
        """Whether muscle activity is in progress (pause decoding meanwhile)."""
        return self._burst_start is not None

    @property
    def last_active(self) -> int | None:
        """Index of the last sample during which muscle activity was in progress.

        Counted from the first sample given to the detector; ``None`` until a
        burst starts. Use it to reject windows that hold a burst which has
        already ended (:attr:`active` only covers the current sample).
        """
        return self._last_active

    @property
    def ready(self) -> bool:
        """Whether a background is known (after 1 s of signal)."""
        return self._background is not None

    def update(self, block: npt.ArrayLike) -> list[float]:
        """Process the next samples ``(n_samples, n_channels)`` in µV.

        Returns:
            Times (s, counted from the first sample given to the detector) at
            which double clenches were recognized.
        """
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.n_channels:
            raise ValueError(f"expected shape (n, {self.n_channels}), got {data.shape}")
        n = data.shape[0]
        if n == 0:
            return []
        if self._band_zi is None:
            self._band_zi = signal.sosfilt_zi(self._sos)[:, :, None] * data[0][None, None, :]
        band, self._band_zi = signal.sosfilt(self._sos, data, axis=0, zi=self._band_zi)
        kernel = np.ones(self._window) / self._window
        power, self._power_zi = signal.lfilter(kernel, [1.0], band**2, axis=0, zi=self._power_zi)
        log_power = 10 * np.log10(np.maximum(power, 1e-12))

        scores = np.full(n, -math.inf)
        pos = 0
        while pos < n:
            index = self._index + pos
            if index % self._step == 0:
                self._refresh_background()
            stop = min(n, pos + self._step - index % self._step)
            segment = log_power[pos:stop]
            if self._background is not None:
                scores[pos:stop] = np.median(segment - self._background[None, :], axis=1)
            self._remember(segment)
            pos = stop

        detections: list[float] = []
        for offset, score in enumerate(scores):
            detections += self._step_burst(self._index + offset, float(score))
            if self._burst_start is not None:
                self._last_active = self._index + offset
        self._index += n
        return detections

    # -- internals -------------------------------------------------------------

    def _remember(self, segment: FloatArray) -> None:
        size = self._history.shape[0]
        for row in segment[-size:]:
            self._history[self._history_pos] = row
            self._history_pos = (self._history_pos + 1) % size
            self._history_count = min(self._history_count + 1, size)

    def _refresh_background(self) -> None:
        if self._history_count >= self.srate:
            self._background = np.median(self._history[: self._history_count], axis=0)

    def _step_burst(self, index: int, score: float) -> list[float]:
        cfg = self.config
        if score > cfg.onset_db:
            if self._burst_start is None:
                self._burst_start, self._burst_peak = index, score
            self._burst_peak = max(self._burst_peak, score)
            self._last_above = index
            return []
        if self._burst_start is None or index - self._last_above <= cfg.merge_gap_s * self.srate:
            return []
        start, peak = self._burst_start, self._burst_peak
        self._burst_start, self._burst_peak = None, -math.inf
        length = (self._last_above - start + 1) / self.srate
        if length < cfg.min_burst_s or length > cfg.max_burst_s or peak < cfg.min_peak_db:
            return []
        return self._on_clench(start, index)

    def _on_clench(self, onset: int, now: int) -> list[float]:
        cfg = self.config
        self.clenches.append(onset / self.srate)
        if onset < self._refractory_until:
            return []
        previous, self._previous = self._previous, onset
        if previous is None:
            return []
        gap = (onset - previous) / self.srate
        if cfg.min_gap_s <= gap <= cfg.max_gap_s:
            self._previous = None
            self._refractory_until = now + round(cfg.refractory_s * self.srate)
            return [now / self.srate]
        return []


def detect_double_clenches(
    data: npt.ArrayLike, srate: float, config: JawConfig | None = None
) -> list[float]:
    """Times (s) of double clenches in a whole recording ``(n_samples, n_channels)`` (µV)."""
    array = np.asarray(data, dtype=np.float64)
    return JawClenchDetector(srate, array.shape[1], config).update(array)
