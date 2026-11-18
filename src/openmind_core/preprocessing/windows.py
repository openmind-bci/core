"""Cutting continuous signals into epochs (offline) and sliding windows (online)."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

#: Length of a decoding window, in seconds.
WINDOW_S: Final = 2.0
#: Step between two online windows, in seconds.
STEP_S: Final = 0.25
#: Start of an offline epoch after the cue, in seconds.
EPOCH_OFFSET_S: Final = 0.5


def epochs_at(
    data: npt.ArrayLike,
    onsets: npt.ArrayLike,
    srate: float,
    offset_s: float = EPOCH_OFFSET_S,
    length_s: float = WINDOW_S,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Cut epochs ``[onset + offset, onset + offset + length)`` from a continuous signal.

    Args:
        data: Continuous signal ``(n_samples, n_channels)``.
        onsets: Event positions in samples (e.g. cue onsets).
        srate: Sampling rate in Hz.
        offset_s: Start of each epoch relative to its onset.
        length_s: Epoch length.

    Returns:
        ``(epochs, kept)``: epochs of shape ``(n_kept, n_channels, n_times)`` and a
        boolean mask over ``onsets`` telling which ones fit inside the signal.
    """
    array = np.asarray(data, dtype=np.float64)
    starts = np.round(np.asarray(onsets, dtype=np.float64) + offset_s * srate).astype(int)
    n_times = round(length_s * srate)
    kept = (starts >= 0) & (starts + n_times <= array.shape[0])
    index = starts[kept, None] + np.arange(n_times)[None, :]
    return np.transpose(array[index], (0, 2, 1)), kept


class SlidingWindow:
    """Ring buffer that yields the latest window every ``step`` samples (online).

    Args:
        n_channels: Number of channels.
        srate: Sampling rate in Hz.
        length_s: Window length.
        step_s: Step between two windows.
    """

    def __init__(
        self,
        n_channels: int,
        srate: float,
        length_s: float = WINDOW_S,
        step_s: float = STEP_S,
    ) -> None:
        self.n_times = round(length_s * srate)
        self.step = max(round(step_s * srate), 1)
        self._buffer = np.zeros((self.n_times, n_channels))
        self._filled = 0
        self._since_last = 0

    def push(self, block: npt.ArrayLike) -> list[npt.NDArray[np.float64]]:
        """Add samples ``(n, n_channels)``; return the windows completed, oldest first.

        Each window has shape ``(n_channels, n_times)``, like an offline epoch.
        """
        windows: list[npt.NDArray[np.float64]] = []
        for sample in np.asarray(block, dtype=np.float64):
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = sample
            self._filled = min(self._filled + 1, self.n_times)
            self._since_last += 1
            if self._filled == self.n_times and self._since_last >= self.step:
                self._since_last = 0
                windows.append(self._buffer.T.copy())
        return windows
