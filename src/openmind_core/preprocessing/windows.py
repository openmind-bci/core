"""Cutting continuous signals into epochs (offline) and sliding windows (online)."""

from __future__ import annotations

import math
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


def window_ends(
    n_samples: int, srate: float, length_s: float = WINDOW_S, step_s: float = STEP_S
) -> npt.NDArray[np.int64]:
    """Ends (exclusive sample indices) of the online windows in ``n_samples`` samples.

    The windows of :class:`SlidingWindow`, for offline use. A step that is not a
    whole number of samples (0.25 s at 250 Hz is 62.5) alternates between the two
    nearest whole steps, so the windows keep the exact step on average.
    """
    n_times = round(length_s * srate)
    step = max(step_s * srate, 1.0)
    ends: list[int] = []
    index = 0
    while (end := _window_end(index, n_times, step)) <= n_samples:
        ends.append(end)
        index += 1
    return np.asarray(ends, dtype=np.int64)


def _window_end(index: int, n_times: int, step: float) -> int:
    """Number of samples seen when window ``index`` (from 0) is complete."""
    return n_times + math.floor(index * step + 0.5)


class SlidingWindow:
    """Ring buffer that yields the latest window every ``step`` samples (online).

    Args:
        n_channels: Number of channels.
        srate: Sampling rate in Hz.
        length_s: Window length.
        step_s: Step between two windows. When it is not a whole number of
            samples, successive steps alternate between the two nearest whole
            numbers so that windows keep the exact step on average.

    Attributes:
        n_times: Samples per window.
        step: Exact step in samples (may be fractional, e.g. 62.5).
    """

    def __init__(
        self,
        n_channels: int,
        srate: float,
        length_s: float = WINDOW_S,
        step_s: float = STEP_S,
    ) -> None:
        self.n_times = round(length_s * srate)
        self.step = max(step_s * srate, 1.0)
        self._buffer = np.zeros((self.n_times, n_channels))
        self._count = 0  # samples pushed so far
        self._n_windows = 0  # windows yielded so far
        self._next_end = self.n_times

    @property
    def until_next(self) -> int:
        """Samples still needed to complete the next window."""
        return self._next_end - self._count

    def end_of(self, index: int) -> int:
        """Number of samples pushed when window ``index`` (from 0) is complete."""
        return _window_end(index, self.n_times, self.step)

    def push(self, block: npt.ArrayLike) -> list[npt.NDArray[np.float64]]:
        """Add samples ``(n, n_channels)``; return the windows completed, oldest first.

        Each window has shape ``(n_channels, n_times)``, like an offline epoch.
        """
        windows: list[npt.NDArray[np.float64]] = []
        for sample in np.asarray(block, dtype=np.float64):
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = sample
            self._count += 1
            if self._count == self._next_end:
                windows.append(self._buffer.T.copy())
                self._n_windows += 1
                self._next_end = self.end_of(self._n_windows)
        return windows
