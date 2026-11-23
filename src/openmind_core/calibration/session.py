"""Live calibration epochs: filtered ring buffer + trials rebuilt from markers.

During a calibration session the sandbox scenes publish markers (trial start,
cue, trial end, block events) while the decoder receives the EEG. The recorder
cuts, for every completed trial, the same epoch the offline loader cuts
(``[cue + 0.5 s, cue + 2.5 s]`` of the causally filtered signal, see
:func:`openmind_core.preprocessing.epochs_at`), so the epochs collected live can
train a model exactly like the public datasets.

Every completed trial is kept, successful or not: the label is the cue, not the
decoder's decision. A trial interrupted before its end is dropped by
:class:`~openmind_core.io.TrialCollector`. A completed trial whose epoch is no
longer in the ring buffer (or has a gap) is counted as *missed*.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from openmind_core.io import Trial, TrialCollector
from openmind_core.models.registry import ModelCard
from openmind_core.online.decoder import _normalize
from openmind_core.preprocessing import EPOCH_OFFSET_S, WINDOW_S, FilterConfig, StreamFilter

FloatArray = npt.NDArray[np.float64]

_BLOCK_START = re.compile(r"^block(?:_(\d+))?_start$")
_BLOCK_END = re.compile(r"^block(?:_(\d+))?_end$")


class EpochBuffer:
    """Causally filtered ring buffer of the picked channels, cut by timestamp.

    Args:
        channels: Labels of the channels fed to :meth:`push` (their order).
        srate: Sampling rate in Hz.
        filters: Band-pass and notch of the training epochs (``None``: defaults).
        keep_s: Seconds of signal kept.
    """

    def __init__(
        self,
        channels: Sequence[str],
        srate: float,
        filters: FilterConfig | None = None,
        keep_s: float = 30.0,
    ) -> None:
        if keep_s <= 0:
            raise ValueError("keep_s must be > 0")
        self.channels = tuple(channels)
        self.srate = srate
        self.keep_s = keep_s
        self._filter = StreamFilter(srate, len(self.channels), filters)
        self._data: FloatArray = np.zeros((0, len(self.channels)))
        self._times: FloatArray = np.zeros(0)
        self._capacity = max(1, round(keep_s * srate))

    @property
    def start_time(self) -> float | None:
        """Timestamp of the oldest sample kept."""
        return float(self._times[0]) if len(self._times) else None

    @property
    def end_time(self) -> float | None:
        """Timestamp of the newest sample."""
        return float(self._times[-1]) if len(self._times) else None

    def push(self, block: npt.ArrayLike, timestamps: npt.ArrayLike | None = None) -> None:
        """Filter and store ``(n_samples, n_channels)`` samples in µV.

        Without ``timestamps`` the samples follow the previous ones at the
        sampling rate (starting at 0).
        """
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(self.channels):
            raise ValueError(f"expected shape (n, {len(self.channels)}), got {data.shape}")
        n = data.shape[0]
        if timestamps is None:
            first = 0.0 if not len(self._times) else float(self._times[-1]) + 1 / self.srate
            times = first + np.arange(n) / self.srate
        else:
            times = np.asarray(timestamps, dtype=np.float64)
            if times.shape != (n,):
                raise ValueError(f"expected {n} timestamps, got shape {times.shape}")
        if n == 0:
            return
        self._data = np.vstack([self._data, self._filter(data)])[-self._capacity :]
        self._times = np.concatenate([self._times, times])[-self._capacity :]

    def window(self, start: float, length_s: float = WINDOW_S) -> FloatArray | None:
        """Samples ``[start, start + length_s)`` as ``(n_channels, n_times)``, or ``None``."""
        return self._cut(start, length_s)[0]

    def epoch_at(self, cue_time: float) -> FloatArray | None:
        """Epoch ``[cue + 0.5 s, cue + 2.5 s]`` of a cue, like the offline loader.

        Returns ``None`` when it is not available (yet): see :meth:`status`.
        """
        return self._cut(cue_time + EPOCH_OFFSET_S, WINDOW_S)[0]

    def status(self, cue_time: float) -> str:
        """``"ok"``, ``"pending"`` (data not received yet) or ``"missed"`` (gone or gap)."""
        return self._cut(cue_time + EPOCH_OFFSET_S, WINDOW_S)[1]

    def _cut(self, start: float, length_s: float) -> tuple[FloatArray | None, str]:
        times, n = self._times, round(length_s * self.srate)
        half = 0.5 / self.srate
        if not len(times):
            return None, "pending"
        if start < times[0] - half:
            return None, "missed"
        i = int(np.searchsorted(times, start - half))
        if i + n > len(times):
            return None, "pending"
        if (
            abs(times[i] - start) > half + 1e-9
            or abs(times[i + n - 1] - times[i] - (n - 1) / self.srate) > half
        ):
            return None, "missed"  # gap in the stream
        return np.ascontiguousarray(self._data[i : i + n].T), "ok"


@dataclass(frozen=True)
class CalibrationTrial:
    """A labelled epoch recorded during the session.

    Attributes:
        epoch: Filtered signal ``(n_channels, n_times)`` in µV, not aligned.
        label: Cue label.
        trial: Trial number from the markers.
        block: Block number (from the ``block_N_start`` events; 0 before any).
        cue_time: LSL timestamp of the cue.
    """

    epoch: FloatArray
    label: str
    trial: int
    block: int
    cue_time: float


class CalibrationRecorder:
    """Collect labelled epochs from the EEG blocks and the scene markers.

    Args:
        card: Model card giving the channels, sampling rate and filters the
            epochs are prepared with (the decoder's).
        stream_labels: Channel labels of the stream, in block order.
        srate: Sampling rate of the stream (the card's).
        source: ``source`` of the markers to keep (``"sandbox"``).
        keep_s: Seconds of filtered signal kept.

    Attributes:
        trials: Recorded trials, in order.
        missed: Completed trials whose epoch was not available.
        interrupted: Trials started and never completed (from the collector).
    """

    def __init__(
        self,
        card: ModelCard,
        stream_labels: Sequence[str],
        srate: float,
        source: str = "sandbox",
        keep_s: float = 30.0,
    ) -> None:
        keys: dict[str, int] = {}
        for index, label in enumerate(stream_labels):
            keys.setdefault(_normalize(label), index)
        missing = [c for c in card.channels if _normalize(c) not in keys]
        if missing:
            raise ValueError(
                f"model {card.name!r} needs channels {missing} missing from the stream"
            )
        self.card = card
        self._picked = [keys[_normalize(c)] for c in card.channels]
        self._n_stream = len(stream_labels)
        filters = FilterConfig(**card.filters) if card.filters else None
        self.buffer = EpochBuffer(card.channels, srate, filters, keep_s)
        self.collector = TrialCollector(source)
        self.trials: list[CalibrationTrial] = []
        self.missed = 0
        self.block_ends = 0
        self._pending: list[Trial] = []
        self._events_seen = 0
        self._block_starts: list[tuple[float, int]] = []

    @property
    def interrupted(self) -> int:
        return self.collector.interrupted

    def push(
        self, block: npt.ArrayLike, timestamps: npt.ArrayLike | None = None
    ) -> list[CalibrationTrial]:
        """Feed a block ``(n_samples, n_stream_channels)``; return the trials it completes."""
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self._n_stream:
            raise ValueError(f"expected shape (n, {self._n_stream}), got {data.shape}")
        self.buffer.push(data[:, self._picked], timestamps)
        return self._resolve()

    def add_markers(self, markers: Iterable[Any]) -> list[CalibrationTrial]:
        """Feed markers (``(timestamp, payload)`` pairs or ``Marker`` objects)."""
        self._pending += self.collector.add(markers)
        self._read_events()
        return self._resolve()

    def finish(self) -> None:
        """End of the session: unresolved trials are missed, an open trial interrupted."""
        self.collector.finish()
        self.missed += len(self._pending)
        self._pending = []

    def epochs(self) -> FloatArray:
        """All epochs ``(n_trials, n_channels, n_times)``."""
        if not self.trials:
            n = round(WINDOW_S * self.buffer.srate)
            return np.zeros((0, len(self.card.channels), n))
        return np.stack([t.epoch for t in self.trials])

    def labels(self) -> list[str]:
        """Cue label of each epoch."""
        return [t.label for t in self.trials]

    def counts_per_class(self) -> dict[str, int]:
        """Number of recorded epochs per label."""
        return dict(Counter(self.labels()))

    # -- internals -------------------------------------------------------------

    def _read_events(self) -> None:
        events = self.collector.events
        for timestamp, label in events[self._events_seen :]:
            if (m := _BLOCK_START.match(label)) is not None:
                number = int(m.group(1)) if m.group(1) else len(self._block_starts) + 1
                self._block_starts.append((timestamp, number))
            elif _BLOCK_END.match(label):
                self.block_ends += 1
        self._events_seen = len(events)

    def _block_of(self, trial: Trial) -> int:
        number = 0
        for timestamp, value in self._block_starts:
            if timestamp <= trial.start:
                number = value
        return number

    def _resolve(self) -> list[CalibrationTrial]:
        recorded: list[CalibrationTrial] = []
        waiting: list[Trial] = []
        for trial in self._pending:
            status = self.buffer.status(trial.cue_time)
            if status == "ok":
                epoch = self.buffer.epoch_at(trial.cue_time)
                assert epoch is not None
                recorded.append(
                    CalibrationTrial(
                        epoch, trial.label, trial.trial, self._block_of(trial), trial.cue_time
                    )
                )
            elif status == "pending":
                waiting.append(trial)
            else:
                self.missed += 1
        self._pending = waiting
        self.trials += recorded
        return recorded
