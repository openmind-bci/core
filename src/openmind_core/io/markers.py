"""Calibration trials rebuilt from the markers of the calibration scenes.

A calibration scene (OpenMind Sandbox) publishes, on its own
``openmind-markers`` stream, for each trial::

    trial_start (trial n) → cue (label, trial n) → trial_end (trial n)

plus ``event`` markers (``session_start``, ``block_start``, ``block_end``,
``session_end``...). The consumer reads every markers stream, so it also
receives markers of other producers, such as the cues of a dataset replayed by
Plug: :class:`TrialCollector` keeps only the markers of one ``source``.

A trial is complete when its ``trial_end`` arrives after its ``cue``. A trial
started and never ended (another ``trial_start``, or the end of the session) was
interrupted and is discarded, as the stream specification recommends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Trial:
    """A complete calibration trial.

    Attributes:
        trial: Trial number (from 1).
        label: Expected command (``left``, ``right``, ``forward``, ``rest``...).
        cue_time: LSL timestamp of the cue: epochs are cut from it.
        start: LSL timestamp of ``trial_start``.
        end: LSL timestamp of ``trial_end``.
        scene: Scene that produced the trial, if given.
    """

    trial: int
    label: str
    cue_time: float
    start: float
    end: float
    scene: str | None = None


@dataclass
class _Open:
    trial: int
    start: float
    scene: str | None
    label: str | None = None
    cue_time: float | None = None


class TrialCollector:
    """Rebuild complete trials from markers, in time order.

    Args:
        source: Keep only markers whose ``source`` field has this value
            (``"sandbox"`` for the calibration scenes); ``None`` keeps all.

    Attributes:
        trials: Every complete trial, in order.
        interrupted: Number of trials started and never completed.
        events: ``(timestamp, label)`` of the ``event`` markers kept.
    """

    def __init__(self, source: str | None = "sandbox") -> None:
        self.source = source
        self.trials: list[Trial] = []
        self.interrupted = 0
        self.events: list[tuple[float, str]] = []
        self._open: _Open | None = None

    def add(self, markers: Iterable[Any]) -> list[Trial]:
        """Feed markers and return the trials they complete.

        Each marker is an object with ``timestamp`` and ``payload`` attributes
        (such as ``openmind_plug.sources.Marker``) or a ``(timestamp, payload)``
        pair. Markers are processed in timestamp order.
        """
        completed: list[Trial] = []
        for timestamp, payload in sorted(map(_unpack, markers), key=lambda item: item[0]):
            if self.source is not None and payload.get("source") != self.source:
                continue
            trial = self._step(timestamp, payload)
            if trial is not None:
                completed.append(trial)
        self.trials += completed
        return completed

    def finish(self) -> None:
        """End of the session: an open trial is interrupted."""
        if self._open is not None:
            self.interrupted += 1
            self._open = None

    def _step(self, timestamp: float, payload: Mapping[str, Any]) -> Trial | None:
        kind, number = payload.get("type"), payload.get("trial")
        if kind == "trial_start" and isinstance(number, int):
            self.finish()  # an unfinished previous trial was interrupted
            self._open = _Open(number, timestamp, payload.get("scene"))
        elif kind == "cue" and self._open is not None and number == self._open.trial:
            label = payload.get("label")
            if isinstance(label, str) and label:
                self._open.label, self._open.cue_time = label, timestamp
        elif kind == "trial_end" and self._open is not None and number == self._open.trial:
            current, self._open = self._open, None
            if current.label is None or current.cue_time is None:
                self.interrupted += 1  # ended without a cue
                return None
            return Trial(
                trial=current.trial,
                label=current.label,
                cue_time=current.cue_time,
                start=current.start,
                end=timestamp,
                scene=current.scene,
            )
        elif kind == "event":
            label = payload.get("label")
            if isinstance(label, str):
                self.events.append((timestamp, label))
                if label == "session_end":
                    self.finish()
        return None


def _unpack(marker: Any) -> tuple[float, Mapping[str, Any]]:
    if isinstance(marker, tuple):
        timestamp, payload = marker
    else:
        timestamp, payload = marker.timestamp, marker.payload
    return float(timestamp), payload
