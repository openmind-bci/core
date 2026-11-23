"""Resume mode: a returning user realigns in about a minute.

A user who already has a profile does not repeat the whole calibration. The
session is short: the user rests, the reference covariance is estimated again
(electrode placement and impedances changed since last time), then a few cued
trials check that the model still works with it.

Expected markers (sandbox scene, source ``sandbox``)::

    event session_start            start of the rest period
    event block_1_start            end of the rest period, start of control cues
      trial_start / cue(label) / trial_end     (control_cues times)
    event block_1_end / session_end

* **Rest**: samples from ``session_start`` (or the first sample) to the first
  ``block_N_start`` are the unlabeled rest; the first 1 s is skipped (filter
  settling) and 2 s windows (step 1 s) with a clean amplitude are used. At
  least ``rest_s`` seconds are required, else :class:`ValueError` is raised
  when the reference is computed.
* **Control cues**: the first ``control_cues`` completed trials after the rest
  are scored (:class:`~openmind_core.scoring.ProgressTracker`) with the
  decisions given to :meth:`ResumeProtocol.add_decisions`, which come from a
  decoder running with the new reference.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from openmind_core.alignment import reference_covariance
from openmind_core.calibration.session import EpochBuffer
from openmind_core.io import Trial, TrialCollector
from openmind_core.online.evidence import Decision
from openmind_core.preprocessing import WINDOW_S, FilterConfig
from openmind_core.preprocessing.artifacts import REJECT_PTP_UV, amplitude_mask
from openmind_core.scoring import BlockScore, ProgressTracker, ScoreConfig

FloatArray = npt.NDArray[np.float64]

_BLOCK_START = re.compile(r"^block(?:_\d+)?_start$")
_SETTLE_S = 1.0
_STEP_S = 1.0


@dataclass(frozen=True)
class ResumeResult:
    """Outcome of the resume protocol.

    Attributes:
        reference: New reference covariance ``(n_channels, n_channels)``.
        accuracy: Mean per-command accuracy of the control cues.
        passed: ``accuracy >= min_accuracy``; otherwise a full calibration
            block is advisable.
        block: Detailed score of the control cues.
        rest_windows: Clean rest windows the reference is estimated on.
    """

    reference: FloatArray
    accuracy: float
    passed: bool
    block: BlockScore
    rest_windows: int


class ResumeProtocol:
    """Rest → realignment → control cues, driven by blocks and markers.

    Args:
        channels: Labels of the channels fed to :meth:`push` (the model's).
        srate: Sampling rate in Hz.
        classes: Commands of the control cues.
        alignment: Reference mean, ``"riemann"`` or ``"euclid"`` (the model's).
        filters: Filters of the model (``None``: defaults).
        rest_s: Minimum seconds of rest (30 to 60 s recommended).
        control_cues: Number of cued trials to score.
        min_accuracy: Accuracy needed to pass.
        reject_uv: Peak-to-peak amplitude above which a rest window is dropped.
        source: ``source`` of the markers to keep.
    """

    def __init__(
        self,
        channels: Sequence[str],
        srate: float,
        classes: Sequence[str],
        alignment: str = "riemann",
        filters: FilterConfig | None = None,
        rest_s: float = 45.0,
        control_cues: int = 5,
        min_accuracy: float = 0.6,
        reject_uv: float = REJECT_PTP_UV,
        source: str = "sandbox",
    ) -> None:
        if rest_s < 2 * WINDOW_S or control_cues < 1:
            raise ValueError("need rest_s >= 4 and control_cues >= 1")
        self.rest_s = rest_s
        self.control_cues = control_cues
        self.min_accuracy = min_accuracy
        self.alignment = alignment
        self.reject_uv = reject_uv
        self.buffer = EpochBuffer(channels, srate, filters, keep_s=rest_s + 30.0)
        self.collector = TrialCollector(source)
        self.tracker = ProgressTracker(classes, ScoreConfig(block_size=None))
        self.result: ResumeResult | None = None
        self._reference: FloatArray | None = None
        self._rest_windows = 0
        self._rest_begin: float | None = None
        self._rest_end: float | None = None
        self._events_seen = 0
        self._decisions: list[Decision] = []
        self._scored = 0
        self._waiting: list[Trial] = []
        self._watermark = float("-inf")

    @property
    def phase(self) -> str:
        """``"rest"``, ``"control"`` or ``"done"``."""
        if self.result is not None:
            return "done"
        return "rest" if self._reference is None else "control"

    @property
    def reference(self) -> FloatArray | None:
        """New reference covariance, once the rest is over."""
        return self._reference

    def push(self, block: npt.ArrayLike, timestamps: npt.ArrayLike | None = None) -> None:
        """Feed EEG samples ``(n, n_channels)`` in µV."""
        self.buffer.push(block, timestamps)
        self._try_reference()

    def add_decisions(self, decisions: Iterable[Decision], until: float | None = None) -> None:
        """Decisions of a decoder running with the new reference (control phase).

        A control trial is scored once the decoder has processed past its end:
        ``until`` is the time (stream clock) the decisions are complete up to,
        by default the latest decision timestamp.
        """
        new = list(decisions)
        self._decisions += new
        stamps = [d.timestamp for d in new if d.timestamp is not None]
        marks = ([until] if until is not None else []) + stamps
        if marks:
            self._watermark = max(self._watermark, *marks)
        self._score_ready()

    def finish(self) -> ResumeResult | None:
        """End of the session: score the trials still waiting for decisions."""
        self.collector.finish()
        self._watermark = float("inf")
        self._score_ready()
        return self.result

    def add_markers(self, markers: Iterable[Any]) -> ResumeResult | None:
        """Feed markers; returns the result once the control cues are scored."""
        self._waiting += self.collector.add(markers)
        events = self.collector.events
        for timestamp, label in events[self._events_seen :]:
            if label == "session_start" and self._rest_begin is None:
                self._rest_begin = timestamp
            elif _BLOCK_START.match(label) and self._rest_end is None:
                self._rest_end = timestamp
        self._events_seen = len(events)
        self._try_reference()
        self._score_ready()
        return self.result

    # -- internals -------------------------------------------------------------

    def _try_reference(self) -> None:
        end = self.buffer.end_time
        if self._reference is not None or self._rest_end is None or end is None:
            return
        if end < self._rest_end:
            return  # the rest samples have not all arrived
        begin = self._rest_begin if self._rest_begin is not None else self.buffer.start_time
        assert begin is not None
        begin = max(begin, self.buffer.start_time or begin) + _SETTLE_S
        windows: list[FloatArray] = []
        t = begin
        while t + WINDOW_S <= self._rest_end + 1e-9:
            window = self.buffer.window(t)
            if window is not None:
                windows.append(window)
            t += _STEP_S
        stack = np.stack(windows) if windows else np.zeros((0, len(self.buffer.channels), 0))
        if len(stack):
            stack = stack[amplitude_mask(stack, self.reject_uv)]
        usable = len(stack) * _STEP_S + (WINDOW_S - _STEP_S)
        if len(stack) == 0 or usable < self.rest_s:
            raise ValueError(
                f"only {usable:.0f} s of clean rest ({len(stack)} windows), "
                f"{self.rest_s:.0f} s needed"
            )
        self._reference = reference_covariance(stack, self.alignment)
        self._rest_windows = len(stack)

    def _score_ready(self) -> None:
        while (
            self.result is None
            and self._reference is not None
            and self._waiting
            and self._waiting[0].end <= self._watermark
        ):
            trial = self._waiting.pop(0)
            if self._rest_end is None or trial.start < self._rest_end:
                continue  # not a control cue
            self.tracker.add_trial(trial, self._decisions)
            self._scored += 1
            if self._scored >= self.control_cues:
                block = self.tracker.close_block()
                self.result = ResumeResult(
                    self._reference,
                    block.overall,
                    block.overall >= self.min_accuracy,
                    block,
                    self._rest_windows,
                )
