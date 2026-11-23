"""Progress score computed from cued trials and the decisions taken meanwhile.

A cued trial gives the expected command (the cue label) and a time window in
which the user imagines it. The decisions of the online pipeline inside that
window are compared with the cue:

* a trial is **correct** when the command held for the longest time (decisions
  are weighted by how long they stay in force) among the movement commands equals
  the cue label; for a ``rest`` cue it is correct when no movement command was
  held for at least ``false_start_s``;
* a **false start** is a movement command held for at least ``false_start_s``
  while the user should be at rest: before the cue of every trial, and during
  the whole window of ``rest`` trials.

Trials are grouped in blocks. A block *passes* when every command reaches
``min_accuracy`` and the false start rate stays under ``max_false_start``; the
user is *ready* after ``stable_blocks_required`` consecutive passed blocks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openmind_core.io.markers import Trial
from openmind_core.online.evidence import MOVEMENTS, Decision
from openmind_protocol import CommandName, Score

_REST = "rest"


@dataclass(frozen=True)
class ScoreConfig:
    """Thresholds and windows of the progress score.

    Args:
        min_accuracy: Minimum accuracy of every command.
        max_false_start: Maximum false start rate.
        stable_blocks_required: Consecutive passed blocks needed to be ready.
        window_start: Start of the imagery window, seconds after the cue (skips
            the visual response).
        window_end: End of the imagery window, seconds after the cue.
        false_start_s: Minimum time a movement command must be held while at
            rest to count as a false start.
        ece_bins: Number of bins of the calibration error.
        block_size: Trials per block (``None``: blocks are closed explicitly).
    """

    min_accuracy: float = 0.70
    max_false_start: float = 0.10
    stable_blocks_required: int = 2
    window_start: float = 0.5
    window_end: float = 4.0
    false_start_s: float = 0.5
    ece_bins: int = 10
    block_size: int | None = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_accuracy <= 1.0 or not 0.0 <= self.max_false_start <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        if self.stable_blocks_required < 1 or self.ece_bins < 1:
            raise ValueError("stable_blocks_required and ece_bins must be >= 1")
        if not 0.0 <= self.window_start < self.window_end:
            raise ValueError("need 0 <= window_start < window_end")
        if self.false_start_s < 0.0:
            raise ValueError("false_start_s must be >= 0")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be >= 1")


@dataclass(frozen=True)
class TrialOutcome:
    """Result of one trial.

    Attributes:
        label: Cue label.
        predicted: Movement command held the longest in the window (``rest``
            when none was held).
        correct: Whether the trial counts as correct.
        confidence: Time-weighted mean confidence of the decisions that agree
            with ``predicted`` (for calibration), ``None`` without decision.
        rest_periods: Periods in which the user should be at rest.
        false_starts: Among them, those with a false start.
    """

    label: str
    predicted: str
    correct: bool
    confidence: float | None
    rest_periods: int
    false_starts: int


@dataclass(frozen=True)
class _Segment:
    command: str
    confidence: float
    start: float
    end: float


def _segments(decisions: Sequence[Decision], start: float, end: float) -> list[_Segment]:
    """Decisions clipped to ``[start, end)``; each lasts until the next one."""
    timed = sorted(
        (d for d in decisions if d.timestamp is not None), key=lambda d: cast(float, d.timestamp)
    )
    out: list[_Segment] = []
    for i, decision in enumerate(timed):
        t0 = cast(float, decision.timestamp)
        t1 = cast(float, timed[i + 1].timestamp) if i + 1 < len(timed) else end
        t0, t1 = max(t0, start), min(t1, end)
        if t1 > t0:
            out.append(_Segment(decision.command, decision.confidence, t0, t1))
    return out


def _has_false_start(segments: Iterable[_Segment], min_hold: float) -> bool:
    """Whether a movement command is held for ``min_hold`` seconds in a row."""
    run_command: str | None = None
    run_start = run_end = 0.0
    for seg in segments:
        if seg.command in MOVEMENTS and seg.command == run_command and seg.start <= run_end + 1e-9:
            run_end = seg.end
        else:
            run_command = seg.command if seg.command in MOVEMENTS else None
            run_start, run_end = seg.start, seg.end
        if run_command is not None and run_end - run_start >= min_hold - 1e-9:
            return True
    return False


def score_trial(
    trial: Trial,
    decisions: Sequence[Decision],
    config: ScoreConfig | None = None,
) -> TrialOutcome:
    """Score one cued trial against the decisions of the pipeline.

    Args:
        trial: A complete trial (label, start, cue and end times).
        decisions: Decisions, with timestamps on the same clock as the trial.
            ``action`` decisions are one-shot and ignored.
        config: Thresholds and windows.
    """
    cfg = config or ScoreConfig()
    win_start = trial.cue_time + cfg.window_start
    win_end = min(trial.cue_time + cfg.window_end, trial.end)
    pre = _segments(decisions, trial.start, trial.cue_time)
    window = _segments(decisions, win_start, win_end)

    rest_periods = 1 if trial.cue_time > trial.start else 0
    false_starts = int(_has_false_start(pre, cfg.false_start_s)) if rest_periods else 0

    weights: dict[str, float] = defaultdict(float)
    for seg in window:
        if seg.command in MOVEMENTS:
            weights[seg.command] += seg.end - seg.start
    predicted = max(weights, key=lambda c: weights[c]) if weights else _REST

    if trial.label == _REST:
        rest_periods += 1
        false_starts += int(_has_false_start(window, cfg.false_start_s))
        correct = not _has_false_start(window, cfg.false_start_s)
    else:
        correct = predicted == trial.label

    agreeing = [s for s in window if s.command == predicted]
    total = sum(s.end - s.start for s in agreeing)
    confidence = (
        sum(s.confidence * (s.end - s.start) for s in agreeing) / total if total > 0 else None
    )
    return TrialOutcome(trial.label, predicted, correct, confidence, rest_periods, false_starts)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 10,
) -> float:
    """Expected calibration error: weighted gap between confidence and accuracy.

    Predictions are split in ``bins`` equal-width confidence bins; the error is
    the mean over bins of ``|accuracy - mean confidence|`` weighted by the
    share of predictions in the bin.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    n = len(confidences)
    if n == 0:
        raise ValueError("no prediction")
    groups: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, ok in zip(confidences, correct, strict=True):
        groups[min(int(conf * bins), bins - 1)].append((conf, ok))
    error = 0.0
    for items in groups.values():
        mean_conf = sum(c for c, _ in items) / len(items)
        accuracy = sum(ok for _, ok in items) / len(items)
        error += len(items) / n * abs(accuracy - mean_conf)
    return error


@dataclass(frozen=True)
class BlockScore:
    """Score of one block of trials.

    Attributes:
        per_command: Accuracy of each command seen in the block.
        false_start_rate: False starts over rest periods.
        ece: Calibration error, ``None`` without confident prediction.
        overall: Mean of the per-command accuracies.
        passed: Every class seen, every accuracy and the false start rate in
            the thresholds.
        n_trials: Number of trials in the block.
    """

    per_command: dict[str, float]
    false_start_rate: float
    ece: float | None
    overall: float
    passed: bool
    n_trials: int


class ProgressTracker:
    """Accumulate trials into blocks and tell when the user is ready.

    Args:
        classes: Commands the user must master (``left``, ``right``, ``rest``...).
            A block in which one is missing does not pass.
        config: Thresholds, windows and block size.
    """

    def __init__(self, classes: Sequence[str], config: ScoreConfig | None = None) -> None:
        if not classes:
            raise ValueError("classes must not be empty")
        self.classes = tuple(classes)
        self.config = config or ScoreConfig()
        self.blocks: list[BlockScore] = []
        self._outcomes: list[TrialOutcome] = []

    @property
    def stable_blocks(self) -> int:
        """Number of consecutive passed blocks, counting back from the last."""
        count = 0
        for block in reversed(self.blocks):
            if not block.passed:
                break
            count += 1
        return count

    @property
    def ready(self) -> bool:
        """Whether enough consecutive blocks passed."""
        return self.stable_blocks >= self.config.stable_blocks_required

    @property
    def pending_trials(self) -> int:
        """Trials of the current, unclosed block."""
        return len(self._outcomes)

    def add_trial(self, trial: Trial, decisions: Sequence[Decision]) -> BlockScore | None:
        """Score a trial; returns the block score when it closes a block."""
        self._outcomes.append(score_trial(trial, decisions, self.config))
        size = self.config.block_size
        if size is not None and len(self._outcomes) >= size:
            return self.close_block()
        return None

    def close_block(self) -> BlockScore:
        """Close the current block (for instance on a ``block_end`` marker)."""
        if not self._outcomes:
            raise ValueError("no trial in the current block")
        cfg, outcomes = self.config, self._outcomes
        self._outcomes = []
        hits: dict[str, list[bool]] = defaultdict(list)
        for outcome in outcomes:
            hits[outcome.label].append(outcome.correct)
        per_command = {c: sum(v) / len(v) for c, v in hits.items()}
        periods = sum(o.rest_periods for o in outcomes)
        false_rate = sum(o.false_starts for o in outcomes) / periods if periods else 0.0
        pairs = [(o.confidence, o.correct) for o in outcomes if o.confidence is not None]
        ece = (
            expected_calibration_error([c for c, _ in pairs], [ok for _, ok in pairs], cfg.ece_bins)
            if pairs
            else None
        )
        overall = sum(per_command.values()) / len(per_command)
        passed = (
            all(c in per_command and per_command[c] >= cfg.min_accuracy for c in self.classes)
            and false_rate <= cfg.max_false_start
        )
        block = BlockScore(per_command, false_rate, ece, overall, passed, len(outcomes))
        self.blocks.append(block)
        return block

    def to_message(self, user: str, ts: float | None = None) -> Score:
        """Protocol ``score`` message with the values of the latest block.

        Before any block, accuracies are empty and the user is not ready.
        """
        last = self.blocks[-1] if self.blocks else None
        fields: dict[str, Any] = {
            "user": user,
            "overall": last.overall if last else 0.0,
            "ready": self.ready,
            "per_command": cast("dict[CommandName, float]", dict(last.per_command)) if last else {},
            "false_start_rate": last.false_start_rate if last else 0.0,
            "stable_blocks": self.stable_blocks,
            "ece": last.ece if last else None,
        }
        if ts is not None:
            fields["ts"] = ts
        return Score(**fields)

    def to_record(self, user: str, ts: float | None = None) -> dict[str, Any]:
        """JSON-ready record for :meth:`ProfileStore.append_score`."""
        record = self.to_message(user, ts).model_dump(mode="json", exclude={"user"})
        record["blocks"] = len(self.blocks)
        return record
