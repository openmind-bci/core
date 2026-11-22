"""Evidence accumulation: from window probabilities to movement commands.

One window of 2 s gives noisy probabilities. The accumulator averages the last
few of them, applies a threshold per command (with hysteresis so the command
does not flicker around the threshold) and falls back to ``rest`` whenever the
evidence is missing or too weak. Double jaw clenches become one-shot ``action``
decisions that do not change the current movement command.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from openmind_core.online.decoder import DecoderStep, Prediction
from openmind_protocol import CommandName

#: Commands a class of the model can stand for; any other class maps to ``rest``.
MOVEMENTS: tuple[CommandName, ...] = ("left", "right", "forward")


def _default_thresholds() -> dict[str, float]:
    return {"left": 0.6, "right": 0.6, "forward": 0.6}


@dataclass(frozen=True)
class EvidenceConfig:
    """Settings of :class:`EvidenceAccumulator`.

    Args:
        windows: Number of last predictions averaged (4 windows of 250 ms step
            is about 1 s).
        min_ok: Fewer decoded windows than this among the last ``windows`` mean
            no evidence: the decision is ``rest``.
        thresholds: Minimum mean probability per command. Objects apply their
            own, stricter ones; ``forward`` is weaker in EEG, so each command is
            tunable separately.
        hysteresis: The current command is kept while its mean probability stays
            above ``threshold - hysteresis``.
        action_confidence: Confidence given to ``action`` decisions.
    """

    windows: int = 4
    min_ok: int = 2
    thresholds: Mapping[str, float] = field(default_factory=_default_thresholds)
    hysteresis: float = 0.05
    action_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.windows < 1 or not 1 <= self.min_ok <= self.windows:
            raise ValueError("need windows >= 1 and 1 <= min_ok <= windows")
        if self.hysteresis < 0:
            raise ValueError("hysteresis must be >= 0")


@dataclass(frozen=True)
class Decision:
    """A command decided at one instant.

    Attributes:
        command: Protocol command name.
        confidence: Between 0 and 1.
        timestamp: Time of the window or action (decoder clock), if known.
        probabilities: Mean class probabilities behind the decision, ``None``
            when there was no evidence and for ``action``.
    """

    command: CommandName
    confidence: float
    timestamp: float | None = None
    probabilities: dict[str, float] | None = None


class EvidenceAccumulator:
    """Turn the predictions of :class:`OnlineDecoder` into :class:`Decision` objects.

    Args:
        classes: Class names of the model (those of ``Prediction.probabilities``).
        config: Settings.
    """

    def __init__(self, classes: Sequence[str], config: EvidenceConfig | None = None) -> None:
        self.classes = tuple(classes)
        self.config = config or EvidenceConfig()
        self._recent: deque[dict[str, float] | None] = deque(maxlen=self.config.windows)
        self._current: CommandName = "rest"

    @property
    def current(self) -> CommandName:
        """Last movement command decided (``rest`` at start); ``action`` never lands here."""
        return self._current

    def reset(self) -> None:
        """Forget the history and go back to ``rest``."""
        self._recent.clear()
        self._current = "rest"

    def update(self, step: DecoderStep) -> list[Decision]:
        """Decide for each prediction of ``step``; actions are inserted in time order."""
        actions = sorted(step.actions)
        out: list[Decision] = []
        i = 0
        for prediction in step.predictions:
            while i < len(actions) and actions[i] < prediction.timestamp:
                out.append(self._action(actions[i]))
                i += 1
            out.append(self._decide(prediction))
        out.extend(self._action(t) for t in actions[i:])
        return out

    # -- internals -------------------------------------------------------------

    def _action(self, t: float) -> Decision:
        return Decision("action", self.config.action_confidence, t)

    def _threshold(self, name: str) -> float | None:
        return self.config.thresholds.get(name)

    def _decide(self, prediction: Prediction) -> Decision:
        self._recent.append(prediction.probabilities if prediction.ok else None)
        ok = [p for p in self._recent if p is not None]
        stamp = prediction.timestamp
        if len(ok) < self.config.min_ok:
            self._current = "rest"
            return Decision("rest", 1.0, stamp)
        mean = {c: sum(p.get(c, 0.0) for p in ok) / len(ok) for c in self.classes}
        best = max(mean, key=lambda c: mean[c])
        threshold = self._threshold(best)
        if threshold is None or best not in MOVEMENTS:
            self._current = "rest"
            return Decision("rest", min(max(mean[best], 0.0), 1.0), stamp, mean)
        if best == self._current:
            threshold -= self.config.hysteresis
        if mean[best] >= threshold:
            command: CommandName = best
            self._current = command
            return Decision(command, min(mean[best], 1.0), stamp, mean)
        self._current = "rest"
        return Decision("rest", min(max(1.0 - mean[best], 0.0), 1.0), stamp, mean)
