"""Build protocol messages from decisions: commands, status heartbeat, latency.

No network here: the caller publishes the returned messages (MQTT lives
elsewhere). Clocks are injectable for tests.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from openmind_core.online.evidence import Decision
from openmind_core.profiles.store import validate_user_id
from openmind_protocol import Command, DecoderState, Event, Status

Clock = Callable[[], float]


class CommandEmitter:
    """Decide which :class:`~openmind_protocol.Command` messages to publish.

    A message is emitted on every change of command and the current command is
    repeated every ``repeat_s`` seconds at most (a lost message is recovered and
    objects with a 2 s heartbeat see a live stream). ``action`` is always emitted.
    ``seq`` starts at 0 and grows by 1 per message; ``ts`` is Unix time.

    Args:
        user: Pseudonymized profile id (``P01``...).
        repeat_s: Minimum delay between two repeats of the same command.
        clock: Monotonic clock for the repeat timing.
        wall_clock: Unix time for ``ts``.
    """

    def __init__(
        self,
        user: str,
        repeat_s: float = 0.5,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ) -> None:
        self.user = validate_user_id(user)
        self.repeat_s = repeat_s
        self._clock = clock
        self._wall = wall_clock
        self._seq = 0
        self._last: str | None = None
        self._last_sent = 0.0

    def process(self, decisions: Iterable[Decision]) -> list[Command]:
        """Return the messages to publish for these decisions, in order."""
        out: list[Command] = []
        for decision in decisions:
            now = self._clock()
            if decision.command == "action":
                out.append(self._make(decision))
                continue
            changed = decision.command != self._last
            if changed or now - self._last_sent >= self.repeat_s:
                self._last = decision.command
                self._last_sent = now
                out.append(self._make(decision))
        return out

    def _make(self, decision: Decision) -> Command:
        message = Command(
            command=decision.command,
            confidence=decision.confidence,
            user=self.user,
            seq=self._seq,
            ts=self._wall(),
        )
        self._seq += 1
        return message


def decoder_state(state: str, healthy: bool) -> DecoderState:
    """Protocol state from the decoder state: calibrating, degraded or running."""
    if state == "calibrating":
        return "calibrating"
    return "running" if healthy else "degraded"


class HealthMonitor:
    """Healthy while a decoded window arrived less than ``timeout_s`` ago.

    A silent stream or permanently rejected windows make it unhealthy.
    """

    def __init__(self, timeout_s: float = 2.0, clock: Clock = time.monotonic) -> None:
        self.timeout_s = timeout_s
        self._clock = clock
        self._last_ok = clock()

    def observe(self, ok_windows: int) -> None:
        """Record a step that produced ``ok_windows`` decoded windows."""
        if ok_windows > 0:
            self._last_ok = self._clock()

    @property
    def healthy(self) -> bool:
        return self._clock() - self._last_ok <= self.timeout_s


class StatusEmitter:
    """Paces the decoder heartbeat (``openmind/v1/status``, at least 2 Hz).

    Args:
        model: Name of the model in use.
        user: Profile id, or ``None``.
        period_s: Delay between statuses (0.25 s is 4 Hz).
        clock: Monotonic clock.
    """

    def __init__(
        self,
        model: str,
        user: str | None,
        period_s: float = 0.25,
        clock: Clock = time.monotonic,
    ) -> None:
        self.model = model
        self.user = None if user is None else validate_user_id(user)
        self.period_s = period_s
        self._clock = clock
        self._last: float | None = None

    def due(self) -> bool:
        """Whether a status should be published now."""
        return self._last is None or self._clock() - self._last >= self.period_s

    def make(
        self, state: DecoderState, assistance: float | None = None, message: str | None = None
    ) -> Status:
        """Build the status and restart the period."""
        self._last = self._clock()
        return Status(
            state=state,
            model=self.model,
            user=self.user,
            assistance=assistance,
            message=message,
        )


class LatencyMeter:
    """Delay between the arrival of a window's last sample and its publication.

    Args:
        clock: Monotonic clock, the same as the one the caller uses to stamp
            arrivals.
        maxlen: Number of last measurements kept.
    """

    def __init__(self, clock: Clock = time.monotonic, maxlen: int = 10_000) -> None:
        self._clock = clock
        self._delays: deque[float] = deque(maxlen=maxlen)

    def record(self, arrival: float, published: float | None = None) -> float:
        """Record a command published now (or at ``published``); return the delay in s."""
        delay = max((self._clock() if published is None else published) - arrival, 0.0)
        self._delays.append(delay)
        return delay

    def summary(self) -> dict[str, float | int]:
        """Count, mean, p50, p95 and max of the delays, in milliseconds."""
        ms = sorted(d * 1000.0 for d in self._delays)
        if not ms:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(ms),
            "mean_ms": sum(ms) / len(ms),
            "p50_ms": _percentile(ms, 50),
            "p95_ms": _percentile(ms, 95),
            "max_ms": ms[-1],
        }

    def to_event(self, wall_clock: Clock = time.time) -> Event:
        """The summary as a free-telemetry ``Event(object="decoder", kind="latency")``."""
        data: dict[str, Any] = dict(self.summary())
        return Event(object="decoder", kind="latency", data=data, ts=wall_clock())


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list."""
    pos = (len(sorted_values) - 1) * q / 100.0
    low, high = math.floor(pos), math.ceil(pos)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)
