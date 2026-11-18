"""Safety rules every real object must apply (see docs/protocol.md).

1. **Heartbeat.** Stop when no ``status`` was received for :data:`STATUS_TIMEOUT_S`
   seconds, or when the last status is not ``running``.
2. **Confidence.** Ignore commands whose ``confidence`` is below the object's own
   threshold.
3. **Order.** Ignore commands whose ``seq`` is not newer than the last one applied.

:class:`Watchdog` implements these rules for Python objects; firmware in other
languages must implement the same logic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

from openmind_protocol.messages import Command, Status

#: Seconds without status after which an object must stop.
STATUS_TIMEOUT_S: Final = 2.0
#: Minimum rate at which the decoder publishes its status.
STATUS_MIN_RATE_HZ: Final = 2.0


class Watchdog:
    """Decide whether an object may act on commands.

    The watchdog uses the **reception** time of messages (its own monotonic clock),
    not their ``ts`` field, so it works even when the object's clock is not set.

    Args:
        min_confidence: Commands below this confidence are refused.
        timeout: Seconds without status after which the object must stop.
        clock: Monotonic time source, for tests.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        timeout: float = STATUS_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.min_confidence = min_confidence
        self.timeout = timeout
        self._clock = clock
        self._last_status_at: float | None = None
        self._last_state: str | None = None
        self._last_seq: int | None = None

    def on_status(self, status: Status) -> None:
        """Record a received status."""
        self._last_status_at = self._clock()
        self._last_state = status.state

    @property
    def alive(self) -> bool:
        """Whether a ``running`` status was received less than ``timeout`` seconds ago."""
        if self._last_status_at is None or self._last_state != "running":
            return False
        return self._clock() - self._last_status_at < self.timeout

    def accept(self, command: Command) -> bool:
        """Whether ``command`` may be applied now; records it when accepted.

        A command is refused when the decoder is not alive, when its confidence is
        below ``min_confidence``, or when its ``seq`` is not newer than the last
        accepted one (duplicate or out-of-order delivery).
        """
        if not self.alive or command.confidence < self.min_confidence:
            return False
        if self._last_seq is not None and command.seq <= self._last_seq:
            return False
        self._last_seq = command.seq
        return True

    def reset_sequence(self) -> None:
        """Forget the last ``seq``, e.g. after the decoder restarted."""
        self._last_seq = None
