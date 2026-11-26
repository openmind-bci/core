"""Online service: EEG blocks in, ``commands``, ``status`` and latency events out.

The loop is independent of LSL and MQTT: it takes an iterator of
``(samples, timestamps)`` blocks and a ``publish`` callable, so it runs the same
on a live stream, a replay or scripted blocks in tests.

**The status heartbeat follows the wall clock, not the data.** Objects stop when
the status stops (2 s), so it must keep flowing when the stream goes silent.
The loop only gets the chance to publish when the iterator yields: the caller
must therefore poll the stream at 10 Hz or faster and yield an *empty* block
when nothing arrived (``EEGInlet.read(timeout=0.1)`` already returns empty
arrays on timeout).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from openmind_core.online.decoder import FloatArray, OnlineDecoder
from openmind_core.online.emitter import (
    Clock,
    CommandEmitter,
    HealthMonitor,
    LatencyMeter,
    StatusEmitter,
    decoder_state,
)
from openmind_core.online.evidence import Decision, EvidenceAccumulator
from openmind_protocol import Command, Message, Status

logger = logging.getLogger(__name__)

Block = tuple[npt.ArrayLike, npt.ArrayLike]


@dataclass(frozen=True)
class OnlineStats:
    """What a run of :func:`run_online` did."""

    blocks: int
    windows: int
    decoded_windows: int
    commands: dict[str, int] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


def run_online(
    blocks: Iterable[Block],
    publish: Callable[[Message], None],
    *,
    decoder: OnlineDecoder,
    accumulator: EvidenceAccumulator,
    command_emitter: CommandEmitter,
    status_emitter: StatusEmitter,
    health: HealthMonitor,
    latency: LatencyMeter,
    clock: Clock = time.monotonic,
    on_reference: Callable[[FloatArray], None] | None = None,
    latency_every_s: float = 10.0,
    stop: threading.Event | None = None,
) -> OnlineStats:
    """Decode ``blocks`` and publish commands, statuses and latency events.

    Args:
        blocks: ``(samples, timestamps)`` blocks, samples ``(n, n_channels)`` in
            µV. An empty block is allowed and only gives the loop a chance to
            publish the status; yield one at least every 100 ms.
        publish: Called with each message (``Command``, ``Status``, ``Event``).
        decoder: Online decoder.
        accumulator: Evidence accumulator, fed with the decoder steps.
        command_emitter: Decides which commands are published.
        status_emitter: Paces the status heartbeat.
        health: Healthy while decoded windows keep arriving.
        latency: Records the delay between the arrival of a block and the
            publication of the commands it produced.
        clock: Monotonic clock (arrival times, durations, latency period); must
            be the one given to the emitters and the monitors.
        on_reference: Called once, with the reference covariance, when the
            decoder leaves warm-up (not called if it started with a reference).
        latency_every_s: Period of the latency ``Event``.
        stop: Set it to end the loop after the current block.

    Returns:
        Counters of the run. On exit (end of ``blocks``, ``stop``, or any
        exception such as ``KeyboardInterrupt``) a final ``rest`` command and an
        ``idle`` status are published so that objects stop at once.
    """
    started = clock()
    last_latency = started
    n_blocks = n_windows = n_decoded = 0
    counts: dict[str, int] = {}
    warmup_windows = 0
    waiting_for_reference = decoder.state == "calibrating"

    def send_commands(commands: list[Command], arrival: float) -> None:
        for command in commands:
            publish(command)
            counts[command.command] = counts.get(command.command, 0) + 1
            latency.record(arrival)

    def status() -> Status:
        healthy = health.healthy
        state = decoder_state(decoder.state, healthy)
        if decoder.state == "calibrating":
            config = decoder.config
            span = 0.0
            if warmup_windows:
                span = (warmup_windows - 1) * config.step_s + config.window_s
            text = f"warm-up {min(span, config.warmup_s):.0f}/{config.warmup_s:.0f} s"
        elif not healthy:
            text = f"no decoded window for {health.timeout_s:g} s"
        else:
            text = None
        return status_emitter.make(state, message=text)

    error: BaseException | None = None
    try:
        for samples, timestamps in blocks:
            if stop is not None and stop.is_set():
                break
            arrival = clock()
            n_blocks += 1
            data = np.asarray(samples)
            if data.shape[0] > 0:
                step = decoder.push(data, timestamps)
                predictions = step.predictions
                n_windows += len(predictions)
                ok = sum(1 for p in predictions if p.ok)
                n_decoded += ok
                warmup_windows += sum(1 for p in predictions if p.reason == "warmup")
                health.observe(ok)
                decisions: list[Decision] = accumulator.update(step)
                send_commands(command_emitter.process(decisions), arrival)
                if waiting_for_reference and decoder.state != "calibrating":
                    waiting_for_reference = False
                    reference = decoder.reference_covariance
                    if on_reference is not None and reference is not None:
                        on_reference(reference)
            if status_emitter.due():
                publish(status())
            now = clock()
            if now - last_latency >= latency_every_s:
                last_latency = now
                publish(latency.to_event())
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            publish(command_emitter.force(Decision("rest", 1.0)))
            publish(status_emitter.make("idle", message="stopped"))
        except Exception:
            if error is None:
                raise
            # Raising here would replace the error that ended the loop (e.g. a decoder
            # failure) with the publish failure (e.g. a lost broker): log it and let the
            # original error propagate.
            logger.exception("could not publish the final rest command and idle status")
    return OnlineStats(
        blocks=n_blocks,
        windows=n_windows,
        decoded_windows=n_decoded,
        commands=counts,
        latency=dict(latency.summary()),
        duration_s=clock() - started,
    )


__all__ = ["Block", "OnlineStats", "run_online"]
