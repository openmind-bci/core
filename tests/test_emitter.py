"""Tests of command, status and latency emission."""

from __future__ import annotations

import time

import pytest

from openmind_core.online import (
    CommandEmitter,
    Decision,
    HealthMonitor,
    LatencyMeter,
    StatusEmitter,
    decoder_state,
)
from openmind_core.profiles.store import InvalidUserIdError
from openmind_protocol import Command, Event, Status, decode, encode


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_emits_on_change_and_repeats_at_most_every_period() -> None:
    clock = Clock()
    emitter = CommandEmitter("P03", repeat_s=0.5, clock=clock)
    first = emitter.process([Decision("left", 0.8)])
    assert [m.command for m in first] == ["left"]
    clock.t += 0.25
    assert emitter.process([Decision("left", 0.8)]) == []
    clock.t += 0.25
    assert [m.command for m in emitter.process([Decision("left", 0.7)])] == ["left"]
    clock.t += 0.1
    assert [m.command for m in emitter.process([Decision("rest", 1.0)])] == ["rest"]


def test_seq_increments_and_messages_validate() -> None:
    clock = Clock()
    emitter = CommandEmitter("P03", clock=clock)
    out = emitter.process([Decision("left", 0.8), Decision("action", 0.95), Decision("right", 0.7)])
    assert [m.seq for m in out] == [0, 1, 2]
    assert [m.command for m in out] == ["left", "action", "right"]
    for message in out:
        assert decode(encode(message), Command) == message
    assert abs(out[0].ts - time.time()) < 5


def test_action_always_emitted_and_does_not_reset_repeat() -> None:
    clock = Clock()
    emitter = CommandEmitter("P03", repeat_s=0.5, clock=clock)
    emitter.process([Decision("left", 0.8)])
    out = emitter.process([Decision("action", 0.95), Decision("action", 0.95)])
    assert len(out) == 2
    clock.t += 0.2
    assert emitter.process([Decision("left", 0.8)]) == []


def test_invalid_user_rejected() -> None:
    with pytest.raises(InvalidUserIdError):
        CommandEmitter("alice")
    with pytest.raises(InvalidUserIdError):
        StatusEmitter("m", "alice")


def test_status_cadence_and_validation() -> None:
    clock = Clock()
    status = StatusEmitter("ts-lr", "P03", clock=clock)
    assert status.due()
    message = status.make("running", assistance=0.2)
    assert decode(encode(message), Status) == message
    assert not status.due()
    clock.t += 0.24
    assert not status.due()
    clock.t += 0.02
    assert status.due()  # 4 Hz, above the 2 Hz minimum
    assert StatusEmitter("m", None).make("idle").user is None


def test_decoder_state_mapping() -> None:
    assert decoder_state("calibrating", True) == "calibrating"
    assert decoder_state("calibrating", False) == "calibrating"
    assert decoder_state("running", True) == "running"
    assert decoder_state("running", False) == "degraded"


def test_health_monitor_degrades_after_two_seconds() -> None:
    clock = Clock()
    health = HealthMonitor(2.0, clock)
    clock.t += 1.9
    assert health.healthy
    clock.t += 0.2
    assert not health.healthy
    health.observe(0)
    assert not health.healthy
    health.observe(2)
    assert health.healthy


def test_latency_summary_and_event() -> None:
    clock = Clock()
    meter = LatencyMeter(clock)
    assert meter.summary()["count"] == 0
    for delay in (0.010, 0.020, 0.030, 0.040, 0.100):
        meter.record(clock.t - delay)
    summary = meter.summary()
    assert summary["count"] == 5
    assert summary["mean_ms"] == pytest.approx(40.0)
    assert summary["p50_ms"] == pytest.approx(30.0)
    assert summary["p95_ms"] == pytest.approx(88.0)
    assert summary["max_ms"] == pytest.approx(100.0)
    event = meter.to_event()
    assert (event.object, event.kind) == ("decoder", "latency")
    assert decode(encode(event), Event) == event
    assert event.data["count"] == 5
