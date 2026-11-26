"""Tests of the online service (scripted model, fake clock) and of the ``online`` command."""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from openmind_core.cli import app
from openmind_core.models import make_baseline
from openmind_core.models.registry import ModelCard, save_model
from openmind_core.online import (
    CommandEmitter,
    EvidenceAccumulator,
    HealthMonitor,
    LatencyMeter,
    OnlineConfig,
    OnlineDecoder,
    StatusEmitter,
)
from openmind_core.online.service import OnlineStats, run_online
from openmind_core.profiles import ProfileStore, Reference
from openmind_protocol import Command, Event, Message, Status, decode, encode

FS = 250.0
CHANNELS = ("C3", "Cz", "C4")
CLASSES = ("left", "right", "forward")
BLOCK = 25  # 0.1 s


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Scripted:
    """Model whose probabilities follow ``script(call_index)``."""

    def __init__(self, script: Callable[[int], tuple[float, float, float]]) -> None:
        self.script = script
        self.calls = 0

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        out = np.array([self.script(self.calls + i) for i in range(len(x))])
        self.calls += len(x)
        return out


def make_card(alignment: str | None = None) -> ModelCard:
    return ModelCard(
        name="scripted",
        architecture="ts-lr",
        channels=CHANNELS,
        srate=FS,
        n_times=500,
        classes=CLASSES,
        scale=(1.0,) * 3,
        format="sklearn",
        alignment=alignment,
    )


def blocks(
    seconds: float, clock: Clock, empty: bool = False, seed: int = 0
) -> Iterator[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """One block per 0.1 s of fake time; the fake clock advances with them."""
    rng = np.random.default_rng(seed)
    for k in range(round(seconds / 0.1)):
        clock.t += 0.1
        n = 0 if empty else BLOCK
        t = 1000.0 + (k * BLOCK + np.arange(n)) / FS
        yield rng.normal(0.0, 5.0, (n, 3)), t


def run(
    model: object,
    stream: Callable[[Clock], Iterator[tuple[Any, Any]]],
    *,
    alignment: str | None = None,
    warmup_s: float = 30.0,
    reference: Reference | None = None,
    on_reference: Callable[[Any], None] | None = None,
    latency_every_s: float = 10.0,
    stop: threading.Event | None = None,
) -> tuple[OnlineStats, list[Message]]:
    clock = Clock()
    card = make_card(alignment)
    decoder = OnlineDecoder(
        model, card, CHANNELS, FS, OnlineConfig(warmup_s=warmup_s), reference=reference
    )
    sent: list[Message] = []
    stats = run_online(
        stream(clock),
        sent.append,
        decoder=decoder,
        accumulator=EvidenceAccumulator(CLASSES),
        command_emitter=CommandEmitter("P01", clock=clock),
        status_emitter=StatusEmitter("scripted", "P01", clock=clock),
        health=HealthMonitor(clock=clock),
        latency=LatencyMeter(clock=clock),
        clock=clock,
        on_reference=on_reference,
        latency_every_s=latency_every_s,
        stop=stop,
    )
    return stats, sent


def left_then_uniform(i: int) -> tuple[float, float, float]:
    return (0.9, 0.05, 0.05) if i < 30 else (0.34, 0.33, 0.33)


def commands(sent: list[Message]) -> list[Command]:
    return [m for m in sent if isinstance(m, Command)]


def statuses(sent: list[Message]) -> list[Status]:
    return [m for m in sent if isinstance(m, Status)]


def test_messages_validate_and_commands_follow_the_model() -> None:
    stats, sent = run(Scripted(left_then_uniform), lambda c: blocks(25.0, c))
    for message in sent:
        assert decode(encode(message), type(message)) == message
    names = [c.command for c in commands(sent)]
    assert "left" in names
    assert names.index("left") < len(names) - 2
    assert names.count("left") >= 2  # repeated every 0.5 s while it holds
    assert names[-1] == "rest"
    assert "rest" in names[: names.index("left")]  # rest before enough evidence
    seqs = [c.seq for c in commands(sent)]
    assert seqs == list(range(len(seqs)))
    assert stats.blocks == 250
    assert stats.decoded_windows > 0
    assert stats.commands["left"] == names.count("left")
    assert stats.latency["count"] == len(names) - 1  # the final rest is not measured
    assert stats.duration_s == pytest.approx(25.0)


class DecoderError(RuntimeError):
    pass


class FailingModel:
    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        raise DecoderError("decoder broke")


def run_with_publish(model: object, publish: Callable[[Message], None]) -> OnlineStats:
    clock = Clock()
    decoder = OnlineDecoder(model, make_card(), CHANNELS, FS, OnlineConfig(warmup_s=0.0))
    return run_online(
        blocks(5.0, clock),
        publish,
        decoder=decoder,
        accumulator=EvidenceAccumulator(CLASSES),
        command_emitter=CommandEmitter("P01", clock=clock),
        status_emitter=StatusEmitter("scripted", "P01", clock=clock),
        health=HealthMonitor(clock=clock),
        latency=LatencyMeter(clock=clock),
        clock=clock,
    )


def test_original_error_survives_a_failing_final_publish(caplog: pytest.LogCaptureFixture) -> None:
    def publish(message: Message) -> None:
        # the failing decoder produces no command, so the only one is the final rest
        if isinstance(message, Command):
            raise ConnectionError("broker lost")

    with (
        caplog.at_level("ERROR", logger="openmind_core.online.service"),
        pytest.raises(DecoderError, match="decoder broke"),
    ):
        run_with_publish(FailingModel(), publish)
    assert "final rest" in caplog.text
    assert "broker lost" in caplog.text


def test_failing_final_publish_is_raised_when_nothing_else_failed() -> None:
    def publish(message: Message) -> None:
        if isinstance(message, Status) and message.state == "idle":
            raise ConnectionError("broker lost")

    with pytest.raises(ConnectionError, match="broker lost"):
        run_with_publish(Scripted(lambda i: (0.34, 0.33, 0.33)), publish)


def test_final_rest_and_idle_even_when_last_command_was_rest() -> None:
    _, sent = run(Scripted(lambda i: (0.34, 0.33, 0.33)), lambda c: blocks(8.0, c))
    assert isinstance(sent[-2], Command) and sent[-2].command == "rest"
    assert isinstance(sent[-1], Status) and sent[-1].state == "idle"
    assert len([c for c in commands(sent) if c.command == "rest"]) >= 2


def test_final_messages_on_interrupt() -> None:
    def stream(clock: Clock) -> Iterator[tuple[Any, Any]]:
        yield from blocks(3.0, clock)
        raise KeyboardInterrupt

    clock_sent: list[Message] = []
    with pytest.raises(KeyboardInterrupt):
        decoder = OnlineDecoder(Scripted(left_then_uniform), make_card(), CHANNELS, FS)
        clock = Clock()
        run_online(
            stream(clock),
            clock_sent.append,
            decoder=decoder,
            accumulator=EvidenceAccumulator(CLASSES),
            command_emitter=CommandEmitter("P01", clock=clock),
            status_emitter=StatusEmitter("scripted", "P01", clock=clock),
            health=HealthMonitor(clock=clock),
            latency=LatencyMeter(clock=clock),
            clock=clock,
        )
    assert isinstance(clock_sent[-2], Command) and clock_sent[-2].command == "rest"
    assert isinstance(clock_sent[-1], Status) and clock_sent[-1].state == "idle"


def test_stop_event_ends_the_loop() -> None:
    stop = threading.Event()

    def stream(clock: Clock) -> Iterator[tuple[Any, Any]]:
        for i, block in enumerate(blocks(60.0, clock)):
            if i == 40:
                stop.set()
            yield block

    stats, sent = run(Scripted(left_then_uniform), stream, stop=stop)
    assert stats.blocks == 40
    assert isinstance(sent[-1], Status) and sent[-1].state == "idle"


def test_status_keeps_flowing_with_empty_blocks() -> None:
    stats, sent = run(Scripted(left_then_uniform), lambda c: blocks(10.0, c, empty=True))
    live = statuses(sent)[:-1]  # without the final idle
    assert len(live) >= 20  # at least 2 Hz over 10 s
    assert live[0].state == "running"
    assert live[-1].state == "degraded"  # no decoded window for 2 s
    assert live[-1].message is not None and "no decoded window" in live[-1].message
    assert stats.windows == 0
    assert [c.command for c in commands(sent)] == ["rest"]  # only the final one


def test_warmup_then_running_and_reference_once() -> None:
    received: list[Any] = []
    stats, sent = run(
        Scripted(lambda i: (0.9, 0.05, 0.05)),
        lambda c: blocks(14.0, c),
        alignment="riemann",
        warmup_s=4.0,
        on_reference=received.append,
    )
    states = [s.state for s in statuses(sent)]
    assert states[0] == "calibrating"
    assert states[-2] == "running"
    assert states[-1] == "idle"
    first = statuses(sent)[0]
    assert first.message is not None and first.message.startswith("warm-up")
    assert len(received) == 1
    assert received[0].shape == (3, 3)
    assert stats.decoded_windows > 0
    # Calibrating, never degraded: "calibrating" precedes "running" without a gap state.
    assert "degraded" not in states


def test_reference_given_up_front_skips_warmup() -> None:
    received: list[Any] = []
    reference = Reference(np.eye(3) * 25.0, CHANNELS)
    _, sent = run(
        Scripted(lambda i: (0.9, 0.05, 0.05)),
        lambda c: blocks(6.0, c),
        alignment="riemann",
        reference=reference,
        on_reference=received.append,
    )
    assert statuses(sent)[0].state == "running"
    assert received == []


def test_latency_event_published() -> None:
    _, sent = run(Scripted(left_then_uniform), lambda c: blocks(10.0, c), latency_every_s=3.0)
    events = [m for m in sent if isinstance(m, Event)]
    assert len(events) == 3
    assert events[0].kind == "latency" and events[0].object == "decoder"


# --- command line -----------------------------------------------------------------

runner = CliRunner()
# Rich colours the help when the CI forces colours (FORCE_COLOR): the escape
# codes split the option names, so they are removed before searching.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def data_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    return tmp_path


def test_cli_help() -> None:
    result = runner.invoke(app, ["online", "--help"])
    assert result.exit_code == 0
    output = _ANSI.sub("", result.output)
    for option in ("--profile", "--model", "--broker", "--warmup-s", "--threshold", "--dry-run"):
        assert option in output


def test_cli_missing_plug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openmind_plug", None)
    monkeypatch.setitem(sys.modules, "openmind_plug.inlet", None)
    result = runner.invoke(app, ["online", "--profile", "P01", "--model", "m", "--dry-run"])
    assert result.exit_code == 1
    assert "uv sync --extra plug" in result.output


def test_cli_unknown_profile(data_dir: Any) -> None:
    pytest.importorskip("openmind_plug")
    result = runner.invoke(app, ["online", "--profile", "P01", "--model", "m", "--dry-run"])
    assert result.exit_code == 1
    assert "openmind-core profile create P01" in result.output


def test_cli_unknown_model(data_dir: Any) -> None:
    pytest.importorskip("openmind_plug")
    ProfileStore().create("P01")
    result = runner.invoke(app, ["online", "--profile", "P01", "--model", "nope", "--dry-run"])
    assert result.exit_code == 1
    assert "nope" in result.output and "openmind-core train" in result.output


def test_cli_rejects_bad_threshold(data_dir: Any) -> None:
    pytest.importorskip("openmind_plug")
    result = runner.invoke(
        app,
        ["online", "--profile", "P01", "--model", "m", "--threshold", "up=0.5", "--dry-run"],
    )
    assert result.exit_code != 0


# --- integration with Plug (skipped without the extra) ------------------------------


def test_synthetic_stream_through_lsl(data_dir: Any) -> None:
    pytest.importorskip("openmind_plug")
    from openmind_plug.inlet import EEGInlet
    from openmind_plug.outlet import EEGOutlet, stream
    from openmind_plug.sources.synthetic import SyntheticSource

    source = SyntheticSource(montage="C", seed=1)
    labels = tuple(source.info.labels)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 10.0, (60, len(labels), 500))
    y = np.array(CLASSES * 20)
    x[y == "left", 4] *= 3.0
    model = make_baseline("ts-lr").fit(x, y)
    save_model(model, "synthetic-c", labels, srate=FS, n_times=500, root=data_dir / "models")
    from openmind_core.models.registry import load_model

    net, card = load_model("synthetic-c", data_dir / "models")

    stop = threading.Event()
    sent: list[Message] = []
    with EEGOutlet(source.info, source_id="openmind-plug:test:online") as outlet:
        thread = threading.Thread(
            target=stream, args=(source,), kwargs={"outlet": outlet, "stop_event": stop}
        )
        thread.start()
        try:
            with EEGInlet(
                timeout=10.0, source_id="openmind-plug:test:online", with_markers=False
            ) as inlet:
                deadline = time.monotonic() + 4.0

                def live() -> Iterator[tuple[Any, Any]]:
                    while time.monotonic() < deadline:
                        yield inlet.read(timeout=0.1)

                decoder = OnlineDecoder(net, card, list(inlet.info.labels), inlet.info.srate)
                stats = run_online(
                    live(),
                    sent.append,
                    decoder=decoder,
                    accumulator=EvidenceAccumulator(card.classes),
                    command_emitter=CommandEmitter("P01"),
                    status_emitter=StatusEmitter("synthetic-c", "P01"),
                    health=HealthMonitor(),
                    latency=LatencyMeter(),
                )
        finally:
            stop.set()
            thread.join(timeout=5.0)
    assert statuses(sent)[0].state in {"running", "degraded"}
    assert len(statuses(sent)) >= 8
    assert commands(sent)
    assert stats.windows > 0
    assert isinstance(sent[-1], Status) and sent[-1].state == "idle"
