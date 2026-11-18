"""Contract tests: examples, schemas, models, codec, topics and safety rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmind_protocol import (
    COMMANDS,
    MODELS,
    BandPower,
    Command,
    Event,
    Marker,
    ProtocolError,
    Score,
    Status,
    UnsupportedVersionError,
    Watchdog,
    decode,
    decode_topic,
    encode,
    load_schema,
    schema_errors,
    topic_of,
    topics,
)
from openmind_protocol.schemas import schema_dir

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
VALID = sorted((EXAMPLES / "valid").glob("*.json"))
INVALID = sorted((EXAMPLES / "invalid").glob("*.json"))


def kind_of_example(path: Path) -> str:
    return path.stem.split("-", 1)[0]


def test_every_kind_has_examples() -> None:
    kinds = set(MODELS)
    assert {kind_of_example(p) for p in VALID} == kinds
    assert {kind_of_example(p) for p in INVALID} == kinds


# --- Acceptance: all valid examples pass, all invalid ones fail ---------------------


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_valid_examples_pass_schema_and_models(path: Path) -> None:
    kind = kind_of_example(path)
    document = json.loads(path.read_text())
    assert schema_errors(kind, document) == []  # type: ignore[arg-type]
    message = decode(path.read_bytes(), kind)  # type: ignore[arg-type]
    assert isinstance(message, MODELS[kind])  # type: ignore[index]
    # Re-encoding keeps every known field; unknown ones are dropped and absent
    # optional fields are written as null.
    again = json.loads(encode(message))
    for key, value in again.items():
        assert document.get(key) == value
    assert set(document) - set(again) <= {"latency_ms"}


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_invalid_examples_fail(path: Path) -> None:
    kind = kind_of_example(path)
    with pytest.raises(ProtocolError) as error:
        decode(path.read_bytes(), kind)  # type: ignore[arg-type]
    assert error.value.problems
    assert error.value.kind == kind
    try:
        document = json.loads(path.read_text(), parse_constant=lambda _: float("nan"))
    except json.JSONDecodeError:
        return  # not JSON at all
    if not any(token in path.stem for token in ("nan",)):
        assert schema_errors(kind, document), "the schema itself must reject it"  # type: ignore[arg-type]


# --- Schemas ----------------------------------------------------------------------


def test_schemas_are_found_and_identified() -> None:
    assert (schema_dir() / "v1").is_dir()
    for kind in MODELS:
        schema = load_schema(kind)
        assert schema["$id"].endswith(f"/v1/{kind}.schema.json")
        assert schema["properties"]["v"] == {"const": 1, "description": "Major protocol version."}
    with pytest.raises(KeyError, match="unknown message kind"):
        load_schema("nope")  # type: ignore[arg-type]


def test_schema_errors_name_the_field() -> None:
    errors = schema_errors(
        "command", {"v": 1, "command": "left", "confidence": 2, "ts": 1.0, "user": "P01", "seq": 0}
    )
    assert errors == ["confidence: 2 is greater than the maximum of 1"]


# --- Models and codec ---------------------------------------------------------------


def test_command_defaults_and_encoding() -> None:
    command = Command(command="right", confidence=0.9, user="P07", seq=3)
    assert command.v == 1
    assert command.ts > 1.7e9
    document = json.loads(encode(command))
    assert document == {
        "v": 1,
        "ts": command.ts,
        "command": "right",
        "confidence": 0.9,
        "user": "P07",
        "seq": 3,
    }
    assert encode(command) == encode(decode(encode(command), Command))


def test_models_reject_invalid_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Command(command="jump", confidence=0.5, user="P01", seq=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Command(command="left", confidence=float("nan"), user="P01", seq=0)
    with pytest.raises(ValidationError):
        Command(command="left", confidence=0.5, user="Alice", seq=0)


def test_messages_are_immutable() -> None:
    from pydantic import ValidationError

    command = Command(command="left", confidence=0.5, user="P01", seq=0)
    with pytest.raises(ValidationError):
        command.confidence = 1.0  # type: ignore[misc]


def test_decode_by_model_or_topic() -> None:
    payload = encode(Status(state="running", model="m", user="P01"))
    assert isinstance(decode(payload, Status), Status)
    assert isinstance(decode(payload.decode(), "status"), Status)
    assert isinstance(decode_topic(topics.STATUS, payload), Status)
    with pytest.raises(KeyError, match="not an OpenMind v1 topic"):
        decode_topic("openmind/v2/status", payload)


def test_decode_rejects_other_major_versions_explicitly() -> None:
    payload = (EXAMPLES / "invalid" / "command-version-2.json").read_bytes()
    with pytest.raises(UnsupportedVersionError, match="version 2 is not supported"):
        decode(payload, Command)


def test_decode_error_messages_are_explicit() -> None:
    with pytest.raises(ProtocolError, match=r"invalid command message: user: 'Thomas' does not"):
        decode((EXAMPLES / "invalid" / "command-real-name-as-user.json").read_bytes(), Command)
    with pytest.raises(ProtocolError, match="NaN is not valid JSON"):
        decode((EXAMPLES / "invalid" / "command-nan-confidence.json").read_bytes(), Command)
    with pytest.raises(ProtocolError, match="not valid UTF-8 JSON"):
        decode(b"\xff\xfe", Command)


def test_event_kind_field_and_bandpower_lookup() -> None:
    event = decode((EXAMPLES / "valid" / "event-robot-telemetry.json").read_bytes(), Event)
    assert event.kind == "telemetry"
    assert Event.KIND == "event"
    power = decode((EXAMPLES / "valid" / "bandpower-montage-a.json").read_bytes(), BandPower)
    assert power.channel("C4").rel_mu == -0.42
    assert not power.channel("Fp1").ok
    assert power.bands["mu"] == (8, 13)
    with pytest.raises(KeyError):
        power.channel("O1")


def test_score_and_marker_models() -> None:
    score = decode((EXAMPLES / "valid" / "score-ready.json").read_bytes(), Score)
    assert score.ready and score.per_command["left"] == 0.81
    marker = Marker(type="cue", label="left", trial=1, scene=None)
    assert topic_of(marker) == "openmind/v1/markers"


# --- Topics -----------------------------------------------------------------------


def test_topics() -> None:
    assert topics.COMMANDS == "openmind/v1/commands"
    assert topics.BANDPOWER == "openmind/v1/bandpower"
    assert set(topics.TOPICS) == set(MODELS)
    assert all(t.startswith(topics.PREFIX + "/") for t in topics.TOPICS.values())
    assert topics.kind_of("openmind/v1/score") == "score"
    assert COMMANDS == ("left", "right", "forward", "rest", "action")


# --- Safety watchdog ------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def running() -> Status:
    return Status(state="running", model="m", user="P01")


def cmd(seq: int, confidence: float = 0.9) -> Command:
    return Command(command="left", confidence=confidence, user="P01", seq=seq)


def test_watchdog_requires_a_recent_running_status() -> None:
    clock = FakeClock()
    dog = Watchdog(min_confidence=0.7, clock=clock)
    assert not dog.alive and not dog.accept(cmd(1))
    dog.on_status(running())
    assert dog.accept(cmd(1))
    clock.now = 1.99
    assert dog.alive
    clock.now = 2.0
    assert not dog.alive  # 2 s without status: stop
    assert not dog.accept(cmd(2))
    dog.on_status(Status(state="degraded", model="m", user="P01"))
    assert not dog.alive  # only "running" allows motion


def test_watchdog_filters_confidence_and_order() -> None:
    dog = Watchdog(min_confidence=0.8, clock=FakeClock())
    dog.on_status(running())
    assert not dog.accept(cmd(1, confidence=0.79))
    assert dog.accept(cmd(5))
    assert not dog.accept(cmd(5))  # duplicate
    assert not dog.accept(cmd(4))  # out of order
    dog.reset_sequence()
    assert dog.accept(cmd(0))
    with pytest.raises(ValueError):
        Watchdog(min_confidence=1.5)
