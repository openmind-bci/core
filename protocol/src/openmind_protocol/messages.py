"""Validated models of the protocol messages, mirroring ``schemas/v1``.

The JSON Schemas are the language-neutral contract and these models follow them;
the test-suite checks every valid and invalid example against both. When building
messages in Python, ``v`` and ``ts`` get defaults (the schema requires them on the
wire, and :func:`~openmind_protocol.decode` enforces it). Unknown fields are
ignored, so minor versions can add optional fields.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from openmind_protocol.topics import Kind

#: Major version carried in every message (``"v"``).
VERSION: Final = 1

CommandName = Literal["left", "right", "forward", "rest", "action"]
DecoderState = Literal["idle", "calibrating", "running", "degraded"]
MarkerType = Literal["cue", "trial_start", "trial_end", "event"]

#: The five commands, in a stable order.
COMMANDS: Final[tuple[CommandName, ...]] = ("left", "right", "forward", "rest", "action")

UserId = Annotated[str, Field(pattern=r"^P[0-9]{2,}$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Timestamp = Annotated[float, Field(ge=0.0)]


def now() -> float:
    """Current Unix time in seconds, the clock of ``ts`` fields."""
    return time.time()


_CONFIG = ConfigDict(frozen=True, extra="ignore", strict=True, allow_inf_nan=False)


class Message(BaseModel):
    """Base of all messages: version field, strict types, unknown fields ignored."""

    model_config = _CONFIG

    #: Message kind (``"command"``, ``"status"``...); set by each subclass.
    KIND: ClassVar[Kind]

    v: Literal[1] = VERSION
    ts: Timestamp = Field(default_factory=now)


class Command(Message):
    """Intent decoded from EEG (``openmind/v1/commands``)."""

    KIND: ClassVar[Kind] = "command"

    command: CommandName
    confidence: Unit
    user: UserId
    seq: Annotated[int, Field(ge=0)]


class Status(Message):
    """Decoder heartbeat (``openmind/v1/status``), at least every 0.5 s."""

    KIND: ClassVar[Kind] = "status"

    state: DecoderState
    model: str
    user: UserId | None
    assistance: Unit | None = None
    message: str | None = None


class Marker(Message):
    """Calibration instruction or scene event (``openmind/v1/markers``)."""

    KIND: ClassVar[Kind] = "marker"

    type: MarkerType
    label: Annotated[str, Field(min_length=1)]
    trial: Annotated[int, Field(ge=1)] | None
    scene: str | None


class Event(Message):
    """Object telemetry or event (``openmind/v1/events``)."""

    KIND: ClassVar[Kind] = "event"

    object: Annotated[str, Field(min_length=1)]
    kind: Annotated[str, Field(min_length=1)]
    data: dict[str, Any]


class Score(Message):
    """User progress (``openmind/v1/score``)."""

    KIND: ClassVar[Kind] = "score"

    user: UserId
    overall: Unit
    ready: bool
    per_command: dict[CommandName, Unit]
    false_start_rate: Unit
    stable_blocks: Annotated[int, Field(ge=0)]
    ece: Unit | None = None


NonNegative = Annotated[float, Field(ge=0.0)]


class ChannelPower(BaseModel):
    """Band power of one electrode in µV²/Hz; ``rel_*`` relative to the rest baseline."""

    model_config = _CONFIG

    label: Annotated[str, Field(min_length=1)]
    alpha: NonNegative | None
    mu: NonNegative | None
    beta: NonNegative | None
    rel_mu: float | None
    rel_beta: float | None
    ok: bool


class BandPower(Message):
    """Power per electrode and band (``openmind/v1/bandpower``), about 10 Hz."""

    KIND: ClassVar[Kind] = "bandpower"

    user: UserId | None
    srate_hz: Annotated[float, Field(gt=0.0)]
    bands: Annotated[
        dict[str, tuple[NonNegative, Annotated[float, Field(gt=0.0)]]], Field(min_length=1)
    ]
    channels: Annotated[list[ChannelPower], Field(min_length=1)]

    def channel(self, label: str) -> ChannelPower:
        """Return the entry of ``label``; raise ``KeyError`` if absent."""
        for entry in self.channels:
            if entry.label == label:
                return entry
        raise KeyError(label)


#: Model class of each message kind.
MODELS: Final[dict[Kind, type[Message]]] = {
    model.KIND: model for model in (Command, Status, Marker, Event, Score, BandPower)
}
