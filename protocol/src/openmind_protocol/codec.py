"""Encode and decode protocol messages, with validation and explicit errors."""

from __future__ import annotations

import json
from typing import Any, TypeVar, overload

from pydantic import ValidationError

from openmind_protocol.messages import MODELS, VERSION, Message
from openmind_protocol.schemas import schema_errors
from openmind_protocol.topics import TOPICS, Kind, kind_of

M = TypeVar("M", bound=Message)


class ProtocolError(ValueError):
    """Raised when a payload is not a valid protocol message.

    Attributes:
        kind: Expected message kind.
        problems: One line per problem, prefixed with the field path.
    """

    def __init__(self, kind: str, problems: list[str]) -> None:
        self.kind = kind
        self.problems = problems
        super().__init__(f"invalid {kind} message: " + "; ".join(problems))


class UnsupportedVersionError(ProtocolError):
    """Raised when a message carries a major version this package does not support."""


def encode(message: Message) -> bytes:
    """Serialize a message as compact UTF-8 JSON, checked against its schema.

    Raises:
        ProtocolError: If the message does not match its schema (cannot happen for
            messages built through the models, but guards against drift).
    """
    document = message.model_dump(mode="json")
    problems = schema_errors(message.KIND, document)
    if problems:
        raise ProtocolError(message.KIND, problems)
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def topic_of(message: Message) -> str:
    """MQTT topic on which ``message`` is published."""
    return TOPICS[message.KIND]


@overload
def decode(payload: bytes | str, model: type[M]) -> M: ...
@overload
def decode(payload: bytes | str, model: Kind) -> Message: ...
def decode(payload: bytes | str, model: type[M] | Kind) -> M | Message:
    """Parse and validate a payload as a given message model or kind.

    Unknown fields are ignored (forward compatibility with minor versions).

    Raises:
        UnsupportedVersionError: If ``v`` is not the supported major version.
        ProtocolError: If the payload is not valid JSON or breaks the schema.
    """
    cls: type[Message] = MODELS[model] if isinstance(model, str) else model
    kind = cls.KIND
    document = _parse(kind, payload)
    if isinstance(document, dict) and "v" in document and document["v"] != VERSION:
        raise UnsupportedVersionError(
            kind, [f"v: version {document['v']!r} is not supported, expected {VERSION}"]
        )
    problems = schema_errors(kind, document)
    if problems:
        raise ProtocolError(kind, problems)
    try:
        return cls.model_validate_json(json.dumps(document))
    except ValidationError as error:  # schema and model disagree: report it clearly
        raise ProtocolError(kind, _pydantic_problems(error)) from error


def decode_topic(topic: str, payload: bytes | str) -> Message:
    """Decode a payload received on an OpenMind topic.

    Raises:
        KeyError: If ``topic`` is not an OpenMind v1 topic.
        ProtocolError: If the payload is invalid for that topic.
    """
    return decode(payload, kind_of(topic))


def _parse(kind: str, payload: bytes | str) -> Any:
    def reject_constant(name: str) -> Any:
        raise ValueError(f"{name} is not valid JSON")

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(text, parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError) as error:
        raise ProtocolError(kind, [f"(root): not valid UTF-8 JSON ({error})"]) from error


def _pydantic_problems(error: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(p) for p in item['loc']) or '(root)'}: {item['msg']}"
        for item in error.errors()
    ]
