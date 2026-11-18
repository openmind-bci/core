"""MQTT topics of protocol version 1."""

from __future__ import annotations

from typing import Final, Literal

#: Prefix of every topic of this major version.
PREFIX: Final = "openmind/v1"

Kind = Literal["command", "status", "marker", "event", "score", "bandpower"]

COMMANDS: Final = f"{PREFIX}/commands"
STATUS: Final = f"{PREFIX}/status"
MARKERS: Final = f"{PREFIX}/markers"
EVENTS: Final = f"{PREFIX}/events"
SCORE: Final = f"{PREFIX}/score"
BANDPOWER: Final = f"{PREFIX}/bandpower"

#: Topic of each message kind.
TOPICS: Final[dict[Kind, str]] = {
    "command": COMMANDS,
    "status": STATUS,
    "marker": MARKERS,
    "event": EVENTS,
    "score": SCORE,
    "bandpower": BANDPOWER,
}

#: Message kind carried by each topic.
KINDS: Final[dict[str, Kind]] = {topic: kind for kind, topic in TOPICS.items()}


def kind_of(topic: str) -> Kind:
    """Return the message kind carried by ``topic``.

    Raises:
        KeyError: If the topic is not an OpenMind v1 topic.
    """
    try:
        return KINDS[topic]
    except KeyError:
        raise KeyError(f"{topic!r} is not an OpenMind v1 topic; known: {sorted(KINDS)}") from None
