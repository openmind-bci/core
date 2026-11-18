"""OpenMind BCI command protocol: topics, schemas and validated messages.

Example::

    from openmind_protocol import Command, decode, encode, topics

    payload = encode(Command(command="left", confidence=0.86, user="P03", seq=41))
    client.publish(topics.COMMANDS, payload)

    message = decode(payload, Command)
"""

from importlib.metadata import PackageNotFoundError, version

from openmind_protocol import topics
from openmind_protocol.codec import (
    ProtocolError,
    UnsupportedVersionError,
    decode,
    decode_topic,
    encode,
    topic_of,
)
from openmind_protocol.messages import (
    COMMANDS,
    MODELS,
    VERSION,
    BandPower,
    ChannelPower,
    Command,
    CommandName,
    DecoderState,
    Event,
    Marker,
    Message,
    Score,
    Status,
)
from openmind_protocol.safety import STATUS_MIN_RATE_HZ, STATUS_TIMEOUT_S, Watchdog
from openmind_protocol.schemas import load_schema, schema_errors

try:
    __version__ = version("openmind-protocol")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

#: Version of the protocol specification implemented by this package.
PROTOCOL_VERSION = "1.0.0"

__all__ = [
    "COMMANDS",
    "MODELS",
    "PROTOCOL_VERSION",
    "STATUS_MIN_RATE_HZ",
    "STATUS_TIMEOUT_S",
    "VERSION",
    "BandPower",
    "ChannelPower",
    "Command",
    "CommandName",
    "DecoderState",
    "Event",
    "Marker",
    "Message",
    "ProtocolError",
    "Score",
    "Status",
    "UnsupportedVersionError",
    "Watchdog",
    "__version__",
    "decode",
    "decode_topic",
    "encode",
    "load_schema",
    "schema_errors",
    "topic_of",
    "topics",
]
