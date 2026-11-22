"""Publishing protocol messages to the MQTT broker (paho-mqtt 2).

Messages are encoded with :func:`openmind_protocol.encode`, which validates them
against the JSON Schema, and sent on the topic of their kind.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from openmind_protocol import Message, encode, topic_of

#: QoS per message kind (suggested QoS of the protocol); other kinds use 0.
QOS: dict[str, int] = {"marker": 1, "score": 1}

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883


class BrokerUnavailableError(ConnectionError):
    """Raised when the MQTT broker cannot be reached."""


def parse_broker(value: str) -> tuple[str, int]:
    """Split ``"host:port"`` (or just ``"host"``) into ``(host, port)``.

    Raises:
        ValueError: If the host is empty or the port is not in 1-65535.
    """
    host, separator, port_text = value.strip().rpartition(":")
    if not separator:
        host, port_text = port_text, ""
    if not host:
        raise ValueError(f"invalid broker {value!r}: expected host[:port]")
    if not port_text:
        return host, DEFAULT_PORT
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"invalid broker port in {value!r}") from None
    if not 0 < port < 65536:
        raise ValueError(f"invalid broker port in {value!r}: must be 1-65535")
    return host, port


class Publisher:
    """Connection to an MQTT broker that publishes protocol messages.

    The network loop runs in a background thread and reconnects automatically
    after a connection loss; a message published while disconnected is lost
    (QoS 0) or queued by the client (QoS 1).

    Args:
        host: Broker host name.
        port: Broker TCP port.
        client_id_prefix: Prefix of the client id (a random suffix keeps it unique).
        client: Client to use instead of a new paho client (for tests).
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        client_id_prefix: str = "openmind-core",
        client: mqtt.Client | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._connected = threading.Event()
        self._client = client or mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"{client_id_prefix}-{uuid.uuid4().hex[:8]}",
        )
        self._client.on_connect = self._handle_connect
        self._client.on_disconnect = self._handle_disconnect

    @property
    def connected(self) -> bool:
        """Whether the publisher is currently connected to the broker."""
        return self._connected.is_set()

    def connect(self, timeout: float = 5.0) -> None:
        """Connect and start the network thread.

        Raises:
            BrokerUnavailableError: If the broker does not accept the connection
                within ``timeout`` seconds.
        """
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        try:
            self._client.connect(self.host, self.port, keepalive=10)
        except OSError as error:
            raise BrokerUnavailableError(self._unreachable(str(error))) from error
        self._client.loop_start()
        if not self._connected.wait(timeout):
            self.close()
            raise BrokerUnavailableError(self._unreachable("no answer"))

    def publish(self, message: Message) -> None:
        """Publish a message on its topic (validated against its schema)."""
        self._client.publish(topic_of(message), encode(message), qos=QOS.get(message.KIND, 0))

    def close(self) -> None:
        """Disconnect and stop the network thread."""
        self._client.disconnect()
        self._client.loop_stop()
        self._connected.clear()

    def __enter__(self) -> Publisher:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _unreachable(self, reason: str) -> str:
        return (
            f"cannot reach the MQTT broker at {self.host}:{self.port} ({reason}). "
            "Start the development broker with "
            "`docker compose -f sandbox/dev/docker-compose.yml up -d`, "
            "or pass --broker host:port."
        )

    def _handle_connect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason: Any, properties: Any
    ) -> None:
        if not reason.is_failure:
            self._connected.set()

    def _handle_disconnect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason: Any, properties: Any
    ) -> None:
        self._connected.clear()
