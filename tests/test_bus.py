"""Tests of the MQTT publisher with a fake client (no broker needed)."""

from __future__ import annotations

from typing import Any, cast

import paho.mqtt.client as mqtt
import pytest

from openmind_core.bus import BrokerUnavailableError, Publisher, parse_broker
from openmind_protocol import Command, Marker, Score, Status, decode


class _Reason:
    def __init__(self, failure: bool = False) -> None:
        self.is_failure = failure


class FakeClient:
    def __init__(self, accept: bool = True, refuse: bool = False) -> None:
        self.accept = accept
        self.refuse = refuse
        self.on_connect: Any = None
        self.on_disconnect: Any = None
        self.published: list[tuple[str, bytes, int]] = []
        self.calls: list[str] = []

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        self.calls.append("reconnect_delay_set")

    def connect(self, host: str, port: int, keepalive: int) -> None:
        if self.refuse:
            raise ConnectionRefusedError("refused")
        self.calls.append(f"connect {host}:{port}")
        if self.accept:
            self.on_connect(self, None, None, _Reason(), None)

    def loop_start(self) -> None:
        self.calls.append("loop_start")

    def loop_stop(self) -> None:
        self.calls.append("loop_stop")

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def publish(self, topic: str, payload: bytes, qos: int = 0) -> None:
        self.published.append((topic, payload, qos))


def _publisher(client: FakeClient) -> Publisher:
    return Publisher("broker", 1884, client=cast(mqtt.Client, client))


def test_parse_broker() -> None:
    assert parse_broker("localhost:1883") == ("localhost", 1883)
    assert parse_broker("broker.local") == ("broker.local", 1883)
    assert parse_broker(" 192.168.1.5:9000 ") == ("192.168.1.5", 9000)


@pytest.mark.parametrize("bad", ["", ":1883", "host:abc", "host:0", "host:70000"])
def test_parse_broker_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="broker"):
        parse_broker(bad)


def test_connect_publish_close() -> None:
    client = FakeClient()
    publisher = _publisher(client)
    publisher.connect()
    assert publisher.connected
    publisher.publish(Command(command="left", confidence=0.8, user="P01", seq=1))
    topic, payload, qos = client.published[0]
    assert topic == "openmind/v1/commands"
    assert decode(payload, Command).command == "left"
    assert qos == 0
    publisher.close()
    assert not publisher.connected
    assert client.calls[-2:] == ["disconnect", "loop_stop"]
    assert "connect broker:1884" in client.calls


def test_qos_per_kind() -> None:
    client = FakeClient()
    publisher = _publisher(client)
    publisher.connect()
    publisher.publish(Status(state="running", model="baseline", user="P01"))
    publisher.publish(Marker(type="cue", label="left", trial=1, scene="calibration"))
    publisher.publish(
        Score(
            user="P01",
            overall=0.5,
            ready=False,
            per_command={"left": 0.5},
            false_start_rate=0.1,
            stable_blocks=0,
        )
    )
    assert [qos for _, _, qos in client.published] == [0, 1, 1]


def test_disconnect_callback_clears_connected() -> None:
    client = FakeClient()
    publisher = _publisher(client)
    publisher.connect()
    client.on_disconnect(client, None, None, _Reason(), None)
    assert not publisher.connected
    client.on_connect(client, None, None, _Reason(), None)
    assert publisher.connected


def test_refused_connection_is_explained() -> None:
    publisher = _publisher(FakeClient(refuse=True))
    with pytest.raises(BrokerUnavailableError, match=r"docker-compose\.yml"):
        publisher.connect()


def test_no_answer_times_out_and_cleans_up() -> None:
    client = FakeClient(accept=False)
    publisher = _publisher(client)
    with pytest.raises(BrokerUnavailableError, match="no answer"):
        publisher.connect(timeout=0.05)
    assert "loop_stop" in client.calls


def test_failed_connack_does_not_connect() -> None:
    client = FakeClient(accept=False)
    publisher = _publisher(client)
    client.on_connect(client, None, None, _Reason(failure=True), None)
    assert not publisher.connected


def test_context_manager() -> None:
    client = FakeClient()
    with _publisher(client) as publisher:
        assert publisher.connected
    assert client.calls[-1] == "loop_stop"
