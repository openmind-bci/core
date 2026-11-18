# OpenMind protocol

The command protocol of [OpenMind BCI](https://github.com/openmind-bci): MQTT topics,
JSON Schemas and validated Python models for the messages exchanged between the
decoder ([core](https://github.com/openmind-bci/core)) and the objects it drives
([sandbox](https://github.com/openmind-bci/sandbox)).

> **Research and experimentation project — not a medical device.**

```bash
pip install openmind-protocol
```

Lightweight on purpose: it depends only on `pydantic` and `jsonschema`, so any
object (a Python simulator, a gateway, a test script) can use it. Schemas are
plain JSON, usable from other languages (C++ on ESP32, GDScript in Godot).

```python
from openmind_protocol import Command, Watchdog, decode_topic, encode, topics

payload = encode(Command(command="left", confidence=0.86, user="P03", seq=41))
client.publish(topics.COMMANDS, payload)
```

| Topic | Message |
|---|---|
| `openmind/v1/commands` | `Command`: `left`, `right`, `forward`, `rest`, `action` with confidence |
| `openmind/v1/status` | `Status`: decoder heartbeat, ≥ 2 Hz |
| `openmind/v1/markers` | `Marker`: calibration cues and scene events |
| `openmind/v1/events` | `Event`: object telemetry |
| `openmind/v1/score` | `Score`: user progress |
| `openmind/v1/bandpower` | `BandPower`: power per electrode and band |

Objects that move something real **must** stop when no `status` arrived for 2 s;
`Watchdog` implements the safety rules. The full specification, safety rules and
versioning policy are in
[docs/protocol.md](https://github.com/openmind-bci/core/blob/main/docs/protocol.md).
JSON Schemas: `schemas/v1/`; examples: `examples/valid/` and `examples/invalid/`.

License: Apache-2.0.
