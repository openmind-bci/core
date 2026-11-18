# OpenMind protocol

**Version 1.0.0** — MQTT topics prefix `openmind/v1/`, messages field `"v": 1`.

This document is the contract between the decoder (OpenMind Core) and everything
that listens to it or talks to it: simulators, the Godot cockpit, the ESP32 robot,
gateways. The normative definitions are the JSON Schemas in
[`protocol/schemas/v1/`](../protocol/schemas/v1); the
[`openmind-protocol`](../protocol) Python package implements them, and
[`protocol/examples/`](../protocol/examples) holds valid and invalid examples.

The key words **MUST**, **SHOULD** and **MAY** are used as in RFC 2119.

## 1. Transport

- MQTT 3.1.1 or 5, any broker (the development broker is Mosquitto on
  `localhost:1883`, WebSocket on `9001`).
- Payloads are **UTF-8 JSON objects**, compact, one message per MQTT message.
  `NaN` and `Infinity` are not JSON and **MUST NOT** be sent.
- Every message carries `"v": 1` (major version) and `"ts"`: the Unix time in
  seconds (UTC, float) at which it was created. `ts` is informative: devices
  without a synchronized clock **MUST NOT** rely on it for safety decisions.
- Users are identified only by pseudonyms matching `^P[0-9]{2,}$` (`P01`, `P03`...).
  A real name **MUST NOT** appear in any message.

## 2. Topics

| Topic | Message | Publisher | Rate | Suggested QoS |
|---|---|---|---|---|
| `openmind/v1/commands` | `command` | core | on change, bounded | 0 |
| `openmind/v1/status` | `status` | core | **≥ 2 Hz** | 0 |
| `openmind/v1/markers` | `marker` | sandbox scenes | on event | 1 |
| `openmind/v1/events` | `event` | objects | free | 0 |
| `openmind/v1/score` | `score` | core | per evaluation block | 1 |
| `openmind/v1/bandpower` | `bandpower` | core | ≈ 10 Hz | 0 |

## 3. Messages

### `command`

```json
{"v":1,"command":"left","confidence":0.86,"ts":1763478000.25,"user":"P03","seq":41}
```

| Field | Type | Meaning |
|---|---|---|
| `command` | `left` \| `right` \| `forward` \| `rest` \| `action` | decoded intent |
| `confidence` | number 0-1 | probability after evidence accumulation |
| `user` | pseudonym | user whose EEG produced the command |
| `seq` | integer ≥ 0 | +1 per command message of this publisher |

The decoder only says what the user intends (`left`), never what an object should
do with it. `rest` is the default. `action` comes from a **double blink**: an eye
signal (EOG) captured by the frontal electrode, not a brain signal. It makes the
interface hybrid, and is documented as such everywhere.

### `status`

```json
{"v":1,"state":"running","ts":1763478000.25,"model":"baseline-riemann","user":"P03"}
```

| Field | Type | Meaning |
|---|---|---|
| `state` | `idle` \| `calibrating` \| `running` \| `degraded` | only `running` allows motion |
| `model` | string | model in use, empty if none |
| `user` | pseudonym or `null` | current profile |
| `assistance` | number 0-1, optional | help given during guided calibration (1 = fully assisted) |
| `message` | string, optional | human-readable detail (e.g. why `degraded`) |

### `marker`

```json
{"v":1,"type":"cue","label":"left","ts":1763478000.25,"trial":12,"scene":"calibration"}
```

`type` is `cue`, `trial_start`, `trial_end` or `event`; for cues, `label` is the
expected command. `trial` counts from 1 (or `null`), `scene` names the emitter (or
`null`). Scenes also relay markers to the LSL markers stream, so EEG and cues
share the LSL clock.

### `event`

```json
{"v":1,"object":"robot","kind":"telemetry","data":{"speed_mps":0.12},"ts":1763478000.25}
```

Free telemetry: `data` is an object whose content is defined per (`object`, `kind`).

### `score`

```json
{"v":1,"ts":1763478000.25,"user":"P03","overall":0.78,"ready":true,
 "per_command":{"left":0.81,"right":0.76,"rest":0.9},"false_start_rate":0.06,
 "stable_blocks":2,"ece":0.04}
```

`ready` is true when every threshold is met (by default ≥ 70 % per command,
≤ 10 % false starts, 2 consecutive blocks): the user may drive real objects.

### `bandpower`

```json
{"v":1,"ts":1763478000.25,"user":"P03","srate_hz":250,
 "bands":{"alpha":[8,12],"mu":[8,13],"beta":[13,30]},
 "channels":[{"label":"C4","alpha":5.0,"mu":6.3,"beta":2.1,"rel_mu":-0.42,"rel_beta":-0.21,"ok":true},
             {"label":"Fp1","alpha":null,"mu":null,"beta":null,"rel_mu":null,"rel_beta":null,"ok":false}]}
```

Power in µV²/Hz per channel and band. `rel_*` is the relative change from the
resting baseline of the profile (negative = desynchronization), `null` before a
baseline exists. `ok: false` marks an absent, flat or saturated channel, whose
values may be `null`.

## 4. Safety rules for objects

Every object that moves something real (robot, wheelchair, actuator) **MUST**:

1. **Stop when the decoder goes silent:** stop within 2 s of the last `status`
   received, measured with the object's own clock at reception, and stop
   immediately when the last `status` is not `running`.
2. **Apply its own confidence threshold:** ignore commands whose `confidence` is
   below a threshold chosen for that object (stricter for real objects).
3. **Ignore stale commands:** ignore a command whose `seq` is not greater than
   the last one applied (duplicates, out-of-order delivery). After a decoder
   restart, `seq` restarts from 0: objects reset their counter when `status`
   returns after a silence.
4. **Fail safe:** on a malformed message, an unknown major version or a broker
   disconnection, stop and wait.

`openmind_protocol.Watchdog` implements rules 1 to 3 in Python:

```python
from openmind_protocol import Watchdog, decode_topic, Command, Status

dog = Watchdog(min_confidence=0.8)


def on_message(topic: str, payload: bytes) -> None:
    message = decode_topic(topic, payload)  # raises ProtocolError if invalid
    if isinstance(message, Status):
        dog.on_status(message)
    elif isinstance(message, Command) and dog.accept(message):
        drive(message.command)


def control_loop() -> None:
    if not dog.alive:
        stop_motors()
```

Firmware in other languages (C++ on the ESP32) **MUST** implement the same rules.

## 5. Versioning and compatibility

The protocol follows [Semantic Versioning](https://semver.org); the major version
is in the topic prefix (`v1`) and in every message (`"v": 1`).

- **Consumers MUST ignore unknown fields**, so a minor version can add optional
  fields without breaking anyone. Decoders of `openmind-protocol` do so.
- **Consumers MUST reject an unknown major version** (`UnsupportedVersionError`).
- **patch** (1.0.x): wording and examples only.
- **minor** (1.x.0): new optional fields, new message kinds or topics.
- **major** (x.0.0): anything else (renamed or removed fields, new required
  fields, new values in an existing enum that old consumers would reject). A new
  major version uses new topics (`openmind/v2/...`) and may run alongside v1.

Changes are made in this repository first, with a version bump and a `CHANGELOG`
entry, then in sandbox.

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-11-18 | First version. |

## 6. Using the Python package

```python
from openmind_protocol import Command, decode, encode, topics

payload = encode(Command(command="left", confidence=0.86, user="P03", seq=41))
client.publish(topics.COMMANDS, payload)  # paho-mqtt client

message = decode(payload, Command)  # or decode_topic(topic, payload)
```

- Building a model validates it; `v` and `ts` (current time) get defaults.
- `encode()` checks the result against the JSON Schema.
- `decode()` rejects invalid JSON (including `NaN`), schema violations and other
  major versions with a `ProtocolError` whose `problems` name each faulty field
  (`confidence: 1.5 is greater than the maximum of 1`).
