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

The topics, schemas and models are being defined; see the
[changelog](https://github.com/openmind-bci/core/blob/main/CHANGELOG.md).

License: Apache-2.0.
