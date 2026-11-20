# OpenMind Core

[![CI](https://github.com/openmind-bci/core/actions/workflows/ci.yml/badge.svg)](https://github.com/openmind-bci/core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The AI core of OpenMind BCI.** Core listens to a standard EEG stream,
recognizes what the user intends (imagined left hand, right hand, feet, rest,
or a double jaw clench), and publishes **generic commands** over MQTT. A new user
should be calibrated in minutes thanks to transfer learning.

> [!WARNING]
> **Research and experimentation project — not a medical device.**
> OpenMind Core must not be used for diagnosis, treatment or any clinical decision,
> nor to control anything where an error could harm someone.

## Where it fits

```
  headset / file / synthetic          intent decoding                 objects
 ┌──────────────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
 │  plug                    │    │  core (this repo)      │    │  sandbox             │
 │  any EEG source          │───▶│  preprocessing         │───▶│  robot, digital twin │
 │  → standard LSL stream   │LSL │  decoder + calibration │MQTT│  wheelchair, bird... │
 └──────────────────────────┘    │  → OpenMind commands   │    └──────────────────────┘
                                 └────────────────────────┘
```

| Repository | Role |
|---|---|
| [plug](https://github.com/openmind-bci/plug) | EEG source → standard LSL stream ([stream specification](https://github.com/openmind-bci/plug/blob/main/docs/stream-spec.md)) |
| **core** (this one) | LSL stream → user intent → MQTT commands |
| [sandbox](https://github.com/openmind-bci/sandbox) | Virtual and real objects driven by those commands |

Core never talks to a headset (that is plug) and knows nothing about objects (that
is sandbox): the decoder says `left`, never "turn the robot's wheel".

## Commands

| Command | Mental task |
|---|---|
| `left` | imagine moving the left hand |
| `right` | imagine moving the right hand |
| `forward` | imagine walking / moving the feet |
| `rest` | relax (default when nothing is detected) |
| `action` | double jaw clench (clench, release at least 0.3 s, clench): a **muscle** signal (EMG), not a brain signal, used as a hybrid switch |

The message format is defined by the **OpenMind protocol**, published from this
repository as the [`openmind-protocol`](protocol/) package.

## Status

Early development (pre-alpha): repository skeleton. See [CHANGELOG.md](CHANGELOG.md).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/openmind-bci/core.git
cd core
uv sync                      # core + protocol (workspace) + dev tools
uv sync --extra research     # + MOABB and MLflow for experiments
uv sync --extra plug         # + openmind-plug, to reuse its LSL inlet
uv run openmind-core --help
```

## Repository layout

| Path | Content |
|---|---|
| `src/openmind_core/` | the decoder: I/O, preprocessing, models, calibration, online loop |
| `protocol/` | the command protocol (`openmind-protocol` package, the contract with sandbox) |
| `research/` | experiments, benchmarks and notebooks; never imported by `src/` |
| `tests/` | tests of `openmind-core` |

## Data and privacy

EEG data are health data. Core processes everything **locally**. Datasets,
profiles and model weights live outside the repository, in `OPENMIND_DATA_DIR`
(default `~/openmind-data`), and users are identified only by pseudonyms
(`P01`, `P02`...).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
