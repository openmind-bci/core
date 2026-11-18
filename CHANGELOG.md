# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
`openmind-core` and `openmind-protocol` are versioned separately; protocol changes
are listed under their own heading.

## [Unreleased]

### Protocol 1.0.0 (`openmind-protocol`)

- JSON Schemas v1 for `command`, `status`, `marker`, `event`, `score` and
  `bandpower`, with valid and invalid examples.
- Topics `openmind/v1/commands|status|markers|events|score|bandpower`.
- Validated pydantic models, `encode()` / `decode()` / `decode_topic()` with
  explicit `ProtocolError`s and rejection of other major versions.
- `Watchdog` implementing the safety rules for objects (2 s heartbeat timeout,
  confidence threshold, sequence order). Specification in `docs/protocol.md`.

### Added

- Preprocessing shared by offline and online decoding: causal 8-30 Hz band-pass
  and notch (`StreamFilter`, identical at once or block by block), 2 s epochs
  from cue + 0.5 s, 2 s sliding windows every 250 ms, 150 µV amplitude rejection.
- Double-blink detector on Fp1 (an eye signal, for the `action` command).
- Public datasets BNCI2014-001 and PhysionetMI through MOABB, restricted to the
  montage A decoding channels, as `EpochSet` objects; `research/experiments/
  make_epochs.py` caches epochs in `$OPENMIND_DATA_DIR`. See `docs/pipeline.md`.
- Repository skeleton: `openmind-core` package with an `openmind-core` command,
  `openmind-protocol` package in `protocol/` (uv workspace), optional extras
  `research` and `plug`, `research/` area.
- Tooling: ruff, strict mypy on `src/` and `protocol/src/`, pytest, pre-commit with
  nbstripout, GitHub Actions CI (Linux, macOS, Windows × Python 3.11, 3.12) building
  both packages, and a manual workflow for slow tests.
