# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
`openmind-core` and `openmind-protocol` are versioned separately; protocol changes
are listed under their own heading.

## [Unreleased]

### Added

- Repository skeleton: `openmind-core` package with an `openmind-core` command,
  `openmind-protocol` package in `protocol/` (uv workspace), optional extras
  `research` and `plug`, `research/` area.
- Tooling: ruff, strict mypy on `src/` and `protocol/src/`, pytest, pre-commit with
  nbstripout, GitHub Actions CI (Linux, macOS, Windows × Python 3.11, 3.12) building
  both packages, and a manual workflow for slow tests.
