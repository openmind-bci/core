# Contributing to OpenMind Core

Thanks for your interest! Bug reports, new decoders, benchmarks and documentation
improvements are all welcome.

## Development setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/openmind-bci/core.git
cd plug
uv sync                       # creates .venv with runtime + dev dependencies
uv run pre-commit install     # run the checks automatically before each commit
```

`uv sync` installs `openmind-core` and the `openmind-protocol` workspace package
in editable mode. Optional extras:

```bash
uv sync --extra research     # MOABB, MLflow
uv sync --extra plug         # openmind-plug, to reuse its LSL inlet
```

## Checks

Every change must pass:

```bash
uv run ruff check             # lint
uv run ruff format --check    # formatting
uv run mypy                   # strict typing on src/ and protocol/src/
uv run pytest -m "not slow"   # fast tests
```

Tests marked `slow` download public datasets or run for a long time; run them
with `uv run pytest -m slow`.

## Guidelines

- Code, comments, docstrings and docs are written in **English**.
- Public functions and classes are typed and documented.
- New behavior comes with tests.
- Offline and online preprocessing share the **same code**.
- Evaluate across subjects or sessions, never on a random split alone.
- `research/` may be exploratory, but `src/` never imports from it. Notebooks are
  committed without outputs (nbstripout).
- Changes to `protocol/` change the public contract with sandbox: bump the protocol
  version, describe the change in `CHANGELOG.md`, keep minor versions compatible.
- **Never commit EEG data** (`.fif`, `.xdf`, `.edf`, recordings), user profiles,
  model weights (`.pt`, `.ckpt`, `.onnx`) or secrets. Use `OPENMIND_DATA_DIR`.

## Commit messages

One commit per coherent change. The first line is a short sentence in the
imperative mood, capitalized, at most 72 characters, without a final period
or prefix:

```
Add synthetic EEG source
Fix microvolt conversion in replay source
```

Add a blank line and a short body explaining *why* when useful.

## Pull requests

Open a pull request against `main`, describe the change and how you tested it,
and make sure CI is green.
