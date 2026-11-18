# Research

Experiments, benchmarks and notebooks: the data-science side of OpenMind Core.

- `configs/`: YAML configurations of experiments (seeds, datasets, pipelines).
- `experiments/`: scripts that run an evaluation and log it to MLflow.
- `notebooks/`: exploration; committed **without outputs** (nbstripout runs in pre-commit).

Rules:

- Nothing in `research/` is imported by `src/`. Reusable code moves to
  `src/openmind_core/` with tests.
- Results must be reproducible: fixed seeds, configuration files, MLflow runs
  recording code and data versions.
- Datasets, MLflow runs and model weights stay in `OPENMIND_DATA_DIR`, never in Git.

Install the research tools with `uv sync --extra research`.
