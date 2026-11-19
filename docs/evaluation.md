# Evaluation and baselines

Every decoder of OpenMind Core is compared with the same classical baselines,
under the same protocols, on the same preprocessed epochs
([pipeline.md](pipeline.md)). New models (deep learning, alignment, fast
calibration) must beat these numbers to be worth their cost.

## Baselines

`openmind_core.models` provides three scikit-learn pipelines. Each one takes
epochs `(n_epochs, n_channels, n_times)` in µV and starts by estimating one
spatial covariance matrix per epoch, with the OAS shrinkage estimator (well
conditioned on 2 s epochs and 7 channels).

| Name | Pipeline | Why it is here |
|---|---|---|
| `csp-lda` | Common Spatial Patterns (log-variance of 6 filters) → shrinkage LDA | The historical reference of motor-imagery BCIs |
| `mdm` | Minimum Distance to Riemannian Mean | No hyper-parameter, robust; basis of Riemannian re-centering |
| `ts-lr` | Tangent space at the Riemannian mean → standardized L2 logistic regression | Usually the strongest classical decoder |

```python
from openmind_core.models import make_baseline

model = make_baseline("ts-lr").fit(epochs.X, epochs.y)
model.predict(new_epochs.X)  # "left", "right", "forward"
```

## Protocols

`openmind_core.evaluation` implements the three MOABB protocols on
`EpochSet` objects:

| Protocol | Train on | Test on | Question answered |
|---|---|---|---|
| `within-session` | 4/5 of a session (stratified 5-fold, seeded) | the remaining fifth | How good can a decoder be for this person, today? (optimistic) |
| `cross-session` | the other session(s) of the subject | one session | Does a model survive to the next day? |
| `cross-subject` | every other subject (leave one subject out) | all epochs of the held-out subject | How good is a model for a **new user with no calibration**? |

The cross-subject score is the starting point of fast calibration (`CORE-060`):
the goal of OpenMind is to bring a new user from that score to the
within-session score with a few minutes of calibration.

Scores are accuracy, balanced accuracy and Cohen's kappa (0 at chance, the BCI
Competition IV metric). Summaries average sessions within each subject first,
then report the mean and standard deviation over subjects, so every subject
weighs the same.

### Why not MOABB's evaluation classes?

MOABB's `WithinSessionEvaluation`, `CrossSessionEvaluation` and
`CrossSubjectEvaluation` re-run their own preprocessing (zero-phase filtering,
their own epoching). OpenMind trains on **causally** filtered epochs, exactly as
the online decoder sees them, so the protocols are reimplemented on top of
`load_epochs()`. The datasets, the splits and the metrics are the same as in
MOABB, which keeps the numbers comparable with the literature.

## Running the benchmark

```bash
uv sync --extra research
uv run python research/experiments/baselines.py
```

The configuration (datasets, subjects, classes, pipelines, protocols, seed) is
in `research/configs/baselines.yaml`. The script:

- caches preprocessed epochs in `$OPENMIND_DATA_DIR/cache/epochs/`; the cache key
  includes the source code of the loader and of the preprocessing, so changing
  them invalidates the cache;
- writes the result tables to
  [`research/results/baselines.md`](../research/results/baselines.md)
  (committed);
- logs one MLflow run per experiment, with a nested run per pipeline and
  protocol (parameters, per-subject and mean metrics, Git commit, package
  versions, per-fold scores as CSV), to `$OPENMIND_DATA_DIR/mlflow/`. Browse
  them with:

  ```bash
  uv run mlflow ui --backend-store-uri "sqlite:///$HOME/openmind-data/mlflow/mlflow.db"
  ```

`--only NAME` runs a single experiment without touching the result file;
`--no-tracking` skips MLflow.
