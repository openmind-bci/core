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
uv sync --extra research                    # add --extra deep for deep models
uv run python research/experiments/benchmark.py --config research/configs/baselines.yaml
uv run python research/experiments/benchmark.py --config research/configs/deep.yaml
```

A configuration lists the datasets, subjects and classes (`experiments`), the
`models` (any name accepted by `openmind_core.models.make_model`), their
training settings (`model_params`), the protocols, the seed, and the deep
models to pre-train on all subjects (`pretrain`). The script:

- caches preprocessed epochs in `$OPENMIND_DATA_DIR/cache/epochs/`; the cache key
  includes the source code of the loader and of the preprocessing, so changing
  them invalidates the cache;
- writes the result tables to the configuration's `output` file
  ([`research/results/baselines.md`](../research/results/baselines.md),
  [`research/results/deep.md`](../research/results/deep.md); committed);
- logs one MLflow run per experiment, with a nested run per model and
  protocol (parameters, per-subject and mean metrics, Git commit, package
  versions, per-fold scores as CSV), to `$OPENMIND_DATA_DIR/mlflow/`. Browse
  them with:

  ```bash
  uv run mlflow ui --backend-store-uri "sqlite:///$HOME/openmind-data/mlflow/mlflow.db"
  ```

- saves the pre-trained deep models to the local registry (see below).

`--only NAME` runs a single experiment without touching the result file;
`--no-tracking` skips MLflow.

## Results (baselines)

Full tables: [`research/results/baselines.md`](../research/results/baselines.md).
Mean accuracy over subjects, 7 decoding channels, 2 s epochs:

| Dataset, classes | Best within-session | Best cross-session | Best cross-subject | Chance |
|---|---|---|---|---|
| BNCI2014-001, left/right (9 subjects) | 76.8 % (`csp-lda`) | 74.8 % (`csp-lda`) | 72.3 % (`ts-lr`) | 50 % |
| BNCI2014-001, left/right/forward (9 subjects) | 68.3 % (`csp-lda`) | 65.6 % (`csp-lda`) | 53.2 % (`csp-lda`) | 33 % |
| PhysionetMI, left/right (16 of subjects 1-20) | 60.5 % (`csp-lda`) | n/a (one session) | 63.1 % (`csp-lda`) | 50 % |

What these numbers say:

- **Orders of magnitude match the literature.** Published MOABB benchmarks
  report roughly 80-85 % within-session for left/right on BNCI2014-001 with all
  22 electrodes and 3-4 s epochs. OpenMind uses 7 electrodes and 2 s epochs, the
  constraints of a light headset and of a responsive online decoder; a few
  points lower is the expected price. As a check, MOABB's own paradigm on
  subject 1 gives 85 % with 22 channels and 83-87 % with our 7 channels.
- **Subjects differ a lot** (standard deviation around 15 points): some reach
  90 %, others stay near chance. This is the well-known variability of motor
  imagery, and the reason why calibration and progress feedback matter.
- **The `forward` (feet) class costs about 9 points** within-session, and much
  more across subjects: feet imagery is weaker and less lateralized.
- **A new user without calibration loses 4-15 points** compared with a
  calibrated one (cross-subject vs within-session on BNCI2014-001). Closing this
  gap with a few minutes of data is the goal of `CORE-050` and `CORE-060`.
- **`mdm` collapses across subjects** (61 % and 43 %): every head shifts the
  covariance matrices, and MDM compares them directly. Riemannian re-centering
  (`CORE-050`) targets exactly this.
- **On PhysionetMI, other people's data beats a short calibration**: with about
  35 epochs per subject, cross-subject training (on about 600 epochs) does as
  well as or better than within-session training. Little personal data plus a lot
  of pooled data is the case for transfer learning.
- PhysionetMI is noisier: the 150 µV rejection leaves subjects 9, 13, 15 and 17
  with fewer than 10 epochs per class, so they are excluded from every protocol.
