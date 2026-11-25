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

## Deep models

With the `deep` extra (`uv sync --extra deep`), `openmind_core.models` also
builds two Braindecode networks behind the same scikit-learn interface:

| Name | Network | Why it is here |
|---|---|---|
| `eegnet` | EEGNet (Lawhern et al., 2018): temporal convolution (0.5 s kernel), depthwise spatial filters, separable convolution | Compact (a few thousand weights), the usual starting point for cross-subject transfer |
| `shallow` | ShallowConvNet (Schirrmeister et al., 2017): temporal and spatial convolutions, squaring, pooling, log | A learned, end-to-end filter-bank CSP |

```python
from openmind_core.models import make_model

model = make_model("eegnet", srate=250.0, seed=0).fit(epochs.X, epochs.y)
model.predict_proba(new_epochs.X)  # columns in the order of model.classes_
```

Training (`DeepClassifier`) is a small PyTorch loop, kept explicit for the
fine-tuning of `CORE-060`:

- inputs divided by the per-channel standard deviation of the training set
  (stored with the model, so live data is scaled the same way);
- AdamW (learning rate 0.001, weight decay 0.001), cosine schedule, cross-entropy;
- 20 % of the training epochs held out (stratified) for early stopping: the
  weights with the lowest validation loss are kept;
- seeded initialization, split and batches. On Apple MPS, results can still
  vary slightly from run to run.

The benchmark uses the same recipe for both networks: at most 300 passes,
patience 60, batches of 32. It was chosen on subjects 1 and 3 of BNCI2014-001,
between two settings only (150 passes, patience 30 and batches of 64 left EEGNet
under-trained on a single session of about 90 training epochs). These two
subjects are also part of the evaluation, a small bias to keep in mind.

## Model registry

Trained deep models are saved in `$OPENMIND_DATA_DIR/models/<name>/`, never in
Git:

| File | Content |
|---|---|
| `weights.pt` | the network's `state_dict` only, read back with `torch.load(weights_only=True)` (no pickled code is ever executed) |
| `card.json` | architecture, channels, sampling rate, epoch length, classes, input scale, training settings, training data and benchmark scores, OpenMind version, date |

```python
from openmind_core.models.registry import list_models, load_model, save_model

save_model(model, "eegnet-bnci2014_001-2class", epochs.channels, training={...})
model, card = load_model("eegnet-bnci2014_001-2class")
[card.name for card in list_models()]
```

The benchmark pre-trains every model listed under `pretrain` on all subjects of
each experiment and registers it as `<model>-<experiment>`: these are the
generic models that fast calibration (`CORE-060`) will start from. Only models
trained on public data may ever be published (for instance on Hugging Face,
with a model card built from `card.json`); models fine-tuned on a volunteer's
data stay on the local machine.

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

Long runs can be interrupted (sleep, power, a slowed-down GPU): each finished
experiment is saved to `$OPENMIND_DATA_DIR/cache/benchmark/<config name>/`, and
running the same command again skips it. The checkpoint is tied to the
experiment, the model settings and the Git version of `src/`; changing any of
them recomputes the experiment (`--fresh` forces it). Run long benchmarks under
`caffeinate -i -m -s` on macOS so that the computer does not sleep.

## Calibration curve

The main question of OpenMind: how many calibration trials does a **new user**
need? `openmind_core.calibration` and `research/experiments/calibration_curve.py`
(configuration `research/configs/calibration-curve.yaml`, results in
`research/results/calibration-curve.md`) answer it with the following protocol.

**Scenario.** A new user, one session. For each target subject (leave one subject
out) and each of its sessions separately:

- the **calibration set** is the first `n` trials *per class* of the session, in
  time order (`n` from `n_per_class`, default 0, 5, 10, 20, 40). An `n` the
  session cannot provide while leaving at least 10 test trials per class
  (`min_test_per_class`) is skipped;
- the **test set** is every trial after the last calibration trial. Each `n` has
  its own test set, so the curve alone would be measured on shrinking data. The
  **common test set** therefore starts after the largest feasible `n` of the
  session and is the same for every `n`: the main table uses it, the CSV and the
  second table of each model also keep the accuracy on each `n`'s own test set.

**Source data.** Every trial of every other subject (all sessions), each session
aligned on its own reference (`alignment.method`, default Riemannian) when the
condition uses alignment.

**Conditions.**

| Condition | Model trained on | Target treatment |
|---|---|---|
| `generic` | unaligned source | unaligned, no calibration (a flat line) |
| `aligned` | aligned source | aligned with a reference from its calibration epochs (signals only, labels unused) |
| `aligned+tuned` | aligned source + the `n` calibration trials, weighted `target_weight` (default 10) | same alignment; at `n = 0` identical to `aligned` |
| `scratch` | the `n` calibration trials only (`n` >= 5 per class), unaligned | unaligned |

Weights: `ts-lr` and `mdm` receive them directly (`sample_weight` of the tangent
space, scaler and logistic regression; weighted class means in MDM). `csp-lda`
(its LDA takes no weights) and deep models repeat each calibration trial
`round(target_weight)` times instead; `fit_weighted()` picks the way from the
final estimator. Deep models (`eegnet`) are not part of the first configuration
but plug in through the same model factories (see *Fine-tuning strategies*).

**Cold start (n = 0).** The aligned conditions have no calibration trials to build
a reference from, so they use the **first 20 trials of the session, without
their labels** (`unlabeled_start`). This start is *transductive*: the reference
covariance is computed on signals of the session that is then tested, though
never on its labels; online it corresponds to the first minute of use. These 20
trials are excluded from the test set of every condition, so all conditions are
scored on the same trials.

**Metrics.** Accuracy, Cohen's kappa and the number of test trials per
(dataset, model, condition, n, subject, session). The summaries average sessions
within each subject first, then give the mean and standard deviation over
subjects (and the number of subjects, which can fall for large `n` when sessions
are short, as in PhysionetMI).

**Running.**

```bash
uv sync --extra research
caffeinate -i -m -s uv run python research/experiments/calibration_curve.py \
    --config research/configs/calibration-curve.yaml     # [--only NAME] [--no-tracking] [--fresh]
```

Epochs are cached like in the benchmark; each finished target subject is
checkpointed in `$OPENMIND_DATA_DIR/cache/calibration-curve/`, so an interrupted
run resumes where it stopped (`--fresh` ignores the checkpoints, which are also
disabled while `src/` has uncommitted changes). Every score is written to
`$OPENMIND_DATA_DIR/results/calibration-curve.csv` and logged to MLflow, one
nested run per model, with the curves as `accuracy.<condition>` metrics whose
step is `n`.

### Fine-tuning strategies

`research/configs/calibration-curve-deep.yaml` extends the curve to a deep model
(EEGNet). For each target subject the network is **pre-trained once per seed on
the aligned trials of the other subjects**; that model is the `aligned` condition
at `n = 0`. For `n > 0` it is then fine-tuned on the `n` calibration trials per
class (aligned on their own reference, like the test trials) by
`DeepClassifier.fine_tune()`, one condition per strategy,
`aligned+finetune-<strategy>`:

| Strategy | What changes | Notes |
|---|---|---|
| `head` | output layer only | the feature extractor, batch norm statistics included, stays frozen |
| `final` | last block (separable convolutions, batch norm) and output layer | `ShallowConvNet`: batch norm and output layer |
| `adapter` | a new 1x1 spatial layer `x + delta @ x` (channels x channels, `delta` = 0 at the start) before the frozen network | identity at the start; weight decay pulls it back to the identity; the registry refuses such a model for now |
| `full` | every weight | low learning rate |

Fine-tuning is deterministic (seeded). With at least 30 calibration trials
(`min_validation_trials`) 20 % are held out (stratified) and the weights of the
best validation loss are kept, stopping after `patience` epochs without
improvement; with fewer trials nothing is held out and training runs for exactly
`epochs` (the best epoch cannot be told apart from so little data). The learning
rates are per strategy and were set once, not tuned on the test subjects.

EEGNet collapses on some initializations (see `research/results/montage-c.md`),
which would turn a curve into noise. `seeds` repeats every model that has a
`seed` parameter (pre-training and fine-tuning together) and averages the scores;
the spread between seeds is kept in `accuracy_seed_std` of the CSV. The
deterministic baselines run once.

```bash
uv sync --extra research --extra deep
caffeinate -i -m -s uv run python research/experiments/calibration_curve.py \
    --config research/configs/calibration-curve-deep.yaml
```

### Foundation model

`research/configs/calibration-curve-foundation.yaml` adds a pre-trained EEG
foundation model as a competitor, on the same protocol (montage C, three
classes, `aligned`, `aligned+finetune-head`, `aligned+finetune-full`,
0/5/10/20/40 trials per class, two seeds).
Code: `openmind_core.models.foundation.FoundationClassifier` (model name
`labram`, `uv sync --extra research --extra foundation`).

**Choice: LaBraM** (Jiang et al., ICLR 2024; 5.8 M parameters, 12-layer
transformer, pre-trained on about 2 500 hours of EEG), loaded with Braindecode's
`Labram.from_pretrained("braindecode/labram-pretrained")`. Reasons: (1) it names
every channel and all eight electrodes of montage C (`FC3 FC4 C3 Cz C4 CP3 CP4
CPz`) are among its 128 position embeddings, so a headset with eight electrodes
needs no interpolation or template; (2) the weights load with Braindecode 2.0 and
our channel names (checked: a forward pass on eight channels, 2 s windows, works);
(3) the research note reports that its personal gains survive across sessions.
MIRepNet (motor-imagery specific) remains the other candidate: Braindecode 2.0
has a `MIRepNet` class, but its weights (`starself/MIRepNet`) were not tried here.
`CBraMod`, `EEGPT` and `REVE` classes exist in the installed Braindecode too.

**Licence of the weights.** The model card of `braindecode/labram-pretrained`
declares `license: bsd-3-clause` in its metadata and says nothing more about the
weights; the original LaBraM code is MIT. The weights were trained on corpora
with their own terms (mostly the TUH EEG corpus, research use after
registration), which the card does not restate. We therefore use them for
research only: they are downloaded at run time into the Hugging Face cache
(`HF_HOME`, for example `$OPENMIND_DATA_DIR/cache/huggingface`), never committed
and never redistributed, and neither are fine-tuned copies (the model registry
refuses `labram`).

**What differs from EEGNet.**

| Aspect | EEGNet | LaBraM |
|---|---|---|
| Sampling rate | 250 Hz | 200 Hz: epochs resampled by 4/5 (`resample_poly`) inside the model |
| Input scale | per-channel std = 1 | per-channel std = 0.2 (about 20 µV / 100 µV, the scale of its pre-training) |
| Window | 2 s | 2 s = two one-second patches per channel, 16 tokens + one class token |
| Channels | positions only | named: each channel selects its position embedding |
| Output layer | trained | new linear layer (the checkpoint has none) |
| `final` strategy | last block | last transformer block, final norms, output layer |
| Source training (`fit`) | from scratch | the whole network at a low learning rate (`fit_strategy: full`; `head` or `final` freeze more) |

Frozen modules stay in evaluation mode, as for EEGNet. Everything else (alignment,
calibration trials, test trials, seeds, early stopping) is shared with the other
deep conditions.

**Caveat on alignment.** The input of every deep model is the *aligned* epochs
(Riemannian re-centering whitens the covariance), so the channels are spatial
mixtures rather than electrodes: LaBraM's position embeddings, which assume real
electrodes, are only approximately meaningful. This is the same pipeline as
EEGNet's, which keeps the comparison fair, but a `generic` (unaligned) condition
would be the more natural input for LaBraM.

**Cost.** The encoder has 5.8 M weights (EEGNet: about 2 000) and the weights file
is 23 MB. Smoke run (subject 1 of BNCI2014_001 as target, one seed, `n` = 0 and
10, four source epochs on about 3 400 aligned trials, 20 fine-tuning epochs, CPU
shared with two other benchmarks): 41 s for the whole subject, of which most is
pre-training. The full configuration (up to 30 source epochs with early stopping,
60 fine-tuning epochs, five values of `n`, 9 subjects x 2 seeds) is estimated at
1.5 to 3 hours on a laptop CPU, i.e. 5 to 10 minutes per subject and seed; the
smoke accuracy (73 % at every `n`) says nothing yet about the final comparison.

```bash
uv sync --extra research --extra foundation
export HF_HOME=$OPENMIND_DATA_DIR/cache/huggingface
caffeinate -i -m -s uv run python research/experiments/calibration_curve.py \
    --config research/configs/calibration-curve-foundation.yaml
```

Offline, `FoundationClassifier` raises a `RuntimeError` naming the repository;
without `huggingface_hub`, an `ImportError` pointing to the `foundation` extra.
Tests use a randomly initialized network and never download; the test that loads
the real weights is marked `slow`.

### Similarity-based cold start

`research/configs/calibration-curve-coldstart.yaml` asks, at `n = 0`, whether the
source model should be trained on **everyone** or only on the subjects most like
the new user. The *signature* of a user is the Riemannian mean covariance of
unlabelled windows (`rest_signature`). For the target it is computed on the same
first 20 trials of the session as the `aligned` reference (labels never read);
for each source subject on the first 20 trials of each of its sessions. The `k`
(`coldstart_k`, default 3) nearest source subjects in the affine-invariant
Riemannian distance (`nearest_subjects`) train the `coldstart-knn` model, aligned
per session like `aligned`, which is the model trained on all subjects. Only `n = 0`
is scored. A `k` equal to the number of source subjects gives `aligned` back, a
useful check; the difference with `aligned` also mixes "similar subjects" with
"fewer subjects", which a random-`k` control would separate.

```bash
uv run python research/experiments/calibration_curve.py \
    --config research/configs/calibration-curve-coldstart.yaml
```

A separate configuration keeps the checkpoints of `calibration-curve.yaml` valid.
An experiment may list `targets: [...]` to evaluate only some subjects (smoke
runs); the source is still every other subject.

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

## Results (deep learning vs baselines)

Full tables: [`research/results/deep.md`](../research/results/deep.md). Same
epochs, protocols and seed as the baselines (whose numbers are reproduced
exactly). Mean accuracy over subjects:

| Dataset, classes | Protocol | `csp-lda` | `ts-lr` | `eegnet` | `shallow` |
|---|---|---|---|---|---|
| BNCI2014-001, left/right | within-session | **76.8** | 74.8 | 68.6 | 68.9 |
| | cross-session | **74.8** | 73.2 | 68.4 | 69.5 |
| | cross-subject | 68.8 | 72.3 | **73.7** | 67.8 |
| BNCI2014-001, left/right/forward | within-session | **68.3** | 67.8 | 61.4 | 61.5 |
| | cross-session | **65.6** | 64.9 | 62.4 | 60.2 |
| | cross-subject | **53.2** | 52.2 | 52.8 | 49.7 |
| PhysionetMI, left/right | within-session | **60.5** | 58.7 | 52.4 | 56.5 |
| | cross-subject | **63.1** | 62.8 | 59.6 | 53.0 |

What these numbers say:

- **With little data, the classical decoders win.** Within a session there are
  about 115 (two classes) to 170 (three classes) training epochs: the networks
  lose 6 to 8 points to CSP + LDA. This matches the literature, where deep
  models only match or beat Riemannian and CSP pipelines with enough data.
- **With more data, EEGNet catches up.** Trained on the other eight BNCI
  subjects (about 2 300 epochs), EEGNet is the best cross-subject decoder for
  left/right (73.7 %), with the smallest spread between subjects. On PhysionetMI
  (about 630 training epochs) and for three classes, it stays level with or
  below the baselines: more subjects are needed for the network to pay off.
- **ShallowConvNet is the weaker network here**, especially across subjects; it
  has about 14 500 weights for 7 channels and two classes, against about 2 200
  for EEGNet: too many for these data sizes.
- **No model solves the new user.** Across subjects, three-class accuracy stays
  around 50-53 % (chance 33 %): the feet class does not transfer well. Hence
  the next steps: align each user's data (`CORE-050`), then fine-tune the
  pre-trained EEGNet on a few minutes of calibration (`CORE-060`). The registry
  already holds the generic models to start from.
- These results use one training recipe for both networks, chosen on two
  subjects; a proper hyper-parameter search (nested, on training subjects only)
  could narrow the gap within a session, but would not change the picture with
  so few epochs.
- Running time: about 50 minutes for left/right on BNCI2014-001 on a laptop
  (EEGNet on Apple MPS, ShallowConvNet on the CPU). The 176 minutes reported for
  three classes include a pause of the run, not computing time.
