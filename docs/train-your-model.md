# Train your own model

This guide trains a decoder on public data, registers it, and runs it on a
live stream. It then shows how to register a classifier of your own and how a
user's personal model is derived from it. No headset is needed: the stream can
come from OpenMind Plug's synthetic source or a replayed dataset.

All models live in `$OPENMIND_DATA_DIR/models/` (default
`~/openmind-data/models/`), never in Git.

## 1. Install

```bash
uv sync --extra research          # public datasets (MOABB, MNE)
uv sync --extra research --extra deep   # also EEGNet and ShallowConvNet
```

## 2. Train a generic model from the command line

```bash
uv run openmind-core train --dataset bnci2014_001 --subjects 1-9 \
    --model ts-lr --classes left,right,forward --channels C --alignment riemann
# registered as ts-lr-bnci2014_001-c-riemann
```

| Option | Choices | Advice |
|---|---|---|
| `--model` | `ts-lr`, `mdm`, `csp-lda`, `eegnet`, `shallow` | `ts-lr`: fast, no GPU, the reference baseline of our calibration results |
| `--channels` | `C` (8 motor electrodes), `A` (7), or a list | keep `C`, the montage of the IronBCI-8 plan |
| `--alignment` | `none`, `euclid`, `riemann` | `riemann`: +6 to +11 points for a new user with no labelled trial ([REPORT](../research/REPORT.md)) |
| `--subjects` | ranges and commas, `1-9`, `1,3,5` | all subjects for a generic model |

The first run downloads the dataset into `$OPENMIND_DATA_DIR/mne_data/`.
Training fits on every epoch; to measure the model, run the benchmark
(`research/experiments/benchmark.py`, see [evaluation.md](evaluation.md)).

## 3. Run it on a stream

```bash
cd ../sandbox/dev && docker compose up -d                    # MQTT broker
cd ../../plug && uv run openmind-plug stream --source synthetic   # or --source replay
cd ../core && uv run openmind-core profile create P00
uv run openmind-core online --profile P00 --model ts-lr-bnci2014_001-c-riemann
```

With an aligned model, the decoder first records 30 s of clean signal to
estimate the user's reference (`calibrating` in the status; keep still), stores
it in the profile, then decodes. `--dry-run` prints the decisions without a
broker. See [pipeline.md](pipeline.md) for the thresholds and the evidence
window.

## 4. Register a classifier of your own

Any scikit-learn compatible classifier can be decoded online if it:

* takes epochs shaped `(n_epochs, n_channels, n_times)` in microvolts,
  filtered as the training epochs (and aligned, if you aligned them);
* has `predict_proba` and `classes_`, with classes among the protocol commands
  (`left`, `right`, `forward`, `rest`).

```python
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from openmind_core.alignment import align_epochs
from openmind_core.datasets import load_epochs
from openmind_core.models.registry import save_model

epochs = load_epochs("bnci2014_001", subjects=range(1, 10))
epochs = align_epochs(epochs, method="riemann", by="session")

model = make_pipeline(Covariances("oas"), TangentSpace(), LinearDiscriminantAnalysis())
model.fit(epochs.X, epochs.y)

save_model(
    model,
    "ts-lda-bnci2014_001-c-riemann",
    epochs.channels,
    training={"dataset": "bnci2014_001", "subjects": list(range(1, 10))},
    alignment="riemann",
    filters=epochs.info["filters"],
    srate=epochs.srate,
    n_times=epochs.X.shape[2],
    architecture="ts-lda",
)
```

The card (`card.json`) records the channels, the sampling rate, the epoch
length, the alignment and the filters: the online decoder repeats exactly
these steps, and refuses a stream that lacks one of the channels. Baselines are
saved with joblib, which executes code when loaded: only load models you
trained or trust. Deep models built with `make_model("eegnet", srate=...)` are
saved as a `state_dict` (`weights.pt`) and loaded safely.

## 5. From the generic model to a personal one

The generic model is the starting point of every new user:

1. **0 trials**: the warm-up reference re-centers the user's signal
   (alignment); optionally the nearest stored profiles give a cold start.
2. **Calibration**: during a guided session the user's epochs train a personal
   classifier blended with the generic one; the blend weight shows up as
   `status.assistance` ([calibration.md](calibration.md)).
3. **Later sessions**: resume mode (rest, re-alignment, 5 cues) and online
   adaptation with rollback.

The personal model and the reference are stored in the user's profile
([profiles.md](profiles.md)); `openmind-core profile delete P00` erases them.

## 6. Sharing models

Only models trained on **public data** may be shared, for instance on Hugging
Face with a model card built from `card.json`. A model fine-tuned on a
volunteer's recordings is health data: it stays on the local machine.
