# Architecture

OpenMind Core turns a standard EEG stream into generic commands. It reads the
LSL stream defined by OpenMind Plug (`openmind-eeg`, microvolts, with
`openmind-markers`), decodes the user's intent with a model from the local
registry and publishes the OpenMind protocol over MQTT. It never talks to a
device or an object directly.

```text
 OpenMind Plug                 OpenMind Core                          OpenMind Sandbox
 (headset, replay,             ┌──────────────────────────────────┐   (Godot scenes,
  synthetic)                   │ online/service   run_online      │    fake robot,
                               │   EEGInlet ─► OnlineDecoder      │    ESP32 robot,
 openmind-eeg ───── LSL ─────► │      StreamFilter (causal)       │    adapters)
 openmind-markers              │      SlidingWindow 2 s / 250 ms  │
                               │      Aligner (profile reference) │
                               │      model.predict_proba         │
                               │   EvidenceAccumulator ─► command │
                               │   JawClenchDetector  ─► action   │ ──► openmind/v1/commands
                               │   StatusEmitter (≥ 2 Hz)         │ ──► openmind/v1/status
                               │ online/bandpower_service         │ ──► openmind/v1/bandpower
                               │ scoring  ProgressTracker         │ ──► openmind/v1/score
                               └──────────────────────────────────┘ ◄── openmind/v1/markers
                                  │ profiles/   models/ (registry)
                                  ▼
                           $OPENMIND_DATA_DIR  (never in Git)
```

## Packages

| Package | Role | Main entry points |
|---|---|---|
| `protocol/` (`openmind-protocol`) | The MQTT contract: JSON Schemas, pydantic models, safety rules. The only package Sandbox depends on. | `openmind_protocol.Command`, `Status`, `safety` |
| `datasets` | Public datasets through MOABB, cut into epochs with the same filters as online decoding. | `load_epochs`, `EpochSet` |
| `preprocessing` | Causal and offline filters, sliding windows, artifact flags, jaw clench and blink detectors, band power. | `StreamFilter`, `SlidingWindow`, `JawClenchDetector`, `BandPowerEstimator` |
| `alignment` | Euclidean and Riemannian alignment (re-centering) per session or per user. | `Aligner`, `align_epochs` |
| `models` | Riemannian baselines, EEGNet and ShallowConvNet, the LaBraM wrapper, and the local registry. | `make_model`, `save_model`, `load_model` |
| `evaluation`, `research/` | Within-, cross- and leave-one-subject-out protocols, the benchmark and the calibration curve. | `research/experiments/*.py` |
| `profiles` | Pseudonymized user profiles: alignment reference, metadata, score history; complete deletion. | `ProfileStore` |
| `online` | The real-time loop: decoder, evidence accumulation, emitters, health and latency, adaptation. | `OnlineDecoder`, `run_online` |
| `calibration` | Guided session: live epochs, staged personal model, resume mode, cold start. | `CalibrationRecorder`, `PersonalModel`, `ResumeProtocol` |
| `scoring` | Progress score (accuracy per command, false starts, stability, ECE). | `ProgressTracker` |
| `bus` | MQTT publisher (paho-mqtt 2). | `Publisher` |
| `cli` | `openmind-core info`, `train`, `online`, `bandpower`, `profile`. | |

Dependencies are layered: `preprocessing` and `alignment` depend only on
NumPy, SciPy and pyRiemann; `models` adds scikit-learn and, with the `deep`
extra, PyTorch and braindecode (imported lazily, so the online decoder of a
baseline runs without them); `datasets` needs the `research` extra (MOABB,
MNE). The online loop needs `pylsl` through OpenMind Plug's `EEGInlet`.

## One offline pipeline, repeated online

The same steps run in training and in real time, so a model behaves online as
it did in the benchmark:

1. **Channels**: montage C, `FC3, FC4, C3, Cz, C4, CP3, CP4, CPz`
   ([montage.md](montage.md)).
2. **Filters**: band-pass 8–30 Hz and notch, causal (`StreamFilter`) both
   offline and online, settings stored in the model card.
3. **Epochs / windows**: 2 s, `[cue + 0.5 s, cue + 2.5 s]` offline; a sliding
   2 s window every 250 ms online.
4. **Alignment**: the inverse square root of a reference covariance, estimated
   per session offline and, online, from the warm-up (30 s of clean signal) or
   from the profile.
5. **Model**: `predict_proba` over the trained classes.

Details: [pipeline.md](pipeline.md).

## From probabilities to commands

Probabilities are averaged over 4 windows (about 1 s); a command is sent when
its average passes its own threshold, otherwise `rest`. A window flagged as
muscle activity is not decoded, and a double jaw clench becomes `action`.
Commands are published when they change and repeated every 0.5 s, the status
at 4 Hz (the protocol requires 2 Hz or more) with `latency_ms` measured from the last EEG sample to publication. On
exit, a final `rest` and an `idle` status are sent. Objects stop on their own
when the status stops for 2 s ([protocol.md](protocol.md), section 4).

## Personalisation

```text
generic model (public data, registry)
   │  warm-up rest ─► alignment reference ─► profile          (0 labelled trials)
   │  nearest profiles (Riemannian distance) ─► cold start    (optional)
   │  calibration trials ─► PersonalModel blend, assistance   (5–40 per class)
   │  next session: resume mode, 30–60 s of rest + 5 cues
   └─ online: sliding re-centering, optional pseudo-labels, rollback on score drop
```

See [calibration.md](calibration.md), [profiles.md](profiles.md) and
[scoring.md](scoring.md); the measured gains are in
[`research/REPORT.md`](../research/REPORT.md).

## Data on disk

Everything personal or downloaded lives under `$OPENMIND_DATA_DIR`
(default `~/openmind-data`), never in the repository:

| Path | Content |
|---|---|
| `mne_data/` | public datasets downloaded by MOABB |
| `cache/` | epoch caches and benchmark checkpoints |
| `models/<name>/` | registered models: `card.json` and `weights.pt` or `model.joblib` |
| `profiles/<P01>/` | reference covariance, metadata, score history, personal models |
| `results/` | CSV scores of the experiments |

The profile store refuses a root inside a Git working tree.
