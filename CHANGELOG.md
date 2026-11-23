# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
`openmind-core` and `openmind-protocol` are versioned separately; protocol changes
are listed under their own heading.

## [Unreleased]

### Added

- Online adaptation with safeguards (`openmind_core.online.adaptation`): bounded
  Riemannian re-centering of the alignment reference (`RecenteringAdapter`), optional
  capped pseudo-labels (`PseudoLabelTuner`, off by default) and `AdaptationGuard`
  (frozen initial model and reference, snapshots, automatic rollback when the score
  drops); `OnlineDecoder.set_reference` and window hooks. Not wired into the CLI yet.
- Live calibration epochs (`openmind_core.calibration`): `EpochBuffer` and
  `CalibrationRecorder` cut the epoch `[cue + 0.5 s, cue + 2.5 s]` of every completed
  trial like the offline loader, keep all trials, drop interrupted ones and count missed.
- Staged model update during calibration (`PersonalModel`): personal tangent-space
  classifier refitted after each block and blended with the generic model, with
  decreasing `assistance`, block scores and storage in the user profile.
- Resume mode (`ResumeProtocol`): rest, new reference covariance and cued control trials.
- Progress score (`openmind_core.scoring`): per-command accuracy, false start rate and
  calibration error per block of trials, `ProgressTracker` with `ready` after
  consecutive passed blocks, `score` messages and profile records.
- Online band power estimator (`BandPowerEstimator`): alpha, mu and beta power per
  electrode every 100 ms, smoothing, resting baseline and channel quality flags,
  as `bandpower` messages.
- MQTT `Publisher` (`openmind_core.bus`): paho-mqtt 2, automatic reconnection,
  QoS per message kind, clear error when the broker is down, `parse_broker()`.
- `openmind-core bandpower`: reads the standard EEG stream and publishes the
  band power at up to `--rate` Hz, with a resting baseline (`--baseline-s`),
  expected electrodes (`--expected`) and `--dry-run` (no broker).

### Protocol 1.0.1 (`openmind-protocol`)

- `action` is described as a double jaw clench (a muscle signal) instead of a
  double blink. Wording only: messages and validation are unchanged.

### Protocol 1.0.0 (`openmind-protocol`)

- JSON Schemas v1 for `command`, `status`, `marker`, `event`, `score` and
  `bandpower`, with valid and invalid examples.
- Topics `openmind/v1/commands|status|markers|events|score|bandpower`.
- Validated pydantic models, `encode()` / `decode()` / `decode_topic()` with
  explicit `ProtocolError`s and rejection of other major versions.
- `Watchdog` implementing the safety rules for objects (2 s heartbeat timeout,
  confidence threshold, sequence order). Specification in `docs/protocol.md`.

### Added

- Evidence accumulation and per-command thresholds for the online decoder
  (`EvidenceAccumulator`, with hysteresis and `rest` by default), command and
  status emitters (`CommandEmitter`, `StatusEmitter`, `HealthMonitor`) and a
  `LatencyMeter` building a `decoder` / `latency` event. No MQTT here.
- Online decoder engine (`openmind_core.online.OnlineDecoder`): EEG blocks in,
  class probabilities per 2 s window out, with the filter, windows, amplitude
  rejection and alignment of offline training (same probabilities, tested).
  Pauses on jaw muscle activity (`muscle`), reports double clenches as
  `action`, and estimates the alignment reference on the first 30 s when the
  profile has none (`Aligner.from_reference` restores a stored one).
- Model registry for every model: baselines are saved as `model.joblib`
  (`format="sklearn"`), the card records the training `alignment` and
  `filters`, torch is imported only for deep models. `openmind-core train`
  trains a model on a public dataset (`--dataset`, `--subjects 1-9`, `--model`,
  `--channels A|C|list`, `--alignment`) and registers it.
- User profile store (`openmind_core.profiles`, `openmind-core profile`):
  pseudonymized ids, reference covariance, scores, personal weights directory,
  refusal of roots inside a Git repository and full erasure on delete.
  See `docs/profiles.md`.
- Calibration trials rebuilt from markers (`openmind_core.io.TrialCollector`):
  keeps the markers of the calibration scenes (`source` `sandbox`), rebuilds
  each `trial_start` → `cue` → `trial_end` trial and discards interrupted ones.
  Markers come from every `openmind-markers` stream through Plug's `EEGInlet`
  (stream specification 1.2.0).
- Alignment of sessions (`openmind_core.alignment`): Euclidean alignment and
  Riemannian re-centering of the signals, with a reference from all epochs of a
  session or its first ones; usable with every decoder.
- Baselines and deep models retrained on montage C (eight motor electrodes,
  `research/configs/montage-c.yaml`, `research/results/montage-c.md`): same as
  montage A for left/right, better at three classes; six pre-trained models
  `-c` in the registry. Benchmarks now resume from finished experiments and log
  each evaluation unit.
- Research note on EEG foundation models and datasets for fast calibration
  (`research/notes/foundation-models.md`).
- Online jaw clench detector (`openmind_core.preprocessing.JawClenchDetector`,
  `detect_double_clenches`): causal, same events offline and block by block,
  flags ongoing muscle activity to pause decoding. Reproduces the public-data
  study (0.6 false double clench per hour). The gesture: clench, release at
  least 0.3 s, clench again.
- Jaw clench study for the `action` gesture (`research/experiments/jaw.py`,
  `docs/action-gesture.md`) on PhysioMotion Artifact (OpenNeuro ds006386): on
  the motor electrodes, a double clench with a strong, short burst gives under
  one false `action` per hour, held-out run included, for about seven clenches
  in ten detected. Swallowing is the main look-alike.
- Montage study (`research/experiments/montage.py`, `docs/montage.md`): CPz is
  the best eighth decoding electrode (small but consistent gain, mostly on
  `forward`); blinks cannot be detected reliably without Fp1, even with a
  spatial filter of the eight motor electrodes; spontaneous double blinks
  (about 47 per hour) rule out the double blink as the `action` gesture. Per-class recall in evaluation scores; `iter_runs()` for
  continuous public recordings.
- Deep decoders EEGNet and ShallowConvNet (`openmind_core.models.deep`, `deep`
  extra): Braindecode networks behind a scikit-learn interface, trained with
  early stopping; `make_model()` builds any baseline or deep model by name.
  Local model registry in `$OPENMIND_DATA_DIR/models` (weights as a
  `state_dict` plus a JSON card). Benchmark against the baselines in
  `research/results/deep.md`. PyTorch CPU builds on Linux and Windows.
- Classical baselines `csp-lda`, `mdm` and `ts-lr` (`openmind_core.models`) and
  evaluation protocols within session, across sessions and leaving one subject
  out (`openmind_core.evaluation`). `research/experiments/benchmark.py` runs a
  benchmark from a YAML configuration (`research/configs/baselines.yaml`),
  writes the result tables to `research/results/` and tracks runs in MLflow.
  See `docs/evaluation.md`.
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
