# Preprocessing pipeline

The decoder sees the signal exactly as it was seen during training: the **same
code**, applied **causally**, processes public datasets offline and the live
stream online. Zero-phase filtering (`filtfilt`) is not used anywhere: it needs
future samples, so a model trained on it would face a different signal in real
time.

## Channels

| Use | Electrodes |
|---|---|
| Decoding (motor imagery) | `FC3, FC4, C3, Cz, C4, CP3, CP4`: montage A without Fp1 (`MONTAGE_A_MI`) |
| Double-blink detection | `Fp1` only |

**Fp1** is the eighth electrode of montage A. It sits above the left eye and is
dominated by eye movements and blinks, so it is not used for decoding. Most
motor-imagery datasets do not record it: BNCI2014-001 has no Fp1; PhysionetMI
does. Models trained on public data therefore use the seven decoding channels
only, and the blink detector runs on the live Fp1 channel.

## Steps

| Step | Offline (datasets) | Online (live stream) | Code |
|---|---|---|---|
| Channel selection | montage order, names normalized (`Fc3.` → `FC3`) | same | `datasets.moabb`, stream metadata |
| Units, rate | volts → µV, resampled to 250 Hz | stream already in µV | |
| Band-pass 8-30 Hz + notch | `StreamFilter` on each continuous run | `StreamFilter` on each block | `preprocessing.filters` |
| Windows | 2 s epochs, from cue + 0.5 s | 2 s window every 250 ms | `preprocessing.windows` |
| Artifacts | epochs above 150 µV peak to peak rejected | same rule | `preprocessing.artifacts` |
| Jaw clench (`action`) | — | `JawClenchDetector` on the motor electrodes | `preprocessing.jaw` |
| Double blink (studies) | — | `DoubleBlinkDetector` on Fp1 | `preprocessing.blink` |

- **Filter**: 4th-order Butterworth band-pass 8-30 Hz (mu and beta rhythms) plus
  a notch (Q = 30) at the power-line frequency of the recording: 50 Hz in Europe
  (BNCI2014-001, Graz), 60 Hz in the Americas (PhysionetMI). The notch is mostly
  a safeguard: both frequencies are already outside the band. The filter state is
  initialized from the first sample, so large DC offsets do not ring.
  `StreamFilter` gives identical samples whether a run is filtered at once or
  block by block (tested).
- **Epochs**: cue onset + 0.5 s to + 2.5 s, which skips the visual evoked response
  to the cue. Events are read as MOABB reads them: from the stim channel when
  the run has one, otherwise from the annotations. The stim channel of
  BNCI2014-001 marks the start of each trial, 2 s before the cue, so the loader
  adds the dataset's `interval[0]`. Its annotations, on the other hand, are
  already on the cue: adding `interval[0]` to them cuts epochs 2 s too late,
  at the end of the imagery period (left/right accuracy then drops from about
  85 % to 64 % for subject 1).
- **Online windows**: `SlidingWindow` keeps the last 2 s and yields a window every
  250 ms, shaped like an offline epoch `(n_channels, n_times)`.
- **Rejection**: 150 µV peak to peak (after band-pass) on any channel, or any
  non-finite value. PhysionetMI is noisier than BNCI: about 15-20 % of its epochs
  are rejected (subject 1: 13 of 69), none for BNCI2014-001 subject 1.

## Train a model

```bash
uv run openmind-core train --dataset bnci2014_001 --subjects 1-9 --model ts-lr \
    --classes left,right,forward --channels C --alignment riemann [--name NAME] [--overwrite]
```

Needs the `research` extra (datasets), and `deep` for `eegnet` and `shallow`.
`--subjects` takes ranges and commas (`1-3,5`); `--channels` a montage letter
(`A`: the seven motor electrodes, `C`: montage A plus CPz) or a comma-separated
list; `--alignment` is `none`, `euclid` or `riemann`, applied per session as in
the benchmark. The model is fitted on all epochs and registered under
`<model>-<dataset>-<montage>[-<alignment>]` unless `--name` is given.

The registry (`$OPENMIND_DATA_DIR/models/<name>/`) holds a `card.json` and the
weights: `weights.pt` for deep models (a `state_dict`, loaded with
`weights_only=True`) or `model.joblib` for baselines. The card records the
channels, sampling rate, epoch length, classes, the **alignment** and the
**filters** of the training epochs, so the online decoder can repeat them.
Joblib files are pickles and execute code when loaded: only load models you
trained or trust.

## Public datasets

```python
from openmind_core.datasets import load_epochs

epochs = load_epochs("bnci2014_001", subjects=[1, 2, 3])  # needs the research extra
epochs.X  # (n_epochs, 7, 500) float32, µV
epochs.y  # "left", "right", "forward"
epochs.subject, epochs.session
```

| Dataset | Subjects | Classes kept | Epochs (subject 1) |
|---|---|---|---|
| `bnci2014_001` (BCI Competition IV 2a) | 9, two sessions | left hand → `left`, right hand → `right`, feet → `forward` (tongue dropped) | 432 (2 × 3 × 72) |
| `physionetmi` (EEGMMIDB) | 109 | same; `rest` available on request | ≈ 56 after rejection |

Downloads go to `$OPENMIND_DATA_DIR/mne_data`; the global MNE configuration is not
modified. `research/experiments/make_epochs.py` builds and caches epochs for a few
subjects in `$OPENMIND_DATA_DIR/cache/epochs/`.

## Jaw clench: a muscle signal

The `action` command comes from a **double jaw clench** (clench, release at
least 0.3 s, clench), detected on the motor electrodes by `JawClenchDetector`:
muscle activity (EMG), not brain activity, which makes OpenMind a hybrid EEG +
EMG interface, to be presented as such. Method, validation and the gesture are in
[action-gesture.md](action-gesture.md). While a muscle burst is in progress, the
decoder pauses (`JawClenchDetector.active`).

## Double blink: an eye signal (not used for `action`)

The double blink was the first `action` gesture; people do it on their own
about 47 times per hour, so it was set aside ([montage.md](montage.md)). The
detector stays available for studies and for headsets with Fp1. A blink is the
electrical activity of the eyes and eyelids (EOG), not brain activity.

Detection (`DoubleBlinkDetector`, identical offline and block by block):

1. causal band-pass 0.1-8 Hz, which keeps the slow blink wave (a 0.5-10 Hz band
   flattens the 0.3 s peak and misses 100 µV blinks);
2. a blink is a peak above `max(60 µV, 5 × robust background deviation)`, the
   deviation being estimated on the last 10 s;
3. a double blink is two blink peaks 0.15 to 0.8 s apart; the detector then
   ignores blinks for 1 s, so a triple blink gives a single event.

Measured on a 15 µV RMS 1/f background: no false detection in 50 minutes;
double blinks detected 40/40 at 120 µV, 37/40 at 100 µV, 24/40 at 80 µV (real
blinks are typically 100-300 µV on Fp1). It is also tested on the synthetic
source of plug. **On real recordings, people double-blink on their own** (about
47 times per hour on PhysionetMI): see [montage.md](montage.md) for the
consequences on the `action` command.

## Calibration markers

The calibration scenes of OpenMind Sandbox publish their own `openmind-markers`
stream (`source` `sandbox`): for each trial, `trial_start`, then the `cue` with
the expected command, then `trial_end`, plus `event` markers for sessions and
blocks. Core reads markers through `openmind_plug.inlet.EEGInlet`, which follows
the LSL stream specification 1.2.0 of Plug: it reads **every** markers stream,
including streams that start later, and merges them by timestamp on the local
LSL clock.

`openmind_core.io.TrialCollector` rebuilds the calibration trials:

```python
from openmind_core.io import TrialCollector

collector = TrialCollector(source="sandbox")  # ignore cues of a replayed dataset
for trial in collector.add(inlet.read_markers()):
    trial.label, trial.cue_time  # epochs are cut from the cue, as offline
```

Only markers of the given `source` are kept, so the cues of a dataset replayed
by Plug (`source` `replay`) do not mix with calibration cues. A trial is kept
when its `trial_end` follows its `cue`; a trial started and never ended
(another `trial_start`, or `session_end`) was interrupted and is discarded
(`collector.interrupted` counts them). Every complete trial is kept for
training, whether the user succeeded or not.
