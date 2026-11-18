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
| Double blink | — | `DoubleBlinkDetector` on Fp1 | `preprocessing.blink` |

- **Filter**: 4th-order Butterworth band-pass 8-30 Hz (mu and beta rhythms) plus
  a notch (Q = 30) at the power-line frequency of the recording: 50 Hz in Europe
  (BNCI2014-001, Graz), 60 Hz in the Americas (PhysionetMI). The notch is mostly
  a safeguard: both frequencies are already outside the band. The filter state is
  initialized from the first sample, so large DC offsets do not ring.
  `StreamFilter` gives identical samples whether a run is filtered at once or
  block by block (tested).
- **Epochs**: cue onset + 0.5 s to + 2.5 s, which skips the visual evoked response
  to the cue. MOABB marks BNCI2014-001 trials 2 s before the cue; the loader adds
  the dataset's `interval[0]` to reach the cue.
- **Online windows**: `SlidingWindow` keeps the last 2 s and yields a window every
  250 ms, shaped like an offline epoch `(n_channels, n_times)`.
- **Rejection**: 150 µV peak to peak (after band-pass) on any channel, or any
  non-finite value. PhysionetMI is noisier than BNCI: about 15-20 % of its epochs
  are rejected (subject 1: 13 of 69), none for BNCI2014-001 subject 1.

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

## Double blink: an eye signal

The `action` command comes from a **double blink**, detected on Fp1. A blink is
the electrical activity of the eyes and eyelids (EOG), not brain activity: this
makes OpenMind a hybrid EEG + EOG interface, and it must be presented as such.

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
source of plug.
