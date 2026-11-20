# The `action` gesture: jaw clench on the motor electrodes

`action` means "grab" or "interact" (open a door, pick an object). The double
blink first planned for it fires on its own about 47 times per hour
([montage.md](montage.md)), and cannot be read without Fp1. This study asks
whether a **double jaw clench**, read on the motor electrodes only, is reliable
enough to replace it.

Tables: [`research/results/jaw.md`](../research/results/jaw.md) (run 1, where
the settings were chosen) and
[`research/results/jaw-run2.md`](../research/results/jaw-run2.md) (run 2,
held-out validation), produced by `uv run python research/experiments/jaw.py
[--run 2]`.

## Data

PhysioMotion Artifact (OpenNeuro ds006386, CC0): 30 participants, 59 channels
(10-10), 1000 Hz, laboratory cap. Each run contains two blocks of five cued
jaw clenches and annotated look-alikes: swallowing, articulation ("la-la-la"),
eyebrow raising, head nodding and turning, eye movements, blinks, rest.
Run 1: 50 annotated clench blocks, 608 minutes; run 2: 57 blocks, 581 minutes.

The seven motor electrodes of montage A are used. **CPz is the recording
reference of this dataset**, so the eighth motor electrode could not be tested;
with a signal present on every electrode, one more is unlikely to change much.

## Detector

1. Signals at 250 Hz (the IronBCI-8 rate); muscle band 30-110 Hz with notches
   at 50 and 100 Hz, above the 8-30 Hz band used for decoding.
2. Power over 100 ms on each electrode, compared with that electrode's own
   running median of the previous 10 s (causal), in dB.
3. Score: the **median over the electrodes**, so a clench must raise the power on
   most of them at once (a bad contact on one electrode cannot trigger it).
4. A clench: the score exceeds 10 dB, **peaks above 20 dB** and lasts **at most
   1 s**. A double clench: two clench onsets 0.25 to 1.2 s apart.

The peak and duration rules come from the shape of the bursts: deliberate
clenches peak at 26 dB above the background (median, run 1), swallowing at
18 dB and eyebrow raising or nodding at 13 dB, and swallowing lasts longer.

## Results

Seven motor electrodes:

| Detector | Clenches detected (run 1 / run 2) | False detections per minute | False double clenches per hour |
|---|---|---|---|
| Score above 15 dB | 91 % / 88 % | 0.8 / 1.0 | 5.1 / 8.3 |
| **Peak ≥ 20 dB, ≤ 1 s** | **76 % / 68 %** | **0.25 / 0.33** | **0.7 / 0.8** |
| Peak ≥ 25 dB, ≤ 1 s | 49 % / 48 % | 0.13 / 0.18 | 0.3 / 0.3 |

- **The jaw clench is readable on the motor electrodes.** With the strict rule,
  false `action` commands drop **below one per hour**, against 47 for the
  spontaneous double blink, in recordings that are much denser in swallows,
  eyebrow raises and nods than daily life. The held-out run confirms the
  setting.
- The motor electrodes do better than Fp1, which catches more forehead muscle
  activity (with the strict rule, Fp1 finds only about half of the clenches),
  and slightly better than C3 and C4 alone (fewer false double clenches).
- **Swallowing is the main look-alike**: at a plain 10 dB threshold, 77 to 92 %
  of swallows trigger a false detection. The peak and duration rules remove
  most of them; eyebrow raising and nodding come next. Articulation triggers
  almost nothing (4-5 %).
- The price is sensitivity: about **seven clenches in ten** are detected. The
  dataset only contains single cued clenches, so the double clench was not
  measured; if the two clenches were independent, about half of the attempts
  would be recognized on the first try. A user who sees the feedback simply
  repeats, and clenching harder helps (the threshold is on the peak).

## The detector in OpenMind Core

`openmind_core.preprocessing.JawClenchDetector` implements this method causally,
block by block (same events as offline, tested), with the study's settings:

```python
from openmind_core.preprocessing import JawClenchDetector

detector = JawClenchDetector(srate=250, n_channels=8)  # the motor electrodes of montage C
for block in stream:                                   # (n_samples, 8) in µV
    for t in detector.update(block):                   # double clenches
        publish_action(t)
    pause_decoding = detector.active                   # muscle burst in progress
```

Run on the public recordings, it reproduces the study: 76 % / 68 % of the
clenches detected and 0.6 false double clench per hour on runs 1 / 2.

**The gesture**: clench for about 0.2-0.4 s, **release for at least 0.3 s**,
clench again. With a release of 0.3 s, synthetic double clenches are always
recognized; below 0.2 s the two bursts merge into one. Finer settings (50 ms
envelope and merge gap) would accept shorter releases, but they also split
other artifacts into pairs: 2.2 instead of 0.6 false double clenches per hour
on the held-out run. The calibration session will teach the gesture, with
feedback.

A double clench is reported about 0.15 s after the end of the second burst.
While a burst is in progress, `detector.active` is true: the online loop
(`CORE-080`) pauses decoding meanwhile, since muscle activity also reaches the
beta band.

## What remains to check with the headset

- **Real double clenches**, done on purpose, with per-user calibration of the
  peak level during the calibration session.
- **Daily-life look-alikes absent from the dataset**: conversation, chewing and
  eating, yawning, smiling.
- **Dry electrodes and the IronBCI-8 reference**: muscle activity is strong, but
  contact noise is higher than with a laboratory cap.
- **Decoding during a clench**: muscle activity also contaminates the beta band;
  the decoder pauses while the detector sees a burst (the same burst detection
  can drive this pause).

## Consequences

- The jaw clench becomes the **candidate gesture for `action`**, readable on the
  eight motor electrodes, without Fp1.
- The IronBCI-8 can therefore put all eight channels on the motor cortex
  (`FC3, FC4, C3, Cz, C4, CP3, CP4, CPz`, see [montage.md](montage.md)).
- `action` stays limited to the contexts where it makes sense, with a
  confirmation before acting on a real object.
- The jaw clench is a muscle signal, not a brain signal: OpenMind becomes an
  EEG + EMG hybrid interface, and must be presented as such.
