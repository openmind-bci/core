# Guided calibration session

A calibration session turns the cued trials of a Sandbox scene into labelled
epochs, adapts the decoder to the user as they come in, and tells how much the
decoder still relies on the generic model. The code lives in
`openmind_core.calibration`. It is pure logic driven by EEG blocks and markers.
`openmind-core online` does not run a calibration session yet: it decodes with
a registered model after the warm-up (a `--calibrate` option is planned,
[#10](https://github.com/openmind-bci/core/issues/10)).

## Session flow

```text
sandbox scene (cues)             core
  markers  ───────────►  TrialCollector ─► completed trials ─┐
  EEG (LSL) ──► OnlineDecoder (generic + personal blend)     │
                 └─ CalibrationRecorder (EpochBuffer) ───────┴─► labelled epochs
                                                                   │
  block_end ─► PersonalModel.end_block(): score the block, refit ◄──┘
               └─► assistance = 1 - w  (status.assistance)
```

The scene (`calibration_default.yaml` in the sandbox: fixation 2 s, cue 1 s,
imagery 4 s, rest, in blocks) publishes `session_start`, `block_N_start`,
`trial_start`, `cue` (with the label), `trial_end`, `block_N_end` and
`session_end` on `openmind-markers`, with `source = "sandbox"`.

## Live epochs

`CalibrationRecorder` filters the model's channels causally with the same
`StreamFilter` settings as training (from the model card) and keeps the last
30 s in an `EpochBuffer`. When a trial completes, the epoch
`[cue + 0.5 s, cue + 2.5 s]` is cut by timestamp, exactly like
`epochs_at` does offline (tested: same samples). Epochs are filtered but not
aligned.

* **All completed trials are kept**, successful or not: the label is the cue,
  not what the decoder decided.
* A trial interrupted before `trial_end` is dropped by the collector
  (`recorder.interrupted`).
* A completed trial whose epoch is no longer in the buffer, or has a gap, is
  counted in `recorder.missed`. Markers arriving before the end of the epoch
  wait for the samples.
* `epochs()`, `labels()`, `counts_per_class()` give the training set; each
  `CalibrationTrial` also carries its trial and block numbers.

## Staged model update

`PersonalModel` wraps the generic model and the recorder. After each block
(`block_end`, `end_block()`), or every `update_every` trials, it fits a
**personal classifier** (tangent space + logistic regression, `ts-lr`) on the
user's epochs, aligned with the session reference, once every class of the
model has at least `min_per_class` (default 5) epochs. The online probabilities
are a blend:

```text
p = (1 - w) * p_generic + w * p_personal      w = min(1, n_min / target_per_class)
```

`n_min` is the number of epochs of the rarest class at the last fit
(`target_per_class` is 20 by default). This is a simple and safe blending: it
needs no source data online, `w` is 0 until a personal model exists, and a model
trained on few epochs weighs little. Fine-tuning a deep model is a later task.

`assistance = 1 - w` goes from 1 (generic model only) to 0 (personal model only)
as the user calibrates; publish it as `status.assistance`. The object has
`predict_proba` and `classes_`, so it replaces `OnlineDecoder.model`:

```python
personal = PersonalModel(model, card, recorder)
decoder.model = personal
personal.set_reference(decoder.reference_covariance)  # when the card has an alignment
...
personal.trial_done(trial, decisions)  # decisions taken during the trial
personal.end_block()                   # on block_end: score the block, refit
```

`trial_done` feeds a `ProgressTracker` (see [scoring](scoring.md)) with the
decisions of the trial; `end_block` closes the block and returns its
`BlockScore`.

## Resume mode

A returning user does not repeat the calibration. `ResumeProtocol(rest_s=45,
control_cues=5)` runs in three steps:

1. **Rest**: unlabeled signal from `session_start` to the first `block_N_start`
   (at least `rest_s` seconds of clean 2 s windows, the first second skipped).
2. **Realignment**: new reference covariance of the rest windows with the
   model's alignment method (`protocol.reference`; give it to the decoder).
3. **Control cues**: the first `control_cues` cued trials after the rest are
   scored with a `ProgressTracker`, using the decisions passed to
   `add_decisions(decisions, until=...)` (a trial is scored once the decoder has
   gone past its end, or at `finish()`).

The result is `ResumeResult(reference, accuracy, passed)` (plus the block score
and the number of rest windows): `accuracy` is the mean per-command accuracy of
the control cues and `passed` means it reaches `min_accuracy` (0.6). A failed
resume suggests a full calibration block.

## What is stored in the profile

`PersonalModel.save(store, user)` writes:

* `weights/personal.joblib`: the personal classifier, the class order, the
  weight, the alignment method and the epochs per class (a joblib pickle: only
  load your own files);
* one record in the scores history: the `score` fields, `assistance` and
  `personal_weight`.

The reference covariance of the session is stored with
`ProfileStore.save_reference`. Epochs themselves are not stored (EEG is health
data and stays out of the repositories).

## Online adaptation

EEG drifts during a session (electrode impedance, fatigue, posture).
`openmind_core.online.adaptation` follows it with three bounded, reversible
mechanisms. It is a library for now: `openmind-core online` does not use it yet
(an `--adapt` option is planned, [#10](https://github.com/openmind-bci/core/issues/10)).

```python
adapter = RecenteringAdapter(half_life_s=120)   # or rate=0.01
adapter.attach(decoder)                         # follows the decoded windows
tuner = PseudoLabelTuner()                      # disabled by default
tuner.attach(decoder)
personal.pseudo = tuner
guard = AdaptationGuard(decoder, adapter, personal, tuner, max_drop=0.10)

guard.begin_block()                             # snapshot
...                                             # trials of the block
rolled_back = guard.end_block(score.overall)    # or end_block() without cues
```

**What adapts**

* *Re-centering* (unsupervised): after `min_windows` (20) decoded windows, the
  alignment reference covariance moves towards the covariance of each window and
  the whitening is rebuilt (`OnlineDecoder.set_reference`). With `riemann` the
  move is a step of weight `rate` along the geodesic between the reference and
  the window; with `euclid` it is a moving average. This follows slow gain and
  spatial drifts without any label.
* *Pseudo-labels* (supervised by the model itself): windows whose smoothed top
  probability exceeds `threshold` (0.9) are added with their predicted label to
  the training set of the personal model, which is refitted at block ends. They
  are stored apart from the true calibration labels and limited to 30 % of the
  set, overall and per class.

**Bounds**

* Each update moves the reference by at most `max_step` (0.05, affine-invariant
  distance), whatever the window.
* The total distance to the initial reference is capped at `max_drift` (1.5):
  an update that would exceed it is not applied, the adapter freezes and
  `status_message()` reports `adaptation frozen (drift limit)`.
* The generic model and the initial reference are never modified
  (`guard.initial_reference` is read-only; `guard.reset()` goes back to it).

**Rollback.** `begin_block()` snapshots the reference, the personal model and
the pseudo-labels. At `end_block(score)`, if the score is more than `max_drop`
(0.10) under the best of the last 3 blocks, the state the best block started
with is restored, adaptation is paused for one block, the rollback is logged
and `status_message()` returns `adaptation rolled back` for the decoder status.
The score is the `overall` of the progress tracker; without cues
`end_block()` uses the mean top probability of the windows decoded in the
block, an unsupervised proxy that drops when the decoder becomes indecisive.

**Why pseudo-labels are off by default.** The model learns from its own
decisions: a systematic error is reinforced (confirmation bias) and a confident
mistake enters the training set as if it were true. Re-centering has no such
loop, since it never sees a label. Enable `PseudoLabelTuner(enabled=True)` only
with the rollback guard, and prefer true labels from calibration blocks.

## Limits

* The personal model is not fitted with source data: with few epochs it can be
  noisy, which the weight `w` limits; the blend is not a fine-tuning.
* Epochs are not rejected on amplitude when recorded; the recorder keeps what the
  cue says. Artifact rejection before training is left to the training step.
* The epoch buffer keeps 30 s: markers must reach the recorder within that delay.
* Resampling is not supported (the stream must run at the model's rate).
