# Progress score

The progress score answers one question: is the user reliable enough to drive
real objects? `ready` in the `score` message (see [protocol](protocol.md)) is
true only when it is. The code lives in `openmind_core.scoring`.

## Definitions

A **cued trial** (see `TrialCollector`) gives the expected command, the cue
label, and the decisions of the pipeline are compared with it.

* **Imagery window**: from 0.5 s to 4.0 s after the cue (the 4 s imagery of the
  calibration scenario, minus the visual response). It stops earlier if the
  trial ends first.
* **Decision duration**: a decision stays in force until the next one.
* **Correct trial**: the movement command held for the longest time in the
  window equals the cue label. For a `rest` cue, the trial is correct when no
  movement command is held for 0.5 s or more. `action` decisions are one-shot
  and ignored.
* **False start**: a movement command held for at least 0.5 s while the user
  should be at rest, i.e. between `trial_start` and the cue of every trial, and
  during the whole window of `rest` trials.
* **False start rate**: false starts over rest periods of the block.
* **Per-command accuracy**: share of correct trials per cue label. `overall` is
  the mean of the per-command accuracies, so a rare class weighs as much as a
  frequent one.
* **ECE** (expected calibration error): predictions (confidence of the winning
  command, correct or not) are split in 10 confidence bins; the error is the
  mean of `|accuracy - mean confidence|` per bin, weighted by bin size. 0 means
  the confidence can be trusted; it is reported, not thresholded.

## Blocks and thresholds

Trials are grouped in blocks of 10 (`ScoreConfig.block_size`, or closed
explicitly with `ProgressTracker.close_block()`, for instance on `block_end`).
A block **passes** when:

| Criterion | Default |
|---|---|
| accuracy of every command (each class must be seen in the block) | >= 70 % |
| false start rate | <= 10 % |

`stable_blocks` counts consecutive passed blocks; a failed block resets it to
0. The user is `ready` when `stable_blocks` >= 2.

## Why it matters for safety

* A per-command threshold, not an average, prevents a user who masters `left`
  but not `right` from steering a robot toward an obstacle.
* The false start limit protects against movement while the user is idle: a
  command the user did not intend is the main risk with a real object.
* Consecutive blocks guard against a lucky block: readiness must be reproduced.
* Objects still stop when the decoder status disappears (2 s); the score does
  not replace that rule.

## Usage

```python
tracker = ProgressTracker(("left", "right", "rest"))
block = tracker.add_trial(trial, decisions)   # BlockScore when a block closes
message = tracker.to_message("P03")           # openmind_protocol.Score
store.append_score("P03", tracker.to_record("P03"))
```

The tracker is not wired into the online loop yet.
