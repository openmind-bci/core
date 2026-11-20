# Research note: EEG foundation models and datasets for fast calibration

Prepared for `CORE-060` (fast calibration), November 2026. Sources at the end;
claims marked *(to verify)* were not checked against the original paper or
files.

## Question

Can a large model pre-trained on thousands of hours of EEG, then fine-tuned on a
few minutes of a new user's data, beat OpenMind's small decoders (EEGNet,
tangent space) on **eight motor electrodes at 250 Hz, on a laptop**? And which
public datasets should feed pre-training and evaluation?

## Summary

- **Several foundation models are usable today**: LaBraM, CBraMod, EEGPT, BIOT,
  MIRepNet, REVE... Braindecode 2.0, already a dependency of the `deep` extra,
  implements them and loads their published weights with `from_pretrained`
  (re-hosted by braindecode on Hugging Face). Trying them costs little code.
- **They are not a sure win for motor imagery.** Independent 2025-2026
  benchmarks find that a well-tuned EEGNet matches or beats several of them on
  motor-imagery datasets, especially when only the output layer is adapted;
  published gains depend heavily on the evaluation protocol.
- **Personal calibration on top of them helps, modestly, and not always across
  days**: small personal adapters on a frozen model gain 1.5-5.4 points on the
  same day; the next day the gain survives for LaBraM and REVE but vanishes for
  CBraMod (Tao and Chen, 2026).
- **Recommendation**: add two foundation models as competitors in `CORE-060`,
  **MIRepNet** (built for motor imagery, small, channel template for arbitrary
  headsets) and **LaBraM** (generic, channels named individually, gains that
  last across sessions), against EEGNet and tangent space with alignment, all on
  montage C and the same calibration budgets. Report the result honestly,
  whichever way it goes: that comparison is itself a contribution.
- **Licences**: the code is MIT-licensed; the weights inherit the terms of their
  pre-training corpora (mostly the TUH EEG corpus, research use after
  registration). Fine for research and the thesis; never committed to a
  repository, never redistributed (nor fine-tuned versions) without checking.

## Candidate models

| Model | Venue | Size | Pre-training data | Electrodes | Weights | Fit for OpenMind |
|---|---|---|---|---|---|---|
| **EEGNet** (reference) | J. Neural Eng. 2018 | ≈ 2 200 weights (7 ch, 2 classes) | ours: public MI datasets | fixed list | our registry | baseline: 73.7 % new user, left/right, montage A |
| **LaBraM** | ICLR 2024 | 5.8 M (Base), 46 M, 369 M | ≈ 2 500 h, ≈ 20 datasets | any subset of its 128-channel list, **named** | `braindecode/labram-pretrained`; code MIT | strong candidate: channel names map directly to montage C |
| **CBraMod** | ICLR 2025 | ≈ 4.0 M | TUH EEG corpus (thousands of hours) | channel-agnostic patches (criss-cross attention) | `braindecode/cbramod-pretrained`; code MIT | strong reported scores; 1 s patches at 200 Hz (resample); personal gains do not carry over across sessions |
| **EEGPT** | NeurIPS 2024 | ≈ 10 M *(to verify)* | 246 h, 5 datasets | 62-channel template, 250 Hz | `braindecode/eegpt-pretrained` | weaker on BCI IV 2a in some comparisons |
| **BIOT** | NeurIPS 2023 | small | clinical (seizures, sleep), 16-18 channels | fixed clinical montages | `braindecode/biot-pretrained-*` | poor fit: clinical montages, not MI |
| **MIRepNet** | Knowledge-Based Systems 2026 | 5.2 M | 7 MI datasets | **channel template for arbitrary headsets** | Hugging Face `starself/MIRepNet`; code MIT | best fit on paper: made for MI, claims fine-tuning with < 30 trials per class |
| REVE, LUNA | 2025-2026 | large | large corpora | flexible | in braindecode | LUNA weights CC-BY-ND (no derivatives): avoid |

Reported numbers are hard to compare: on BCI Competition IV 2a (four classes),
the CBraMod paper reports 51.4 % balanced accuracy against 48.7 % for LaBraM;
another paper (CoMET) reports 55.9 % (CBraMod), 52.5 % (LaBraM) and 38.4 %
(EEGPT). Splits, epoch windows and fine-tuning differ from paper to paper and
from our protocols (causal filtering, 2 s windows, 7-8 electrodes).

## What independent evaluations say

- **EEG-FM-Audit (2026)**: a well-optimized EEGNet surpasses several
  foundation models on BCI IV 2b motor imagery, with a fraction of the
  parameters; reported gains may be inflated by weak baselines.
- **Beyond Accuracy (2026)**: with head-only adaptation, every foundation model
  stays below EEGNet; full fine-tuning closes most of the gap.
- **OmniEEG-Bench (2026)**: with full fine-tuning, nine models out of the
  benchmark outperform EEGNet on average rank, but many do not beat a
  task-specific baseline (EEGConformer).
- **EEG-FM-Compass (National Science Review, 2026)**: 12 open models, 13
  datasets; EEGNet, with only thousands of parameters, remains strong overall.
- **Fine-tuning strategy matters**: LoRA is the worst strategy on BCI IV 2a in
  NeuroTTT, competitive in EEG-FM-Bench when pre-training already helps; a 2026
  study adapts LaBraM with LoRA for "plug-and-play" MI *(results to verify)*.
- **Calibration of new users** (Tao and Chen, 2026; CBraMod, REVE, LaBraM;
  235 held-out subjects): personal adapters gain 1.5-5.4 points over the
  population model on the same day; on another day LaBraM keeps 83 % of the gain
  and REVE 61 %, CBraMod none. Same-day recalibration stays best. Their
  evaluation advice: compare each personal model with the **population model**
  and with **other users' adapters**, and test **across sessions**.

None of these evaluations targets eight electrodes and dry electrodes: our
setting remains to be measured.

## Datasets with the eight motor electrodes of montage C

Needed electrodes: `FC3, FC4, C3, Cz, C4, CP3, CP4, CPz`. *Verified* means that
OpenMind loaded them; *expected* means the published montage should contain
them (to verify when loading).

| Dataset (MOABB) | Subjects | Sessions | Classes | Montage C | Use |
|---|---|---|---|---|---|
| BNCI2014_001 | 9 | 2 | left, right, **feet**, tongue | verified | benchmark reference, cross-session |
| PhysionetMI | 109 | 1 | left, right, **feet**, hands, **rest** | verified | large pre-training pool; `rest` for false starts |
| BNCI2015_001 | 12 | 2 | right, **feet** | expected (13 central electrodes) | `forward` (feet), cross-session |
| Schirrmeister2017 | 14 | 1 | left, right, **feet**, **rest** | expected (128 ch) | many trials per subject (≈ 1 000) |
| Lee2019_MI | 54 | 2 | left, right | expected (62 ch) | pre-training, cross-session |
| Cho2017 | 52 | 1 | left, right | expected (64 ch) | pre-training |
| Stieger2021 | 62 | up to 11 | left, right, both, **rest** | expected (62 ch) | long-term use, session-to-session drift |
| Dreyer2023 | 87 | 1 | left, right | expected (27 central ch) | pre-training, users with low performance |
| Weibo2014 | 10 | 1 | left, right, **feet**, hands, rest... | expected (60 ch) | `forward` |
| Zhou2020 | 20 | 7 | left, right, **feet**, **rest** | to check | `forward`, sessions |

Besides more subjects for pre-training, two gaps of the current benchmarks can
be closed with these datasets: the **`rest` class** (PhysionetMI,
Schirrmeister2017, Stieger2021, Zhou2020), needed to measure false starts, and
**`forward`** with more than nine subjects (BNCI2015_001, Schirrmeister2017,
Weibo2014).

## Proposed plan for CORE-060

1. **Data**: extend the loader to BNCI2015_001, Schirrmeister2017, Lee2019_MI,
   Cho2017 and Stieger2021 with montage C, checking every channel list.
2. **Competitors**, all on montage C, causal 8-30 Hz, 2 s windows: tangent space
   + Riemannian re-centering (`CORE-050`); EEGNet pre-trained on the pool then
   fine-tuned; **MIRepNet** fine-tuned; **LaBraM** fine-tuned (head only,
   LoRA, full).
3. **Protocol**: leave-subjects-out pre-training, then 0, 5, 10, 20, 40
   calibration trials per class; report each personal model against the
   population model and against other users' adapters, and across sessions
   (BNCI2014_001, Lee2019_MI, Stieger2021), as Tao and Chen recommend.
4. **Cost**: training time on the laptop (MPS / CPU) and inference latency per
   2 s window: a model that cannot calibrate in minutes or decode in real time
   does not fit OpenMind, whatever its accuracy.

## Sources

- LaBraM: Jiang et al., "Large Brain Model for Learning Generic Representations
  with Tremendous EEG Data in BCI", ICLR 2024. <https://arxiv.org/abs/2405.18765>,
  <https://github.com/935963004/LaBraM>
- CBraMod: Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG
  Decoding", ICLR 2025. <https://arxiv.org/abs/2412.07236>,
  <https://github.com/wjq-learning/CBraMod>
- MIRepNet: "MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor
  Imagery Classification", Knowledge-Based Systems, 2026.
  <https://arxiv.org/abs/2507.20254>, <https://github.com/staraink/MIRepNet>
- CoMET (comparison table on BCI IV 2a): <https://arxiv.org/abs/2509.00314>
- Braindecode, loading pretrained foundation models:
  <https://braindecode.org/stable/auto_examples/model_building/plot_load_pretrained_models.html>
- EEG-FM-Audit: <https://www.alphaxiv.org/abs/2605.26910>
- Beyond Accuracy: <https://arxiv.org/abs/2605.17562>
- OmniEEG-Bench: <https://arxiv.org/abs/2606.00815>
- EEG-FM-Compass (press release):
  <https://www.eurekalert.org/news-releases/1140272>
- NeuroTTT: <https://arxiv.org/abs/2509.26301>
- EEG-FM-Bench: <https://arxiv.org/abs/2508.17742>
- Parameter-efficient fine-tuning of EEG foundation models for plug-and-play MI
  BCIs (VU Amsterdam):
  <https://research.vu.nl/en/publications/parameter-efficient-fine-tuning-of-eeg-foundation-models-for-plug/>
- Tao and Chen, "Do Personal Adapters Carry Over Across Sessions? Separating
  Personal From Population Gains When Calibrating EEG Foundation Models for
  Motor Imagery", 2026. <https://arxiv.org/abs/2609.34801>
- NeuroTechX collection of EEG foundation models:
  <https://huggingface.co/collections/NeuroTechX/eeg-foundation-models>
