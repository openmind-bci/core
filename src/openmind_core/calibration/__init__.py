"""Fast calibration of a new user: how many trials does a decoder need?

The experiment behind the main question of OpenMind. A *new user* sits down for
one session; the first ``n`` trials per class are their calibration set and the
trials that follow are the test set. Four ways of building the decoder are
compared (see :mod:`openmind_core.calibration.curve` for the exact protocol):

- ``generic``: a model trained on other users, applied as is;
- ``aligned``: the same, after aligning every session on its own reference
  covariance (the reference uses the new user's signals, never their labels);
- ``aligned+tuned``: ``aligned``, plus the user's calibration trials added to the
  training set with a larger weight;
- ``scratch``: a model trained on the calibration trials only.

Two more families build on them: ``aligned+finetune-<strategy>`` adapts a deep
model pre-trained on the aligned source to the calibration trials
(:mod:`~openmind_core.calibration.finetune`), and ``coldstart-knn`` trains on the
source subjects nearest to the new user's unlabelled signals only
(:mod:`~openmind_core.calibration.coldstart`).
"""

from openmind_core.calibration.coldstart import (
    nearest_subjects,
    rest_signature,
    subject_signatures,
)
from openmind_core.calibration.curve import (
    ALIGNED,
    ALL_CONDITIONS,
    COLDSTART,
    CONDITIONS,
    DEFAULT_COLDSTART_K,
    DEFAULT_N_PER_CLASS,
    FINETUNE_CONDITIONS,
    GENERIC,
    SCRATCH,
    TUNED,
    CurveScore,
    CurveSummary,
    Split,
    align_with_reference,
    calibration_curve,
    condition_rank,
    finetune_strategy,
    fit_weighted,
    is_condition,
    run_condition,
    split_calibration,
    summarize_curve,
)
from openmind_core.calibration.finetune import DEFAULT_LR, STRATEGIES, FineTuneSettings

__all__ = [
    "ALIGNED",
    "ALL_CONDITIONS",
    "COLDSTART",
    "CONDITIONS",
    "DEFAULT_COLDSTART_K",
    "DEFAULT_LR",
    "DEFAULT_N_PER_CLASS",
    "FINETUNE_CONDITIONS",
    "GENERIC",
    "SCRATCH",
    "STRATEGIES",
    "TUNED",
    "CurveScore",
    "CurveSummary",
    "FineTuneSettings",
    "Split",
    "align_with_reference",
    "calibration_curve",
    "condition_rank",
    "finetune_strategy",
    "fit_weighted",
    "is_condition",
    "nearest_subjects",
    "rest_signature",
    "run_condition",
    "split_calibration",
    "subject_signatures",
    "summarize_curve",
]
