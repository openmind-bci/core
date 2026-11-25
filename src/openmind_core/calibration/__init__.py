"""Calibration of a new user: guided live session and offline calibration curve.

Live session: epochs cut from the stream (:class:`CalibrationRecorder`), staged
model update (:class:`PersonalModel`) and resume mode (:class:`ResumeProtocol`).

Offline curve: how many trials does a decoder need? The experiment behind the
main question of OpenMind. A *new user* sits down for one session; the first
``n`` trials per class are their calibration set and the trials that follow are
the test set. Four ways of building the decoder are
compared (see :mod:`openmind_core.calibration.curve` for the exact protocol):

- ``generic``: a model trained on other users, applied as is;
- ``aligned``: the same, after aligning every session on its own reference
  covariance (the reference uses the new user's signals, never their labels);
- ``aligned+tuned``: ``aligned``, plus the user's calibration trials added to the
  training set with a larger weight;
- ``scratch``: a model trained on the calibration trials only.
"""

from openmind_core.calibration.adaptive import PERSONAL_MODEL, PersonalModel
from openmind_core.calibration.curve import (
    ALIGNED,
    CONDITIONS,
    DEFAULT_N_PER_CLASS,
    GENERIC,
    SCRATCH,
    TUNED,
    CurveScore,
    CurveSummary,
    Split,
    align_with_reference,
    calibration_curve,
    fit_weighted,
    run_condition,
    split_calibration,
    summarize_curve,
)
from openmind_core.calibration.resume import ResumeProtocol, ResumeResult
from openmind_core.calibration.session import (
    CalibrationRecorder,
    CalibrationTrial,
    EpochBuffer,
)

__all__ = [
    "ALIGNED",
    "CONDITIONS",
    "DEFAULT_N_PER_CLASS",
    "GENERIC",
    "PERSONAL_MODEL",
    "SCRATCH",
    "TUNED",
    "CalibrationRecorder",
    "CalibrationTrial",
    "CurveScore",
    "CurveSummary",
    "EpochBuffer",
    "PersonalModel",
    "ResumeProtocol",
    "ResumeResult",
    "Split",
    "align_with_reference",
    "calibration_curve",
    "fit_weighted",
    "run_condition",
    "split_calibration",
    "summarize_curve",
]
