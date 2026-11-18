"""Preprocessing shared by offline training and the online decoder."""

from openmind_core.preprocessing.artifacts import REJECT_PTP_UV, amplitude_mask
from openmind_core.preprocessing.blink import (
    BlinkConfig,
    DoubleBlinkDetector,
    detect_double_blinks,
)
from openmind_core.preprocessing.filters import FilterConfig, StreamFilter, filter_continuous
from openmind_core.preprocessing.windows import (
    EPOCH_OFFSET_S,
    STEP_S,
    WINDOW_S,
    SlidingWindow,
    epochs_at,
)

__all__ = [
    "EPOCH_OFFSET_S",
    "REJECT_PTP_UV",
    "STEP_S",
    "WINDOW_S",
    "BlinkConfig",
    "DoubleBlinkDetector",
    "FilterConfig",
    "SlidingWindow",
    "StreamFilter",
    "amplitude_mask",
    "detect_double_blinks",
    "epochs_at",
    "filter_continuous",
]
