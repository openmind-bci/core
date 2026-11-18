"""Public motor-imagery datasets, restricted to the OpenMind montage.

See :func:`openmind_core.datasets.moabb.load_epochs`.
"""

from openmind_core.datasets.epochs import EpochSet
from openmind_core.datasets.moabb import (
    DATASETS,
    MONTAGE_A_MI,
    DatasetSpec,
    epochs_from_raw,
    load_epochs,
)

__all__ = ["DATASETS", "MONTAGE_A_MI", "DatasetSpec", "EpochSet", "epochs_from_raw", "load_epochs"]
