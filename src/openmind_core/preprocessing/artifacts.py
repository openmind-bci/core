"""Simple artifact rejection by amplitude."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

#: Peak-to-peak amplitude (µV, after band-pass) above which an epoch is rejected.
REJECT_PTP_UV: Final = 150.0


def amplitude_mask(
    epochs: npt.ArrayLike, max_ptp_uv: float = REJECT_PTP_UV
) -> npt.NDArray[np.bool_]:
    """Return ``True`` for epochs ``(n, n_channels, n_times)`` to keep.

    An epoch is rejected when any channel exceeds ``max_ptp_uv`` peak to peak, or
    contains a non-finite value.
    """
    array = np.asarray(epochs, dtype=np.float64)
    finite = np.all(np.isfinite(array), axis=(1, 2))
    ptp = np.ptp(np.nan_to_num(array), axis=2).max(axis=1) if array.size else np.zeros(0)
    return np.asarray(finite & (ptp <= max_ptp_uv), dtype=np.bool_)
