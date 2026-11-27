"""Channel labels: matching the spellings used by headsets, files and model cards."""

from __future__ import annotations

__all__ = ["normalize_label"]


def normalize_label(label: str) -> str:
    """Case-insensitive channel key (``Fc3.``, ``EEG FC3`` and ``fc3`` match ``FC3``).

    Strips surrounding spaces, trailing dots and an ``EEG `` prefix, then folds
    the case. Compare keys, never raw labels, when matching a stream to a model.
    """
    return label.strip().rstrip(".").removeprefix("EEG ").strip().casefold()
