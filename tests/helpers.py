"""Shared test helpers (importable because ``tests`` is on the pytest path)."""

from __future__ import annotations

import numpy as np

from openmind_core.datasets import MONTAGE_A_MI, EpochSet

CLASSES = ("left", "right", "forward")


def synthetic_epochs(
    subjects: tuple[int, ...] = (1, 2),
    sessions: tuple[str, ...] = ("0train", "1test"),
    per_class: int = 20,
    separation: float = 3.0,
    seed: int = 0,
) -> EpochSet:
    """Epochs whose class sets which channels carry extra variance.

    Mimics motor imagery: each class changes the power of a different group
    of channels, which is what covariance-based decoders pick up.
    """
    rng = np.random.default_rng(seed)
    n_channels, n_times = len(MONTAGE_A_MI), 250
    patterns = {
        "left": np.array([0, 0, 0, 0, 1, 0, 1], float),  # C4, CP4
        "right": np.array([1, 0, 1, 0, 0, 1, 0], float),  # FC3, C3, CP3
        "forward": np.array([0, 0, 0, 1, 0, 0, 0], float),  # Cz
    }
    X, y, subject, session = [], [], [], []
    for s in subjects:
        mixing = np.eye(n_channels) + 0.1 * rng.normal(size=(n_channels, n_channels))
        for sess in sessions:
            for label in CLASSES:
                for _ in range(per_class):
                    noise = rng.normal(0, 5, (n_channels, n_times))
                    source = rng.normal(0, 5 * separation, n_times)
                    X.append(mixing @ (noise + np.outer(patterns[label], source)))
                    y.append(label)
                    subject.append(s)
                    session.append(sess)
    return EpochSet(
        X=np.array(X, dtype=np.float32),
        y=np.array(y, dtype=np.str_),
        subject=np.array(subject, dtype=np.int64),
        session=np.array(session, dtype=np.str_),
        channels=MONTAGE_A_MI,
        srate=125.0,
        dataset="synthetic",
    )
