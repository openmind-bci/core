"""Container for labelled epochs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class EpochSet:
    """Labelled epochs from one or more subjects.

    Attributes:
        X: Signals in µV, shape ``(n_epochs, n_channels, n_times)``, float32.
        y: Command labels (``"left"``, ``"right"``, ``"forward"``, ``"rest"``).
        subject: Subject number of each epoch.
        session: Session name of each epoch.
        channels: Channel labels, in the order of ``X``.
        srate: Sampling rate in Hz.
        dataset: Name of the source dataset.
        info: Free metadata (filters, rejection counts...).
    """

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.str_]
    subject: npt.NDArray[np.int64]
    session: npt.NDArray[np.str_]
    channels: tuple[str, ...]
    srate: float
    dataset: str
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.X.ndim != 3 or self.X.shape[1] != len(self.channels):
            raise ValueError(
                f"X must be (n_epochs, {len(self.channels)}, n_times), got {self.X.shape}"
            )
        for name in ("y", "subject", "session"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one entry per epoch ({n})")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def classes(self) -> tuple[str, ...]:
        """Labels present, sorted."""
        return tuple(sorted(set(self.y.tolist())))

    @property
    def subjects(self) -> tuple[int, ...]:
        """Subjects present, sorted."""
        return tuple(sorted(set(self.subject.tolist())))

    def counts(self) -> dict[str, int]:
        """Number of epochs per label."""
        labels, numbers = np.unique(self.y, return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, numbers, strict=True)}

    def select(self, mask: npt.ArrayLike) -> EpochSet:
        """Subset of the epochs where ``mask`` is true."""
        keep = np.asarray(mask, dtype=bool)
        return EpochSet(
            X=self.X[keep],
            y=self.y[keep],
            subject=self.subject[keep],
            session=self.session[keep],
            channels=self.channels,
            srate=self.srate,
            dataset=self.dataset,
            info=dict(self.info),
        )

    @staticmethod
    def concatenate(sets: list[EpochSet]) -> EpochSet:
        """Join epoch sets that share channels, rate and dataset."""
        if not sets:
            raise ValueError("nothing to concatenate")
        first = sets[0]
        for other in sets[1:]:
            if (other.channels, other.srate) != (first.channels, first.srate):
                raise ValueError("epoch sets must share channels and sampling rate")
        return EpochSet(
            X=np.concatenate([s.X for s in sets]),
            y=np.concatenate([s.y for s in sets]),
            subject=np.concatenate([s.subject for s in sets]),
            session=np.concatenate([s.session for s in sets]),
            channels=first.channels,
            srate=first.srate,
            dataset="+".join(dict.fromkeys(s.dataset for s in sets)),
            info=dict(first.info),
        )

    def save(self, path: str | Path) -> Path:
        """Write to a compressed ``.npz`` file (keep it in ``OPENMIND_DATA_DIR``)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            X=self.X,
            y=self.y,
            subject=self.subject,
            session=self.session,
            channels=np.array(self.channels),
            srate=np.array(self.srate),
            dataset=np.array(self.dataset),
        )
        return target

    @staticmethod
    def load(path: str | Path) -> EpochSet:
        """Read a file written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            return EpochSet(
                X=data["X"],
                y=data["y"],
                subject=data["subject"],
                session=data["session"],
                channels=tuple(str(c) for c in data["channels"]),
                srate=float(data["srate"]),
                dataset=str(data["dataset"]),
            )
