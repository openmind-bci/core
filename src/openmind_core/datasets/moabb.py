"""Load public motor-imagery datasets through MOABB, ready for decoding.

For every run of a subject the processing is:

1. keep the motor-imagery electrodes of **montage A** (``FC3, FC4, C3, Cz, C4,
   CP3, CP4``), in that order. ``Fp1``, the eighth electrode of montage A, is
   used by the blink detector only and is absent from most motor-imagery
   datasets, so it is not part of the decoding channels;
2. convert to microvolts and resample to 250 Hz;
3. filter **causally** (8-30 Hz band-pass and a notch at the dataset's power-line
   frequency) with :class:`~openmind_core.preprocessing.filters.StreamFilter`,
   exactly as the online decoder will;
4. cut 2 s epochs starting 0.5 s after each cue;
5. reject epochs above 150 µV peak to peak.

Downloads go to ``$OPENMIND_DATA_DIR/mne_data`` and need the ``research`` extra
(MOABB). The global MNE configuration is left untouched.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import mne
import numpy as np

from openmind_core.datasets.epochs import EpochSet
from openmind_core.paths import data_dir
from openmind_core.preprocessing.artifacts import REJECT_PTP_UV, amplitude_mask
from openmind_core.preprocessing.filters import FilterConfig, StreamFilter
from openmind_core.preprocessing.windows import EPOCH_OFFSET_S, WINDOW_S, epochs_at

#: Decoding channels: montage A without Fp1, in montage order.
MONTAGE_A_MI: Final[tuple[str, ...]] = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")

#: Target sampling rate of every dataset.
TARGET_SRATE: Final = 250.0

#: Dataset event names → OpenMind commands. Other events (tongue, both hands) are dropped.
EVENT_TO_COMMAND: Final[dict[str, str]] = {
    "left_hand": "left",
    "right_hand": "right",
    "feet": "forward",
    "rest": "rest",
}


@dataclass(frozen=True)
class DatasetSpec:
    """A supported dataset."""

    key: str
    moabb_class: str
    mne_sign: str
    default_classes: tuple[str, ...]
    description: str


DATASETS: Final[dict[str, DatasetSpec]] = {
    "bnci2014_001": DatasetSpec(
        key="bnci2014_001",
        moabb_class="BNCI2014_001",
        mne_sign="BNCI",
        default_classes=("left", "right", "forward"),
        description="BCI Competition IV 2a: 9 subjects, 2 sessions, 250 Hz, Graz (50 Hz mains).",
    ),
    "physionetmi": DatasetSpec(
        key="physionetmi",
        moabb_class="PhysionetMI",
        mne_sign="EEGBCI",
        default_classes=("left", "right", "forward"),
        description="PhysioNet EEGMMIDB: 109 subjects, 160 Hz, United States (60 Hz mains).",
    ),
}


def load_epochs(
    dataset: str,
    subjects: Sequence[int],
    classes: Sequence[str] | None = None,
    channels: Sequence[str] = MONTAGE_A_MI,
    srate: float = TARGET_SRATE,
    filters: FilterConfig | None = None,
    reject_uv: float | None = REJECT_PTP_UV,
) -> EpochSet:
    """Load, preprocess and epoch subjects of a public dataset.

    Args:
        dataset: Key of :data:`DATASETS` (``"bnci2014_001"``, ``"physionetmi"``).
        subjects: Subject numbers, as numbered by the dataset.
        classes: Commands to keep (default: left, right, forward).
        channels: Electrodes to keep, in order.
        srate: Target sampling rate.
        filters: Filter settings; by default 8-30 Hz with a notch at the dataset's
            power-line frequency.
        reject_uv: Peak-to-peak rejection threshold, or ``None`` to keep everything.

    Raises:
        KeyError: If the dataset is unknown.
        ImportError: If MOABB (``research`` extra) is not installed.
    """
    spec = _spec(dataset)
    try:
        import moabb.datasets
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("loading public datasets needs the research extra (MOABB)") from error

    moabb_dataset = getattr(moabb.datasets, spec.moabb_class)()
    cue_offset_s = float(moabb_dataset.interval[0])
    wanted = tuple(classes or spec.default_classes)
    sets: list[EpochSet] = []
    with _mne_data_paths(spec.mne_sign):
        data = moabb_dataset.get_data(subjects=list(subjects))
    for subject in subjects:
        for session, runs in sorted(data[subject].items()):
            for raw in runs.values():
                sets.append(
                    epochs_from_raw(
                        raw,
                        subject=subject,
                        session=session,
                        dataset=spec.key,
                        classes=wanted,
                        channels=channels,
                        srate=srate,
                        filters=filters,
                        reject_uv=reject_uv,
                        cue_offset_s=cue_offset_s,
                        event_id=moabb_dataset.event_id,
                    )
                )
    result = EpochSet.concatenate([s for s in sets if len(s)] or sets[:1])
    result.info["rejected"] = sum(int(s.info.get("rejected", 0)) for s in sets)
    return result


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    subject: int,
    session: str,
    dataset: str,
    classes: Sequence[str],
    channels: Sequence[str] = MONTAGE_A_MI,
    srate: float = TARGET_SRATE,
    filters: FilterConfig | None = None,
    reject_uv: float | None = REJECT_PTP_UV,
    cue_offset_s: float = 0.0,
    event_map: Mapping[str, str] = EVENT_TO_COMMAND,
    event_id: Mapping[str, Any] | None = None,
) -> EpochSet:
    """Preprocess one continuous run and cut labelled epochs.

    Events are read like MOABB reads them: from the stim channel when ``raw``
    has one and ``event_id`` (event name → stim code) is given, otherwise from
    the annotations of ``raw``. Their names are mapped with ``event_map`` and
    only ``classes`` are kept. The cue is ``cue_offset_s`` after each event: the
    stim channel of BNCI2014-001 marks the start of each trial, 2 s before the
    cue (the dataset's ``interval[0]``).

    The two sources are not interchangeable: since MOABB 1.x the annotations of
    BNCI2014-001 are already placed on the cue, so adding ``interval[0]`` to them
    would cut the epochs 2 s too late.

    Raises:
        ValueError: If a channel is missing from the run.
    """
    picks = _pick_channels(raw, channels)
    run = raw.copy().pick(picks)
    run.load_data(verbose=False)
    original_srate = float(run.info["sfreq"])
    if not np.isclose(original_srate, srate):
        run.resample(srate, verbose=False)
    data_uv = run.get_data().T * 1e6

    line_freq = raw.info.get("line_freq") or 50.0
    config = filters or FilterConfig(notch_hz=float(line_freq))
    filtered = StreamFilter(srate, len(channels), config)(data_uv)

    onsets: list[float] = []
    labels: list[str] = []
    for onset, name in _events(raw, event_id):
        command = event_map.get(name)
        if command is not None and command in classes:
            onsets.append((onset + cue_offset_s) * srate)
            labels.append(command)
    epochs, kept = epochs_at(filtered, onsets, srate, EPOCH_OFFSET_S, WINDOW_S)
    y = np.array(labels, dtype=np.str_)[kept]

    mask = amplitude_mask(epochs, reject_uv) if reject_uv is not None else np.ones(len(y), bool)
    n = int(mask.sum())
    result = EpochSet(
        X=epochs[mask].astype(np.float32),
        y=y[mask],
        subject=np.full(n, subject, dtype=np.int64),
        session=np.array([session] * n, dtype=np.str_).reshape(n),
        channels=tuple(channels),
        srate=srate,
        dataset=dataset,
        info={"filters": asdict(config), "rejected": int((~mask).sum())},
    )
    return result


def _events(raw: mne.io.BaseRaw, event_id: Mapping[str, Any] | None) -> list[tuple[float, str]]:
    """Events of a run as ``(seconds from the first sample, name)``, in time order."""
    codes = {code: name for name, code in (event_id or {}).items() if isinstance(code, int)}
    if codes and len(mne.pick_types(raw.info, meg=False, stim=True)):
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        srate = float(raw.info["sfreq"])
        return [
            ((int(sample) - raw.first_samp) / srate, codes[int(code)])
            for sample, _, code in events
            if int(code) in codes
        ]
    return [
        (float(onset) - raw.first_time, str(description))
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description, strict=True
        )
    ]


def _spec(dataset: str) -> DatasetSpec:
    key = dataset.strip().lower().replace("-", "_")
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; supported: {sorted(DATASETS)}")
    return DATASETS[key]


def _pick_channels(raw: mne.io.BaseRaw, channels: Sequence[str]) -> list[str]:
    def clean(name: str) -> str:
        return name.strip().rstrip(".").removeprefix("EEG ").strip().casefold()

    by_name = {clean(name): name for name in raw.ch_names}
    missing = [label for label in channels if clean(label) not in by_name]
    if missing:
        raise ValueError(f"the recording lacks channels {missing}")
    return [by_name[clean(label)] for label in channels]


@contextmanager
def _mne_data_paths(sign: str) -> Iterator[Path]:
    """Point MNE/MOABB downloads to ``$OPENMIND_DATA_DIR/mne_data`` for a while."""
    path = data_dir(create=True) / "mne_data"
    path.mkdir(parents=True, exist_ok=True)
    keys = ["MNE_DATA", f"MNE_DATASETS_{sign}_PATH"]
    saved: dict[str, Any] = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.setdefault(key, str(path))
    try:
        yield path
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
