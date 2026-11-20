"""Jaw clench as the action gesture (CORE-047): is it readable on motor electrodes?

    uv sync --extra research
    uv run python research/experiments/jaw.py [--subjects 1 2 3] [--run 1]

Data: PhysioMotion Artifact (OpenNeuro ds006386, CC0): 30 participants, 59
channels (10-10), 1000 Hz, with cued jaw clenches (2 blocks of 5 per run) and
annotated look-alikes (articulation "la-la-la", swallowing, eyebrow raising,
head and eye movements, blinks, rest). One run per subject contains every task;
the needed files are downloaded to $OPENMIND_DATA_DIR/datasets/ds006386
(about 150 MB per run).

Method: signals resampled to 250 Hz (the IronBCI-8 rate), muscle band 30-110 Hz
with notches at 50 and 100 Hz, power envelope over 100 ms. Each channel's power
is compared with its own running median (previous 10 s, causal); the score is
the median of these ratios (dB) over the channels, so a clench must raise the
power on most electrodes at once. A clench is a score above the threshold for
at least 100 ms; a double clench is two clench onsets 0.25 to 1.2 s apart.

Writes research/results/jaw.md for run 1, where the settings were explored,
and research/results/jaw-run<N>.md for another run (held-out validation).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import logging
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
from benchmark import ROOT, code_version, package_versions
from scipy import ndimage, signal

from openmind_core.paths import data_dir
from openmind_core.preprocessing import JawClenchDetector, JawConfig

RESULTS_DIR = ROOT / "research" / "results"
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds006386"
SRATE = 250.0
# The motor electrodes of montage A. CPz, the eighth motor electrode, is the
# recording reference of this dataset, so it has no signal of its own here.
MOTOR_7 = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
CHANNEL_SETS = {
    "7 motor electrodes (montage A)": MOTOR_7,
    "C3 + C4 only": ("C3", "C4"),
    "Fp1 alone (reference)": ("Fp1",),
}
THRESHOLDS_DB = (6.0, 10.0, 15.0, 20.0)
# Stricter variants: a burst starts at ONSET_DB but must peak above a level and
# stay short. Deliberate clenches are strong; swallowing is weaker and longer.
ONSET_DB = 10.0
STRICT = tuple((peak, duration) for peak in (20.0, 25.0) for duration in (1.0, 0.8))
DETAIL_DB = 10.0  # threshold of the per-event breakdown
CLENCHES_PER_BLOCK = 5
MIN_BURST_S = 0.1
MERGE_GAP_S = 0.15
DOUBLE_GAP_S = (0.25, 1.2)
BACKGROUND_S = 10.0
MARGIN_S = 1.0  # around clench blocks, detections are neither hits nor false

logger = logging.getLogger("jaw")


@dataclass
class Run:
    """One recording: channels in µV at 250 Hz and annotated intervals."""

    subject: int
    data: dict[str, np.ndarray]
    intervals: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    scores: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)

    def score(self, channels: tuple[str, ...]) -> np.ndarray:
        """Muscle score of a channel set, computed once."""
        if channels not in self.scores:
            self.scores[channels] = muscle_score([self.data[c] for c in channels])
        return self.scores[channels]

    @property
    def minutes(self) -> float:
        return len(next(iter(self.data.values()))) / SRATE / 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    start = time.perf_counter()
    runs = []
    for subject in args.subjects:
        try:
            runs.append(load_run(subject, args.run))
        except (OSError, ValueError) as error:
            logger.warning("skip subject %s: %s", subject, error)
    if not runs:
        raise SystemExit("no usable run")
    report(runs, args.run, time.perf_counter() - start)


# -- Data ---------------------------------------------------------------------------


def cache_dir() -> Path:
    return data_dir(create=True) / "datasets" / "ds006386"


def fetch(url: str, target: Path, attempts: int = 4) -> Path:
    """Download once to ``target``, with a socket timeout and retries."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        logger.info("download %s (attempt %d)", url, attempt)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            partial.rename(target)
            return target
        except OSError as error:  # includes timeouts and connection resets
            logger.warning("download failed: %s", error)
            partial.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    raise OSError(f"could not download {url}")


def load_run(subject: int, run: int) -> Run:
    name = f"sub-{subject}_task-artifact_run-{run:02d}_eeg.edf"
    edf = fetch(f"{BASE_URL}/sub-{subject}/eeg/{name}", cache_dir() / name)
    notes = fetch(
        f"{BASE_URL}/derivatives/Manual_Annotations/sub{subject}_run{run:02d}.csv",
        cache_dir() / f"sub{subject}_run{run:02d}_annotations.csv",
    )
    wanted = sorted({c for channels in CHANNEL_SETS.values() for c in channels})
    raw = mne.io.read_raw_edf(edf, include=wanted, preload=True, verbose=False)
    missing = set(wanted) - set(raw.ch_names)
    if missing:
        raise ValueError(f"missing channels {sorted(missing)}")
    raw.resample(SRATE, verbose=False)
    data = {name: raw.get_data(picks=[name])[0] * 1e6 for name in wanted}
    return Run(subject=subject, data=data, intervals=read_intervals(notes))


def read_intervals(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Annotated intervals per label, merged over channels."""
    by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    text = path.read_text(encoding="utf-8-sig").replace("\r", "")
    for row in csv.DictReader(io.StringIO(text)):
        by_label[row["label"].strip()].append((float(row["start_time"]), float(row["stop_time"])))
    return {label: merge(spans) for label, spans in by_label.items()}


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, stop in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


# -- Detector ----------------------------------------------------------------------------


def muscle_score(channels: list[np.ndarray]) -> np.ndarray:
    """Median over channels of the muscle power above each channel's background, in dB."""
    sos = signal.butter(4, (30.0, 110.0), btype="bandpass", fs=SRATE, output="sos")
    for line in (50.0, 100.0):
        b, a = signal.iirnotch(line, 30.0, fs=SRATE)
        sos = np.vstack([sos, signal.tf2sos(b, a)])
    window = round(0.1 * SRATE)
    step = round(0.25 * SRATE)
    ratios = []
    for x in channels:
        band = signal.sosfilt(sos, x - x[0])
        power = np.convolve(band**2, np.ones(window) / window)[: len(band)] + 1e-12
        log_power = 10 * np.log10(power)
        # Causal running median of the previous BACKGROUND_S, refreshed every 0.25 s.
        history = round(BACKGROUND_S * SRATE)
        background = np.empty_like(log_power)
        for i in range(0, len(log_power), step):
            past = log_power[max(0, i - history) : max(i, 1)]
            background[i : i + step] = np.median(past)
        ratios.append(log_power - background)
    return np.median(np.vstack(ratios), axis=0)


def bursts(
    score: np.ndarray,
    threshold_db: float,
    min_peak_db: float | None = None,
    max_duration_s: float | None = None,
) -> list[float]:
    """Onset times (s) of bursts above the threshold.

    Optionally keep only bursts that peak above ``min_peak_db`` and last at
    most ``max_duration_s``.
    """
    above = score > threshold_db
    # Bridge short gaps, then keep long enough segments.
    above = ndimage.binary_closing(above, structure=np.ones(round(MERGE_GAP_S * SRATE)))
    labels, n = ndimage.label(above)
    onsets = []
    for index in range(1, n + 1):
        where = np.flatnonzero(labels == index)
        if len(where) < MIN_BURST_S * SRATE:
            continue
        if min_peak_db is not None and score[where].max() < min_peak_db:
            continue
        if max_duration_s is not None and len(where) > max_duration_s * SRATE:
            continue
        onsets.append(where[0] / SRATE)
    return onsets


def double_clenches(onsets: list[float]) -> list[float]:
    """Times of double clenches: two onsets DOUBLE_GAP_S apart, non-overlapping."""
    found, i = [], 0
    while i + 1 < len(onsets):
        gap = onsets[i + 1] - onsets[i]
        if DOUBLE_GAP_S[0] <= gap <= DOUBLE_GAP_S[1]:
            found.append(onsets[i + 1])
            i += 2
        else:
            i += 1
    return found


# -- Evaluation --------------------------------------------------------------------------


def inside(t: float, spans: list[tuple[float, float]], margin: float = 0.0) -> bool:
    return any(start - margin <= t <= stop + margin for start, stop in spans)


def evaluate(
    runs: list[Run],
    channels: tuple[str, ...],
    threshold: float,
    min_peak_db: float | None = None,
    max_duration_s: float | None = None,
) -> dict[str, float]:
    hits = expected = extra = false = false_double = 0
    minutes = 0.0
    triggered: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # label -> [hit, total]
    for run in runs:
        onsets = bursts(run.score(channels), threshold, min_peak_db, max_duration_s)
        chew = run.intervals.get("chew", [])
        for start, stop in chew:
            n = sum(start - 0.5 <= t <= stop + 0.5 for t in onsets)
            hits += min(n, CLENCHES_PER_BLOCK)
            extra += max(n - CLENCHES_PER_BLOCK, 0)
            expected += CLENCHES_PER_BLOCK
        outside = [t for t in onsets if not inside(t, chew, MARGIN_S)]
        false += len(outside)
        false_double += len(double_clenches(outside))
        clench_minutes = sum(stop - start + 2 * MARGIN_S for start, stop in chew) / 60
        minutes += run.minutes - clench_minutes
        for label, spans in run.intervals.items():
            if label == "chew":
                continue
            for start, stop in spans:
                triggered[label][1] += 1
                triggered[label][0] += any(start - 0.2 <= t <= stop + 0.2 for t in outside)
    result = {
        "detected": hits / max(expected, 1),
        "extra_per_block": extra / max(expected / CLENCHES_PER_BLOCK, 1),
        "false_per_min": false / minutes,
        "false_double_per_hour": 60 * false_double / minutes,
        "minutes": minutes,
    }
    for label, (hit, total) in triggered.items():
        result[f"triggered.{label}"] = hit / total
    return result


def evaluate_detector(
    runs: list[Run], channels: tuple[str, ...], config: JawConfig
) -> dict[str, float]:
    """The online detector of openmind_core, run over each recording."""
    hits = expected = extra = false = false_double = 0
    minutes = 0.0
    for run in runs:
        data = np.column_stack([run.data[c] for c in channels])
        detector = JawClenchDetector(SRATE, len(channels), config)
        doubles = detector.update(data)
        chew = run.intervals.get("chew", [])
        for start, stop in chew:
            n = sum(start - 0.5 <= t <= stop + 0.5 for t in detector.clenches)
            hits += min(n, CLENCHES_PER_BLOCK)
            extra += max(n - CLENCHES_PER_BLOCK, 0)
            expected += CLENCHES_PER_BLOCK
        false += sum(not inside(t, chew, MARGIN_S) for t in detector.clenches)
        false_double += sum(not inside(t, chew, MARGIN_S + 1.0) for t in doubles)
        minutes += run.minutes - sum(stop - start + 2 * MARGIN_S for start, stop in chew) / 60
    return {
        "detected": hits / max(expected, 1),
        "extra_per_block": extra / max(expected / CLENCHES_PER_BLOCK, 1),
        "false_per_min": false / minutes,
        "false_double_per_hour": 60 * false_double / minutes,
    }


def detector_section(runs: list[Run]) -> list[str]:
    """Rows for the detector of openmind_core, default and study settings."""
    variants = {
        "default settings (100 ms envelope, 150 ms merge gap)": JawConfig(),
        "finer settings, not kept (50 ms envelope, 50 ms merge gap)": JawConfig(
            power_window_s=0.05, merge_gap_s=0.05
        ),
    }
    lines = [
        "## Detector of openmind_core",
        "",
        "`openmind_core.preprocessing.JawClenchDetector` (causal, block by block) on the "
        "7 motor electrodes: onset 10 dB, peak ≥ 20 dB, burst ≤ 1 s, the settings of the "
        "study above. Finer settings would split double clenches with a release "
        "shorter than 0.3 s, but also split other artifacts into pairs.",
        "",
        "| Settings | Clenches detected | Extra / block | False / min "
        "| False double clenches / hour |",
        "|---|---|---|---|---|",
    ]
    for label, config in variants.items():
        r = evaluate_detector(runs, MOTOR_7, config)
        lines.append(
            f"| {label} | {100 * r['detected']:.0f} % | {r['extra_per_block']:.1f} "
            f"| {r['false_per_min']:.2f} | {r['false_double_per_hour']:.1f} |"
        )
    return [*lines, ""]


def report(runs: list[Run], run_index: int, elapsed: float) -> None:
    blocks = sum(len(r.intervals.get("chew", [])) for r in runs)
    lines = [
        "# Jaw clench on motor electrodes",
        "",
        "Generated by `uv run python research/experiments/jaw.py`. Do not edit by hand.",
        "Data: PhysioMotion Artifact (OpenNeuro ds006386, CC0), run "
        f"{run_index} of {len(runs)} subjects: {blocks} annotated clench blocks of "
        f"{CLENCHES_PER_BLOCK} cued clenches, {sum(r.minutes for r in runs):.0f} minutes. "
        "Discussion in `docs/action-gesture.md`.",
        "",
        "Threshold: a burst starts when the score exceeds it. Strict variants start "
        f"at {ONSET_DB:.0f} dB but keep only bursts that peak above a level and stay short. "
        "Detected: share of the cued clenches found inside the clench blocks. "
        "Extra: detections beyond five per block. False: detections outside the "
        f"clench blocks (± {MARGIN_S:.0f} s); false double clenches are false `action` "
        "commands.",
        "",
    ]
    for set_name, channels in CHANNEL_SETS.items():
        lines += [
            f"## {set_name}",
            "",
            "| Threshold | Clenches detected | Extra / block | False / min "
            "| False double clenches / hour |",
            "|---|---|---|---|---|",
        ]
        for threshold in THRESHOLDS_DB:
            r = evaluate(runs, channels, threshold)
            lines.append(
                f"| {threshold:.0f} dB | {100 * r['detected']:.0f} % | {r['extra_per_block']:.1f} "
                f"| {r['false_per_min']:.2f} | {r['false_double_per_hour']:.1f} |"
            )
        for peak, duration in STRICT:
            r = evaluate(runs, channels, ONSET_DB, peak, duration)
            lines.append(
                f"| peak ≥ {peak:.0f} dB, ≤ {duration:g} s | {100 * r['detected']:.0f} % "
                f"| {r['extra_per_block']:.1f} | {r['false_per_min']:.2f} "
                f"| {r['false_double_per_hour']:.1f} |"
            )
        r = evaluate(runs, channels, DETAIL_DB)
        triggered = sorted(
            ((k.split(".", 1)[1], v) for k, v in r.items() if k.startswith("triggered.")),
            key=lambda item: -item[1],
        )
        lines += [
            "",
            f"Annotated events that trigger at least one false detection, at {DETAIL_DB:.0f} dB: "
            + ", ".join(f"{label} {100 * v:.0f} %" for label, v in triggered)
            + ".",
            "",
        ]
    lines += detector_section(runs)
    code = code_version()
    dirty = " (with uncommitted changes)" if code["dirty"] == "true" else ""
    lines += [
        "## Run",
        "",
        f"- Date: {dt.date.today().isoformat()}, {elapsed / 60:.0f} min",
        f"- Code: commit `{code['commit']}`{dirty}",
        "- Versions: "
        + ", ".join(f"{name} {version}" for name, version in package_versions().items()),
        "",
    ]
    output = RESULTS_DIR / ("jaw.md" if run_index == 1 else f"jaw-run{run_index}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    logger.info("results written to %s", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
