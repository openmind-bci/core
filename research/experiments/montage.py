"""Montage study (CORE-045): which 8th motor electrode, and blinks without Fp1.

    uv sync --extra research
    uv run python research/experiments/montage.py [--part channels|blinks|all] [--no-tracking]

Part ``channels``: the seven decoding electrodes of montage A, alone and with
one central candidate (FCz, CPz, C1 or C2) as an eighth, evaluated with the
classical baselines under the three protocols on BNCI2014-001 and PhysionetMI
(left, right, forward). The criterion is the recognition of ``forward``.

Part ``blinks``: on the continuous PhysionetMI recordings, blinks found on Fp1
serve as the reference; the double-blink detector then runs on the mean of FC3
and FC4. Reports the blink amplitude ratio, the share of strong blinks
detected, and the false detections per minute, for several thresholds.

Writes research/results/montage.md (parts not run are kept from the previous
file) and logs to the MLflow experiment "montage" in $OPENMIND_DATA_DIR/mlflow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from benchmark import ROOT, cached_epochs, code_version, package_versions
from scipy import signal, stats

from openmind_core.datasets import MONTAGE_A_MI, iter_runs
from openmind_core.evaluation import PROTOCOLS, Score, drop_small_subjects, evaluate
from openmind_core.models import make_model
from openmind_core.preprocessing.blink import BlinkConfig, DoubleBlinkDetector

RESULTS = ROOT / "research" / "results" / "montage.md"
CANDIDATES = ("FCz", "CPz", "C1", "C2")
MODELS = ("csp-lda", "ts-lr")
SEED = 42
N_SPLITS = 5
MIN_EPOCHS_PER_CLASS = 10
CLASSES = ("left", "right", "forward")
CHANNEL_EXPERIMENTS = (
    {"dataset": "bnci2014_001", "subjects": list(range(1, 10))},
    # Subjects 88, 92, 100 and 104 of PhysionetMI have another sampling rate;
    # the first 50 subjects avoid them.
    {"dataset": "physionetmi", "subjects": list(range(1, 51))},
)
BLINK_SUBJECTS = list(range(1, 51))
# Detector settings tried on FC3/FC4: signal ("mean" of the two channels, or
# "min", which needs both to rise as a symmetric blink does), adaptive factor k,
# minimum amplitude in µV.
FC_CONFIGS = tuple(
    (feature, k, amplitude)
    for feature in ("mean", "min")
    for k in (2.0, 3.0, 5.0)
    for amplitude in (30.0, 50.0)
)
MATCH_S = 0.15  # a detection within 150 ms of a reference blink is the same blink
STRONG_UV = 100.0  # deliberate blinks are typically 100-300 µV on Fp1
VERY_STRONG_UV = 200.0

logger = logging.getLogger("montage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("channels", "blinks", "all"), default="all")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    code = code_version()
    sections = read_sections()
    metrics: dict[str, float] = {}
    if args.part in ("channels", "all"):
        text, values = channels_part()
        sections["channels"] = text
        metrics |= values
    if args.part in ("blinks", "all"):
        text, values = blinks_part()
        sections["blinks"] = text
        metrics |= values
    write_report(sections, code, package_versions())
    if not args.no_tracking:
        track(args.part, metrics, code)
    logger.info("results written to %s", RESULTS.relative_to(ROOT))


# -- Part A: the eighth electrode ----------------------------------------------


def channels_part() -> tuple[str, dict[str, float]]:
    montages = {"montage A (7)": list(MONTAGE_A_MI)} | {
        f"+ {name}": [*MONTAGE_A_MI, name] for name in CANDIDATES
    }
    lines = [
        "## Part A: the eighth motor electrode",
        "",
        f"Models {', '.join(f'`{m}`' for m in MODELS)}, classes {', '.join(CLASSES)} "
        "(chance 33 %). Accuracy and recall of `forward` in %, mean over subjects. "
        "Δ: mean paired difference with montage A over subjects, with the Wilcoxon "
        "signed-rank p-value.",
        "",
    ]
    metrics: dict[str, float] = {}
    for spec in CHANNEL_EXPERIMENTS:
        start = time.perf_counter()
        per_montage: dict[str, list[Score]] = {}
        dropped: list[int] = []
        for label, channels in montages.items():
            experiment = spec | {"classes": list(CLASSES), "channels": channels}
            epochs, dropped = drop_small_subjects(cached_epochs(experiment), MIN_EPOCHS_PER_CLASS)
            per_montage[label] = evaluate(
                epochs,
                {m: (lambda m=m: make_model(m)) for m in MODELS},
                n_splits=N_SPLITS,
                seed=SEED,
            )
        reference = per_montage["montage A (7)"]
        n_subjects = len({s.subject for s in reference})
        lines += [
            f"### {spec['dataset']}",
            "",
            f"{n_subjects} subjects"
            + (f" (subjects {', '.join(map(str, dropped))} excluded)" if dropped else "")
            + f", {format_duration(time.perf_counter() - start)}.",
            "",
        ]
        for protocol in PROTOCOLS:
            if not any(s.protocol == protocol for s in reference):
                continue
            lines += [
                f"**{protocol}**",
                "",
                "| Montage | Model | Accuracy | Δ accuracy (p) | Forward recall | Δ forward (p) |",
                "|---|---|---|---|---|---|",
            ]
            for label, scores in per_montage.items():
                for model in MODELS:
                    acc = per_subject(scores, model, protocol, "accuracy")
                    fwd = per_subject(scores, model, protocol, "forward")
                    ref_acc = per_subject(reference, model, protocol, "accuracy")
                    ref_fwd = per_subject(reference, model, protocol, "forward")
                    lines.append(
                        f"| {label} | `{model}` | {100 * mean(acc):.1f} | "
                        f"{delta(acc, ref_acc)} | {100 * mean(fwd):.1f} | {delta(fwd, ref_fwd)} |"
                    )
                    key = f"{spec['dataset']}.{protocol}.{model}.{label.strip('+ ')}"
                    metrics[f"{key}.accuracy"] = mean(acc)
                    metrics[f"{key}.forward"] = mean(fwd)
            lines.append("")
    return "\n".join(lines), {sanitize(k): v for k, v in metrics.items()}


def per_subject(scores: list[Score], model: str, protocol: str, what: str) -> dict[int, float]:
    """Per-subject value (sessions averaged): accuracy or the recall of a class."""
    values: dict[int, list[float]] = {}
    for s in scores:
        if (s.pipeline, s.protocol) == (model, protocol):
            value = s.accuracy if what == "accuracy" else s.recall.get(what, np.nan)
            values.setdefault(s.subject, []).append(value)
    return {subject: float(np.nanmean(v)) for subject, v in values.items()}


def mean(values: dict[int, float]) -> float:
    return float(np.nanmean(list(values.values())))


def delta(values: dict[int, float], reference: dict[int, float]) -> str:
    subjects = sorted(set(values) & set(reference))
    diff = np.array([values[s] - reference[s] for s in subjects])
    if not np.any(diff):
        return "—"
    p = float(stats.wilcoxon(diff).pvalue) if len(diff) >= 5 else float("nan")
    return f"{100 * diff.mean():+.1f} (p {p:.2f})"


# -- Part B: blinks without Fp1 --------------------------------------------------


@dataclass
class BlinkRun:
    """Blink measurements of one continuous run."""

    minutes: float
    reference: list[tuple[float, float]]  # (time s, Fp1 amplitude µV) of Fp1 blinks
    ratios: list[float]  # FC mean amplitude / Fp1 amplitude at each reference blink
    fp1_doubles: list[float]
    fc_blinks: dict[tuple[str, float, float], list[float]]  # config -> blink times
    fc_doubles: dict[tuple[str, float, float], list[float]]


def blinks_part() -> tuple[str, dict[str, float]]:
    start = time.perf_counter()
    runs: list[BlinkRun] = []
    subjects: set[int] = set()
    for subject, _session, raw in iter_runs("physionetmi", BLINK_SUBJECTS):
        traces = channel_traces(raw, ("Fp1", "FC3", "FC4"))
        if traces is None:
            continue
        runs.append(analyse_run(traces, float(raw.info["sfreq"])))
        subjects.add(subject)
    minutes = sum(r.minutes for r in runs)
    reference = [amp for r in runs for _, amp in r.reference]
    strong = sum(amp >= STRONG_UV for amp in reference)
    very_strong = sum(amp >= VERY_STRONG_UV for amp in reference)
    ratios = np.array([x for r in runs for x in r.ratios])
    fp1_doubles = sum(len(r.fp1_doubles) for r in runs)
    lines = [
        "## Part B: double blink without Fp1",
        "",
        f"PhysionetMI, {len(subjects)} subjects, {len(runs)} imagery runs, "
        f"{minutes:.0f} minutes of signal, {format_duration(time.perf_counter() - start)}. "
        f"Reference: the detector on Fp1 (default thresholds) finds {len(reference)} blinks "
        f"({len(reference) / minutes:.1f} per minute), {strong} of them strong "
        f"(≥ {STRONG_UV:.0f} µV, like a deliberate blink), and {fp1_doubles} double blinks.",
        "",
        "Blink amplitude on the mean of FC3 and FC4, relative to Fp1 (0.1-8 Hz): "
        f"median {np.median(ratios):.2f}, quartiles {np.quantile(ratios, 0.25):.2f}-"
        f"{np.quantile(ratios, 0.75):.2f}.",
        "",
        "Detector on FC3/FC4: signal (`mean` of the two channels, or `min`, which needs "
        "both to rise as a symmetric blink does), adaptive factor k (threshold = the "
        "larger of the minimum amplitude and k times the robust background deviation; "
        "Fp1 uses k = 5, minimum 60 µV). A detection matches a reference blink within "
        f"{1000 * MATCH_S:.0f} ms. False detections are blinks found on FC3/FC4 with no "
        "Fp1 blink nearby; double blinks found on FC3/FC4 with no Fp1 double blink nearby "
        "are false `action` commands.",
        "",
        f"| Signal | k | Minimum | Blinks ≥ {STRONG_UV:.0f} µV found "
        f"| Blinks ≥ {VERY_STRONG_UV:.0f} µV found | False detections / min "
        "| Fp1 double blinks found | False double blinks / hour |",
        "|---|---|---|---|---|---|---|---|",
    ]
    metrics: dict[str, float] = {"blinks.ratio_median": float(np.median(ratios))}
    for config in FC_CONFIGS:
        feature, k, threshold = config
        hit_strong = hit_very = false = 0
        double_hit = false_double = 0
        for r in runs:
            found = r.fc_blinks[config]
            for t, amp in r.reference:
                matched = any(abs(t - f) <= MATCH_S for f in found)
                hit_strong += matched and amp >= STRONG_UV
                hit_very += matched and amp >= VERY_STRONG_UV
            false += sum(not any(abs(f - t) <= MATCH_S for t, _ in r.reference) for f in found)
            doubles = r.fc_doubles[config]
            double_hit += sum(any(abs(d - f) <= MATCH_S for f in doubles) for d in r.fp1_doubles)
            false_double += sum(
                not any(abs(f - d) <= MATCH_S for d in r.fp1_doubles) for f in doubles
            )
        lines.append(
            f"| {feature} | {k:.0f} | {threshold:.0f} µV | {pct(hit_strong, strong)} "
            f"| {pct(hit_very, very_strong)} | {false / minutes:.2f} "
            f"| {pct(double_hit, fp1_doubles)} | {60 * false_double / minutes:.1f} |"
        )
        key = f"blinks.{feature}.k{k:.0f}.{threshold:.0f}uv"
        metrics |= {
            f"{key}.strong_detected": hit_strong / max(strong, 1),
            f"{key}.false_per_min": false / minutes,
            f"{key}.false_doubles_per_hour": 60 * false_double / minutes,
        }
    hours = minutes / 60
    triples = {
        gap: sum(count_triples([t for t, _ in r.reference], gap) for r in runs)
        for gap in (0.6, 0.8)
    }
    lines += [
        "",
        "Spontaneous blinking (no instruction to blink in this dataset), from the Fp1 "
        f"reference: {fp1_doubles / hours:.1f} double blinks per hour (gaps "
        f"{BlinkConfig().min_gap_s}-{BlinkConfig().max_gap_s} s); triple blinks "
        + ", ".join(
            f"{triples[gap] / hours:.1f} per hour with gaps up to {gap} s" for gap in triples
        )
        + ".",
        "",
    ]
    metrics |= {"blinks.fp1_spontaneous_doubles_per_hour": fp1_doubles / hours}
    metrics |= {
        f"blinks.fp1_spontaneous_triples_{gap}s_per_hour": n / hours for gap, n in triples.items()
    }
    return "\n".join(lines), metrics


def count_triples(times: list[float], max_gap_s: float, min_gap_s: float = 0.15) -> int:
    """Non-overlapping runs of three blinks, each gap within the given bounds."""
    count = i = 0
    while i + 2 < len(times):
        gaps = (times[i + 1] - times[i], times[i + 2] - times[i + 1])
        if all(min_gap_s <= gap <= max_gap_s for gap in gaps):
            count += 1
            i += 3
        else:
            i += 1
    return count


def channel_traces(raw: Any, names: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """The given channels of a run in µV, or ``None`` if one is missing."""

    def clean(name: str) -> str:
        return name.strip().rstrip(".").casefold()

    by_name = {clean(name): name for name in raw.ch_names}
    if any(clean(name) not in by_name for name in names):
        return None
    data = raw.get_data(picks=[by_name[clean(name)] for name in names]) * 1e6
    return dict(zip(names, data, strict=True))


def analyse_run(traces: dict[str, np.ndarray], srate: float) -> BlinkRun:
    fp1 = traces["Fp1"]
    fc = (traces["FC3"] + traces["FC4"]) / 2
    # Both channels must rise together: the smaller of the two band-passed
    # signals (the detector band-passes it again, which leaves it unchanged
    # in shape).
    fc_min = np.minimum(blink_band(traces["FC3"], srate), blink_band(traces["FC4"], srate))
    reference_detector = DoubleBlinkDetector(srate)
    fp1_doubles = reference_detector.update(fp1)
    fp1_slow, fc_slow = blink_band(fp1, srate), blink_band(fc, srate)
    window = round(0.05 * srate)
    reference, ratios = [], []
    for t in reference_detector.blinks:
        i = round(t * srate)
        amp = float(fp1_slow[i])
        reference.append((t, amp))
        segment = fc_slow[max(i - window, 0) : i + window + 1]
        ratios.append(float(segment.max()) / amp if amp > 0 else np.nan)
    fc_blinks: dict[tuple[str, float, float], list[float]] = {}
    fc_doubles: dict[tuple[str, float, float], list[float]] = {}
    for config in FC_CONFIGS:
        feature, k, threshold = config
        detector = DoubleBlinkDetector(srate, BlinkConfig(min_amplitude_uv=threshold, k=k))
        fc_doubles[config] = detector.update(fc if feature == "mean" else fc_min)
        fc_blinks[config] = detector.blinks
    return BlinkRun(
        minutes=len(fp1) / srate / 60,
        reference=reference,
        ratios=[r for r in ratios if np.isfinite(r)],
        fp1_doubles=fp1_doubles,
        fc_blinks=fc_blinks,
        fc_doubles=fc_doubles,
    )


def blink_band(data: np.ndarray, srate: float) -> np.ndarray:
    """The detector's causal 0.1-8 Hz band-pass."""
    sos = signal.butter(2, (0.1, 8.0), btype="bandpass", fs=srate, output="sos")
    filtered, _ = signal.sosfilt(sos, data, zi=signal.sosfilt_zi(sos) * data[0])
    return np.asarray(filtered)


# -- Report and tracking -----------------------------------------------------------


def read_sections() -> dict[str, str]:
    """Sections of the previous report, to keep the parts that are not rerun."""
    if not RESULTS.exists():
        return {}
    text = RESULTS.read_text()
    sections = {}
    for key, title in (("channels", "## Part A"), ("blinks", "## Part B")):
        match = re.search(rf"^{re.escape(title)}.*?(?=^## |\Z)", text, re.S | re.M)
        if match:
            sections[key] = match.group(0).rstrip() + "\n"
    return sections


def write_report(sections: dict[str, str], code: dict[str, str], versions: dict[str, str]) -> None:
    dirty = " (with uncommitted changes)" if code["dirty"] == "true" else ""
    lines = [
        "# Montage study",
        "",
        "Generated by `uv run python research/experiments/montage.py`. Do not edit by hand.",
        "Preprocessing as in `docs/pipeline.md` (causal 8-30 Hz, 2 s epochs from cue",
        "+ 0.5 s, 250 Hz). Discussion in `docs/montage.md`.",
        "",
        *(sections[key] for key in ("channels", "blinks") if key in sections),
        "## Run",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- Code: commit `{code['commit']}`{dirty}",
        f"- Seed {SEED}, {N_SPLITS}-fold within-session",
        "- Versions: " + ", ".join(f"{name} {version}" for name, version in versions.items()),
        "",
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines))


def track(part: str, metrics: dict[str, float], code: dict[str, str]) -> None:
    import mlflow

    from openmind_core.paths import data_dir

    store = data_dir(create=True) / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("montage") is None:
        mlflow.create_experiment("montage", artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment("montage")
    tags = {"git_commit": code["commit"], "git_dirty": code["dirty"], "task": "CORE-045"}
    with mlflow.start_run(run_name=f"montage-{part}", tags=tags):
        mlflow.log_params({"part": part, "seed": SEED, "candidates": ",".join(CANDIDATES)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(RESULTS))


def sanitize(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-/ ]", "_", key).replace(" ", "_")


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.0f} %" if denominator else "n/a"


def format_duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


if __name__ == "__main__":
    main()
