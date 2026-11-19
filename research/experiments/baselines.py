"""Benchmark the classical baselines on public datasets and track runs in MLflow.

    uv sync --extra research
    uv run python research/experiments/baselines.py [--config research/configs/baselines.yaml]
                                                    [--only NAME ...] [--no-tracking]

Writes the result tables to research/results/baselines.md and, unless
--no-tracking is given, one MLflow run per experiment (with a nested run per
pipeline and protocol) to $OPENMIND_DATA_DIR/mlflow. Preprocessed epochs are
cached in $OPENMIND_DATA_DIR/cache/epochs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

import openmind_core.datasets.moabb as loader
import openmind_core.preprocessing as preprocessing
from openmind_core.datasets import DATASETS, EpochSet, load_epochs
from openmind_core.evaluation import Score, Summary, drop_small_subjects, evaluate, summarize
from openmind_core.models import make_baseline
from openmind_core.paths import data_dir

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "research" / "configs" / "baselines.yaml"
RESULTS = ROOT / "research" / "results" / "baselines.md"
PACKAGES = ("openmind-core", "numpy", "scipy", "mne", "moabb", "pyriemann", "scikit-learn")

logger = logging.getLogger("baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", nargs="+", help="run only these experiments")
    parser.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    experiments = [e for e in config["experiments"] if not args.only or e["name"] in set(args.only)]
    code = code_version()
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    results: list[tuple[dict[str, Any], EpochSet, list[Summary], float]] = []
    for experiment in experiments:
        epochs, dropped = drop_small_subjects(
            cached_epochs(experiment), config["min_epochs_per_class"]
        )
        experiment = experiment | {"dropped": dropped}
        start = time.perf_counter()
        scores = evaluate(
            epochs,
            {name: (lambda name=name: make_baseline(name)) for name in config["pipelines"]},
            protocols=config["protocols"],
            n_splits=config["n_splits"],
            seed=config["seed"],
        )
        elapsed = time.perf_counter() - start
        summaries = summarize(scores, n_classes=len(experiment["classes"]))
        for summary in summaries:
            logger.info(
                "%s %s %s: %.1f ± %.1f %%",
                experiment["name"],
                summary.protocol,
                summary.pipeline,
                100 * summary.accuracy_mean,
                100 * summary.accuracy_std,
            )
        if not args.no_tracking:
            track(config, experiment, epochs, scores, summaries, code, versions)
        results.append((experiment, epochs, summaries, elapsed))

    if args.only:
        logger.info("partial run (--only): %s left unchanged", RESULTS.relative_to(ROOT))
    else:
        write_report(config, results, code, versions)
        logger.info("results written to %s", RESULTS.relative_to(ROOT))


def cached_epochs(experiment: dict[str, Any]) -> EpochSet:
    """Load the epochs of an experiment, from the cache when possible.

    The cache key includes the source code of the loader and of the
    preprocessing, so any change to them invalidates the cached epochs.
    """
    key = json.dumps({k: experiment[k] for k in ("dataset", "subjects", "classes")}, sort_keys=True)
    digest_input = key.encode() + source_bytes(loader.__file__)
    for module in sorted(Path(preprocessing.__file__).parent.glob("*.py")):
        digest_input += source_bytes(str(module))
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    path = data_dir() / "cache" / "epochs" / f"{experiment['dataset']}_{digest}.npz"
    if path.exists():
        logger.info("epochs from cache %s", path)
        return EpochSet.load(path)
    epochs = load_epochs(experiment["dataset"], experiment["subjects"], experiment["classes"])
    epochs.save(path)
    return epochs


def source_bytes(path: str | None) -> bytes:
    return Path(path).read_bytes() if path else b""


def code_version() -> dict[str, str]:
    """Commit of the repository and whether the working tree has changes."""

    def git(*command: str) -> str:
        try:
            return subprocess.run(
                ["git", *command], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "dirty": "true" if git("status", "--porcelain", "--untracked-files=no") else "false",
    }


def track(
    config: dict[str, Any],
    experiment: dict[str, Any],
    epochs: EpochSet,
    scores: list[Score],
    summaries: list[Summary],
    code: dict[str, str],
    versions: dict[str, str],
) -> None:
    """Log one experiment to MLflow: a parent run, one nested run per pipeline and protocol."""
    import mlflow

    store = data_dir(create=True) / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("baselines") is None:
        mlflow.create_experiment("baselines", artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment("baselines")

    tags = {"git_commit": code["commit"], "git_dirty": code["dirty"], "task": "CORE-030"}
    tags |= {f"version.{name}": version for name, version in versions.items()}
    data_params = {
        "dataset": experiment["dataset"],
        "subjects": ",".join(map(str, experiment["subjects"])),
        "classes": ",".join(experiment["classes"]),
        "channels": ",".join(epochs.channels),
        "srate": epochs.srate,
        "n_epochs": len(epochs),
        "seed": config["seed"],
        "n_splits": config["n_splits"],
        "min_epochs_per_class": config["min_epochs_per_class"],
        "dropped_subjects": ",".join(map(str, experiment["dropped"])) or "none",
    }
    with mlflow.start_run(run_name=experiment["name"], tags=tags):
        mlflow.log_params(data_params)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(Score.__dataclass_fields__))
                writer.writeheader()
                writer.writerows(asdict(score) for score in scores)
            mlflow.log_artifact(str(path))
        for summary in summaries:
            name = f"{summary.pipeline}/{summary.protocol}"
            with mlflow.start_run(run_name=name, nested=True, tags=tags):
                mlflow.log_params(
                    data_params | {"pipeline": summary.pipeline, "protocol": summary.protocol}
                )
                mlflow.log_metrics(
                    {
                        "accuracy_mean": summary.accuracy_mean,
                        "accuracy_std": summary.accuracy_std,
                        "kappa_mean": summary.kappa_mean,
                        "chance": summary.chance,
                    }
                )
                for score in scores:
                    if (score.pipeline, score.protocol) == (summary.pipeline, summary.protocol):
                        key = f"accuracy_s{score.subject:03d}_{score.session}"
                        mlflow.log_metric(key, score.accuracy)


def write_report(
    config: dict[str, Any],
    results: list[tuple[dict[str, Any], EpochSet, list[Summary], float]],
    code: dict[str, str],
    versions: dict[str, str],
) -> None:
    """Write the Markdown result tables."""
    protocols = config["protocols"]
    lines = [
        "# Baselines",
        "",
        "Generated by `uv run python research/experiments/baselines.py` from",
        "`research/configs/baselines.yaml`. Do not edit by hand.",
        "",
        "Accuracy in %, mean ± standard deviation over subjects (sessions averaged",
        "within each subject first); κ is Cohen's kappa. Preprocessing: causal",
        "8-30 Hz band-pass and notch, 2 s epochs from cue + 0.5 s, montage A",
        "decoding channels (`FC3, FC4, C3, Cz, C4, CP3, CP4`), 250 Hz.",
        "See `docs/evaluation.md` for the protocols and the discussion.",
        "",
    ]
    for experiment, epochs, summaries, elapsed in results:
        spec = DATASETS[experiment["dataset"]]
        chance = 100 / len(experiment["classes"])
        dropped = experiment["dropped"]
        excluded = (
            f" Subjects {', '.join(map(str, dropped))} excluded (fewer than "
            f"{config['min_epochs_per_class']} epochs per class after rejection)."
            if dropped
            else ""
        )
        lines += [
            f"## {experiment['name']}",
            "",
            f"{spec.description} Subjects {format_subjects(experiment['subjects'])}, "
            f"classes {', '.join(experiment['classes'])} (chance {chance:.0f} %), "
            f"{len(epochs)} epochs after rejection.{excluded} Evaluation time {elapsed:.0f} s.",
            "",
            "| Pipeline | " + " | ".join(protocols) + " |",
            "|---|" + "---|" * len(protocols),
        ]
        for pipeline in config["pipelines"]:
            cells = []
            for protocol in protocols:
                match = [s for s in summaries if (s.pipeline, s.protocol) == (pipeline, protocol)]
                if match:
                    s = match[0]
                    cells.append(
                        f"{100 * s.accuracy_mean:.1f} ± {100 * s.accuracy_std:.1f} "
                        f"(κ {s.kappa_mean:.2f})"
                    )
                else:
                    cells.append("n/a")
            lines.append(f"| `{pipeline}` | " + " | ".join(cells) + " |")
        lines.append("")
    dirty = " (with uncommitted changes)" if code["dirty"] == "true" else ""
    lines += [
        "## Run",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- Code: commit `{code['commit']}`{dirty}",
        f"- Seed {config['seed']}, {config['n_splits']}-fold within-session",
        "- Versions: " + ", ".join(f"{name} {version}" for name, version in versions.items()),
        "",
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines))


def format_subjects(subjects: list[int]) -> str:
    if subjects == list(range(subjects[0], subjects[-1] + 1)):
        return f"{subjects[0]}-{subjects[-1]}"
    return ", ".join(map(str, subjects))


if __name__ == "__main__":
    main()
